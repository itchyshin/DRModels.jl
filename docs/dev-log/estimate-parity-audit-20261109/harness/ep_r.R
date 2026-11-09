# Estimate-parity extractor, R side. Fits one cell (same builder as the wave-8 gap-map harness) and dumps
# user-facing quantities as long TSV: id, section, key, v1, v2, v3.
# Usage: Rscript ep_r.R <ids.tsv> <outfile>      (cwd holds ep_cell_r.R, cells_all.tsv, data/)
source("ep_cell_r.R")
args <- commandArgs(TRUE)
ids_all <- readLines(args[1])[-1]; outfile <- args[2]
NSIM <- 1000L
ENG <- Sys.getenv("EP_ENGINE", "tmb")   # "julia" runs the drmTMB(..., engine = "julia") bridge
con <- file(outfile, "w")
emit <- function(id, sec, key, v1 = "", v2 = "", v3 = "") {
  cat(paste(c(id, sec, key, v1, v2, v3), collapse = "\t"), "\n", file = con, sep = ""); flush(con)
}
num <- function(x) sprintf("%.12g", x)
vec <- function(id, sec, v) for (i in seq_along(v)) emit(id, sec, as.character(i), num(v[i]))
tryf <- function(id, sec, expr) tryCatch(expr, error = function(e) emit(id, sec, "ERROR", gsub("[\t\n]+", " ", substr(conditionMessage(e), 1, 300))))
for (id in ids_all) {
  cl <- cells[id, ]
  kk <- if (cl$kind %in% c("indep", "slope")) "ordinary" else cl$kind
  d <- read.csv(file.path(DD, sprintf("data_%s_%s.csv", cl$family, kk)))
  d$g <- factor(d$g, levels = unique(d$g)); d$g2 <- factor(d$g2)
  t0 <- Sys.time()
  fit <- tryCatch(if (ENG == "tmb") drmTMB(build(cl), family = fam_of(cl$family), data = d, REML = (cl$est == "REML")) else drmTMB(build(cl), family = fam_of(cl$family), data = d, REML = (cl$est == "REML"), engine = ENG),
                  error = function(e) { emit(id, "meta", "FIT_ERROR", gsub("[\t\n]+", " ", substr(conditionMessage(e), 1, 300))); NULL })
  if (is.null(fit)) next
  emit(id, "meta", "secs", num(as.numeric(Sys.time() - t0, units = "secs")))
  tryf(id, "meta", {
    ll <- logLik(fit)
    emit(id, "meta", "logLik", num(as.numeric(ll))); emit(id, "meta", "df", num(attr(ll, "df")))
    emit(id, "meta", "AIC", num(AIC(fit))); emit(id, "meta", "BIC", num(BIC(fit))); emit(id, "meta", "nobs", num(nobs(fit)))
    emit(id, "meta", "converged", as.character(isTRUE(is_converged(fit))))
    emit(id, "meta", "pdHess", as.character(isTRUE(fit$sdr$pdHess)))
  })
  s <- tryCatch(summary(fit), error = function(e) { emit(id, "meta", "SUMMARY_ERROR", substr(conditionMessage(e), 1, 300)); NULL })
  if (!is.null(s)) {
    tryf(id, "coef", { cf <- s$coefficients; for (i in seq_len(nrow(cf))) emit(id, "coef", rownames(cf)[i], num(cf$estimate[i]), num(cf$std_error[i])) })
    tryf(id, "par", { p <- s$parameters; for (i in seq_len(nrow(p))) emit(id, "par", paste(p$component[i], p$dpar[i], p$term[i], sep = "|"), num(p$estimate[i]), num(p$std_error[i]), p$scale[i]) })
    tryf(id, "cov", { cv <- s$covariance; if (is.data.frame(cv) && nrow(cv)) for (i in seq_len(nrow(cv))) emit(id, "cov", paste(cv$group[i], cv$class[i], i, sep = "|"), num(cv$correlation[i]), num(cv$from_sd[i]), num(cv$to_sd[i])) })
  }
  tryf(id, "ci", { ci <- confint(fit); for (i in seq_len(nrow(ci))) emit(id, "ci", paste(ci$parm[i], ci$scale[i], sep = "|"), num(ci$lower[i]), num(ci$upper[i]), as.character(ci$method[i])) })
  # newdata: 8 in-sample rows spanning the data
  idx <- unique(round(seq(1, nrow(d), length.out = 8))); nd <- d[idx, , drop = FALSE]
  tryf(id, "pred_resp", vec(id, "pred_resp", as.numeric(predict(fit, newdata = nd, type = "response"))))
  tryf(id, "pred_link", vec(id, "pred_link", as.numeric(predict(fit, newdata = nd, type = "link"))))
  tryf(id, "fitted", vec(id, "fitted", as.numeric(fitted(fit))))
  tryf(id, "resid_q", { set.seed(1); vec(id, "resid_q", as.numeric(residuals(fit, type = "quantile"))) })
  tryf(id, "resid_r", vec(id, "resid_r", as.numeric(residuals(fit, type = "response"))))
  for (mode in c("marginal", "cond")) tryf(id, paste0("simerr:", mode), {
    sm <- if (mode == "marginal") as.matrix(simulate(fit, nsim = NSIM, seed = 1)) else as.matrix(simulate(fit, nsim = NSIM, seed = 1, re.form = NA))
    n <- nrow(d); if (nrow(sm) != n) sm <- t(sm)
    emit(id, "sim", paste0(mode, ":grand_mean"), num(mean(sm))); emit(id, "sim", paste0(mode, ":grand_var"), num(var(as.numeric(sm))))
    emit(id, "sim", paste0(mode, ":mean_perobs_var"), num(mean(apply(sm, 1, var)))); emit(id, "sim", paste0(mode, ":p_zero"), num(mean(sm == 0)))
    emit(id, "sim", paste0(mode, ":mean_perobs_mean_first8"), num(mean(rowMeans(sm)[idx]))); emit(id, "sim", paste0(mode, ":dim"), paste(dim(sm), collapse = "x"))
  })
}
close(con)
