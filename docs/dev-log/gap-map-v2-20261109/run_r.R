# Native R drmTMB runner for the gap-map v2 grid. One process, many cells, one result line per cell.
# Usage: Rscript run_r.R <cells.tsv> <idlist.txt> <start_index> <outfile>
# Env: FUZZ_DATA (data dir), DRMTMB_LIB (library holding drmTMB 0.8.1).
args <- commandArgs(TRUE)
cells_file <- args[1]; idlist <- readLines(args[2]); start <- as.integer(args[3]); outfile <- args[4]
DD <- Sys.getenv("FUZZ_DATA", "data")
if (nzchar(Sys.getenv("DRMTMB_LIB"))) .libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
cells <- read.delim(cells_file, stringsAsFactors = FALSE, quote = "", comment.char = "")
rownames(cells) <- cells$id

d <- read.csv(file.path(DD, "data_v2.csv"), stringsAsFactors = FALSE, na.strings = "NA")
for (v in c("g", "g2", "h", "f", "f2", "gna", "gu", "y_cat")) d[[v]] <- factor(d[[v]])
d$y_ordf <- factor(d$y_ordf, levels = c("o1", "o2", "o3", "o4"), ordered = TRUE)

full_msg <- function(x) substr(trimws(gsub("[[:space:]]+", " ", paste(x, collapse = " "))), 1, 600)
first_line <- function(x) { x <- gsub("[\t\r]", " ", strsplit(paste(x, collapse = "\n"), "\n")[[1]]); x <- x[nzchar(trimws(x))]; if (length(x)) substr(trimws(x[1]), 1, 400) else "" }
clean_ansi <- function(x) gsub("\033\\[[0-9;]*m", "", x)
fnum <- function(x) sprintf("%.10g", x)
fin <- function(v) { v <- suppressWarnings(as.numeric(v)); v[is.finite(v)] }
probe <- function(expr) tryCatch(list(ok = TRUE, v = expr), error = function(e) list(ok = FALSE, v = substr(gsub("[\t\n]+", " ", clean_ansi(conditionMessage(e))), 1, 120)))
pstat <- function(p) if (p$ok) "ok" else paste0("ERR:", p$v)

run_one <- function(id) {
  cl <- cells[id, ]
  warn <- character(0)
  t0 <- Sys.time()
  res <- tryCatch(
    withCallingHandlers({
      fit <- eval(parse(text = cl$r))
      list(ok = TRUE, fit = fit)
    }, warning = function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") }),
    error = function(e) list(ok = FALSE, e = e))
  secs <- as.numeric(Sys.time() - t0, units = "secs")
  w1 <- if (length(warn)) first_line(clean_ansi(warn[1])) else ""
  if (!res$ok) {
    e <- res$e
    cond <- if (inherits(e, "rlang_error")) "abort" else "base"
    return(c(status = "ERROR", etype = paste0(cond, ":", class(e)[1]), msg = first_line(clean_ansi(conditionMessage(e))),
             msgfull = full_msg(clean_ansi(conditionMessage(e))), ll = "", conv = "", extra = "", secs = sprintf("%.2f", secs),
             warn = w1, metrics = "", probes = ""))
  }
  fit <- res$fit
  ll <- tryCatch(as.numeric(logLik(fit)), error = function(e) NA_real_)
  conv <- tryCatch(fit$opt$convergence, error = function(e) NA)
  pd <- tryCatch(isTRUE(fit$sdr$pdHess), error = function(e) NA)
  isc <- tryCatch(isTRUE(drmTMB::is_converged(fit)), error = function(e) NA)
  ok <- is.finite(ll) && identical(as.integer(conv), 0L)
  m <- character(0); pr <- character(0)
  pa <- function(k, v) m <<- c(m, paste0(k, "=", v))
  pp <- function(k, p) pr <<- c(pr, paste0(k, "=", pstat(p)))
  pa("aic", tryCatch(fnum(AIC(fit)), error = function(e) "NA")); pa("bic", tryCatch(fnum(BIC(fit)), error = function(e) "NA"))
  pa("nobs", tryCatch(as.character(nobs(fit)), error = function(e) "NA"))
  p <- probe(fitted(fit)); pp("fitted", p)
  if (p$ok) { v <- fin(p$v); pa("fit_n", length(v)); pa("fit_mean", fnum(mean(v))); pa("fit_sd", fnum(sd(v))) }
  p <- probe(predict(fit)); pp("predict", p)
  if (p$ok) { v <- fin(p$v); pa("pred_mean", fnum(mean(v))) }
  idx <- c(5, 1, 9)
  pf <- probe(predict(fit, newdata = d[idx, , drop = FALSE])); pp("predict_new", pf)
  if (pf$ok) { v <- as.numeric(pf$v); pa("pred_new", paste(fnum(v), collapse = ",")) }
  if (p$ok && pf$ok) { vf <- as.numeric(p$v)[idx]; pa("pred_new_consistent", isTRUE(all.equal(as.numeric(pf$v), vf, tolerance = 1e-6))) }
  p <- probe(residuals(fit)); pp("resid", p)
  if (p$ok) { v <- fin(p$v); pa("resid_sd", fnum(sd(v))) }
  p <- probe(simulate(fit, nsim = 1)); pp("simulate", p)
  if (p$ok) { v <- fin(p$v[[1]]); pa("sim_n", length(v)) }
  p <- probe(vcov(fit)); pp("vcov", p)
  if (p$ok) { pa("npar", nrow(p$v)); pa("se_finite", all(is.finite(sqrt(diag(p$v))))) }
  p <- probe(coef(fit)); pp("coef", p)
  if (p$ok) pa("ncoef", length(unlist(p$v)))
  if (cl$probes == "ext") {
    nd <- d[idx, , drop = FALSE]
    pp("predict_link", probe(predict(fit, newdata = nd, type = "link")))
    pp("predict_parameters", probe(predict_parameters(fit, newdata = nd)))
    ndn <- d[idx, , drop = FALSE]; if ("f" %in% names(ndn)) ndn$f <- factor(c("a", "b", "z"), levels = c("a", "b", "z"))
    pp("predict_unseen_level", probe(predict(fit, newdata = ndn)))
    p <- probe(simulate(fit, nsim = 3)); pp("simulate3", p); if (p$ok) pa("sim3_ncol", ncol(p$v))
    p <- probe(confint(fit)); pp("confint", p); if (p$ok) { pa("ci_rows", nrow(p$v)); pa("ci_finite", mean(is.finite(as.matrix(p$v)))) }
    pp("summary", probe(capture.output(summary(fit))))
  }
  c(status = if (ok) "FITS" else "NONCONVERGED", etype = "", msg = "", msgfull = "", ll = sprintf("%.10f", ll), conv = as.character(conv),
    extra = sprintf("pdHess=%s;is_converged=%s", pd, isc), secs = sprintf("%.2f", secs), warn = w1,
    metrics = paste(m, collapse = ";"), probes = paste(pr, collapse = ";"))
}

FIELDS <- c("status", "etype", "msg", "msgfull", "ll", "conv", "extra", "secs", "warn", "metrics", "probes")
con <- file(outfile, "a")
for (i in seq(start, length(idlist))) {
  id <- idlist[i]
  cat(sprintf("S\t%d\t%s\n", i, id), file = con); flush(con)
  r <- tryCatch(run_one(id), error = function(e) c(status = "RUNNER_ERROR", etype = "runner", msg = first_line(conditionMessage(e)), msgfull = full_msg(conditionMessage(e)), ll = "", conv = "", extra = "", secs = "", warn = "", metrics = "", probes = ""))
  cat(sprintf("R\t%d\t%s\t%s\n", i, id, paste(gsub("[\t\n]", " ", r[FIELDS]), collapse = "\t")), file = con); flush(con)
}
close(con)
