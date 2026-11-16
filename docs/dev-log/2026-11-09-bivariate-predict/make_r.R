# Bivariate Gaussian with missing responses: what does drmTMB 0.8.1 do by default?
# Writes data_seed<k>.csv and r_seed<k>.txt. Usage: Rscript make_r.R <outdir>
lib <- Sys.getenv("DRMTMB_LIB", path.expand("~/drm_campaign_20261108/Rlib"))
.libPaths(c(lib, .libPaths()))
suppressMessages(library(drmTMB))
out <- commandArgs(TRUE)[1]
cat("drmTMB", as.character(packageVersion("drmTMB")), "\n")
gen <- function(seed, n = 160) {
  set.seed(seed)
  x1 <- rnorm(n); x2 <- rnorm(n)
  e1 <- rnorm(n); e2 <- 0.5 * e1 + sqrt(1 - 0.25) * rnorm(n)
  y1 <- 0.5 + 0.8 * x1 + 0.7 * e1
  y2 <- -0.3 + 0.6 * x2 + 1.1 * e2
  y1m <- y1; y2m <- y2
  y1m[sample(n, 20)] <- NA; y2m[sample(n, 25)] <- NA
  d <- data.frame(y1 = y1, y2 = y2, y1m = y1m, y2m = y2m, x1 = x1, x2 = x2)
  d
}
fm <- function(cell) switch(cell,
  both = bf(mu1 = y1m ~ x1, mu2 = y2m ~ x2, sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1),
  y1 = bf(mu1 = y1m ~ x1, mu2 = y2 ~ x2, sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1),
  y2 = bf(mu1 = y1 ~ x1, mu2 = y2m ~ x2, sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1))
for (seed in 1:3) {
  d <- gen(seed); write.csv(d, file.path(out, sprintf("data_seed%d.csv", seed)), row.names = FALSE, na = "NA")
  for (mode in c("default", "include", "drop")) {
    for (cell in c("both", "y1", "y2")) {
      r1 <- if (cell %in% c("both", "y1")) "y1m" else "y1"
      r2 <- if (cell %in% c("both", "y2")) "y2m" else "y2"
      msgs <- character()
      fit <- withCallingHandlers(
        tryCatch(
          if (mode == "default") drmTMB(fm(cell), family = biv_gaussian(), data = d)
          else drmTMB(fm(cell), family = biv_gaussian(), data = d, missing = miss_control(response = mode)),
          error = function(e) e),
        warning = function(w) { msgs <<- c(msgs, paste("W:", conditionMessage(w))); invokeRestart("muffleWarning") },
        message = function(m) { msgs <<- c(msgs, paste("M:", conditionMessage(m))); invokeRestart("muffleMessage") })
      tag <- sprintf("seed=%d mode=%s cell=%s", seed, mode, cell)
      if (inherits(fit, "error")) { cat(tag, "ERROR", gsub("\n", " ", conditionMessage(fit)), "\n"); next }
      sm <- summary(fit)
      co <- tryCatch(unlist(coef(fit)), error = function(e) NULL)
      V <- tryCatch(vcov(fit), error = function(e) NULL)
      se <- if (!is.null(V)) sqrt(diag(V)) else NULL
      fv <- tryCatch(fitted(fit), error = function(e) NULL)
      if (!is.null(fv)) write.table(format(as.matrix(fv), digits = 15), file.path(out, sprintf("fitted_seed%d_%s_%s.csv", seed, mode, cell)), sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
      cat(tag, sprintf("logLik=%.10f nobs=%s npar=%d", as.numeric(logLik(fit)), tryCatch(nobs(fit), error=function(e) NA), length(co)), "\n")
      cat("  msgs:", if (length(msgs)) paste(unique(msgs), collapse = " | ") else "none", "\n")
      cat("  coef:", paste(sprintf("%s=%.10f", names(co), co), collapse = " "), "\n")
      if (!is.null(se)) cat("  se:", paste(sprintf("%.10f", se), collapse = " "), "\n")
      if (!is.null(fv)) cat("  fitted: len=", NROW(fv), " class=", class(fv)[1], " head=", paste(sprintf("%.6f", head(as.numeric(fv), 3)), collapse = ","),
                           " sum=", sprintf("%.6f", sum(as.numeric(fv), na.rm=TRUE)), "\n")
    }
  }
}

# Bivariate lognormal: fitted() is E[Y_k] = exp(mu_k + sigma_k^2 / 2), one column per response.
d <- gen(1); d$l1 <- exp(d$y1); d$l2 <- exp(d$y2)
fl <- drmTMB(bf(mu1 = l1 ~ x1, mu2 = l2 ~ x2, sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1), family = biv_lognormal(), data = d)
write.table(format(as.matrix(fitted(fl)), digits = 15), file.path(out, "fitted_lognormal_seed1.csv"), sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
cat(sprintf("lognormal seed=1 logLik=%.10f nobs=%d\n", as.numeric(logLik(fl)), nobs(fl)))
