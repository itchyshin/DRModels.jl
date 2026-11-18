# R reference for test_sigma_re_parity.jl (drmTMB 0.8.1, Laplace).
# Usage: DRMTMB_LIB=<lib with drmTMB> SRE_CSV=<sigma_re_data.csv> Rscript this.R
# The CSV is written by `sre_write_csv()` in sigma_re_fixture.jl. Prints, per cell, the log-likelihood,
# coefficients with SEs and the convergence flags; the numbers are pasted into the Julia test.
if (nzchar(Sys.getenv("DRMTMB_LIB"))) .libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressMessages(library(drmTMB))
stopifnot(as.character(packageVersion("drmTMB")) == "0.8.1")
d <- read.csv(Sys.getenv("SRE_CSV", "sigma_re_data.csv"))
d$g <- factor(d$g); d$g2 <- factor(d$g2)
cells <- list(
  lognormal = quote(drmTMB(bf(y_ln ~ x1, sigma ~ x1 + (1 | g)), family = lognormal(), data = d)),
  gamma     = quote(drmTMB(bf(y_gam ~ x1, sigma ~ x1 + (1 | g)), family = stats::Gamma(link = "log"), data = d)),
  nb2       = quote(drmTMB(bf(y_nb ~ x1, sigma ~ x1 + (1 | g)), family = nbinom2(), data = d)),
  bb        = quote(drmTMB(bf(cbind(sb, flb) ~ x1, sigma ~ x1 + (1 | g)), family = beta_binomial(), data = d)),
  gauss_slope0 = quote(drmTMB(bf(y_g ~ x1, sigma ~ x1 + (0 + x1 | g)), family = gaussian(), data = d)),
  gauss_slope  = quote(drmTMB(bf(y_g ~ x1, sigma ~ x1 + (1 + x1 | g)), family = gaussian(), data = d)),
  gauss_two    = quote(drmTMB(bf(y_g ~ x1, sigma ~ x1 + (1 | g) + (1 | g2)), family = gaussian(), data = d)),
  gauss_diff   = quote(drmTMB(bf(y_gm ~ x1 + (1 | g), sigma ~ x1 + (1 | g2)), family = gaussian(), data = d)))
for (cell in names(cells)) {
  fit <- eval(cells[[cell]])
  V <- tryCatch(vcov(fit), error = function(e) NULL)
  cat("\n##", cell, "ll", sprintf("%.10f", as.numeric(logLik(fit))), "conv", fit$opt$convergence,
      "pdHess", fit$sdr$pdHess, "is_converged", is_converged(fit), "\n")
  print(coef(fit)); print(summary(fit))
  if (!is.null(V)) { print(data.frame(name = rownames(V), est = unlist(coef(fit), use.names = FALSE), se = sqrt(diag(V))), digits = 10) }
  print(unlist(fit$opt$par), digits = 10)
}
