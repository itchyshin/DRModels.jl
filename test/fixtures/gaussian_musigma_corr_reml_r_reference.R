# R reference for test_gaussian_musigma_corr_reml.jl (drmTMB 0.8.1, REML = TRUE).
# Usage: DRMTMB_LIB=<lib with drmTMB> GMC_DIR=<dir with data_<cell>.csv> Rscript this.R
# Model: y ~ x... + (1 | p | g), sigma ~ 1 [+ x1] + (1 | p | g), family = gaussian(), REML = TRUE.
# Writes rcoef_<cell>.csv (name, est, se from vcov()) and rll_<cell>.txt; the numbers are pasted into
# the Julia test as `_GMC_R`.
if (nzchar(Sys.getenv("DRMTMB_LIB"))) .libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressMessages(library(drmTMB))
O <- Sys.getenv("GMC_DIR", ".")
for (cell in c("harness", "s1", "s2")) {
  d <- read.csv(file.path(O, paste0("data_", cell, ".csv"))); d$g <- factor(d$g, levels = unique(d$g))
  fm <- switch(cell,
    harness = bf(y ~ x1 + x2 + x3 + (1 | p | g), sigma = ~ 1 + (1 | p | g)),
    s1 = bf(y ~ x1 + x2 + (1 | p | g), sigma = ~ 1 + (1 | p | g)),
    s2 = bf(y ~ x1 + x2 + (1 | p | g), sigma = ~ 1 + x1 + (1 | p | g)))
  fit <- drmTMB(fm, family = gaussian(), data = d, REML = TRUE)
  V <- vcov(fit); nm <- rownames(V)
  cat("\n##", cell, "ll", sprintf("%.10f", as.numeric(logLik(fit))), "conv", fit$opt$convergence,
      "pdHess", fit$sdr$pdHess, "is_converged", isTRUE(is_converged(fit)), "exact", fit$estimator_exact, "\n")
  print(names(coef(fit))); print(nm)
  print(unlist(coef(fit)))
  print(fit$opt$par)
  cf <- unlist(coef(fit), use.names = FALSE)
  # beta rows from vcov() (sdreport covariance of the integrated coefficients); then the three outer
  # parameters log_sd_mu, log_sd_sigma, eta_cor_mu_sigma (rho = tanh(eta)) with their sdreport SEs.
  sf <- summary(fit$sdr, "fixed")
  pn <- names(fit$opt$par)
  out <- data.frame(name = c(nm, "log_sd_mu", "log_sd_sigma", "eta_cor_mu_sigma"),
                    est = c(cf[seq_along(nm)], fit$opt$par[c("log_sd_mu", "log_sd_sigma", "eta_cor_mu_sigma")]),
                    se = c(sqrt(diag(V)), sf[match(c("log_sd_mu", "log_sd_sigma", "eta_cor_mu_sigma"), rownames(sf)), 2]))
  print(out)
  write.csv(out, file.path(O, paste0("rcoef_", cell, ".csv")), row.names = FALSE)
  write(sprintf("%.12f", as.numeric(logLik(fit))), file.path(O, paste0("rll_", cell, ".txt")))
}
