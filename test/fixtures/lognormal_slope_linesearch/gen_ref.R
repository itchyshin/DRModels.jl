# R reference for the LogNormal (1 + x1 | g) line-search regression test (test_lognormal_corr_linesearch.jl):
#   y ~ x1 + (1 + x1 | g), sigma ~ 1, data.csv (four 120-row datasets, 12 groups of 10, column `dataset`):
#   s2, s10 are LogNormal draws of the parity grid with a random slope (Julia 1.10.12 threw a HagerZhang
#   AssertionError on both; s10 also on 1.12.7); b1, b10 are draws with NO random slope (slope SD = 0,
#   rho on the boundary): b1 threw on Julia 1.13.3, b10 on 1.10.12.
# drmTMB 0.8.1, engine = "tmb", library ~/drm_campaign_20261108/Rlib on Totoro.
# Usage: Rscript gen_ref.R r_reference.csv
.libPaths(c("~/drm_campaign_20261108/Rlib", .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
stopifnot(as.character(packageVersion("drmTMB")) == "0.8.1")
all <- read.csv("data.csv"); all$g <- factor(all$g)
res <- list()
for (ds in unique(all$dataset)) {
  d <- all[all$dataset == ds, ]
  fit <- drmTMB(bf(y ~ x1 + (1 + x1 | g), sigma ~ 1), family = lognormal(), data = d)
  s <- summary(fit)
  co <- as.data.frame(s$coefficients); sp <- as.data.frame(s$parameters)
  out <- rbind(data.frame(term = rownames(co), estimate = co$estimate, se = co$std_error),
               data.frame(term = sp$parm, estimate = sp$estimate, se = sp$std_error))
  out$dataset <- ds; out$loglik <- as.numeric(logLik(fit)); out$conv <- fit$opt$convergence
  out$pdhess <- isTRUE(fit$sdr$pdHess); out$is_converged <- isTRUE(drmTMB::is_converged(fit))
  res[[ds]] <- out
}
res <- do.call(rbind, res); rownames(res) <- NULL
print(res, digits = 10); write.csv(res, commandArgs(TRUE)[1], row.names = FALSE)
