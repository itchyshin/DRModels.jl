# drmTMB reference for test_vc_laplace_corr_slope.jl: SDs, correlations and log-likelihood of the
# correlated block `(1 + x | g)` (and `(1 + x + x2 | g)` for Binomial) fitted by maximum likelihood.
#   Rscript gen_ref.R r_reference.csv        (from this directory, after gen_data.jl)
# Reference: drmTMB 0.8.1, engine = "tmb", library ~/drm_campaign_20261108/Rlib on Totoro.
.libPaths(c("~/drm_campaign_20261108/Rlib", .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
out <- commandArgs(TRUE)[1]
rows <- list()
fit_one <- function(id, fam, formula, d) {
  d$g <- factor(d$g, levels = unique(d$g))
  f <- drmTMB(formula, family = fam, data = d)
  sd <- unlist(f$sdpars$mu); cr <- unlist(f$corpars$mu)
  data.frame(id = id, sd1 = sd[1], sd2 = sd[2], sd3 = if (length(sd) > 2) sd[3] else NA,
             rho12 = cr[1], rho13 = if (length(cr) > 1) cr[2] else NA, rho23 = if (length(cr) > 2) cr[3] else NA,
             logLik = as.numeric(logLik(f)), converged = isTRUE(f$opt$convergence == 0))
}
for (seed in 1:4) {
  d <- read.csv(sprintf("data_binomial_s%d.csv", seed))
  rows[[length(rows) + 1]] <- fit_one(sprintf("binomial_s%d", seed), binomial(), bf(cbind(s, fl) ~ x + (1 + x | g)), d)
  d <- read.csv(sprintf("data_poisson_s%d.csv", seed))
  rows[[length(rows) + 1]] <- fit_one(sprintf("poisson_s%d", seed), poisson(), bf(y ~ x + (1 + x | g)), d)
  d <- read.csv(sprintf("data_nb2_s%d.csv", seed))
  rows[[length(rows) + 1]] <- fit_one(sprintf("nb2_s%d", seed), nbinom2(), bf(y ~ x + (1 + x | g)), d)
}
for (seed in 1:2) {
  d <- read.csv(sprintf("data_binomial_q3_s%d.csv", seed))
  rows[[length(rows) + 1]] <- fit_one(sprintf("binomial_q3_s%d", seed), binomial(), bf(cbind(s, fl) ~ x + x2 + (1 + x + x2 | g)), d)
}
res <- do.call(rbind, rows); row.names(res) <- NULL
write.csv(res, out, row.names = FALSE)
