# Regenerates reference.tsv and sigma_reference.tsv: drmTMB 0.8.1 fits of the bivariate Gaussian with an
# ordinary correlated random intercept (1 | p | g) on sigma1 and sigma2 (Laplace, ML).
# Usage: Rscript make_reference.R <data_v2_subset.csv> <sre_bal.csv> <sre_unbal.csv> <sre_cov.csv> <out_reference.tsv> <out_sigma.tsv>
# harness.csv: columns g, x1, x2, y1, y2 of the gap-map v2 data (160 rows, 16 groups): the harness cell,
# on which drmTMB itself sits on a boundary (group correlation 1).
args <- commandArgs(TRUE)
suppressPackageStartupMessages(library(drmTMB))
rd <- function(f) { x <- read.csv(f); x$g <- factor(x$g); x }
h <- rd(args[1]); bal <- rd(args[2]); unb <- rd(args[3]); cv <- rd(args[4])
b0 <- bf(mu1 = y1 ~ x1, mu2 = y2 ~ x2, sigma1 = ~ 1 + (1 | p | g), sigma2 = ~ 1 + (1 | p | g), rho12 = ~ 1)
bc <- bf(mu1 = y1 ~ x1, mu2 = y2 ~ x2, sigma1 = ~ x1 + (1 | p | g), sigma2 = ~ 1 + (1 | p | g), rho12 = ~ x1)
cells <- list(
  harness   = quote(drmTMB(b0, family = biv_gaussian(), data = h)),
  bal       = quote(drmTMB(b0, family = biv_gaussian(), data = bal)),
  unbal     = quote(drmTMB(b0, family = biv_gaussian(), data = unb)),
  cov       = quote(drmTMB(bc, family = biv_gaussian(), data = cv))
)
out <- list(); so <- list()
for (nm in names(cells)) {
  fit <- eval(cells[[nm]])
  out[[length(out) + 1]] <- data.frame(cell = nm, kind = "logLik", name = "logLik", estimate = as.numeric(logLik(fit)), se = NA)
  sdr <- summary(fit$sdr); rn <- rownames(sdr)
  keep <- rn %in% c("beta_mu1","beta_mu2","beta_sigma1","beta_sigma2","beta_rho12","log_sd_sigma","eta_cor_sigma","sd_sigma_re","rho_sigma_re")
  sd2 <- sdr[keep & !duplicated(paste(rn, round(sdr[,1], 8))), , drop = FALSE]
  out[[length(out) + 1]] <- data.frame(cell = nm, kind = "par", name = make.unique(rownames(sd2)), estimate = sd2[, 1], se = sd2[, 2])
  out[[length(out) + 1]] <- data.frame(cell = nm, kind = "nobs", name = "nobs", estimate = nobs(fit), se = NA)
  out[[length(out) + 1]] <- data.frame(cell = nm, kind = "pdHess", name = "pdHess", estimate = as.numeric(isTRUE(fit$sdr$pdHess)), se = NA)
  out[[length(out) + 1]] <- data.frame(cell = nm, kind = "conv", name = "conv", estimate = fit$opt$convergence, se = NA)
  s1 <- as.numeric(predict(fit, dpar = "sigma1")); s2 <- as.numeric(predict(fit, dpar = "sigma2"))
  fv <- fitted(fit)
  so[[length(so) + 1]] <- data.frame(cell = nm, i = seq_along(s1), sigma1 = sprintf("%.15g", s1), sigma2 = sprintf("%.15g", s2),
                                     mu1 = sprintf("%.15g", fv[, 1]), mu2 = sprintf("%.15g", fv[, 2]))
}
write.table(do.call(rbind, out), args[5], sep = "\t", quote = FALSE, row.names = FALSE)
write.table(do.call(rbind, so), args[6], sep = "\t", quote = FALSE, row.names = FALSE)
