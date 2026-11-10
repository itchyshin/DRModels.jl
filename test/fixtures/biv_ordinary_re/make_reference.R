# Regenerates reference.tsv: drmTMB 0.8.1 fits of the bivariate Gaussian with an
# ordinary correlated random intercept (1 | p | g) in mu1 and mu2.
# Usage: Rscript make_reference.R <data_biv_re.csv> <out.tsv>
# data_biv_re.csv is the columns id,g,x1,x2,y1,y2,y1m,y2m of the gap-map v2 data
# (harness/gen_data.py, seed 20261109).
args <- commandArgs(TRUE)
suppressPackageStartupMessages(library(drmTMB))
d <- read.csv(args[1], na.strings = "NA"); d$g <- factor(d$g)
cells <- list(
  ml_rho1   = quote(drmTMB(bf(mu1 = y1 ~ x1 + (1 | p | g), mu2 = y2 ~ x2 + (1 | p | g), sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1), family = biv_gaussian(), data = d)),
  ml_rhox1  = quote(drmTMB(bf(mu1 = y1 ~ x1 + (1 | p | g), mu2 = y2 ~ x2 + (1 | p | g), sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ x1), family = biv_gaussian(), data = d)),
  ml_sig1x1 = quote(drmTMB(bf(mu1 = y1 ~ x1 + (1 | p | g), mu2 = y2 ~ x2 + (1 | p | g), sigma1 = ~ x1, sigma2 = ~ 1, rho12 = ~ 1), family = biv_gaussian(), data = d)),
  reml_rho1 = quote(drmTMB(bf(mu1 = y1 ~ x1 + (1 | p | g), mu2 = y2 ~ x2 + (1 | p | g), sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1), family = biv_gaussian(), data = d, REML = TRUE)),
  miss_drop = quote(drmTMB(bf(mu1 = y1m ~ x1 + (1 | p | g), mu2 = y2m ~ x2 + (1 | p | g), sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1), family = biv_gaussian(), data = d))
)
out <- list()
for (nm in names(cells)) {
  fit <- eval(cells[[nm]])
  ll <- as.numeric(logLik(fit))
  out[[length(out) + 1]] <- data.frame(cell = nm, kind = "logLik", name = "logLik", estimate = ll, se = NA)
  sdr <- summary(fit$sdr)
  rn <- rownames(sdr)
  keep <- rn %in% c("beta_mu1","beta_mu2","beta_sigma1","beta_sigma2","beta_rho12","log_sd_mu","eta_cor_mu","sd_mu_re","rho_mu_re")
  sd2 <- sdr[keep & !duplicated(paste(rn, round(sdr[,1], 8))), , drop = FALSE]
  out[[length(out) + 1]] <- data.frame(cell = nm, kind = "par", name = make.unique(rownames(sd2)),
                                       estimate = sd2[, 1], se = sd2[, 2])
  out[[length(out) + 1]] <- data.frame(cell = nm, kind = "nobs", name = "nobs", estimate = nobs(fit), se = NA)
  out[[length(out) + 1]] <- data.frame(cell = nm, kind = "pdHess", name = "pdHess", estimate = as.numeric(isTRUE(fit$sdr$pdHess)), se = NA)
}
write.table(do.call(rbind, out), args[2], sep = "\t", quote = FALSE, row.names = FALSE)
