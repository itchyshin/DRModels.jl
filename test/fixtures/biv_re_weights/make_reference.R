# Regenerates reference.tsv and fitted_reference.tsv: drmTMB 0.8.1 fits of the bivariate Gaussian with
# an ordinary correlated random intercept (1 | p | g) in mu1 and mu2 plus prior weights.
# Usage: Rscript make_reference.R <data_weights.csv> <unbal_weights.csv> <out_reference.tsv> <out_fitted.tsv>
# data_weights.csv: gap-map v2 data (160 rows, 16 groups) with weights w (non-integer), iw (integer 1 to 3)
# and wz (w with zeros on rows 3, 50, 99 and on all of group g01). unbal_weights.csv: unbalanced groups
# with non-integer weights uw in [0.2, 4].
args <- commandArgs(TRUE)
suppressPackageStartupMessages(library(drmTMB))
d <- read.csv(args[1], na.strings = "NA"); d$g <- factor(d$g)
u <- read.csv(args[2]); u$g <- factor(u$g)
b0 <- bf(mu1 = y1 ~ x1 + (1 | p | g), mu2 = y2 ~ x2 + (1 | p | g), sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1)
bc <- bf(mu1 = y1 ~ x1 + (1 | p | g), mu2 = y2 ~ x2 + (1 | p | g), sigma1 = ~ x1, sigma2 = ~ 1, rho12 = ~ x1)
bm <- bf(mu1 = y1m ~ x1 + (1 | p | g), mu2 = y2m ~ x2 + (1 | p | g), sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1)
cells <- list(
  w_ml       = quote(drmTMB(b0, family = biv_gaussian(), data = d, weights = w)),
  w_reml     = quote(drmTMB(b0, family = biv_gaussian(), data = d, weights = w, REML = TRUE)),
  iw_ml      = quote(drmTMB(b0, family = biv_gaussian(), data = d, weights = iw)),
  iw_reml    = quote(drmTMB(b0, family = biv_gaussian(), data = d, weights = iw, REML = TRUE)),
  wz_ml      = quote(drmTMB(b0, family = biv_gaussian(), data = d, weights = wz)),
  wz_reml    = quote(drmTMB(b0, family = biv_gaussian(), data = d, weights = wz, REML = TRUE)),
  wcov_ml    = quote(drmTMB(bc, family = biv_gaussian(), data = d, weights = w)),
  wmiss_ml   = quote(drmTMB(bm, family = biv_gaussian(), data = d, weights = w)),
  wunbal_ml  = quote(drmTMB(b0, family = biv_gaussian(), data = u, weights = uw)),
  wunbal_reml = quote(drmTMB(b0, family = biv_gaussian(), data = u, weights = uw, REML = TRUE))
)
out <- list(); fo <- list()
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
  fv <- fitted(fit)
  fo[[length(fo) + 1]] <- data.frame(cell = nm, i = seq_len(nrow(fv)), mu1 = fv[, 1], mu2 = fv[, 2])
}
write.table(do.call(rbind, out), args[3], sep = "\t", quote = FALSE, row.names = FALSE)
fd <- do.call(rbind, fo); fd$mu1 <- sprintf("%.15g", fd$mu1); fd$mu2 <- sprintf("%.15g", fd$mu2)
write.table(fd, args[4], sep = "\t", quote = FALSE, row.names = FALSE)
