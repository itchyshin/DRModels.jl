# Regenerates reference_review.tsv: drmTMB 0.8.1 fits of the bivariate Gaussian with
# an ordinary (1 | p | g) on mu1 and mu2, for the interior and large-n fixtures.
# Usage: Rscript make_reference_review.R <dir with the csv files> <out.tsv>
# (csv files from gen_review_data.R). Same extraction as make_reference.R.
args <- commandArgs(TRUE)
suppressPackageStartupMessages(library(drmTMB))
rd <- function(f) { d <- read.csv(file.path(args[1], f)); d$g <- factor(d$g); d }
F0 <- bf(mu1 = y1 ~ x1 + (1 | p | g), mu2 = y2 ~ x2 + (1 | p | g), sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1)
Fc <- bf(mu1 = y1 ~ x1 + (1 | p | g), mu2 = y2 ~ x2 + (1 | p | g), sigma1 = ~ x1, sigma2 = ~ 1, rho12 = ~ x1)
cells <- list(
  int_bal_ml     = list(F0, "interior_bal.csv", FALSE),
  int_unbal_ml   = list(F0, "interior_unbal.csv", FALSE),
  int_unbal_reml = list(F0, "interior_unbal.csv", TRUE),
  int_cov_ml     = list(Fc, "interior_cov.csv", FALSE),
  int_cov_reml   = list(Fc, "interior_cov.csv", TRUE),
  large1200_ml   = list(F0, "large_1200.csv", FALSE),
  large1200_reml = list(F0, "large_1200.csv", TRUE),
  large2400_ml   = list(F0, "large_2400.csv", FALSE)
)
out <- list()
for (nm in names(cells)) {
  cl <- cells[[nm]]
  fit <- drmTMB(cl[[1]], family = biv_gaussian(), data = rd(cl[[2]]), REML = cl[[3]])
  out[[length(out) + 1]] <- data.frame(cell = nm, kind = "logLik", name = "logLik", estimate = as.numeric(logLik(fit)), se = NA)
  sdr <- summary(fit$sdr); rn <- rownames(sdr)
  keep <- rn %in% c("beta_mu1","beta_mu2","beta_sigma1","beta_sigma2","beta_rho12","log_sd_mu","eta_cor_mu","sd_mu_re","rho_mu_re")
  sd2 <- sdr[keep & !duplicated(paste(rn, round(sdr[,1], 8))), , drop = FALSE]
  out[[length(out) + 1]] <- data.frame(cell = nm, kind = "par", name = make.unique(rownames(sd2)), estimate = sd2[, 1], se = sd2[, 2])
  out[[length(out) + 1]] <- data.frame(cell = nm, kind = "nobs", name = "nobs", estimate = nobs(fit), se = NA)
  out[[length(out) + 1]] <- data.frame(cell = nm, kind = "pdHess", name = "pdHess", estimate = as.numeric(isTRUE(fit$sdr$pdHess)), se = NA)
}
write.table(do.call(rbind, out), args[2], sep = "\t", quote = FALSE, row.names = FALSE)
