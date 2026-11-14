# Regenerates reference.tsv: drmTMB 0.8.1 fits of the bivariate Gaussian with a correlated random
# intercept pair (1 | p | g) on mu1 and mu2, on three small simulated data sets (d80.csv, d132.csv, d135.csv;
# made by make_data.jl). On these cells the default start of DRModels.jl slides to group SDs of 0 (logLik
# lower by 3e-3 to 7e-2) while drmTMB finds a higher mode with a group correlation of +1 (d80) or -1 (d132,
# d135). d80 and d132 are run with ML and REML, d135 with ML only.
# Usage: Rscript make_reference.R <dir with the d*.csv files> <out.tsv>
args <- commandArgs(TRUE)
suppressPackageStartupMessages(library(drmTMB))
b0 <- bf(mu1 = y1 ~ x1 + (1 | p | g), mu2 = y2 ~ x2 + (1 | p | g), sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1)
out <- list()
cells <- list(c(80, 0), c(80, 1), c(132, 0), c(132, 1), c(135, 0))
for (cl in cells) {
  s <- cl[1]; reml <- cl[2] == 1
  d <- read.csv(file.path(args[1], paste0("d", s, ".csv"))); d$g <- factor(d$g)
  nm <- paste0("d", s, "_", ifelse(reml, "reml", "ml"))
  fit <- drmTMB(b0, family = biv_gaussian(), data = d, REML = reml)
  out[[length(out) + 1]] <- data.frame(cell = nm, kind = "logLik", name = "logLik", estimate = as.numeric(logLik(fit)), se = NA)
  sdr <- summary(fit$sdr); rn <- rownames(sdr)
  keep <- rn %in% c("beta_mu1", "beta_mu2", "beta_sigma1", "beta_sigma2", "beta_rho12", "log_sd_mu", "eta_cor_mu")
  sd2 <- sdr[keep & !duplicated(paste(rn, round(sdr[, 1], 8))), , drop = FALSE]   # summary() lists each parameter twice
  out[[length(out) + 1]] <- data.frame(cell = nm, kind = "par", name = make.unique(rownames(sd2)), estimate = sd2[, 1], se = sd2[, 2])
  out[[length(out) + 1]] <- data.frame(cell = nm, kind = "nobs", name = "nobs", estimate = nobs(fit), se = NA)
  out[[length(out) + 1]] <- data.frame(cell = nm, kind = "pdHess", name = "pdHess", estimate = as.numeric(isTRUE(fit$sdr$pdHess)), se = NA)
}
write.table(do.call(rbind, out), args[2], sep = "\t", quote = FALSE, row.names = FALSE)
