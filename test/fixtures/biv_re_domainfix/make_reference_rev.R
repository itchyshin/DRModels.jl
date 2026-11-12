# Regenerates reference_rev.tsv: drmTMB 0.8.1 fits of the bivariate Gaussian with a correlated random
# intercept pair (1 | p | g) on mu1 and mu2, on the three data sets made by make_data_rev.jl
# (r7080.csv REML and ML, r7083.csv REML and ML, r7084.csv REML and ML). On these cells DRModels.jl crashes in the
# line search (AssertionError: isfinite(phi_c) ...) if a rejected trial returns Inf or NaN instead of 1e18
# (7080 and 7083 REML on Julia 1.13.3, 7084 on Julia 1.10.12). r7080 ML crashed in an earlier version
# of the fix, which returned Inf from some rejections and 1e18 from others (the same assertion).
# Usage: Rscript make_reference_rev.R <dir with the r*.csv files> <out.tsv>
args <- commandArgs(TRUE)
suppressPackageStartupMessages(library(drmTMB))
b0 <- bf(mu1 = y1 ~ x1 + (1 | p | g), mu2 = y2 ~ x2 + (1 | p | g), sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1)
out <- list()
cells <- list(c(7080, 1), c(7080, 0), c(7083, 1), c(7083, 0), c(7084, 1), c(7084, 0))
for (cl in cells) {
  s <- cl[1]; reml <- cl[2] == 1
  d <- read.csv(file.path(args[1], paste0("r", s, ".csv"))); d$g <- factor(d$g)
  nm <- paste0("r", s, "_", ifelse(reml, "reml", "ml"))
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
