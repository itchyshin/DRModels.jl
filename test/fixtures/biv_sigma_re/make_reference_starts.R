# Regenerates reference_starts.tsv: a drmTMB 0.8.1 fit of the bivariate Gaussian with a correlated random
# intercept pair (1 | p | g) on sigma1 and sigma2 (Laplace, ML) on ss16.csv (60 rows, 12 unbalanced groups,
# columns id, g, x1, x2, y1, y2). On this data one or two of the three starts of the Julia route stop with an
# assertion from the line search (Julia 1.10.12 and 1.13.3 differ); the fit must still come back from the third start.
# Usage: Rscript make_reference_starts.R <ss16.csv> <out.tsv>
args <- commandArgs(TRUE)
suppressPackageStartupMessages(library(drmTMB))
d <- read.csv(args[1]); d$g <- factor(d$g)
b0 <- bf(mu1 = y1 ~ x1, mu2 = y2 ~ x2, sigma1 = ~ 1 + (1 | p | g), sigma2 = ~ 1 + (1 | p | g), rho12 = ~ 1)
fit <- drmTMB(b0, family = biv_gaussian(), data = d)
nm <- "ss16"
out <- list(data.frame(cell = nm, kind = "logLik", name = "logLik", estimate = as.numeric(logLik(fit)), se = NA))
sdr <- summary(fit$sdr); rn <- rownames(sdr)
keep <- rn %in% c("beta_mu1", "beta_mu2", "beta_sigma1", "beta_sigma2", "beta_rho12", "log_sd_sigma", "eta_cor_sigma")
sd2 <- sdr[keep & !duplicated(paste(rn, round(sdr[, 1], 8))), , drop = FALSE]
out[[2]] <- data.frame(cell = nm, kind = "par", name = make.unique(rownames(sd2)), estimate = sd2[, 1], se = sd2[, 2])
out[[3]] <- data.frame(cell = nm, kind = "nobs", name = "nobs", estimate = nobs(fit), se = NA)
out[[4]] <- data.frame(cell = nm, kind = "pdHess", name = "pdHess", estimate = as.numeric(isTRUE(fit$sdr$pdHess)), se = NA)
write.table(do.call(rbind, out), args[2], sep = "\t", quote = FALSE, row.names = FALSE)
