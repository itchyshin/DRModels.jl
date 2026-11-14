# Reference fit for NB2 `spatial(1 | g, coords)` (drmTMB 0.8.1, ML), written to reference_drmtmb.csv.
# Usage: DRMTMB_LIB=<lib> Rscript ref_nb2_spatial_intercept.R <data_nb2_spatial.csv> <coords.csv> <out.csv>
# Both inputs are the committed files in ../struct_sd_counts/.
args <- commandArgs(TRUE)
if (nzchar(Sys.getenv("DRMTMB_LIB"))) .libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
d <- read.csv(args[1]); d$g <- factor(d$g, levels = unique(d$g))
cc <- read.csv(args[2]); coords <- as.matrix(cc[, c("easting", "northing")]); rownames(coords) <- cc$id
fit <- drmTMB(bf(y ~ x1 + x2 + x3 + spatial(1 | g, coords = coords), sigma = ~ 1), family = nbinom2(), data = d)
est <- as.numeric(fit$opt$par); se <- sqrt(diag(vcov(fit)))
stopifnot(length(est) == 6, length(se) == 5)   # 4 betas, log sigma, log SD(spatial); vcov lists the 5 coefficients
out <- data.frame(name = c("mu:(Intercept)", "mu:x1", "mu:x2", "mu:x3", "sigma:(Intercept)", "log_sd_spatial"),
                  estimate = est, se = c(se, NA), loglik = as.numeric(logLik(fit)),
                  is_converged = isTRUE(drmTMB::is_converged(fit)))
write.csv(out, args[3], row.names = FALSE)
print(out, digits = 12)
