.libPaths(c(path.expand("~/drm_campaign_20261108/Rlib"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
d <- read.csv("~/spsp/pq2.csv"); d$g <- factor(d$g, levels = unique(d$g))
cc <- read.csv("~/spsp/coords.csv"); coords <- as.matrix(cc[, c("easting","northing")]); rownames(coords) <- cc$id
fm <- bf(y ~ x1 + x2 + x3 + spatial(1 + x1 | p | g, coords = coords),
         sd(g, level = "spatial", coef = "(Intercept)") ~ z, sd(g, level = "spatial", coef = "x1") ~ z)
for (ms in c(1, 8)) {
  fit <- tryCatch(if (ms == 1) drmTMB(fm, family = poisson(), data = d) else drmTMB(fm, family = poisson(), data = d, control = list(multi_start = ms)), error = function(e) e)
  if (inherits(fit, "error")) { cat("multi_start", ms, "ERROR", conditionMessage(fit), "\n"); next }
  cat("multi_start", ms, "logLik", sprintf("%.4f", as.numeric(logLik(fit))), "conv", fit$opt$convergence, "pdHess", fit$sdr$pdHess, "\n")
}
