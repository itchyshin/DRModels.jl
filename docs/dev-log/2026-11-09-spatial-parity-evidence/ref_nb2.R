.libPaths(c(path.expand("~/drm_campaign_20261108/Rlib"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
d <- read.csv("~/spsp/nb2cell.csv"); d$g <- factor(d$g, levels = unique(d$g))
cc <- read.csv("~/spsp/coords.csv"); coords <- as.matrix(cc[, c("easting","northing")]); rownames(coords) <- cc$id
fm <- bf(y ~ x1 + x2 + x3 + spatial(1 + x1 | p | g, coords = coords), sigma = ~ 1,
         sd(g, level = "spatial", coef = "(Intercept)") ~ z, sd(g, level = "spatial", coef = "x1") ~ z)
fit <- drmTMB(fm, family = nbinom2(), data = d)
cat("R default logLik", sprintf("%.4f", as.numeric(logLik(fit))), "conv", fit$opt$convergence, "pdHess", fit$sdr$pdHess, "\n")
