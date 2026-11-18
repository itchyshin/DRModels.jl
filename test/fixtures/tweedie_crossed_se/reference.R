# Regenerates the drmTMB reference numbers pinned in test/test_tweedie_crossed_se.jl.
# Usage: Rscript reference.R   (from this folder; needs drmTMB 0.8.1, engine "tmb")
# Data provenance: data.csv is columns y, x1, x2, x3, g, g2 of `data_tweedie_crossed.csv`
# from the 2026-11-10 gap-map rerun harness (a fixed 400-row simulated data set, one per
# family and random-effect kind; the generator is not part of this repository). The R fit is
# the harness cell `tweedie|crossed|lab0|q1|sd-none|cm0|ML`.
suppressPackageStartupMessages(library(drmTMB))
stopifnot(as.character(packageVersion("drmTMB")) == "0.8.1")
d <- read.csv("data.csv")
d$g <- factor(d$g, levels = unique(d$g)); d$g2 <- factor(d$g2)
fit <- drmTMB(bf(y ~ x1 + x2 + x3 + (1 | g) + (1 | g2)), family = tweedie(), data = d)
cat("logLik", sprintf("%.10f", as.numeric(logLik(fit))), "\n")
cat("opt convergence", fit$opt$convergence, " pdHess", fit$sdr$pdHess, "\n")
sm <- summary(fit$sdr, "fixed")   # R order: mu (4), sigma, nu, log SD g, log SD g2
print(sm, digits = 10)
