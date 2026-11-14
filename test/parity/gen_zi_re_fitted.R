# gen_zi_re_fitted.R -- drmTMB 0.8.1 `fitted()` for the zero-inflated Poisson (1 | g) cell of
# test/test_zi_re_laplace_parity.jl.
#
#   DRMTMB_LIB=<library with drmTMB 0.8.1> Rscript test/parity/gen_zi_re_fitted.R
#
# Writes test/fixtures/zi_re_parity/zip_g/fitted_r.csv with, per row of data.csv, the drmTMB
# `fitted()` value (`cond`: random-effect modes included) and the value with the random effect at
# zero (`pop`: (1 - zero-inflation probability) * exp(mu), from `predict()` on the response scale).
.libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
stopifnot(as.character(utils::packageVersion("drmTMB")) == "0.8.1")
d <- read.csv(file.path("test", "fixtures", "zi_re_parity", "zip_g", "data.csv"), stringsAsFactors = FALSE)
d$g <- factor(d$g)
fit <- drmTMB(bf(y ~ x + (1 | g), zi ~ 1), family = stats::poisson(), data = d)
mu <- as.numeric(predict(fit, newdata = d, dpar = "mu", type = "response"))
zi <- as.numeric(predict(fit, newdata = d, dpar = "zi", type = "response"))
out <- data.frame(row = seq_len(nrow(d)), cond = as.numeric(fitted(fit)), pop = (1 - zi) * mu)
write.csv(out, file.path("test", "fixtures", "zi_re_parity", "zip_g", "fitted_r.csv"), row.names = FALSE)
cat(sprintf("rows %d  logLik %.10f  max|cond-pop| %.4f\n", nrow(out), as.numeric(logLik(fit)), max(abs(out$cond - out$pop))))
