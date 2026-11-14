# Regenerates include_reference.csv for test/test_biv_re_include.jl.
#
# Run on a machine with drmTMB 0.8.1 installed (the reference was made on Totoro, R 4.6.1):
#
#   DRMTMB_LIB=<library with drmTMB 0.8.1> Rscript include_r.R \
#       ../fitted_modes_bivariate/data.csv include_reference.csv
#
# Bivariate Gaussian with `(1 | p | g)` in mu1 and mu2 and `miss_control(response = "include")`:
# rows with one response missing stay in the fit and contribute the marginal likelihood of the
# response that was observed. One row per data row; `fit1`, `fit2` are the two columns of
# fitted(fit) (random-effect modes included), `ll` is logLik(fit) and `nobs` is nobs(fit),
# both repeated only in the first row.
args <- commandArgs(TRUE)
lib <- Sys.getenv("DRMTMB_LIB")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
d <- read.csv(args[1], stringsAsFactors = FALSE, na.strings = "NA")
d$g <- factor(d$g)
fit <- drmTMB(bf(mu1 = y1m ~ x1 + (1 | p | g), mu2 = y2m ~ x2 + (1 | p | g),
                 sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1),
              family = biv_gaussian(), data = d, missing = miss_control(response = "include"))
f <- fitted(fit)
n <- nrow(f)
out <- data.frame(row = seq_len(n), fit1 = f[, 1], fit2 = f[, 2],
                  ll = c(as.numeric(logLik(fit)), rep(NA_real_, n - 1)),
                  nobs = c(nobs(fit), rep(NA_real_, n - 1)))
cat("OK", n, as.numeric(logLik(fit)), nobs(fit), "\n")
write.csv(out, args[2], row.names = FALSE)
