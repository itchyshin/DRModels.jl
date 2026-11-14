# Regenerates fitted_reference.csv and ranef_reference.csv for test/test_fitted_modes_bivariate_weights.jl.
#
# Run on a machine with drmTMB 0.8.1 installed (the reference was made with R 4.6.1):
#
#   DRMTMB_LIB=<library with drmTMB 0.8.1> Rscript make_reference.R \
#       ../biv_re_weights/data_weights.csv ../biv_re_weights/unbal_weights.csv .
#
# Bivariate Gaussian with `(1 | p | g)` in mu1 and mu2 and prior weights. One row per data row and
# cell. `fit1`, `fit2` are the two columns of fitted(fit) (group modes included); `ll` is logLik(fit)
# and `nobs` is nobs(fit), repeated only in the first row of the cell. `ranef_reference.csv` has
# the modes themselves, one row per group and cell (ranef(fit)$mu$terms).
# Cells: w_ml, w_reml (non-integer weights), wz_ml (zero weights, one whole group at zero),
# wunbal_ml (unbalanced groups), wincl_ml (weights plus miss_control(response = "include")).
args <- commandArgs(TRUE)
lib <- Sys.getenv("DRMTMB_LIB")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
d <- read.csv(args[1], na.strings = "NA"); d$g <- factor(d$g)
u <- read.csv(args[2]); u$g <- factor(u$g)
b0 <- bf(mu1 = y1 ~ x1 + (1 | p | g), mu2 = y2 ~ x2 + (1 | p | g), sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1)
bm <- bf(mu1 = y1m ~ x1 + (1 | p | g), mu2 = y2m ~ x2 + (1 | p | g), sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1)
inc <- miss_control(response = "include")
cells <- list(
  w_ml      = quote(drmTMB(b0, family = biv_gaussian(), data = d, weights = w)),
  w_reml    = quote(drmTMB(b0, family = biv_gaussian(), data = d, weights = w, REML = TRUE)),
  wz_ml     = quote(drmTMB(b0, family = biv_gaussian(), data = d, weights = wz)),
  wunbal_ml = quote(drmTMB(b0, family = biv_gaussian(), data = u, weights = uw)),
  wincl_ml  = quote(drmTMB(bm, family = biv_gaussian(), data = d, weights = w, missing = inc))
)
out <- list(); outr <- list()
for (nm in names(cells)) {
  fit <- suppressWarnings(eval(cells[[nm]]))
  f <- fitted(fit)
  tm <- ranef(fit)$mu$terms
  outr[[nm]] <- data.frame(cell = nm, group = names(tm[[1]]), u1 = sprintf("%.15g", as.numeric(tm[[1]])),
                           u2 = sprintf("%.15g", as.numeric(tm[[2]])))
  ll <- as.numeric(logLik(fit)); n <- nrow(f)
  out[[nm]] <- data.frame(cell = nm, row = seq_len(n), fit1 = sprintf("%.15g", f[, 1]), fit2 = sprintf("%.15g", f[, 2]),
                          ll = c(sprintf("%.15g", ll), rep(NA, n - 1)), nobs = c(nobs(fit), rep(NA, n - 1)))
  cat("OK", nm, n, ll, nobs(fit), "\n")
}
write.csv(do.call(rbind, out), file.path(args[3], "fitted_reference.csv"), row.names = FALSE, na = "NA")
write.csv(do.call(rbind, outr), file.path(args[3], "ranef_reference.csv"), row.names = FALSE)
