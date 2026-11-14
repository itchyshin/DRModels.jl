# Regenerates fitted_modes_reference.csv for test/test_fitted_modes_zi_ordinal.jl.
#
# Run on a machine with drmTMB 0.8.1 installed (the reference was made with R 4.6.1):
#
#   DRMTMB_LIB=<library with drmTMB 0.8.1> Rscript fitted_modes_r.R \
#       ../p17_fitted/data.csv fitted_modes_reference.csv
#
# One row per used data row and cell. `cond` is drmTMB's fitted(fit): the random-effect
# modes are included. For the zero-inflated models drmTMB returns (1 - zi) * mu, for the
# zero-truncated NB2 model mu / (1 - P(0)), for the ordinal model the expected category
# score; mu is the conditional mean (modes included) in every case.
# `ll` repeats the log-likelihood of the fit in the first row of each cell.
args <- commandArgs(TRUE)
lib <- Sys.getenv("DRMTMB_LIB")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
d <- read.csv(args[1], stringsAsFactors = FALSE, na.strings = "NA")
for (v in c("g", "g2", "h", "f", "f2", "gna", "gu", "y_cat")) d[[v]] <- factor(d[[v]])

cells <- list(
  list(id = "E|zip|mu=(1|g)|zi=1",
       fit = function(d) drmTMB(bf(y_zip ~ x1 + (1 | g), zi ~ 1), family = stats::poisson(link = "log"), data = d)),
  list(id = "E|zinb|mu=(1|g)|zi=1",
       fit = function(d) drmTMB(bf(y_zinb ~ x1 + (1 | g), sigma ~ 1, zi ~ 1), family = nbinom2(), data = d)),
  list(id = "E|tnb|mu=(1|g)",
       fit = function(d) drmTMB(bf(y_tnb ~ x1 + (1 | g), sigma ~ 1), family = truncated_nbinom2(), data = d)),
  list(id = "I|cumul|mu=x1+(1|g)+(0+x1|g)",
       fit = function(d) drmTMB(bf(y_ord ~ x1 + (1 | g) + (0 + x1 | g)), family = cumulative_logit(), data = d))
)
out <- list()
for (cl in cells) {
  fit <- suppressWarnings(cl$fit(d))
  cond <- as.numeric(fitted(fit))
  ll <- as.numeric(logLik(fit))
  out[[cl$id]] <- data.frame(cell = cl$id, row = seq_along(cond), cond = cond,
                             ll = c(ll, rep(NA_real_, length(cond) - 1)))
  cat("OK", cl$id, length(cond), ll, "\n")
}
write.csv(do.call(rbind, out), args[2], row.names = FALSE)
