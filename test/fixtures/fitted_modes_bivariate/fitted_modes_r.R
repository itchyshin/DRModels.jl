# Regenerates data.csv and fitted_modes_reference.csv for test/test_fitted_modes_bivariate.jl.
#
# Run on a machine with drmTMB 0.8.1 installed (the reference was made on Totoro, R 4.6.1):
#
#   DRMTMB_LIB=<library with drmTMB 0.8.1> Rscript fitted_modes_r.R <data_v2.csv> .
#
# `data_v2.csv` is the 160-row data set of the parity harness (columns id, g, x1, x2, y1, y2,
# y1m, y2m are kept here as data.csv). One row per used data row and cell. `fit1` and `fit2`
# are the two columns of drmTMB's fitted(fit) matrix: the group effects (random-effect modes)
# are included. `ranef_reference.csv` has the modes themselves, one row per group and cell
# (`u1`, `u2` from ranef(fit)$mu$terms).
# `ll` repeats logLik(fit) in the first row of each cell.
args <- commandArgs(TRUE)
lib <- Sys.getenv("DRMTMB_LIB")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
d <- read.csv(args[1], stringsAsFactors = FALSE, na.strings = "NA")
d <- d[, c("id", "g", "x1", "x2", "y1", "y2", "y1m", "y2m")]
write.csv(d, file.path(args[2], "data.csv"), row.names = FALSE, na = "NA")
d$g <- factor(d$g)
re <- function(mu1, mu2, s1 = ~ 1, s2 = ~ 1, rho = ~ 1, ...)
  function(d) drmTMB(bf(mu1 = mu1, mu2 = mu2, sigma1 = s1, sigma2 = s2, rho12 = rho),
                     family = biv_gaussian(), data = d, ...)
cells <- list(
  list(id = "G|gauss|RE=(1|p|g)|rho=1",
       fit = re(y1 ~ x1 + (1 | p | g), y2 ~ x2 + (1 | p | g))),
  list(id = "G|gauss|RE=(1|p|g)|rho=x1",
       fit = re(y1 ~ x1 + (1 | p | g), y2 ~ x2 + (1 | p | g), rho = ~ x1)),
  list(id = "G|gauss|RE=(1|p|g)|sigma1=x1",
       fit = re(y1 ~ x1 + (1 | p | g), y2 ~ x2 + (1 | p | g), s1 = ~ x1)),
  list(id = "G|gauss|RE=(1|p|g)|REML",
       fit = re(y1 ~ x1 + (1 | p | g), y2 ~ x2 + (1 | p | g), REML = TRUE))
)
out <- list(); outr <- list()
for (cl in cells) {
  fit <- suppressWarnings(cl$fit(d))
  f <- fitted(fit)
  tm <- ranef(fit)$mu$terms
  outr[[cl$id]] <- data.frame(cell = cl$id, group = names(tm[[1]]), u1 = as.numeric(tm[[1]]), u2 = as.numeric(tm[[2]]))
  ll <- as.numeric(logLik(fit))
  out[[cl$id]] <- data.frame(cell = cl$id, row = seq_len(nrow(f)), fit1 = f[, 1], fit2 = f[, 2],
                             ll = c(ll, rep(NA_real_, nrow(f) - 1)))
  cat("OK", cl$id, nrow(f), ll, "\n")
}
write.csv(do.call(rbind, out), file.path(args[2], "fitted_modes_reference.csv"), row.names = FALSE)
write.csv(do.call(rbind, outr), file.path(args[2], "ranef_reference.csv"), row.names = FALSE)
