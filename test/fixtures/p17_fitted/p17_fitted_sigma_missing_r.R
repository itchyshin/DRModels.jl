# R reference values for the fitted() cells on the sigma and weights routes with a missing response
# (MR) and with a dropped predictor row (DP). Used by test/test_fitted_modes_sigma_weights.jl.
#
#   DRMTMB_LIB=~/drm_campaign_20261108/Rlib Rscript p17_fitted_sigma_missing_r.R data.csv p17_fitted_sigma_missing_reference.csv
#
# Same layout as p17_fitted_review_r.R: per used row, the drmTMB 0.8.1 `fitted()` value (`cond`).
# The missing responses sit on the 16 rows where `y_gm` is NA; the dropped predictor rows are the
# rows where `x1m` is NA.
args <- commandArgs(TRUE)
.libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
d <- read.csv(args[1], stringsAsFactors = FALSE, na.strings = "NA")
for (v in c("g", "g2", "h", "f", "f2", "gna", "gu", "y_cat")) d[[v]] <- factor(d[[v]])
na <- is.na(d$y_gm)
for (v in c("y_g", "y_ln", "yi")) d[[paste0(v, "_m")]] <- ifelse(na, NA, d[[v]])

cells <- list(
  list("MR|lognormal|sigma=x1+(1|g)", function(d) drmTMB(bf(y_ln_m ~ x1, sigma ~ x1 + (1 | g)), family = lognormal(), data = d)),
  list("MR|gaussian|mu=(1|g)|sigma=(1|g2)", function(d) drmTMB(bf(y_g_m ~ x1 + (1 | g), sigma ~ x1 + (1 | g2)), family = gaussian(), data = d)),
  list("MR|gaussian|(1|g)+weights", function(d) drmTMB(bf(y_g_m ~ x1 + (1 | g), sigma ~ 1), family = gaussian(), data = d, weights = w)),
  list("MR|gaussian|(1|g)+weights+REML", function(d) drmTMB(bf(y_g_m ~ x1 + (1 | g), sigma ~ 1), family = gaussian(), data = d, weights = w, REML = TRUE)),
  list("MR|meta|(1|g)|REML", function(d) drmTMB(bf(yi_m ~ x1 + (1 | g) + meta_V(V = vi), sigma ~ 1), family = gaussian(), data = d, REML = TRUE)),
  list("DP|lognormal|sigma=x1m+(1|g)", function(d) drmTMB(bf(y_ln ~ x1m, sigma ~ x1m + (1 | g)), family = lognormal(), data = d)),
  list("DP|gaussian|mu=(1|g)|sigma=(1|g2)", function(d) drmTMB(bf(y_g ~ x1m + (1 | g), sigma ~ x1m + (1 | g2)), family = gaussian(), data = d)),
  list("DP|gaussian|(1|g)+weights", function(d) drmTMB(bf(y_g ~ x1m + (1 | g), sigma ~ 1), family = gaussian(), data = d, weights = w)),
  list("DP|gaussian|(1|g)+weights+REML", function(d) drmTMB(bf(y_g ~ x1m + (1 | g), sigma ~ 1), family = gaussian(), data = d, weights = w, REML = TRUE)),
  list("DP|meta|(1|g)|REML", function(d) drmTMB(bf(yi ~ x1m + (1 | g) + meta_V(V = vi), sigma ~ 1), family = gaussian(), data = d, REML = TRUE)),
  list("MRDP|lognormal|sigma=x1m+(1|g)", function(d) drmTMB(bf(y_ln_m ~ x1m, sigma ~ x1m + (1 | g)), family = lognormal(), data = d))
)
out <- list()
for (cl in cells) {
  fit <- tryCatch(suppressWarnings(cl[[2]](d)), error = function(e) { cat("FAIL", cl[[1]], conditionMessage(e), "\n"); NULL })
  if (is.null(fit)) next
  cond <- as.numeric(fitted(fit))
  out[[cl[[1]]]] <- data.frame(cell = cl[[1]], row = seq_along(cond), cond = cond, loglik = as.numeric(logLik(fit)))
  cat("OK", cl[[1]], length(cond), "\n")
}
write.csv(do.call(rbind, out), args[2], row.names = FALSE)
