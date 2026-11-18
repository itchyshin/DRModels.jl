# R reference values for test/test_laplace_fitted_modes.jl: three cells of the opt-in Laplace
# `fitted()` check (Poisson on an unbalanced group, Binomial on a numeric group code, Poisson with a
# missing response and a missing predictor).
#
#   DRMTMB_LIB=~/drm_campaign_20261108/Rlib Rscript laplace_fitted_modes_r.R data.csv laplace_fitted_modes_reference.csv
#
# Same layout as p17_fitted_review_r.R: per used row, the drmTMB 0.8.1 `fitted()` value (`cond`, random-effect
# modes included) and `pop`, the population value (every random effect at zero; NA when a row has a missing
# predictor). drmTMB integrates a random intercept by the Laplace approximation, which is what
# `marginal = :Laplace` fits in DRModels.jl.
args <- commandArgs(TRUE)
.libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
d <- read.csv(args[1], stringsAsFactors = FALSE, na.strings = "NA")
for (v in c("g", "gu")) d[[v]] <- factor(d[[v]])
pois <- stats::poisson(link = "log")
cells <- list(
  list("L|poisson|gu", function(d) drmTMB(bf(y_pois ~ x1 + (1 | gu)), family = pois, data = d)),
  list("L|binomial|gn", function(d) drmTMB(bf(y_bin ~ x1 + (1 | gn)), family = stats::binomial(link = "logit"), data = d)),
  list("L|poisson|NA in both", function(d) drmTMB(bf(y_pm ~ x1m + (1 | g)), family = pois, data = d))
)
out <- list()
for (cl in cells) {
  fit <- suppressWarnings(cl[[2]](d))
  cond <- as.numeric(fitted(fit))
  pop <- tryCatch(as.numeric(predict(fit, newdata = d, dpar = "mu", type = "response")), error = function(e) rep(NA_real_, length(cond)))
  if (length(pop) != length(cond)) pop <- pop[!is.na(pop)]
  cat("OK", cl[[1]], length(cond), "logLik", as.numeric(logLik(fit)), "\n")
  out[[cl[[1]]]] <- data.frame(cell = cl[[1]], row = seq_along(cond), cond = cond, pop = pop)
}
write.csv(do.call(rbind, out), args[2], row.names = FALSE)
