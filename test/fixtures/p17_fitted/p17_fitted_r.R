# Regenerates the R reference values for test/test_fitted_conditional.jl.
#
# Run on a machine with drmTMB 0.8.1 installed (the reference was made on a Linux server, library
# ~/drm_campaign_20261108/Rlib, R 4.6.1):
#
#   DRMTMB_LIB=~/drm_campaign_20261108/Rlib Rscript p17_fitted_r.R data.csv p17_fitted_r_reference.csv
#
# For each cell the script fits the model in drmTMB and writes, per used row, the drmTMB
# `fitted()` value (`cond`: random-effect modes included, the drmTMB default) and, where it
# can be formed, the population-level value (`pop`: every random effect at zero).
# `pop` is `predict(fit, newdata = d)` on the response scale, turned into E[Y] by the
# family's own mean formula (LogNormal: exp(mu + sigma^2 / 2); cumulative: the expected
# category score). It is NA for a cell whose used rows differ from `d`.
args <- commandArgs(TRUE)
.libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
d <- read.csv(args[1], stringsAsFactors = FALSE, na.strings = "NA")
for (v in c("g", "g2", "h", "f", "f2", "gna", "gu", "y_cat")) d[[v]] <- factor(d[[v]])
d$y_ordf <- factor(d$y_ordf, levels = c("o1", "o2", "o3", "o4"), ordered = TRUE)

cells <- list(
  list(id = "C|gaussian|(1|g)+weights",
       fit = function(d) drmTMB(bf(y_g ~ x1 + (1 | g), sigma ~ 1), family = gaussian(), data = d, weights = w)),
  list(id = "C|gaussian|(1|g)+weights+REML",
       fit = function(d) drmTMB(bf(y_g ~ x1 + (1 | g), sigma ~ 1), family = gaussian(), data = d, weights = w, REML = TRUE)),
  list(id = "H|meta|(1|g)",
       fit = function(d) drmTMB(bf(yi ~ x1 + (1 | g) + meta_V(V = vi), sigma ~ 1), family = gaussian(), data = d)),
  list(id = "H|meta|(1|g)+(1|g2)",
       fit = function(d) drmTMB(bf(yi ~ x1 + (1 | g) + (1 | g2) + meta_V(V = vi), sigma ~ 1), family = gaussian(), data = d)),
  list(id = "H|meta|(1|g)|REML",
       fit = function(d) drmTMB(bf(yi ~ x1 + (1 | g) + meta_V(V = vi), sigma ~ 1), family = gaussian(), data = d, REML = TRUE)),
  list(id = "I|cumul|mu=x1+(1|g)+(1|g2)",
       fit = function(d) drmTMB(bf(y_ord ~ x1 + (1 | g) + (1 | g2)), family = cumulative_logit(), data = d)),
  list(id = "J|poisson|(1|g)+(1|g2)+(1|h)",
       fit = function(d) drmTMB(bf(y_pois ~ x1 + (1 | g) + (1 | g2) + (1 | h)), family = stats::poisson(link = "log"), data = d)),
  list(id = "J|gaussian|mu=(1|g)|sigma=x1",
       fit = function(d) drmTMB(bf(y_g ~ x1 + (1 | g), sigma ~ x1), family = gaussian(), data = d)),
  list(id = "J|lognormal|mu=(1|g)|sigma=x1",
       fit = function(d) drmTMB(bf(y_ln ~ x1 + (1 | g), sigma ~ x1), family = lognormal(), data = d)),
  list(id = "J|lognormal|sigma=x1+(1|g)",
       fit = function(d) drmTMB(bf(y_ln ~ x1, sigma ~ x1 + (1 | g)), family = lognormal(), data = d)),
  list(id = "J|gaussian|mu=(1|g)|sigma=(1|g2)",
       fit = function(d) drmTMB(bf(y_g ~ x1 + (1 | g), sigma ~ x1 + (1 | g2)), family = gaussian(), data = d)),
  list(id = "J|gaussian|group=numeric code gn",
       fit = function(d) drmTMB(bf(y_g ~ x1 + (1 | gn), sigma ~ 1), family = gaussian(), data = d)),
  list(id = "J|gaussian|group=unbalanced with singletons gu",
       fit = function(d) drmTMB(bf(y_g ~ x1 + (1 | gu), sigma ~ 1), family = gaussian(), data = d)),
  list(id = "J|gaussian|group has 2 levels (1|f2)",
       fit = function(d) drmTMB(bf(y_g ~ x1 + (1 | f2), sigma ~ 1), family = gaussian(), data = d)),
  list(id = "J|gaussian|slopes on two different groups",
       fit = function(d) drmTMB(bf(y_g ~ x1 + (1 | g) + (0 + x1 | g) + (0 + x2 | g2), sigma ~ 1), family = gaussian(), data = d)),
  list(id = "K|gaussian|NA in both + (1|g)",
       fit = function(d) drmTMB(bf(y_gm ~ x1m + (1 | g), sigma ~ 1), family = gaussian(), data = d)),
  list(id = "M|gaussian|mu=x1+(1|g)|entry-points",
       fit = function(d) drmTMB(bf(y_g ~ x1 + (1 | g), sigma ~ 1), family = gaussian(), data = d)),
  list(id = "K|gaussian|NA in response",
       fit = function(d) drmTMB(bf(y_gm ~ x1, sigma ~ 1), family = gaussian(), data = d)),
  list(id = "K|gaussian|NA in response|REML",
       fit = function(d) drmTMB(bf(y_gm ~ x1, sigma ~ 1), family = gaussian(), data = d, REML = TRUE)),
  list(id = "K|gaussian|NA in response + weights",
       fit = function(d) drmTMB(bf(y_gm ~ x1, sigma ~ 1), family = gaussian(), data = d, weights = w)),
  list(id = "K|poisson|NA in response",
       fit = function(d) drmTMB(bf(y_pm ~ x1), family = stats::poisson(link = "log"), data = d))
)

pop_value <- function(fit, d, id) {
  if (grepl("cumul", id)) return(drmTMB:::ordinal_expected_score(fit, newdata = d))
  mu <- as.numeric(predict(fit, newdata = d, dpar = "mu", type = "response"))
  if (grepl("lognormal", id)) {
    sg <- as.numeric(predict(fit, newdata = d, dpar = "sigma", type = "response"))
    return(exp(mu + sg^2 / 2))
  }
  mu
}

out <- list()
for (cl in cells) {
  fit <- suppressWarnings(cl$fit(d))
  cond <- as.numeric(fitted(fit))
  pop <- tryCatch(pop_value(fit, d, cl$id), error = function(e) rep(NA_real_, length(cond)))
  if (length(pop) != length(cond)) pop <- rep(NA_real_, length(cond))
  out[[cl$id]] <- data.frame(cell = cl$id, row = seq_along(cond), cond = cond, pop = pop)
  cat("OK", cl$id, length(cond), "\n")
}
write.csv(do.call(rbind, out), args[2], row.names = FALSE)
