# Regenerates the R reference for test/test_predict_missing_predictor.jl.
# Usage (repository root): Rscript test/fixtures/predict_miss/gen_fixtures.R
# Needs drmTMB 0.8.1 (the gap-map reference version).  Reads data.csv (a column
# subset of the gap-map v2 data, `NA` = missing) and writes
#   reference.csv : cell, call, id, value   (predictions, one row per row id)
#   fits.csv      : cell, logLik, nobs, term, estimate, se  (fixed effects of `mu`)
#   refusals.csv  : cell, message  (R's error for predict(fit, newdata = <data with NA>))
suppressPackageStartupMessages(library(drmTMB))
stopifnot(as.character(packageVersion("drmTMB")) == "0.8.1")
here <- if (length(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))) {
  dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
} else "."
d <- read.csv(file.path(here, "data.csv"), na.strings = "NA", stringsAsFactors = FALSE)
d$g <- factor(d$g)
d$y_ordf <- factor(d$y_ordf, levels = c("o1", "o2", "o3", "o4"), ordered = TRUE)

cells <- list(
  K_gauss_mu      = quote(drmTMB(bf(y_g ~ x1m, sigma ~ 1), family = gaussian(), data = d)),
  K_gauss_sigma   = quote(drmTMB(bf(y_g ~ x1, sigma ~ x1m), family = gaussian(), data = d)),
  K_pois          = quote(drmTMB(bf(y_pois ~ x1m), family = stats::poisson(link = "log"), data = d)),
  K_nb2           = quote(drmTMB(bf(y_nb ~ x1m, sigma ~ 1), family = nbinom2(), data = d)),
  K_gauss_both_re = quote(drmTMB(bf(y_gm ~ x1m + (1 | g), sigma ~ 1), family = gaussian(), data = d)),
  K_pois_both_re  = quote(drmTMB(bf(y_pm ~ x1m + (1 | g)), family = stats::poisson(link = "log"), data = d))
)
ref <- list(); fits <- list(); ref_msg <- list()
add <- function(cell, call, ids, v) ref[[length(ref) + 1L]] <<-
  data.frame(cell = cell, call = call, id = ids, value = as.numeric(v))
for (nm in names(cells)) {
  fit <- suppressWarnings(eval(cells[[nm]]))
  b <- fixef(fit)
  mu <- if (is.list(b)) b[["mu"]] else b
  fits[[nm]] <- data.frame(cell = nm, logLik = as.numeric(logLik(fit)), nobs = nobs(fit),
                           term = names(mu), estimate = as.numeric(mu),
                           se = sqrt(diag(vcov(fit)))[paste0("mu:", names(mu))])
  ok <- !is.na(d$x1m)
  nd <- d[ok, c("id", "x1", "x1m", "g")]
  add(nm, "newdata_response", nd$id, predict(fit, newdata = nd))
  add(nm, "newdata_link", nd$id, predict(fit, newdata = nd, type = "link"))
  msg <- tryCatch({ predict(fit, newdata = d); "NO-ERROR" }, error = function(e) gsub("\\s+", " ", conditionMessage(e)))
  ref_msg[[nm]] <- data.frame(cell = nm, message = msg)
  if (nm == "K_gauss_mu") add(nm, "fit_rows", d$id[ok], predict(fit))   # no newdata: the fitted rows
  if (nm == "K_gauss_sigma") add(nm, "newdata_full_with_NA", d$id, predict(fit, newdata = d))
}
# Ordinal response given as a label (ordered factor in R); predict(fit, newdata) is called with the label column too.
fit <- drmTMB(bf(y_ordf ~ x1), family = cumulative_logit(), data = d)
fits[["I_ord_label"]] <- data.frame(cell = "I_ord_label", logLik = as.numeric(logLik(fit)), nobs = nobs(fit),
                                    term = names(fixef(fit)[["mu"]]), estimate = as.numeric(fixef(fit)[["mu"]]),
                                    se = sqrt(diag(vcov(fit)))[paste0("mu:", names(fixef(fit)[["mu"]]))])
add("I_ord_label", "fit_rows", d$id, predict(fit))
idx <- c(5, 1, 9)
add("I_ord_label", "newdata_with_label", d$id[idx], predict(fit, newdata = d[idx, , drop = FALSE]))
add("I_ord_label", "newdata_all_with_label", d$id, predict(fit, newdata = d))
write.csv(do.call(rbind, ref), file.path(here, "reference.csv"), row.names = FALSE)
write.csv(do.call(rbind, fits), file.path(here, "fits.csv"), row.names = FALSE)
write.csv(do.call(rbind, ref_msg), file.path(here, "refusals.csv"), row.names = FALSE)
