# Second set of R reference values for test/test_fitted_conditional.jl (review fixes for #1324).
#
#   DRMTMB_LIB=~/drm_campaign_20261108/Rlib Rscript p17_fitted_review_r.R data.csv tree16.newick p17_fitted_review_reference.csv
#
# Same layout as p17_fitted_r.R: per used row, the drmTMB `fitted()` value (`cond`, random-effect modes
# included). The MR cells put a missing response on the rows where `y_gm` is NA (16 rows) and
# check that the used rows only are returned. `pop` (W cells only) is the population value, to tell a
# difference between the two fits from a difference in the modes.
args <- commandArgs(TRUE)
.libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages({library(drmTMB); library(ape)})
d <- read.csv(args[1], stringsAsFactors = FALSE, na.strings = "NA")
for (v in c("g", "g2", "h", "f", "f2", "gna", "gu", "y_cat")) d[[v]] <- factor(d[[v]])
tree <- read.tree(args[2])
na <- is.na(d$y_gm)
for (v in c("y_g", "y_ln", "y_pois", "y_gam")) d[[paste0(v, "_m")]] <- ifelse(na, NA, d[[v]])
pois <- stats::poisson(link = "log")

cells <- list(
  list("W|beta|(1|g)", function(d) drmTMB(bf(y_beta ~ x1 + (1 | g), sigma ~ 1), family = beta_family(), data = d)),
  list("W|gamma|(1|g)", function(d) drmTMB(bf(y_gam ~ x1 + (1 | g), sigma ~ 1), family = stats::Gamma(link = "log"), data = d)),
  list("W|nb2|(1|g)", function(d) drmTMB(bf(y_nb ~ x1 + (1 | g), sigma ~ 1), family = nbinom2(), data = d)),
  list("W|student|(1|g)", function(d) drmTMB(bf(y_st ~ x1 + (1 | g), sigma ~ 1, nu ~ 1), family = student(), data = d)),
  list("W|tweedie|(1|g)", function(d) drmTMB(bf(y_tw ~ x1 + (1 | g), sigma ~ 1, nu ~ 1), family = tweedie(), data = d)),
  list("W|betabinomial|(1|g)", function(d) drmTMB(bf(cbind(s, fl) ~ x1 + (1 | g), sigma ~ 1), family = beta_binomial(), data = d)),
  list("W|cumul|(1|g)", function(d) drmTMB(bf(y_ord ~ x1 + (1 | g)), family = cumulative_logit(), data = d)),
  list("W|poisson|offset+(1|g)", function(d) drmTMB(bf(y_pois ~ x1 + offset(off) + (1 | g)), family = pois, data = d)),
  list("W|nb2|offset+(1|g)", function(d) drmTMB(bf(y_nb ~ x1 + offset(off) + (1 | g), sigma ~ 1), family = nbinom2(), data = d)),
  list("MR|gaussian|(1|g)", function(d) drmTMB(bf(y_g_m ~ x1 + (1 | g), sigma ~ 1), family = gaussian(), data = d)),
  list("MR|lognormal|(1|g)", function(d) drmTMB(bf(y_ln_m ~ x1 + (1 | g), sigma ~ 1), family = lognormal(), data = d)),
  list("MR|poisson|(1|g)", function(d) drmTMB(bf(y_pois_m ~ x1 + (1 | g)), family = pois, data = d)),
  list("MR|gamma|(1|g)", function(d) drmTMB(bf(y_gam_m ~ x1 + (1 | g), sigma ~ 1), family = stats::Gamma(link = "log"), data = d)),
  list("MR|poisson|(1|g)+(1|g2)+(1|h)", function(d) drmTMB(bf(y_pois_m ~ x1 + (1 | g) + (1 | g2) + (1 | h)), family = pois, data = d)),
  list("MR|gaussian|phylo(1|g)", function(d) drmTMB(bf(y_g_m ~ x1 + phylo(1 | g, tree = tree), sigma ~ 1), family = gaussian(), data = d)),
  list("MR|lognormal|phylo(1|g)", function(d) drmTMB(bf(y_ln_m ~ x1 + phylo(1 | g, tree = tree), sigma ~ 1), family = lognormal(), data = d))
)
out <- list()
for (cl in cells) {
  fit <- tryCatch(suppressWarnings(cl[[2]](d)), error = function(e) { cat("FAIL", cl[[1]], conditionMessage(e), "\n"); NULL })
  if (is.null(fit)) next
  cond <- as.numeric(fitted(fit))
  # `pop`: every random effect at zero, for the W cells (same rows as `d`); NA for the others
  pop <- rep(NA_real_, length(cond))
  if (startsWith(cl[[1]], "W|")) {
    pop <- tryCatch({
      if (grepl("cumul", cl[[1]])) drmTMB:::ordinal_expected_score(fit, newdata = d)
      else as.numeric(predict(fit, newdata = d, dpar = "mu", type = "response"))
    }, error = function(e) rep(NA_real_, length(cond)))
    if (length(pop) != length(cond)) pop <- rep(NA_real_, length(cond))
  }
  out[[cl[[1]]]] <- data.frame(cell = cl[[1]], row = seq_along(cond), cond = cond, pop = pop)
  cat("OK", cl[[1]], length(cond), "\n")
}
write.csv(do.call(rbind, out), args[3], row.names = FALSE)
