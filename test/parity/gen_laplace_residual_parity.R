# gen_laplace_residual_parity.R -- native drmTMB 0.8.1 reference `residuals()` for three fits whose
# `fitted()` values are pinned in test_laplace_fitted_modes_more.jl:
#   * Binomial `(1 | g)` + `sd(g) ~ gx`, logit and probit (data and estimates as in
#     gen_binomial_sd_fitted_parity.R);
#   * ZeroOneBeta `(1 | g)` with `sigma ~ x1` (data as in gen_laplace_sigma_formula_parity.R).
#
#   Rscript test/parity/gen_laplace_residual_parity.R <binomial sd data.csv> <zob data.csv> <output dir>
#
# Output: one ref_<cell>.toml per model. Besides `theta` and `logLik` it holds drmTMB's
#   response  = residuals(fit, "response"),
#   pearson   = residuals(fit, "pearson"),
#   q_lower, q_upper = qnorm of the lower and upper end of the interval a randomised quantile residual
#               is drawn from (F(y-) and F(y); equal for a continuous row), from fitted_distribution(fit),
# and the rows' response values. drmTMB's randomised draw cannot be reproduced from another language, so
# the test compares the deterministic ends of the interval and checks that a draw falls inside it.
# Numbers and simulated data only. A fit with max |gradient| >= 1e-6 is reported and NOT written.

if (nzchar(Sys.getenv("DRMTMB_LIB"))) .libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
stopifnot(as.character(utils::packageVersion("drmTMB")) == "0.8.1")
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3)
dir.create(args[3], recursive = TRUE, showWarnings = FALSE)
# TOML spells the infinities `inf` and `-inf` (the lower end of the interval at y = 0 is qnorm(0)).
fmt <- function(x) {
  s <- trimws(formatC(x, digits = 17, format = "g"))
  s[is.infinite(x)] <- ifelse(x[is.infinite(x)] > 0, "inf", "-inf")
  paste0("[", paste(s, collapse = ", "), "]")
}

interval_ends <- function(f) {
  fd <- drmTMB::fitted_distribution(f)
  y <- f$model$y
  upper <- fd$p(y)
  lower <- if (isTRUE(fd$discrete)) {
    ifelse(y <= 0, 0, fd$p(y - 1))
  } else if (isTRUE(fd$has_atom)) {
    drmTMB:::drm_atom_left_limit(fd, y, upper)
  } else upper
  list(lower = stats::qnorm(lower), upper = stats::qnorm(upper))
}

write_cell <- function(id, f) {
  par <- f$opt$par
  gr <- as.numeric(f$obj$gr(par))
  ends <- interval_ends(f)
  cat(sprintf("%s n=%d max|grad|=%.2e ll=%.10f\n", id, length(f$model$y), max(abs(gr)), as.numeric(logLik(f))))
  if (max(abs(gr)) >= 1e-6) { cat("  -> NOT WRITTEN\n"); return(invisible()) }
  writeLines(c(
    sprintf("cell = \"%s\"", id),
    sprintf("drmTMB_version = \"%s\"", as.character(utils::packageVersion("drmTMB"))),
    sprintf("loglik = %s", formatC(as.numeric(logLik(f)), digits = 17, format = "g")),
    sprintf("max_grad = %s", formatC(max(abs(gr)), digits = 6, format = "e")),
    sprintf("theta = %s", fmt(as.numeric(par))),
    sprintf("fitted = %s", fmt(as.numeric(fitted(f)))),
    sprintf("response = %s", fmt(as.numeric(residuals(f, type = "response")))),
    sprintf("pearson = %s", fmt(as.numeric(residuals(f, type = "pearson")))),
    sprintf("q_lower = %s", fmt(ends$lower)),
    sprintf("q_upper = %s", fmt(ends$upper))
  ), file.path(args[3], sprintf("ref_%s.toml", id)))
}

d <- utils::read.csv(args[1], stringsAsFactors = FALSE)
for (cl in list(list(id = "binom_logit_sd_gx", fam = stats::binomial(link = "logit")),
                list(id = "binom_probit_sd_gx", fam = stats::binomial(link = "probit")))) {
  write_cell(cl$id, drmTMB(bf(cbind(s, fl) ~ x1 + (1 | g), sd(g) ~ gx), family = cl$fam, data = d))
}
z <- utils::read.csv(args[2], stringsAsFactors = FALSE)
write_cell("zob_sigma_x1", drmTMB(bf(y_zoib ~ x1 + (1 | g), sigma ~ x1), family = zero_one_beta(), data = z))
