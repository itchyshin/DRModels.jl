# gen_laplace_sigma_formula_parity.R -- native drmTMB 0.8.1 reference for four models that
# DRModels.jl fits with the opt-in `marginal = :Laplace` (drmTMB integrates the random
# intercept by the Laplace approximation; the default `:LA` of DRModels.jl is quadrature):
#   Gamma, Beta, NegBinomial2 and ZeroOneBeta with (1 | g) on mu and `sigma ~ x1`.
#
#   Rscript test/parity/gen_laplace_sigma_formula_parity.R <input csv> <output dir>
#
# The input is the simulated data set the comparison harness uses (columns g, x1, y_gam, y_beta, y_nb, y_zoib). Output: data.csv (those columns only) and one ref_<cell>.toml per
# model with drmTMB's opt$par, sdreport SEs, logLik, max |gradient| and fitted values. Numbers and
# simulated data only; no drmTMB source is copied. A fit with max|grad| >= 1e-6 or a non-finite SE
# is reported and NOT written.

if (nzchar(Sys.getenv("DRMTMB_LIB"))) .libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
stopifnot(as.character(utils::packageVersion("drmTMB")) == "0.8.1")
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
d <- utils::read.csv(args[1], stringsAsFactors = FALSE)
keep <- c("g", "x1", "y_gam", "y_beta", "y_nb", "y_zoib")
d <- d[, keep]
dir.create(args[2], recursive = TRUE, showWarnings = FALSE)
utils::write.csv(d, file.path(args[2], "data.csv"), row.names = FALSE)
fmt <- function(x) paste0("[", paste(formatC(x, digits = 17, format = "g"), collapse = ", "), "]")
qs  <- function(x) paste0("[", paste(sprintf("\"%s\"", x), collapse = ", "), "]")

cells <- list(
  list(id = "gamma_sigma_x1", fam = stats::Gamma(link = "log"), f = bf(y_gam ~ x1 + (1 | g), sigma ~ x1)),
  list(id = "beta_sigma_x1",  fam = beta_family(),              f = bf(y_beta ~ x1 + (1 | g), sigma ~ x1)),
  list(id = "nb2_sigma_x1",   fam = nbinom2(),                  f = bf(y_nb ~ x1 + (1 | g), sigma ~ x1)),
  list(id = "zob_sigma_x1",   fam = zero_one_beta(),            f = bf(y_zoib ~ x1 + (1 | g), sigma ~ x1))
)
for (cl in cells) {
  f <- drmTMB(cl$f, family = cl$fam, data = d)
  par <- f$opt$par
  gr <- as.numeric(f$obj$gr(par))
  se <- tryCatch(sqrt(diag(f$sdr$cov.fixed)), error = function(e) rep(NA_real_, length(par)))
  fv <- as.numeric(fitted(f))
  cat(sprintf("%s n=%d max|grad|=%.2e allSEfinite=%s ll=%.10f\n  names: %s\n  par: %s\n  se : %s\n",
              cl$id, nrow(d), max(abs(gr)), all(is.finite(se)), as.numeric(logLik(f)),
              paste(names(par), collapse = " "), paste(signif(par, 6), collapse = " "),
              paste(signif(se, 4), collapse = " ")))
  if (max(abs(gr)) >= 1e-6 || !all(is.finite(se))) { cat("  -> NOT WRITTEN\n"); next }
  writeLines(c(
    sprintf("cell = \"%s\"", cl$id),
    sprintf("drmTMB_version = \"%s\"", as.character(utils::packageVersion("drmTMB"))),
    sprintf("R_version = \"%s\"", paste(R.version$major, R.version$minor, sep = ".")),
    sprintf("tmb_names = %s", qs(names(par))),
    sprintf("loglik = %s", formatC(as.numeric(logLik(f)), digits = 17, format = "g")),
    sprintf("max_grad = %s", formatC(max(abs(gr)), digits = 6, format = "e")),
    sprintf("theta = %s", fmt(as.numeric(par))),
    sprintf("se = %s", fmt(se)),
    sprintf("fitted = %s", fmt(fv))
  ), file.path(args[2], sprintf("ref_%s.toml", cl$id)))
}
