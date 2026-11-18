# gen_binomial_sd_fitted_parity.R -- native drmTMB 0.8.1 reference `fitted()` values for two Binomial
# random-intercept models with a random-effect SD regression, `sd(g) ~ gx`, one with a logit and one with a
# probit link. DRModels.jl fits them with its default integrator (quadrature); drmTMB uses the Laplace
# approximation, so the two differ by about 3e-4 in `fitted`. The Julia test therefore uses drmTMB's
# estimates (`theta`) and `fitted` values as a reference for the conditional-mode computation, not for
# the package's own estimates.
#
#   Rscript test/parity/gen_binomial_sd_fitted_parity.R <output dir>
#
# Output: data.csv (columns g, gx, x1, s, fl; simulated here with a fixed seed) and one ref_<cell>.toml per
# model with drmTMB's opt$par, logLik, max |gradient| and fitted values. Numbers and simulated data only.
# A fit with max|gradient| >= 1e-6 is reported and NOT written.

if (nzchar(Sys.getenv("DRMTMB_LIB"))) .libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
stopifnot(as.character(utils::packageVersion("drmTMB")) == "0.8.1")
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1)
dir.create(args[1], recursive = TRUE, showWarnings = FALSE)

set.seed(20261114)
G <- 30; m <- 8
g <- rep(sprintf("g%02d", 1:G), each = m)
gx_group <- rep(c(-1, 0, 1), length.out = G)
gx <- rep(gx_group, each = m)
x1 <- rnorm(G * m)
sd_group <- exp(-0.4 + 0.5 * gx_group)
u <- rnorm(G, 0, sd_group)
size <- 6
p <- stats::pnorm(-0.2 + 0.6 * x1 + rep(u, each = m))
s <- stats::rbinom(G * m, size, p)
d <- data.frame(g = g, gx = gx, x1 = x1, s = s, fl = size - s)
utils::write.csv(d, file.path(args[1], "data.csv"), row.names = FALSE)

fmt <- function(x) paste0("[", paste(formatC(x, digits = 17, format = "g"), collapse = ", "), "]")
qs  <- function(x) paste0("[", paste(sprintf("\"%s\"", x), collapse = ", "), "]")
cells <- list(
  list(id = "binom_logit_sd_gx",  fam = stats::binomial(link = "logit")),
  list(id = "binom_probit_sd_gx", fam = stats::binomial(link = "probit"))
)
for (cl in cells) {
  f <- drmTMB(bf(cbind(s, fl) ~ x1 + (1 | g), sd(g) ~ gx), family = cl$fam, data = d)
  par <- f$opt$par
  gr <- as.numeric(f$obj$gr(par))
  fv <- as.numeric(fitted(f))
  cat(sprintf("%s n=%d max|grad|=%.2e ll=%.10f\n  names: %s\n  par: %s\n",
              cl$id, nrow(d), max(abs(gr)), as.numeric(logLik(f)),
              paste(names(par), collapse = " "), paste(signif(par, 6), collapse = " ")))
  if (max(abs(gr)) >= 1e-6) { cat("  -> NOT WRITTEN\n"); next }
  writeLines(c(
    sprintf("cell = \"%s\"", cl$id),
    sprintf("drmTMB_version = \"%s\"", as.character(utils::packageVersion("drmTMB"))),
    sprintf("R_version = \"%s\"", paste(R.version$major, R.version$minor, sep = ".")),
    sprintf("tmb_names = %s", qs(names(par))),
    sprintf("loglik = %s", formatC(as.numeric(logLik(f)), digits = 17, format = "g")),
    sprintf("max_grad = %s", formatC(max(abs(gr)), digits = 6, format = "e")),
    sprintf("theta = %s", fmt(as.numeric(par))),
    sprintf("fitted = %s", fmt(fv))
  ), file.path(args[1], sprintf("ref_%s.toml", cl$id)))
}
