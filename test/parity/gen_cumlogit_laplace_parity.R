# gen_cumlogit_laplace_parity.R -- native drmTMB 0.8.1 reference for CumulativeLogit with one
# ordinary random intercept `(1 | g)`. drmTMB integrates it by the Laplace approximation;
# DRModels.jl reproduces that with `marginal = :Laplace` (its default `:LA` uses quadrature).
#
#   Rscript test/parity/gen_cumlogit_laplace_parity.R
#
# Four simulated data sets (3, 4 and 5 categories; 8, 15, 20 and 40 groups), one folder per data
# set under test/fixtures/cumlogit_laplace_parity/ with data.csv and ref.toml (drmTMB opt$par,
# sdreport SEs, logLik, max |gradient|, and the population-level category probabilities
# P(y = k) = logistic(cut_k - x beta) - logistic(cut_{k-1} - x beta) at the fitted values).
# Numbers and simulated data only; no drmTMB source is copied. A data set whose R fit is not
# converged (max|grad| >= 1e-6 or a non-finite SE) is reported and NOT written.

if (nzchar(Sys.getenv("DRMTMB_LIB"))) .libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
stopifnot(as.character(utils::packageVersion("drmTMB")) == "0.8.1")
out_root <- file.path("test", "fixtures", "cumlogit_laplace_parity")
fmt <- function(x) paste0("[", paste(formatC(x, digits = 17, format = "g"), collapse = ", "), "]")
qs  <- function(x) paste0("[", paste(sprintf("\"%s\"", x), collapse = ", "), "]")

cfg <- list(
  list(route = "cl_1", seed = 20261301, G = 8,  m = 20, K = 3, sd = 0.8, beta = 0.7),
  list(route = "cl_2", seed = 20261302, G = 40, m = 8,  K = 4, sd = 0.6, beta = -0.5),
  list(route = "cl_3", seed = 20261303, G = 20, m = 12, K = 5, sd = 1.0, beta = 0.9),
  list(route = "cl_4", seed = 20261304, G = 15, m = 15, K = 4, sd = 0.4, beta = 0.6)
)
for (s in cfg) {
  set.seed(s$seed)
  n <- s$G * s$m
  g <- rep(seq_len(s$G), each = s$m); x <- stats::rnorm(n)
  cut <- seq(-1.2, 1.2, length.out = s$K - 1)
  lat <- s$beta * x + stats::rnorm(s$G, 0, s$sd)[g] + stats::rlogis(n)
  y <- findInterval(lat, cut) + 1L
  stopifnot(all(seq_len(s$K) %in% y))
  d <- data.frame(g = sprintf("g%02d", g), x = x, y = y)
  d$y_ord <- ordered(d$y, levels = seq_len(s$K))
  f <- drmTMB(bf(y_ord ~ x + (1 | g)), family = cumulative_logit(), data = d)
  par <- f$opt$par
  gr <- as.numeric(f$obj$gr(par))
  se <- tryCatch(sqrt(diag(f$sdr$cov.fixed)), error = function(e) rep(NA_real_, length(par)))
  cuts <- unname(f$ordinal$cutpoints)
  cat(sprintf("%s K=%d G=%d n=%d max|grad|=%.2e allSEfinite=%s ll=%.10f\n  tmb: %s\n  par: %s\n  se : %s\n  cuts: %s\n",
              s$route, s$K, s$G, n, max(abs(gr)), all(is.finite(se)), as.numeric(logLik(f)),
              paste(names(par), collapse = " "), paste(signif(par, 5), collapse = " "),
              paste(signif(se, 4), collapse = " "), paste(signif(cuts, 5), collapse = " ")))
  if (max(abs(gr)) >= 1e-6 || !all(is.finite(se))) { cat("  -> NOT WRITTEN\n"); next }
  nc <- s$K - 1
  names_j <- c("mu_x", paste0("cutpoints_theta", seq_len(nc)), "resd_g")
  stopifnot(length(par) == length(names_j))
  eta <- unname(par[1]) * d$x
  cdf <- cbind(0, vapply(cuts, function(ct) stats::plogis(ct - eta), numeric(n)), 1)
  prob <- cdf[, -1, drop = FALSE] - cdf[, -ncol(cdf), drop = FALSE]
  dir <- file.path(out_root, s$route)
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(d[, c("g", "x", "y")], file.path(dir, "data.csv"), row.names = FALSE)
  writeLines(c(
    sprintf("route = \"%s\"", s$route),
    sprintf("drmTMB_version = \"%s\"", as.character(utils::packageVersion("drmTMB"))),
    sprintf("R_version = \"%s\"", paste(R.version$major, R.version$minor, sep = ".")),
    sprintf("note = \"CumulativeLogit, K = %d, one (1 | g) with %d groups\"", s$K, s$G),
    sprintf("K = %d", s$K),
    sprintf("tmb_names = %s", qs(names(par))),
    sprintf("names = %s", qs(names_j)),
    sprintf("loglik = %s", formatC(as.numeric(logLik(f)), digits = 17, format = "g")),
    sprintf("max_grad = %s", formatC(max(abs(gr)), digits = 6, format = "e")),
    sprintf("theta = %s", fmt(as.numeric(par))),
    sprintf("se = %s", fmt(se)),
    sprintf("cutpoints = %s", fmt(cuts)),
    sprintf("prob = %s", fmt(as.numeric(t(prob))))   # row-major: n rows of K probabilities
  ), file.path(dir, "ref.toml"))
}
