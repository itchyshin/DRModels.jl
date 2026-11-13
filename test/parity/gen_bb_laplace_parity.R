# gen_bb_laplace_parity.R -- native drmTMB 0.8.1 reference for BetaBinomial with one ordinary
# random intercept `(1 | g)` (P22c). drmTMB integrates it by the Laplace approximation; DRModels.jl
# reproduces that with `marginal = :Laplace` (its default `:LA` integrates by quadrature).
#
#   Rscript --vanilla test/parity/gen_bb_laplace_parity.R
#
# Seven data sets (seeds, sizes, group counts, random-effect SD and precision vary), one folder per
# data set under test/fixtures/bb_laplace_parity/ with data.csv and ref.toml (drmTMB opt$par, sdreport
# SEs, logLik, max |gradient|). Numbers and simulated data only; no drmTMB source is copied. A data set
# whose R fit is not converged (max|grad| >= 1e-6 or a non-finite SE) is reported and NOT written.

suppressPackageStartupMessages(library(drmTMB))
stopifnot(as.character(utils::packageVersion("drmTMB")) == "0.8.1")
args <- commandArgs(trailingOnly = TRUE)
out_root <- file.path("test", "fixtures", "bb_laplace_parity")
fmt <- function(x) paste0("[", paste(formatC(x, digits = 17, format = "g"), collapse = ", "), "]")
qs  <- function(x) paste0("[", paste(sprintf("\"%s\"", x), collapse = ", "), "]")

write_ref <- function(route, fit, names, data, note) {
  par <- fit$opt$par
  gr <- as.numeric(fit$obj$gr(par))
  se <- tryCatch(sqrt(diag(fit$sdr$cov.fixed)), error = function(e) rep(NA_real_, length(par)))
  cat(sprintf("%-14s n=%d max|grad|=%.2e allSEfinite=%s\n  tmb: %s\n  par: %s\n  se : %s\n  ll : %.10f\n",
              route, nrow(data), max(abs(gr)), all(is.finite(se)), paste(names(par), collapse = " "),
              paste(signif(par, 5), collapse = " "), paste(signif(se, 4), collapse = " "),
              as.numeric(logLik(fit))))
  if (length(names) != length(par) || max(abs(gr)) >= 1e-6 || !all(is.finite(se))) {
    cat("  -> NOT WRITTEN (not a valid target)\n"); return(invisible(FALSE))
  }
  dir <- file.path(out_root, route)
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(data, file.path(dir, "data.csv"), row.names = FALSE)
  writeLines(c(
    sprintf("route = \"%s\"", route),
    sprintf("drmTMB_version = \"%s\"", as.character(utils::packageVersion("drmTMB"))),
    sprintf("R_version = \"%s\"", paste(R.version$major, R.version$minor, sep = ".")),
    sprintf("note = \"%s\"", note),
    sprintf("tmb_names = %s", qs(names(par))),
    sprintf("names = %s", qs(names)),
    sprintf("loglik = %s", formatC(as.numeric(logLik(fit)), digits = 17, format = "g")),
    sprintf("max_grad = %s", formatC(max(abs(gr)), digits = 6, format = "e")),
    sprintf("theta = %s", fmt(as.numeric(par))),
    sprintf("se = %s", fmt(se))
  ), file.path(dir, "ref.toml"))
  invisible(TRUE)
}

cfg <- list(
  list(G = 30, m = 12, sd = 0.6, phi = 12, nmin = 15, nmax = 25),
  list(G = 20, m = 20, sd = 0.9, phi = 8,  nmin = 10, nmax = 30),
  list(G = 50, m = 8,  sd = 0.4, phi = 20, nmin = 15, nmax = 25),
  list(G = 15, m = 25, sd = 0.7, phi = 15, nmin = 20, nmax = 40),
  list(G = 40, m = 10, sd = 1.0, phi = 10, nmin = 12, nmax = 20),
  list(G = 25, m = 15, sd = 0.5, phi = 6,  nmin = 10, nmax = 25),
  list(G = 35, m = 12, sd = 0.8, phi = 25, nmin = 20, nmax = 35)
)
for (k in seq_along(cfg)) {
  s <- cfg[[k]]
  set.seed(20261170 + k)
  n <- s$G * s$m
  g <- rep(seq_len(s$G), each = s$m); x <- stats::rnorm(n)
  eta <- -0.2 + 0.5 * x + stats::rnorm(s$G, 0, s$sd)[g]
  mu <- stats::plogis(eta)
  ntr <- sample(s$nmin:s$nmax, n, TRUE)
  pp <- stats::rbeta(n, mu * s$phi, (1 - mu) * s$phi)
  succ <- stats::rbinom(n, ntr, pp)
  d <- data.frame(g = paste0("g", g), x = x, successes = succ, failures = ntr - succ)
  f <- drmTMB(bf(cbind(successes, failures) ~ x + (1 | g), sigma ~ 1), family = beta_binomial(), data = d)
  write_ref(paste0("bb_", k), f, c("mu_(Intercept)", "mu_x", "sigma_(Intercept)", "resd_g"), d,
            "BetaBinomial, one (1 | g), sigma ~ 1")
}

# Three near-binomial data sets (precision phi 30 to 150, small group SD) where the MLE precision is
# large. An unbounded precision let the optimiser run off to phi of 1e30 and return a nonsense
# log-likelihood with `converged = false`; these are the regression targets for the bound on log sigma.
hard <- list(
  list(route = "bb_8",  seed = 8802, G = 45, m = 6,  sd = 0.3, phi = 30),
  list(route = "bb_9",  seed = 5013, G = 45, m = 6,  sd = 0.3, phi = 150),
  list(route = "bb_10", seed = 5014, G = 20, m = 12, sd = 0.3, phi = 150)
)
for (s in hard) {
  set.seed(s$seed)
  g <- rep(seq_len(s$G), each = s$m); n <- length(g); x <- stats::rnorm(n)
  eta <- -0.3 + 0.5 * x + stats::rnorm(s$G, 0, s$sd)[g]; mu <- stats::plogis(eta)
  ntr <- sample(8:30, n, TRUE)
  pp <- stats::rbeta(n, mu * s$phi, (1 - mu) * s$phi); succ <- stats::rbinom(n, ntr, pp)
  d <- data.frame(g = paste0("g", g), x = x, successes = succ, failures = ntr - succ)
  f <- drmTMB(bf(cbind(successes, failures) ~ x + (1 | g), sigma ~ 1), family = beta_binomial(), data = d)
  write_ref(s$route, f, c("mu_(Intercept)", "mu_x", "sigma_(Intercept)", "resd_g"), d,
            sprintf("BetaBinomial, one (1 | g), near-binomial (phi %g), sigma ~ 1", s$phi))
}
