# gen_zi_re_parity.R -- native drmTMB 0.8.1 reference for random-effect cells that DRModels.jl
# used to refuse (zero-inflation / hurdle / truncated NB2 with (1 | g); SkewNormal (1 | g);
# Binomial with three random intercepts; CumulativeLogit with (1 | g) + (0 + x | g)).
#
#   Rscript --vanilla test/parity/gen_zi_re_parity.R [only_route ...]
#
# One folder per route under test/fixtures/zi_re_parity/<route>/ with data.csv and ref.toml
# (drmTMB opt$par, sdreport SEs, logLik, max |gradient|). Numbers and simulated data only; no
# drmTMB source is copied. Fixed seeds, so the fixtures regenerate given the same versions.
# A route whose R fit is not converged (max|grad| >= 1e-6 or a non-finite SE) is reported and
# NOT written: it is not a valid target.

suppressPackageStartupMessages(library(drmTMB))
stopifnot(as.character(utils::packageVersion("drmTMB")) == "0.8.1")
args <- commandArgs(trailingOnly = TRUE)
out_root <- file.path("test", "fixtures", "zi_re_parity")
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
want <- function(route) length(args) == 0 || route %in% args

grp <- function(seed, G, m, sd_g) {
  set.seed(seed)
  g <- rep(seq_len(G), each = m); x <- stats::rnorm(G * m)
  list(d = data.frame(g = paste0("g", g), x = x), n = G * m,
       eta = 0.4 + 0.4 * x + stats::rnorm(G, 0, sd_g)[g])
}
rtnb <- function(mu, size) { y <- stats::rnbinom(length(mu), mu = mu, size = size)
  while (any(y == 0)) { k <- y == 0; y[k] <- stats::rnbinom(sum(k), mu = mu[k], size = size) }; y }

if (want("zip_g")) {
  s <- grp(20261110, 30, 10, 0.5)
  s$d$y <- stats::rpois(s$n, exp(s$eta)) * stats::rbinom(s$n, 1, 0.75)
  f <- drmTMB(bf(y ~ x + (1 | g), zi ~ 1), family = stats::poisson(), data = s$d)
  write_ref("zip_g", f, c("mu_(Intercept)", "mu_x", "zi_(Intercept)", "resd_g"), s$d,
            "zero-inflated Poisson, mean (1 | g), zi ~ 1")
}
if (want("zinb_g")) {
  s <- grp(20261111, 30, 10, 0.5)
  s$d$y <- stats::rnbinom(s$n, mu = exp(s$eta), size = 3) * stats::rbinom(s$n, 1, 0.75)
  f <- drmTMB(bf(y ~ x + (1 | g), sigma ~ 1, zi ~ 1), family = nbinom2(), data = s$d)
  write_ref("zinb_g", f, c("mu_(Intercept)", "mu_x", "sigma_(Intercept)", "zi_(Intercept)", "resd_g"), s$d,
            "zero-inflated NB2, mean (1 | g), sigma ~ 1, zi ~ 1")
}
if (want("tnb_g")) {
  s <- grp(20261112, 30, 10, 0.5)
  s$d$y <- rtnb(exp(s$eta), 3)
  f <- drmTMB(bf(y ~ x + (1 | g), sigma ~ 1), family = truncated_nbinom2(), data = s$d)
  write_ref("tnb_g", f, c("mu_(Intercept)", "mu_x", "sigma_(Intercept)", "resd_g"), s$d,
            "zero-truncated NB2, mean (1 | g), sigma ~ 1")
}
if (want("hnb_g")) {
  s <- grp(20261113, 30, 10, 0.5)
  s$d$y <- rtnb(exp(s$eta), 3) * stats::rbinom(s$n, 1, 0.7)
  f <- drmTMB(bf(y ~ x + (1 | g), sigma ~ 1, hu ~ 1), family = truncated_nbinom2(), data = s$d)
  write_ref("hnb_g", f, c("mu_(Intercept)", "mu_x", "sigma_(Intercept)", "hu_(Intercept)", "resd_g"), s$d,
            "hurdle NB2, mean (1 | g), sigma ~ 1, hu ~ 1")
}
if (want("skew_g")) {
  s <- grp(20261114, 40, 10, 0.5)
  al <- 3; dl <- al / sqrt(1 + al^2)
  z <- dl * abs(stats::rnorm(s$n)) + sqrt(1 - dl^2) * stats::rnorm(s$n)
  s$d$y <- s$eta + 1.0 * z
  f <- drmTMB(bf(y ~ x + (1 | g), sigma ~ 1, nu ~ 1), family = skew_normal(), data = s$d)
  write_ref("skew_g", f, c("mu_(Intercept)", "mu_x", "sigma_(Intercept)", "nu_(Intercept)", "resd_g"), s$d,
            "SkewNormal, mean (1 | g), sigma ~ 1, nu ~ 1")
}
if (want("binom3")) {
  set.seed(20261115)
  n <- 900; g <- sample(30, n, TRUE); g2 <- sample(20, n, TRUE); h <- sample(15, n, TRUE)
  x <- stats::rnorm(n)
  eta <- -0.2 + 0.5 * x + stats::rnorm(30, 0, 0.8)[g] + stats::rnorm(20, 0, 0.6)[g2] +
    stats::rnorm(15, 0, 0.5)[h]
  d <- data.frame(y = stats::rbinom(n, 1, stats::plogis(eta)), x = x,
                  g = paste0("g", g), g2 = paste0("k", g2), h = paste0("h", h))
  f <- drmTMB(bf(y ~ x + (1 | g) + (1 | g2) + (1 | h)), family = stats::binomial(), data = d)
  write_ref("binom3", f, c("mu_(Intercept)", "mu_x", "resd_g", "resd_g2", "resd_h"), d,
            "Binomial, three crossed random intercepts")
}
if (want("cumul_gx")) {
  set.seed(20261116)
  G <- 40; m <- 12; g <- rep(seq_len(G), each = m); x <- stats::rnorm(G * m)
  eta <- 0.7 * x + stats::rnorm(G, 0, 0.7)[g] + stats::rnorm(G, 0, 0.5)[g] * x
  th <- c(-1, 0.4, 1.6)
  u <- stats::rlogis(G * m)
  y <- 1 + (eta + u > th[1]) + (eta + u > th[2]) + (eta + u > th[3])
  d <- data.frame(y = as.numeric(y), x = x, g = paste0("g", g))
  f <- drmTMB(bf(y ~ x + (1 | g) + (0 + x | g)), family = cumulative_logit(), data = d)
  write_ref("cumul_gx", f, c("mu_x", "cutpoints_theta1", "cutpoints_theta2", "cutpoints_theta3", "resd_g", "resd_g:x"), d, "CumulativeLogit, (1 | g) + (0 + x | g)")
}

# ---- non-constant zi / hu / sigma / nu formulas (same data generators, richer parameter formulas)
if (want("zip_zix")) {
  s <- grp(20261110, 30, 10, 0.5)
  s$d$y <- stats::rpois(s$n, exp(s$eta)) * stats::rbinom(s$n, 1, stats::plogis(1.0 - 0.8 * s$d$x))
  f <- drmTMB(bf(y ~ x + (1 | g), zi ~ x), family = stats::poisson(), data = s$d)
  write_ref("zip_zix", f, c("mu_(Intercept)", "mu_x", "zi_(Intercept)", "zi_x", "resd_g"), s$d, "ZIP, zi ~ x")
}
if (want("zinb_sx_zx")) {
  # seed 20261111 gave a drmTMB local optimum 1.08 logLik below the better mode DRModels.jl found
  # (R's own objective at the Julia optimum is higher), so it is not a valid target; this seed
  # has one mode both engines reach.
  s <- grp(20261121, 40, 10, 0.5)
  s$d$y <- stats::rnbinom(s$n, mu = exp(s$eta), size = exp(1 + 0.4 * s$d$x)) * stats::rbinom(s$n, 1, stats::plogis(1.0 - 0.8 * s$d$x))
  f <- drmTMB(bf(y ~ x + (1 | g), sigma ~ x, zi ~ x), family = nbinom2(), data = s$d)
  write_ref("zinb_sx_zx", f, c("mu_(Intercept)", "mu_x", "sigma_(Intercept)", "sigma_x", "zi_(Intercept)", "zi_x", "resd_g"), s$d, "ZINB, sigma ~ x, zi ~ x")
}
if (want("tnb_sx")) {
  s <- grp(20261112, 30, 10, 0.5)
  s$d$y <- rtnb(exp(s$eta), exp(1 + 0.4 * s$d$x))
  f <- drmTMB(bf(y ~ x + (1 | g), sigma ~ x), family = truncated_nbinom2(), data = s$d)
  write_ref("tnb_sx", f, c("mu_(Intercept)", "mu_x", "sigma_(Intercept)", "sigma_x", "resd_g"), s$d, "truncated NB2, sigma ~ x")
}
if (want("hnb_sx_hx")) {
  s <- grp(20261113, 30, 10, 0.5)
  s$d$y <- rtnb(exp(s$eta), exp(1 + 0.4 * s$d$x)) * stats::rbinom(s$n, 1, stats::plogis(0.8 + 0.6 * s$d$x))
  f <- drmTMB(bf(y ~ x + (1 | g), sigma ~ x, hu ~ x), family = truncated_nbinom2(), data = s$d)
  write_ref("hnb_sx_hx", f, c("mu_(Intercept)", "mu_x", "sigma_(Intercept)", "sigma_x", "hu_(Intercept)", "hu_x", "resd_g"), s$d, "hurdle NB2, sigma ~ x, hu ~ x")
}
if (want("skew_sx")) {
  s <- grp(20261114, 40, 10, 0.5)
  al <- 3; dl <- al / sqrt(1 + al^2)
  z <- dl * abs(stats::rnorm(s$n)) + sqrt(1 - dl^2) * stats::rnorm(s$n)
  s$d$y <- s$eta + exp(0.0 + 0.3 * s$d$x) * z
  f <- drmTMB(bf(y ~ x + (1 | g), sigma ~ x, nu ~ 1), family = skew_normal(), data = s$d)
  write_ref("skew_sx", f, c("mu_(Intercept)", "mu_x", "sigma_(Intercept)", "sigma_x", "nu_(Intercept)", "resd_g"), s$d, "SkewNormal, sigma ~ x, nu ~ 1")
}
if (want("skew_nx")) {
  s <- grp(20261115, 40, 10, 0.5)
  al <- 3; dl <- al / sqrt(1 + al^2)
  z <- dl * abs(stats::rnorm(s$n)) + sqrt(1 - dl^2) * stats::rnorm(s$n)
  s$d$y <- s$eta + 1.0 * z
  f <- drmTMB(bf(y ~ x + (1 | g), sigma ~ 1, nu ~ x), family = skew_normal(), data = s$d)
  write_ref("skew_nx", f, c("mu_(Intercept)", "mu_x", "sigma_(Intercept)", "nu_(Intercept)", "nu_x", "resd_g"), s$d, "SkewNormal, nu ~ x")
}

# ---- two zero-inflated NB2 data sets on which the fixed-effects start sat on the zero-inflation
# boundary (logit about -30), so the Laplace fit stalled 8 to 10 log-likelihood units below drmTMB
# before the interior restarts of `src/zi_re_laplace.jl` (regression targets; set 3: 25 unbalanced
# groups, group SD 0.9; set 6: 40 groups of 6, group SD 1.2, slope 0.8).
zinb_r <- function(route, seed, G, m, sd_g, bx) {
  set.seed(seed)
  sz <- if (m == 0) sample(2:20, G, TRUE) else rep(m, G)
  g <- rep(seq_len(G), times = sz); n <- length(g); x <- stats::rnorm(n)
  eta <- 0.4 + bx * x + stats::rnorm(G, 0, sd_g)[g]
  d <- data.frame(g = paste0("g", g), x = x)
  d$y <- stats::rnbinom(n, mu = exp(eta), size = 3) * stats::rbinom(n, 1, 0.75)
  f <- drmTMB(bf(y ~ x + (1 | g), sigma ~ 1, zi ~ 1), family = nbinom2(), data = d)
  write_ref(route, f, c("mu_(Intercept)", "mu_x", "sigma_(Intercept)", "zi_(Intercept)", "resd_g"), d,
            "zero-inflated NB2, mean (1 | g), sigma ~ 1, zi ~ 1 (start on the zero-inflation boundary)")
}
if (want("zinb_r3")) zinb_r("zinb_r3", 7303, 25, 0, 0.9, 0.4)
if (want("zinb_r6")) zinb_r("zinb_r6", 7603, 40, 6, 1.2, 0.8)
