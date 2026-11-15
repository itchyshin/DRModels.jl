pkg <- "/home/snakagaw/hsq_work/drmtmb-biv-lognormal-main-b535bfbc-20261028/source"
jl_path <- "/home/snakagaw/hsq_work/DRModels-ci-64ccb2d8"

# Standalone hurdle-NB2 data prototype; no package source is imported.
# The caller's RNGkind is retained; the seed is set explicitly.
# Positive and hurdle uniforms are drawn as separate vectors,
# in that order. The positive uniform is clamped below at machine epsilon.
# NB2 has mean mu, size 1/sigma^2 and variance mu + sigma^2 * mu^2.
make_hurdle_data <- function(n = 1800L, seed = 20260623L) {
  set.seed(seed)
  x <- rnorm(n)
  z <- rnorm(n)
  w <- rnorm(n)
  habitat <- factor(rep(c("open", "closed"), length.out = n))
  X <- model.matrix(~ x + habitat)
  S <- model.matrix(~ z)
  H <- model.matrix(~ w + habitat)
  mu <- exp(drop(X %*% c(0.40, -0.22, 0.20)))
  sigma <- exp(drop(S %*% c(-0.75, 0.18)))
  hurdle <- plogis(drop(H %*% c(-0.85, 0.45, -0.35)))
  size <- 1 / sigma^2
  p0 <- pnbinom(0, mu = mu, size = size)
  u_positive <- pmax(runif(n), .Machine$double.eps)
  u_hurdle <- runif(n)
  positive <- qnbinom(p0 + (1 - p0) * u_positive, mu = mu, size = size)
  y <- ifelse(u_hurdle < hurdle, 0, positive)
  data.frame(y, x, z, w, habitat, mu, sigma, hurdle, size, p0, u_positive, u_hurdle, positive)
}

fixture <- make_hurdle_data()
dat <- fixture[c("x", "z", "w", "habitat")]
dat$count <- fixture$y
row.names(dat) <- NULL
n <- nrow(dat)

out <- callr::r(function(pkg, jl_path, dat) {
  julia_home <- Sys.getenv("DRM_JL_JULIA_HOME", Sys.getenv("JULIA_HOME", ""))
  if (nzchar(julia_home)) Sys.setenv(JULIA_HOME = julia_home)
  options(drmTMB.DRM.jl.path = jl_path)
  Sys.setenv(DRM_JL_PATH = jl_path)
  suppressMessages(pkgload::load_all(pkg, quiet = TRUE))
  fml <- drmTMB::bf(count ~ x + habitat, sigma ~ z, hu ~ w + habitat)
  fam <- drmTMB::truncated_nbinom2()
  ft <- drmTMB::drmTMB(fml, family = fam, data = dat, engine = "tmb")
  fj <- drmTMB::drmTMB(fml, family = fam, data = dat, engine = "julia")
  dpar <- function(f, par) as.numeric(stats::predict(f, dpar = par))
  fitted_r <- as.numeric(stats::fitted(ft))
  fitted_j <- as.numeric(stats::fitted(fj))
  residuals_r <- as.numeric(stats::residuals(ft))
  residuals_j <- as.numeric(stats::residuals(fj))
  mu_j <- dpar(fj, "mu")
  sigma_j <- dpar(fj, "sigma")
  hu_j <- dpar(fj, "hu")
  fitted_expected_r <- drmTMB:::hurdle_nbinom2_mean(mu_j, sigma_j, hu_j)
  list(
    convergence_r = drmTMB::is_converged(ft),
    convergence_j = drmTMB::is_converged(fj),
    n = length(dat$count), zeros = sum(dat$count == 0), positive = sum(dat$count > 0),
    max_abs_dpar_rj = max(vapply(c("mu", "sigma", "hu"), function(p) {
      max(abs(dpar(ft, p) - dpar(fj, p)))
    }, numeric(1))),
    max_abs_r_mean_vs_reconstructed = max(abs(fitted_r - fitted_expected_r)),
    max_abs_j_mean_vs_mu = max(abs(fitted_j - mu_j)),
    max_abs_fitted_rj = max(abs(fitted_r - fitted_j)),
    max_abs_residual_rj = max(abs(residuals_r - residuals_j)),
    max_abs_residual_r_yminusfit = max(abs(residuals_r - (dat$count - fitted_r))),
    max_abs_residual_j_yminusfit = max(abs(residuals_j - (dat$count - fitted_j))),
    logLik_r = as.numeric(stats::logLik(ft)),
    logLik_j = as.numeric(stats::logLik(fj))
  )
}, args = list(pkg = pkg, jl_path = jl_path, dat = dat), show = TRUE)
cat("R_version=", as.character(getRversion()), "\n", sep = "")
cat("fixture_seed=20260623\nfixture_n=", n, "\nzeros=", sum(dat$count == 0), "\npositive=", sum(dat$count > 0), "\n", sep = "")
for (nm in names(out)) cat(nm, "=", paste(format(out[[nm]], digits = 16), collapse = ","), "\n", sep = "")
