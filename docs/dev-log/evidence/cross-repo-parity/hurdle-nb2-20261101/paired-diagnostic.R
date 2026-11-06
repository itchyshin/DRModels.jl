pkg <- "/home/snakagaw/hsq_work/drmtmb-biv-lognormal-main-b535bfbc-20261028/source"
jl_path <- "/home/snakagaw/hsq_work/DRModels-ci-64ccb2d8"
set.seed(20260623)
n <- 1800L
dat <- data.frame(
  x = stats::rnorm(n),
  z = stats::rnorm(n),
  w = stats::rnorm(n),
  habitat = factor(rep(c("open", "closed"), length.out = n))
)
X_mu <- stats::model.matrix(~ x + habitat, dat)
X_sigma <- stats::model.matrix(~ z, dat)
X_hu <- stats::model.matrix(~ w + habitat, dat)
mu <- exp(as.vector(X_mu %*% c(0.40, -0.22, 0.20)))
sigma <- exp(as.vector(X_sigma %*% c(-0.75, 0.18)))
hu <- stats::plogis(as.vector(X_hu %*% c(-0.85, 0.45, -0.35)))
p0 <- stats::dnbinom(0, size = 1 / sigma^2, mu = mu)
positive_u <- p0 + pmax(stats::runif(n), .Machine$double.eps) * (1 - p0)
dat$count <- ifelse(
  stats::runif(n) < hu,
  0,
  stats::qnbinom(positive_u, size = 1 / sigma^2, mu = mu)
)

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
