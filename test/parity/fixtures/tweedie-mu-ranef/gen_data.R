# gen_data.R — R oracle for the tweedie() mu random-intercept parity fixture
# (#563 S8). Run with drmTMB 0.7.0:
#
#   Rscript test/parity/fixtures/tweedie-mu-ranef/gen_data.R
#
# Mirrors drmTMB's own `new_tweedie_random_intercept_data()` helper
# (tests/testthat/test-tweedie-location-scale.R) with a smaller n_id/n_each so
# the fixture is easy to embed, and a fixed seed so the CSV here is
# reproducible byte-for-byte.

suppressMessages(library(drmTMB))

new_tweedie_random_intercept_data <- function(
  n_id = 40, n_each = 8, seed = 20260901,
  beta_mu = c("(Intercept)" = 0.5, x = 0.3),
  nu = 1.5, phi = 1.0, sd_id = 0.5
) {
  set.seed(seed)
  id <- factor(rep(seq_len(n_id), each = n_each))
  n <- length(id)
  dat <- data.frame(id = id, x = stats::runif(n, -1, 1))
  u_id <- stats::rnorm(n_id, sd = sd_id)
  u_id <- u_id - mean(u_id)
  names(u_id) <- levels(id)
  eta_mu <- beta_mu[[1L]] + beta_mu[[2L]] * dat$x + u_id[id]
  mu <- exp(eta_mu)
  dat$y <- drmTMB:::rtweedie_compound(n, mu = mu, phi = phi, power = nu)
  list(data = dat, beta_mu = beta_mu, nu = nu, phi = phi, sd_id = sd_id, u_id = u_id)
}

sim <- new_tweedie_random_intercept_data()
cat("n zeros:", sum(sim$data$y == 0), "of", nrow(sim$data), "\n")

fit <- drmTMB(bf(y ~ x + (1 | id), nu ~ 1), family = tweedie(), data = sim$data)

cat("convergence:", fit$opt$convergence, "pdHess:", fit$sdr$pdHess, "\n")
cat("coef mu:", coef(fit, "mu"), "\n")
cat("coef sigma (log sigma):", coef(fit, "sigma"), "\n")
cat("coef nu (logit):", coef(fit, "nu"), "\n")
cat("sd_id estimate:", unname(fit$sdpars$mu[["(1 | id)"]]), "\n")
cat("loglik:", as.numeric(logLik(fit)), "\n")

# The correlated random slope `(1 + x | id)` stays rejected by drmTMB
# 0.7.0 (`validate_tweedie_mu_random_terms()`): only `(1 | id)` and the
# INDEPENDENT slope `(0 + x | id)` are implemented for tweedie() mu. DRModels.jl
# matches that same-target scope this slice.
res <- tryCatch(
  drmTMB(bf(y ~ x + (1 + x | id), nu ~ 1), family = tweedie(), data = sim$data),
  error = function(e) conditionMessage(e)
)
cat("slope-fit error:", res, "\n")

write.csv(sim$data, "test/parity/fixtures/tweedie-mu-ranef/data.csv", row.names = FALSE)
