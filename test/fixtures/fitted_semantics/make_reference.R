# Reference values for test/test_fitted_semantics_r_parity.jl.
# Run once, single process, on drmTMB 0.8.1:
#   R_LIBS=~/drm_campaign_20261108/Rlib OPENBLAS_NUM_THREADS=1 Rscript make_reference.R
# Writes data.csv (committed simulated data; set.seed only here, never in the Julia
# test) and reference.csv (long format: model, kind, row, value).
suppressPackageStartupMessages(library(drmTMB))
stopifnot(as.character(packageVersion("drmTMB")) == "0.8.1")
set.seed(20261109)
n <- 150
x <- round(runif(n, -1, 1), 4)
z <- round(runif(n, -1, 1), 4)
mu_ln <- 0.3 + 0.5 * x
sg_ln <- exp(-0.4 + 0.5 * z)
ln <- round(exp(rnorm(n, mu_ln, sg_ln)), 5)
mu_c <- exp(0.7 + 0.5 * x)
zi <- plogis(-0.8 + 0.7 * z)
zip <- ifelse(runif(n) < zi, 0L, rpois(n, mu_c))
zinb <- ifelse(runif(n) < zi, 0L, rnbinom(n, mu = mu_c, size = 2))
rtnb <- function(m, size) {                  # zero-truncated NB2 by rejection
  out <- integer(length(m))
  for (i in seq_along(m)) repeat { v <- rnbinom(1, mu = m[i], size = size); if (v > 0) { out[i] <- v; break } }
  out
}
tnb <- rtnb(mu_c, 2)
hnb <- ifelse(runif(n) < plogis(-0.5 + 0.6 * z), 0L, tnb)
dat <- data.frame(x = x, z = z, ln = ln, zip = zip, zinb = zinb, tnb = tnb, hnb = hnb)
write.csv(dat, "data.csv", row.names = FALSE)
dat <- read.csv("data.csv")                   # fit exactly what is committed
nd <- data.frame(x = c(-0.9, -0.2, 0.3, 0.8), z = c(0.5, -0.7, 0.1, 0.9))
fits <- list(
  lognormal = drmTMB(bf(ln ~ x, sigma ~ z), family = lognormal(), data = dat),
  zi_poisson = drmTMB(bf(zip ~ x, zi ~ z), family = stats::poisson(link = "log"), data = dat),
  zi_nbinom2 = drmTMB(bf(zinb ~ x, sigma ~ 1, zi ~ z), family = nbinom2(), data = dat),
  truncated_nbinom2 = drmTMB(bf(tnb ~ x, sigma ~ 1), family = truncated_nbinom2(), data = dat),
  hurdle_nbinom2 = drmTMB(bf(hnb ~ x, sigma ~ 1, hu ~ z), family = truncated_nbinom2(), data = dat)
)
rows <- list()
add <- function(m, kind, v) rows[[length(rows) + 1L]] <<-
  data.frame(model = m, kind = kind, row = seq_along(v), value = as.numeric(v))
for (m in names(fits)) {
  f <- fits[[m]]
  add(m, "logLik", as.numeric(logLik(f)))
  add(m, "fitted", fitted(f))
  add(m, "residuals_response", residuals(f, type = "response"))
  add(m, "predict_response_new", predict(f, newdata = nd, type = "response"))
  add(m, "predict_mean_new", predict(f, newdata = nd, type = "mean"))
  add(m, "predict_link_new", predict(f, newdata = nd, type = "link"))
  sm <- simulate(f, nsim = 400, seed = 1)
  add(m, "simulate_mean_per_row", rowMeans(as.matrix(sm)))
}
write.csv(do.call(rbind, rows), "reference.csv", row.names = FALSE)
