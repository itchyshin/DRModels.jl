# Generates the pinned residual values in test/fixtures/pearson/pearson_<family>.csv.
# Run on a machine with drmTMB 0.8.1:
#   Rscript make_pearson_fixture.R <output directory>
# Each CSV has the data (y, x) and R's residuals(fit, type = "pearson") for one fixed-effect cell.
suppressPackageStartupMessages(library(drmTMB))
out <- commandArgs(trailingOnly = TRUE)[1]
n <- 200
cells <- list(
  gaussian = list(
    sim = function(x) rnorm(n, 0.5 + 0.8 * x, exp(-0.3 + 0.4 * x)),
    fit = function(d) drmTMB(bf(mu = y ~ x, sigma = ~ x), family = gaussian(), data = d)),
  poisson = list(
    sim = function(x) rpois(n, exp(0.4 + 0.5 * x)),
    fit = function(d) drmTMB(bf(mu = y ~ x), family = poisson(), data = d)),
  nbinom2 = list(
    sim = function(x) rnbinom(n, mu = exp(0.6 + 0.4 * x), size = 3),
    fit = function(d) drmTMB(bf(mu = y ~ x, sigma = ~ 1), family = nbinom2(), data = d)),
  gamma = list(
    sim = function(x) rgamma(n, shape = 4, scale = exp(0.3 + 0.4 * x) / 4),
    fit = function(d) drmTMB(bf(mu = y ~ x, sigma = ~ 1), family = Gamma(link = "log"), data = d)),
  beta = list(
    sim = function(x) { m <- plogis(0.2 + 0.6 * x); rbeta(n, m * 8, (1 - m) * 8) },
    fit = function(d) drmTMB(bf(mu = y ~ x, sigma = ~ 1), family = beta_family(), data = d))
)
set.seed(20261113)
for (nm in names(cells)) {
  cl <- cells[[nm]]
  x <- rnorm(n)
  d <- data.frame(x = x)
  d$y <- cl$sim(x)
  fit <- cl$fit(d)
  r <- as.numeric(residuals(fit, type = "pearson"))
  stopifnot(length(r) == n, all(is.finite(r)))
  write.csv(data.frame(y = d$y, x = d$x, r_pearson = r),
            file.path(out, paste0("pearson_", nm, ".csv")), row.names = FALSE)
}
