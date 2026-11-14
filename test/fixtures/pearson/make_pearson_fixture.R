# Generates the pinned residual values in test/fixtures/pearson/pearson_<family>.csv.
# Run on a machine with drmTMB 0.8.1:
#   Rscript make_pearson_fixture.R <output directory>
# Each CSV has the data (y, x) and R's residuals(fit, type = "pearson") for one fixed-effect cell.
# Two extra cells follow the loop: pearson_gaussian_metav.csv (diagonal meta_V, column v) and
# pearson_poisson_miss.csv (five rows with a missing response; R returns the kept rows only, so the
# residual is padded with NA at the dropped rows here).
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

# Gaussian with a diagonal known sampling variance, `meta_V(V = v)`, no random effect.
n <- 120
x <- rnorm(n); v <- runif(n, 0.05, 0.4)
d <- data.frame(x = x, v = v)
d$y <- rnorm(n, 0.5 + 0.8 * x, sqrt(v + 0.3^2))
fit <- drmTMB(bf(mu = y ~ x + meta_V(V = v), sigma = ~ 1), family = gaussian(), data = d)
r <- as.numeric(residuals(fit, type = "pearson"))
stopifnot(length(r) == n, all(is.finite(r)))
write.csv(data.frame(y = d$y, x = d$x, v = d$v, r_pearson = r),
          file.path(out, "pearson_gaussian_metav.csv"), row.names = FALSE)

# Poisson with five rows whose response is missing: R returns the kept rows only.
n <- 100
x <- rnorm(n)
d <- data.frame(x = x)
d$y <- rpois(n, exp(0.4 + 0.5 * x))
drop <- c(2, 9, 41, 60, 97)
d$y[drop] <- NA
fit <- drmTMB(bf(mu = y ~ x), family = poisson(), data = d)
r <- as.numeric(residuals(fit, type = "pearson"))
stopifnot(length(r) == n - length(drop))
rfull <- rep(NA_real_, n)
rfull[-drop] <- r
write.csv(data.frame(y = d$y, x = d$x, r_pearson = rfull),
          file.path(out, "pearson_poisson_miss.csv"), row.names = FALSE)
