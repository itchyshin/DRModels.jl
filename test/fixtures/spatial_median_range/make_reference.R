# Reference values for the fixed median-distance spatial range (drmTMB 0.8.1).
# Simulates one Gaussian and one Binomial data set with a spatial site effect, writes the
# data and the site coordinates, fits drmTMB `spatial(1 | g, coords = ...)` (Gaussian ML,
# Gaussian REML, Binomial ML) and writes the reference table.
# Usage: Rscript make_reference.R <outdir>
# Needs drmTMB 0.8.1 (a library path is set below for the compute host).
.libPaths(c("~/drm_campaign_20261108/Rlib", .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
out <- commandArgs(TRUE)[1]; dir.create(out, showWarnings = FALSE)
set.seed(20261113)
G <- 20; m <- 6; N <- G * m
ids <- sprintf("s%02d", 1:G)
xy <- cbind(runif(G)^2, runif(G)^2)                     # clustered sites
rownames(xy) <- ids
D <- as.matrix(dist(xy))
rng <- median(D[upper.tri(D)][D[upper.tri(D)] > 0])
K <- exp(-D / rng) + 1e-6 * diag(G)
u <- drop(t(chol(K)) %*% rnorm(G))
g <- rep(ids, times = m)
x <- rnorm(N)
write.csv(data.frame(id = ids, easting = xy[, 1], northing = xy[, 2]),
          file.path(out, "coords.csv"), row.names = FALSE)
ug <- u[match(g, ids)]
yg <- 0.3 + 0.5 * x + 0.8 * ug + 0.4 * rnorm(N)
eta <- -0.2 + 0.6 * x + 0.9 * ug
tr <- 8
s <- rbinom(N, tr, plogis(eta))
d <- data.frame(y = yg, s = s, fl = tr - s, x = x, g = g)
write.csv(d, file.path(out, "data.csv"), row.names = FALSE)

rows <- list()
add <- function(cell, fit) {
  sm <- summary(fit)
  co <- as.data.frame(sm$coefficients); sp <- as.data.frame(sm$parameters)
  r <- rbind(data.frame(term = rownames(co), estimate = co$estimate),
             data.frame(term = rownames(sp), estimate = sp$estimate))
  rows[[length(rows) + 1]] <<- cbind(cell = cell, r, loglik = as.numeric(logLik(fit)))
}
fg <- function(reml) drmTMB(bf(y ~ x + spatial(1 | g, coords = xy), sigma = ~ 1),
                            family = gaussian(), data = d, REML = reml)
add("gaussian_ML", fg(FALSE))
add("gaussian_REML", fg(TRUE))
add("binomial_ML", drmTMB(bf(cbind(s, fl) ~ x + spatial(1 | g, coords = xy)),
                          family = binomial(), data = d))
ref <- do.call(rbind, rows)
write.csv(ref, file.path(out, "reference.csv"), row.names = FALSE)
print(ref)
