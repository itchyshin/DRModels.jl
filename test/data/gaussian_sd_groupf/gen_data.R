# Three committed datasets for the Gaussian SD "Group F" closure (docs/dev-log/2026-11-09-gaussian-sd-groupf.md).
# Base R only. n = 300 rows; `g` has 30 levels (10 rows each), `g2` has 15 levels; z is constant within g,
# z2 constant within g2. Two responses share the covariates:
#   y  : correlated block (1 + x1 | g) with SDs depending on z, an independent slope x3 on g,
#        and a correlated block (1 + x2 | g2) with SDs depending on z2 (shapes with a block beside slope terms);
#   yf : mean intercept on g with SD depending on z, correlated with a sigma intercept on g (rho = 0.5),
#        plus an independent sigma intercept on g2 (mean + sigma random-effect shapes).
# Run: Rscript gen_data.R   (writes gsf_s1.csv, gsf_s2.csv, gsf_s3.csv next to this script)
out_dir <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])))
for (s in 1:3) {
  set.seed(20261110 + s)
  G <- 30; G2 <- 15; n <- 300
  g <- rep(seq_len(G), each = n / G)
  g2 <- sample(rep(seq_len(G2), length.out = n))
  z <- scale(rnorm(G))[, 1]; z2 <- scale(rnorm(G2))[, 1]
  X <- cbind(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
  # family G
  Bg <- t(sapply(seq_len(G), function(k) {
    D <- diag(c(0.5, 0.4) * exp(0.2 * z[k])); R <- matrix(c(1, 0.3, 0.3, 1), 2)
    t(chol(D %*% R %*% D)) %*% rnorm(2)
  }))
  cg <- rnorm(G, 0, 0.3)
  Bh <- t(sapply(seq_len(G2), function(k) {
    D <- diag(c(0.4, 0.35) * exp(0.2 * z2[k])); R <- matrix(c(1, -0.3, -0.3, 1), 2)
    t(chol(D %*% R %*% D)) %*% rnorm(2)
  }))
  mu <- 0.3 + X %*% c(0.5, -0.3, 0.2) + Bg[g, 1] + X[, 1] * Bg[g, 2] + X[, 3] * cg[g] +
    Bh[g2, 1] + X[, 2] * Bh[g2, 2]
  y <- as.numeric(mu + rnorm(n, 0, 0.5))
  # family F
  a <- rnorm(G); w <- rnorm(G)
  bg <- 0.6 * exp(0.3 * z) * a
  dg <- 0.4 * (0.5 * a + sqrt(0.75) * w)
  d2 <- rnorm(G2, 0, 0.35)
  yf <- as.numeric(0.3 + 0.5 * X[, 1] + bg[g] + exp(log(0.5) + dg[g] + d2[g2]) * rnorm(n))
  d <- data.frame(y = y, yf = yf, X, z = z[g], z2 = z2[g2], g = sprintf("g%02d", g), g2 = sprintf("h%02d", g2))
  write.csv(d, file.path(out_dir, sprintf("gsf_s%d.csv", s)), row.names = FALSE, quote = FALSE)
}
