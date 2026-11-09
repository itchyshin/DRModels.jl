# Three committed datasets for the Gaussian SD-gap closure (docs/dev-log/2026-11-09-gaussian-sd-gaps.md).
# Base R only. One file serves every shape: a correlated 4-coefficient block on `g` (30 levels),
# an extra random intercept on `g2` (12 levels), group-level predictors `z` (on g) and `z2` (on g2).
# Run: Rscript gen_data.R   (writes gsd_s1.csv, gsd_s2.csv, gsd_s3.csv next to this script)
out_dir <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])))
for (s in 1:3) {
  set.seed(20261109 + s)
  G <- 30; G2 <- 12; n <- 240
  g <- rep(seq_len(G), each = n / G)
  g2 <- sample(rep(seq_len(G2), length.out = n))
  z <- scale(rnorm(G))[, 1]; z2 <- scale(rnorm(G2))[, 1]
  X <- cbind(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
  R <- diag(4); R[lower.tri(R)] <- c(0.3, 0.2, 0.1, 0.25, 0.15, 0.2); R <- R + t(R) - diag(4)
  B <- matrix(0, G, 4)
  for (k in seq_len(G)) {
    D <- diag(c(0.6, 0.4, 0.3, 0.25) * exp(0.2 * z[k]))
    B[k, ] <- t(chol(D %*% R %*% D)) %*% rnorm(4)
  }
  h <- rnorm(G2, 0, 0.4 * exp(0.2 * z2))
  U <- cbind(1, X)
  mu <- 0.3 + X %*% c(0.5, -0.3, 0.2) + rowSums(U * B[g, ]) + h[g2]
  y <- as.numeric(mu + rnorm(n, 0, 0.5))
  d <- data.frame(y = y, X, z = z[g], z2 = z2[g2], g = sprintf("g%02d", g), g2 = sprintf("h%02d", g2))
  write.csv(d, file.path(out_dir, sprintf("gsd_s%d.csv", s)), row.names = FALSE, quote = FALSE)
}
