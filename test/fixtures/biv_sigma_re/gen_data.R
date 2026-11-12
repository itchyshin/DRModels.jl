# Writes the interior data sets of the bivariate sigma random-effect tests (seed 20261112):
# a correlated random intercept pair on log sigma1 and log sigma2 (SD 0.35 and 0.25, correlation 0.5),
# residual correlation 0.45. Usage: Rscript gen_data.R <outdir>
out <- commandArgs(TRUE)[1]
sim <- function(sizes, seed, xs1 = 0) {
  set.seed(seed)
  G <- length(sizes); g <- rep(sprintf("g%02d", seq_len(G)), sizes); n <- length(g)
  S <- matrix(c(0.35^2, 0.5 * 0.35 * 0.25, 0.5 * 0.35 * 0.25, 0.25^2), 2)
  u <- matrix(rnorm(2 * G), G) %*% chol(S)
  x1 <- rnorm(n); x2 <- rnorm(n)
  e1 <- rnorm(n); e2 <- 0.45 * e1 + sqrt(1 - 0.45^2) * rnorm(n)
  s1 <- exp(-0.1 + xs1 * x1 + u[match(g, unique(g)), 1]); s2 <- exp(-0.3 + u[match(g, unique(g)), 2])
  y1 <- 0.3 + 0.5 * x1 + s1 * e1; y2 <- -0.3 + 0.4 * x2 + s2 * e2
  data.frame(g = g, x1 = x1, x2 = x2, y1 = y1, y2 = y2)
}
write.csv(sim(rep(10, 30), 1), file.path(out, "sre_bal.csv"), row.names = FALSE)
write.csv(sim(c(3:12, 14, 16, 18, 20, 3:12), 2), file.path(out, "sre_unbal.csv"), row.names = FALSE)
write.csv(sim(c(3:12, 14, 16, 18, 20, 3:12), 3, xs1 = 0.3), file.path(out, "sre_cov.csv"), row.names = FALSE)
