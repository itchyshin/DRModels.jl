# Regenerates the review fixtures (data only): interior_{bal,unbal,cov}.csv (same
# draws as the independent review, seed 20261110) and large_{1200,2400}.csv.
# Usage: Rscript gen_review_data.R <outdir>
od <- commandArgs(TRUE)[1]
set.seed(20261110)
gen <- function(sizes, cov = FALSE) {
  G <- length(sizes); g <- rep(sprintf("g%02d", seq_len(G)), sizes); n <- length(g)
  sd1 <- 0.8; sd2 <- 0.6; rg <- 0.5
  z1 <- rnorm(G); z2 <- rnorm(G)
  u1 <- sd1 * z1; u2 <- sd2 * (rg * z1 + sqrt(1 - rg^2) * z2)
  gi <- match(g, unique(g))
  x1 <- rnorm(n); x2 <- rnorm(n)
  s1 <- exp(0.1 + if (cov) 0.3 * x1 else 0); s2 <- exp(-0.1)
  rho <- if (cov) tanh(0.3 + 0.4 * x1) else 0.3
  e1 <- rnorm(n); e2 <- rnorm(n)
  r1 <- s1 * e1; r2 <- s2 * (rho * e1 + sqrt(1 - rho^2) * e2)
  data.frame(g = g, x1 = x1, x2 = x2, y1 = 0.3 + 0.5 * x1 + u1[gi] + r1,
             y2 = -0.3 + 0.4 * x2 + u2[gi] + r2)
}
write.csv(gen(rep(8, 40)), file.path(od, "interior_bal.csv"), row.names = FALSE)
sz <- c(sample(2:12, 30, replace = TRUE), 1, 1, 1, 1)
write.csv(gen(sz), file.path(od, "interior_unbal.csv"), row.names = FALSE)
write.csv(gen(rep(8, 40), cov = TRUE), file.path(od, "interior_cov.csv"), row.names = FALSE)
big <- function(seed, G, k) {
  set.seed(seed); g <- rep(sprintf("g%03d", 1:G), each = k); n <- G * k; gi <- rep(1:G, each = k)
  u1 <- rnorm(G, 0, .8); u2 <- .6 * (.5 * (u1 / .8) + sqrt(.75) * rnorm(G))
  x1 <- rnorm(n); x2 <- rnorm(n); e1 <- rnorm(n); e2 <- .3 * e1 + sqrt(1 - .09) * rnorm(n)
  data.frame(g = g, x1 = x1, x2 = x2, y1 = .3 + .5 * x1 + u1[gi] + e1, y2 = -.3 + .4 * x2 + u2[gi] + .9 * e2)
}
write.csv(big(7, 120, 10), file.path(od, "large_1200.csv"), row.names = FALSE)
write.csv(big(11, 150, 16), file.path(od, "large_2400.csv"), row.names = FALSE)
