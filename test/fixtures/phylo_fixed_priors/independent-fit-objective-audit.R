# Independent dense-objective and objective-only central-difference check.
# Created after preliminary package comparisons; this is not a blind freeze.
d <- jsonlite::fromJSON('public-tree-fit-frozen.json')
K <- d$K; g <- d$group; x <- d$x; z <- d$z; y <- d$y
X <- cbind(1, x); H <- cbind(1, z); q <- d$theta
f <- function(t) {
  a <- exp(H %*% t[5:6])
  v <- K[g, g] * tcrossprod(a[g]) + diag(exp(2 * X %*% t[3:4])[, 1])
  u <- chol(v); r <- y - X %*% t[1:2]
  nll <- (length(y) * log(2 * pi) + 2 * sum(log(diag(u))) +
          sum(r * solve(v, r))) / 2
  nll - sum(dnorm(t[c(2, 4)], c(.1, -.2), c(.7, .5), log = TRUE))
}
h <- 1e-4; E <- diag(h, 6); q0 <- f(q)
grad <- vapply(1:6, function(i) (f(q + E[, i]) - f(q - E[, i])) / (2 * h), 0.0)
HH <- outer(1:6, 1:6, Vectorize(function(i, j)
  (f(q + E[, i] + E[, j]) - f(q + E[, i] - E[, j]) -
   f(q - E[, i] + E[, j]) + f(q - E[, i] - E[, j])) / (4 * h * h)))
obj_error <- abs(q0 - d$objective)
score_error <- max(abs(grad))
hessian_error <- max(abs(HH - d$hessian))
hessian_scaled <- max(abs(HH - d$hessian) / pmax(1, abs(d$hessian)))
min_eig <- min(eigen(HH, symmetric = TRUE, only.values = TRUE)$values)
cat('objective abs error', obj_error, 'score fd max', score_error,
    'hessian max abs', hessian_error, 'hessian scaled max', hessian_scaled,
    'min eig', min_eig, '\n')
stopifnot(obj_error < 1e-8, score_error < 1e-6, hessian_scaled < 1e-5, min_eig > 0)
cat('INDEPENDENT_FIT_OBJECTIVE_AUDIT_PASS\n')
