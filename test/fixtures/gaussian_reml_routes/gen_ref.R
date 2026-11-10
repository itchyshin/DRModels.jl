# Gaussian REML routes: simulate (3 seeds x 5 designs), fit with R drmTMB REML = TRUE, write data + reference.
# Usage: Rscript gen_ref.R <outdir>   (data_gaussian_*.csv, tree.nwk, A.csv, A_ids.csv alongside)
.libPaths(c("~/drm_campaign_20261108/Rlib", .libPaths()))
suppressPackageStartupMessages({library(drmTMB); library(ape)})
out <- commandArgs(TRUE)[1]; dir.create(out, showWarnings = FALSE)
tree <- read.tree("tree.nwk")
ids <- read.csv("A_ids.csv")$id
A <- as.matrix(read.csv("A.csv", header = FALSE)); dimnames(A) <- list(ids, ids); K <- A
C <- vcv(tree, corr = TRUE)
beta <- c(0.3, 0.5, -0.4, 0.3); se <- 0.8
base <- function(kind) { d <- read.csv(sprintf("data_gaussian_%s.csv", kind)); d <- d[seq(1, nrow(d), by = 2), c("x1", "x2", "x3", "g", "g2")]; rownames(d) <- NULL; d }
mvn_struct <- function(M, sd) { lv <- rownames(M); L <- chol(M + 1e-8 * diag(nrow(M))); setNames(sd * drop(t(L) %*% rnorm(nrow(M))), lv) }
refs <- list()
add <- function(shape, seed, fit) {
  s <- summary(fit)
  co <- as.data.frame(s$coefficients); sp <- as.data.frame(s$parameters)
  ll <- as.numeric(logLik(fit)); conv <- fit$opt$convergence; pd <- isTRUE(fit$sdr$pdHess)
  rows <- rbind(data.frame(term = rownames(co), estimate = co$estimate, se = co$std_error),
                data.frame(term = sp$parm, estimate = sp$estimate, se = sp$std_error))
  refs[[length(refs) + 1]] <<- cbind(shape = shape, seed = seed, rows, loglik = ll, conv = conv, pdhess = pd, nobs = fit$nobs)
}
fixed <- function(d) beta[1] + beta[2] * d$x1 + beta[3] * d$x2 + beta[4] * d$x3
fitit <- function(fm, d, ...) { fm <- as.formula(fm, env = globalenv()); drmTMB(bf(fm), family = gaussian(), data = d, REML = TRUE) }
for (seed in c(11, 12, 13)) {
  set.seed(seed)
  # ordinary covariates: slope + indep designs
  d <- base("ordinary"); g <- factor(d$g, levels = unique(d$g)); gi <- as.integer(g); G <- nlevels(g)
  u0 <- rnorm(G, 0, .6); u1 <- rnorm(G, 0, .5); u2 <- rnorm(G, 0, .4); u3 <- rnorm(G, 0, .4)
  eps <- rnorm(nrow(d), 0, se)
  d$y_slope <- fixed(d) + u1[gi] * d$x1 + eps
  d$y_i2 <- fixed(d) + u0[gi] + u1[gi] * d$x1 + eps
  d$y_i3 <- fixed(d) + u0[gi] + u1[gi] * d$x1 + u2[gi] * d$x2 + eps
  d$y_i4 <- fixed(d) + u0[gi] + u1[gi] * d$x1 + u2[gi] * d$x2 + u3[gi] * d$x3 + eps
  write.csv(d, file.path(out, sprintf("ord_s%d.csv", seed)), row.names = FALSE)
  d$g <- g
  add("slope", seed, fitit("y_slope ~ x1 + x2 + x3 + (0 + x1 | g)", d))
  add("indep2", seed, fitit("y_i2 ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g)", d))
  add("indep3", seed, fitit("y_i3 ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g) + (0 + x2 | g)", d))
  add("indep4", seed, fitit("y_i4 ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g) + (0 + x2 | g) + (0 + x3 | g)", d))
  # crossed intercepts
  d <- base("crossed"); g <- factor(d$g, levels = unique(d$g)); g2 <- factor(d$g2)
  d$y_cross <- fixed(d) + rnorm(nlevels(g), 0, .7)[as.integer(g)] + rnorm(nlevels(g2), 0, .5)[as.integer(g2)] + rnorm(nrow(d), 0, se)
  write.csv(d, file.path(out, sprintf("crossed_s%d.csv", seed)), row.names = FALSE)
  d$g <- g; d$g2 <- g2
  add("cross1", seed, fitit("y_cross ~ x1 + x2 + x3 + (1 | g) + (1 | g2)", d))
  # animal / relmat
  for (kind in c("animal", "relmat")) {
    d <- base(kind); a <- mvn_struct(A, .8)
    d$y <- fixed(d) + a[d$g] + rnorm(nrow(d), 0, se)
    write.csv(d, file.path(out, sprintf("%s_s%d.csv", kind, seed)), row.names = FALSE)
    d$g <- factor(d$g, levels = ids)
    if (kind == "animal") add("animal", seed, fitit("y ~ x1 + x2 + x3 + animal(1 | g, A = A)", d, A = A))
    else add("relmat", seed, fitit("y ~ x1 + x2 + x3 + relmat(1 | g, K = K)", d, K = K))
  }
  # phylo intercept + slope
  d <- base("phylo"); tips <- rownames(C)
  a <- mvn_struct(C, .7); b <- mvn_struct(C, .4)
  d$y <- fixed(d) + a[d$g] + b[d$g] * d$x1 + rnorm(nrow(d), 0, se)
  write.csv(d, file.path(out, sprintf("phylo_s%d.csv", seed)), row.names = FALSE)
  add("phylo2", seed, fitit("y ~ x1 + x2 + x3 + phylo(1 + x1 | g, tree = tree)", d, tree = tree))
}
ref <- do.call(rbind, refs)
write.csv(ref, file.path(out, "r_reml_reference.csv"), row.names = FALSE)
print(unique(ref[, c("shape", "seed", "loglik", "conv", "pdhess")]))
