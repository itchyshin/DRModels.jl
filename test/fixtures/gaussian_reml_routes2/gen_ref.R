# Gaussian REML, second campaign (2026-11-09): correlated blocks of 3-4 coefficients, a correlated block
# beside crossed intercepts, and structured intercepts with an sd() formula. Simulates 3 seeds per shape,
# fits R drmTMB 0.8.1 with REML = TRUE, and writes the data and the reference table.
# Usage (from a folder holding the refusal-fuzz data data_gaussian_*.csv, A.csv, A_ids.csv, coords.csv):
#   Rscript gen_ref.R <outdir>
# The covariates are the fixed 400-row refusal-fuzz covariates, halved to n = 200 (odd rows), as in
# fixtures/gaussian_reml_routes/gen_ref.R. Thresholds are in docs/dev-log/2026-11-09-gaussian-reml-routes2.md.
.libPaths(c("~/drm_campaign_20261108/Rlib", .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
out <- commandArgs(TRUE)[1]; dir.create(out, showWarnings = FALSE)
ids <- read.csv("A_ids.csv")$id
A <- as.matrix(read.csv("A.csv", header = FALSE)); dimnames(A) <- list(ids, ids); K <- A
cc <- read.csv("coords.csv"); coords <- as.matrix(cc[, c("easting", "northing")]); rownames(coords) <- cc$id
Dm <- as.matrix(dist(coords)); rng <- median(Dm[upper.tri(Dm)][Dm[upper.tri(Dm)] > 0])
Csp <- exp(-Dm / rng) + 1e-6 * diag(nrow(Dm)); dimnames(Csp) <- list(cc$id, cc$id)   # drmTMB's fixed kernel
beta <- c(0.3, 0.5, -0.4, 0.3); se <- 0.7
base <- function(kind) { d <- read.csv(sprintf("data_gaussian_%s.csv", kind)); d <- d[seq(1, nrow(d), by = 2), c("x1", "x2", "x3", "z", "z2", "g", "g2")]; rownames(d) <- NULL; d }
fixed <- function(d) beta[1] + beta[2] * d$x1 + beta[3] * d$x2 + beta[4] * d$x3
sds4 <- c(0.6, 0.5, 0.4, 0.4)
R4 <- matrix(c(1, .4, .3, .2,  .4, 1, .2, .1,  .3, .2, 1, .3,  .2, .1, .3, 1), 4)
ranblock <- function(G, q) { S <- diag(sds4[1:q]) %*% R4[1:q, 1:q] %*% diag(sds4[1:q]); matrix(rnorm(G * q), G) %*% chol(S) }
mvn <- function(M, sd) { L <- chol(M + 1e-8 * diag(nrow(M))); setNames(drop(sd * t(L) %*% rnorm(nrow(M))), rownames(M)) }
refs <- list()
add <- function(shape, seed, fit) {
  s <- summary(fit)
  co <- as.data.frame(s$coefficients); sp <- as.data.frame(s$parameters)
  sp <- sp[!grepl("s[0-9][0-9]$", rownames(sp)), , drop = FALSE]   # per-group SDs of an sd() formula
  rows <- rbind(data.frame(term = rownames(co), estimate = co$estimate, se = co$std_error),
                data.frame(term = rownames(sp), estimate = sp$estimate, se = sp$std_error))
  refs[[length(refs) + 1]] <<- cbind(shape = shape, seed = seed, rows, loglik = as.numeric(logLik(fit)),
    conv = fit$opt$convergence, pdhess = isTRUE(fit$sdr$pdHess), nobs = fit$nobs)
}
fitit <- function(fm, d) { fm <- as.formula(fm, env = globalenv()); drmTMB(bf(fm, sigma = ~ 1), family = gaussian(), data = d, REML = TRUE) }
fitsd <- function(fm, sdf, d) { fm <- as.formula(fm, env = globalenv()); b <- do.call(bf, list(fm, sigma = ~ 1, as.formula(sdf, env = globalenv()))); drmTMB(b, family = gaussian(), data = d, REML = TRUE) }
for (seed in c(21, 22, 23)) {
  set.seed(seed)
  # ordinary correlated blocks, q = 3 and 4
  d <- base("ordinary"); g <- factor(d$g, levels = unique(d$g)); gi <- as.integer(g); G <- nlevels(g)
  u <- ranblock(G, 4); eps <- rnorm(nrow(d), 0, se)
  d$y3 <- fixed(d) + u[gi, 1] + u[gi, 2] * d$x1 + u[gi, 3] * d$x2 + eps
  d$y4 <- fixed(d) + u[gi, 1] + u[gi, 2] * d$x1 + u[gi, 3] * d$x2 + u[gi, 4] * d$x3 + eps
  write.csv(d, file.path(out, sprintf("ord_s%d.csv", seed)), row.names = FALSE)
  d$g <- g
  add("ord3", seed, fitit("y3 ~ x1 + x2 + x3 + (1 + x1 + x2 | g)", d))
  add("ord4", seed, fitit("y4 ~ x1 + x2 + x3 + (1 + x1 + x2 + x3 | g)", d))
  add("ord3lab", seed, fitit("y3 ~ x1 + x2 + x3 + (1 + x1 + x2 | p | g)", d))
  # a correlated block beside a crossed intercept, q = 2, 3, 4
  d <- base("crossed"); g <- factor(d$g, levels = unique(d$g)); gi <- as.integer(g); G <- nlevels(g)
  g2 <- factor(d$g2); u <- ranblock(G, 4); v <- rnorm(nlevels(g2), 0, .5)[as.integer(g2)]; eps <- rnorm(nrow(d), 0, se)
  d$y2 <- fixed(d) + u[gi, 1] + u[gi, 2] * d$x1 + v + eps
  d$y3 <- fixed(d) + u[gi, 1] + u[gi, 2] * d$x1 + u[gi, 3] * d$x2 + v + eps
  d$y4 <- fixed(d) + u[gi, 1] + u[gi, 2] * d$x1 + u[gi, 3] * d$x2 + u[gi, 4] * d$x3 + v + eps
  write.csv(d, file.path(out, sprintf("crossed_s%d.csv", seed)), row.names = FALSE)
  d$g <- g; d$g2 <- g2
  add("cross2", seed, fitit("y2 ~ x1 + x2 + x3 + (1 + x1 | g) + (1 | g2)", d))
  add("cross3", seed, fitit("y3 ~ x1 + x2 + x3 + (1 + x1 + x2 | g) + (1 | g2)", d))
  add("cross4", seed, fitit("y4 ~ x1 + x2 + x3 + (1 + x1 + x2 + x3 | g) + (1 | g2)", d))
  # structured intercept with an sd() formula: animal, relmat, spatial; sd ~ 1 and sd ~ z
  for (kind in c("animal", "relmat", "spatial")) {
    d <- base(kind); Cm <- switch(kind, animal = A, relmat = K, spatial = Csp)
    zg <- tapply(d$z, d$g, mean); sdg <- exp(-0.4 + 0.8 * zg)[ids]
    a <- mvn(Cm, 1)[ids]
    d$y_sd1 <- fixed(d) + 0.7 * a[d$g] + rnorm(nrow(d), 0, se)
    d$y_sdz <- fixed(d) + (sdg * a)[d$g] + rnorm(nrow(d), 0, se)
    write.csv(d, file.path(out, sprintf("%s_s%d.csv", kind, seed)), row.names = FALSE)
    d$g <- factor(d$g, levels = ids)
    term <- switch(kind, animal = "animal(1 | g, A = A)", relmat = "relmat(1 | g, K = K)", spatial = "spatial(1 | g, coords = coords)")
    lv <- sprintf('sd(g, level = "%s")', kind)
    add(paste0(kind, "_sd1"), seed, fitsd(paste("y_sd1 ~ x1 + x2 + x3 +", term), paste(lv, "~ 1"), d))
    add(paste0(kind, "_sdz"), seed, fitsd(paste("y_sdz ~ x1 + x2 + x3 +", term), paste(lv, "~ z"), d))
  }
}
ref <- do.call(rbind, refs)
write.csv(ref, file.path(out, "r_reml_reference.csv"), row.names = FALSE)
print(unique(ref[, c("shape", "seed", "loglik", "conv", "pdhess")]), row.names = FALSE)
