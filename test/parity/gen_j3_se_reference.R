# gen_j3_se_reference.R -- native drmTMB reference SEs for the #1073 Hessian rewrite (lane J3).
#
#   Rscript --vanilla test/parity/gen_j3_se_reference.R            # all routes
#   DRM_J3_ONLY=poisson_phylo,nb2_phylo Rscript --vanilla ...      # a subset
#
# One folder per route under test/fixtures/j3_se/<route>/ holding the data DRModels.jl
# re-fits (data.csv, tree.newick when the route has a tree) and ref.toml with the drmTMB
# reference numbers, all re-ordered into DRModels.jl's theta order:
#   theta     drmTMB opt$par
#   se        sqrt(diag(sdreport cov.fixed)) at opt$par
#   max_grad  max |gradient of the TMB marginal objective at opt$par| (must be < 1e-6)
#   loglik    drmTMB logLik
# License boundary: numbers and simulated data only. No drmTMB source is copied (see
# test/parity/GENERATING.md). The data are simulated here from a fixed seed, so the
# fixtures regenerate bit-for-bit given the same drmTMB / R versions (recorded in ref.toml).
#
# The design is chosen so every route has an interior optimum (no variance at a boundary):
# a boundary SE is not an estimand, and the Hessian there is singular by construction.

suppressPackageStartupMessages({
  library(drmTMB)
  library(ape)
})

out_root <- file.path("test", "fixtures", "j3_se")
only <- Sys.getenv("DRM_J3_ONLY", "")
only <- if (nzchar(only)) strsplit(only, ",")[[1]] else NULL

fmt <- function(x) paste0("[", paste(formatC(x, digits = 17, format = "g"), collapse = ", "), "]")

write_ref <- function(route, fit, perm, data, tree = NULL, extra = character(),
                      se_ok = TRUE) {
  dir <- file.path(out_root, route)
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(data, file.path(dir, "data.csv"), row.names = FALSE)
  if (!is.null(tree)) {
    ape::write.tree(tree, file.path(dir, "tree.newick"))
    # ape's own tip covariance (height-1 tree, so covariance = correlation): the independent
    # dense reference in test/j3_dense_reference.jl builds its prior from this, not from
    # DRModels' sparse augmented tree precision.
    utils::write.csv(ape::vcv(tree)[levels(data$species), levels(data$species)],
                     file.path(dir, "cov.csv"))
  }
  par <- fit$opt$par
  gr <- as.numeric(fit$obj$gr(par))
  se <- sqrt(diag(fit$sdr$cov.fixed))
  stopifnot(length(perm) == length(par), all(sort(perm) == seq_along(par)))
  lines <- c(
    sprintf("route = \"%s\"", route),
    sprintf("drmTMB_version = \"%s\"", as.character(utils::packageVersion("drmTMB"))),
    sprintf("R_version = \"%s\"", paste(R.version$major, R.version$minor, sep = ".")),
    sprintf("tmb_names = %s",
            paste0("[", paste(sprintf("\"%s\"", names(par)[perm]), collapse = ", "), "]")),
    sprintf("loglik = %s", formatC(as.numeric(logLik(fit)), digits = 17, format = "g")),
    sprintf("max_grad = %s", formatC(max(abs(gr)), digits = 6, format = "e")),
    sprintf("theta = %s", fmt(as.numeric(par)[perm])),
    sprintf("se = %s", fmt(se[perm])),
    extra
  )
  writeLines(lines, file.path(dir, "ref.toml"))
  cat(sprintf("%-18s n=%d  max|grad|=%.2e  se=%s\n", route, nrow(data), max(abs(gr)),
              paste(signif(se[perm], 4), collapse = " ")))
  stopifnot(max(abs(gr)) < 1e-6)
}

want <- function(route) is.null(only) || route %in% only

make_tree <- function(p) {
  tree <- ape::rcoal(p)
  tree$tip.label <- paste0("sp", seq_len(p))
  # Height 1: DRModels' phylo SD is on the raw branch-length scale, drmTMB's on the
  # correlation scale (ape::vcv(tree, corr = TRUE)); they coincide only at tip height 1.
  tree$edge.length <- tree$edge.length / max(ape::node.depth.edgelength(tree))
  tree
}

# ---- phylogenetic count / binomial mean routes ------------------------------------------
phylo_data <- function(seed, p, m, sd_phylo) {
  set.seed(seed)
  tree <- make_tree(p)
  A <- ape::vcv(tree, corr = TRUE)
  u <- as.numeric(t(chol(A)) %*% stats::rnorm(p, 0, sd_phylo))
  sp <- factor(rep(tree$tip.label, each = m), levels = tree$tip.label)
  x <- stats::rnorm(length(sp))
  assign("tree", tree, envir = globalenv())   # phylo() needs a NAMED phylogeny object
  list(tree = tree, u = u, sp = sp, x = x, n = length(sp))
}

if (want("poisson_phylo")) {              # sparse_laplace_glmm.jl :619 (general covariance)
  g <- phylo_data(20261001, 40, 8, 1.0)
  eta <- 0.4 + 0.35 * g$x + g$u[as.integer(g$sp)]
  d <- data.frame(species = g$sp, x = g$x, y = stats::rpois(g$n, exp(eta)))
  f <- drmTMB(bf(y ~ x + phylo(1 | species, tree = tree)), family = poisson(), data = d)
  write_ref("poisson_phylo", f, 1:3, d, g$tree)
}

if (want("nb2_phylo")) {                  # :898 (general-mean nuisance)
  g <- phylo_data(20261002, 40, 8, 1.0)
  eta <- 0.4 + 0.35 * g$x + g$u[as.integer(g$sp)]
  d <- data.frame(species = g$sp, x = g$x,
                  y = stats::rnbinom(g$n, mu = exp(eta), size = 4))
  f <- drmTMB(bf(y ~ x + phylo(1 | species, tree = tree), sigma ~ 1),
              family = nbinom2(), data = d)
  write_ref("nb2_phylo", f, 1:4, d, g$tree)
}

# ---- crossed random intercepts ----------------------------------------------------------
crossed_data <- function(seed, G, H, n, sd_a, sd_b, K3 = FALSE, sd_c = 0.6) {
  set.seed(seed)
  a <- factor(sample(seq_len(G), n, replace = TRUE), levels = seq_len(G))
  b <- factor(sample(seq_len(H), n, replace = TRUE), levels = seq_len(H))
  x <- stats::rnorm(n)
  eta <- 0.3 + 0.4 * x + stats::rnorm(G, 0, sd_a)[a] + stats::rnorm(H, 0, sd_b)[b]
  d <- data.frame(a = paste0("a", a), b = paste0("b", b), x = x)
  if (K3) {
    C <- 15L
    cc <- factor(sample(seq_len(C), n, replace = TRUE), levels = seq_len(C))
    d$c <- paste0("c", cc)
    eta <- eta + stats::rnorm(C, 0, sd_c)[cc]
  }
  list(d = d, eta = eta)
}

if (want("poisson_crossed2")) {           # :1851 (two crossed intercepts)
  g <- crossed_data(20261005, 30, 25, 600, 0.6, 0.5)
  g$d$y <- stats::rpois(600, exp(g$eta))
  f <- drmTMB(drm_formula(y ~ x + (1 | a) + (1 | b)), family = poisson(), data = g$d)
  write_ref("poisson_crossed2", f, 1:4, g$d)
}

if (want("poisson_crossed3")) {           # :2023 (K = 3 components)
  g <- crossed_data(20261006, 30, 25, 800, 0.6, 0.5, K3 = TRUE)
  g$d$y <- stats::rpois(800, exp(g$eta))
  f <- drmTMB(drm_formula(y ~ x + (1 | a) + (1 | b) + (1 | c)), family = poisson(), data = g$d)
  write_ref("poisson_crossed3", f, 1:5, g$d)
}

if (want("binomial_crossed")) {           # :2997 (crossed mean, fixed aux) -- #1073 route 7
  g <- crossed_data(20261007, 30, 25, 700, 0.6, 0.5)
  ntr <- rep(8L, 700)
  g$d$succ <- stats::rbinom(700, ntr, stats::plogis(g$eta)); g$d$fail <- ntr - g$d$succ
  f <- drmTMB(drm_formula(cbind(succ, fail) ~ x + (1 | a) + (1 | b)), family = binomial(),
              data = g$d)
  write_ref("binomial_crossed", f, 1:4, g$d)
}

# ---- ordinary (1 | g) Laplace ----------------------------------------------------------
ordinary_data <- function(seed, G, m, sd_g) {
  set.seed(seed)
  g <- factor(rep(seq_len(G), each = m))
  x <- stats::rnorm(G * m)
  list(d = data.frame(g = paste0("g", g), x = x),
       eta = 0.3 + 0.4 * x + stats::rnorm(G, 0, sd_g)[g], n = G * m)
}

# ---- cumulative logit, phylogenetic mean intercept --------------------------------------

# ---- Gaussian phylogenetic mean (location-only sparse route) ----------------------------

# ---- bivariate Gaussian q = 2, shared phylogenetic intercept (value-closure route) ----------
# DRModels parameterises the 2x2 group covariance by a Cholesky factor, drmTMB by
# (log sd, log sd, atanh corr), so only the first 8 coordinates (the betas, the two log
# residual SDs, atanh rho12 and the log of the first group SD) share a parameterisation. A block of the covariance of
# the other coordinates is invariant under a reparameterisation of THEM, so the SEs of
# those 8 are comparable; `n_shared` tells the test how many to compare.
