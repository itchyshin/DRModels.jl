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

drmtmb_commit <- Sys.getenv("DRMTMB_COMMIT", unset = NA_character_)

write_ref <- function(route, fit, perm, data, tree = NULL, extra = character(),
                      se_ok = TRUE) {
  dir <- file.path(out_root, route)
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(data, file.path(dir, "data.csv"), row.names = FALSE)
  if (!is.null(tree)) ape::write.tree(tree, file.path(dir, "tree.newick"), digits = 17)
  par <- fit$opt$par
  gr <- as.numeric(fit$obj$gr(par))
  se <- sqrt(diag(fit$sdr$cov.fixed))
  stopifnot(length(perm) == length(par), all(sort(perm) == seq_along(par)))
  # Per-row validity of the REFERENCE: drmTMB's own optimiser converged and its sdreport
  # Hessian is positive definite. If either fails the row is "not comparable" (neither pass
  # nor fail) and the Julia test reports it as such instead of comparing numbers.
  conv_ok <- identical(as.integer(fit$opt$convergence), 0L)
  pd_ok <- isTRUE(fit$sdr$pdHess)
  comparable <- conv_ok && pd_ok && all(is.finite(se))
  lines <- c(
    sprintf("route = \"%s\"", route),
    sprintf("drmTMB_version = \"%s\"", as.character(utils::packageVersion("drmTMB"))),
    sprintf("drmTMB_commit = \"%s\"", if (is.na(drmtmb_commit)) "unrecorded" else drmtmb_commit),
    sprintf("R_version = \"%s\"", paste(R.version$major, R.version$minor, sep = ".")),
    sprintf("opt_convergence = %d", as.integer(fit$opt$convergence)),
    sprintf("pd_hess = %s", tolower(as.character(pd_ok))),
    sprintf("comparable = %s", tolower(as.character(comparable))),
    sprintf("tmb_names = %s",
            paste0("[", paste(sprintf("\"%s\"", names(par)[perm]), collapse = ", "), "]")),
    sprintf("loglik = %s", formatC(as.numeric(logLik(fit)), digits = 17, format = "g")),
    sprintf("max_grad = %s", formatC(max(abs(gr)), digits = 6, format = "e")),
    sprintf("theta = %s", fmt(as.numeric(par)[perm])),
    sprintf("se = %s", fmt(se[perm])),
    extra
  )
  writeLines(lines, file.path(dir, "ref.toml"))
  cat(sprintf("%-22s n=%d  conv=%d pd.hess=%s  max|grad|=%.2e  se=%s\n", route, nrow(data),
              as.integer(fit$opt$convergence), pd_ok, max(abs(gr)),
              paste(signif(se[perm], 4), collapse = " ")))
  if (comparable) stopifnot(max(abs(gr)) < 1e-6)
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

if (want("nb2_phylo_hetero")) {           # :1091 (phylo mean, covariate dispersion)
  g <- phylo_data(20261003, 40, 8, 1.0)
  eta <- 0.4 + 0.35 * g$x + g$u[as.integer(g$sp)]
  size <- exp(1.2 + 0.5 * g$x)
  d <- data.frame(species = g$sp, x = g$x, y = stats::rnbinom(g$n, mu = exp(eta), size = size))
  f <- drmTMB(bf(y ~ x + phylo(1 | species, tree = tree), sigma ~ x),
              family = nbinom2(), data = d)
  write_ref("nb2_phylo_hetero", f, 1:5, d, g$tree)
}

if (want("binomial_phylo")) {             # :1244 (phylo mean, fixed aux)
  g <- phylo_data(20261004, 40, 8, 1.0)
  eta <- 0.2 + 0.5 * g$x + g$u[as.integer(g$sp)]
  ntr <- rep(10L, g$n)
  d <- data.frame(species = g$sp, x = g$x, succ = stats::rbinom(g$n, ntr, stats::plogis(eta)),
                  fail = NA_integer_)
  d$fail <- ntr - d$succ
  f <- drmTMB(bf(cbind(succ, fail) ~ x + phylo(1 | species, tree = tree)),
              family = binomial(), data = d)
  write_ref("binomial_phylo", f, 1:3, d, g$tree)
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

if (want("ordinary_poisson")) {           # ordinary_laplace.jl :193 (Poisson kernel)
  o <- ordinary_data(20261008, 40, 8, 0.7)
  o$d$y <- stats::rpois(o$n, exp(o$eta))
  f <- drmTMB(drm_formula(y ~ x + (1 | g)), family = poisson(), data = o$d)
  write_ref("ordinary_poisson", f, 1:3, o$d)
}

if (want("ordinary_nb2")) {               # ordinary_laplace.jl :193 (scale kernel)
  o <- ordinary_data(20261009, 40, 8, 0.7)
  o$d$y <- stats::rnbinom(o$n, mu = exp(o$eta), size = 4)
  f <- drmTMB(drm_formula(y ~ x + (1 | g), sigma ~ 1), family = nbinom2(), data = o$d)
  write_ref("ordinary_nb2", f, 1:4, o$d)
}

# ---- cumulative logit, phylogenetic mean intercept --------------------------------------
if (want("cumulative_phylo")) {           # cumulative.jl :614
  g <- phylo_data(1, 60, 6, 1.6)
  lat <- 0.6 * g$x + g$u[as.integer(g$sp)] + stats::rlogis(g$n)
  y <- findInterval(lat, c(-0.5, 0.6)) + 1L
  d <- data.frame(species = g$sp, x = g$x, y = y)
  f <- drmTMB(bf(y ~ x + phylo(1 | species, tree = tree)), family = cumulative_logit(),
              data = within(d, y <- ordered(y, levels = 1:3)))
  write_ref("cumulative_phylo", f, 1:4, d, g$tree)
}

# ---- Gaussian phylogenetic mean (location-only sparse route) ----------------------------
if (want("gaussian_phylo")) {             # location_only.jl :3540
  g <- phylo_data(20261010, 40, 6, 1.0)
  d <- data.frame(species = g$sp, x = g$x,
                  y = 0.3 + 0.5 * g$x + g$u[as.integer(g$sp)] + stats::rnorm(g$n, 0, 0.7))
  f <- drmTMB(bf(y ~ x + phylo(1 | species, tree = tree), sigma ~ 1), family = gaussian(),
              data = d)
  write_ref("gaussian_phylo", f, 1:4, d, g$tree)
}

# ---- bivariate Gaussian q = 2, shared phylogenetic intercept (value-closure route) ----------
# DRModels parameterises the 2x2 group covariance by a Cholesky factor, drmTMB by
# (log sd, log sd, atanh corr), so only the first 8 coordinates (the betas, the two log
# residual SDs, atanh rho12 and the log of the first group SD) share a parameterisation. A block of the covariance of
# the other coordinates is invariant under a reparameterisation of THEM, so the SEs of
# those 8 are comparable; `n_shared` tells the test how many to compare.
if (want("biv_phylo")) {                  # gaussian_bivariate.jl :887 (no exact gradient)
  set.seed(901)
  p <- 120L; m <- 4L
  tree <- make_tree(p)
  A <- ape::vcv(tree, corr = TRUE)
  Lam <- matrix(c(0.6, 0.3, 0.3, 0.5), 2)
  U <- t(chol(A)) %*% matrix(stats::rnorm(p * 2), p, 2) %*% chol(Lam)
  species <- factor(rep(tree$tip.label, each = m), levels = tree$tip.label)
  n <- length(species); x <- stats::rnorm(n)
  e <- matrix(stats::rnorm(2 * n), n, 2) %*% chol(matrix(c(0.4, 0.1, 0.1, 0.5), 2))
  d <- data.frame(species = species, x = x,
                  y1 = 0.2 + 0.3 * x + U[as.integer(species), 1] + e[, 1],
                  y2 = -0.1 + 0.2 * x + U[as.integer(species), 2] + e[, 2])
  assign("tree", tree, envir = globalenv())
  f <- drmTMB(bf(mu1 = y1 ~ x + phylo(1 | species, tree = tree),
                 mu2 = y2 ~ x + phylo(1 | species, tree = tree),
                 sigma1 ~ 1, sigma2 ~ 1, rho12 ~ 1),
              family = biv_gaussian(), data = d)
  write_ref("biv_phylo", f, 1:10, d, tree, extra = "n_shared = 8")
}

# ---- Engine A additions (#1073) -----------------------------------------------------------
# Rows added after the Simulator / Data fitter audits. The Simulator's `gen.R` is not in
# this checkout, so the three crossed families below are simulated HERE from the audit's
# stated design (n = 400, 40/25/15 levels, sd 0.5/0.4/0.3, shape 4, 10 binomial trials) and
# its seed formula; they are the same designs, not byte-identical datasets.
sim_seed <- function(route, rep) (20281016000 + route * 100 + rep) %% 2147483647

sim_crossed <- function(route, rep, K3 = FALSE) {
  set.seed(sim_seed(route, rep))
  n <- 400L
  g <- sample(seq_len(40L), n, replace = TRUE); h <- sample(seq_len(25L), n, replace = TRUE)
  x <- stats::rnorm(n)
  eta <- 0.3 + 0.4 * x + stats::rnorm(40L, 0, 0.5)[g] + stats::rnorm(25L, 0, 0.4)[h]
  d <- data.frame(g = paste0("g", g), h = paste0("h", h), x = x)
  if (K3) {
    k <- sample(seq_len(15L), n, replace = TRUE)
    d$k <- paste0("k", k)
    eta <- eta + stats::rnorm(15L, 0, 0.3)[k]
  }
  list(d = d, eta = eta, n = n)
}

for (rep in 1:4) {
  r <- sprintf("sim_poisson_crossed3_s%d", rep)       # sparse_laplace_glmm.jl :2023
  if (want(r)) {
    s <- sim_crossed(6, rep, K3 = TRUE); s$d$y <- stats::rpois(s$n, exp(s$eta))
    f <- drmTMB(drm_formula(y ~ x + (1 | g) + (1 | h) + (1 | k)), family = poisson(), data = s$d)
    write_ref(r, f, 1:5, s$d)
  }
  r <- sprintf("sim_binomial_crossed_s%d", rep)       # :2997
  if (want(r)) {
    s <- sim_crossed(7, rep)
    s$d$succ <- stats::rbinom(s$n, 10L, stats::plogis(s$eta)); s$d$fail <- 10L - s$d$succ
    f <- drmTMB(drm_formula(cbind(succ, fail) ~ x + (1 | g) + (1 | h)), family = binomial(), data = s$d)
    write_ref(r, f, 1:4, s$d)
  }
  r <- sprintf("sim_gamma_crossed_s%d", rep)          # :3170 (Gamma)
  if (want(r)) {
    s <- sim_crossed(8, rep)
    s$d$y <- stats::rgamma(s$n, shape = 4, rate = 4 / exp(s$eta))
    f <- drmTMB(drm_formula(y ~ x + (1 | g) + (1 | h), sigma ~ 1), family = Gamma(link = "log"),
                data = s$d)
    write_ref(r, f, 1:5, s$d)
  }
}

if (want("alfalfa_gamma_crossed")) {      # Wave E: nlme::Alfalfa, Block x Variety
  a <- nlme::Alfalfa
  d <- data.frame(Yield = a$Yield, Date = as.character(a$Date), Block = as.character(a$Block),
                  Variety = as.character(a$Variety))
  f <- drmTMB(drm_formula(Yield ~ Date + (1 | Block) + (1 | Variety), sigma ~ 1),
              family = Gamma(link = "log"), data = d)
  write_ref("alfalfa_gamma_crossed", f, 1:7, d)
}

# ---- F4: ape::carnivora, Gaussian + phylo mean (location_only.jl :3540), ML and REML ---------
carnivora_data <- function() {
  utils::data("carnivora", package = "ape", envir = environment())
  cv <- get("carnivora", envir = environment())
  cv$Species <- gsub(" ", "_", as.character(cv$Species))
  cv <- cv[!is.na(cv$SB) & !is.na(cv$SW) & !duplicated(cv$Species), ]
  for (k in c("SuperFamily", "Family", "Genus", "Species")) cv[[k]] <- factor(as.character(cv[[k]]))
  tr <- ape::compute.brlen(ape::as.phylo(~SuperFamily/Family/Genus/Species, data = cv),
                           method = "Grafen")
  tr <- ape::multi2di(tr, random = FALSE)
  tr$edge.length[tr$edge.length <= 0] <- 1e-6
  tr$node.label <- NULL
  dep <- ape::node.depth.edgelength(tr); n <- ape::Ntip(tr)
  te <- match(seq_len(n), tr$edge[, 2])
  tr$edge.length[te] <- tr$edge.length[te] + (max(dep[seq_len(n)]) - dep[seq_len(n)])
  tr$edge.length <- tr$edge.length / max(ape::node.depth.edgelength(tr)[seq_len(n)])
  list(tree = tr, data = data.frame(species = cv$Species, lSB = log(cv$SB), lSW = log(cv$SW)))
}

for (reml in c(FALSE, TRUE)) {
  r <- if (reml) "carnivora_reml" else "carnivora_ml"
  if (want(r)) {
    cc <- carnivora_data(); assign("tree", cc$tree, envir = globalenv())
    f <- drmTMB(bf(lSB ~ lSW + phylo(1 | species, tree = tree), sigma ~ 1), family = gaussian(),
                data = cc$data, REML = reml)
    write_ref(r, f, 1:4, cc$data, cc$tree)
  }
}
