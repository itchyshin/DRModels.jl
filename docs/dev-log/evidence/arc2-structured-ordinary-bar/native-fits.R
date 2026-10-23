# Arc 2 `structured_with_ordinary_bar` receipt -- NATIVE side (drmTMB, engine = "tmb").
#
# Builds every fixture, fits it natively, and writes
#   fixtures/<id>.data.tsv      the rows (columns: y, x, grouping factors)
#   fixtures/<id>.tree.nwk      the tree (phylo fixtures)
#   fixtures/<id>.K.tsv         the fixed K / A over the marker levels in
#                               FIRST-SEEN data order (what the bridge sends)
#   native.tsv                  fixture, param, value  (df, logLik, estimates)
# julia-fits.jl reads the same files and fits them through DRModels.drm_bridge.
#
# Run from this directory:
#   Rscript native-fits.R <path-to-drmTMB-checkout>
# Measured against drmTMB at ~/local-scratch/lanes/drmTMB-arc1-pr1304-fold
# (709efbeb0, origin/main + Arc 1 evidence files, DLL built).
args <- commandArgs(trailingOnly = TRUE)
pkg <- if (length(args) >= 1L) args[[1L]] else "~/local-scratch/lanes/drmTMB-arc1-pr1304-fold"
suppressMessages(pkgload::load_all(pkg, quiet = TRUE))
dir.create("fixtures", showWarnings = FALSE)

ar1 <- function(k, rho) rho^abs(outer(seq_len(k), seq_len(k), "-"))
first_seen <- function(v) unique(as.character(v))
rmvn <- function(S, sd) as.numeric(t(chol(S)) %*% stats::rnorm(nrow(S))) * sd

# Each fixture: a list(id, formula (native), julia (formula string for
# drm_bridge), kind, data, tree, K, kwarg).
fixtures <- list()

# F1/F2: phylo(1 | sp) + (1 | h), crossed, two seeds and two shapes.
mk_phylo_h <- function(id, seed, ntip, nh, sigma_x = FALSE) {
  set.seed(seed)
  tree <- ape::rcoal(ntip, tip.label = paste0("t", seq_len(ntip)))
  C <- ape::vcv(tree, corr = TRUE)
  dat <- expand.grid(sp = tree$tip.label, h = paste0("h", seq_len(nh)),
                     stringsAsFactors = FALSE)
  dat <- dat[sample.int(nrow(dat)), ]
  a <- rmvn(C, 0.8); names(a) <- rownames(C)
  b <- stats::rnorm(nh, 0, 0.7); names(b) <- paste0("h", seq_len(nh))
  dat$x <- stats::rnorm(nrow(dat))
  sd_e <- if (sigma_x) exp(-0.5 + 0.3 * dat$x) else 0.6
  dat$y <- 1 + 0.5 * dat$x + a[dat$sp] + b[dat$h] + stats::rnorm(nrow(dat), 0, sd_e)
  sig <- if (sigma_x) "sigma ~ x" else "sigma ~ 1"
  sigf <- stats::as.formula(sig)
  list(id = id, data = dat, tree = tree, kind = "phylo",
       nf = drmTMB::bf(y ~ x + phylo(1 | sp, tree = tree) + (1 | h), sigf),
       julia = paste0("y ~ x + phylo(1 | sp) + (1 | h); ", sig))
}
fixtures$phylo_h_seed11 <- mk_phylo_h("phylo_h_seed11", 11, 20, 6)
fixtures$phylo_h_seed22 <- mk_phylo_h("phylo_h_seed22", 22, 30, 4)
fixtures$phylo_h_sigmax <- mk_phylo_h("phylo_h_sigmax", 33, 24, 5, sigma_x = TRUE)

# F3: (1 | sp) + phylo(1 | sp) on the SAME grouping, no sd() submodels.
local({
  set.seed(44)
  tree <- ape::rcoal(30, tip.label = paste0("t", 1:30))
  C <- ape::vcv(tree, corr = TRUE)
  dat <- data.frame(sp = rep(tree$tip.label, each = 6), stringsAsFactors = FALSE)
  a <- rmvn(C, 0.8); names(a) <- rownames(C)
  u <- stats::rnorm(30, 0, 0.5); names(u) <- tree$tip.label
  dat$x <- stats::rnorm(nrow(dat))
  dat$y <- 1 + 0.5 * dat$x + a[dat$sp] + u[dat$sp] + stats::rnorm(nrow(dat), 0, 0.5)
  fixtures$phylo_sp_same_group <<- list(
    id = "phylo_sp_same_group", data = dat, tree = tree, kind = "phylo",
    nf = drmTMB::bf(y ~ x + (1 | sp) + phylo(1 | sp, tree = tree), sigma ~ 1),
    julia = "y ~ x + (1 | sp) + phylo(1 | sp); sigma ~ 1")
})

# F4: phylo(1 | sp) + two ordinary bars (1 | h) + (1 | g).
local({
  set.seed(55)
  tree <- ape::rcoal(20, tip.label = paste0("t", 1:20))
  C <- ape::vcv(tree, corr = TRUE)
  dat <- expand.grid(sp = tree$tip.label, h = paste0("h", 1:6), stringsAsFactors = FALSE)
  dat$g <- paste0("g", sample.int(8, nrow(dat), replace = TRUE))
  a <- rmvn(C, 0.8); names(a) <- rownames(C)
  b <- stats::rnorm(6, 0, 0.7); names(b) <- paste0("h", 1:6)
  gg <- stats::rnorm(8, 0, 0.5); names(gg) <- paste0("g", 1:8)
  dat$x <- stats::rnorm(nrow(dat))
  dat$y <- 1 + 0.5 * dat$x + a[dat$sp] + b[dat$h] + gg[dat$g] + stats::rnorm(nrow(dat), 0, 0.6)
  fixtures$phylo_h_g <<- list(
    id = "phylo_h_g", data = dat, tree = tree, kind = "phylo",
    nf = drmTMB::bf(y ~ x + phylo(1 | sp, tree = tree) + (1 | h) + (1 | g), sigma ~ 1),
    julia = "y ~ x + phylo(1 | sp) + (1 | h) + (1 | g); sigma ~ 1")
})

# F5/F6: relmat(1 | id, K = K) + (1 | h) and animal(1 | id, A = A) + (1 | h).
mk_K_h <- function(id, seed, kind) {
  set.seed(seed)
  nid <- 25; nh <- 5
  lev <- paste0("i", seq_len(nid))
  K <- if (kind == "relmat") ar1(nid, 0.6) else {
    # an additive-relatedness-like matrix: 5 full-sib families, r = 0.5
    fam <- rep(1:5, each = 5); A <- outer(fam, fam, function(a, b) ifelse(a == b, 0.5, 0))
    diag(A) <- 1; A
  }
  dimnames(K) <- list(lev, lev)
  dat <- expand.grid(id = lev, h = paste0("h", seq_len(nh)), stringsAsFactors = FALSE)
  dat <- dat[sample.int(nrow(dat)), ]
  a <- rmvn(K, 0.8); names(a) <- lev
  b <- stats::rnorm(nh, 0, 0.7); names(b) <- paste0("h", seq_len(nh))
  dat$x <- stats::rnorm(nrow(dat))
  dat$y <- 1 + 0.5 * dat$x + a[dat$id] + b[dat$h] + stats::rnorm(nrow(dat), 0, 0.6)
  nf <- if (kind == "relmat") {
    drmTMB::bf(y ~ x + relmat(1 | id, K = K) + (1 | h), sigma ~ 1)
  } else {
    drmTMB::bf(y ~ x + animal(1 | id, A = K) + (1 | h), sigma ~ 1)
  }
  list(id = id, data = dat, K = K, kind = kind, nf = nf,
       kwarg = if (kind == "relmat") "K" else "A",
       julia = paste0("y ~ x + ", kind, "(1 | id) + (1 | h); sigma ~ 1"))
}
fixtures$relmat_h <- mk_K_h("relmat_h", 66, "relmat")
fixtures$animal_h <- mk_K_h("animal_h", 77, "animal")

# F7: spatial(1 | site, coords = co) + (1 | h). Native builds its fixed-range
# exponential K (drm_spatial_coords_precision); the bridge sends that same K to
# DRModels.jl as relmat(1 | site). Julia gets exactly that K here.
local({
  set.seed(88)
  ns <- 20; nh <- 6
  lev <- paste0("s", seq_len(ns))
  co <- data.frame(cx = stats::runif(ns), cy = stats::runif(ns), row.names = lev)
  dat <- expand.grid(site = lev, h = paste0("h", seq_len(nh)), stringsAsFactors = FALSE)
  dat <- dat[sample.int(nrow(dat)), ]
  prec <- drm_spatial_coords_precision(as.matrix(co), site = dat$site, group = "site")
  Ks <- solve(as.matrix(prec$precision)); dimnames(Ks) <- list(prec$site_levels, prec$site_levels)
  a <- rmvn(Ks, 0.8); names(a) <- rownames(Ks)
  b <- stats::rnorm(nh, 0, 0.7); names(b) <- paste0("h", seq_len(nh))
  dat$x <- stats::rnorm(nrow(dat))
  dat$y <- 1 + 0.5 * dat$x + a[dat$site] + b[dat$h] + stats::rnorm(nrow(dat), 0, 0.6)
  fixtures$spatial_h <<- list(
    id = "spatial_h", data = dat, K = Ks, kind = "spatial", kwarg = "K",
    nf = drmTMB::bf(y ~ x + spatial(1 | site, coords = co) + (1 | h), sigma ~ 1),
    julia = "y ~ x + relmat(1 | site) + (1 | h); sigma ~ 1")
})

# F8: phylo(1 | sp) + an independent ordinary random SLOPE (0 + x | h).
local({
  set.seed(99)
  tree <- ape::rcoal(20, tip.label = paste0("t", 1:20))
  C <- ape::vcv(tree, corr = TRUE)
  dat <- expand.grid(sp = tree$tip.label, h = paste0("h", 1:6), stringsAsFactors = FALSE)
  a <- rmvn(C, 0.8); names(a) <- rownames(C)
  b <- stats::rnorm(6, 0, 0.5); names(b) <- paste0("h", 1:6)
  dat$x <- stats::rnorm(nrow(dat))
  dat$y <- 1 + 0.5 * dat$x + a[dat$sp] + b[dat$h] * dat$x + stats::rnorm(nrow(dat), 0, 0.6)
  fixtures$phylo_slope_h <<- list(
    id = "phylo_slope_h", data = dat, tree = tree, kind = "phylo",
    nf = drmTMB::bf(y ~ x + phylo(1 | sp, tree = tree) + (0 + x | h), sigma ~ 1),
    julia = "y ~ x + phylo(1 | sp) + (0 + x | h); sigma ~ 1")
})

rows <- list()
add <- function(id, param, value) {
  rows[[length(rows) + 1L]] <<- data.frame(fixture = id, param = param,
                                           value = sprintf("%.10f", value))
}
for (fx in fixtures) {
  dat <- fx$data
  fit <- drmTMB(fx$nf, family = stats::gaussian(), data = dat, engine = "tmb")
  stopifnot(isTRUE(fit$opt$convergence == 0))
  add(fx$id, "df", attr(logLik(fit), "df"))
  add(fx$id, "logLik", as.numeric(logLik(fit)))
  for (dp in names(fit$coefficients)) {
    cf <- fit$coefficients[[dp]]
    for (nm in names(cf)) add(fx$id, paste0(dp, ":", nm), cf[[nm]])
  }
  sd <- fit$sdpars$mu
  for (nm in names(sd)) add(fx$id, paste0("sd:", nm), sd[[nm]])
  write.table(dat, file.path("fixtures", paste0(fx$id, ".data.tsv")),
              sep = "\t", quote = FALSE, row.names = FALSE)
  if (!is.null(fx$tree)) ape::write.tree(fx$tree, file.path("fixtures", paste0(fx$id, ".tree.nwk")))
  if (!is.null(fx$K)) {
    grp <- if (fx$kind == "spatial") "site" else "id"
    ord <- first_seen(dat[[grp]])
    K <- fx$K[ord, ord]
    write.table(cbind(level = ord, as.data.frame(K)),
                file.path("fixtures", paste0(fx$id, ".", fx$kwarg, ".tsv")),
                sep = "\t", quote = FALSE, row.names = FALSE)
  }
  cat(fx$id, "\t", fx$julia, "\n", file = file.path("fixtures", "formulas.tsv"),
      append = !identical(fx$id, fixtures[[1L]]$id), sep = "")
  message(fx$id, ": df ", attr(logLik(fit), "df"), " logLik ",
          format(as.numeric(logLik(fit)), digits = 12))
}
write.table(do.call(rbind, rows), "native.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
