# Arc 2 receipt: native drmTMB (engine = "tmb") REML/ML target values for the
# Gaussian location-scale model with a phylo() random intercept on sigma
# (sigma-only) and on both mu and sigma (coupled, with the mean-scale phylo
# correlation), plus the mean-only phylo neighbour. Writes the fixtures (data CSV + unit-height Newick tree) that
# julia-fit.jl reads, and native.tsv.
#
# Run (from anywhere):
#   DRMTMB_PATH=~/local-scratch/lanes/drmTMB-arc1-pr1304-fold \
#     Rscript --no-init-file native-fit.R
# drmTMB pin: see the `drmtmb_sha` column of native.tsv.
here <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f)) dirname(normalizePath(f)) else getwd()
})
drmtmb_path <- Sys.getenv("DRMTMB_PATH", "~/local-scratch/lanes/drmTMB-arc1-pr1304-fold")
suppressMessages(pkgload::load_all(drmtmb_path, quiet = TRUE))
drmtmb_sha <- tryCatch(
  system2("git", c("-C", drmtmb_path, "rev-parse", "--short=9", "HEAD"), stdout = TRUE),
  error = function(e) NA_character_
)

unit_height <- function(tree) {
  h <- max(ape::node.depth.edgelength(tree))
  tree$edge.length <- tree$edge.length / h
  tree
}

# F1: the Arc 1 probe fixture, sweep_fixture("gaussian") in drmTMB
# tools/julia-model-identity-sweep.R (seed 20260924, n = 90, 15 tips), copied
# here so the receipt is self-contained. Only y, x and sp are used.
fixture_f1 <- function(n = 90L, ntip = 15L, seed = 20260924L) {
  set.seed(seed)
  tree <- ape::rcoal(ntip)
  tree$tip.label <- paste0("t", seq_len(ntip))
  grp <- factor(rep(tree$tip.label, each = n %/% ntip), levels = tree$tip.label)
  x <- stats::rnorm(n)
  group_effect <- stats::rnorm(nlevels(grp), 0, 0.4)[as.integer(grp)]
  y <- 0.3 + 0.5 * x + group_effect + stats::rnorm(n, 0, 1)
  list(data = data.frame(y = y, x = x, sp = grp), tree = unit_height(tree))
}

# F2: a genuine phylogenetic location-scale draw (50 tips x 6 rows) with
# correlated mean and log-sigma phylogenetic effects (true SDs 0.6 / 0.5,
# correlation -0.4). Seed 11 was chosen (among 4242, 11..14) because every
# native fit, ML and REML, converges with an interior correlation.
fixture_f2 <- function(ntip = 50L, m = 6L, seed = 11L) {
  set.seed(seed)
  tree <- unit_height(ape::rcoal(ntip))
  tree$tip.label <- paste0("s", seq_len(ntip))
  C <- ape::vcv(tree)[tree$tip.label, tree$tip.label]
  L <- t(chol(C))
  z1 <- as.vector(L %*% stats::rnorm(ntip))
  z2 <- as.vector(L %*% stats::rnorm(ntip))
  rho <- -0.4
  a_mu <- 0.6 * z1
  a_sig <- 0.5 * (rho * z1 + sqrt(1 - rho^2) * z2)
  sp <- factor(rep(tree$tip.label, each = m), levels = tree$tip.label)
  n <- ntip * m
  x <- stats::rnorm(n)
  j <- as.integer(sp)
  y <- 1 + 0.4 * x + a_mu[j] + exp(-0.3 + a_sig[j]) * stats::rnorm(n)
  list(data = data.frame(y = y, x = x, sp = sp), tree = tree)
}

fixtures <- list(F1 = fixture_f1(), F2 = fixture_f2())
shapes <- list(
  # Neighbour guard (D-273): the mean-only phylo REML cell, a different DRModels
  # route (the sparse location-only spine) that must not move.
  mu_only = function(tree) bf(y ~ x + phylo(1 | sp, tree = tree), sigma ~ 1),
  sigma_only = function(tree) bf(y ~ x, sigma ~ 1 + phylo(1 | sp, tree = tree)),
  mu_sigma = function(tree) bf(y ~ x + phylo(1 | sp, tree = tree), sigma ~ 1 + phylo(1 | sp, tree = tree))
)

getv <- function(v, nm) if (nm %in% names(v)) unname(v[[nm]]) else NA_real_
rows <- list()
for (fx in names(fixtures)) {
  d <- fixtures[[fx]]$data
  tree <- fixtures[[fx]]$tree
  utils::write.csv(d, file.path(here, sprintf("fixture-%s.csv", fx)), row.names = FALSE)
  ape::write.tree(tree, file.path(here, sprintf("fixture-%s.nwk", fx)), digits = 17)
  for (shape in names(shapes)) for (reml in c(FALSE, TRUE)) {
    fit <- drmTMB(shapes[[shape]](tree), family = gaussian(), data = d, engine = "tmb", REML = reml)
    ll <- logLik(fit)
    v <- unlist(c(coef(fit), fit$sdpars, fit$corpars))
    nm <- names(v)
    sd_mu <- v[grepl("^mu\\..*phylo", nm)]
    sd_sig <- v[grepl("^sigma\\..*phylo", nm)]
    cor <- v[grepl("^phylo\\.cor", nm)]
    # NA when sdreport has no PD Hessian (the F1 mu_sigma ML fit at cor = -1).
    se <- tryCatch(sqrt(diag(vcov(fit))), error = function(e) c(x = NA_real_))
    getse <- function(nm) if (nm %in% names(se)) unname(se[[nm]]) else NA_real_
    rows[[length(rows) + 1L]] <- data.frame(
      fixture = fx, shape = shape, estimator = if (reml) "REML" else "ML",
      engine = "tmb", n = nrow(d), df = attr(ll, "df"),
      logLik = sprintf("%.8f", as.numeric(ll)),
      mu_intercept = sprintf("%.8f", getv(v, "mu.(Intercept)")),
      mu_x = sprintf("%.8f", getv(v, "mu.x")),
      sigma_intercept = sprintf("%.8f", getv(v, "sigma.(Intercept)")),
      sd_mu = sprintf("%.8f", if (length(sd_mu)) sd_mu[[1]] else NA_real_),
      sd_sigma = sprintf("%.8f", if (length(sd_sig)) sd_sig[[1]] else NA_real_),
      cor = sprintf("%.8f", if (length(cor)) cor[[1]] else NA_real_),
      se_mu_intercept = sprintf("%.8f", getse("mu:(Intercept)")),
      se_mu_x = sprintf("%.8f", getse("mu:x")),
      se_sigma_intercept = sprintf("%.8f", getse("sigma:(Intercept)")),
      converged = isTRUE(fit$opt$convergence == 0),
      drmtmb_sha = drmtmb_sha,
      stringsAsFactors = FALSE
    )
    if (reml) {
      # What native integrates out under REML (the TMB random vector).
      message(fx, " ", shape, " REML: TMB random = ",
              paste(unique(names(fit$obj$env$par[fit$obj$env$random])), collapse = ","))
    }
  }
}
res <- do.call(rbind, rows)
utils::write.table(res, file.path(here, "native.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
print(res[, c("fixture", "shape", "estimator", "df", "logLik", "sigma_intercept", "sd_mu", "sd_sigma", "cor", "converged")])
