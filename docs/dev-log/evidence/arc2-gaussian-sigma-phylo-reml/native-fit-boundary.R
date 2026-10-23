# Arc 2 receipt, correlation-boundary fixtures: native drmTMB (engine = "tmb")
# ML/REML target values for two coupled mu+sigma phylo fixtures whose native
# optimum sits on the phylo-correlation bound, rho = 0.999999 * tanh(eta_cor_phylo)
# (drmTMB.cpp). A reviewer found Julia's coupled REML stopping at a worse local
# optimum on both. Writes fixture-G1/G2 (data CSV + Newick tree) and
# native-boundary.tsv, which test/test_reml_sigma_phylo_joint.jl reads.
#
# Run (from anywhere):
#   DRMTMB_PATH=~/local-scratch/lanes/drmTMB-arc1-pr1304-fold \
#     Rscript --no-init-file native-fit-boundary.R
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

# Unequal species sizes (2-9 rows), correlated mean and log-sigma phylo effects.
# G1: unit-height tree, sigma ~ 1. G2: tree height 7.22 (not unit), sigma ~ z.
fixture_g <- function(seed, ntip, unit, rho, sdm, sds) {
  set.seed(seed)
  tree <- ape::rcoal(ntip)
  if (unit) tree$edge.length <- tree$edge.length / max(ape::node.depth.edgelength(tree))
  else tree$edge.length <- tree$edge.length * 1.7
  tree$tip.label <- paste0("q", seq_len(ntip))
  C <- ape::vcv(tree)[tree$tip.label, tree$tip.label]
  L <- t(chol(C))
  z1 <- as.vector(L %*% stats::rnorm(ntip))
  z2 <- as.vector(L %*% stats::rnorm(ntip))
  m <- sample(2:9, ntip, replace = TRUE)
  sp <- factor(rep(tree$tip.label, times = m), levels = tree$tip.label)
  n <- length(sp)
  x <- stats::rnorm(n)
  z <- stats::runif(n, -1, 1)
  j <- as.integer(sp)
  a_mu <- sdm * z1
  a_sig <- sds * (rho * z1 + sqrt(1 - rho^2) * z2)
  y <- -0.5 + 0.7 * x + a_mu[j] + exp(0.2 + 0.35 * z + a_sig[j]) * stats::rnorm(n)
  list(data = data.frame(y = y, x = x, z = z, sp = sp), tree = tree)
}

fixtures <- list(G1 = fixture_g(90210L, 37L, TRUE, 0.5, 0.5, 0.45),
                 G2 = fixture_g(31337L, 44L, FALSE, -0.6, 0.4, 0.4))
# Literal formulas: drmTMB's bf() takes formulas, not built expressions.
shapes <- list(
  G1 = list(
    mu_only = function(tree) bf(y ~ x + phylo(1 | sp, tree = tree), sigma ~ 1),
    sigma_only = function(tree) bf(y ~ x, sigma ~ 1 + phylo(1 | sp, tree = tree)),
    mu_sigma = function(tree) bf(y ~ x + phylo(1 | sp, tree = tree), sigma ~ 1 + phylo(1 | sp, tree = tree))),
  G2 = list(
    mu_only = function(tree) bf(y ~ x + phylo(1 | sp, tree = tree), sigma ~ z),
    sigma_only = function(tree) bf(y ~ x, sigma ~ z + phylo(1 | sp, tree = tree)),
    mu_sigma = function(tree) bf(y ~ x + phylo(1 | sp, tree = tree), sigma ~ z + phylo(1 | sp, tree = tree)))
)

getv <- function(v, nm) if (nm %in% names(v)) unname(v[[nm]]) else NA_real_
rows <- list()
for (fx in names(fixtures)) {
  d <- fixtures[[fx]]$data
  tree <- fixtures[[fx]]$tree
  utils::write.csv(d, file.path(here, sprintf("fixture-%s.csv", fx)), row.names = FALSE)
  ape::write.tree(tree, file.path(here, sprintf("fixture-%s.nwk", fx)), digits = 17)
  for (shape in names(shapes[[fx]])) for (reml in c(FALSE, TRUE)) {
    fit <- drmTMB(shapes[[fx]][[shape]](tree), family = gaussian(), data = d,
                  engine = "tmb", REML = reml)
    ll <- logLik(fit)
    v <- unlist(c(coef(fit), fit$sdpars, fit$corpars))
    nm <- names(v)
    sd_mu <- v[grepl("^mu\\..*phylo", nm)]
    sd_sig <- v[grepl("^sigma\\..*phylo", nm)]
    cor <- v[grepl("^phylo\\.cor", nm)]
    rows[[length(rows) + 1L]] <- data.frame(
      fixture = fx, shape = shape, estimator = if (reml) "REML" else "ML",
      engine = "tmb", n = nrow(d), df = attr(ll, "df"),
      logLik = sprintf("%.8f", as.numeric(ll)),
      mu_intercept = sprintf("%.8f", getv(v, "mu.(Intercept)")),
      mu_x = sprintf("%.8f", getv(v, "mu.x")),
      sigma_intercept = sprintf("%.8f", getv(v, "sigma.(Intercept)")),
      sigma_z = sprintf("%.8f", getv(v, "sigma.z")),
      sd_mu = sprintf("%.8f", if (length(sd_mu)) sd_mu[[1]] else NA_real_),
      sd_sigma = sprintf("%.8f", if (length(sd_sig)) sd_sig[[1]] else NA_real_),
      cor = sprintf("%.8f", if (length(cor)) cor[[1]] else NA_real_),
      converged = isTRUE(fit$opt$convergence == 0),
      drmtmb_sha = drmtmb_sha,
      stringsAsFactors = FALSE
    )
  }
}
res <- do.call(rbind, rows)
utils::write.table(res, file.path(here, "native-boundary.tsv"), sep = "\t", quote = FALSE,
                   row.names = FALSE, na = "NA")
print(res[, c("fixture", "shape", "estimator", "df", "logLik", "sd_mu", "sd_sigma", "cor", "converged")])
