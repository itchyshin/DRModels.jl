# Arc 2 receipt, speed fixture H2: native drmTMB (engine = "tmb") ML/REML target
# values for a coupled mu+sigma phylo fixture with a strong NEGATIVE phylo
# correlation (true -0.95), unequal species sizes (1-12 rows, 3 singleton
# species) and a tree of height 3.58 (not unit). A reviewer measured Julia's
# coupled REML at about 29 minutes here (native: under 1 second); the cause was
# the inner-mode re-solve, not the target. Writes fixture-H2 (data CSV + Newick
# tree) and native-speed.tsv, which test/test_reml_sigma_phylo_joint.jl reads.
# The generator is the reviewer's (seed 8675309, 41 tips).
#
# Run (from anywhere):
#   DRMTMB_PATH=~/local-scratch/lanes/drmTMB-arc1-pr1304-fold \
#     Rscript --no-init-file native-fit-speed.R
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

set.seed(8675309L)
ntip <- 41L
tree <- ape::rcoal(ntip)
tree$edge.length <- tree$edge.length * 2.3
tree$tip.label <- paste0("t", seq_len(ntip))
C <- ape::vcv(tree)[tree$tip.label, tree$tip.label]
L <- t(chol(C))
z1 <- as.vector(L %*% stats::rnorm(ntip))
z2 <- as.vector(L %*% stats::rnorm(ntip))
m <- sample(1:12, ntip, replace = TRUE)
sp <- factor(rep(tree$tip.label, times = m), levels = tree$tip.label)
n <- length(sp)
x <- stats::rnorm(n)
z <- stats::runif(n, -1, 1)
j <- as.integer(sp)
rho <- -0.95
a_mu <- 0.5 * z1
a_sig <- 0.45 * (rho * z1 + sqrt(1 - rho^2) * z2)
y <- 0.3 + 0.6 * x + a_mu[j] + exp(-0.1 + 0.4 * z + a_sig[j]) * stats::rnorm(n)
d <- data.frame(y = y, x = x, z = z, sp = sp)
utils::write.csv(d, file.path(here, "fixture-H2.csv"), row.names = FALSE)
ape::write.tree(tree, file.path(here, "fixture-H2.nwk"), digits = 17)

getv <- function(v, nm) if (nm %in% names(v)) unname(v[[nm]]) else NA_real_
rows <- list()
for (reml in c(FALSE, TRUE)) {
  t0 <- proc.time()[["elapsed"]]
  fit <- drmTMB(bf(y ~ x + phylo(1 | sp, tree = tree), sigma ~ z + phylo(1 | sp, tree = tree)),
                family = gaussian(), data = d, engine = "tmb", REML = reml)
  el <- proc.time()[["elapsed"]] - t0
  ll <- logLik(fit)
  v <- unlist(c(coef(fit), fit$sdpars, fit$corpars))
  nm <- names(v)
  rows[[length(rows) + 1L]] <- data.frame(
    fixture = "H2", shape = "mu_sigma", estimator = if (reml) "REML" else "ML",
    engine = "tmb", n = nrow(d), df = attr(ll, "df"),
    logLik = sprintf("%.8f", as.numeric(ll)),
    mu_intercept = sprintf("%.8f", getv(v, "mu.(Intercept)")),
    mu_x = sprintf("%.8f", getv(v, "mu.x")),
    sigma_intercept = sprintf("%.8f", getv(v, "sigma.(Intercept)")),
    sigma_z = sprintf("%.8f", getv(v, "sigma.z")),
    sd_mu = sprintf("%.8f", v[grepl("^mu\\..*phylo", nm)][[1]]),
    sd_sigma = sprintf("%.8f", v[grepl("^sigma\\..*phylo", nm)][[1]]),
    cor = sprintf("%.8f", v[grepl("^phylo\\.cor", nm)][[1]]),
    converged = isTRUE(fit$opt$convergence == 0),
    elapsed_s = sprintf("%.2f", el),
    drmtmb_sha = drmtmb_sha,
    stringsAsFactors = FALSE
  )
}
res <- do.call(rbind, rows)
utils::write.table(res, file.path(here, "native-speed.tsv"), sep = "\t", quote = FALSE,
                   row.names = FALSE, na = "NA")
print(res)
