# Arc 2 receipt: native drmTMB (engine = "tmb") reference fits for Gaussian
# meta-analysis with known sampling variances PLUS a random effect on the mean.
#
# Writes, next to this script:
#   fixtures/<fixture>.tsv    the data each fit used (read by julia-fit.jl)
#   fixtures/<fixture>.nwk    the tree, for the phylo fixtures
#   fixtures/<fixture>-K.tsv  the relatedness matrix, for the relmat fixture
#   native.tsv                one row per (fixture, quantity): df, logLik, estimates
#
# Run from the DRModels.jl repository root:
#   DRMTMB_PATH=/path/to/drmTMB Rscript docs/dev-log/evidence/arc2-metav-random-effect/native-fit.R
# The drmTMB tree must have its TMB DLL built (pkgload::load_all compiles it).
suppressMessages({
  pkgload::load_all(Sys.getenv("DRMTMB_PATH", "~/local-scratch/lanes/drmTMB-arc1-pr1304-fold"),
                    quiet = TRUE)
  library(ape)
})
here <- "docs/dev-log/evidence/arc2-metav-random-effect"
dir.create(file.path(here, "fixtures"), showWarnings = FALSE)

make_study <- function(seed, S, m, tau, sd_study, sigma_slope = 0) {
  set.seed(seed)
  n <- S * m
  d <- data.frame(study = sprintf("s%02d", rep(seq_len(S), each = m)),
                  x = round(rnorm(n), 6), v = round(runif(n, 0.05, 0.3), 6))
  u <- rnorm(S, 0, sd_study)
  e_sd <- tau * exp(sigma_slope * d$x)
  d$y <- round(0.3 + 0.5 * d$x + u[match(d$study, unique(d$study))] +
                 rnorm(n, 0, e_sd) + rnorm(n, 0, sqrt(d$v)), 6)
  d
}

make_phylo <- function(seed, ntip, m, tau, sd_phylo, height = NULL, file) {
  set.seed(seed)
  tree <- ape::rcoal(ntip)
  if (!is.null(height)) {
    tree$edge.length <- tree$edge.length * height / max(ape::node.depth.edgelength(tree))
  }
  # Round-trip through the Newick file the Julia side reads, so both engines
  # see byte-identical branch lengths.
  ape::write.tree(tree, file)
  tree <- ape::read.tree(file)
  C <- ape::vcv(tree, corr = TRUE)
  a <- as.vector(t(chol(C)) %*% rnorm(ntip, 0, sd_phylo))
  names(a) <- rownames(C)
  n <- ntip * m
  d <- data.frame(sp = rep(tree$tip.label, each = m),
                  x = round(rnorm(n), 6), v = round(runif(n, 0.05, 0.3), 6))
  d$y <- round(0.3 + 0.5 * d$x + a[d$sp] + rnorm(n, 0, tau) +
                 rnorm(n, 0, sqrt(d$v)), 6)
  list(data = d, tree = tree)
}

make_relmat <- function(seed, G, m, tau, sd_rel) {
  set.seed(seed)
  # A dense, well-conditioned relatedness matrix: the tip correlation of a
  # random coalescent tree, relabelled as individuals.
  tr <- ape::rcoal(G)
  K <- ape::vcv(tr, corr = TRUE)
  ids <- sprintf("i%02d", seq_len(G))
  dimnames(K) <- list(ids, ids)
  a <- as.vector(t(chol(K)) %*% rnorm(G, 0, sd_rel))
  n <- G * m
  d <- data.frame(id = rep(ids, each = m),
                  x = round(rnorm(n), 6), v = round(runif(n, 0.05, 0.3), 6))
  d$y <- round(0.3 + 0.5 * d$x + a[match(d$id, ids)] + rnorm(n, 0, tau) +
                 rnorm(n, 0, sqrt(d$v)), 6)
  list(data = d, K = round(K, 10))
}

rows <- list()
record <- function(fixture, formula_txt, fit) {
  cf <- coef(fit)
  est <- c(
    setNames(cf$mu, paste0("mu:", names(cf$mu))),
    setNames(cf$sigma, paste0("sigma:", names(cf$sigma))),
    unlist(lapply(names(fit$sdpars), function(p) {
      s <- fit$sdpars[[p]]
      setNames(unname(s), paste0("sd:", names(s)))
    }))
  )
  ll <- logLik(fit)
  rows[[length(rows) + 1L]] <<- data.frame(
    fixture = fixture, formula = formula_txt, engine = "tmb",
    df = attr(ll, "df"), logLik = sprintf("%.9f", as.numeric(ll)),
    converged = isTRUE(fit$opt$convergence == 0),
    quantity = names(est), estimate = sprintf("%.9f", unname(est)),
    stringsAsFactors = FALSE
  )
}
fit_tmb <- function(f, data) drmTMB(f, family = gaussian(), data = data, engine = "tmb")

# F1, F2: meta_V + (1 | study), two seeds and two shapes (F2 adds sigma ~ x).
d1 <- make_study(20260924, S = 36, m = 5, tau = 0.3, sd_study = 0.6)
write.table(d1, file.path(here, "fixtures/study-a.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
record("study-a", "y ~ x + meta_V(V = v) + (1 | study)",
       fit_tmb(bf(y ~ x + meta_V(V = v) + (1 | study)), d1))

d2 <- make_study(7, S = 24, m = 6, tau = 0.35, sd_study = 0.45, sigma_slope = 0.3)
write.table(d2, file.path(here, "fixtures/study-b.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
record("study-b", "y ~ x + meta_V(V = v) + (1 | study)",
       fit_tmb(bf(y ~ x + meta_V(V = v) + (1 | study)), d2))
record("study-b", "y ~ x + meta_V(V = v) + (1 | study); sigma ~ x",
       fit_tmb(bf(y ~ x + meta_V(V = v) + (1 | study), sigma ~ x), d2))

# F3, F4: meta_V + phylo(1 | sp) on two coalescent trees, neither of unit
# height (native requires an ultrametric tree and uses its tip correlation,
# ape::vcv(corr = TRUE), so the answer must not depend on the height).
p3 <- make_phylo(11, ntip = 30, m = 5, tau = 0.3, sd_phylo = 0.7,
                 file = file.path(here, "fixtures/phylo-a.nwk"))
write.table(p3$data, file.path(here, "fixtures/phylo-a.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
tree <- p3$tree
record("phylo-a", "y ~ x + meta_V(V = v) + phylo(1 | sp)",
       fit_tmb(bf(y ~ x + meta_V(V = v) + phylo(1 | sp, tree = tree)), p3$data))

p4 <- make_phylo(29, ntip = 24, m = 6, tau = 0.25, sd_phylo = 0.6, height = 3,
                 file = file.path(here, "fixtures/phylo-b.nwk"))
write.table(p4$data, file.path(here, "fixtures/phylo-b.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
tree <- p4$tree
record("phylo-b", "y ~ x + meta_V(V = v) + phylo(1 | sp)",
       fit_tmb(bf(y ~ x + meta_V(V = v) + phylo(1 | sp, tree = tree)), p4$data))

# F5: meta_V + relmat(1 | id).
r5 <- make_relmat(41, G = 25, m = 6, tau = 0.3, sd_rel = 0.6)
write.table(r5$data, file.path(here, "fixtures/relmat-a.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(r5$K, file.path(here, "fixtures/relmat-a-K.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE)
K <- r5$K
record("relmat-a", "y ~ x + meta_V(V = v) + relmat(1 | id)",
       fit_tmb(bf(y ~ x + meta_V(V = v) + relmat(1 | id, K = K)), r5$data))

# F6: meta_V + phylo(1 | sp) + (1 | study): both components in one fit.
p6 <- p3$data
p6$study <- sprintf("s%02d", rep(seq_len(25), length.out = nrow(p6)))
set.seed(5)
us <- rnorm(25, 0, 0.4)
p6$y <- round(p6$y + us[match(p6$study, sprintf("s%02d", 1:25))], 6)
write.table(p6, file.path(here, "fixtures/phylo-study.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
tree <- p3$tree
record("phylo-study", "y ~ x + meta_V(V = v) + phylo(1 | sp) + (1 | study)",
       fit_tmb(bf(y ~ x + meta_V(V = v) + phylo(1 | sp, tree = tree) + (1 | study)), p6))

out <- do.call(rbind, rows)
write.table(out, file.path(here, "native.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
cat("drmTMB commit:", tryCatch(system2("git", c("-C", Sys.getenv("DRMTMB_PATH",
    path.expand("~/local-scratch/lanes/drmTMB-arc1-pr1304-fold")), "rev-parse", "HEAD"),
    stdout = TRUE), error = function(e) NA), "\n")
print(out[, c("fixture", "formula", "df", "logLik", "quantity", "estimate")], row.names = FALSE)
