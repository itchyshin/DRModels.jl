# gen_data.R -- R oracle for the cumulative_logit() mu phylogenetic-intercept
# parity fixture (#563 S8 follow-on). Run with drmTMB 0.7.0:
#
#   Rscript test/parity/fixtures/cumlogit-mu-phylo/gen_data.R
#
# DGP: a 60-tip coalescent tree (ape::rcoal), a genuinely tree-correlated
# Brownian random intercept on the ordinal latent mean
# (u = chol(vcv(tree, corr = TRUE)) %*% rnorm(60, 0, sd_phylo_true)) -- NOT iid
# noise indexed by species (that pitfall was diagnosed in
# test/parity/gen_gaussian_phylo_mean.R's #483 note) -- plus n_each = 5
# replicate observations per tip (n = 300) so the phylo variance component has
# real signal to recover. Uses drmTMB's own `validate_ordinal_phylo_mu_structured_term()`
# route (R/drmTMB.R:10500), the same route exercised by
# tests/testthat/test-cumulative-logit.R's "cumulative-logit admits the first
# phylogenetic mu intercept gate" test -- this fixture reuses that test's own
# `control = drm_control(se = FALSE)` choice (drmTMB's own gate test disables
# SE for this combination too).

suppressPackageStartupMessages({
  library(drmTMB)
  library(ape)
})

set.seed(20260902)
n_tip <- 60L; n_each <- 5L
tree <- ape::rcoal(n_tip)
tree$tip.label <- paste0("sp", seq_len(n_tip))
A <- ape::vcv(tree, corr = TRUE)
L <- t(chol(A))
sd_phylo_true <- 0.9
u <- as.numeric(L %*% stats::rnorm(n_tip, 0, sd_phylo_true))

species <- factor(rep(tree$tip.label, each = n_each), levels = tree$tip.label)
n <- length(species)
x <- stats::rnorm(n)
beta_true <- 0.6
cut_true <- c(-0.5, 0.6)
eta <- beta_true * x + u[as.integer(species)]
lat <- eta + stats::rlogis(n)
y_int <- findInterval(lat, cut_true) + 1L
dat <- data.frame(species = species, x = x, y_int = y_int)

fit <- drmTMB(
  bf(y_int ~ x + phylo(1 | species, tree = tree)),
  family = cumulative_logit(),
  data = within(dat, y_int <- ordered(y_int, levels = 1:3)),
  control = drm_control(se = FALSE)
)

cat("convergence:", fit$opt$convergence, "\n")
cat("coef mu (slope):", coef(fit, "mu"), "\n")
cat("cutpoints:", unname(fit$ordinal$cutpoints), "\n")
cat("sdpars$mu names:", paste(names(fit$sdpars$mu), collapse = " | "), "\n")
phylo_name <- grep("^phylo\\(", names(fit$sdpars$mu), value = TRUE)
cat("phylo term name:", phylo_name, "\n")
cat("sd_phylo estimate:", unname(fit$sdpars$mu[[phylo_name]]), "\n")
cat("loglik:", as.numeric(logLik(fit)), "\n")
tree_height <- max(ape::node.depth.edgelength(tree)[seq_along(tree$tip.label)])
cat("tree_height:", tree_height, "\n")

# Refusals this slice keeps (mirrors drmTMB's own validator + DRModels.jl's iid
# random-effect slice's own refusal cells): a slope structured term, and a
# labelled marker.
res_slope <- tryCatch(
  drmTMB(bf(y_int ~ x + phylo(1 + x | species, tree = tree)),
         family = cumulative_logit(), data = within(dat, y_int <- ordered(y_int, levels = 1:3))),
  error = function(e) conditionMessage(e)
)
cat("phylo-slope-fit error:", res_slope, "\n")

res_label <- tryCatch(
  drmTMB(bf(y_int ~ x + phylo(1 | q | species, tree = tree)),
         family = cumulative_logit(), data = within(dat, y_int <- ordered(y_int, levels = 1:3))),
  error = function(e) conditionMessage(e)
)
cat("labelled-phylo-fit error:", res_label, "\n")

write.csv(dat, "test/parity/fixtures/cumlogit-mu-phylo/data.csv", row.names = FALSE)
ape::write.tree(tree, "test/parity/fixtures/cumlogit-mu-phylo/tree.newick")
cat("wrote fixture\n")
