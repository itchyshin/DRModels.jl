# Third set of R reference values for test/test_fitted_conditional.jl (P20, PR #1324): the two phylo groups.
#
#   DRMTMB_LIB=~/drm_campaign_20261108/Rlib Rscript p17_fitted_phylo_r.R data.csv tree16.newick p17_fitted_phylo_reference.csv
#
# Group 1: a Gaussian mean with ONE structured intercept: relmat(1 | g, K), animal(1 | g, A) and
#   phylo(1 | g, tree), constant sigma. Group 2: phylo(1 | g, tree) for Poisson, NB2, Binomial,
#   BetaBinomial, Gamma and Beta (Laplace in both engines). Each cell is also fitted with a missing
#   response on the rows where `y_gm` is NA (16 rows): the used rows only come back.
# Output, per used row: the drmTMB `fitted()` value (`cond`, modes included), `pop` (the value with
# every random effect at zero) and, for the Gaussian cells, nothing else. `logLik` rows (row 0)
# give each fit's log-likelihood and, in `pop`, the first fixed coefficient, to tell a difference
# between two fits from a difference between two sets of modes.
# The two relatedness matrices are written next to the output (k_relmat.csv, a_animal.csv); their row
# order is the first-seen order of `g` (g01 ... g16), which is also the factor-level order.
args <- commandArgs(TRUE)
.libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages({library(drmTMB); library(ape)})
d <- read.csv(args[1], stringsAsFactors = FALSE, na.strings = "NA")
d$g <- factor(d$g)
tree <- read.tree(args[2])
lev <- levels(d$g)
G <- length(lev)
K <- 0.4^abs(outer(seq_len(G), seq_len(G), "-")) + 0.6 * diag(G);  dimnames(K) <- list(lev, lev)
A <- 0.7^abs(outer(seq_len(G), seq_len(G), "-")) + 0.3 * diag(G);  dimnames(A) <- list(lev, lev)
outdir <- dirname(args[3])
write.table(K, file.path(outdir, "k_relmat.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(A, file.path(outdir, "a_animal.csv"), sep = ",", row.names = FALSE, col.names = FALSE)

na <- is.na(d$y_gm)
for (v in c("y_g", "y_pois", "y_nb", "y_gam", "y_beta", "s")) d[[paste0(v, "_m")]] <- ifelse(na, NA, d[[v]])
d$fl_m <- ifelse(na, NA, d$fl)

ph <- "phylo(1 | g, tree = tree)"
cell <- function(id, call) list(id, function(d) eval(parse(text = call)))
# drmTMB needs literal formulas in bf(), so each cell is a call written out in full.
cells <- list(
  cell("G1|relmat",       "drmTMB(bf(y_g ~ x1 + relmat(1 | g, K = K), sigma ~ 1), family = gaussian(), data = d)"),
  cell("G1|animal",       "drmTMB(bf(y_g ~ x1 + animal(1 | g, A = A), sigma ~ 1), family = gaussian(), data = d)"),
  cell("G1|phylo-dense",  "drmTMB(bf(y_g ~ x1 + phylo(1 | g, tree = tree), sigma ~ 1), family = gaussian(), data = d)"),
  cell("G1M|relmat",      "drmTMB(bf(y_g_m ~ x1 + relmat(1 | g, K = K), sigma ~ 1), family = gaussian(), data = d)"),
  cell("G1M|animal",      "drmTMB(bf(y_g_m ~ x1 + animal(1 | g, A = A), sigma ~ 1), family = gaussian(), data = d)"),
  cell("G1M|phylo-dense", "drmTMB(bf(y_g_m ~ x1 + phylo(1 | g, tree = tree), sigma ~ 1), family = gaussian(), data = d)"),
  cell("G2|poisson",      "drmTMB(bf(y_pois ~ x1 + phylo(1 | g, tree = tree)), family = stats::poisson(link = 'log'), data = d)"),
  cell("G2|nb2",          "drmTMB(bf(y_nb ~ x1 + phylo(1 | g, tree = tree), sigma ~ 1), family = nbinom2(), data = d)"),
  cell("G2|binomial",     "drmTMB(bf(cbind(s, fl) ~ x1 + phylo(1 | g, tree = tree)), family = binomial(), data = d)"),
  cell("G2|betabinomial", "drmTMB(bf(cbind(s, fl) ~ x1 + phylo(1 | g, tree = tree), sigma ~ 1), family = beta_binomial(), data = d)"),
  cell("G2|gamma",        "drmTMB(bf(y_gam ~ x1 + phylo(1 | g, tree = tree), sigma ~ 1), family = stats::Gamma(link = 'log'), data = d)"),
  cell("G2|beta",         "drmTMB(bf(y_beta ~ x1 + phylo(1 | g, tree = tree), sigma ~ 1), family = beta_family(), data = d)"),
  cell("G2M|poisson",     "drmTMB(bf(y_pois_m ~ x1 + phylo(1 | g, tree = tree)), family = stats::poisson(link = 'log'), data = d)"),
  cell("G2M|nb2",         "drmTMB(bf(y_nb_m ~ x1 + phylo(1 | g, tree = tree), sigma ~ 1), family = nbinom2(), data = d)"),
  cell("G2M|binomial",    "drmTMB(bf(cbind(s_m, fl_m) ~ x1 + phylo(1 | g, tree = tree)), family = binomial(), data = d)"),
  cell("G2M|betabinomial","drmTMB(bf(cbind(s_m, fl_m) ~ x1 + phylo(1 | g, tree = tree), sigma ~ 1), family = beta_binomial(), data = d)"),
  cell("G2M|gamma",       "drmTMB(bf(y_gam_m ~ x1 + phylo(1 | g, tree = tree), sigma ~ 1), family = stats::Gamma(link = 'log'), data = d)"),
  cell("G2M|beta",        "drmTMB(bf(y_beta_m ~ x1 + phylo(1 | g, tree = tree), sigma ~ 1), family = beta_family(), data = d)")
)
out <- list()
for (cl in cells) {
  fit <- tryCatch(suppressWarnings(cl[[2]](d)), error = function(e) { cat("FAIL", cl[[1]], conditionMessage(e), "\n"); NULL })
  if (is.null(fit)) next
  cond <- as.numeric(fitted(fit))
  pop <- tryCatch(as.numeric(predict(fit, newdata = d, dpar = "mu", type = "response")), error = function(e) rep(NA_real_, nrow(d)))
  keep <- if (grepl("M\\|", cl[[1]])) !na else rep(TRUE, nrow(d))
  if (length(pop) == nrow(d)) pop <- pop[keep] else pop <- rep(NA_real_, length(cond))
  ll <- as.numeric(logLik(fit))
  b1 <- unname(coef(fit, dpar = "mu")[1])
  out[[cl[[1]]]] <- data.frame(cell = cl[[1]], row = c(0, seq_along(cond)), cond = c(ll, cond), pop = c(b1, pop))
  cat("OK", cl[[1]], length(cond), ll, "\n")
}
write.csv(do.call(rbind, out), args[3], row.names = FALSE)
