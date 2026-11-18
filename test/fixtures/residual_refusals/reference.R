# drmTMB 0.8.1 reference for the residual bare structured cells of the wave-10 gap map.
# Usage: Rscript reference.R <outfile.tsv>   (run in this directory; DRMTMB_LIB = library holding drmTMB 0.8.1)
args <- commandArgs(TRUE); out <- args[1]
if (nzchar(Sys.getenv("DRMTMB_LIB"))) .libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages({library(drmTMB); library(ape)})
ids <- read.csv("A_ids.csv")$id
A <- as.matrix(read.csv("A.csv", header = FALSE)); dimnames(A) <- list(ids, ids); K <- A
tree <- ape::read.tree("tree.nwk")
cc <- read.csv("coords.csv"); coords <- as.matrix(cc[, c("easting", "northing")]); rownames(coords) <- cc$id
ld <- function(f) { d <- read.csv(sprintf("data_%s.csv", f)); d$g <- factor(d$g, levels = unique(d$g)); d }
bin <- binomial(link = "logit")
cells <- list(
  binom_animal_q1  = list(d = "binomial_animal",  fam = bin, f = bf(y ~ x1 + x2 + x3 + animal(1 | g, A = A))),
  binom_relmat_q1  = list(d = "binomial_relmat",  fam = bin, f = bf(y ~ x1 + x2 + x3 + relmat(1 | g, K = K))),
  binom_spatial_q1 = list(d = "binomial_spatial", fam = bin, f = bf(y ~ x1 + x2 + x3 + spatial(1 | g, coords = coords))),
  beta_spatial_q1  = list(d = "beta_spatial", fam = beta_family(),
                          f = bf(y ~ x1 + x2 + x3 + spatial(1 | g, coords = coords), sigma = ~ 1)),
  beta_spatial_q2  = list(d = "beta_spatial", fam = beta_family(),
                          f = bf(y ~ x1 + x2 + x3 + spatial(1 + x1 | g, coords = coords), sigma = ~ 1)),
  beta_phylo_q2    = list(d = "beta_phylo", fam = beta_family(),
                          f = bf(y ~ x1 + x2 + x3 + phylo(1 + x1 | g, tree = tree), sigma = ~ 1)),
  nb2_phylo_lab_q2 = list(d = "nb2_phylo", fam = nbinom2(),
                          f = bf(y ~ x1 + x2 + x3 + phylo(1 + x1 | p | g, tree = tree), sigma = ~ 1)))
rows <- list()
for (nm in names(cells)) {
  cl <- cells[[nm]]
  fit <- drmTMB(cl$f, family = cl$fam, data = ld(cl$d))
  s <- summary(fit$sdr, "fixed")
  rows[[nm]] <- data.frame(cell = nm, idx = seq_len(nrow(s)), par = rownames(s), estimate = s[, 1], se = s[, 2],
    loglik = as.numeric(logLik(fit)), converged = isTRUE(drmTMB::is_converged(fit)), pdhess = isTRUE(fit$sdr$pdHess))
  cat(nm, as.numeric(logLik(fit)), "conv", drmTMB::is_converged(fit), "\n")
}
write.table(do.call(rbind, rows), out, sep = "\t", quote = FALSE, row.names = FALSE)
