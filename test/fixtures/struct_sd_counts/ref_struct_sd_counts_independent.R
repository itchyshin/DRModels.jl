#!/usr/bin/env Rscript
# Independent public-API reproduction of the structured SD fixture.
# Usage: Rscript ref_struct_sd_counts_independent.R FIXTURE_DIR NEW_OUTPUT.csv
# Run with a deliberately selected drmTMB library; record its build identity
# separately. Never overwrite the checked-in reference or an existing output.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Supply fixture directory and a new output CSV path.")
root <- normalizePath(args[[1L]], mustWork = TRUE)
out <- args[[2L]]
if (file.exists(out)) stop("Output already exists; choose a new path.")
if (!requireNamespace("drmTMB", quietly = TRUE) ||
    !requireNamespace("ape", quietly = TRUE)) stop("Install drmTMB and ape first.")
library(drmTMB)
cells <- read.delim(file.path(root, "cells.tsv"), stringsAsFactors = FALSE)
xy <- read.csv(file.path(root, "coords.csv"), row.names = 1)
tree <- ape::read.tree(file.path(root, "tree.nwk"))
A <- as.matrix(read.csv(file.path(root, "A.csv"), header = FALSE))
ids <- read.csv(file.path(root, "A_ids.csv"))[[1L]]
dimnames(A) <- list(ids, ids)
make_formula <- function(family, kind, pat) {
  if (kind == "spatial") {
    if (family == "poisson") {
      if (pat == "one") return(bf(y ~ x1 + x2 + x3 + spatial(1 | g, coords = xy), sd(g, level = "spatial") ~ 1))
      return(bf(y ~ x1 + x2 + x3 + spatial(1 | g, coords = xy), sd(g, level = "spatial") ~ z))
    }
    if (pat == "one") return(bf(y ~ x1 + x2 + x3 + spatial(1 | g, coords = xy), sigma ~ 1, sd(g, level = "spatial") ~ 1))
    return(bf(y ~ x1 + x2 + x3 + spatial(1 | g, coords = xy), sigma ~ 1, sd(g, level = "spatial") ~ z))
  }
  if (kind == "phylo") {
    if (pat == "one") return(bf(y ~ x1 + x2 + x3 + phylo(1 | g, tree = tree), sigma ~ 1, sd(g, level = "phylogenetic") ~ 1))
    return(bf(y ~ x1 + x2 + x3 + phylo(1 | g, tree = tree), sigma ~ 1, sd(g, level = "phylogenetic") ~ z))
  }
  if (kind == "relmat") {
    if (pat == "one") return(bf(y ~ x1 + x2 + x3 + relmat(1 + x1 | g, K = A), sd(g, level = "relmat", coef = "(Intercept)") ~ 1, sd(g, level = "relmat", coef = "x1") ~ 1))
    return(bf(y ~ x1 + x2 + x3 + relmat(1 + x1 | g, K = A), sd(g, level = "relmat", coef = "(Intercept)") ~ z, sd(g, level = "relmat", coef = "x1") ~ z))
  }
  if (kind == "crossed") {
    if (pat == "one") return(bf(y ~ x1 + x2 + x3 + (1 | g) + (1 | g2), sigma ~ 1, sd(g) ~ 1, sd(g2) ~ 1))
    return(bf(y ~ x1 + x2 + x3 + (1 | g) + (1 | g2), sigma ~ 1, sd(g) ~ z, sd(g2) ~ z2))
  }
  if (kind == "indep") {
    if (pat == "first") return(bf(y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g), sigma ~ 1, sd(g, coef = "(Intercept)") ~ z))
    return(bf(y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g), sigma ~ 1, sd(g, coef = "x1") ~ z))
  }
  stop("Unknown fixture kind: ", kind)
}
family_for <- function(x) switch(x, poisson = poisson(), nb2 = nbinom2(), binomial = binomial(), gamma = Gamma(link = "log"), lognormal = lognormal(), student = student(), stop("Unknown family: ", x))
rows <- lapply(seq_len(nrow(cells)), function(i) {
  cell <- cells[i, ]
  message("Fitting ", cell$cell_id)
  d <- read.csv(file.path(root, paste0("data_", cell$family, "_", if (cell$kind == "indep") "ordinary" else cell$kind, ".csv")))
  fit <- drmTMB(make_formula(cell$family, cell$kind, cell$pat), data = d, family = family_for(cell$family))
  co <- coef(fit)
  v <- vcov(fit)
  se <- sqrt(diag(v))
  blocks <- lapply(names(co), function(block) {
    values <- co[[block]]
    keys <- paste(block, names(values), sep = ":")
    if (!all(keys %in% names(se))) stop("Covariance names do not match coefficients for ", cell$cell_id, ": ", block)
    data.frame(cell_id = cell$cell_id, block = gsub('[,"]', "", block), name = names(values), estimate = unname(values), se = unname(se[keys]), loglik = as.numeric(logLik(fit)), convergence = fit$opt$convergence, pdHess = fit$sdr$pdHess, is_converged = is_converged(fit), stringsAsFactors = FALSE)
  })
  do.call(rbind, blocks)
})
write.csv(do.call(rbind, rows), out, row.names = FALSE, na = "NA")
