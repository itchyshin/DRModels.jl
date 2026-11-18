# R reference for test_gaussian_structured_sigma_reml.jl (drmTMB 0.8.1, REML = TRUE).
# Usage: DRMTMB_LIB=<lib with drmTMB> GSR_DIR=<dir> Rscript this.R
# GSR_DIR must hold data_<cell>.csv, K_<cell>.csv, tree_phylo.nwk, coords_spatial.csv written from
# `gsr_fixture(cell)` (gaussian_structured_sigma_reml_fixture.jl). Writes rcoef_<cell>.csv / rll_<cell>.txt;
# the numbers are pasted into the Julia test as `_GSR_R`.
if (nzchar(Sys.getenv("DRMTMB_LIB"))) .libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressMessages({library(drmTMB); library(ape)})
O <- Sys.getenv("GSR_DIR", ".")
for (cell in c("animal","relmat","phylo","spatial","relmat_sx")) {
  d <- read.csv(file.path(O, paste0("data_", cell, ".csv"))); d$id <- factor(d$id)
  ids <- levels(d$id)
  K <- as.matrix(read.csv(file.path(O, paste0("K_", cell, ".csv")), header=FALSE)); dimnames(K) <- list(ids, ids)
  fm <- switch(cell,
    animal = bf(y ~ x, sigma ~ 1 + animal(1|id, A = K), sd(id, level="animal", dpar="sigma") ~ z),
    relmat = bf(y ~ x, sigma ~ 1 + relmat(1|id, K = K), sd(id, level="relmat", dpar="sigma") ~ z),
    relmat_sx = bf(y ~ x, sigma ~ x + relmat(1|id, K = K), sd(id, level="relmat", dpar="sigma") ~ z),
    phylo = { tr <- read.tree(file.path(O, "tree_phylo.nwk")); bf(y ~ x, sigma ~ 1 + phylo(1|id, tree = tr), sd(id, level="phylogenetic", dpar="sigma") ~ z) },
    spatial = { cc <- as.matrix(read.csv(file.path(O, "coords_spatial.csv"), header=FALSE)); rownames(cc) <- ids; bf(y ~ x, sigma ~ 1 + spatial(1|id, coords = cc), sd(id, level="spatial", dpar="sigma") ~ z) })
  fit <- drmTMB(fm, family = gaussian(), data = d, REML = TRUE)
  V <- vcov(fit); cf <- unlist(coef(fit), use.names = FALSE); nm <- rownames(V)
  cat("\n##", cell, "ll", sprintf("%.10f", as.numeric(logLik(fit))), "conv", fit$opt$convergence, "pdHess", fit$sdr$pdHess, "exact", fit$estimator_exact, "\n")
  print(nm)
  out <- data.frame(name = nm, est = cf, se = sqrt(diag(V)))
  write.csv(out, file.path(O, paste0("rcoef_", cell, ".csv")), row.names = FALSE)
  write(sprintf("%.12f", as.numeric(logLik(fit))), file.path(O, paste0("rll_", cell, ".txt")))
  print(out)
}
