# R side of the agreement protocol (docs/dev-log/2026-11-08-tweedie-structured-sd.md).
# Rscript agreement_r.R <fixture_dir> <out.tsv>      Env: DRMTMB_LIB (library holding drmTMB 0.8.1)
args <- commandArgs(TRUE); dir <- args[1]; out <- args[2]
if (nzchar(Sys.getenv("DRMTMB_LIB"))) .libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
rows <- list()
add <- function(ds, model, fit, secs) {
  est <- unname(fit$opt$par)
  V <- tryCatch(vcov(fit), error = function(e) NULL)
  pd <- isTRUE(fit$sdr$pdHess)
  sev <- unname(if (!is.null(V)) sqrt(diag(V)) else rep(NA_real_, length(est)))
  nm <- tryCatch(names(coef(fit)), error = function(e) rep("", length(est)))
  rows[[length(rows) + 1]] <<- data.frame(engine = "R", dataset = ds, model = model, idx = seq_along(est),
    name = if (length(nm) == length(est)) unname(nm) else "", est = est, se = sev[seq_along(est)],
    conv = fit$opt$convergence, pdhess = pd, secs = secs, row.names = NULL)
}
for (ds in c("phylo_1", "phylo_2", "phylo_3", "relmat_1", "relmat_2")) {
  d <- read.csv(file.path(dir, paste0(ds, ".csv")))
  d$g <- factor(d$g, levels = unique(d$g))
  if (startsWith(ds, "phylo")) {
    tree <- ape::read.tree(file.path(dir, paste0(ds, "_tree.nwk")))
    re <- "phylo(1 | g, tree = tree)"; lvl <- "phylogenetic"; ex <- list(tree = tree)
  } else {
    K <- as.matrix(read.csv(file.path(dir, paste0(ds, "_K.csv"))))
    dimnames(K) <- list(levels(d$g), levels(d$g))
    re <- "relmat(1 | g, K = K)"; lvl <- "relmat"
  }
  for (model in c("sd-none", "sd-z")) {
    mu <- as.formula(paste("y ~ x +", re)); environment(mu) <- environment()
    parts <- list(mu)
    if (model == "sd-z") parts <- c(parts, list(as.formula(sprintf('sd(g, level = "%s") ~ z', lvl))))
    fm <- do.call(bf, parts)
    t0 <- Sys.time()
    fit <- tryCatch(drmTMB(fm, family = tweedie(), data = d), error = function(e) e)
    secs <- as.numeric(Sys.time() - t0, units = "secs")
    if (inherits(fit, "error")) { message(ds, " ", model, ": ", conditionMessage(fit)); next }
    add(ds, model, fit, secs)
  }
}
write.table(do.call(rbind, rows), out, sep = "\t", quote = FALSE, row.names = FALSE)
