#!/usr/bin/env Rscript
# Reproduce the R side of the lower-mode check through drmTMB's public API.
# Usage: Rscript agree_independent.R GENERATOR_OUTPUT_DIR JOBS.tsv NEW_OUTPUT.tsv
# JOBS.tsv has three tab-separated columns: route, cell ID, and seed, without
# a header. GENERATOR_OUTPUT_DIR is the parent directory containing gen_data.jl's data/ directory. Never overwrite
# an existing output. Run with one BLAS/OpenMP thread and a selected R library.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Supply the generator output directory (parent of data/), jobs TSV, and new output path.")
if (file.exists(args[[3L]])) stop("Output exists; choose a new path.")
if (!requireNamespace("drmTMB", quietly = TRUE) || !requireNamespace("ape", quietly = TRUE)) stop("Install drmTMB and ape first.")
library(drmTMB)
root <- normalizePath(args[[1L]], mustWork = TRUE)
dir <- file.path(root, "data")
jobs <- read.delim(args[[2L]], header = FALSE, stringsAsFactors = FALSE, col.names = c("route", "cell", "seed"), quote = "", comment.char = "")
if (ncol(jobs) != 3L || nrow(jobs) == 0L || anyNA(jobs)) stop("Jobs need route, cell ID and seed columns.")
coords <- read.csv(file.path(dir, "coords.csv"), row.names = 1)
tree <- ape::read.tree(file.path(dir, "tree.nwk"))
formula_for <- function(kind, q) {
  if (kind == "phylo" && q == 1L) return(bf(y ~ x1 + x2 + x3 + phylo(1 | g, tree = tree), sigma ~ 1, sd(g, level = "phylogenetic") ~ z))
  if (kind == "spatial" && q == 1L) return(bf(y ~ x1 + x2 + x3 + spatial(1 | g, coords = coords), sigma ~ 1, sd(g, level = "spatial") ~ z))
  if (kind == "phylo" && q == 2L) return(bf(y ~ x1 + x2 + x3 + phylo(1 + x1 | g, tree = tree), sigma ~ 1, sd(g, level = "phylogenetic", coef = "(Intercept)") ~ z, sd(g, level = "phylogenetic", coef = "x1") ~ z))
  if (kind == "spatial" && q == 2L) return(bf(y ~ x1 + x2 + x3 + spatial(1 + x1 | g, coords = coords), sigma ~ 1, sd(g, level = "spatial", coef = "(Intercept)") ~ z, sd(g, level = "spatial", coef = "x1") ~ z))
  stop("Unsupported kind or coefficient count.")
}
results <- lapply(seq_len(nrow(jobs)), function(i) {
  job <- jobs[i, ]
  fields <- strsplit(job$cell, "|", fixed = TRUE)[[1L]]
  if (length(fields) != 7L || !fields[1] %in% c("gamma", "nb2") || !fields[2] %in% c("phylo", "spatial") || !fields[4] %in% c("q1", "q2") || !identical(fields[c(3, 5, 6, 7)], c("lab0", "sd-z", "cm0", "ML")) || !grepl("^[0-9]+$", job$seed)) stop("Unsupported job: ", job$cell)
  q <- as.integer(sub("q", "", fields[4], fixed = TRUE))
  expected <- paste(fields[1], fields[2], fields[4], sep = "_")
  if (job$route != expected) stop("Route does not match cell: ", job$route)
  data_file <- file.path(dir, paste0("data_", fields[1], "_", fields[2], "_s", job$seed, ".csv"))
  d <- read.csv(data_file)
  message("Fitting ", job$route, " seed ", job$seed)
  started <- proc.time()[["elapsed"]]
  fit <- drmTMB(formula_for(fields[2], q), data = d, family = if (fields[1] == "gamma") Gamma(link = "log") else nbinom2())
  data.frame(route = job$route, seed = job$seed, llR = as.numeric(logLik(fit)), R_opt_conv = fit$opt$convergence, R_flags = paste0("pdHess=", fit$sdr$pdHess, ";is_converged=", is_converged(fit)), secsR = proc.time()[["elapsed"]] - started)
})
write.table(do.call(rbind, results), args[[3L]], sep = "\t", row.names = FALSE, quote = FALSE)
