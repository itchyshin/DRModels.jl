# Phase 1: R fit of one cell -> out/xr/<n>.tsv : id, ll, pdHess, conv, par(;), names(;)
id <- commandArgs(TRUE)[1]; n <- commandArgs(TRUE)[2]
suppressPackageStartupMessages(library(drmTMB)); source("r_cell.R")
dir.create("../out/xr", showWarnings = FALSE)
fit <- fit_cell(id)
ll <- as.numeric(logLik(fit))
line <- paste(id, sprintf("%.10f", ll), isTRUE(fit$sdr$pdHess), fit$opt$convergence,
  paste(sprintf("%.12g", fit$opt$par), collapse = ";"), paste(names(fit$opt$par), collapse = ";"), sep = "\t")
writeLines(line, file.path("../out/xr", paste0(n, ".tsv")))
