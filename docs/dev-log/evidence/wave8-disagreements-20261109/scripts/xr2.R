# Phase 3: R objective at Julia's estimate.
id <- commandArgs(TRUE)[1]; n <- commandArgs(TRUE)[2]
suppressPackageStartupMessages(library(drmTMB)); source("r_cell.R")
fit <- fit_cell(id)
jl <- strsplit(readLines(file.path("../out/xj", paste0(n, ".tsv"))), "\t")[[1]]
thJ <- as.numeric(strsplit(jl[5], ";")[[1]])
res <- NA_real_; note <- ""
if (length(thJ) == length(fit$opt$par)) {
  drmTMB:::drm_pin_tmb_object_to_optimum(fit$obj, fit$opt, fit$tmb_state)
  res <- -as.numeric(fit$obj$fn(thJ))
  drmTMB:::drm_pin_tmb_object_to_optimum(fit$obj, fit$opt, fit$tmb_state)
  chk <- -as.numeric(fit$obj$fn(fit$opt$par))
  note <- sprintf("selfcheck_ll=%.10f", chk)
} else note <- sprintf("npar differs: R %d vs Julia %d", length(fit$opt$par), length(thJ))
writeLines(paste(id, sprintf("%.10f", as.numeric(logLik(fit))), sprintf("%.10f", res), note, sep = "\t"),
           file.path("../out/xr", paste0(n, ".cross.tsv")))
