#!/usr/bin/env Rscript
# Run an UNMODIFIED drmTMB tools/ runner against the private installed library
# (DRMTMB_PRIVLIB) instead of pkgload::load_all() on the checkout. Nothing is
# copied from drmTMB: the runner file is sourced in place, so its own
# sha('tools/<runner>') records the genuine runner bytes. Only load_all is
# replaced by a no-op after library(drmTMB) has attached the private build.
.shim_args <- commandArgs(TRUE)
.shim_tool <- .shim_args[[1]]
.shim_lib <- normalizePath(Sys.getenv("DRMTMB_PRIVLIB"), mustWork = TRUE)
.libPaths(c(.shim_lib, .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
stopifnot(identical(normalizePath(find.package("drmTMB")), file.path(.shim_lib, "drmTMB")))
utils::assignInNamespace("load_all", function(...) invisible(NULL), ns = "pkgload")
# The drmTMB public runners record the loaded Julia source as pathof(DRM), the
# module's pre-rename name. drmTMB binds the loaded backend as Main.drmTMB_backend,
# so alias DRM to it (read-only identity; no computation changes).
suppressMessages(trace("drm_julia_setup", where = asNamespace("drmTMB"), print = FALSE,
  exit = quote(if (!isTRUE(JuliaCall::julia_eval("isdefined(Main, :DRM)")))
    JuliaCall::julia_command("const DRM = drmTMB_backend"))))
commandArgs <- function(trailingOnly = FALSE) if (trailingOnly) .shim_args[-1] else base::commandArgs(FALSE)
cat("SHIM drmTMB from", find.package("drmTMB"), "version", as.character(packageVersion("drmTMB")), "\n")
source(.shim_tool, echo = FALSE)
