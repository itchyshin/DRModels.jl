## gen_temporal_parity.R -- generate the temporal AR1 / OU parity cells (D-310).
##
## Maintainer-only (never run by the Julia tests). Needs R + a drmTMB build
## that has `temporal()` (drmTMB PRs #1446 / #1447). Fits each cell below by
## ML with drmTMB and writes GENERATED NUMBERS ONLY into
## test/parity/temporal/<cell>/expected.toml (+ expected.meta.toml). No
## drmTMB source is copied (drmTMB is GPL; DRModels.jl is MIT).
##
##   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
##     Rscript test/parity/gen_temporal_parity.R [drmTMB_sha]
##
## The input data are the committed CSVs in test/fixtures/temporal/ (see the
## README there for where each one comes from).
##
## WHEN TO RERUN. The committed numbers come from drmTMB dc81bb368, the head
## of the then-unmerged drmTMB PR #1447 (stacked on #1446). That commit lives
## only on an unmerged branch, so once #1446 / #1447 merge, reinstall drmTMB
## at the MERGED main SHA, rerun this script with that SHA, and confirm the
## Julia tests still pass (test/test_parity_temporal.jl and, with
## DRM_PARITY_TESTS=1, test/parity/runparity_temporal.jl). Rerun also after
## any drmTMB change to the temporal likelihood or its optimiser defaults.
##
## For the two article cells (vignette-ar1-ri, vignette-ou-ri) the script also
## records drmTMB's conditional fitted values, the AR1 Wald intervals of the
## mean coefficients and the profile interval for mu:treatment, which
## docs/src/tutorials/temporal-random-effects.md compares against.

suppressPackageStartupMessages(library(drmTMB))

repo_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    return(normalizePath(file.path(dirname(sub("^--file=", "", file_arg[[1]])), "..", "..")))
  }
  normalizePath(getwd())
}

toml_string <- function(x) paste0('"', gsub('"', '\\"', as.character(x), fixed = TRUE), '"')
toml_num <- function(x) {
  if (!is.finite(x)) stop("cannot write non-finite TOML number: ", x)
  format(as.numeric(x), digits = 17, scientific = TRUE, trim = TRUE)
}
toml_bool <- function(x) if (isTRUE(x)) "true" else "false"

root <- repo_root()
fixdir <- file.path(root, "test", "fixtures", "temporal")
outdir <- file.path(root, "test", "parity", "temporal")
sha <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(sha)) sha <- "unknown"

## cell name, data file, drmTMB mean formula, Julia mean formula (for the
## runner), time column, structure, has (1 | id)
cells <- list(
  list(name = "ar1-gapped", file = "ar1_gapped.csv",
       r = y ~ x + temporal(1 | id, time = occ, structure = "ar1"),
       jl = "y ~ x + temporal(1 | id, occ, ar1)", group = "id", iid = FALSE),
  list(name = "ar1-gapped-ri", file = "ar1_gapped.csv",
       r = y ~ x + (1 | id) + temporal(1 | id, time = occ, structure = "ar1"),
       jl = "y ~ x + (1 | id) + temporal(1 | id, occ, ar1)", group = "id", iid = TRUE),
  list(name = "ou-irregular", file = "ou_irregular.csv",
       r = y ~ x + temporal(1 | id, time = elapsed, structure = "ou"),
       jl = "y ~ x + temporal(1 | id, elapsed, ou)", group = "id", iid = FALSE),
  list(name = "ou-irregular-ri", file = "ou_irregular.csv",
       r = y ~ x + (1 | id) + temporal(1 | id, time = elapsed, structure = "ou"),
       jl = "y ~ x + (1 | id) + temporal(1 | id, elapsed, ou)", group = "id", iid = TRUE),
  list(name = "vignette-ar1", file = "vignette_ar1_sites.csv",
       r = y ~ treatment + occasion + temporal(1 | site, time = occasion, structure = "ar1"),
       jl = "y ~ treatment + occasion + temporal(1 | site, occasion, ar1)", group = "site", iid = FALSE),
  list(name = "vignette-ar1-ri", file = "vignette_ar1_sites.csv",
       r = y ~ treatment + occasion + (1 | site) +
         temporal(1 | site, time = occasion, structure = "ar1"),
       jl = "y ~ treatment + occasion + (1 | site) + temporal(1 | site, occasion, ar1)",
       group = "site", iid = TRUE),
  list(name = "vignette-ou", file = "vignette_ou_sites.csv",
       r = y ~ treatment + temporal(1 | site, time = elapsed_days, structure = "ou"),
       jl = "y ~ treatment + temporal(1 | site, elapsed_days, ou)", group = "site", iid = FALSE),
  list(name = "vignette-ou-ri", file = "vignette_ou_sites.csv",
       r = y ~ treatment + (1 | site) + temporal(1 | site, time = elapsed_days, structure = "ou"),
       jl = "y ~ treatment + (1 | site) + temporal(1 | site, elapsed_days, ou)",
       group = "site", iid = TRUE)
)

for (cell in cells) {
  dat <- read.csv(file.path(fixdir, cell$file))
  ## bf() wants literal formulas, so splice the cell's formula in as a call.
  fit <- eval(bquote(drmTMB(bf(.(cell$r), sigma ~ 1), data = dat,
                            family = gaussian(), REML = FALSE)))
  sd_mu <- fit$sdpars$mu
  is_t <- startsWith(names(sd_mu), "temporal_sd")
  stopifnot(sum(is_t) == 1L)
  structure <- if (grepl("\"ar1\"", deparse1(cell$r))) "ar1" else "ou"
  persist <- if (structure == "ar1") fit$corpars$temporal else fit$decaypars$temporal
  beta <- fit$coefficients$mu
  conv <- fit$opt$convergence
  pd <- isTRUE(fit$sdr$pdHess)

  lines <- c(
    "[fit]",
    'family = "gaussian"',
    paste0("formula = ", toml_string(paste0(deparse1(cell$r), "; sigma ~ 1"))),
    paste0("julia_formula = ", toml_string(cell$jl)),
    paste0("data_file = ", toml_string(cell$file)),
    paste0("group = ", toml_string(cell$group)),
    paste0("structure = ", toml_string(structure)),
    paste0("ordinary_intercept = ", toml_bool(cell$iid)),
    'method = "ML"',
    paste0("loglik = ", toml_num(as.numeric(logLik(fit)))),
    paste0("n = ", nrow(dat)),
    "",
    "[coef]",
    paste0(toml_string(paste0("mu_", names(beta))), " = ", vapply(beta, toml_num, "")),
    "",
    "[temporal]",
    paste0("sd = ", toml_num(sd_mu[is_t])),
    if (structure == "ar1") paste0("phi = ", toml_num(persist)) else paste0("decay = ", toml_num(persist)),
    if (cell$iid) paste0("sd_iid = ", toml_num(sd_mu[[paste0("(1 | ", cell$group, ")")]])),
    paste0("sigma = ", toml_num(exp(fit$coefficients$sigma[[1]]))),
    "",
    "[status]",
    paste0("converged = ", toml_bool(identical(as.integer(conv), 0L))),
    paste0("pdHess = ", toml_bool(pd)),
    "",
    "[tol]",
    ## Tightened from the 1e-6 / 1e-5 contract after the first comparison
    ## measured max diffs of 9e-12 (logLik) and 1e-8 (parameters).
    "atol_loglik = 1e-8",
    "rtol_par = 1e-6"
  )
  if (cell$name %in% c("vignette-ar1-ri", "vignette-ou-ri")) {
    ci_rows <- function(ci) unlist(lapply(seq_len(nrow(ci)), function(i) c(
      paste0("[[", if (ci$method[i] == "wald") "wald" else "profile", "]]"),
      paste0("parm = ", toml_string(ci$parm[i])),
      paste0("lower = ", toml_num(ci$lower[i])),
      paste0("upper = ", toml_num(ci$upper[i])),
      paste0("conf_status = ", toml_string(ci$conf.status[i])), "")))
    lines <- c(lines, "",
      "# drmTMB fitted() is CONDITIONAL (adds the site and temporal modes);",
      "# data-row order.",
      "[conditional]",
      paste0("fitted = [", paste(vapply(as.numeric(fitted(fit)), toml_num, ""),
                                 collapse = ", "), "]"), "")
    if (structure == "ar1") lines <- c(lines, ci_rows(confint(fit, method = "wald")))
    lines <- c(lines, ci_rows(confint(fit, parm = "mu:treatment", method = "profile")))
    ## drmTMB's own gate for reporting a temporal mean profile interval: no
    ## `temporal_mean_profile` WARNING from check_drm() (a "note" is fine).
    cd <- as.data.frame(check_drm(fit))
    tmp <- cd[cd$check == "temporal_mean_profile", ]
    stopifnot(nrow(tmp) == 1L)
    lines <- c(lines, "[check_drm]",
      paste0("temporal_mean_profile_status = ", toml_string(tmp$status)),
      paste0("temporal_mean_profile_value = ", toml_string(tmp$value)), "")
  }
  meta <- c(
    paste0("drmtmb_version = ", toml_string(as.character(packageVersion("drmTMB")))),
    paste0("drmtmb_sha = ", toml_string(sha)),
    paste0("r_version = ", toml_string(R.version.string)),
    paste0("generated_on = ", toml_string(format(Sys.Date()))),
    paste0("r_call = ", toml_string(paste0(
      "drmTMB(bf(", deparse1(cell$r), ", sigma ~ 1), data = read.csv(\"", cell$file,
      "\"), family = gaussian(), REML = FALSE)"))),
    paste0("opt_convergence = ", as.integer(conv)),
    paste0("max_abs_gradient = ", toml_num(max(abs(unlist(fit$gradient))))),
    paste0("note = ", toml_string(paste(
      "Generated outputs only; no drmTMB source vendored.",
      "Point estimates at the ML optimum; no interval or calibration claim.")))
  )
  d <- file.path(outdir, cell$name)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  writeLines(lines, file.path(d, "expected.toml"))
  writeLines(meta, file.path(d, "expected.meta.toml"))
  cat(sprintf("%-18s logLik %.10f  conv %s  pdHess %s\n", cell$name,
              as.numeric(logLik(fit)), conv, pd))
}
