## gen_temporal_wave2_parity.R -- generate the temporal WAVE-2 parity cells
## (D-311): the paired phylogenetic stable intercept + OU provider
## (`phylo(1 | species, tree = tree) + temporal(1 | species, time = elapsed,
## structure = "ou")`) and the homogeneous Toeplitz structure
## (`temporal(1 | id, time = occ, structure = "homtoep")`).
##
## Maintainer-only (never run by the Julia tests). Needs R, ape, and a drmTMB
## build that has the provider being regenerated, installed into a PRIVATE
## library (never the shared one): drmTMB draft PR #1448
## (`claude/temporal-phylo-ou-land`, rescued from
## `codex/phylo-temporal-ou-exec-v1-20260909`) for the phylo + OU cells, and
## draft PR #1449 (`claude/temporal-homtoep-land`, rescued from
## `codex/temporal-homtoep-v1-20260910`) for the homtoep cells. The optional
## third argument selects the cell set (`phylo`, `homtoep`, or `all`):
##
##   R_LIBS=~/scratch/drmtmb-wave2-lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
##     Rscript test/parity/gen_temporal_wave2_parity.R <drmTMB_sha> [drmTMB_source_dir] [phylo|homtoep|all]
##
## Writes GENERATED NUMBERS ONLY into test/parity/temporal/<cell>/expected.toml
## (+ expected.meta.toml, stamped with tools/drmtmb_provenance_lib.R's
## code hash, which test/test_fixture_provenance.jl requires). No drmTMB source
## is copied (drmTMB is GPL; DRModels.jl is MIT).
##
## Inputs: test/fixtures/temporal/phylo_ou_species.{csv,newick} (simulated by
## test/fixtures/temporal/generate.jl, no drmTMB involvement) and, for the
## article cell, vignette_phylo_ou_species.{csv,newick}: the data of drmTMB's
## *Phylogenetic stable effects and temporal OU deviations* article. When those
## two files are absent and `drmTMB_source_dir` is given, the script runs that
## article's `simulate` chunk (knitr::purl + sys.source, unchanged) and writes
## its `dat` and `tree` with 17 significant digits; only those generated data
## are committed, never the article's code.

suppressPackageStartupMessages({
  library(drmTMB)
  library(ape)
})

repo_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    return(normalizePath(file.path(dirname(sub("^--file=", "", file_arg[[1]])), "..", "..")))
  }
  normalizePath(getwd())
}

source(file.path(repo_root(), "tools", "drmtmb_provenance_lib.R"))

toml_string <- function(x) paste0('"', gsub('"', '\\"', as.character(x), fixed = TRUE), '"')
toml_num <- function(x) {
  if (!is.finite(x)) stop("cannot write non-finite TOML number: ", x)
  format(as.numeric(x), digits = 17, scientific = TRUE, trim = TRUE)
}
toml_bool <- function(x) if (isTRUE(x)) "true" else "false"

root <- repo_root()
fixdir <- file.path(root, "test", "fixtures", "temporal")
outdir <- file.path(root, "test", "parity", "temporal")
args <- commandArgs(trailingOnly = TRUE)
sha <- if (length(args) >= 1L) args[[1L]] else "unknown"
srcdir <- if (length(args) >= 2L) args[[2L]] else NA_character_
which_cells <- if (length(args) >= 3L) args[[3L]] else "all"
stopifnot(which_cells %in% c("phylo", "homtoep", "all"))
do_phylo <- which_cells %in% c("phylo", "all")
do_homtoep <- which_cells %in% c("homtoep", "all")

## --- the article's data (exported once, then committed) -------------------
vig_csv <- file.path(fixdir, "vignette_phylo_ou_species.csv")
vig_nwk <- file.path(fixdir, "vignette_phylo_ou_species.newick")
if (do_phylo && !file.exists(vig_csv) && !is.na(srcdir)) {
  rmd <- file.path(srcdir, "vignettes", "phylogenetic-temporal-effects.Rmd")
  code <- tempfile(fileext = ".R")
  has_ape <- TRUE   # the article's chunks are `eval = has_ape`; purl resolves it here
  knitr::purl(rmd, output = code, quiet = TRUE, documentation = 0L)
  lines <- readLines(code)
  ## keep everything up to the end of the `simulate` chunk (before the fit)
  stop_at <- grep("^fit <- drmTMB", lines)[[1L]] - 1L
  env <- new.env()
  writeLines(lines[seq_len(stop_at)], code)
  sys.source(code, envir = env)
  out <- env$dat
  out[] <- lapply(out, function(v) if (is.double(v)) sprintf("%.17g", v) else as.character(v))
  utils::write.table(out, vig_csv, sep = ",", quote = FALSE, row.names = FALSE)
  ape::write.tree(env$tree, vig_nwk, digits = 17)
  back <- utils::read.csv(vig_csv)
  stopifnot(identical(back$y, env$dat$y))
  cat("exported article data:", nrow(out), "rows\n")
}

cells <- list(
  list(name = "phylo-ou-species", file = "phylo_ou_species.csv", tree = "phylo_ou_species.newick",
       r = y ~ x + phylo(1 | species, tree = tree) +
         temporal(1 | species, time = elapsed, structure = "ou"),
       jl = "y ~ x + phylo(1 | species) + temporal(1 | species, elapsed, ou)",
       group = "species", article = FALSE),
  list(name = "vignette-phylo-ou", file = "vignette_phylo_ou_species.csv",
       tree = "vignette_phylo_ou_species.newick",
       r = y ~ treatment + phylo(1 | species, tree = tree) +
         temporal(1 | species, time = elapsed_days, structure = "ou"),
       jl = "y ~ treatment + phylo(1 | species) + temporal(1 | species, elapsed_days, ou)",
       group = "species", article = TRUE)
)

for (cell in if (do_phylo) cells else list()) {
  if (!file.exists(file.path(fixdir, cell$file))) {
    cat("skip", cell$name, "(no data file)\n")
    next
  }
  dat <- utils::read.csv(file.path(fixdir, cell$file), stringsAsFactors = FALSE)
  tree <- ape::read.tree(file.path(fixdir, cell$tree))
  fit <- eval(bquote(drmTMB(bf(.(cell$r), sigma ~ 1), data = dat,
                            family = gaussian(), REML = FALSE)))
  sd_mu <- fit$sdpars$mu
  stopifnot(all(c("sd_temporal", "sd_phylo_stable") %in% names(sd_mu)))
  decay <- fit$decaypars$temporal[["decay_temporal"]]
  beta <- fit$coefficients$mu
  conv <- fit$opt$convergence
  pd <- isTRUE(fit$sdr$pdHess)
  lines <- c(
    "[fit]",
    'family = "gaussian"',
    paste0("formula = ", toml_string(paste0(deparse1(cell$r), "; sigma ~ 1"))),
    paste0("julia_formula = ", toml_string(cell$jl)),
    paste0("data_file = ", toml_string(cell$file)),
    paste0("tree_file = ", toml_string(cell$tree)),
    paste0("group = ", toml_string(cell$group)),
    'structure = "ou"',
    "ordinary_intercept = false",
    'method = "ML"',
    paste0("loglik = ", toml_num(as.numeric(logLik(fit)))),
    paste0("n = ", nrow(dat)),
    "",
    "[coef]",
    paste0(toml_string(paste0("mu_", names(beta))), " = ", vapply(beta, toml_num, "")),
    "",
    "[temporal]",
    paste0("sd = ", toml_num(sd_mu[["sd_temporal"]])),
    paste0("decay = ", toml_num(decay)),
    paste0("sd_phylo = ", toml_num(sd_mu[["sd_phylo_stable"]])),
    paste0("sigma = ", toml_num(exp(fit$coefficients$sigma[[1]]))),
    "",
    "[status]",
    paste0("converged = ", toml_bool(identical(as.integer(conv), 0L))),
    paste0("pdHess = ", toml_bool(pd)),
    "",
    "[tol]",
    "atol_loglik = 1e-8",
    "rtol_par = 1e-6",
    "",
    "# drmTMB fitted() is CONDITIONAL (adds the phylogenetic stable and the",
    "# temporal modes); data-row order.",
    "[conditional]",
    paste0("fitted = [", paste(vapply(as.numeric(fitted(fit)), toml_num, ""),
                               collapse = ", "), "]"),
    ""
  )
  if (isTRUE(cell$article)) {
    ## drmTMB's paired provider exposes fixed-mean profiles only; record the
    ## treatment profile for the article comparison (no calibration claim --
    ## drmTMB's own article says not to report it yet).
    ci <- try(confint(fit, parm = "mu:treatment", method = "profile"), silent = TRUE)
    if (!inherits(ci, "try-error")) {
      lines <- c(lines, "[[profile]]",
        paste0("parm = ", toml_string(ci$parm[[1]])),
        paste0("lower = ", toml_num(ci$lower[[1]])),
        paste0("upper = ", toml_num(ci$upper[[1]])),
        paste0("conf_status = ", toml_string(ci$conf.status[[1]])), "")
    }
  }
  prov <- drmtmb_provenance()
  meta <- c(
    paste0("drmtmb_version = ", toml_string(as.character(packageVersion("drmTMB")))),
    paste0("drmtmb_code_hash = ", toml_string(prov$code_hash)),
    paste0("drmtmb_built = ", toml_string(prov$built)),
    paste0("drmtmb_sha = ", toml_string(sha)),
    paste0("r_version = ", toml_string(R.version.string)),
    paste0("ape_version = ", toml_string(as.character(packageVersion("ape")))),
    paste0("generated_on = ", toml_string(format(Sys.Date()))),
    paste0("r_call = ", toml_string(paste0(
      "tree <- ape::read.tree(\"", cell$tree, "\"); drmTMB(bf(", deparse1(cell$r),
      ", sigma ~ 1), data = read.csv(\"", cell$file, "\"), family = gaussian(), REML = FALSE)"))),
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
  cat(sprintf("%-20s logLik %.10f  conv %s  pdHess %s\n", cell$name,
              as.numeric(logLik(fit)), conv, pd))
}

## --- homogeneous Toeplitz cells (Slice B) ---------------------------------
## `temporal(1 | id, time = occ, structure = "homtoep")`: `sigma` is the
## TOTAL within-series SD and `corpars$temporal` the lag correlations
## (`cor_lag1`, ...). drmTMB's fitted() is the marginal mean X beta here (no
## latent temporal states), so no conditional block is needed beyond it.
## The article's Toeplitz data (`complete_regular_panel`, `set.seed(20261002)`
## in drmTMB's *Temporal AR1, OU, and Toeplitz effects* article) are exported
## once, like the phylo article data above: the article's chunks run unchanged
## up to its Toeplitz fit, and only the generated data frame is written.
vig_toep <- file.path(fixdir, "vignette_homtoep_sites.csv")
if (do_homtoep && !file.exists(vig_toep) && !is.na(srcdir)) {
  rmd <- file.path(srcdir, "vignettes", "temporal-random-effects.Rmd")
  code <- tempfile(fileext = ".R")
  knitr::purl(rmd, output = code, quiet = TRUE, documentation = 0L)
  lines <- readLines(code)
  stop_at <- grep("^toeplitz_fit <- drmTMB", lines)[[1L]] - 1L
  writeLines(lines[seq_len(stop_at)], code)
  env <- new.env()
  sys.source(code, envir = env)
  out <- env$complete_regular_panel
  out[] <- lapply(out, function(v) if (is.double(v)) sprintf("%.17g", v) else as.character(v))
  utils::write.table(out, vig_toep, sep = ",", quote = FALSE, row.names = FALSE)
  back <- utils::read.csv(vig_toep)
  stopifnot(identical(back$y, env$complete_regular_panel$y))
  cat("exported Toeplitz article data:", nrow(out), "rows\n")
}

homtoep_cells <- list(
  list(name = "homtoep-panel6", file = "homtoep_panel6.csv", group = "id", parm = "mu:x",
       r = y ~ x + temporal(1 | id, time = occ, structure = "homtoep"),
       jl = "y ~ x + temporal(1 | id, occ, homtoep)"),
  list(name = "homtoep-neg4", file = "homtoep_neg4.csv", group = "id", parm = "mu:x",
       r = y ~ x + temporal(1 | id, time = occ, structure = "homtoep"),
       jl = "y ~ x + temporal(1 | id, occ, homtoep)"),
  list(name = "vignette-homtoep", file = "vignette_homtoep_sites.csv", group = "site",
       parm = "mu:treatment",
       r = y ~ treatment + temporal(1 | site, time = occasion, structure = "homtoep"),
       jl = "y ~ treatment + temporal(1 | site, occasion, homtoep)")
)
for (cell in if (do_homtoep) homtoep_cells else list()) {
  dat <- utils::read.csv(file.path(fixdir, cell$file), stringsAsFactors = FALSE)
  fit <- eval(bquote(drmTMB(bf(.(cell$r), sigma ~ 1), data = dat,
                            family = gaussian(), REML = FALSE)))
  rho <- fit$corpars$temporal
  stopifnot(identical(names(rho), paste0("cor_lag", seq_along(rho))))
  beta <- fit$coefficients$mu
  conv <- fit$opt$convergence
  pd <- isTRUE(fit$sdr$pdHess)
  ## drmTMB's article uses the tmbprofile engine for this interval.
  ci <- try(confint(fit, parm = cell$parm, method = "profile",
                    profile_engine = "tmbprofile"), silent = TRUE)
  lines <- c(
    "[fit]",
    'family = "gaussian"',
    paste0("formula = ", toml_string(paste0(deparse1(cell$r), "; sigma ~ 1"))),
    paste0("julia_formula = ", toml_string(cell$jl)),
    paste0("data_file = ", toml_string(cell$file)),
    paste0("group = ", toml_string(cell$group)),
    'structure = "homtoep"',
    "ordinary_intercept = false",
    'method = "ML"',
    paste0("loglik = ", toml_num(as.numeric(logLik(fit)))),
    paste0("n = ", nrow(dat)),
    "",
    "[coef]",
    paste0(toml_string(paste0("mu_", names(beta))), " = ", vapply(beta, toml_num, "")),
    "",
    "[temporal]",
    paste0("sigma = ", toml_num(exp(fit$coefficients$sigma[[1]]))),
    paste0("cor = [", paste(vapply(unname(rho), toml_num, ""), collapse = ", "), "]"),
    "",
    "[status]",
    paste0("converged = ", toml_bool(identical(as.integer(conv), 0L))),
    paste0("pdHess = ", toml_bool(pd)),
    "",
    "[tol]",
    "atol_loglik = 1e-8",
    "rtol_par = 1e-6",
    "",
    "# drmTMB's Pearson residuals: the Levinson-whitened L^-1 (y - X beta), with",
    "# L the Cholesky factor of sigma^2 R per series; data-row order.",
    "[residuals]",
    paste0("pearson = [", paste(vapply(as.numeric(residuals(fit, type = "pearson")), toml_num, ""),
                                collapse = ", "), "]"),
    "",
    if (!inherits(ci, "try-error")) c("[[profile]]",
      paste0("parm = ", toml_string(ci$parm[[1]])),
      paste0("lower = ", toml_num(ci$lower[[1]])),
      paste0("upper = ", toml_num(ci$upper[[1]])),
      paste0("conf_status = ", toml_string(ci$conf.status[[1]])), "")
  )
  prov <- drmtmb_provenance()
  meta <- c(
    paste0("drmtmb_version = ", toml_string(as.character(packageVersion("drmTMB")))),
    paste0("drmtmb_code_hash = ", toml_string(prov$code_hash)),
    paste0("drmtmb_built = ", toml_string(prov$built)),
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
  cat(sprintf("%-20s logLik %.10f  conv %s  pdHess %s\n", cell$name,
              as.numeric(logLik(fit)), conv, pd))
}
