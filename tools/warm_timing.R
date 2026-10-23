# tools/warm_timing.R — S12 (#563), root gate G5.
#
# Native-R warm-workflow timing harness (drmTMB, engine="tmb" -- never
# engine="julia"). Reads the SAME fixture CSV/Newick files
# tools/warm_timing_fixtures.jl wrote (never regenerates data), fits each
# registered workflow via drmTMB's own public front end, and writes a TSV with
# the identical column set tools/warm_timing.jl writes so
# tools/warm_timing_compare.jl can join them.
#
# Registry: docs/dev-log/evidence/julia-r-parity/warm-workflow-registry.md
#
# Timer floor (fixed 2026-09-02, see registry doc §1): `proc.time()` is
# MILLISECOND-floored on this host (confirmed empirically: successive calls
# only ever differ by whole 0.001s steps) -- useless for a leg that finishes
# in microseconds (predict, forced-uncertainty on small fixtures). Two fixes,
# together:
#   (a) sub-millisecond clock: `microbenchmark::get_nanotime()` (nanosecond
#       resolution) if the `microbenchmark` package is installed (it IS, on
#       this host, checked 2026-09-02) -- falls back to `Sys.time()`
#       (~1-10 microsecond resolution measured on this host, still far above
#       proc.time()'s 1ms floor) otherwise. Printed at startup so a run's
#       receipt says which was used.
#   (b) loop-until-0.25s: each timed repetition calls the leg repeatedly
#       until cumulative wall time >= 0.25s (minimum 1 call), reports
#       per-call = total/calls. `reps` such loops are run per leg;
#       median_s/min_s are the median/min of the per-call numbers, and
#       `calls` (new TSV column) is the mean calls-per-loop, rounded.
#
# Uncertainty leg (fixed 2026-09-02, see registry doc §1): `vcov(fit)` /
# `confint(fit)` on a drmTMB object read `object$sdr` -- a `TMB::sdreport()`
# result computed ONCE inside `drmTMB()` and cached on the fit
# (`drmTMB:::drm_sdreport_cov_coefficients` reads `object$sdr$cov.fixed`
# directly) -- timing `confint(fit)` measures a cached list-field read, not
# the actual delta-method/Hessian computation. This harness instead times
# `TMB::sdreport(fit$obj)` directly -- `fit$obj` is the live TMB ADFun object
# every drmTMB fit carries, and calling `sdreport()` on it fresh re-invokes
# TMB's own C++ AD Hessian and covariance computation (verified to reproduce
# `fit$sdr$cov.fixed` to machine precision on 2026-09-02 for
# `lognormal_locscale`). Unlike the Julia side, this generalises to every
# family/route here (TMB::sdreport operates on the compiled ADFun, not on any
# family-specific R closure) -- confirmed for 7 of the 10 workflows directly;
# NOT independently re-verified for `large_sparse_lss_p2000` (p~2048) due to
# its cost (see registry doc "Estimate").
#
# Grammar notes found and fixed during this same validation pass (differ from
# DRModels.jl's Julia grammar -- see registry doc §3 for the full matched-call
# tables):
#   - `phylo(1 | grp, tree = TREE)` takes the tree INLINE in the formula on
#     the R side (Julia: `tree=` is a separate `drm()` keyword argument).
#   - `meta_V(V = v)` requires the NAMED argument `V` (Julia: `meta_V(v)`
#     positional).
#   - `sd(g) ~ zg` / `sd(species, level = "phylogenetic") ~ xs`: the sd()
#     predictor must be constant WITHIN each level of the grouping factor on
#     BOTH engines -- an observation-level covariate (this script's earlier,
#     wrong draft used the same column as the mean-model predictor) is
#     refused by drm()/drmTMB() alike; the fixtures now carry a dedicated
#     group-/species-level column (`zg`, `xs`) for this.
#
# Threading:
#   - TMB::openmp(N) sets TMB's C++ AD-backend OpenMP thread count -- the
#     lever that actually matters for a TMB Laplace fit.
#   - RhpcBLASctl::blas_set_num_threads(N) would additionally pin R's BLAS if
#     installed; on THIS host (Mac, R 4.6.0, checked 2026-09-02) it is NOT
#     installed, so this script falls back to OMP_NUM_THREADS/OPENBLAS_NUM_THREADS
#     env vars (set before the R process starts -- setting them mid-session via
#     Sys.setenv() is NOT guaranteed to change an already-loaded BLAS's active
#     thread count) and prints a warning naming the gap. Install RhpcBLASctl
#     for a stronger guarantee before citing R-side thread scaling as verified.
#
# Usage (run once per thread count; set OMP_NUM_THREADS/OPENBLAS_NUM_THREADS
# in the environment BEFORE starting R, since the BLAS may already be loaded
# by the time this script runs):
#   OMP_NUM_THREADS=N OPENBLAS_NUM_THREADS=N Rscript tools/warm_timing.R \
#       --threads N --reps 3 \
#       --fixtures docs/dev-log/evidence/julia-r-parity/warm-fixtures \
#       --out docs/dev-log/evidence/julia-r-parity/warm-timing-r-tN.tsv
#   (add --workflow NAME to run a single workflow, e.g. for the smoke pre-run)

suppressMessages(library(drmTMB))
suppressMessages(library(ape))

# ---- CLI --------------------------------------------------------------------

parse_args <- function(argv) {
  a <- list(threads = NULL, reps = 3L, out = NULL,
            fixtures = "docs/dev-log/evidence/julia-r-parity/warm-fixtures",
            workflow = NULL)
  i <- 1L
  while (i <= length(argv)) {
    key <- argv[i]
    if (key == "--threads") { a$threads <- as.integer(argv[i + 1]); i <- i + 2L }
    else if (key == "--reps") { a$reps <- as.integer(argv[i + 1]); i <- i + 2L }
    else if (key == "--out") { a$out <- argv[i + 1]; i <- i + 2L }
    else if (key == "--fixtures") { a$fixtures <- argv[i + 1]; i <- i + 2L }
    else if (key == "--workflow") { a$workflow <- argv[i + 1]; i <- i + 2L }
    else stop("warm_timing.R: unknown arg ", key)
  }
  if (is.null(a$threads)) stop("warm_timing.R: --threads N is required")
  if (is.null(a$out)) stop("warm_timing.R: --out FILE is required")
  a
}
args <- parse_args(commandArgs(trailingOnly = TRUE))

# ---- thread control -----------------------------------------------------

TMB::openmp(args$threads)
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
  RhpcBLASctl::blas_set_num_threads(args$threads)
} else {
  cat(sprintf("WARNING: RhpcBLASctl not installed -- R BLAS thread count NOT pinned to %d at runtime; only TMB::openmp(%d) is active. Set OMP_NUM_THREADS=%d OPENBLAS_NUM_THREADS=%d before starting R for a stronger guarantee.\n",
              args$threads, args$threads, args$threads, args$threads))
}

# ---- sub-millisecond clock ---------------------------------------------------
# proc.time() is millisecond-floored on this host -- see header note.

if (requireNamespace("microbenchmark", quietly = TRUE)) {
  hires_time <- function() microbenchmark::get_nanotime() / 1e9   # -> seconds
  TIMER_LABEL <- "microbenchmark::get_nanotime() (nanosecond resolution)"
} else {
  hires_time <- function() as.numeric(Sys.time())
  TIMER_LABEL <- "Sys.time() (microbenchmark not installed -- microsecond-ish, NOT nanosecond)"
}
cat("timer:", TIMER_LABEL, "\n")

# ---- timing: loop until >=0.25s cumulative, per-call = total/calls ----------

TIMER_FLOOR_S <- 0.25

timed_loop <- function(f, floor_s = TIMER_FLOOR_S) {
  calls <- 0L; total <- 0; val <- NULL
  repeat {
    t0 <- hires_time()
    val <- f()
    total <- total + (hires_time() - t0)
    calls <- calls + 1L
    if (total >= floor_s) break
  }
  list(val = val, per_call = total / calls, calls = calls)
}

time_reps <- function(f, k, floor_s = TIMER_FLOOR_S) {
  per_call <- numeric(k); calls_vec <- integer(k); val <- NULL
  for (i in seq_len(k)) {
    r <- timed_loop(f, floor_s)
    val <- r$val; per_call[i] <- r$per_call; calls_vec[i] <- r$calls
  }
  list(val = val, times = per_call, median_s = stats::median(per_call),
       min_s = min(per_call), calls = round(mean(calls_vec)))
}

fit_converged <- function(fit) tryCatch(isTRUE(fit$opt$convergence == 0L), error = function(e) NA)
fit_loglik <- function(fit) tryCatch(as.numeric(stats::logLik(fit)), error = function(e) NA_real_)

# Forces a genuine covariance recomputation from the live TMB ADFun object
# (NOT drmTMB's cached `fit$sdr` -- see header note).
force_sdreport <- function(fit) TMB::sdreport(fit$obj)

# ---- workflow registry (must match warm-workflow-registry.md) ---------------
# Each entry returns list(fit = function() ..., data = <training frame>).
# Every workflow here supports the forced-sdreport uncertainty leg (fit$obj
# is a generic TMB ADFun regardless of family/route on the R side -- see
# header note); large_sparse_lss_p2000 is UNTESTED for cost reasons, run with
# the same code path and watched, not special-cased.

workflows <- list(
  gauss_mixed_phylo_mean = function(fixdir) {
    d <- read.csv(file.path(fixdir, "gauss_mixed_phylo_mean.csv"), stringsAsFactors = FALSE)
    d$study <- as.integer(d$study)
    tree <- ape::read.tree(file.path(fixdir, "gauss_mixed_phylo_mean.nwk"))
    f <- bf(y ~ x + (1 | study) + phylo(1 | species, tree = tree), sigma ~ 1)
    list(fit = function() drmTMB(f, family = stats::gaussian(), data = d, engine = "tmb"),
         data = d)
  },
  gauss_lss_sd_group = function(fixdir) {
    d <- read.csv(file.path(fixdir, "gauss_lss_sd_group.csv"), stringsAsFactors = FALSE)
    d$g <- as.integer(d$g)
    f <- bf(y ~ x + (1 | g), sigma ~ z, sd(g) ~ zg)
    list(fit = function() drmTMB(f, family = stats::gaussian(), data = d, engine = "tmb"),
         data = d)
  },
  gauss_lss_sd_phylo = function(fixdir) {
    d <- read.csv(file.path(fixdir, "gauss_lss_sd_phylo.csv"), stringsAsFactors = FALSE)
    tree <- ape::read.tree(file.path(fixdir, "gauss_lss_sd_phylo.nwk"))
    f <- bf(y ~ x + phylo(1 | species, tree = tree), sigma ~ x,
            sd(species, level = "phylogenetic") ~ xs)
    list(fit = function() drmTMB(f, family = stats::gaussian(), data = d, engine = "tmb"),
         data = d)
  },
  biv_q4_phylo_ml = function(fixdir) {
    d <- read.csv(file.path(fixdir, "biv_q4_phylo.csv"), stringsAsFactors = FALSE)
    tree <- ape::read.tree(file.path(fixdir, "biv_q4_phylo.nwk"))
    f <- bf(mu1 = y1 ~ x + phylo(1 | p | species, tree = tree),
            mu2 = y2 ~ x + phylo(1 | p | species, tree = tree),
            sigma1 = ~1, sigma2 = ~1, rho12 = ~1)
    list(fit = function() drmTMB(f, family = biv_gaussian(), data = d, REML = FALSE, engine = "tmb"),
         data = d)
  },
  biv_q4_phylo_reml = function(fixdir) {
    d <- read.csv(file.path(fixdir, "biv_q4_phylo.csv"), stringsAsFactors = FALSE)
    tree <- ape::read.tree(file.path(fixdir, "biv_q4_phylo.nwk"))
    f <- bf(mu1 = y1 ~ x + phylo(1 | p | species, tree = tree),
            mu2 = y2 ~ x + phylo(1 | p | species, tree = tree),
            sigma1 = ~1, sigma2 = ~1, rho12 = ~1)
    list(fit = function() drmTMB(f, family = biv_gaussian(), data = d, REML = TRUE, engine = "tmb"),
         data = d)
  },
  bernoulli_mixed = function(fixdir) {
    d <- read.csv(file.path(fixdir, "bernoulli_mixed.csv"), stringsAsFactors = FALSE)
    d$g <- as.integer(d$g)
    f <- bf(y ~ x + (1 | g))
    list(fit = function() drmTMB(f, family = stats::binomial(), data = d, engine = "tmb"),
         data = d)
  },
  poisson_mixed = function(fixdir) {
    d <- read.csv(file.path(fixdir, "poisson_mixed.csv"), stringsAsFactors = FALSE)
    d$g <- as.integer(d$g)
    f <- bf(y ~ x + (1 | g))
    list(fit = function() drmTMB(f, family = stats::poisson(), data = d, engine = "tmb"),
         data = d)
  },
  lognormal_locscale = function(fixdir) {
    d <- read.csv(file.path(fixdir, "lognormal_locscale.csv"), stringsAsFactors = FALSE)
    f <- bf(y ~ x, sigma ~ z)
    list(fit = function() drmTMB(f, family = drmTMB::lognormal(), data = d, engine = "tmb"),
         data = d)
  },
  meta_analysis_meta_V = function(fixdir) {
    d <- read.csv(file.path(fixdir, "meta_analysis.csv"), stringsAsFactors = FALSE)
    f <- bf(y ~ x + meta_V(V = v), sigma ~ 1)
    list(fit = function() drmTMB(f, family = stats::gaussian(), data = d, engine = "tmb"),
         data = d)
  },
  large_sparse_lss_p2000 = function(fixdir) {
    d <- read.csv(file.path(fixdir, "large_sparse_lss_p2000.csv"), stringsAsFactors = FALSE)
    tree <- ape::read.tree(file.path(fixdir, "large_sparse_lss_p2000.nwk"))
    f <- bf(y ~ x + phylo(1 | species, tree = tree), sigma ~ x,
            sd(species, level = "phylogenetic") ~ xs)
    list(fit = function() drmTMB(f, family = stats::gaussian(), data = d, engine = "tmb"),
         data = d)
  }
)

# ---- run --------------------------------------------------------------------

commit <- tryCatch(system2("git", c("rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE),
                    error = function(e) "unknown")
if (length(commit) == 0 || !nzchar(commit)) commit <- "unknown"

names_to_run <- if (!is.null(args$workflow)) args$workflow else names(workflows)
stopifnot(all(names_to_run %in% names(workflows)))

rows <- character(0)
header <- paste("engine", "workflow", "leg", "threads", "reps", "calls", "median_s", "min_s",
                 "converged", "loglik", "r_version", "blas_threads", "commit", sep = "\t")
rows <- c(rows, header)

for (wname in names_to_run) {
  cat(sprintf("== %s ==\n", wname))
  w <- workflows[[wname]](args$fixtures)
  cat("  warm-up (untimed) ... ")
  fit0 <- w$fit()
  unc0 <- tryCatch(force_sdreport(fit0), error = function(e) NULL)
  pr0 <- tryCatch(predict(fit0, newdata = w$data), error = function(e) NULL)
  cat("done\n")

  r_fit <- time_reps(w$fit, args$reps)
  conv <- fit_converged(r_fit$val); ll <- fit_loglik(r_fit$val)
  cat(sprintf("  fit:         median=%.6fs min=%.6fs calls~%d conv=%s loglik=%.4f\n",
              r_fit$median_s, r_fit$min_s, r_fit$calls, conv, ll))
  rows <- c(rows, paste("drmTMB", wname, "fit", args$threads, args$reps, r_fit$calls,
                         sprintf("%.6f", r_fit$median_s), sprintf("%.6f", r_fit$min_s),
                         conv, sprintf("%.6f", ll), R.version.string, args$threads, commit, sep = "\t"))

  fitref <- r_fit$val
  r_unc <- time_reps(function() force_sdreport(fitref), args$reps)
  cat(sprintf("  uncertainty: median=%.6fs min=%.6fs calls~%d (forced TMB::sdreport, not cached read)\n",
              r_unc$median_s, r_unc$min_s, r_unc$calls))
  rows <- c(rows, paste("drmTMB", wname, "uncertainty", args$threads, args$reps, r_unc$calls,
                         sprintf("%.6f", r_unc$median_s), sprintf("%.6f", r_unc$min_s),
                         conv, sprintf("%.6f", ll), R.version.string, args$threads, commit, sep = "\t"))

  r_pred <- time_reps(function() predict(fitref, newdata = w$data), args$reps)
  cat(sprintf("  predict:     median=%.6fs min=%.6fs calls~%d\n",
              r_pred$median_s, r_pred$min_s, r_pred$calls))
  rows <- c(rows, paste("drmTMB", wname, "predict", args$threads, args$reps, r_pred$calls,
                         sprintf("%.6f", r_pred$median_s), sprintf("%.6f", r_pred$min_s),
                         conv, sprintf("%.6f", ll), R.version.string, args$threads, commit, sep = "\t"))
}

dir.create(dirname(args$out), showWarnings = FALSE, recursive = TRUE)
writeLines(rows, args$out)
cat("wrote ", args$out, "\n", sep = "")
