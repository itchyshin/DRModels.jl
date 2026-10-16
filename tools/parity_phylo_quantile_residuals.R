# parity_phylo_quantile_residuals.R — drmTMB vs DRModels.jl quantile residuals
# for a Gaussian `phylo(1 | g)` mean model (#995, PR #1078).
#
# WHY THIS EXISTS. PR #1078 makes DRModels.jl's `residuals(fit; type = :quantile)`
# the population-level PIT, (y − Xβ̂) / √(σ̂_e² + σ̂_b² C_ii). Whether that now
# disagrees with drmTMB's `residuals(fit, type = "quantile")` on the same fit had
# not been measured. This script measures it and separates the two possible
# causes of a difference:
#
#   engine      the two fits land on different estimates or modes;
#   convention  the two engines define the residual differently.
#
# For each case it records, besides the headline max |q_julia − q_drmTMB|:
#   * the Julia CONDITIONAL residual (y − fitted) / σ̂_e against drmTMB's
#     residual (an engine-level comparison under drmTMB's convention);
#   * drmTMB's own estimates under the POPULATION-level convention,
#     (y − Xβ̂) / √(σ̂² + σ̂_phylo²), against the Julia residual (an
#     engine-level comparison under DRModels.jl's convention; drmTMB's tree is
#     normalised to tip variance 1, so C_ii = 1 on that side);
#   * that drmTMB's residual is exactly (y − fitted(fit)) / sigma(fit);
#   * logLik, coefficients, σ̂, σ̂_phylo (drmTMB's correlation scale) and both
#     convergence flags.
#
# drmTMB requires an ultrametric tree and normalises it to height 1
# (`drm_phylo_augmented_precision(..., correlation = TRUE)`); DRModels.jl uses
# the branch lengths as given. On an ultrametric tree of height h the two
# models are the same model (σ_phylo,drmTMB² = h · σ_b,raw²), so the fixture's
# raw-height tree is run as well as its height-1 copy.
#
# THE LICENSE BOUNDARY. This script only CALLS drmTMB and records its outputs;
# no drmTMB source is copied into DRModels.jl.
#
# Usage (from the DRModels.jl root, drmTMB installed, julia on PATH):
#
#   Rscript tools/parity_phylo_quantile_residuals.R
#
# Environment: DRM_JL_PATH (default "."), JULIA (default "julia"),
# PQR_OUT (default "docs/dev-log/evidence/parity-phylo-quantile-residuals.tsv"),
# DRMTMB_GIT_SHA (the drmTMB commit the installed build came from, if known;
# the code hash identifies the build either way).

suppressMessages({
  library(drmTMB)
  library(ape)
})

drm_path <- Sys.getenv("DRM_JL_PATH", ".")
julia <- Sys.getenv("JULIA", "julia")
out_path <- Sys.getenv("PQR_OUT", "docs/dev-log/evidence/parity-phylo-quantile-residuals.tsv")
source(file.path(drm_path, "tools", "drmtmb_provenance_lib.R"))

work <- tempfile("pqr-")
dir.create(work)
fmt17 <- function(v) sprintf("%.17g", v)

write_case <- function(case, newick, dat) {
  writeLines(newick, file.path(work, paste0(case, ".tree.nwk")))
  utils::write.csv(data.frame(y = fmt17(dat$y), x = fmt17(dat$x), species = dat$species),
                   file.path(work, paste0(case, ".data.csv")), row.names = FALSE, quote = FALSE)
}

# --- cases -------------------------------------------------------------------
fx <- file.path(drm_path, "test", "parity", "phylo-mean", "gaussian-phylo-mean")
fx_dat <- utils::read.csv(file.path(fx, "data.csv"), stringsAsFactors = FALSE)
cases <- list(
  list(id = "fixture_h1",
       note = "test/parity gaussian-phylo-mean data (seed 404), tree_h1.0.newick",
       newick = trimws(paste(readLines(file.path(fx, "tree_h1.0.newick")), collapse = "")),
       dat = fx_dat),
  list(id = "fixture_hraw",
       note = "same data, raw-height tree.newick (h = 2.474)",
       newick = trimws(paste(readLines(file.path(fx, "tree.newick")), collapse = "")),
       dat = fx_dat)
)

# A strong-signal case: σ_phylo / σ_e = 5 on drmTMB's scale, 120 tips, 3 rows
# per tip, raw coalescent height.
local({
  set.seed(995)
  tr <- ape::rcoal(120)
  tr$tip.label <- paste0("sp_", seq_len(120))
  nwk <- ape::write.tree(tr)
  tr <- ape::read.tree(text = nwk)                   # what both engines read
  A <- ape::vcv(tr) / max(diag(ape::vcv(tr)))         # tip variance 1 (drmTMB's scale)
  u <- 2.0 * as.vector(t(chol(A)) %*% rnorm(120))
  sp <- rep(tr$tip.label, each = 3)
  x <- rnorm(length(sp))
  y <- 0.3 + 0.6 * x + u[match(sp, tr$tip.label)] + 0.4 * rnorm(length(sp))
  cases[[length(cases) + 1L]] <<- list(
    id = "strong_signal",
    note = "seed 995, rcoal(120), 3 rows/tip, sigma_phylo = 2, sigma = 0.4",
    newick = nwk, dat = data.frame(y = y, x = x, species = sp, stringsAsFactors = FALSE))
})

for (cs in cases) write_case(cs$id, cs$newick, cs$dat)

# --- DRModels.jl --------------------------------------------------------------
jl_script <- file.path(drm_path, "tools", "parity_phylo_quantile_residuals.jl")
status <- system2(julia, c(paste0("--project=", drm_path), jl_script, work,
                           vapply(cases, `[[`, "", "id")))
if (!identical(status, 0L)) stop("the Julia half failed (exit ", status, ")")

read_toml_flat <- function(path) {
  kv <- grep("=", readLines(path), value = TRUE)
  k <- trimws(sub("=.*$", "", kv))
  v <- trimws(sub("^[^=]*=", "", kv))
  stats::setNames(as.list(gsub('^"|"$', "", v)), k)
}

# --- drmTMB, and the comparison ---------------------------------------------
prov <- drmtmb_provenance()
rows <- list()
for (cs in cases) {
  tree <- ape::read.tree(file = file.path(work, paste0(cs$id, ".tree.nwk")))
  dat <- utils::read.csv(file.path(work, paste0(cs$id, ".data.csv")), stringsAsFactors = FALSE)
  fit <- drmTMB(bf(y ~ x + phylo(1 | species, tree = tree), sigma ~ 1),
                family = gaussian(), data = dat)
  q_r <- as.numeric(residuals(fit, type = "quantile"))
  fitted_r <- as.numeric(fitted(fit))
  sig_r <- as.numeric(sigma(fit))
  b_r <- as.numeric(fit$coefficients$mu)
  sdp_r <- as.numeric(fit$sdpars$mu[["phylo(1 | species)"]])
  xb_r <- b_r[1] + b_r[2] * dat$x
  q_r_marg <- (dat$y - xb_r) / sqrt(sig_r^2 + sdp_r^2)

  jc <- utils::read.csv(file.path(work, paste0(cs$id, ".julia.csv")))
  jt <- read_toml_flat(file.path(work, paste0(cs$id, ".julia.toml")))
  num <- function(k) as.numeric(jt[[k]])

  rows[[length(rows) + 1L]] <- data.frame(
    case = cs$id, note = cs$note, n = nrow(dat), n_tip = length(tree$tip.label),
    tree_height = num("tree_height"),
    max_abs_q_julia_vs_drmtmb = max(abs(jc$q_julia - q_r)),
    max_abs_qcond_julia_vs_drmtmb = max(abs(jc$q_cond - q_r)),
    max_abs_q_julia_vs_drmtmb_marginal = max(abs(jc$q_julia - q_r_marg)),
    max_abs_drmtmb_q_vs_its_pearson = max(abs(q_r - (dat$y - fitted_r) / sig_r)),
    max_abs_fitted_diff = max(abs(jc$fitted - fitted_r)),
    max_abs_xbeta_diff = max(abs(jc$mu_fixed - xb_r)),
    max_abs_q_gh32_vs_closed = max(abs(jc$q_gh32 - jc$q_julia)),
    cor_q_julia_drmtmb = stats::cor(jc$q_julia, q_r),
    sd_q_julia = stats::sd(jc$q_julia), sd_q_drmtmb = stats::sd(q_r),
    sd_pit_expected_under_model = sqrt(num("pit_expected_sample_var")),
    loglik_julia = num("loglik"), loglik_drmtmb = as.numeric(logLik(fit)),
    beta_julia = paste(fmt17(c(num("beta0"), num("beta1"))), collapse = ";"),
    beta_drmtmb = paste(fmt17(b_r), collapse = ";"),
    sigma_julia = num("sigma_e"), sigma_drmtmb = unique(sig_r)[1],
    sd_phylo_julia_drmtmb_scale = num("sd_phylo_drmtmb_scale"), sd_phylo_drmtmb = sdp_r,
    ratio_sdphylo_sigma = sdp_r / unique(sig_r)[1],
    converged_julia = jt[["converged"]],
    converged_drmtmb_opt_code = fit$opt$convergence,
    is_converged_drmtmb = isTRUE(is_converged(fit)),
    pdHess_drmtmb = isTRUE(fit$sdr$pdHess),
    drmodels_commit = jt[["drmodels_commit"]], julia_version = jt[["julia_version"]],
    drmtmb_version = prov$version, drmtmb_code_hash = prov$code_hash,
    drmtmb_git_sha = Sys.getenv("DRMTMB_GIT_SHA", "unrecorded"),
    r_version = paste(R.version$major, R.version$minor, sep = "."),
    stringsAsFactors = FALSE)
}
res <- do.call(rbind, rows)
utils::write.table(res, out_path, sep = "\t", row.names = FALSE, quote = FALSE)

for (i in seq_len(nrow(res))) {
  r <- res[i, ]
  cat(sprintf(paste0("%-14s headline %.3e | engine (conditional) %.3e | engine (population) %.3e | ",
                     "logLik %.8f / %.8f | conv julia=%s drmTMB code=%d pdHess=%s\n"),
              r$case, r$max_abs_q_julia_vs_drmtmb, r$max_abs_qcond_julia_vs_drmtmb,
              r$max_abs_q_julia_vs_drmtmb_marginal, r$loglik_julia, r$loglik_drmtmb,
              r$converged_julia, r$converged_drmtmb_opt_code, r$pdHess_drmtmb))
}
cat("wrote ", out_path, "\n", sep = "")
