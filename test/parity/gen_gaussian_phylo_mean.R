## gen_gaussian_phylo_mean.R -- generate the gaussian_phylo_mean fixture.
##
## Maintainer machine with local R + installed drmTMB. Writes generated
## data/numbers only into test/parity/phylo-mean/gaussian-phylo-mean/.
## Never edits the shared drmTMB checkout. Never vendors drmTMB source.
## Do not edit gen_fixtures.R / runparity.jl.
##
## #483: the previous cell (seed 111, iid "phylo" noise mislabelled onto tree
## tips, n_each=1) was UNIDENTIFIABLE -- the phylo variance component was
## boundary-adjacent (estimate ~6.7e-06) and the profiled nll moved by <1e-4
## across twenty orders of magnitude of log_sd_phylo. A comparison at that
## seed would pass silently on a parameter neither engine can estimate.
##
## This generator instead simulates a GENUINELY phylogenetically-correlated
## random effect (u = chol(vcv(tree, corr = TRUE)) %*% rnorm(n_tip, 0, sd_phylo),
## not iid noise indexed by species) with real signal (sd_phylo_true = 1.5 vs.
## sd_resid = 0.3) and n_each = 4 replicate observations per tip, so there is
## enough information to move the profiled likelihood. Seed 404 was selected
## BY MEASURING: see the profiled-nll sweep below and nll_profile.csv, which
## show real curvature at the fitted estimate, not a flat plateau.

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

toml_string <- function(x) {
  paste0('"', gsub('"', '\\"', as.character(x), fixed = TRUE), '"')
}

toml_num <- function(x) {
  if (!is.finite(x)) stop("cannot write non-finite TOML number: ", x)
  format(as.numeric(x), digits = 17, scientific = TRUE, trim = TRUE)
}

toml_bool <- function(x) {
  if (isTRUE(x)) return("true")
  if (identical(x, FALSE)) return("false")
  "\"NA\""
}

flat_coef <- function(fit) {
  cf <- coef(fit)
  out <- numeric()
  for (param in names(cf)) {
    vals <- as.numeric(cf[[param]])
    nm <- names(cf[[param]])
    if (is.null(nm)) nm <- as.character(seq_along(vals))
    names(vals) <- paste0(param, "_", nm)
    out <- c(out, vals)
  }
  names(out) <- sub(":", "_", names(out), fixed = TRUE)
  out
}

extract_pdHess <- function(fit) {
  tryCatch({
    if (!is.null(fit$sdr) && !is.null(fit$sdr$pdHess)) return(isTRUE(fit$sdr$pdHess))
    if (!is.null(fit$sdreport) && !is.null(fit$sdreport$pdHess)) return(isTRUE(fit$sdreport$pdHess))
    NA
  }, error = function(e) NA)
}

## The row's DEFINING quantity: the fitted phylogenetic SD. Read from the
## sdreport's FINAL fixed-parameter vector (matches `coef(fit)`) -- NOT
## `fit$obj$par`, which holds the fitter's STARTING values and never updates
## post-optimisation (verified 2026-08-24, reconfirmed while picking this
## seed: `obj$par["log_sd_phylo"]` silently holds the START value, off by
## more than a log-unit from the actual MLE for this cell). This is on
## drmTMB's CORRELATION-SCALE convention: drmTMB standardises the
## phylogenetic covariance via `ape::vcv(tree, corr = TRUE)` internally, so
## the value does not depend on the tree's raw branch-length height.
## Cross-checked against the public `fit$sdpars$mu` field, which must agree
## exactly.
extract_sd_phylo <- function(fit, term_label) {
  nm <- names(fit$sdr$par.fixed)
  idx <- which(nm == "log_sd_phylo")
  if (length(idx) != 1) {
    stop("expected exactly one 'log_sd_phylo' fixed parameter, found ", length(idx))
  }
  sd_from_sdr <- as.numeric(exp(fit$sdr$par.fixed[idx]))
  sd_from_sdpars <- tryCatch(as.numeric(fit$sdpars$mu[[term_label]]), error = function(e) NA_real_)
  if (is.finite(sd_from_sdpars) &&
      !isTRUE(all.equal(sd_from_sdr, sd_from_sdpars, tolerance = 1e-8))) {
    stop("sd_phylo mismatch: sdr$par.fixed gives ", sd_from_sdr,
         " but sdpars$mu gives ", sd_from_sdpars)
  }
  sd_from_sdr
}

## log_sd_phylo's fitted value and Wald SE from sdreport -- a large SE
## relative to 1 log-unit is itself a symptom of a boundary-adjacent /
## weakly-identified estimate, independent of the nll sweep below.
extract_log_sd_phylo_se <- function(fit) {
  idx <- which(names(fit$sdr$par.fixed) == "log_sd_phylo")
  list(
    est = as.numeric(fit$sdr$par.fixed[idx]),
    se = as.numeric(sqrt(diag(fit$sdr$cov.fixed))[idx])
  )
}

## Maximum root-to-tip path length -- the tip variance the tree implies when
## DRModels.jl's `re_sd()` (RAW branch-length scale) is compared to drmTMB's
## CORRELATION-scale `sd_phylo`: species_corr_scale == raw * sqrt(tree_height).
tree_height <- function(tree) {
  max(ape::node.depth.edgelength(tree)[seq_along(tree$tip.label)])
}

extract_interval_status <- function(fit) {
  ci <- tryCatch(confint(fit, method = "wald"), error = function(e) e)
  if (inherits(ci, "error")) {
    return(paste0("wald_unavailable:", conditionMessage(ci)))
  }
  if (is.null(ci)) return("wald_unavailable")
  nums <- suppressWarnings(as.numeric(as.matrix(ci)))
  if (length(nums) && all(is.finite(nums))) return("wald_finite")
  "wald_unavailable"
}

## Genuinely phylogenetically-correlated random effect: u = L %*% rnorm(),
## L = chol(vcv(tree, corr = TRUE)) -- NOT iid noise indexed by species
## (that was the retired seed-111 cell's bug: it drew `rnorm(n, 0, 0.45)`
## per species with no tree structure at all, so there was no real
## phylogenetic signal in the data for either engine to recover).
simulate_phylo_signal <- function(seed, n_tip, n_each, sd_phylo_true, sd_resid) {
  set.seed(as.integer(seed))
  tree <- ape::rcoal(as.integer(n_tip))
  tree$tip.label <- paste0("sp_", seq_len(n_tip))
  A <- ape::vcv(tree, corr = TRUE)
  L <- t(chol(A))
  u <- as.numeric(L %*% stats::rnorm(n_tip, 0, sd_phylo_true))
  tip <- rep(seq_len(n_tip), each = as.integer(n_each))
  n <- length(tip)
  x <- seq(-1, 1, length.out = n)
  dat <- data.frame(
    y = 0.4 + 0.7 * x + u[tip] + stats::rnorm(n, 0, sd_resid),
    x = x,
    species = factor(tree$tip.label[tip], levels = tree$tip.label)
  )
  list(data = dat, tree = tree, seed = as.integer(seed),
       n_tip = as.integer(n_tip), n_each = as.integer(n_each),
       sd_phylo_true = sd_phylo_true, sd_resid_true = sd_resid, with_x = TRUE)
}

fit_cell <- function(cell) {
  tree <- cell$tree
  dat <- cell$data
  form <- drmTMB::bf(
    y ~ x + phylo(1 | species, tree = tree),
    sigma ~ 1
  )
  drmTMB::drmTMB(form, family = stats::gaussian(), data = dat, engine = "tmb")
}

formula_text <- function() {
  "y ~ x + phylo(1 | species); sigma ~ 1"
}

r_call_text <- function() {
  "drmTMB(bf(y ~ x + phylo(1 | species, tree = tree), sigma ~ 1), family = gaussian(), data = dat, engine = \"tmb\")"
}

## Profiled negative log-likelihood over log_sd_phylo, holding the other
## FINAL fixed-effect estimates fixed (sdr$par.fixed, not obj$par). TMB's
## obj$fn re-optimises the random effects (Laplace) internally for each
## call, so this is a proper profile slice through the fitted surface, and
## its curvature at the MLE is the seed-selection criterion #483 asked for.
profile_nll <- function(fit, grid) {
  idx <- which(names(fit$sdr$par.fixed) == "log_sd_phylo")
  par0 <- as.numeric(fit$sdr$par.fixed)
  names(par0) <- names(fit$sdr$par.fixed)
  nll0 <- fit$obj$fn(par0)
  nlls <- vapply(grid, function(v) {
    p <- par0
    p[idx] <- v
    fit$obj$fn(p)
  }, numeric(1))
  data.frame(log_sd_phylo = grid, nll = nlls, delta = nlls - nll0)
}

write_fixture <- function(cell, fit, out_dir) {
  if (dir.exists(out_dir)) unlink(out_dir, recursive = TRUE)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  utils::write.csv(cell$data, file.path(out_dir, "data.csv"), row.names = FALSE)
  ape::write.tree(cell$tree, file.path(out_dir, "tree.newick"))

  coefs <- flat_coef(fit)
  ll <- as.numeric(stats::logLik(fit))
  n <- nrow(cell$data)
  converged <- isTRUE(identical(fit$opt$convergence, 0L) || identical(fit$opt$convergence, 0))
  pd <- extract_pdHess(fit)
  interval_status <- extract_interval_status(fit)
  sd_phylo_corr <- extract_sd_phylo(fit, "phylo(1 | species)")
  log_sd <- extract_log_sd_phylo_se(fit)
  h <- tree_height(cell$tree)

  ## Identifiability evidence: a local sweep (curvature at the estimate) and
  ## a wide absolute sweep matching the retired seed-111 report's range
  ## ([-30,-10]) to show the minimum is NOT a boundary artifact -- pushing
  ## to that plateau costs real likelihood here, unlike at seed 111.
  local_grid <- seq(log_sd$est - 3, log_sd$est + 3, by = 0.25)
  wide_grid <- seq(-30, -10, by = 2)
  prof_local <- profile_nll(fit, local_grid)
  prof_wide <- profile_nll(fit, wide_grid)
  utils::write.csv(prof_local, file.path(out_dir, "nll_profile.csv"), row.names = FALSE)
  delta_at_1 <- prof_local$delta[which.min(abs(prof_local$log_sd_phylo - (log_sd$est + 1)))]
  delta_at_boundary <- min(prof_wide$delta)
  boundary_adjacent <- sd_phylo_corr < 1e-3 || log_sd$se > 3

  ## Multi-height round trip (#483 requirement: kept, now on an identifiable
  ## cell). Same data, tree rescaled to three absolute heights, refit
  ## natively each time. drmTMB's value must be height-invariant; DRModels.jl's
  ## raw re_sd must equal it after x sqrt(height).
  heights <- c(0.5, 1.0, 3.0)
  height_rows <- lapply(heights, function(hh) {
    tr_h <- cell$tree
    tr_h$edge.length <- cell$tree$edge.length * (hh / h)
    ape::write.tree(tr_h, file.path(out_dir, sprintf("tree_h%s.newick", format(hh, nsmall = 1))))
    fit_h <- drmTMB::drmTMB(
      drmTMB::bf(y ~ x + phylo(1 | species, tree = tr_h), sigma ~ 1),
      family = stats::gaussian(), data = cell$data, engine = "tmb"
    )
    list(
      height = hh,
      measured_height = tree_height(tr_h),
      sd_phylo_corr = extract_sd_phylo(fit_h, "phylo(1 | species)"),
      converged = isTRUE(identical(fit_h$opt$convergence, 0L) || identical(fit_h$opt$convergence, 0)),
      pdHess = extract_pdHess(fit_h)
    )
  })

  con <- file(file.path(out_dir, "heights.toml"), "w")
  on.exit(close(con), add = TRUE)
  writeLines(c(
    "# Multi-height round trip (#483): SAME data, tree rescaled to three",
    "# absolute heights, refit natively in drmTMB each time. drmTMB's",
    "# sd_phylo is on the CORRELATION scale (ape::vcv(tree, corr = TRUE)),",
    "# so it should be height-invariant here -- that invariance IS the",
    "# scale-convention evidence. DRModels.jl's re_sd(fit)[:species] is on the",
    "# RAW branch-length scale; test_parity_gaussian_phylo_mean.jl compares",
    "# raw * sqrt(measured_height) against sd_phylo_corr at each row below."
  ), con)
  for (row in height_rows) {
    writeLines(c(
      sprintf("[[height]]"),
      sprintf("target_height = %s", toml_num(row$height)),
      sprintf("measured_height = %s", toml_num(row$measured_height)),
      sprintf("sd_phylo_corr = %s", toml_num(row$sd_phylo_corr)),
      sprintf("converged = %s", toml_bool(row$converged)),
      sprintf("pdHess = %s", toml_bool(row$pdHess)),
      ""
    ), con)
  }
  close(con)
  on.exit()

  ## atol_re_sd re-derivation (#483): the retired 1e-4 was sized against an
  ## unidentifiable ~6.7e-06 estimate and means nothing here. NOT re-derived
  ## from the multi-height round trip's own spread: on the CORRELATION scale
  ## the three refits solve an objective function that is bit-identical
  ## regardless of the tree's raw height (ape::vcv(tree, corr = TRUE)
  ## standardises height away), so that spread (~1e-15, machine epsilon)
  ## measures floating-point reproducibility of the SAME optimum, not
  ## cross-engine numerical agreement -- using it here would size a
  ## meaningless, far-too-tight tolerance by the same mistake in a new
  ## disguise.
  ##
  ## Instead: two engines solving the identical deterministic MLE problem
  ## for the SAME data should agree to within a small fraction of the
  ## parameter's OWN statistical uncertainty, not to within that
  ## uncertainty itself (that would be testing recovery of the true value
  ## against sampling noise, not cross-engine numerical agreement). Take
  ## 1% of the Wald SE on the log scale, propagated to the SD scale by the
  ## delta method (d(sd)/d(log_sd) = sd):
  ##   atol_re_sd = 0.01 * log_sd_phylo_se * sd_phylo_corr
  ## This is set BEFORE looking at DRModels.jl's answer, from drmTMB's own fit
  ## alone -- so a subsequent disagreement is not being tolerance-shopped.
  atol_re_sd <- 0.01 * log_sd$se * sd_phylo_corr

  fit_con <- file(file.path(out_dir, "expected.toml"), "w")
  on.exit(close(fit_con), add = TRUE)
  writeLines(c(
    "[fit]",
    paste0("family = ", toml_string("gaussian")),
    paste0("formula = ", toml_string(formula_text())),
    paste0("method = ", toml_string("ML")),
    paste0("engine = ", toml_string("tmb")),
    paste0("loglik = ", toml_num(ll)),
    paste0("n = ", as.integer(n)),
    "",
    "[coef]"
  ), fit_con)
  for (name in sort(names(coefs))) {
    writeLines(paste0(toml_string(name), " = ", toml_num(coefs[[name]])), fit_con)
  }
  writeLines(c(
    "",
    "# The row's DEFINING quantity: the fitted phylogenetic SD, on drmTMB's",
    "# CORRELATION-SCALE convention (ape::vcv(tree, corr = TRUE) standardises the",
    "# tip variance to 1, independent of the tree's raw branch-length height).",
    "# DRModels.jl's re_sd(fit)[:species] is on the RAW branch-length scale instead",
    "# (tip variance = tree_height); the two are related by",
    "#   species_corr_scale == re_sd(fit)[:species] * sqrt(tree_height)",
    "# tree_height is recorded in expected.meta.toml. This cell's estimate is",
    "# NOT boundary-adjacent -- see expected.meta.toml [identifiability] and",
    "# nll_profile.csv for the curvature evidence #483 asked for.",
    "[re_sd]"
  ), fit_con)
  writeLines(paste0("species_corr_scale = ", toml_num(sd_phylo_corr)), fit_con)
  writeLines(c(
    "",
    "[status]",
    paste0("converged = ", toml_bool(converged)),
    paste0("pdHess = ", toml_bool(pd)),
    paste0("interval_status = ", toml_string(interval_status)),
    "",
    "[tol]",
    "atol_loglik = 1e-6",
    "atol_coef = 1e-5",
    "rtol_coef = 1e-5",
    "# Re-derived for #483 (NOT the retired seed-111 1e-4, which was sized",
    "# against an unidentifiable ~6.7e-06 estimate and meant nothing).",
    "# = 1% of log_sd_phylo's Wald SE, delta-method-mapped to the SD scale",
    "# (0.01 * log_sd_phylo_se * sd_phylo_corr); see expected.meta.toml",
    "# [identifiability] for the SE and the full derivation note.",
    paste0("atol_re_sd = ", toml_num(atol_re_sd))
  ), fit_con)

  note <- paste(
    "Generated outputs only; no drmTMB source vendored.",
    "Workflow G fixtures remain 0.6.0 / ML / no tree.",
    "This cell is 0.7.0 ML + tree, outside the fixtures/ glob.",
    "Not a TSV supported flip. Not the last fixture-gap.",
    "ML, univariate, sigma ~ 1. Tight ML tols (not #434 atol_loglik=6).",
    "#483: reseeded off the retired seed-111 Route A cell, whose 'phylo'",
    "effect was actually IID NOISE with no tree structure (rnorm() per",
    "species, not chol(vcv(tree)) %*% rnorm()) and 1 obs/tip -- undetectable",
    "and unidentifiable (profiled nll moved < 1e-4 over 20 orders of",
    "magnitude of log_sd_phylo). This cell instead simulates a genuinely",
    "tree-correlated random effect with real signal and n_each=4 replicate",
    "obs/tip; see [identifiability] below and nll_profile.csv for the",
    "curvature evidence that this seed was chosen by measuring, not",
    "guessing. Multi-height round trip (#483, kept): heights.toml records",
    "three native drmTMB refits (heights 0.5/1.0/3.0, same data) confirming",
    "sd_phylo's correlation-scale height-invariance on THIS identifiable",
    "cell (the retired cell only established the scale convention; it could",
    "not evidence numeric phylo-SD agreement)."
  )

  prov <- tryCatch({
    out <- system2("Rscript", c(shQuote(file.path(repo_root(), "tools", "drmtmb_provenance.R")), "--toml"),
                    stdout = TRUE, stderr = FALSE)
    vals <- list()
    for (line in out) {
      m <- regmatches(line, regexec('^(\\w+)\\s*=\\s*"(.*)"$', line))[[1]]
      if (length(m) == 3) vals[[m[2]]] <- m[3]
    }
    vals
  }, error = function(e) list())
  drmtmb_built <- if (!is.null(prov$drmtmb_built)) prov$drmtmb_built else as.character(utils::packageDescription("drmTMB")$Built)
  drmtmb_code_hash <- if (!is.null(prov$drmtmb_code_hash)) prov$drmtmb_code_hash else NA_character_

  meta_lines <- c(
    paste0("drmtmb_version = ", toml_string(as.character(utils::packageVersion("drmTMB")))),
    paste0("drmtmb_built = ", toml_string(drmtmb_built)),
    if (!is.na(drmtmb_code_hash)) paste0("drmtmb_code_hash = ", toml_string(drmtmb_code_hash)),
    paste0("generated_on = ", toml_string(as.character(Sys.Date()))),
    paste0("r_call = ", toml_string(r_call_text())),
    paste0("seed = ", as.integer(cell$seed)),
    paste0("n_tip = ", as.integer(cell$n_tip)),
    paste0("n_each = ", as.integer(cell$n_each)),
    paste0("sd_phylo_true = ", toml_num(cell$sd_phylo_true)),
    paste0("sd_resid_true = ", toml_num(cell$sd_resid_true)),
    paste0("with_x = ", toml_bool(cell$with_x)),
    paste0("tree_height = ", toml_num(h)),
    "",
    "[identifiability]",
    paste0("boundary_adjacent = ", toml_bool(boundary_adjacent)),
    "# log_sd_phylo's fitted value and Wald SE (sdr$par.fixed / sdr$cov.fixed).",
    "# A boundary-adjacent estimate would show se >> 1 log-unit; it does not.",
    paste0("log_sd_phylo_est = ", toml_num(log_sd$est)),
    paste0("log_sd_phylo_se = ", toml_num(log_sd$se)),
    "# Curvature evidence (the #483 seed-selection criterion): cost, in nll",
    "# units, of moving log_sd_phylo 1 unit off the estimate (full grid in",
    "# nll_profile.csv), versus the cost of pushing all the way out to the",
    "# retired fixture's [-30,-10] plateau (proves the minimum is interior,",
    "# not a boundary artifact).",
    paste0("nll_delta_at_plus_1_log_unit = ", toml_num(delta_at_1)),
    paste0("nll_delta_at_boundary_plateau = ", toml_num(delta_at_boundary)),
    "nll_profile_file = \"nll_profile.csv\"",
    "heights_file = \"heights.toml\"",
    "",
    paste0("note = ", toml_string(note))
  )
  writeLines(meta_lines[!is.na(meta_lines)], file.path(out_dir, "expected.meta.toml"))
}

## Seed chosen BY MEASURING (#483): see candidate scan in the PR description /
## commit message. seed=404, n_tip=18, n_each=4, sd_phylo_true=1.5,
## sd_resid=0.3 converges cleanly (pdHess = TRUE) with a profiled nll that
## moves by double digits within +-1.5 log-units of the estimate -- not the
## <1e-4-over-20-orders-of-magnitude flatness at the retired seed 111.
cell <- simulate_phylo_signal(seed = 404L, n_tip = 18L, n_each = 4L,
                               sd_phylo_true = 1.5, sd_resid = 0.3)
fit <- fit_cell(cell)
conv <- fit$opt$convergence
ll <- as.numeric(stats::logLik(fit))
message(sprintf("seed=404 n_tip=18 n_each=4 convergence=%s logLik=%s", conv, ll))
if (!(identical(conv, 0L) || identical(conv, 0))) {
  stop("HANDS TO Codex: the measured seed (404) no longer converges natively -- do not reseed blindly.")
}

out_dir <- file.path(repo_root(), "test", "parity", "phylo-mean", "gaussian-phylo-mean")
write_fixture(cell, fit, out_dir)
message("wrote ", out_dir)
message("coef names: ", paste(names(flat_coef(fit)), collapse = ", "))
invisible(NULL)
