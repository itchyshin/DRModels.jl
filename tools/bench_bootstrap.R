#!/usr/bin/env Rscript
#
# Head-to-head parametric bootstrap benchmark: drmTMB engine = "tmb" vs
# engine = "julia", same DGP / n / B / seed, driven through the SAME public
# entry point (drmTMB() + confint(method = "bootstrap")) so this is a
# same-target comparison across the bridge, not a bridge-vs-direct-call
# comparison. Julia-engine bootstrap intervals currently support only the
# Gaussian phylogenetic SD target (see confint.drmTMB_julia() in
# drmTMB/R/julia-bridge.R), so the DGP here is a phylogenetic Gaussian
# regression (the AVONET/Hackett example already used by drmTMB's own
# tools/benchmark-r-julia-bootstrap-refits.R), scaled down for a fast,
# honest pre-run test.
#
# Owned by: Curie (DRModels.jl performance lane), branch parity/se-axis.
# D-139: this script is meant to be run first at small B (pre-run test)
# before any larger B is committed to.

env_value <- function(name, default = "") {
  value <- Sys.getenv(name, unset = NA_character_)
  if (is.na(value) || !nzchar(value)) return(default)
  value
}
env_int <- function(name, default) {
  v <- suppressWarnings(as.integer(env_value(name, as.character(default))))
  if (length(v) != 1L || is.na(v)) default else v
}

B            <- env_int("DRM_BOOT_B", 5L)
n_species    <- env_int("DRM_BOOT_SPECIES", 100L)
seed         <- env_int("DRM_BOOT_SEED", 20260824L)
drmtmb_path  <- env_value("DRMTMB_PATH", normalizePath(file.path(dirname(getwd()), "drmTMB"), mustWork = FALSE))
drm_jl_path  <- env_value("DRM_JL_PATH", "")
out_tsv      <- env_value(
  "DRM_BOOT_OUT_TSV",
  file.path(getwd(), "docs", "dev-log", "evidence", "bootstrap-h2h.tsv")
)

Sys.setenv(
  OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1", VECLIB_MAXIMUM_THREADS = "1"
)
if (nzchar(drm_jl_path)) Sys.setenv(DRM_JL_PATH = drm_jl_path)

message(sprintf(
  "bench_bootstrap.R: B=%d species=%d seed=%d drmtmb_path=%s",
  B, n_species, seed, drmtmb_path
))

stopifnot(requireNamespace("ape", quietly = TRUE))
stopifnot(requireNamespace("JuliaCall", quietly = TRUE))

## Use the INSTALLED drmTMB, matching every other parity/benchmark script in this
## repo (tools/parity_fixture.R, parity_se.R, parity_classc.R, bench_fit_h2h.R).
##
## This previously called devtools::load_all(drmtmb_path), which loads drmTMB from
## the SOURCE TREE. On 2026-08-24 that silently measured version 0.6.0.9000 off a
## working tree sitting on branch claude/handover-freshness-0718 with 102
## uncommitted files -- while another lane was concurrently running
## devtools::test() against that same tree, so the code under measurement could
## change mid-run. The result was reported as a drmTMB-vs-DRModels.jl benchmark; it was
## nothing of the kind, and the printed banner said 0.6.0.9000 while the installed
## package was 0.7.0. A benchmark whose subject is a dirty checkout measures
## nothing reproducible.
suppressMessages(library(drmTMB))
message(sprintf("bench_bootstrap.R: using INSTALLED drmTMB %s from %s",
                as.character(utils::packageVersion("drmTMB")),
                find.package("drmTMB")))

# ---- DGP: AVONET/Hackett phylogenetic Gaussian regression (subset) --------

find_avonet_files <- function() {
  candidates <- list(
    list(
      data = file.path(dirname(getwd()), "pigauto", "avonet", "AVONET3_BirdTree.csv"),
      tree = file.path(dirname(getwd()), "pigauto", "avonet", "Stage2_Hackett_MCC_no_neg.tre")
    ),
    list(
      data = file.path(dirname(getwd()), "BACE", "dev", "testing_data", "AVONET.csv"),
      tree = file.path(dirname(getwd()), "BACE", "dev", "testing_data", "Hackett_tree.tre")
    )
  )
  for (cand in candidates) {
    if (file.exists(cand$data) && file.exists(cand$tree)) return(cand)
  }
  stop("AVONET and Hackett tree files not found in sibling checkouts.", call. = FALSE)
}

make_positive_branch_lengths <- function(tree, eps = 1e-8) {
  tree$edge.length <- pmax(tree$edge.length, eps)
  tree
}

make_avonet_phylo_data <- function(n_species) {
  paths <- find_avonet_files()
  avonet <- utils::read.csv(paths$data, check.names = FALSE)
  tree0 <- ape::read.tree(paths$tree)
  dat0 <- data.frame(
    species = gsub(" ", "_", avonet$Species3, fixed = TRUE),
    mass = avonet$Mass,
    hand_wing = avonet[["Hand-Wing.Index"]],
    beak = avonet[["Beak.Length_Culmen"]]
  )
  dat0 <- dat0[stats::complete.cases(dat0), , drop = FALSE]
  available <- tree0$tip.label[tree0$tip.label %in% dat0$species]
  if (n_species > length(available)) {
    stop(sprintf("Requested %d species, only %d available.", n_species, length(available)), call. = FALSE)
  }
  available <- available[seq_len(n_species)]
  tree <- make_positive_branch_lengths(ape::keep.tip(tree0, available))
  dat <- dat0[match(tree$tip.label, dat0$species), , drop = FALSE]
  dat$species <- factor(dat$species, levels = tree$tip.label)
  dat$log_mass <- log(dat$mass)
  dat$hand_wing_z <- as.numeric(scale(dat$hand_wing))
  dat$beak_z <- as.numeric(scale(dat$beak))
  list(data = dat, tree = tree, source = paths)
}

time_it <- function(expr_fn) {
  gc()
  t0 <- proc.time()[["elapsed"]]
  val <- expr_fn()
  list(value = val, elapsed = proc.time()[["elapsed"]] - t0)
}

av <- make_avonet_phylo_data(n_species)
tree <- av$tree
form <- bf(
  log_mass ~ hand_wing_z + beak_z + phylo(1 | species, tree = tree),
  sigma ~ 1
)
level <- 0.95

# ---- TMB engine -------------------------------------------------------

message("Fitting native TMB base model...")
base_tmb <- time_it(function() {
  drmTMB(form, family = stats::gaussian(), data = av$data, engine = "tmb",
         control = drm_control(keep_data = TRUE))
})
fit_tmb <- base_tmb$value
targets <- profile_targets(fit_tmb)
target <- grep("^sd:mu:phylo", targets$parm, value = TRUE)[[1L]]
message(sprintf("Target: %s", target))

message(sprintf("TMB bootstrap, B=%d ...", B))
boot_tmb <- time_it(function() {
  stats::confint(fit_tmb, parm = target, method = "bootstrap", R = B, seed = seed, parallel = "none")
})
ci_tmb <- boot_tmb$value

# ---- Julia engine -------------------------------------------------------

message("Julia startup (JuliaCall::julia_setup)...")
julia_startup <- time_it(function() JuliaCall::julia_setup())

message("Fitting Julia-engine base model (first call after startup)...")
base_julia <- time_it(function() {
  drmTMB(form, family = stats::gaussian(), data = av$data, engine = "julia")
})
fit_julia <- base_julia$value

message(sprintf("Julia bootstrap, B=%d ...", B))
boot_julia <- time_it(function() {
  stats::confint(fit_julia, parm = target, method = "bootstrap", R = B, seed = seed, threads = FALSE)
})
ci_julia <- boot_julia$value

targets_julia <- profile_targets(fit_julia)
estimate_tmb   <- targets$estimate[targets$parm == target][[1L]]
estimate_julia <- targets_julia$estimate[targets_julia$parm == target][[1L]]

# ---- Report --------------------------------------------------------------

cpu <- tryCatch(system("sysctl -n machdep.cpu.brand_string", intern = TRUE), error = function(e) NA_character_)

summary_row <- function(label, fit_elapsed, boot, boot_elapsed, estimate) {
  data.frame(
    label = label,
    B = B,
    n_species = n_species,
    rows = stats::nobs(fit_tmb),
    seed = seed,
    base_fit_s = fit_elapsed,
    bootstrap_elapsed_s = boot_elapsed,
    sec_per_refit = boot_elapsed / B,
    used = boot$bootstrap.n[[1L]],
    failed = boot$bootstrap.failed[[1L]],
    lower = boot$lower[[1L]],
    upper = boot$upper[[1L]],
    estimate = estimate,
    status = boot$conf.status[[1L]],
    stringsAsFactors = FALSE
  )
}

row_tmb   <- summary_row("tmb",   base_tmb$elapsed,   ci_tmb,   boot_tmb$elapsed,   estimate_tmb)
row_julia <- summary_row("julia", base_julia$elapsed, ci_julia, boot_julia$elapsed, estimate_julia)

cat("\n==== bench_bootstrap.R summary ====\n")
cat(sprintf("machine: %s\n", cpu))
cat(sprintf("R: %s | drmTMB: %s | Julia: julia --version (see below) | JuliaCall: %s\n",
            R.version.string, as.character(utils::packageVersion("drmTMB")),
            as.character(utils::packageVersion("JuliaCall"))))
cat(sprintf("target: %s | n_species: %d | rows: %d | B: %d | seed: %d\n",
            target, n_species, stats::nobs(fit_tmb), B, seed))
cat(sprintf("julia_startup_s: %.3f\n", julia_startup$elapsed))
cat("\n-- TMB --\n"); print(row_tmb)
cat("\n-- Julia --\n"); print(row_julia)

ratio <- row_tmb$sec_per_refit / row_julia$sec_per_refit
cat(sprintf("\nsec_per_refit ratio (tmb / julia): %.3f\n", ratio))
cat(sprintf("interval overlap: tmb [%.5f, %.5f] vs julia [%.5f, %.5f]\n",
            row_tmb$lower, row_tmb$upper, row_julia$lower, row_julia$upper))

dir.create(dirname(out_tsv), recursive = TRUE, showWarnings = FALSE)
out <- rbind(row_tmb, row_julia)
out$julia_startup_s <- julia_startup$elapsed
out$machine <- cpu
out$r_version <- R.version.string
out$drmtmb_version <- as.character(utils::packageVersion("drmTMB"))
out$juliacall_version <- as.character(utils::packageVersion("JuliaCall"))
out$timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
write_header <- !file.exists(out_tsv)
utils::write.table(out, out_tsv, sep = "\t", row.names = FALSE, col.names = write_header,
                    append = !write_header, quote = FALSE)
cat(sprintf("\nAppended TSV rows to %s\n", out_tsv))
