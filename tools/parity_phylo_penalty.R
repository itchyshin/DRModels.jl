# parity_phylo_penalty.R — penalized-MAP phylo parity: drmTMB's
# `drm_phylo_penalty()` against DRModels.jl's, on identical data and an identical tree.
#
# `penalty` is not a bridge argument, so this cannot ride the per-cell
# `parity_fixture.R` harness; like `parity_associate.R` it drives DRModels.jl directly
# through JuliaCall and compares one cell at a time.
#
# WHAT IS COMPARED, and why all three:
#   sd_phylo — the estimate the penalty is supposed to move. On its own it could
#              agree for the wrong reason (e.g. both at a boundary).
#   penalty  — the value of the added term. Pins the two implementations to the
#              same arithmetic, not merely to the same answer.
#   logLik   — must be the UNPENALIZED data log-likelihood on both sides. If one
#              side folded the penalty into logLik, sd_phylo could still match
#              while the reported fit statistics silently diverged.
#
# THE ML BASELINE CELL IS NOT DECORATION. It runs the same model with no penalty
# and must pass first. If the fixture, the tree, or the species↔leaf alignment
# differed between the two sides, the baseline is what catches it — a penalized
# cell failing tells you nothing about WHICH of the two things broke.
#
# SPECIES ORDER IS LOad-BEARING. drmTMB matches species to tips BY NAME; DRModels.jl's
# univariate sparse phylo route matches them POSITIONALLY (group levels are
# numbered by first appearance in the data, tree leaves by Newick left-to-right
# order). The fixture therefore emits rows in Newick tip order so that the two
# conventions coincide. Do not reorder the rows.
#
#   DRM_JL_PATH=/path/to/DRModels.jl Rscript tools/parity_phylo_penalty.R

suppressMessages(library(drmTMB))

tol <- 1e-4
out_path <- "docs/dev-log/evidence/parity-phylo-penalty.tsv"

make_fixture <- function(seed = 244L, n_tip = 10L, n_each = 6L, sd_phy = 0.7) {
  stopifnot(requireNamespace("ape", quietly = TRUE))
  set.seed(seed)
  tree <- ape::rcoal(n_tip)
  tree$tip.label <- paste0("sp_", seq_len(n_tip))
  # THE TREE IS RESCALED TO UNIT HEIGHT, and that is load-bearing twice over.
  # drmTMB builds its phylogenetic covariance from `ape::vcv(tree, corr = TRUE)`,
  # i.e. tips normalised to variance 1. DRModels.jl builds its sparse precision from
  # the Newick branch lengths AS GIVEN, so its tip variance is the tree height h.
  # Consequences, both measured on the first run of this fixture:
  #   (a) reporting — sd_drmTMB = sd_DRM * sqrt(h) exactly (observed ratio
  #       1.23630740 vs sqrt(h) = 1.23630753 on the raw rcoal tree), while the
  #       logLiks agreed to 5 decimals: the same fit, two SD scales;
  #   (b) the PRIOR — `sd_u` is a threshold ON that SD, so on a non-unit-height
  #       tree the same `sd_u` is a DIFFERENT prior on the two sides.
  # Normalising h to 1 makes the two conventions coincide, so this fixture
  # compares the penalty rather than the tree scaling.
  h <- max(diag(ape::vcv(tree)))
  tree$edge.length <- tree$edge.length / h
  nwk <- ape::write.tree(tree)
  # Tip labels in NEWICK left-to-right order — the order DRModels.jl's parser assigns
  # leaf indices in. Taken from the string itself rather than from `tree$tip.label`
  # so the alignment cannot drift with ape's internal ordering.
  tip_order <- regmatches(nwk, gregexpr("[A-Za-z0-9_.-]+(?=:)", nwk, perl = TRUE))[[1]]
  stopifnot(setequal(tip_order, tree$tip.label), length(tip_order) == n_tip)

  A <- ape::vcv(tree, corr = TRUE)[tip_order, tip_order]
  u <- as.vector(t(chol(A)) %*% rnorm(n_tip)) * sd_phy
  species <- factor(rep(tip_order, each = n_each), levels = tip_order)
  x <- rnorm(n_tip * n_each)
  dat <- data.frame(
    y = 0.3 + 0.5 * x + u[rep(seq_len(n_tip), each = n_each)] +
        rnorm(n_tip * n_each, sd = 0.5),
    x = x,
    species = species
  )
  list(data = dat, tree = tree, newick = nwk,
       species_code = as.integer(species))   # 1..n_tip in Newick order
}

cells <- list(
  list(id = "phylo_penalty_ml_baseline", sd_u = NA_real_,  sd_alpha = NA_real_,
       label = "unpenalized ML baseline (validates the fixture + species alignment)"),
  list(id = "phylo_penalty_sd_moderate", sd_u = 0.50, sd_alpha = 0.05,
       label = "PC prior, P(sd > 0.50) = 0.05"),
  list(id = "phylo_penalty_sd_tight",    sd_u = 0.25, sd_alpha = 0.01,
       label = "PC prior, P(sd > 0.25) = 0.01 (drmTMB's own test setting)")
)

fx <- make_fixture()
tree <- fx$tree

drmTMB:::drm_julia_setup()
JuliaCall::julia_assign("pp_y", as.numeric(fx$data$y))
JuliaCall::julia_assign("pp_x", as.numeric(fx$data$x))
JuliaCall::julia_assign("pp_sp", as.integer(fx$species_code))
JuliaCall::julia_assign("pp_newick", fx$newick)
JuliaCall::julia_command("pp_d = (; y = Float64.(pp_y), x = Float64.(pp_x), species = Int.(pp_sp))")
JuliaCall::julia_command("pp_tree = DRModels.augmented_phy(pp_newick)")

# UPSTREAM BUG, MEASURED HERE — read this before "fixing" the Julia side.
#
# drmTMB reads its penalty from `fit$obj$report()` (R/drmTMB.R ~line 635), called
# with NO argument. TMB then reports at `obj$env$last.par`, which after the
# Hessian/sdreport step is a FINITE-DIFFERENCE PERTURBATION of the optimum — here
# max|last.par - last.par.best| = 1e-3. So `fit$phylo_penalty` is evaluated a step
# away from the MLE, and because `fit$logLik <- -opt$objective + phylo_penalty`,
# the reported log-likelihood of every penalized drmTMB fit inherits the error.
#
# Measured on this fixture (sd_u = 0.5):
#   drmTMB fit$phylo_penalty          2.81823157781175   <- reported, off-optimum
#   obj$report(last.par.best)         2.82150351154948   <- at the optimum
#   lam*sd - log(sd) - log(lam)       2.82150351154948   <- drmTMB's own formula
#   DRModels.jl fit.phylo_penalty          2.82150351154948   <- agrees to 15 digits
#
# Filed upstream as drmTMB#1036 (2026-08-15).
# DRModels.jl matches drmTMB's DOCUMENTED FORMULA and its own internal parameter; the
# R value is the outlier. Comparing against the reported number would therefore
# force the port to reproduce an upstream defect, so this fixture evaluates the
# penalty at `last.par.best` and reconstructs logLik from it. The raw reported
# values are kept in the TSV as `penalty_tmb_reported` so the gap stays visible
# and is never silently absorbed.
r_fit <- function(cell) {
  form <- bf(y ~ x + phylo(1 | species, tree = tree), sigma ~ 1)
  pen <- if (is.na(cell$sd_u)) NULL else
    drm_phylo_penalty(sd_u = cell$sd_u, sd_alpha = cell$sd_alpha)
  fit <- drmTMB(form, data = fx$data, penalty = pen)
  reported <- if (is.null(fit$phylo_penalty)) 0 else as.numeric(fit$phylo_penalty)
  if (is.na(cell$sd_u)) {
    return(c(sd = as.numeric(fit$sdpars$mu[[1L]]),
             loglik = as.numeric(fit$logLik), penalty = 0, reported = 0))
  }
  at_opt <- fit$obj$report(fit$obj$env$last.par.best)
  pen_opt <- as.numeric(at_opt$phylo_penalty)
  c(sd = as.numeric(fit$sdpars$mu[[1L]]),
    loglik = -as.numeric(fit$opt$objective) + pen_opt,
    penalty = pen_opt,
    reported = reported)
}

j_fit <- function(cell) {
  pen_txt <- if (is.na(cell$sd_u)) "nothing" else
    sprintf("DRModels.drm_phylo_penalty(sd_u = %.17g, sd_alpha = %.17g)", cell$sd_u, cell$sd_alpha)
  as.numeric(JuliaCall::julia_eval(sprintf(
    'let d = pp_d, tr = pp_tree
       f = DRModels.drm(DRModels.bf(DRModels.@formula(y ~ x + phylo(1 | species)), DRModels.@formula(sigma ~ 1)),
                   DRModels.Gaussian(); data = d, tree = tr, penalty = %s)
       p = isnan(f.phylo_penalty) ? 0.0 : f.phylo_penalty
       [DRModels.re_sd(f)[:species], DRModels.loglik(f), p]
     end', pen_txt)))
}

rows <- list()
for (cell in cells) {
  res <- list(capability_id = cell$id, label = cell$label, status = NA_character_,
              sd_tmb = NA_real_, sd_julia = NA_real_,
              loglik_tmb = NA_real_, loglik_julia = NA_real_,
              penalty_tmb = NA_real_, penalty_julia = NA_real_,
              penalty_tmb_reported = NA_real_,
              max_abs_diff = NA_real_, tolerance = tol, note = "")

  rv <- tryCatch(r_fit(cell),
                 error = function(e) { res$note <<- conditionMessage(e); rep(NA_real_, 4) })
  jv <- tryCatch(j_fit(cell),
                 error = function(e) { res$note <<- paste(res$note, conditionMessage(e)); rep(NA_real_, 3) })

  res$sd_tmb <- rv[1]; res$loglik_tmb <- rv[2]; res$penalty_tmb <- rv[3]
  res$penalty_tmb_reported <- rv[4]
  res$sd_julia <- jv[1]; res$loglik_julia <- jv[2]; res$penalty_julia <- jv[3]

  if (all(is.finite(rv)) && all(is.finite(jv))) {
    res$max_abs_diff <- max(abs(rv[1:3] - jv))
    res$status <- if (res$max_abs_diff < tol) "PARITY_PASS" else "PARITY_FAIL"
    if (is.finite(rv[4]) && abs(rv[4] - rv[3]) > 1e-8) {
      res$note <- sprintf(
        "drmTMB fit$phylo_penalty reports %.15g (off-optimum, obj$report() at last.par); compared at last.par.best = %.15g",
        rv[4], rv[3])
    }
  } else {
    res$status <- if (all(is.finite(rv))) "JULIA_FAILED" else
                  if (all(is.finite(jv))) "NATIVE_FAILED" else "BOTH_FAILED"
  }
  rows[[length(rows) + 1L]] <- as.data.frame(res, stringsAsFactors = FALSE)
  cat(sprintf("%-26s %-13s sd %.7f/%.7f  ll %.5f/%.5f  pen %.6f/%.6f  max|d|=%.3e\n",
              res$capability_id, res$status, res$sd_tmb, res$sd_julia,
              res$loglik_tmb, res$loglik_julia, res$penalty_tmb, res$penalty_julia,
              res$max_abs_diff))
  if (nzchar(res$note)) cat("    note: ", res$note, "\n", sep = "")
}

tab <- do.call(rbind, rows)
dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write.table(tab, out_path, sep = "\t", row.names = FALSE, quote = FALSE)
cat("\nwrote ", out_path, "\n", sep = "")
cat("OVERALL: ", if (all(tab$status == "PARITY_PASS")) "ALL CELLS PASS" else "SOME CELLS FAILED",
    "\n", sep = "")
