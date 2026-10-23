#!/usr/bin/env Rscript
# parity_classc_largep.R — phylo_count_large_p SE-precision recheck (issue #487).
#
# tools/parity_classc.R already measures phylo_count_large_p at p=300 (coef +
# logLik only, no SE columns). A separate large-p probe extended p to 1000 and
# 3000 but only ever printed the fitted SE vectors via `%.6f` and never
# computed max_abs_se_diff / max_rel_se_diff — so the ~1.6e-03 / ~1.5e-03
# figures quoted for p=1000/p=3000 were read off ROUNDED display output, not
# measured. This script fixes that gap: it fits the SAME target twice
# (`engine = "tmb"` vs `engine = "julia"`) at p = 300, 1000, 3000, on the SAME
# seed/DGP for all three (so the three points are comparable), and computes
# max_abs_se_diff / max_rel_se_diff at FULL PRECISION from the fitted SE
# vectors before any rounding — the same pattern tools/parity_classc.R uses
# for its own SE columns (fixed-effect Wald SE = sqrt(diag(vcov(fit)))).
#
# tools/parity_classc.R itself is NOT modified — this script reuses its
# make_phylo_count_fixture(seed, p, m, ...) verbatim (copied below) and calls
# it at the three target p values.
#
# rcond: src/vcov_guard.jl's `_vcov_from_hessian` guard is NOT actually in the
# call path for this route — src/sparse_laplace_glmm.jl's
# `_fit_poisson_general_laplace` (which `_fit_poisson_phylo_laplace`
# delegates to) computes the ML fixed-effect vcov inline via
# `try inv(Symmetric(Hθ)) catch → identity`, with no conditioning diagnostic
# exposed anywhere in that path. So rcond is computed here independently, as
# the reciprocal condition number (min|eig| / max|eig|) of the 2x2 vcov(fit)
# matrix each engine actually returns for `y ~ x + phylo(1 | species)` — the
# same matrix se_of() takes sqrt(diag()) of, on both the native TMB side and
# the DRModels.jl side.
#
# Each engine's fit is timed separately (D-139); native TMB is wrapped in a
# wall-clock ceiling so a divergent/hanging dense factorisation is a finding,
# not a script hang.
#
#   DRM_JL_PATH=$(pwd) NOT_CRAN=true Rscript tools/parity_classc_largep.R

suppressMessages(library(drmTMB))
stopifnot(requireNamespace("ape", quietly = TRUE))

tol <- 1e-4
seed <- 20260824L
m <- 4L
ps <- c(300L, 1000L, 3000L)
tmb_timeout_sec <- 1500  # 25 min ceiling per fit, per D-139

out_path <- "docs/dev-log/evidence/parity-classc.tsv"

# Copied verbatim from tools/parity_classc.R's make_phylo_count_fixture() —
# that harness is not modified, just called here at different p.
make_phylo_count_fixture <- function(seed, p, m, beta = c(0.30, 0.35),
                                      sd_phy = 0.5, size_nb = 4) {
  set.seed(seed)
  tree <- ape::rcoal(p)
  tree$tip.label <- paste0("sp_", seq_len(p))
  A <- ape::vcv(tree, corr = TRUE)
  u <- as.vector(t(chol(A)) %*% rnorm(p)) * sd_phy
  species <- factor(rep(tree$tip.label, each = m), levels = tree$tip.label)
  idx <- rep(seq_len(p), each = m)
  n <- p * m
  x <- rnorm(n)
  eta <- beta[1] + beta[2] * x + u[idx]
  list(
    tree = tree,
    data_pois = data.frame(y = rpois(n, exp(eta)), x = x, species = species),
    data_nb2  = data.frame(
      y = rnbinom(n, mu = exp(eta), size = size_nb), x = x, species = species
    )
  )
}

# Fixed-effect Wald SE = sqrt(diag(vcov())); NULL (not an error) if vcov()
# itself fails on that side.
se_of <- function(fit) {
  tryCatch({
    v <- diag(as.matrix(vcov(fit)))
    ifelse(v > 0, sqrt(v), NA_real_)
  }, error = function(e) NULL)
}
fmt_se <- function(x) if (is.null(x)) "NA" else paste(sprintf("%.15g", x), collapse = ";")

# Reciprocal condition number (min|eig| / max|eig|) of the vcov(fit) matrix
# actually returned — see the file header for why this, not a vcov_guard.jl
# log line, is what is available for this route.
rcond_of <- function(fit) {
  tryCatch({
    V <- as.matrix(vcov(fit))
    ev <- eigen(V, symmetric = TRUE, only.values = TRUE)$values
    min(abs(ev)) / max(abs(ev))
  }, error = function(e) NA_real_)
}

check_real <- function(fit, label) {
  if (is.null(fit)) { cat(sprintf("  %s: NULL fit\n", label)); return(FALSE) }
  co <- unlist(fixef(fit))
  ll <- tryCatch(as.numeric(logLik(fit)), error = function(e) NA_real_)
  ok <- length(co) > 0 && !any(is.na(co)) && is.finite(ll)
  cat(sprintf("  %s: coef=%s logLik=%.4f real=%s\n", label,
              paste(sprintf("%.4f", co), collapse = ","), ll, ok))
  ok
}

rows <- list()
rcond_log <- list()

for (p in ps) {
  cat(sprintf("\n=== phylo_count_large_p SE recheck: p=%d, m=%d, seed=%d ===\n", p, m, seed))
  fx <- make_phylo_count_fixture(seed = seed, p = p, m = m)
  tree <- fx$tree
  dat <- fx$data_pois
  cat(sprintf("Fixture built: n=%d rows, p=%d tips, m=%d reps/tip\n", nrow(dat), p, m))

  cat(sprintf("Fitting engine = 'tmb' (native, dense phylo factorisation), timeout=%.0fs...\n",
              tmb_timeout_sec))
  ft <- NULL
  t_tmb <- system.time({
    ft <- tryCatch({
      setTimeLimit(elapsed = tmb_timeout_sec, transient = TRUE)
      on.exit(setTimeLimit(elapsed = Inf, transient = TRUE), add = TRUE)
      drmTMB(bf(y ~ x + phylo(1 | species, tree = tree)),
             family = stats::poisson(link = "log"), data = dat, engine = "tmb")
    }, error = function(e) {
      cat("NATIVE TMB FAILED/TIMED OUT: ", conditionMessage(e), "\n")
      NULL
    })
  })
  cat(sprintf("  wall clock (tmb): %.2f sec\n", t_tmb[["elapsed"]]))

  cat("Fitting engine = 'julia' (DRModels.jl sparse route)...\n")
  fj <- NULL
  t_jl <- system.time({
    fj <- tryCatch(
      drmTMB(bf(y ~ x + phylo(1 | species, tree = tree)),
             family = stats::poisson(link = "log"), data = dat, engine = "julia"),
      error = function(e) { cat("JULIA ENGINE FAILED: ", conditionMessage(e), "\n"); NULL }
    )
  })
  cat(sprintf("  wall clock (julia): %.2f sec\n", t_jl[["elapsed"]]))

  ok_tmb <- check_real(ft, "tmb")
  ok_jl  <- check_real(fj, "julia")

  res <- list(
    capability_id = "phylo_count_large_p",
    cell_id = sprintf("poisson_phylo_p%d", p),
    label = sprintf("Poisson, phylo(1 | species) mean intercept, p=%d", p),
    status = NA_character_, max_abs_coef_diff = NA_real_,
    loglik_tmb = NA_real_, loglik_julia = NA_real_, loglik_diff = NA_real_,
    max_abs_se_diff = NA_real_, max_rel_se_diff = NA_real_,
    se_tmb = "NA", se_julia = "NA",
    tolerance = tol, note = ""
  )

  rc_tmb <- NA_real_; rc_jl <- NA_real_
  if (ok_tmb && ok_jl) {
    ct <- unlist(fixef(ft)); cj <- unlist(fixef(fj))
    k <- min(length(ct), length(cj))
    res$max_abs_coef_diff <- max(abs(ct[seq_len(k)] - cj[seq_len(k)]))
    res$loglik_tmb <- as.numeric(logLik(ft))
    res$loglik_julia <- as.numeric(logLik(fj))
    res$loglik_diff <- abs(res$loglik_tmb - res$loglik_julia)

    se_t <- se_of(ft); se_j <- se_of(fj)
    res$se_tmb <- fmt_se(se_t); res$se_julia <- fmt_se(se_j)
    if (!is.null(se_t) && !is.null(se_j)) {
      ks <- min(length(se_t), length(se_j))
      d <- abs(se_t[seq_len(ks)] - se_j[seq_len(ks)])
      res$max_abs_se_diff <- max(d)
      res$max_rel_se_diff <- max(d / pmax(abs(se_t[seq_len(ks)]), abs(se_j[seq_len(ks)])))
    }

    rc_tmb <- rcond_of(ft); rc_jl <- rcond_of(fj)
    status <- if (res$max_abs_coef_diff < tol && res$loglik_diff < tol) "PARITY_PASS" else "PARITY_FAIL"
    res$status <- status
    res$note <- sprintf(
      "large-p SE recheck (#487), full precision, seed=%d m=%d; %d coefficient(s) compared; rcond(vcov) tmb=%.3e julia=%.3e",
      seed, m, k, rc_tmb, rc_jl)
  } else if (ok_tmb && !ok_jl) {
    res$status <- "JULIA_FAILED"
  } else if (!ok_tmb && ok_jl) {
    res$status <- "NO_NATIVE_COMPARATOR_AT_SCALE"
  } else {
    res$status <- "BOTH_FAILED"
  }

  cat(sprintf("max|coef diff| = %.3e, |logLik diff| = %.3e, max_rel_se_diff = %.3e\n",
              res$max_abs_coef_diff, res$loglik_diff, res$max_rel_se_diff))
  cat(sprintf("rcond(vcov): tmb=%.6e julia=%.6e\n", rc_tmb, rc_jl))
  cat("STATUS: ", res$status, "\n")
  cat(sprintf("SUMMARY p=%d m=%d n=%d: tmb=%.2fs julia=%.2fs\n",
              p, m, nrow(dat), t_tmb[["elapsed"]], t_jl[["elapsed"]]))

  rows[[length(rows) + 1L]] <- as.data.frame(res, stringsAsFactors = FALSE)
  rcond_log[[length(rcond_log) + 1L]] <- data.frame(p = p, rcond_tmb = rc_tmb, rcond_julia = rc_jl)
}

new_tab <- do.call(rbind, rows)
cat("\n=== new rows ===\n")
print(new_tab[, c("cell_id", "status", "max_abs_coef_diff", "loglik_diff",
                   "max_abs_se_diff", "max_rel_se_diff")])
cat("\n=== rcond by p ===\n")
print(do.call(rbind, rcond_log))

# ---------------------------------------------------------------------------
# Merge into docs/dev-log/evidence/parity-classc.tsv: add the 4 SE columns to
# the header if absent (NA for every pre-existing row this script does not
# re-measure — general_covariance_structured, poisson_phylo_smoke_p20,
# nb2_phylo_p300 are all out of scope here), then upsert one row per p in
# `ps` by cell_id (poisson_phylo_p300 already exists — same seed/p/m as its
# original measurement, so this updates it in place with full-precision SE
# columns rather than adding a duplicate; poisson_phylo_p1000/p3000 are new).
# ---------------------------------------------------------------------------
# --- provenance stamp (#473) -------------------------------------------------
# Record WHICH drmTMB build produced these numbers, not just its version string.
# "drmTMB 0.7.0" identifies at least 16 different builds, so a version alone
# cannot tell a later reader whether a disagreement is DRModels.jl regressing or the
# COMPARATOR having moved underneath the fixture. Stamped at write time from the
# single definition in drmtmb_provenance_lib.R.
.tools_dir <- tryCatch({
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(f)) dirname(sub("^--file=", "", f[1])) else "tools"
}, error = function(e) "tools")
source(file.path(.tools_dir, "drmtmb_provenance_lib.R"))
.drmtmb_stamp <- drmtmb_code_hash()
new_tab$drmtmb_code_hash <- .drmtmb_stamp

se_cols <- c("max_abs_se_diff", "max_rel_se_diff", "se_tmb", "se_julia",
             "drmtmb_code_hash")
old <- read.delim(out_path, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE,
                   colClasses = "character", na.strings = character(0), quote = "")
missing_cols <- setdiff(se_cols, names(old))
for (col in missing_cols) old[[col]] <- NA_character_
# column order: ... loglik_diff, <se_cols>, tolerance, note
base_order <- c("capability_id", "cell_id", "label", "status", "max_abs_coef_diff",
                 "loglik_tmb", "loglik_julia", "loglik_diff",
                 "max_abs_se_diff", "max_rel_se_diff", "se_tmb", "se_julia",
                 "tolerance", "note", "drmtmb_code_hash")
old <- old[, base_order]

new_tab_chr <- as.data.frame(lapply(new_tab, as.character), stringsAsFactors = FALSE)
new_tab_chr <- new_tab_chr[, base_order]

for (i in seq_len(nrow(new_tab_chr))) {
  cid <- new_tab_chr$cell_id[i]
  hit <- which(old$cell_id == cid)
  if (length(hit) == 1L) {
    old[hit, ] <- new_tab_chr[i, ]
  } else {
    old <- rbind(old, new_tab_chr[i, ])
  }
}

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write.table(old, out_path, sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
cat("\nwrote ", out_path, "\n", sep = "")
