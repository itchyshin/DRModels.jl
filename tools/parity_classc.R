# parity_classc.R — native-vs-Julia same-target comparison for the two
# remaining reachable class-(c) capability rows: `phylo_count_large_p` and
# `general_covariance_structured`.
#
# Both rows are MEASUREMENT ONLY: the engine works on both sides and only the
# number is missing. Each cell fits the SAME target twice through drmTMB,
# `engine = "tmb"` vs `engine = "julia"`, and compares coefficients + logLik +
# fixed-effect SE (where obtainable — `vcov()` on either side can fail
# independently of the fit itself; a failure there does not retract the
# coefficient/logLik comparison already made for that cell).
# `engine = "julia"` routes a `phylo(1 | group, tree = tree)` mean term and a
# `relmat(1 | group, K = K)` mean term through the DRModels.jl bridge automatically
# (drmTMB_julia_bridge / drmTMB_julia_structured_bridge); no manual JuliaCall
# marshalling is needed here (unlike the bivariate cells in parity_fixture.R).
#
# KNOWN TRAP (docs/dev-log/evidence/2026-08-24-sd-floor-asymmetry.md): DRModels.jl
# clamps variance components at `_LAPLACE_LOG_SD_FLOOR = log(1e-6)`; drmTMB has
# no equivalent bound. Generating SDs here are kept well away from that floor
# (0.5), and the fitted phylo/relmat SD on each side is recorded so a
# near-floor draw reads as `BOUNDARY_NOT_COMPARABLE`, not PARITY_FAIL.
#
#   DRM_JL_PATH=/path/to/DRModels.jl NOT_CRAN=true Rscript tools/parity_classc.R

suppressMessages(library(drmTMB))
stopifnot(requireNamespace("ape", quietly = TRUE))

tol <- 1e-4
sd_floor <- 1e-6
out_path <- "docs/dev-log/evidence/parity-classc.tsv"

`%||%` <- function(a, b) if (is.null(a)) b else a

# cli::cli_abort() messages wrap onto multiple lines; collapse to keep one
# TSV row per cell (tools/parity_crosscheck.py and every sibling evidence TSV
# assume that).
one_line <- function(x) gsub("\\s+", " ", trimws(paste(x, collapse = " ")))

# Pull the single named structured-effect SD (phylo/relmat) out of a fit's
# sdpars$mu, regardless of engine. Returns NA if none is present.
structured_sd <- function(fit) {
  v <- tryCatch(fit$sdpars$mu, error = function(e) NULL)
  if (is.null(v) || length(v) == 0L) {
    return(NA_real_)
  }
  hits <- v[grepl("phylo\\(|relmat\\(", names(v))]
  if (length(hits) == 0L) {
    return(NA_real_)
  }
  unname(hits[[1L]])
}

# Fixed-effect Wald SEs = sqrt(diag(vcov())), in whatever coefficient order
# the fit itself returns (same positional convention `run_cell` already uses
# for the coefficient-diff comparison below — this file does not do the
# name-normalising match tools/parity_se.R uses). NA (not an error) if
# vcov() fails on that side; SE is a second-order quantity and the two
# engines are not guaranteed to produce one for every model.
se_of <- function(fit) {
  tryCatch({
    v <- diag(as.matrix(vcov(fit)))
    ifelse(v > 0, sqrt(v), NA_real_)
  }, error = function(e) NULL)
}
fmt_se <- function(x) if (is.null(x)) "NA" else paste(sprintf("%.6g", x), collapse = ";")

run_cell <- function(capability_id, cell_id, label, native_expr, julia_expr,
                     note_prefix = "") {
  res <- list(
    capability_id = capability_id, cell_id = cell_id, label = label,
    status = NA_character_, max_abs_coef_diff = NA_real_,
    loglik_tmb = NA_real_, loglik_julia = NA_real_, loglik_diff = NA_real_,
    max_abs_se_diff = NA_real_, max_rel_se_diff = NA_real_,
    se_tmb = "NA", se_julia = "NA",
    tolerance = tol, note = ""
  )
  ft <- tryCatch(eval(native_expr), error = function(e) {
    res$note <<- one_line(paste0(note_prefix, "native: ", conditionMessage(e)))
    NULL
  })
  fj <- tryCatch(eval(julia_expr), error = function(e) {
    res$note <<- one_line(paste(res$note, paste0(note_prefix, "julia: ", conditionMessage(e))))
    NULL
  })

  sd_tmb <- if (!is.null(ft)) structured_sd(ft) else NA_real_
  sd_jl <- if (!is.null(fj)) structured_sd(fj) else NA_real_

  if (is.null(ft)) {
    res$status <- if (grepl("planned, not implemented|Structured non-Gaussian|not.*implemented",
                            res$note, ignore.case = TRUE)) {
      "NO_NATIVE_COMPARATOR"
    } else {
      "NATIVE_FAILED"
    }
  } else if (is.null(fj)) {
    res$status <- "JULIA_FAILED"
  } else {
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

    near_floor <- (!is.na(sd_jl) && sd_jl < 5 * sd_floor) ||
      (!is.na(sd_tmb) && sd_tmb < 5 * sd_floor)
    if (near_floor) {
      res$status <- "BOUNDARY_NOT_COMPARABLE"
      res$note <- paste0(res$note,
        sprintf("fitted structured SD tmb=%.3e julia=%.3e (floor=%.0e)",
                sd_tmb, sd_jl, sd_floor))
    } else {
      agree <- res$max_abs_coef_diff < tol && res$loglik_diff < tol
      res$status <- if (agree) "PARITY_PASS" else "PARITY_FAIL"
      res$note <- sprintf("structured SD tmb=%.3e julia=%.3e; %d coefficient(s) compared",
                           sd_tmb, sd_jl, k)
    }
  }
  cat(sprintf("%-28s %-24s %-22s coef_diff=%.3e  loglik_diff=%.3e\n",
              capability_id, cell_id, res$status,
              res$max_abs_coef_diff, res$loglik_diff))
  if (nzchar(res$note)) cat("    note: ", substr(res$note, 1, 200), "\n", sep = "")
  as.data.frame(res, stringsAsFactors = FALSE)
}

rows <- list()

## ---------------------------------------------------------------------
## (a) phylo_count_large_p — Poisson / NB2 with phylo(1 | species, tree = tree)
## ---------------------------------------------------------------------

make_phylo_count_fixture <- function(seed, p, m, beta = c(0.30, 0.35),
                                      sd_phy = 0.5, size_nb = 4) {
  set.seed(seed)
  tree <- ape::rcoal(p)
  tree$tip.label <- paste0("sp_", seq_len(p))
  A <- ape::vcv(tree, corr = TRUE)                 # unit-diagonal phylo corr
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

# Smoke first, small p — read the output before scaling up (D-139).
smoke <- make_phylo_count_fixture(seed = 20260824L, p = 20L, m = 5L)
tree <- smoke$tree
dat <- smoke$data_pois
rows[[length(rows) + 1L]] <- run_cell(
  "phylo_count_large_p", "poisson_phylo_smoke_p20",
  "Poisson, phylo(1 | species) mean intercept, smoke p=20",
  quote(drmTMB(bf(y ~ x + phylo(1 | species, tree = tree)),
               family = stats::poisson(link = "log"), data = dat, engine = "tmb")),
  quote(drmTMB(bf(y ~ x + phylo(1 | species, tree = tree)),
               family = stats::poisson(link = "log"), data = dat, engine = "julia"))
)

# Scale up. p=300 is "large-ish" relative to the DRModels.jl unit tests (p=32) and
# cheap: dense p x p phylo covariance factorisation is O(p^3) ~ 2.7e7, trivial
# for both TMB and the DRModels.jl sparse route.
big <- make_phylo_count_fixture(seed = 20260824L, p = 300L, m = 4L)
tree <- big$tree

dat <- big$data_pois
rows[[length(rows) + 1L]] <- run_cell(
  "phylo_count_large_p", "poisson_phylo_p300",
  "Poisson, phylo(1 | species) mean intercept, p=300",
  quote(drmTMB(bf(y ~ x + phylo(1 | species, tree = tree)),
               family = stats::poisson(link = "log"), data = dat, engine = "tmb")),
  quote(drmTMB(bf(y ~ x + phylo(1 | species, tree = tree)),
               family = stats::poisson(link = "log"), data = dat, engine = "julia"))
)

dat <- big$data_nb2
rows[[length(rows) + 1L]] <- run_cell(
  "phylo_count_large_p", "nb2_phylo_p300",
  "NegBinomial2, phylo(1 | species) mean intercept, p=300",
  quote(drmTMB(bf(y ~ x + phylo(1 | species, tree = tree), sigma ~ 1),
               family = nbinom2(), data = dat, engine = "tmb")),
  quote(drmTMB(bf(y ~ x + phylo(1 | species, tree = tree), sigma ~ 1),
               family = nbinom2(), data = dat, engine = "julia"))
)

## ---------------------------------------------------------------------
## (b) general_covariance_structured — relmat(1 | group, K = K), sigma ~ 1
## families both sides admit per docs/dev-log/evidence/
## 2026-08-16-a9-general-covariance-audit.md: Gaussian, Poisson, NB2, Gamma.
## Beta is DRModels.jl-only (audit finding 1) and Binomial is refused by both; ergo
## neither belongs in a same-target native-vs-Julia comparison. `engine =
## "julia"` itself gates relmat/animal/spatial structured terms to exactly
## these four univariate families (R/julia-bridge.R
## drm_julia_structured_family_tag), so this list is not a guess.
## ---------------------------------------------------------------------

make_relmat_fixture <- function(seed, n_id = 12L, n_each = 8L,
                                 beta = c(0.30, 0.30), sd_known = 0.5,
                                 sigma_gauss = 0.25, size_nb = 4, shape_g = 8) {
  set.seed(seed)
  id_levels <- paste0("id", seq_len(n_id))
  K <- outer(seq_len(n_id), seq_len(n_id), function(i, j) 0.35^abs(i - j))
  diag(K) <- diag(K) + 0.15
  dimnames(K) <- list(id_levels, id_levels)
  known <- as.vector(t(chol(K)) %*% rnorm(n_id)) * sd_known
  names(known) <- id_levels
  id <- rep(id_levels, each = n_each)
  n <- n_id * n_each
  x <- rnorm(n)
  eta <- beta[1] + beta[2] * x + known[id]
  list(
    K = K,
    data_gauss = data.frame(y = eta + sigma_gauss * rnorm(n), x = x, id = id),
    data_pois  = data.frame(y = rpois(n, exp(eta)), x = x, id = id),
    data_nb2   = data.frame(y = rnbinom(n, mu = exp(eta), size = size_nb), x = x, id = id),
    data_gamma = data.frame(y = rgamma(n, shape = shape_g, rate = shape_g / exp(eta)),
                             x = x, id = id)
  )
}

rf <- make_relmat_fixture(seed = 20260824L)
K <- rf$K

dat <- rf$data_gauss
rows[[length(rows) + 1L]] <- run_cell(
  "general_covariance_structured", "gaussian_relmat",
  "Gaussian, relmat(1 | id, K = K) mean intercept, sigma ~ 1",
  quote(drmTMB(bf(y ~ x + relmat(1 | id, K = K), sigma ~ 1),
               family = stats::gaussian(), data = dat, engine = "tmb")),
  quote(drmTMB(bf(y ~ x + relmat(1 | id, K = K), sigma ~ 1),
               family = stats::gaussian(), data = dat, engine = "julia"))
)

dat <- rf$data_pois
rows[[length(rows) + 1L]] <- run_cell(
  "general_covariance_structured", "poisson_relmat",
  "Poisson, relmat(1 | id, K = K) mean intercept",
  quote(drmTMB(bf(y ~ x + relmat(1 | id, K = K)),
               family = stats::poisson(link = "log"), data = dat, engine = "tmb")),
  quote(drmTMB(bf(y ~ x + relmat(1 | id, K = K)),
               family = stats::poisson(link = "log"), data = dat, engine = "julia"))
)

dat <- rf$data_nb2
rows[[length(rows) + 1L]] <- run_cell(
  "general_covariance_structured", "nb2_relmat",
  "NegBinomial2, relmat(1 | id, K = K) mean intercept, sigma ~ 1",
  quote(drmTMB(bf(y ~ x + relmat(1 | id, K = K), sigma ~ 1),
               family = nbinom2(), data = dat, engine = "tmb")),
  quote(drmTMB(bf(y ~ x + relmat(1 | id, K = K), sigma ~ 1),
               family = nbinom2(), data = dat, engine = "julia"))
)

dat <- rf$data_gamma
rows[[length(rows) + 1L]] <- run_cell(
  "general_covariance_structured", "gamma_relmat",
  "Gamma (log link), relmat(1 | id, K = K) mean intercept, sigma ~ 1",
  quote(drmTMB(bf(y ~ x + relmat(1 | id, K = K), sigma ~ 1),
               family = stats::Gamma(link = "log"), data = dat, engine = "tmb")),
  quote(drmTMB(bf(y ~ x + relmat(1 | id, K = K), sigma ~ 1),
               family = stats::Gamma(link = "log"), data = dat, engine = "julia"))
)

# Beta + relmat: DRModels.jl fits it (A9 audit), drmTMB natively refuses it
# ("Structured non-Gaussian paths ... remain deferred"). Recorded as
# NO_NATIVE_COMPARATOR, not a failure — that refusal is the honest finding.
# Build a small (0,1) response directly from the same K/id structure.
{
  n_id <- nrow(K); id_levels <- rownames(K)
  set.seed(20260824L)
  known_beta <- as.vector(t(chol(K)) %*% rnorm(n_id)) * 0.5
  names(known_beta) <- id_levels
  id <- rep(id_levels, each = 8L)
  n <- length(id)
  x <- rnorm(n)
  eta <- 0.1 + 0.2 * x + known_beta[id]
  dat <- data.frame(
    y = pmin(pmax(plogis(eta) + 0.02 * rnorm(n), 1e-3), 1 - 1e-3),
    x = x, id = id
  )
}
rows[[length(rows) + 1L]] <- run_cell(
  "general_covariance_structured", "beta_relmat",
  "Beta, relmat(1 | id, K = K) mean intercept, sigma ~ 1 (DRModels.jl admits; drmTMB refuses)",
  quote(drmTMB(bf(y ~ x + relmat(1 | id, K = K), sigma ~ 1),
               family = drmTMB::beta(), data = dat, engine = "tmb")),
  quote(drmTMB(bf(y ~ x + relmat(1 | id, K = K), sigma ~ 1),
               family = drmTMB::beta(), data = dat, engine = "julia"))
)

## ---------------------------------------------------------------------

tab <- do.call(rbind, rows)
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
tab$drmtmb_code_hash <- .drmtmb_stamp

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write.table(tab, out_path, sep = "\t", row.names = FALSE, quote = FALSE)
cat("\nwrote ", out_path, "\n", sep = "")
comparable <- tab[!tab$status %in% c("NO_NATIVE_COMPARATOR"), , drop = FALSE]
cat("OVERALL: ",
    if (nrow(comparable) > 0 && all(comparable$status == "PARITY_PASS")) {
      sprintf("ALL COMPARABLE CELLS PASS (%d of %d; %d have no native comparator)",
              nrow(comparable), nrow(tab), nrow(tab) - nrow(comparable))
    } else {
      "SOME COMPARABLE CELLS DID NOT PASS OR HIT A BOUNDARY"
    },
    "\n", sep = "")
