# bench_fit_h2h.R — broad single-fit head-to-head, drmTMB engine="tmb" vs
# engine="julia", through the SAME drmTMB() call wherever the R bridge admits
# the family/random-effect combination (the "bridge" cells below). Two
# families (bivariate lognormal, bivariate Student-t) are never routed by
# `engine = "julia"` at all (drmTMB's own gate: "Gaussian one-/two-response
# ... or large-p phylogenetic Poisson, NB2, Gamma, Beta, or Binomial models"),
# so those are timed as engine="tmb" (drmTMB()) vs a direct DRModels.jl bridge call
# (JuliaCall::julia_call), mirroring the `biv_cells` convention already used
# in tools/parity_fixture.R.
#
# Tests the owner's claim in two parts (see task brief):
#   (1) breadth  -- is Julia faster across many single fits, not one lucky cell?
#   (2) mechanism -- does the speed gap track optimizer choice, or something else
#       (sparsity handling / exact O(p) gradient / AD backend / no marshalling)?
# It also checks whether a "speed win" cell even HAS a same-target TMB
# comparator: some non-Gaussian phylo families (Gamma, Beta, Binomial) are
# native-TMB-REJECTED ("structured-effect syntax is planned, not implemented"
# on the TMB side) -- for those, DRModels.jl is not "faster", it is the only route,
# a different claim entirely. This script records that distinction per cell
# rather than silently forcing a ratio.
#
#   DRM_JL_PATH=/path/to/DRModels.jl Rscript tools/bench_fit_h2h.R
#
# Env vars:
#   BENCH_PILOT=1   -- pilot mode: n_reps=1, small cell subset, label
#                      everything provisional. Use to validate plumbing only.
#   BENCH_REPS=<n>  -- override timed-rep count (default 3, min for median).

suppressMessages(library(drmTMB))

pilot_mode <- identical(Sys.getenv("BENCH_PILOT"), "1")
n_reps <- as.integer(Sys.getenv("BENCH_REPS", if (pilot_mode) "1" else "3"))
tol <- 1e-4

out_tsv <- "docs/dev-log/evidence/fit-speed-h2h.tsv"
out_doc <- "docs/dev-log/evidence/2026-08-24-fit-speed-h2h.md"

cat(sprintf("bench_fit_h2h.R -- pilot_mode=%s n_reps=%d\n", pilot_mode, n_reps))

# ---- timing helper ---------------------------------------------------------
# One untimed warmup (absorbs Julia JIT / TMB DLL compile on first call to a
# new family), then n_reps timed reps; report the MEDIAN, not a single draw.
time_median <- function(f, n_reps) {
  warm <- tryCatch(list(ok = TRUE, val = f(), t_warm = NA_real_), error = function(e) {
    list(ok = FALSE, err = conditionMessage(e))
  })
  if (!warm$ok) return(list(ok = FALSE, err = warm$err, times = numeric(0)))
  times <- numeric(n_reps)
  val <- NULL
  for (i in seq_len(n_reps)) {
    t0 <- proc.time()[["elapsed"]]
    val <- f()
    times[i] <- proc.time()[["elapsed"]] - t0
  }
  list(ok = TRUE, val = val, times = times, median_s = stats::median(times))
}

# Coerce a possibly-NULL / possibly-zero-length / possibly-non-finite value to
# exactly one NA_integer_ or one integer -- a data.frame() column must always
# get length-1 values, and drmTMB's internal accessors are not guaranteed to
# return a scalar for every engine/family combination.
scalar_int <- function(x) {
  if (is.null(x) || length(x) != 1L || !is.finite(x)) return(NA_integer_)
  as.integer(x)
}

fit_info <- function(fit, engine) {
  if (engine == "tmb") {
    list(
      converged = tryCatch(isTRUE(fit$opt$convergence == 0L), error = function(e) NA),
      loglik = tryCatch(as.numeric(stats::logLik(fit)), error = function(e) NA_real_),
      iterations = scalar_int(tryCatch(drmTMB:::drm_qgt2_scalar_numeric(fit$opt$iterations), error = function(e) NA_integer_)),
      optimizer = "nlminb (R/TMB, CppAD)"
    )
  } else {
    iters <- tryCatch(fit$bridge$iterations, error = function(e) NULL)
    list(
      converged = tryCatch(drmTMB::is_converged(fit), error = function(e) NA),
      loglik = tryCatch(as.numeric(stats::logLik(fit)), error = function(e) NA_real_),
      iterations = scalar_int(iters),
      optimizer = "DRModels.jl sparse Laplace, exact O(p) gradient, LBFGS (Optim.jl)"
    )
  }
}

# ---- cell builders ----------------------------------------------------------

make_tree <- function(n_tip, seed) {
  stopifnot(requireNamespace("ape", quietly = TRUE))
  set.seed(seed)
  tree <- ape::rcoal(n_tip)
  tree$tip.label <- paste0("sp_", seq_len(n_tip))
  h <- max(diag(ape::vcv(tree)))
  tree$edge.length <- tree$edge.length / h   # unit tip variance -- see
  tree                                       # parity_phylo_nongaussian.R note 1
}

phylo_fixture <- function(n_tip, n_each, seed, sd_phy = 0.5) {
  tree <- make_tree(n_tip, seed)
  A <- ape::vcv(tree, corr = TRUE)
  u <- as.vector(t(chol(A)) %*% rnorm(n_tip)) * sd_phy
  species <- factor(rep(tree$tip.label, each = n_each), levels = tree$tip.label)
  idx <- rep(seq_len(n_tip), each = n_each)
  n <- n_tip * n_each
  x <- rnorm(n)
  eta <- 0.3 + 0.4 * x + u[idx]
  list(
    data = data.frame(
      x = x, species = species,
      y_pois  = rpois(n, exp(eta)),
      y_nb2   = rnbinom(n, mu = exp(eta), size = 4),
      y_gamma = rgamma(n, shape = 8, rate = 8 / exp(eta)),
      y_binom = rbinom(n, 1, plogis(eta)),
      y_beta  = pmin(pmax(plogis(eta) + 0.03 * rnorm(n), 1e-3), 1 - 1e-3)
    ),
    tree = tree, n = n, p = n_tip
  )
}

gauss_locscale_fixture <- function(n, seed) {
  set.seed(seed)
  x <- rnorm(n); z <- rnorm(n)
  data.frame(y = 0.4 + 0.9 * x + exp(-0.3 + 0.25 * z) * rnorm(n), x = x, z = z)
}

biv_gauss_fixture <- function(n, seed) {
  set.seed(seed)
  x <- rnorm(n)
  e1 <- rnorm(n); e2 <- 0.5 * e1 + sqrt(0.75) * rnorm(n)
  data.frame(x = x, y1 = 0.2 + 0.4 * x + 0.8 * e1, y2 = -0.1 + 0.3 * x + 0.7 * e2)
}

biv_gauss_q2_phylo_fixture <- function(n_tip, n_each, seed) {
  tree <- make_tree(n_tip, seed)
  V_phy <- ape::vcv.phylo(tree)
  C_phy <- t(chol(V_phy + diag(1e-10, n_tip)))
  Sigma_a <- matrix(c(0.22, 0.07, 0.07, 0.18), 2L, 2L)
  C_axis <- chol(Sigma_a)
  U <- C_phy %*% matrix(rnorm(n_tip * 2L), n_tip, 2L) %*% C_axis
  species <- rep(tree$tip.label, each = n_each)
  row <- match(species, tree$tip.label)
  n <- length(species)
  x <- rnorm(n)
  e1 <- rnorm(n, sd = 0.32); e2 <- 0.25 * e1 + sqrt(1 - 0.25^2) * rnorm(n, sd = 0.36)
  list(
    data = data.frame(species = species, x = x,
                       y1 = 0.20 + 0.30 * x + U[row, 1L] + e1,
                       y2 = -0.15 + 0.15 * x + U[row, 2L] + e2),
    tree = tree, n = n, p = n_tip
  )
}

biv_lognormal_fixture <- function(n, seed) {
  set.seed(seed); x <- rnorm(n)
  s1 <- 0.5; s2 <- 0.8; rho <- 0.6
  z1 <- rnorm(n); z2 <- rho * z1 + sqrt(1 - rho^2) * rnorm(n)
  data.frame(y1 = exp(0.4 + 0.9 * x + s1 * z1), y2 = exp(-0.2 + 0.5 * x + s2 * z2), x = x)
}

biv_student_fixture <- function(n, seed) {
  set.seed(seed); x <- rnorm(n)
  s1 <- 0.7; s2 <- 1.1; rho <- 0.5; nu <- 6
  z1 <- rnorm(n); z2 <- rho * z1 + sqrt(1 - rho^2) * rnorm(n)
  sh <- sqrt(nu / rchisq(n, df = nu))
  data.frame(y1 = 0.5 + 0.8 * x + s1 * z1 * sh, y2 = -0.3 + 0.4 * x + s2 * z2 * sh, x = x)
}

# ---- cell registry -----------------------------------------------------------
# type "bridge": drmTMB(..., engine = "tmb") vs drmTMB(..., engine = "julia"),
#   SAME call shape both sides -- this is the real user-facing comparison.
# type "direct": drmTMB(..., engine = "tmb") vs a raw DRModels.jl bridge call
#   (JuliaCall) -- used only where drmTMB's own gate never admits
#   engine = "julia" for this family at all (biv_lognormal, biv_student).

bridge_cells <- list(
  list(id = "gauss_locscale_n100", family = "Gaussian location-scale", n = 100,
       fixture = function() gauss_locscale_fixture(100, 20260814),
       formula = function() bf(y ~ x, sigma ~ z), fam = function() gaussian()),
  list(id = "gauss_locscale_n1000", family = "Gaussian location-scale", n = 1000,
       fixture = function() gauss_locscale_fixture(1000, 20260815),
       formula = function() bf(y ~ x, sigma ~ z), fam = function() gaussian()),

  list(id = "biv_gaussian_rho12_n100", family = "Bivariate Gaussian (rho12)", n = 100,
       fixture = function() biv_gauss_fixture(100, 7),
       formula = function() bf(mu1 = y1 ~ x, mu2 = y2 ~ x, sigma1 = ~1, sigma2 = ~1, rho12 = ~1),
       fam = function() biv_gaussian()),
  list(id = "biv_gaussian_rho12_n1000", family = "Bivariate Gaussian (rho12)", n = 1000,
       fixture = function() biv_gauss_fixture(1000, 8),
       formula = function() bf(mu1 = y1 ~ x, mu2 = y2 ~ x, sigma1 = ~1, sigma2 = ~1, rho12 = ~1),
       fam = function() biv_gaussian()),

  list(id = "biv_gaussian_q2_phylo_p12x3", family = "Bivariate Gaussian, q2 phylo (mu1,mu2)",
       n = NA, phylo = TRUE,
       fixture = function() biv_gauss_q2_phylo_fixture(12, 3, 20260625),
       formula = function() bf(mu1 = y1 ~ x + phylo(1 | p | species, tree = tree),
                               mu2 = y2 ~ x + phylo(1 | p | species, tree = tree),
                               sigma1 = ~1, sigma2 = ~1, rho12 = ~1),
       fam = function() biv_gaussian()),

  list(id = "poisson_phylo_p12x6", family = "Poisson, phylo(1|species)",
       n = NA, phylo = TRUE,
       fixture = function() phylo_fixture(12, 6, 4242),
       resp = "y_pois", extra = "",
       fam = function() poisson()),
  list(id = "poisson_phylo_p40x6", family = "Poisson, phylo(1|species)",
       n = NA, phylo = TRUE,
       fixture = function() phylo_fixture(40, 6, 4243),
       resp = "y_pois", extra = "",
       fam = function() poisson()),

  list(id = "nbinom2_phylo_p12x6", family = "NegBinomial2, phylo(1|species)",
       n = NA, phylo = TRUE,
       fixture = function() phylo_fixture(12, 6, 5151),
       resp = "y_nb2", extra = ", sigma ~ 1",
       fam = function() nbinom2()),

  list(id = "gamma_phylo_p12x6", family = "Gamma (log link), phylo(1|species)",
       n = NA, phylo = TRUE, tmb_rejects = TRUE,
       fixture = function() phylo_fixture(12, 6, 6161),
       resp = "y_gamma", extra = ", sigma ~ 1",
       fam = function() Gamma(link = "log")),

  list(id = "beta_phylo_p12x6", family = "Beta, phylo(1|species)",
       n = NA, phylo = TRUE, tmb_rejects = TRUE,
       fixture = function() phylo_fixture(12, 6, 7171),
       resp = "y_beta", extra = ", sigma ~ 1",
       fam = function() drmTMB::beta()),

  list(id = "binomial_phylo_p12x6", family = "Binomial (logit), phylo(1|species)",
       n = NA, phylo = TRUE, tmb_rejects = TRUE, mu_only = TRUE,
       fixture = function() phylo_fixture(12, 6, 8181),
       resp = "y_binom", extra = "",
       fam = function() stats::binomial())
)

direct_cells <- list(
  list(id = "biv_lognormal_n500", family = "Bivariate lognormal", n = 500,
       fixture = function() biv_lognormal_fixture(500, 11),
       formula = function() bf(mu1 = y1 ~ x, mu2 = y2 ~ x, sigma1 = ~1, sigma2 = ~1, rho12 = ~1),
       fam = function() biv_lognormal(),
       jfam = "biv_lognormal",
       jformula = list(mu1 = "y1 ~ x", mu2 = "y2 ~ x", sigma1 = "sigma1 ~ 1",
                       sigma2 = "sigma2 ~ 1", rho12 = "rho12 ~ 1")),
  list(id = "biv_student_n800", family = "Bivariate Student-t", n = 800,
       fixture = function() biv_student_fixture(800, 21),
       formula = function() bf(mu1 = y1 ~ x, mu2 = y2 ~ x, sigma1 = ~1, sigma2 = ~1, nu = ~1, rho12 = ~1),
       fam = function() biv_student(),
       jfam = "biv_student",
       jformula = list(mu1 = "y1 ~ x", mu2 = "y2 ~ x", sigma1 = "sigma1 ~ 1",
                       sigma2 = "sigma2 ~ 1", nu = "nu ~ 1", rho12 = "rho12 ~ 1"))
)

if (pilot_mode) {
  # Small, diverse subset: proves plumbing (native family, phylo-with-TMB-
  # comparator family, phylo-TMB-rejects family, bivariate) without paying
  # for the full grid while five sibling agents share this machine.
  keep_ids <- c("gauss_locscale_n100", "poisson_phylo_p12x6", "gamma_phylo_p12x6",
                "biv_gaussian_rho12_n100")
  all_ids <- vapply(bridge_cells, `[[`, "", "id")
  bridge_cells <- bridge_cells[all_ids %in% keep_ids]
  direct_cells <- direct_cells[1]
}

rows <- list()
# Coerce any candidate cell value to exactly one scalar of the requested type.
# data.frame() errors ("differing number of rows") the instant one column's
# value is length 0 (e.g. a NULL field on an unexpected return shape) or
# length >1 -- this is the single choke point that guarantees every add_row()
# call produces a valid one-row data.frame no matter what a fit/error object
# actually contains.
sc <- function(x, type = c("character", "numeric", "integer", "logical")) {
  type <- match.arg(type)
  na <- switch(type, character = NA_character_, numeric = NA_real_,
               integer = NA_integer_, logical = NA)
  if (is.null(x) || length(x) != 1L) return(na)
  out <- tryCatch(switch(type, character = as.character(x), numeric = as.numeric(x),
                          integer = as.integer(x), logical = as.logical(x)),
                   error = function(e) na, warning = function(w) na)
  if (length(out) != 1L) na else out
}

add_row <- function(cell_id, family, n, p, engine, res, info, note = "") {
  err_msg <- if (!res$ok) {
    m <- res$err
    if (is.null(m) || length(m) == 0L || !nzchar(paste(m, collapse = ""))) "(no error message captured)" else paste(m, collapse = "; ")
  } else ""
  rows[[length(rows) + 1L]] <<- data.frame(
    cell_id = sc(cell_id, "character"), family = sc(family, "character"),
    n = sc(n, "integer"), p = sc(p, "integer"), engine = sc(engine, "character"),
    wall_clock_s_median = sc(if (res$ok) res$median_s else NA, "numeric"),
    n_reps = sc(if (res$ok) length(res$times) else 0L, "integer"),
    all_times_s = sc(if (res$ok) paste(sprintf("%.4f", res$times), collapse = ";") else "", "character"),
    converged = sc(if (res$ok) isTRUE(info$converged) else NA, "logical"),
    loglik = sc(if (res$ok) info$loglik else NA, "numeric"),
    iterations = sc(if (res$ok) info$iterations else NA, "integer"),
    optimizer = sc(if (res$ok) info$optimizer else NA, "character"),
    note = sc(if (res$ok) note else paste0("FAILED: ", err_msg), "character"),
    stringsAsFactors = FALSE
  )
}

cat("\n=== bridge cells (drmTMB engine='tmb' vs engine='julia') ===\n")
for (cell in bridge_cells) {
  fx <- cell$fixture()
  d <- if (is.list(fx) && !is.null(fx$data)) fx$data else fx
  tree <- if (is.list(fx) && !is.null(fx$tree)) fx$tree else NULL
  n <- if (!is.null(cell$n) && !is.na(cell$n)) cell$n else (if (is.list(fx)) fx$n else NA)
  p <- if (isTRUE(cell$phylo)) (if (is.list(fx)) fx$p else NA) else NA

  if (isTRUE(cell$phylo) && is.null(cell$formula)) {
    # non-Gaussian phylo cells built from a text formula (resp varies by cell)
    bf_txt <- if (isTRUE(cell$mu_only)) {
      sprintf("bf(%s ~ x + phylo(1 | species, tree = tree))", cell$resp)
    } else {
      sprintf("bf(%s ~ x + phylo(1 | species, tree = tree)%s)", cell$resp, cell$extra)
    }
    form_expr <- str2lang(bf_txt)
    formula_env <- environment()
    make_formula <- function() eval(form_expr, envir = formula_env)
  } else {
    make_formula <- cell$formula
  }

  cat(sprintf("-- %s (%s, n=%s, p=%s)\n", cell$id, cell$family,
              ifelse(is.na(n), "NA", n), ifelse(is.na(p), "NA", p)))

  res_tmb <- if (isTRUE(cell$tmb_rejects)) {
    list(ok = FALSE, err = "native TMB rejects phylo() for this family (\"structured-effect syntax is planned, not implemented\") -- NO_NATIVE_COMPARATOR, not a timing failure")
  } else {
    time_median(function() {
      drmTMB(make_formula(), family = cell$fam(), data = d, engine = "tmb")
    }, n_reps)
  }
  info_tmb <- if (res_tmb$ok) fit_info(res_tmb$val, "tmb") else NULL
  note_tmb <- if (isTRUE(cell$tmb_rejects) && !res_tmb$ok) "NO_NATIVE_COMPARATOR (see note in err)" else ""
  add_row(cell$id, cell$family, n, p, "tmb", res_tmb, info_tmb, note_tmb)
  cat(sprintf("   tmb:   %s\n", if (res_tmb$ok)
    sprintf("median=%.4fs conv=%s loglik=%.4f iters=%s", res_tmb$median_s,
            info_tmb$converged, info_tmb$loglik, info_tmb$iterations)
    else paste("FAILED:", substr(res_tmb$err, 1, 140))))

  res_jl <- time_median(function() {
    drmTMB(make_formula(), family = cell$fam(), data = d, engine = "julia")
  }, n_reps)
  info_jl <- if (res_jl$ok) fit_info(res_jl$val, "julia") else NULL
  add_row(cell$id, cell$family, n, p, "julia", res_jl, info_jl)
  cat(sprintf("   julia: %s\n", if (res_jl$ok)
    sprintf("median=%.4fs conv=%s loglik=%.4f iters=%s", res_jl$median_s,
            info_jl$converged, info_jl$loglik, info_jl$iterations)
    else paste("FAILED:", substr(res_jl$err, 1, 140))))

  if (res_tmb$ok && res_jl$ok) {
    ll_diff <- abs(info_tmb$loglik - info_jl$loglik)
    ratio <- res_tmb$median_s / res_jl$median_s
    flag <- if (ll_diff > 1e-4) " *** LOGLIK DISAGREEMENT > 1e-4 -- CORRECTNESS FLAG, not a speed win ***" else ""
    cat(sprintf("   ratio (tmb/julia) = %.2fx %s | loglik_diff=%.2e%s\n",
                ratio, if (ratio > 1) "(julia faster)" else "(tmb faster)", ll_diff, flag))
  }
}

cat("\n=== direct cells (drmTMB engine='tmb' vs raw DRModels.jl bridge call) ===\n")
drmTMB:::drm_julia_setup()
for (cell in direct_cells) {
  d <- cell$fixture()
  cat(sprintf("-- %s (%s, n=%s)\n", cell$id, cell$family, cell$n))

  res_tmb <- time_median(function() {
    drmTMB(cell$formula(), family = cell$fam(), data = d, engine = "tmb")
  }, n_reps)
  info_tmb <- if (res_tmb$ok) fit_info(res_tmb$val, "tmb") else NULL
  add_row(cell$id, cell$family, cell$n, NA, "tmb", res_tmb, info_tmb)
  cat(sprintf("   tmb:   %s\n", if (res_tmb$ok)
    sprintf("median=%.4fs conv=%s loglik=%.4f iters=%s", res_tmb$median_s,
            info_tmb$converged, info_tmb$loglik, info_tmb$iterations)
    else paste("FAILED:", substr(res_tmb$err, 1, 140))))

  res_jl <- time_median(function() {
    JuliaCall::julia_call("drmTMB_drm_bridge", cell$jformula, cell$jfam,
                          as.list(d), NULL, NULL)
  }, n_reps)
  info_jl <- if (res_jl$ok) list(
    converged = NA, loglik = tryCatch(res_jl$val$loglik, error = function(e) NA_real_),
    iterations = tryCatch(res_jl$val$iterations, error = function(e) NA_integer_),
    optimizer = "DRModels.jl direct bridge call (JuliaCall), engine='julia' does not route this family"
  ) else NULL
  add_row(cell$id, cell$family, cell$n, NA, "julia_direct", res_jl, info_jl,
          "direct DRModels.jl bridge call -- engine='julia' never routes this family (drmTMB gate)")
  cat(sprintf("   julia (direct bridge): %s\n", if (res_jl$ok)
    sprintf("median=%.4fs loglik=%.4f", res_jl$median_s, info_jl$loglik)
    else paste("FAILED:", substr(res_jl$err, 1, 140))))

  if (res_tmb$ok && res_jl$ok) {
    ll_diff <- abs(info_tmb$loglik - info_jl$loglik)
    ratio <- res_tmb$median_s / res_jl$median_s
    flag <- if (ll_diff > 1e-4) " *** LOGLIK DISAGREEMENT > 1e-4 -- CORRECTNESS FLAG ***" else ""
    cat(sprintf("   ratio (tmb/julia_direct) = %.2fx %s | loglik_diff=%.2e%s\n",
                ratio, if (ratio > 1) "(julia faster)" else "(tmb faster)", ll_diff, flag))
  }
}

tab <- do.call(rbind, rows)
dir.create(dirname(out_tsv), showWarnings = FALSE, recursive = TRUE)
write.table(tab, out_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
cat("\nwrote ", out_tsv, "\n", sep = "")
cat("pilot_mode=", pilot_mode, " n_reps=", n_reps, "\n", sep = "")
