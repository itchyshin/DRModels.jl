# parity_fixture.R — native-vs-Julia same-target comparison per bridge cell.
#
# The promotion gate. A drmTMB capability row may only move toward `supported`
# on evidence that `engine = "julia"` and `engine = "tmb"` fit the SAME target
# and agree — matching coefficients and logLik within a declared tolerance.
# Direct DRModels.jl evidence is not R-via-Julia bridge support, so this must run
# through drmTMB itself.
#
#   DRM_JL_PATH=/path/to/DRModels.jl Rscript tools/parity_fixture.R
#
# Writes a TSV of results next to this script's --out argument (default below).

source("tools/parity_numeric.R")
suppressMessages(library(drmTMB))

out_path <- "docs/dev-log/evidence/parity-fixtures.tsv"
tol <- 1e-4

cells <- list(
  list(
    id      = "base_gaussian_location_scale",
    label   = "Gaussian location-scale, fixed effects",
    build   = function() {
      set.seed(20260814)
      n <- 120
      x <- rnorm(n); z <- rnorm(n)
      data.frame(y = 0.4 + 0.9 * x + exp(-0.3 + 0.25 * z) * rnorm(n), x = x, z = z)
    },
    formula = function() bf(y ~ x, sigma ~ z),
    family  = function() gaussian()
  ),
  list(
    id      = "base_gaussian_intercept_only",
    label   = "Gaussian, intercept-only sigma",
    build   = function() {
      set.seed(99)
      n <- 100
      x <- rnorm(n)
      data.frame(y = 1.2 - 0.6 * x + 0.7 * rnorm(n), x = x)
    },
    formula = function() bf(y ~ x, sigma ~ 1),
    family  = function() gaussian()
  ),
  # Gaussian observed-response mask. The drmTMB ledger row
  # `gaussian_response_mask` (inst/extdata/julia-capabilities.tsv:5) had no
  # receipt row here, so drmTMB's parity scoreboard read the capability
  # `Missing-response handling (native, per fitted route)` as UNCITED even
  # though both engines fit it. Same fixture drmTMB's own
  # tests/testthat/test-julia-missing.R uses, so the two are about one target.
  # `fit_args` (below) is what lets a cell carry the `missing` policy.
  list(
    id       = "gaussian_response_mask",
    label    = "Gaussian observed-response mask (miss_control(response = 'include'))",
    build    = function() {
      set.seed(1)
      n <- 60
      x <- rnorm(n)
      y <- 0.3 + 0.5 * x + rnorm(n) * exp(0.1 * x)
      y[1:6] <- NA
      data.frame(y = y, x = x)
    },
    formula  = function() bf(y ~ x, sigma ~ x),
    family   = function() gaussian(),
    fit_args = function() list(missing = miss_control(response = "include"))
  )
)

# Fixed-effect non-Gaussian cells. The R gate REFUSES these through
# `engine = "julia"` ("routes <family> models only with a phylo() random
# intercept"), and the gate registry's stated reason is the absence of a
# coefficient-scale parity test. So compare native TMB against the DRModels.jl bridge
# payload directly — the payload the bridge WOULD return if the gate opened.
# This is the evidence the gate's own `review_due` asks for.
fe_cells <- list(
  list(id = "fe_poisson", label = "Poisson, fixed effects", jfam = "poisson",
       family = function() poisson(),
       build = function() { set.seed(4242); n <- 150; x <- rnorm(n)
                            data.frame(y = rpois(n, exp(0.6 + 0.4 * x)), x = x) }),
  list(id = "fe_nbinom2", label = "NegBinomial2, fixed effects", jfam = "nbinom2",
       family = function() nbinom2(),
       build = function() { set.seed(4242); n <- 150; x <- rnorm(n)
                            data.frame(y = rnbinom(n, mu = exp(0.6 + 0.4 * x), size = 3), x = x) }),
  list(id = "fe_gamma", label = "Gamma (log link), fixed effects", jfam = "gamma",
       family = function() Gamma(link = "log"),
       build = function() { set.seed(4242); n <- 150; x <- rnorm(n)
                            data.frame(y = rgamma(n, shape = 4, rate = 4 / exp(0.5 + 0.3 * x)), x = x) }),
  # plain_binomial_nonphylo. Added 2026-08-24: tools/parity_crosscheck.py found this
  # capability row carried a committed Route-1 fixture (test/parity/fixtures/
  # binomial-trials, drmTMB 0.6.0) but NO live engine="tmb" vs engine="julia"
  # measurement -- offline replay only. Mirrors that fixture's shape
  # (cbind(successes, failures), logit link, mean-only) so the two routes are
  # about the same capability, while remaining a separate draw.
  list(id = "plain_binomial_nonphylo", label = "Binomial (logit, trials), fixed effects",
       jfam = "binomial",
       family = function() binomial(),
       formula = function() bf(cbind(successes, failures) ~ x),
       build = function() { set.seed(4242); n <- 150; x <- rnorm(n)
                            size <- 12L
                            p <- plogis(0.3 + 0.7 * x)
                            s <- rbinom(n, size, p)
                            data.frame(successes = s, failures = size - s, x = x) }),
  # skew_normal (A4, 2026-09-05). drmTMB's `skew_normal()` (dpars mu, sigma, nu;
  # public moment form: mu = E[y], sigma = SD[y], nu = Azzalini's slant) through
  # engine = "julia". Needs the `_bridge_family("skew_normal") -> SkewNormal()`
  # case in src/bridge.jl. Same draw as drmTMB's
  # tests/testthat/test-skew-normal-location-scale.R (n = 500, seed 20260608,
  # nu = 1.6) so the receipt is about the committed fixture.
  list(id = "skew_normal", label = "Skew-normal (mu ~ x, sigma ~ z, nu ~ 1), fixed effects",
       jfam = "skew_normal",
       family = function() skew_normal(),
       formula = function() bf(y ~ x, sigma ~ z, nu ~ 1),
       build = function() {
         set.seed(20260608); n <- 500; nu <- 1.6
         x <- rnorm(n); z <- rnorm(n)
         mu <- 0.20 + 0.45 * x; sigma <- exp(-0.35 + 0.18 * z)
         delta <- nu / sqrt(1 + nu^2); ms <- delta * sqrt(2 / pi)
         omega <- sigma / sqrt(1 - ms^2); xi <- mu - omega * ms
         data.frame(y = xi + omega * (delta * abs(rnorm(n)) + sqrt(1 - delta^2) * rnorm(n)),
                    x = x, z = z) })
)

# Bivariate lognormal (drmTMB biv_lognormal). Compared native-TMB vs the DRModels.jl
# bridge payload: `engine = "julia"` does not route this family at the anchor, so
# this is capability parity for the Julia implementation, NOT bridge admission.
biv_cells <- list(
  list(id = "biv_lognormal", label = "Bivariate lognormal, fixed effects",
       jfam = "biv_lognormal",
       family = function() biv_lognormal(),
       formula = function() bf(mu1 = y1 ~ x, mu2 = y2 ~ x,
                               sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1),
       jformula = list(mu1 = "y1 ~ x", mu2 = "y2 ~ x", sigma1 = "sigma1 ~ 1",
                       sigma2 = "sigma2 ~ 1", rho12 = "rho12 ~ 1"),
       build = function() {
         set.seed(11); n <- 500; x <- rnorm(n)
         s1 <- 0.5; s2 <- 0.8; rho <- 0.6
         z1 <- rnorm(n); z2 <- rho * z1 + sqrt(1 - rho^2) * rnorm(n)
         data.frame(y1 = exp(0.4 + 0.9 * x + s1 * z1),
                    y2 = exp(-0.2 + 0.5 * x + s2 * z2), x = x)
       }),
  list(id = "biv_student", label = "Bivariate Student-t, fixed effects",
       jfam = "biv_student",
       family = function() biv_student(),
       formula = function() bf(mu1 = y1 ~ x, mu2 = y2 ~ x,
                               sigma1 = ~ 1, sigma2 = ~ 1, nu = ~ 1, rho12 = ~ 1),
       jformula = list(mu1 = "y1 ~ x", mu2 = "y2 ~ x", sigma1 = "sigma1 ~ 1",
                       sigma2 = "sigma2 ~ 1", nu = "nu ~ 1", rho12 = "rho12 ~ 1"),
       build = function() {
         set.seed(21); n <- 800; x <- rnorm(n)
         s1 <- 0.7; s2 <- 1.1; rho <- 0.5; nu <- 6
         z1 <- rnorm(n); z2 <- rho * z1 + sqrt(1 - rho^2) * rnorm(n)
         sh <- sqrt(nu / rchisq(n, df = nu))
         data.frame(y1 = 0.5 + 0.8 * x + s1 * z1 * sh,
                    y2 = -0.3 + 0.4 * x + s2 * z2 * sh, x = x)
       })
)

rows <- list()
for (cell in cells) {
  d <- cell$build()
  res <- list(
    capability_id = cell$id, label = cell$label,
    status = NA_character_, max_abs_coef_diff = NA_real_,
    loglik_tmb = NA_real_, loglik_julia = NA_real_, loglik_diff = NA_real_,
    tolerance = tol, note = ""
  )

  # A cell may need more than formula/family/data -- the observed-response mask
  # cell carries a `missing` policy. Without this hook the loop would silently
  # fit the DEFAULT policy and record a receipt for the wrong target.
  extra <- if (is.null(cell$fit_args)) list() else cell$fit_args()
  fit_one <- function(engine) {
    try(do.call(drmTMB, c(list(cell$formula(), family = cell$family(), data = d,
                               engine = engine), extra)), silent = TRUE)
  }
  ft <- fit_one("tmb")
  fj <- fit_one("julia")

  if (inherits(ft, "try-error")) {
    res$status <- "NATIVE_FAILED"; res$note <- conditionMessage(attr(ft, "condition"))
  } else if (inherits(fj, "try-error")) {
    res$status <- "JULIA_FAILED"; res$note <- conditionMessage(attr(fj, "condition"))
  } else {
    ct <- unlist(fixef(ft)); cj <- unlist(fixef(fj))
    comparison <- parity_numeric(ct, cj, tol)
    res$max_abs_coef_diff <- comparison$max_abs_diff
    res$loglik_tmb <- as.numeric(logLik(ft))
    res$loglik_julia <- as.numeric(logLik(fj))
    res$loglik_diff <- abs(res$loglik_tmb - res$loglik_julia)
    agree <- comparison$pass && is.finite(res$loglik_diff) && res$loglik_diff < tol
    res$status <- if (agree) "PARITY_PASS" else "PARITY_FAIL"
    res$note <- comparison$reason
  }
  rows[[length(rows) + 1L]] <- as.data.frame(res, stringsAsFactors = FALSE)
  cat(sprintf("%-32s %-14s coef_diff=%.3e  loglik_diff=%.3e\n",
              res$capability_id, res$status,
              res$max_abs_coef_diff, res$loglik_diff))
}

drmTMB:::drm_julia_setup()
for (cell in fe_cells) {
  d <- cell$build()
  res <- list(
    capability_id = cell$id, label = cell$label,
    status = NA_character_, max_abs_coef_diff = NA_real_,
    loglik_tmb = NA_real_, loglik_julia = NA_real_, loglik_diff = NA_real_,
    ## MEASURE the drmTMB version, never assert it. This note used to hardcode
    ## "drmTMB 0.7.0", so the TSV claimed a version it had not checked. That is
    ## not hypothetical: on 2026-08-24 a concurrent `devtools::test()` in the
    ## drmTMB checkout transiently put 0.6.0.9000 on the search path, and a run
    ## in that window would have recorded "0.7.0" while measuring something else.
    tolerance = tol,
    note = sprintf("R-via-Julia bridge parity (engine='julia'), drmTMB %s",
                   as.character(utils::packageVersion("drmTMB")))
  )
  # Most FE cells are a plain `y ~ x`. A cell may override this when its response
  # is not a bare `y` -- e.g. the binomial trials cell, whose response is
  # cbind(successes, failures). Without the override the loop would silently fit
  # the wrong formula rather than erroring.
  fml <- if (is.null(cell$formula)) bf(y ~ x) else cell$formula()
  ft <- try(drmTMB(fml, family = cell$family(), data = d, engine = "tmb"), silent = TRUE)
  jb <- try(drmTMB(fml, family = cell$family(), data = d, engine = "julia"), silent = TRUE)
  if (inherits(ft, "try-error")) {
    res$status <- "NATIVE_FAILED"
  } else if (inherits(jb, "try-error")) {
    res$status <- "JULIA_FAILED"
  } else {
    ct <- unlist(fixef(ft)); cj <- unlist(fixef(jb))
    comparison <- parity_numeric(ct, cj, tol)
    res$max_abs_coef_diff <- comparison$max_abs_diff
    res$loglik_tmb <- as.numeric(logLik(ft)); res$loglik_julia <- as.numeric(logLik(jb))
    res$loglik_diff <- abs(res$loglik_tmb - res$loglik_julia)
    res$status <- if (comparison$pass && is.finite(res$loglik_diff) && res$loglik_diff < tol)
      "PARITY_PASS" else "PARITY_FAIL"
    res$note <- paste(res$note, comparison$reason, sep = "; ")
  }
  rows[[length(rows) + 1L]] <- as.data.frame(res, stringsAsFactors = FALSE)
  cat(sprintf("%-32s %-14s coef_diff=%.3e  loglik_diff=%.3e\n",
              res$capability_id, res$status, res$max_abs_coef_diff, res$loglik_diff))
}

for (cell in biv_cells) {
  d <- cell$build()
  res <- list(
    capability_id = cell$id, label = cell$label,
    status = NA_character_, max_abs_coef_diff = NA_real_,
    loglik_tmb = NA_real_, loglik_julia = NA_real_, loglik_diff = NA_real_,
    tolerance = tol, note = "capability parity vs DRModels.jl bridge payload; not bridge admission"
  )
  ft <- try(drmTMB(cell$formula(), family = cell$family(), data = d, engine = "tmb"),
            silent = TRUE)
  jb <- try(JuliaCall::julia_call("drmTMB_drm_bridge", cell$jformula, cell$jfam,
                                  as.list(d), NULL, NULL), silent = TRUE)
  if (inherits(ft, "try-error")) {
    res$status <- "NATIVE_FAILED"
  } else if (inherits(jb, "try-error")) {
    res$status <- "JULIA_FAILED"
  } else {
    ct <- unlist(fixef(ft))
    cj <- unlist(jb$coefficients, use.names = FALSE)
    if (length(jb$coef_names) == length(cj)) names(cj) <- as.character(jb$coef_names)
    comparison <- parity_numeric(ct, cj, tol)
    res$max_abs_coef_diff <- comparison$max_abs_diff
    res$loglik_tmb <- as.numeric(logLik(ft)); res$loglik_julia <- jb$loglik
    res$loglik_diff <- abs(res$loglik_tmb - res$loglik_julia)
    res$status <- if (comparison$pass && is.finite(res$loglik_diff) && res$loglik_diff < tol)
      "PARITY_PASS" else "PARITY_FAIL"
    res$note <- paste(res$note, comparison$reason, sep = "; ")
  }
  rows[[length(rows) + 1L]] <- as.data.frame(res, stringsAsFactors = FALSE)
  cat(sprintf("%-32s %-14s coef_diff=%.3e  loglik_diff=%.3e\n",
              res$capability_id, res$status, res$max_abs_coef_diff, res$loglik_diff))
}

tab <- do.call(rbind, rows)
dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write.table(tab, out_path, sep = "\t", row.names = FALSE, quote = FALSE)
cat("\nwrote ", out_path, "\n", sep = "")
cat("OVERALL: ", if (all(tab$status == "PARITY_PASS")) "ALL CELLS PASS" else "SOME CELLS FAILED", "\n", sep = "")

# A failed row must also fail automation, after preserving the complete table.
if (!all(tab$status == "PARITY_PASS")) quit(status = 1L)
