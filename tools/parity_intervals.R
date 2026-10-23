# parity_intervals.R — the interval TRIO at parity: Wald, profile, bootstrap.
#
# Point estimates and logLik agreeing is not the same as INFERENCE agreeing. Both
# engines advertise the same three interval methods:
#
#   confint(object, method = c("wald", "profile", "bootstrap"))
#
# so the capability SURFACE matches on both sides. This script asks the question
# the surface cannot answer: for the same fit, does each method actually run on
# each engine, and do the intervals agree?
#
# Capability parity first. A method that errors on one side is a capability gap;
# a method that runs on both but returns materially different intervals is a
# correctness gap. Both are recorded, and they are NOT the same finding.
#
#   DRM_JL_PATH=/path/to/DRModels.jl Rscript tools/parity_intervals.R
#
# Deliberately uses the INSTALLED drmTMB, like every other parity script here --
# never devtools::load_all() on the source tree, which on 2026-08-24 silently
# measured a dirty 0.6.0.9000 checkout while another lane was editing it.

suppressMessages(library(drmTMB))

out_path <- "docs/dev-log/evidence/parity-intervals.tsv"
R_boot   <- as.integer(Sys.getenv("DRM_INT_R", "99"))
boot_seed <- 20260824L
tol_rel  <- 1e-3   # relative agreement bar for an interval endpoint

message(sprintf("parity_intervals.R: INSTALLED drmTMB %s | R_boot=%d | seed=%d",
                as.character(utils::packageVersion("drmTMB")), R_boot, boot_seed))

## ---- cells ---------------------------------------------------------------

cells <- list(
  list(
    id    = "gauss_locscale_fe",
    label = "Gaussian location-scale, fixed effects",
    build = function() {
      set.seed(20260824); n <- 200; x <- rnorm(n); z <- rnorm(n)
      data.frame(y = 0.4 + 0.9 * x + exp(-0.3 + 0.25 * z) * rnorm(n), x = x, z = z)
    },
    formula = function() bf(y ~ x, sigma ~ z),
    family  = function() gaussian()
  ),
  list(
    id    = "gauss_mean_only",
    label = "Gaussian, mean-only (sigma ~ 1)",
    build = function() {
      set.seed(20260824); n <- 200; x <- rnorm(n)
      data.frame(y = 0.4 + 0.9 * x + rnorm(n), x = x)
    },
    formula = function() bf(y ~ x, sigma ~ 1),
    family  = function() gaussian()
  ),
  list(
    id    = "poisson_fe",
    label = "Poisson, fixed effects",
    build = function() {
      set.seed(20260824); n <- 200; x <- rnorm(n)
      data.frame(y = rpois(n, exp(0.6 + 0.4 * x)), x = x)
    },
    formula = function() bf(y ~ x),
    family  = function() poisson()
  ),
list(
    id    = "fe_beta",
    label = "Beta (logit mu), fixed effects",
    build = function() {
      set.seed(4242); n <- 150; x <- rnorm(n)
      mu <- plogis(0.3 + 0.7 * x); phi <- 8
      data.frame(y = rbeta(n, mu * phi, (1 - mu) * phi), x = x)
    },
    formula = function() bf(y ~ x, sigma ~ 1),
    family  = function() beta()
  ),
# gamma_fe. Gamma (log link), mean-only, through engine = "julia". The build()
  # is copied VERBATIM from the `fe_gamma` cell in tools/parity_fixture.R:83-86
  # so the interval receipt and the coefficient receipt are about ONE target,
  # not two separate draws. That cell carries no `formula` element (the fe_cells
  # loop defaults to bf(y ~ x)); this file's loop calls cell$formula()
  # unconditionally, so it is spelled out here.
  list(
    id    = "fe_gamma",
    label = "Gamma (log link), fixed effects",
    build = function() {
      set.seed(4242); n <- 150; x <- rnorm(n)
      data.frame(y = rgamma(n, shape = 4, rate = 4 / exp(0.5 + 0.3 * x)), x = x)
    },
    formula = function() bf(y ~ x),
    family  = function() Gamma(link = "log")
  ),
list(
    id    = "fe_lognormal",
    label = "Lognormal, fixed effects",
    build = function() {
      set.seed(4242); n <- 150; x <- rnorm(n)
      data.frame(y = exp(0.6 + 0.4 * x + 0.5 * rnorm(n)), x = x)
    },
    formula = function() bf(y ~ x),
    family  = function() lognormal()
  ),
list(
    id    = "fe_nbinom2",
    label = "NegBinomial2, fixed effects",
    build = function() {
      set.seed(4242); n <- 150; x <- rnorm(n)
      data.frame(y = rnbinom(n, mu = exp(0.6 + 0.4 * x), size = 3), x = x)
    },
    formula = function() bf(y ~ x),
    family  = function() nbinom2()
  ),
# skew_normal, fixed effects (measured 2026-09-07). Profile is at parity: both
  # engines return a finite interval for `fixef:mu:x` agreeing to ~2e-6 relative.
  # Bootstrap is a MEASURED one-sided gap: native TMB gives 19/19 successful
  # refits, while the Julia side fails every replicate because DRModels.jl's
  # `_simulate_once` (src/gaussian_core.jl) has no SkewNormal branch and hits its
  # terminal "simulate: not yet supported" error. Expect UNSUPPORTED_JULIA on the
  # bootstrap row until that draw kernel gains the family. Fixture draw is the one
  # already committed in tools/parity_fixture.R (n = 500, seed 20260608, nu = 1.6).
  list(
    id    = "fe_skew_normal",
    label = "Skew-normal (mu ~ x, sigma ~ z, nu ~ 1), fixed effects",
    build = function() {
      set.seed(20260608); n <- 500; nu <- 1.6
      x <- rnorm(n); z <- rnorm(n)
      mu <- 0.20 + 0.45 * x; sigma <- exp(-0.35 + 0.18 * z)
      delta <- nu / sqrt(1 + nu^2); ms <- delta * sqrt(2 / pi)
      omega <- sigma / sqrt(1 - ms^2); xi <- mu - omega * ms
      data.frame(y = xi + omega * (delta * abs(rnorm(n)) + sqrt(1 - delta^2) * rnorm(n)),
                 x = x, z = z)
    },
    formula = function() bf(y ~ x, sigma ~ z, nu ~ 1),
    family  = function() skew_normal()
  ),
list(
    id    = "zi_nbinom2",
    label = "Zero-inflated NB2, fixed effects (zi ~ 1)",
    build = function() {
      set.seed(20260824); n <- 400; x <- rnorm(n)
      mu <- exp(0.45 - 0.30 * x); sigma <- exp(-0.75); zi <- plogis(-1.15)
      data.frame(
        y = ifelse(runif(n) < zi, 0L, rnbinom(n, size = 1 / sigma^2, mu = mu)),
        x = x
      )
    },
    formula = function() bf(y ~ x, sigma ~ 1, zi ~ 1),
    family  = function() nbinom2()
  ),
list(
    id    = "zi_poisson",
    label = "Zero-inflated Poisson, fixed effects (zi ~ x)",
    build = function() {
      set.seed(20260824); n <- 400; x <- rnorm(n)
      lambda <- exp(0.6 + 0.4 * x); pz <- plogis(-0.8 + 0.5 * x)
      y <- ifelse(rbinom(n, 1, pz) == 1L, 0L, rpois(n, lambda))
      data.frame(y = y, x = x)
    },
    formula = function() bf(y ~ x, zi ~ x),
    family  = function() poisson()
  ),
# fe_student, fixed effects (measured 2026-09-07). Profile and bootstrap are BOTH
  # at parity through engine = "julia". Profile: both engines return a finite
  # interval for `fixef:mu:x` agreeing to 9.2e-07 absolute (1.5e-06 relative,
  # width ratio 1.0000133). Bootstrap at R = 99 / seed 20260824: native TMB 99/99
  # refits [0.499768, 0.602050], Julia 98/99 refits [0.499710, 0.594623] — scored
  # as DISTRIBUTIONAL OVERLAP (TRUE), which is the only correct bootstrap test
  # since each engine draws its own resamples. Note the Julia side emits a
  # numerically-singular-Hessian warning for coordinate 5 (the nu intercept,
  # rcond 9.1e-09); it does not touch the mu:x target measured here. There was no
  # univariate student cell in either parity script, so the draw is taken verbatim
  # from the committed fixture in drmTMB tests/testthat/test-student-location-scale.R
  # (n = 600, seed 20260509, nu = 8) — the same provenance rule the fe_skew_normal
  # cell follows.
  list(
    id    = "fe_student",
    label = "Student-t (mu ~ x, sigma ~ z, nu ~ 1), fixed effects",
    build = function() {
      set.seed(20260509); n <- 600
      dat <- data.frame(x = rnorm(n), z = rnorm(n))
      mu <- 0.25 + 0.6 * dat$x
      sigma <- exp(-0.3 + 0.25 * dat$z)
      nu <- 2 + exp(log(6))
      q <- qt((seq_len(n) - 0.5) / n, df = nu)
      dat$y <- mu + sigma * sample(q)
      dat
    },
    formula = function() bf(y ~ x, sigma ~ z, nu ~ 1),
    family  = function() student()
  ),
# fe_truncated_nbinom2. Zero-truncated NegBinomial2 (log mu, log sigma), fixed
  # effects, measured 2026-09-07. NO fixture for this family existed in
  # tools/parity_fixture.R or in this file, so the build() below is NEW and this
  # cell rests on a SINGLE seed rather than a draw already committed elsewhere.
  # Zeros are rejected per observation, so the draw is genuinely truncated
  # (min(y) = 1, n_zero = 0) and the truncated likelihood is the right one.
  # Both engines reproduce the same fit (logLik -667.3957, fixef:mu:x 0.3645748)
  # and the whole interval trio runs on both: profile agrees to 1.8e-5 relative
  # (max abs 8.97e-06), and the bootstrap intervals OVERLAP -- which is the only
  # correct bootstrap test, since each engine draws its own resamples.
  list(
    id    = "fe_truncated_nbinom2",
    label = "Zero-truncated NegBinomial2, fixed effects",
    build = function() {
      set.seed(20260824); n <- 400; x <- rnorm(n)
      mu <- exp(0.6 + 0.4 * x); sigma <- exp(-0.5); size <- 1 / sigma^2
      y <- integer(n)
      for (i in seq_len(n)) {
        repeat { v <- rnbinom(1, size = size, mu = mu[i]); if (v > 0L) break }
        y[i] <- v
      }
      data.frame(y = y, x = x)
    },
    formula = function() bf(y ~ x, sigma ~ 1),
    family  = function() truncated_nbinom2()
  ),
# fe_tweedie. Tweedie (log mu, log sigma, logit12 nu), fixed effects, measured
  # 2026-09-07. Profile and bootstrap are BOTH at parity: profile endpoints agree
  # to 7.29e-06 absolute (~1.3e-05 relative) and both engines return 99/99
  # successful bootstrap refits with OVERLAPPING intervals for `fixef:mu:x`.
  # NOTE ON THE FIXTURE: unlike fe_gamma / fe_skew_normal, this draw is NOT copied
  # from tools/parity_fixture.R -- that file carries no tweedie cell. The shape
  # follows the fe_* house style (seed 4242, n = 150) and the draw kernel is the
  # one already committed at tools/parity_ranef.R:270
  # (drmTMB:::rtweedie_compound, phi = 1.2, power = 1.5), with the random
  # intercept dropped so the DGP is purely fixed-effect. Single-seed measurement.
  # `nu` is a FREE dpar in drmTMB 0.7.1 (logit12 link) and is estimated here; it is
  # spelled out in the formula because this file's loop calls cell$formula()
  # unconditionally.
  list(
    id    = "fe_tweedie",
    label = "Tweedie (log mu, log sigma, logit12 nu), fixed effects",
    build = function() {
      set.seed(4242); n <- 150; x <- rnorm(n)
      mu <- exp(0.5 + 0.3 * x)
      data.frame(y = drmTMB:::rtweedie_compound(n, mu = mu, phi = 1.2, power = 1.5),
                 x = x)
    },
    formula = function() bf(y ~ x, sigma ~ 1, nu ~ 1),
    family  = function() tweedie()
  ),
# fe_zero_one_beta. Zero-one-inflated beta (mu ~ x, sigma ~ 1, zoi ~ 1, coi ~ 1),
  # measured 2026-09-07. NOTE the provenance: unlike fe_gamma and fe_skew_normal, this
  # build() is NOT copied from tools/parity_fixture.R -- that file carries no zero-one
  # cell, so this is a fresh single-seed draw written for the interval receipt. If a
  # coefficient-parity cell for this family is added later, copy THIS build() into it
  # rather than drawing again, so the two receipts stay about one target.
  # The draw deliberately lands both inflation components (37 zeros, 61 ones, 302
  # interior at n = 400); a draw that empties either one makes zoi/coi unidentified.
  # Measured at parity: profile agrees to 3.3e-06 absolute on `fixef:mu:x`, and both
  # engines return 99/99 successful bootstrap refits whose intervals OVERLAP
  # (TMB [0.69647, 0.86950] sits inside Julia [0.69615, 0.89163]).
  list(
    id    = "fe_zero_one_beta",
    label = "Zero-one-inflated beta (mu ~ x, sigma ~ 1, zoi ~ 1, coi ~ 1), fixed effects",
    build = function() {
      set.seed(20260824); n <- 400; x <- rnorm(n)
      mu  <- plogis(0.30 + 0.70 * x); phi <- 8
      zoi <- plogis(-1.20); coi <- plogis(0.40)
      infl <- rbinom(n, 1, zoi); one <- rbinom(n, 1, coi)
      yb   <- rbeta(n, mu * phi, (1 - mu) * phi)
      data.frame(y = ifelse(infl == 1L, as.numeric(one), yb), x = x)
    },
    formula = function() bf(y ~ x, sigma ~ 1, zoi ~ 1, coi ~ 1),
    family  = function() zero_one_beta()
  )
)

methods <- c("wald", "profile", "bootstrap")

## ---- helpers -------------------------------------------------------------

one_line <- function(x) gsub("[\r\n]+", " ", paste(as.character(x), collapse = " "))

# Return a named list(lower=, upper=, ok=, note=) for one engine x method.
# `parm` matters: profile intervals REQUIRE an explicit target on both engines
# ("Profile confidence intervals currently require explicit target names"), so
# calling confint(method="profile") with no parm reports UNSUPPORTED on both
# sides and hides the real asymmetry underneath.
get_ci <- function(fit, method, parm = NULL) {
  res <- tryCatch({
    ci <- if (method == "bootstrap") {
      if (is.null(parm)) confint(fit, method = method, R = R_boot, seed = boot_seed)
      else confint(fit, parm = parm, method = method, R = R_boot, seed = boot_seed)
    } else if (!is.null(parm)) {
      confint(fit, parm = parm, method = method)
    } else {
      confint(fit, method = method)
    }
    ci <- as.data.frame(ci)
    list(ok = TRUE, ci = ci, note = "")
  }, error = function(e) list(ok = FALSE, ci = NULL, note = one_line(conditionMessage(e))))
  res
}

# Align two CI tables on their common rows, comparing endpoints by row name where
# possible and positionally otherwise (the two engines spell coefficient names
# differently -- `mu:x` native vs `mu_x` julia -- so name matching must normalise).
norm_names <- function(x) gsub(":", "_", x, fixed = TRUE)

compare_ci <- function(a, b) {
  # confint.drmTMB returns NAMED columns (parm, level, lower, upper, tmb_parameter,
  # ...), not positional endpoints. Reading columns 1:2 as lower/upper silently
  # coerced the `parm` STRINGS to NA and produced width_ratio = -Inf -- a
  # comparison that looked like it ran and compared nothing.
  # Key on `parm`, NOT `tmb_parameter`. `parm` is the shared, human-meaningful id
  # and matches across engines ("fixef:mu:x"). `tmb_parameter` does not: native
  # TMB reports internal block names (beta_mu, beta_sigma) which are NOT UNIQUE
  # across rows, while the Julia bridge reports flat coefficient names
  # (mu_(Intercept)). Keying on it matched nothing and every cell came back
  # INDETERMINATE while the endpoints in fact agreed exactly.
  key <- function(x) norm_names(as.character(x$parm))
  need <- c("lower", "upper")
  if (!all(need %in% names(a)) || !all(need %in% names(b))) {
    return(list(n = 0L, max_abs = NA_real_, max_rel = NA_real_,
                width_ratio = NA_real_, how = "no lower/upper columns",
                only_tmb = NA_integer_, only_julia = NA_integer_))
  }
  ka <- key(a); kb <- key(b)
  common <- intersect(ka, kb)
  only_a <- setdiff(ka, kb); only_b <- setdiff(kb, ka)
  if (length(common) == 0L) {
    return(list(n = 0L, max_abs = NA_real_, max_rel = NA_real_,
                width_ratio = NA_real_, how = "no shared parameters",
                only_tmb = length(only_a), only_julia = length(only_b)))
  }
  ia <- match(common, ka); ib <- match(common, kb)
  la <- as.numeric(a$lower[ia]); ua <- as.numeric(a$upper[ia])
  lb <- as.numeric(b$lower[ib]); ub <- as.numeric(b$upper[ib])
  d  <- c(abs(la - lb), abs(ua - ub))
  sc <- c(pmax(abs(la), abs(lb)), pmax(abs(ua), abs(ub)))
  wa <- ua - la; wb <- ub - lb
  ok <- is.finite(wa) & is.finite(wb) & wa > 0 & wb > 0
  wr <- if (any(ok)) max(pmax(wa[ok] / wb[ok], wb[ok] / wa[ok])) else NA_real_
  list(n = length(common),
       max_abs = suppressWarnings(max(d, na.rm = TRUE)),
       max_rel = suppressWarnings(max(d / pmax(sc, 1e-12), na.rm = TRUE)),
       # Width ratio is its own gate: two intervals can share endpoints on the
       # parameters they both report and still disagree on SPREAD elsewhere.
       width_ratio = wr,
       how = sprintf("by name (%d shared)", length(common)),
       only_tmb = length(only_a), only_julia = length(only_b))
}

## ---- run -----------------------------------------------------------------

rows <- list()
for (cell in cells) {
  d <- cell$build()
  ft <- tryCatch(drmTMB(cell$formula(), family = cell$family(), data = d, engine = "tmb"),
                 error = function(e) e)
  fj <- tryCatch(drmTMB(cell$formula(), family = cell$family(), data = d, engine = "julia"),
                 error = function(e) e)
  cat(sprintf("\n-- %s (%s)\n", cell$id, cell$label))

  for (m in methods) {
    rec <- list(cell_id = cell$id, label = cell$label, method = m,
                status = NA_character_, n_params = NA_integer_,
                max_abs_diff = NA_real_, max_rel_diff = NA_real_,
                width_ratio = NA_real_, matched = NA_character_,
                only_tmb = NA_integer_, only_julia = NA_integer_,
                tolerance = tol_rel, note = "")

    if (inherits(ft, "error") || inherits(fj, "error")) {
      rec$status <- "FIT_FAILED"
      rec$note <- one_line(if (inherits(ft, "error")) conditionMessage(ft) else conditionMessage(fj))
    } else {
      # Profile needs an explicit target; use a fixed effect present in every cell.
      # BOOTSTRAP NEEDS AN EXPLICIT parm TOO (2026-09-07). With parm = NULL the
      # bridge refuses: "`method = \"bootstrap\"` confidence intervals require an
      # explicit `parm` naming exactly one target", and every julia bootstrap cell
      # recorded UNSUPPORTED_JULIA -- a DRIVER convention gap that was being read as
      # an engine limitation. Measured otherwise: with an explicit parm the bridge
      # returns finite bootstrap intervals for 7 non-Gaussian families.
      parm_m <- if (m %in% c("profile", "bootstrap")) "fixef:mu:x" else NULL
      a <- get_ci(ft, m, parm_m); b <- get_ci(fj, m, parm_m)
      if (!a$ok && !b$ok) {
        rec$status <- "UNSUPPORTED_BOTH"; rec$note <- one_line(a$note)
      } else if (!a$ok) {
        rec$status <- "UNSUPPORTED_TMB"; rec$note <- one_line(a$note)
      } else if (!b$ok) {
        rec$status <- "UNSUPPORTED_JULIA"; rec$note <- one_line(b$note)
      } else {
        cmp <- compare_ci(a$ci, b$ci)
        rec$n_params <- cmp$n; rec$max_abs_diff <- cmp$max_abs
        rec$max_rel_diff <- cmp$max_rel; rec$width_ratio <- cmp$width_ratio
        rec$matched <- cmp$how
        # A parameter reported by one engine and not the other is a CAPABILITY
        # difference, not an agreement one, and must not be hidden by comparing
        # only the intersection.
        rec$only_tmb <- cmp$only_tmb
        rec$only_julia <- cmp$only_julia
        # An interval can match at its endpoints and still be the wrong SHAPE, so
        # the width ratio is a gate in its own right, not decoration.
        rec$status <- if (!is.finite(cmp$max_rel)) {
          "INTERVAL_INDETERMINATE"
        } else if (cmp$only_tmb > 0L || cmp$only_julia > 0L) {
          "PARAM_COVERAGE_DIFF"
        } else if (cmp$max_rel <= tol_rel &&
                   (is.na(cmp$width_ratio) || cmp$width_ratio <= 1.05)) {
          "INTERVAL_PASS"
        } else {
          "INTERVAL_MISMATCH"
        }
      }
    }
    cat(sprintf("   %-10s %-18s rel=%-10.3g width_ratio=%-10.3g %s\n", m, rec$status,
                ifelse(is.na(rec$max_rel_diff), NA_real_, rec$max_rel_diff),
                ifelse(is.na(rec$width_ratio), NA_real_, rec$width_ratio),
                substr(rec$note, 1, 60)))
    rows[[length(rows) + 1L]] <- rec
  }
}

df <- do.call(rbind, lapply(rows, function(r) as.data.frame(r, stringsAsFactors = FALSE)))
df$drmtmb_version <- as.character(utils::packageVersion("drmTMB"))
df$R_boot <- R_boot
df$seed <- boot_seed
write.table(df, out_path, sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("\nwrote %s\n", out_path))

tally <- table(df$status)
cat("STATUS TALLY: ", paste(sprintf("%s=%d", names(tally), as.integer(tally)), collapse = "  "), "\n")
