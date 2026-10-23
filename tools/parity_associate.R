# parity_associate.R — staged-association parity: drmTMB's `associate_pairs()`
# against DRModels.jl's, on identical data and identical frozen margins.
#
# The staged route needs TWO fitted margins plus a staged call, which the
# per-cell `parity_fixture.R` harness does not construct — hence its own script.
#
# Both sides fit their margins independently, so this compares the whole staged
# pipeline (margin fits + freezing + association), not just the association
# optimiser. That is the honest comparison: a margin-fitting difference would
# show up here, and should.
#
#   DRM_JL_PATH=/path/to/DRModels.jl Rscript tools/parity_associate.R

suppressMessages(library(drmTMB))

tol <- 1e-4
out_path <- "docs/dev-log/evidence/parity-associate.tsv"

make_data <- function(seed, n = 1200, eta = 0.55) {
  set.seed(seed)
  x <- rnorm(n)
  z1 <- rnorm(n)
  z2 <- eta * z1 + sqrt(1 - eta^2) * rnorm(n)
  u1 <- pnorm(z1); u2 <- pnorm(z2)
  sg <- 0.6; size <- 1 / sg^2
  data.frame(
    x  = x,
    y  = 0.3 + 0.7 * x + z1,                                   # gaussian  <- z1
    b1 = as.numeric(u1 > 1 - plogis(0.2 + 0.5 * x)),           # bernoulli <- u1
    b2 = as.numeric(u2 > 1 - plogis(0.1 - 0.4 * x)),           # bernoulli <- u2
    c1 = qnbinom(u1, size = size, mu = exp(1.5 + 0.3 * x)),    # nbinom2   <- u1
    c2 = qnbinom(u2, size = size, mu = exp(1.4 - 0.2 * x))     # nbinom2   <- u2
  )
}

cells <- list(
  list(id = "gaussian_bernoulli",  f1 = "y",  fam1 = "gaussian", f2 = "b2", fam2 = "binomial"),
  list(id = "gaussian_nbinom2",    f1 = "y",  fam1 = "gaussian", f2 = "c2", fam2 = "nbinom2"),
  list(id = "bernoulli_bernoulli", f1 = "b1", fam1 = "binomial", f2 = "b2", fam2 = "binomial"),
  list(id = "bernoulli_nbinom2",   f1 = "b1", fam1 = "binomial", f2 = "c2", fam2 = "nbinom2"),
  list(id = "nbinom2_nbinom2",     f1 = "c1", fam1 = "nbinom2",  f2 = "c2", fam2 = "nbinom2")
)

# `bf()` inspects its unevaluated arguments and rejects `as.formula(...)`
# ("inputs must be formulas"), so the call is built as an expression and eval'd —
# that keeps the formulas literal at the point bf() sees them.
fit_margin <- function(resp, fam, d) {
  call_txt <- if (fam == "gaussian") {
    sprintf('drmTMB(bf(%s ~ x, sigma ~ 1), family = gaussian(), data = d)', resp)
  } else if (fam == "binomial") {
    sprintf('drmTMB(bf(%s ~ x), family = binomial(), data = d)', resp)
  } else {
    sprintf('drmTMB(bf(%s ~ x, sigma ~ 1), family = nbinom2(), data = d)', resp)
  }
  eval(str2lang(call_txt))
}

# The Julia-side margin fit for a response/family pair, as a Julia expression.
julia_margin <- function(resp, fam) {
  if (fam == "gaussian") {
    sprintf('DRModels.drm(DRModels.bf(DRModels.@formula(%s ~ x), DRModels.@formula(sigma ~ 1)), DRModels.Gaussian(); data = d)', resp)
  } else if (fam == "binomial") {
    sprintf('DRModels.drm(DRModels.bf(DRModels.@formula(%s ~ x)), DRModels.Binomial(); data = d)', resp)
  } else {
    sprintf('DRModels.drm(DRModels.bf(DRModels.@formula(%s ~ x), DRModels.@formula(sigma ~ 1)), DRModels.NegBinomial2(); data = d)', resp)
  }
}

drmTMB:::drm_julia_setup()
d <- make_data(4)
JuliaCall::julia_assign("pa_df", as.list(d))
JuliaCall::julia_command("pa_d = (; (Symbol(k) => Float64.(v) for (k,v) in pa_df)...)")

rows <- list()
for (cell in cells) {
  res <- list(pair_class = cell$id, status = NA_character_,
              eta_r = NA_real_, eta_julia = NA_real_, abs_diff = NA_real_,
              tolerance = tol, note = "")

  r_eta <- tryCatch({
    f1 <- fit_margin(cell$f1, cell$fam1, d)
    f2 <- fit_margin(cell$f2, cell$fam2, d)
    association(associate_pairs(f1, f2, kernel = latent_normal(),
                                association = ~ 1))$eta
  }, error = function(e) {res$note <<- conditionMessage(e); NA_real_})

  j_eta <- tryCatch({
    JuliaCall::julia_eval(sprintf(
      'let d = pa_d
         m1 = %s; m2 = %s
         DRModels.association(DRModels.associate_pairs(m1, m2; kernel = DRModels.latent_normal())).eta
       end',
      julia_margin(cell$f1, cell$fam1), julia_margin(cell$f2, cell$fam2)))
  }, error = function(e) {res$note <<- paste(res$note, conditionMessage(e)); NA_real_})

  res$eta_r <- r_eta; res$eta_julia <- j_eta
  if (is.finite(r_eta) && is.finite(j_eta)) {
    res$abs_diff <- abs(r_eta - j_eta)
    res$status <- if (res$abs_diff < tol) "PARITY_PASS" else "PARITY_FAIL"
  } else {
    res$status <- "FAILED"
  }
  rows[[length(rows) + 1L]] <- as.data.frame(res, stringsAsFactors = FALSE)
  cat(sprintf("%-22s %-12s R=%.7f  julia=%.7f  diff=%.3e\n",
              res$pair_class, res$status, res$eta_r, res$eta_julia, res$abs_diff))
}

tab <- do.call(rbind, rows)
dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write.table(tab, out_path, sep = "\t", row.names = FALSE, quote = FALSE)
cat("\nwrote ", out_path, "\n", sep = "")
cat("OVERALL: ", if (all(tab$status == "PARITY_PASS")) "ALL CELLS PASS" else "SOME CELLS FAILED",
    "\n", sep = "")
