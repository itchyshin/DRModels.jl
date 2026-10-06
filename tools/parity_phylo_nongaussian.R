# parity_phylo_nongaussian.R — native-vs-Julia parity for the
# `phylo_gamma_beta_binomial` capability row.
#
# WHY THIS EXISTS. That row's own claim_boundary says it has
# "Finite-and-sane bridge smoke evidence only; no native TMB parity", and its
# next_action is "Add comparator or parity evidence before promoting beyond
# experimental." Smoke evidence shows a fit RUNS. It cannot show the two engines
# agree. This produces the missing comparator.
#
# WHAT IT DOES NOT DO: promote anything. `claim_status` lives in drmTMB's own
# release process and is the owner's call. This script produces evidence and
# stops there.
#
# THREE THINGS CARRIED OVER FROM THE PENALTY FIXTURE (each cost a debugging round
# there, so they are stated rather than rediscovered):
#
#  1. THE TREE IS NORMALISED TO UNIT HEIGHT. drmTMB builds its phylogenetic
#     covariance from `ape::vcv(tree, corr = TRUE)` — tips at variance 1 — while
#     DRModels.jl builds its sparse precision from the Newick branch lengths AS GIVEN,
#     so its tip variance is the tree height h. Without normalising, the fitted
#     phylo SDs differ by exactly sqrt(h) while the log-likelihoods agree, and the
#     failure reads as an engine bug instead of a units mismatch.
#
#  2. SPECIES ARE PASSED AS TIP-LABEL STRINGS. DRModels.jl's non-Gaussian phylo route
#     (`sparse_laplace_glmm.jl`) matches labels to leaves BY NAME, as drmTMB does.
#     (Its univariate Gaussian sparse route matches POSITIONALLY instead — a real
#     asymmetry, and the reason the penalty fixture had to emit rows in Newick
#     order. Do not assume one route's convention holds for another.)
#
#  3. MORE THAN ONE QUANTITY IS COMPARED. Coefficients AND log-likelihood. Two
#     engines can land on the same estimate while disagreeing about what they
#     report; the penalty fixture caught exactly that twice.
#
#   DRM_JL_PATH=/path/to/DRModels.jl Rscript tools/parity_phylo_nongaussian.R

suppressMessages(library(drmTMB))

tol <- 1e-4
out_path <- "docs/dev-log/evidence/parity-phylo-nongaussian.tsv"

make_fixture <- function(seed = 411L, n_tip = 12L, n_each = 6L, sd_phy = 0.5) {
  stopifnot(requireNamespace("ape", quietly = TRUE))
  set.seed(seed)
  tree <- ape::rcoal(n_tip)
  tree$tip.label <- paste0("sp_", seq_len(n_tip))
  h <- max(diag(ape::vcv(tree)))          # see note 1
  tree$edge.length <- tree$edge.length / h
  nwk <- ape::write.tree(tree)

  A <- ape::vcv(tree, corr = TRUE)
  u <- as.vector(t(chol(A)) %*% rnorm(n_tip)) * sd_phy
  species <- factor(rep(tree$tip.label, each = n_each), levels = tree$tip.label)
  idx <- rep(seq_len(n_tip), each = n_each)
  n <- n_tip * n_each
  x <- rnorm(n)
  eta <- 0.3 + 0.4 * x + u[idx]

  dat <- data.frame(
    x = x,
    species = species,
    # gamma: positive, log link
    y_gamma = rgamma(n, shape = 8, rate = 8 / exp(eta)),
    # binomial: Bernoulli, logit link
    y_binom = rbinom(n, 1, plogis(eta)),
    # beta: strictly interior, logit link
    y_beta = pmin(pmax(plogis(eta) + 0.03 * rnorm(n), 1e-3), 1 - 1e-3)
  )
  list(data = dat, tree = tree, newick = nwk,
       species_chr = as.character(species))
}

cells <- list(
  list(id = "phylo_gamma",    resp = "y_gamma", rfam = "Gamma(link = \"log\")",
       jfam = "DRModels.Gamma()",    label = "Gamma() with phylo(1 | species)"),
  # drmTMB REFUSES `sigma ~ 1` for binomial ("Binomial models currently support
  # only the `mu` event-probability formula"), so this cell is mu-only on BOTH
  # sides -- the comparison has to use each engine's admitted syntax, not a
  # uniform template.
  list(id = "phylo_binomial", resp = "y_binom", rfam = "stats::binomial()",
       jfam = "DRModels.Binomial()", mu_only = TRUE,
       label = "binomial() with phylo(1 | species)"),
  list(id = "phylo_beta",     resp = "y_beta",  rfam = "drmTMB::beta()",
       jfam = "DRModels.Beta()",     label = "beta() with phylo(1 | species)")
)

fx <- make_fixture()
tree <- fx$tree

drmTMB:::drm_julia_setup()
JuliaCall::julia_assign("pn_x", as.numeric(fx$data$x))
JuliaCall::julia_assign("pn_sp", fx$species_chr)
JuliaCall::julia_assign("pn_newick", fx$newick)
for (cell in cells) {
  JuliaCall::julia_assign(paste0("pn_", cell$id), as.numeric(fx$data[[cell$resp]]))
}
JuliaCall::julia_command("pn_tree = DRModels.augmented_phy(pn_newick)")

r_fit <- function(cell) {
  call_txt <- sprintf(
    'drmTMB(bf(%s ~ x + phylo(1 | species, tree = tree), sigma ~ 1), family = %s, data = fx$data)',
    cell$resp, cell$rfam)
  fit <- eval(str2lang(call_txt))
  list(coef = as.numeric(fit$sdpars$mu$coefficients %||% NA), loglik = as.numeric(fit$logLik),
       fit = fit)
}
`%||%` <- function(a, b) if (is.null(a)) b else a

# Mu-block Wald SEs = sqrt(diag(vcov())) filtered to the mu parameter block by
# label ("mu:(Intercept)", "mu:x"). Mirrors tools/parity_classc.R: the SE is a
# second-order quantity, recorded as evidence columns, and NA (not an error) if
# vcov() fails on one side.
se_mu_of <- function(fit) {
  tryCatch({
    V <- as.matrix(vcov(fit))
    v <- diag(V)
    labels <- rownames(V) %||% names(v)
    keep <- grepl("^mu[:_]", labels)
    v <- v[keep]
    ifelse(v > 0, sqrt(v), NA_real_)
  }, error = function(e) NULL)
}
fmt_se <- function(x) if (is.null(x) || !length(x)) "NA" else paste(sprintf("%.6g", x), collapse = ";")

j_fit <- function(cell) {
  JuliaCall::julia_command(sprintf(
    "pn_d = (; y = Float64.(pn_%s), x = Float64.(pn_x), species = String.(pn_sp))", cell$id))
  jbf <- if (isTRUE(cell$mu_only)) {
    "DRModels.bf(DRModels.@formula(y ~ x + phylo(1 | species)))"
  } else {
    "DRModels.bf(DRModels.@formula(y ~ x + phylo(1 | species)), DRModels.@formula(sigma ~ 1))"
  }
  as.numeric(JuliaCall::julia_eval(sprintf(
    'let d = pn_d, tr = pn_tree
       f = DRModels.drm(%s, %s; data = d, tree = tr)
       se = DRModels.stderror(f)
       se_mu = first(se[r] for (p, r) in f.blocks if p === :mu)
       vcat(DRModels.coef(f, :mu), DRModels.loglik(f), se_mu)
     end', jbf, cell$jfam)))
}

rows <- list()
for (cell in cells) {
  res <- list(capability_id = "phylo_gamma_beta_binomial", cell_id = cell$id,
              label = cell$label, status = NA_character_,
              max_abs_coef_diff = NA_real_, loglik_tmb = NA_real_,
              loglik_julia = NA_real_, loglik_diff = NA_real_,
              max_abs_se_diff = NA_real_, max_rel_se_diff = NA_real_,
              se_tmb = "NA", se_julia = "NA",
              tolerance = tol, note = "")

  rv <- tryCatch({
    bf_txt <- if (isTRUE(cell$mu_only)) {
      sprintf('bf(%s ~ x + phylo(1 | species, tree = tree))', cell$resp)
    } else {
      sprintf('bf(%s ~ x + phylo(1 | species, tree = tree), sigma ~ 1)', cell$resp)
    }
    fit <- eval(str2lang(sprintf('drmTMB(%s, family = %s, data = fx$data)',
                                 bf_txt, cell$rfam)))
    cf <- fit$coefficients$mu
    c(as.numeric(cf), as.numeric(fit$logLik), as.numeric(se_mu_of(fit)))
  }, error = function(e) { res$note <<- paste("native:", conditionMessage(e)); NULL })

  jv <- tryCatch(j_fit(cell),
                 error = function(e) { res$note <<- paste(res$note, "julia:", conditionMessage(e)); NULL })

  if (!is.null(rv) && !is.null(jv) && length(rv) == length(jv) &&
      all(is.finite(rv)) && all(is.finite(jv))) {
    # Each vector is [k mu coefficients, logLik, k mu SEs]; length 2k + 1.
    k <- (length(rv) - 1L) %/% 2L
    res$max_abs_coef_diff <- max(abs(rv[seq_len(k)] - jv[seq_len(k)]))
    res$loglik_tmb <- rv[k + 1L]; res$loglik_julia <- jv[k + 1L]
    res$loglik_diff <- abs(res$loglik_tmb - res$loglik_julia)
    se_t <- rv[(k + 2L):length(rv)]; se_j <- jv[(k + 2L):length(jv)]
    res$se_tmb <- fmt_se(se_t); res$se_julia <- fmt_se(se_j)
    d <- abs(se_t - se_j)
    res$max_abs_se_diff <- max(d)
    res$max_rel_se_diff <- max(d / pmax(abs(se_t), abs(se_j)))
    res$status <- if (max(res$max_abs_coef_diff, res$loglik_diff) < tol) "PARITY_PASS" else "PARITY_FAIL"
  } else if (is.null(rv)) {
    # A native engine that REFUSES the syntax is not a failed comparison -- it is
    # the absence of a comparator, and the two must not be reported as the same
    # thing. drmTMB rejects binomial + phylo() ("Structured-effect syntax is
    # planned, not implemented"), so that third of the capability row simply has
    # nothing to be compared against on this version.
    res$status <- if (grepl("planned, not implemented|not.*supported|Unsupported",
                            res$note, ignore.case = TRUE))
      "NO_NATIVE_COMPARATOR" else "NATIVE_FAILED"
  } else if (is.null(jv)) {
    res$status <- "JULIA_FAILED"
  } else {
    res$status <- "SHAPE_MISMATCH"
    res$note <- paste(res$note, sprintf("lengths native=%d julia=%d", length(rv), length(jv)))
  }
  rows[[length(rows) + 1L]] <- as.data.frame(res, stringsAsFactors = FALSE)
  cat(sprintf("%-16s %-15s coef|d|=%-10.3e  ll %.5f/%.5f  |d|=%.3e\n",
              res$cell_id, res$status, res$max_abs_coef_diff,
              res$loglik_tmb, res$loglik_julia, res$loglik_diff))
  if (nzchar(res$note)) cat("    note: ", substr(res$note, 1, 160), "\n", sep = "")
}

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
cmp <- tab[tab$status != "NO_NATIVE_COMPARATOR", , drop = FALSE]
cat("OVERALL: ",
    if (nrow(cmp) > 0 && all(cmp$status == "PARITY_PASS"))
      sprintf("ALL COMPARABLE CELLS PASS (%d of %d; %d have no native comparator)",
              nrow(cmp), nrow(tab), nrow(tab) - nrow(cmp))
    else "SOME COMPARABLE CELLS DID NOT PASS",
    "\n", sep = "")
