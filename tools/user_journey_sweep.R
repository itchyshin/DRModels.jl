# user_journey_sweep.R — the USER-SEAT sweep of engine="julia".
#
# The owner's charge (2026-08-28): real users will consume DRModels.jl from R, the
# owner personally hit "bugs and snags" doing so, and the parity harnesses—
# which fit curated fixtures—cannot see that class of problem. This sweep types
# models the way a user types them (factors, interactions, I()/poly()/scale(),
# cbind, missing data, REML, wrong-but-reasonable inputs) and, for EVERY cell:
#
#   * fits with BOTH engines;
#   * runs the standard post-fit workflow on the julia fit
#     (print, summary, coef, fixef, vcov, confint-wald, predict(newdata),
#     residuals) and records any method that errors;
#   * name-matches fixed-effect estimates between engines (tol 1e-3 — a
#     USER-visible mismatch bar, not the parity bar);
#   * classifies: OK_MATCH · OK_MISMATCH · JULIA_ERROR · TMB_ERROR ·
#     BOTH_ERROR · JULIA_REFUSED (a clean, intentional gate) — and captures
#     error text VERBATIM, because a confusing message is itself a snag.
#
# Deliberately includes cells both engines should REFUSE: the sweep grades the
# refusal's clarity, not just success.
#
#   DRM_JL_PATH=$(pwd) NOT_CRAN=true Rscript tools/user_journey_sweep.R

if (!"drmTMB" %in% loadedNamespaces()) suppressMessages(library(drmTMB))
stopifnot(requireNamespace("ape", quietly = TRUE))
out_tsv <- "docs/dev-log/evidence/user-journey-sweep.tsv"

set.seed(20260828)
n <- 200
d <- data.frame(
  y  = rnorm(n, 1, 0.5), x = rnorm(n), z = rnorm(n),
  g  = factor(sample(letters[1:6], n, TRUE)),
  s  = rbinom(n, 20, 0.4), f = 20L - rbinom(n, 20, 0.4),
  v  = runif(n, 0.05, 0.2)
)
d$y  <- 0.4 + 0.6 * d$x - 0.3 * d$z + 0.5 * (as.integer(d$g) - 3.5) / 3 + rnorm(n, 0, exp(-0.5 + 0.2 * d$z))
# Student cell data from an ACTUAL t (df = 5) so nu is identified — the first
# sweep generated Gaussian data and read a 14-vs-17 nu "mismatch" off a flat
# ridge (logLik agreed to 1e-5): the #483 lesson, in our own instrument.
d$yt <- 0.4 + 0.6 * d$x + 0.5 * rt(n, df = 5)
d$yp <- rpois(n, exp(0.2 + 0.4 * d$x))
d$yg <- rgamma(n, shape = 6, rate = 6 / exp(0.3 + 0.3 * d$x))
d$yb <- pmin(pmax(plogis(0.2 + 0.5 * d$x) + 0.05 * rnorm(n), 1e-3), 1 - 1e-3)
d$yna <- d$y; d$yna[c(3, 50, 111)] <- NA

ntip <- 24; m <- 5
tree <- ape::rcoal(ntip); tree$tip.label <- paste0("sp_", seq_len(ntip))
h <- max(diag(ape::vcv(tree))); tree$edge.length <- tree$edge.length / h
A <- ape::vcv(tree, corr = TRUE)
u <- as.vector(t(chol(A)) %*% rnorm(ntip)) * 0.5
pd <- data.frame(species = rep(tree$tip.label, each = m),
                 x = rnorm(ntip * m))
pd$yph <- 0.3 + 0.5 * pd$x + u[rep(seq_len(ntip), each = m)] + 0.4 * rnorm(ntip * m)
pd$ypc <- rpois(ntip * m, exp(0.3 + 0.4 * pd$x + u[rep(seq_len(ntip), each = m)]))
pd_chr <- pd; pd_chr$species <- as.character(pd_chr$species)   # character, not factor

G <- 20; Z <- matrix(rnorm(G * G), G); K <- crossprod(Z) / G; K <- K / mean(diag(K))
rownames(K) <- colnames(K) <- paste0("k", 1:G)
kd <- data.frame(gk = factor(rep(rownames(K), each = 8), levels = rownames(K)), x = rnorm(G * 8))
kd$yk <- 0.3 + 0.4 * kd$x + (as.vector(t(chol(K + diag(1e-8, G))) %*% rnorm(G)) * 0.5)[as.integer(kd$gk)] + 0.4 * rnorm(G * 8)

J <- list()  # journeys
add <- function(id, call_fn, newdata = NULL) J[[length(J) + 1L]] <<- list(id = id, fn = call_fn, nd = newdata)

add("gaussian_basic",        function(e) drmTMB(bf(y ~ x, sigma ~ 1), family = gaussian(), data = d, engine = e), data.frame(x = c(-1, 1)))
add("gaussian_sigma_covar",  function(e) drmTMB(bf(y ~ x, sigma ~ z), family = gaussian(), data = d, engine = e), data.frame(x = 0, z = c(-1, 1)))
add("factor_covariate",      function(e) drmTMB(bf(y ~ x + g, sigma ~ 1), family = gaussian(), data = d, engine = e), data.frame(x = 0, g = factor("c", levels = levels(d$g))))
add("interaction_star",      function(e) drmTMB(bf(y ~ x * z, sigma ~ 1), family = gaussian(), data = d, engine = e), data.frame(x = 1, z = 1))
add("interaction_colon",     function(e) drmTMB(bf(y ~ x + z + x:z, sigma ~ 1), family = gaussian(), data = d, engine = e), NULL)
add("I_square",              function(e) drmTMB(bf(y ~ x + I(x^2), sigma ~ 1), family = gaussian(), data = d, engine = e), NULL)
add("poly_2",                function(e) drmTMB(bf(y ~ poly(x, 2), sigma ~ 1), family = gaussian(), data = d, engine = e), NULL)
add("scale_x",               function(e) drmTMB(bf(y ~ scale(x), sigma ~ 1), family = gaussian(), data = d, engine = e), NULL)
add("minus_one_intercept",   function(e) drmTMB(bf(y ~ x - 1, sigma ~ 1), family = gaussian(), data = d, engine = e), NULL)
add("crossing_power",        function(e) drmTMB(bf(y ~ (x + z)^2, sigma ~ 1), family = gaussian(), data = d, engine = e), NULL)
add("poisson_basic",         function(e) drmTMB(bf(yp ~ x), family = poisson(), data = d, engine = e), data.frame(x = 0))
add("gamma_log",             function(e) drmTMB(bf(yg ~ x, sigma ~ 1), family = Gamma(link = "log"), data = d, engine = e), NULL)
add("beta_basic",            function(e) drmTMB(bf(yb ~ x, sigma ~ 1), family = drmTMB::beta(), data = d, engine = e), NULL)
add("binomial_cbind",        function(e) drmTMB(bf(cbind(s, f) ~ x), family = stats::binomial(), data = d, engine = e), NULL)
add("meta_V",                function(e) drmTMB(bf(y ~ x + meta_V(V = v), sigma ~ 1), family = gaussian(), data = d, engine = e), NULL)
add("missing_y_default",     function(e) drmTMB(bf(yna ~ x, sigma ~ 1), family = gaussian(), data = d, engine = e), NULL)
add("missing_y_include",     function(e) drmTMB(bf(yna ~ x, sigma ~ 1), family = gaussian(), data = d, engine = e,
                                                missing = miss_control(response = "include")), NULL)
add("gaussian_reml",         function(e) drmTMB(bf(y ~ x, sigma ~ 1), family = gaussian(), data = d, engine = e, REML = TRUE), NULL)
add("phylo_gaussian",        function(e) drmTMB(bf(yph ~ x + phylo(1 | species, tree = tree), sigma ~ 1),
                                                family = gaussian(), data = pd, engine = e), NULL)
add("phylo_species_chr",     function(e) drmTMB(bf(yph ~ x + phylo(1 | species, tree = tree), sigma ~ 1),
                                                family = gaussian(), data = pd_chr, engine = e), NULL)
add("phylo_poisson",         function(e) drmTMB(bf(ypc ~ x + phylo(1 | species, tree = tree)),
                                                family = poisson(), data = pd, engine = e), NULL)
add("phylo_missing_include", function(e) { pdna <- pd; pdna$yph[c(2, 9)] <- NA
                                           drmTMB(bf(yph ~ x + phylo(1 | species, tree = tree), sigma ~ 1),
                                                  family = gaussian(), data = pdna, engine = e,
                                                  missing = miss_control(response = "include")) }, NULL)
add("relmat_gaussian",       function(e) drmTMB(bf(yk ~ x + relmat(1 | gk, K = K), sigma ~ 1),
                                                family = gaussian(), data = kd, engine = e), NULL)
# Data built ONCE, outside the closure: the first sweep drew rnorm(n) inside
# the per-engine fit function, so the two engines fitted DIFFERENT data and a
# 0.065 "mismatch" was pure noise — the harness's own bug, not the engines'.
bd_biv <- data.frame(y1 = d$y, y2 = 0.5 * d$y + 0.5 * rnorm(n), x = d$x)
add("biv_gaussian",          function(e) drmTMB(bf(mu1 = y1 ~ x, mu2 = y2 ~ x, sigma1 = ~1, sigma2 = ~1, rho12 = ~1),
                                                family = biv_gaussian(), data = bd_biv, engine = e), NULL)
add("student_nu",            function(e) drmTMB(bf(yt ~ x, sigma ~ 1, nu ~ 1), family = student(), data = d, engine = e), NULL)
cd_xf <- data.frame(y1 = d$y, y2 = d$yp, x = d$x)
add("cross_family",          function(e) drmTMB(bf(mu1 = y1 ~ x, mu2 = y2 ~ x),
                                                family = c(gaussian(), poisson()), data = cd_xf, engine = e), NULL)
# -- cells both engines are EXPECTED to gate; the sweep grades the message ----
add("weights_gated",         function(e) drmTMB(bf(y ~ x, sigma ~ 1), family = gaussian(), data = d, engine = e,
                                                weights = rep(1, n)), NULL)
add("unknown_column",        function(e) drmTMB(bf(y ~ nosuchcol, sigma ~ 1), family = gaussian(), data = d, engine = e), NULL)
add("probit_binomial",       function(e) drmTMB(bf(cbind(s, f) ~ x), family = stats::binomial(link = "probit"),
                                                data = d, engine = e), NULL)

fit_try <- function(fn, e) tryCatch(list(ok = TRUE, fit = fn(e)),
                                    error = function(err) list(ok = FALSE, msg = conditionMessage(err)))
first_line <- function(s) { s <- gsub("\n", " | ", s); substr(s, 1, 220) }

rows <- list()
for (j in J) {
  rt <- fit_try(j$fn, "tmb"); rj <- fit_try(j$fn, "julia")
  res <- list(journey = j$id, status = NA_character_, max_coef_diff = NA_real_,
              julia_methods_ok = "", julia_methods_err = "",
              tmb_error = if (rt$ok) "" else first_line(rt$msg),
              julia_error = if (rj$ok) "" else first_line(rj$msg))
  if (rt$ok && rj$ok) {
    ct <- unlist(fixef(rt$fit)); cj <- unlist(fixef(rj$fit))
    # Cross-engine name normalisation, measured not guessed: Julia says
    # "g: b" / "x & z" where R says "gb" / "x:z"; block prefixes join with
    # ".". Interactions first, then factor colons, then unify separators.
    norm_nm <- function(v) gsub("[ .:]+", "_", gsub(": ", "", gsub(" & ", ":", v)))
    names(ct) <- norm_nm(names(ct)); names(cj) <- norm_nm(names(cj))
    shared <- intersect(names(ct), names(cj))
    dmax <- if (length(shared)) max(abs(ct[shared] - cj[shared])) else {
      k <- min(length(ct), length(cj)); max(abs(ct[seq_len(k)] - cj[seq_len(k)])) }
    res$max_coef_diff <- dmax
    res$status <- if (is.finite(dmax) && dmax < 1e-3) "OK_MATCH" else "OK_MISMATCH"
    okm <- c(); errm <- c()
    for (gen in c("print", "summary", "coef", "fixef", "vcov", "confint", "residuals")) {
      r <- tryCatch({
        switch(gen,
          print    = capture.output(print(rj$fit)),
          summary  = capture.output(summary(rj$fit)),
          coef     = coef(rj$fit),
          fixef    = fixef(rj$fit),
          vcov     = vcov(rj$fit),
          confint  = confint(rj$fit, method = "wald"),
          residuals = residuals(rj$fit))
        TRUE }, error = function(e2) first_line(conditionMessage(e2)))
      if (isTRUE(r)) okm <- c(okm, gen) else errm <- c(errm, paste0(gen, ": ", r))
    }
    if (!is.null(j$nd)) {
      r <- tryCatch({ predict(rj$fit, newdata = j$nd); TRUE },
                    error = function(e2) first_line(conditionMessage(e2)))
      if (isTRUE(r)) okm <- c(okm, "predict") else errm <- c(errm, paste0("predict: ", r))
    }
    res$julia_methods_ok <- paste(okm, collapse = ",")
    res$julia_methods_err <- paste(errm, collapse = " || ")
    if (nzchar(res$julia_methods_err) && res$status == "OK_MATCH") res$status <- "OK_METHOD_SNAG"
  } else if (!rt$ok && !rj$ok) {
    res$status <- "BOTH_ERROR"
  } else if (!rt$ok && grepl("gaussian.*gaussian|cross-family|composed", res$tmb_error, ignore.case = TRUE)) {
    res$status <- "NO_NATIVE_COMPARATOR"
  } else if (!rj$ok) {
    res$status <- if (grepl("engine = \"julia\"|DRModels.jl|bridge|not support", res$julia_error, ignore.case = TRUE))
      "JULIA_REFUSED" else "JULIA_ERROR"
  } else res$status <- "TMB_ERROR"
  rows[[length(rows) + 1L]] <- as.data.frame(res, stringsAsFactors = FALSE)
  cat(sprintf("%-22s %-15s coefdiff=%-9.2g %s%s\n", res$journey, res$status,
              res$max_coef_diff,
              if (nzchar(res$julia_error)) paste0("JERR: ", substr(res$julia_error, 1, 80)) else "",
              if (nzchar(res$julia_methods_err)) paste0("MERR: ", substr(res$julia_methods_err, 1, 80)) else ""))
}

tab <- do.call(rbind, rows)
source(file.path("tools", "drmtmb_provenance_lib.R"))
tab$drmtmb_code_hash <- drmtmb_code_hash()
dir.create(dirname(out_tsv), showWarnings = FALSE, recursive = TRUE)
write.table(tab, out_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
cat("\nwrote ", out_tsv, "\n", sep = "")
cat("STATUS TALLY:\n"); print(table(tab$status))
