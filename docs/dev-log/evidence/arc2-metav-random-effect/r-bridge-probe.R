# Arc 2 receipt, R side: the same fixtures through drmTMB(engine = "julia")
# against this DRModels.jl branch, next to engine = "tmb".
#
# PROBE, NOT A drmTMB CHANGE. drmTMB at the pinned commit (origin/main, before
# the Arc 1 `meta_v_with_random_effect` refusal) still admits these shapes, but
# its coefficient-label builder sends no `resd` label for a meta_V fit, so
# DRModels' label echo aborts ("coef_labels is missing an entry for dpar
# resd"). This script patches drm_julia_bridge_payload_coef_labels() IN THIS
# R SESSION ONLY to add one `resd` label per random component -- ordinary bars
# first, then structured markers, each in formula order, which is the order
# DRModels fits them in -- so the rest of the bridge can be observed. The
# conductor owns the real drmTMB change.
#
# Run from the DRModels.jl repository root:
#   JULIA_HOME=<julia 1.13 bin> DRMODELS_JL_PATH=$PWD DRMTMB_PATH=<drmTMB> \
#     Rscript docs/dev-log/evidence/arc2-metav-random-effect/r-bridge-probe.R
suppressMessages({
  pkgload::load_all(Sys.getenv("DRMTMB_PATH", "~/local-scratch/lanes/drmTMB-arc1-pr1304-fold"),
                    quiet = TRUE)
  library(ape)
})
orig <- drmTMB:::drm_julia_bridge_payload_coef_labels
patched <- function(formula, data, env, family_type = NULL, method = "ML") {
  labels <- orig(formula, data, env, family_type, method)
  has_meta <- any(vapply(formula$entries, function(e) formula_contains_call(e$rhs, "meta_V"), logical(1)))
  if (!has_meta || length(labels) == 0L) return(labels)
  mu <- Filter(function(e) identical(e$dpar, "mu"), formula$entries)
  bars <- unlist(lapply(mu, function(e) Filter(is_random_bar_call, flatten_plus_terms(e$rhs))),
                 recursive = FALSE)
  bar_groups <- vapply(bars, function(b) as.character(strip_parens(strip_parens(b)[[3L]])), character(1))
  st <- Filter(function(t) t$type %in% c("phylo", "relmat", "animal"),
               unlist(lapply(mu, function(e) e$structured), recursive = FALSE))
  groups <- c(bar_groups, vapply(st, function(t) t$group, character(1)))
  if (length(groups)) labels[["resd"]] <- groups
  labels
}
environment(patched) <- asNamespace("drmTMB")
assignInNamespace("drm_julia_bridge_payload_coef_labels", patched, "drmTMB")

here <- "docs/dev-log/evidence/arc2-metav-random-effect/fixtures"
rd <- function(f) read.delim(file.path(here, f), stringsAsFactors = FALSE)
show <- function(tag, f, data) {
  for (eng in c("tmb", "julia")) {
    fit <- tryCatch(drmTMB(f, family = gaussian(), data = data, engine = eng),
                    error = function(e) e)
    if (inherits(fit, "error")) {
      cat(sprintf("%-12s %-5s ERROR: %s\n", tag, eng, strsplit(conditionMessage(fit), "\n")[[1]][2]))
      next
    }
    ll <- logLik(fit)
    est <- c(unlist(coef(fit)), unlist(fit$sdpars))
    cat(sprintf("%-12s %-5s df=%d logLik=%.9f  %s\n", tag, eng, attr(ll, "df"), as.numeric(ll),
                paste(sprintf("%s=%.7f", names(est), est), collapse = " ")))
  }
}
d1 <- rd("study-a.tsv")
show("study-a", bf(y ~ x + meta_V(V = v) + (1 | study)), d1)
show("study-b sx", bf(y ~ x + meta_V(V = v) + (1 | study), sigma ~ x), rd("study-b.tsv"))
tree <- read.tree(file.path(here, "phylo-a.nwk"))
show("phylo-a", bf(y ~ x + meta_V(V = v) + phylo(1 | sp, tree = tree)), rd("phylo-a.tsv"))
K <- as.matrix(read.delim(file.path(here, "relmat-a-K.tsv"), header = FALSE))
dimnames(K) <- list(sprintf("i%02d", 1:25), sprintf("i%02d", 1:25))
show("relmat-a", bf(y ~ x + meta_V(V = v) + relmat(1 | id, K = K)), rd("relmat-a.tsv"))
show("phylo-study", bf(y ~ x + meta_V(V = v) + phylo(1 | sp, tree = tree) + (1 | study)),
     rd("phylo-study.tsv"))
# Neighbours that already matched (D-273): unchanged.
show("meta-only", bf(y ~ x + meta_V(V = v)), d1)
show("meta-sx", bf(y ~ x + meta_V(V = v), sigma ~ x), rd("study-b.tsv"))
