suppressMessages({library(ape); library(drmTMB)})
D <- "/tmp/claude-0/-home-user-DRModels-jl/bf8c2b27-03aa-5044-87f6-e06cfadb9c02/scratchpad/n5"
d <- read.csv(file.path(D, "carni70.csv"))
cat("drmTMB version:", as.character(packageVersion("drmTMB")), "\n")
trees <- list(raw = read.tree(file.path(D, "carni70.nwk")), unit = read.tree(file.path(D, "carni70_unit.nwk")))
for (lab in names(trees)) cat(lab, "tree height =", max(node.depth.edgelength(trees[[lab]])), "\n")

mk <- list(
  mu_only_phylo = function(tree) drm_formula(lsize ~ lrange + phylo(1 | species, tree = tree), sigma ~ 1),
  sigma_only_phylo = function(tree) drm_formula(lsize ~ lrange, sigma ~ 1 + phylo(1 | species, tree = tree)),
  mu_and_sigma_phylo = function(tree) drm_formula(lsize ~ lrange + phylo(1 | species, tree = tree),
                                                  sigma ~ 1 + phylo(1 | species, tree = tree)),
  coupled_1_p_phylo = function(tree) drm_formula(lsize ~ lrange + phylo(1 | p | species, tree = tree),
                                                 sigma ~ 1 + phylo(1 | p | species, tree = tree)),
  nu_phylo_extra = function(tree) drm_formula(lsize ~ lrange, sigma ~ 1, nu ~ 1 + phylo(1 | species, tree = tree))
)

one <- function(name, lab, f_or_fun) {
  t0 <- proc.time()[["elapsed"]]
  res <- tryCatch({
    fit <- drmTMB(f_or_fun, family = student(), data = d)
    ll <- logLik(fit)
    sprintf("RFIT variant=%s tree=%s status=ok loglik=%.10g df=%d converged=%s time_s=%.1f par=%s",
            name, lab, as.numeric(ll), as.integer(attr(ll, "df")), isTRUE(is_converged(fit)),
            proc.time()[["elapsed"]] - t0,
            paste(sprintf("%s=%.8g", names(fit$opt$par), fit$opt$par), collapse = ","))
  }, error = function(e) sprintf("RFIT variant=%s tree=%s status=ERROR time_s=%.1f error=%s", name, lab,
                                 proc.time()[["elapsed"]] - t0, gsub("\n", " | ", conditionMessage(e))))
  cat(res, "\n"); flush.console()
  res
}

one("control_no_phylo", "none", drm_formula(lsize ~ lrange, sigma ~ 1))
for (name in names(mk)) {
  for (lab in names(trees)) {
    tree <- trees[[lab]]
    f <- tryCatch(mk[[name]](tree), error = function(e) e)
    if (inherits(f, "error")) {
      cat(sprintf("RFIT variant=%s tree=%s status=ERROR(formula) error=%s\n", name, lab, conditionMessage(f)))
    } else one(name, lab, f)
  }
}
cat("DONE\n")
