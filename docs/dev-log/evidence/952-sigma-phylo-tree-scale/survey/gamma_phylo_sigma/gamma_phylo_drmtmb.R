suppressPackageStartupMessages({ library(drmTMB); library(ape) })
n5 <- "/tmp/claude-0/-home-user-DRModels-jl/bf8c2b27-03aa-5044-87f6-e06cfadb9c02/scratchpad/n5"
d <- read.csv(file.path(n5, "carni70.csv"))
d$size <- exp(d$lsize)
trees <- list(raw = read.tree(file.path(n5, "carni70.nwk")),
              unit = read.tree(file.path(n5, "carni70_unit.nwk")))
cat("# drmTMB version", as.character(packageVersion("drmTMB")), "\n")
for (tn in names(trees)) cat("# tree", tn, "height", max(node.depth.edgelength(trees[[tn]])), "\n")

variants <- list(
  G1_mu_phylo_sigma1 = function() drm_formula(size ~ lrange + phylo(1 | species, tree = tree), sigma ~ 1),
  G2_mu_phylo_sigmaX = function() drm_formula(size ~ lrange + phylo(1 | species, tree = tree), sigma ~ 1 + lrange),
  G3_coupled_phylo_sigma1 = function() drm_formula(size ~ lrange + phylo(1 | p | species, tree = tree),
                                                   sigma ~ 1 + phylo(1 | p | species, tree = tree)),
  G4_coupled_phylo_sigmaX = function() drm_formula(size ~ lrange + phylo(1 | p | species, tree = tree),
                                                   sigma ~ 1 + lrange + phylo(1 | p | species, tree = tree)),
  R1_separate_phylo_mu_and_sigma = function() drm_formula(size ~ lrange + phylo(1 | species, tree = tree),
                                                          sigma ~ 1 + phylo(1 | species, tree = tree)),
  R2_sigma_only_phylo = function() drm_formula(size ~ lrange, sigma ~ 1 + phylo(1 | species, tree = tree))
)

for (vn in names(variants)) for (tn in names(trees)) {
  tree <- trees[[tn]]
  t0 <- proc.time()[["elapsed"]]
  res <- tryCatch({
    f <- drmTMB(variants[[vn]](), family = Gamma(link = "log"), data = d)
    ll <- logLik(f)
    sprintf("FIT variant=%s tree=%s loglik=%.10g loglik_full=%.17g df=%d converged=%s time_s=%.1f par=%s",
            vn, tn, as.numeric(ll), as.numeric(ll), as.integer(attr(ll, "df")), drmTMB::is_converged(f),
            proc.time()[["elapsed"]] - t0,
            paste(sprintf("%s=%.10g", names(f$opt$par), f$opt$par), collapse = ","))
  }, error = function(e) sprintf("ERR variant=%s tree=%s time_s=%.1f error=%s", vn, tn,
                                 proc.time()[["elapsed"]] - t0,
                                 gsub("\n", " ", conditionMessage(e))))
  cat(res, "\n")
}
cat("# done\n")
