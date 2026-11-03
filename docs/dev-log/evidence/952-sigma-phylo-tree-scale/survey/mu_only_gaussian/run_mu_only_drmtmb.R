# drmTMB (ca3ddef) parity for the Gaussian mu-only phylo route on carni70, raw vs unit tree.
D <- "/tmp/claude-0/-home-user-DRModels-jl/bf8c2b27-03aa-5044-87f6-e06cfadb9c02/scratchpad/n5"
d <- read.csv(file.path(D, "carni70.csv"))
cat("packageVersion drmTMB:", as.character(packageVersion("drmTMB")), "\n")
trees <- list(raw = ape::read.tree(file.path(D, "carni70.nwk")),
              unit = ape::read.tree(file.path(D, "carni70_unit.nwk")))
for (nm in names(trees)) cat("TREE", nm, "height", max(ape::node.depth.edgelength(trees[[nm]])), "\n")
for (meth in c("ML", "REML")) for (nm in names(trees)) {
  tree <- trees[[nm]]
  t0 <- proc.time()[["elapsed"]]
  res <- tryCatch({
    args <- list(drmTMB::drm_formula(lsize ~ lrange + phylo(1 | species, tree = tree), sigma ~ 1),
                 family = gaussian(), data = d)
    if (meth == "REML") args$REML <- TRUE
    f <- do.call(drmTMB::drmTMB, args)
    el <- proc.time()[["elapsed"]] - t0
    ll <- logLik(f)
    sprintf("DRMTMB method=%s tree=%s loglik=%s df=%s is_converged=%s time_s=%.2f par=%s",
            meth, nm, formatC(as.numeric(ll), digits = 10, format = "g"), attr(ll, "df"),
            drmTMB::is_converged(f), el,
            paste(names(f$opt$par), formatC(f$opt$par, digits = 10, format = "g"), sep = "=", collapse = ","))
  }, error = function(e) sprintf("DRMTMB method=%s tree=%s ERROR %s", meth, nm, conditionMessage(e)))
  cat(res, "\n")
}
