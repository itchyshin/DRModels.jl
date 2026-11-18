suppressMessages({library(ape); library(drmTMB)})
N5 <- "/tmp/claude-0/-home-user-DRModels-jl/bf8c2b27-03aa-5044-87f6-e06cfadb9c02/scratchpad/n5"
d <- read.csv(file.path(N5, "carni70.csv"))
cat("drmTMB version:", as.character(packageVersion("drmTMB")), "\n")
for (reml in c(FALSE, TRUE)) for (tn in c("raw", "unit")) {
  tree <- ape::read.tree(file.path(N5, if (tn == "raw") "carni70.nwk" else "carni70_unit.nwk"))
  t0 <- Sys.time()
  r <- tryCatch({
    f <- drmTMB::drmTMB(drmTMB::drm_formula(lsize ~ lrange, sigma ~ 1 + phylo(1 | species, tree = tree)),
                        family = gaussian(), data = d, REML = reml)
    ll <- logLik(f)
    cat(sprintf("RESULT drmTMB variant=%s tree=%s loglik=%.15g df=%s converged=%s time_s=%.1f height=%.4f\n",
                if (reml) "REML" else "ML", tn, as.numeric(ll), attr(ll, "df"),
                drmTMB::is_converged(f), as.numeric(difftime(Sys.time(), t0, units = "secs")),
                max(ape::node.depth.edgelength(tree))))
    cat("   par=", paste(sprintf("%s=%.8g", names(f$opt$par), f$opt$par), collapse = " "), "\n")
    cat("   opt$convergence=", f$opt$convergence, " message=", f$opt$message, "\n")
    as.numeric(ll)
  }, error = function(e) {
    cat(sprintf("RESULT drmTMB variant=%s tree=%s ERROR: %s\n", if (reml) "REML" else "ML", tn, conditionMessage(e)))
    NA
  })
}
