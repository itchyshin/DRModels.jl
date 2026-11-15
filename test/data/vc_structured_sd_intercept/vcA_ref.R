# Pin for the DRModels.jl regression test test_vc_structured_sd_intercept.jl.
# Needs vcA.csv written by the Julia data generator in that test (same deterministic data).
# Run: Rscript vcA_ref.R   (drmTMB 0.8.1)
library(drmTMB)
d <- read.csv("vcA.csv"); G <- 20; ids <- paste0("a", 1:G)
A <- exp(-abs(outer(1:G, 1:G, "-")) / 3); dimnames(A) <- list(ids, ids)
f <- drmTMB(bf(cbind(s, fl) ~ x + animal(1 | g, A = A)), family = binomial(), data = d)
cat("drmTMB", as.character(packageVersion("drmTMB")), "R", as.character(getRversion()), "\n")
cat(sprintf("logLik %.7f\n", as.numeric(logLik(f))))
print(f$sdpars)
cat(sprintf("sd_animal %.7f\n", unlist(f$sdpars)[1]))
