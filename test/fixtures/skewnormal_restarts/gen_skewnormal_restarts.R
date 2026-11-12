# gen_skewnormal_restarts.R -- drmTMB reference numbers for test/test_skewnormal_restarts.jl.
#
#   Rscript test/fixtures/skewnormal_restarts/gen_skewnormal_restarts.R
#
# Fits the three data sets in this folder with drmTMB (the version recorded below) and prints
# the log-likelihood, the estimates and the standard errors that the test hard-codes. Numbers
# only; no drmTMB source is copied. Reference run: drmTMB 0.8.1, R 4.6.1, 2026-11-12.
suppressMessages(library(drmTMB))
dir <- file.path("test", "fixtures", "skewnormal_restarts")
show <- function(label, fit) {
  cat("\n==", label, "==\n")
  cat(sprintf("logLik %.10f  convergence %s\n", as.numeric(logLik(fit)), fit$opt$convergence))
  cat("par ", paste(sprintf("%.10g", fit$opt$par), collapse = ";"), "\n")
  cat("se  ", paste(sprintf("%.10g", sqrt(diag(vcov(fit)))), collapse = ";"), "\n")
}
# sim159: simulated set (see gen_skewnormal_restarts_sim.jl); the default start of the Julia
# fit stops 1.28 nats below the optimum drmTMB finds.
d <- read.csv(file.path(dir, "sim159.csv"))
show("sim159", drmTMB(bf(y ~ x, sigma ~ x, nu ~ x), family = skew_normal(), data = d))
# cell_y_sk and cell_y_g: columns of the gap-map data set (data_v2.csv, 160 rows).
d <- read.csv(file.path(dir, "cell_y_sk.csv"))
show("cell_y_sk", drmTMB(bf(y_sk ~ x1, sigma ~ x2, nu ~ x1), family = skew_normal(), data = d))
d <- read.csv(file.path(dir, "cell_y_g.csv"))
show("cell_y_g", drmTMB(bf(y_g ~ x1, sigma ~ x2, nu ~ x1), family = skew_normal(), data = d))
