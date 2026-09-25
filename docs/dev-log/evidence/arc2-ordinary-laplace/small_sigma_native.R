# Arc 2 small-sigma receipt: native drmTMB (TMB Laplace) side.
#
# (1) Regenerates the two small-sigma test fixtures added after review
#     (gamma_sigma0003.csv, nbinom2_sigma003.csv; seed 99173, unbalanced groups
#     of 4 to 13). The loop simulates six review cells in one RNG stream; only
#     the Gamma sigma = 0.003 and the near-Poisson NB2 (sigma = 0.03) cells are
#     kept, so the loop must run in full to reproduce them.
# (2) Fits engine = "tmb" on all four fixtures in test/fixtures/ordinary_laplace/
#     and writes small_sigma_native.tsv (df, logLik, convergence code, max
#     |gradient|, every free outer parameter on its working scale).
#
# Run from the DRModels.jl repo root:
#   DRMTMB_PATH=~/local-scratch/lanes/drmTMB-arc1-pr1304-fold \
#     Rscript docs/dev-log/evidence/arc2-ordinary-laplace/small_sigma_native.R
suppressPackageStartupMessages({
  pkgload::load_all(Sys.getenv("DRMTMB_PATH"), quiet = TRUE)
})
out_dir <- "docs/dev-log/evidence/arc2-ordinary-laplace"
fix_dir <- "test/fixtures/ordinary_laplace"

set.seed(99173)
mk <- function(fam, n_g, msizes, sd_g, sig, b0, b1) {
  g <- factor(sprintf("h%03d", rep(seq_len(n_g), times = msizes)))
  n <- length(g); x <- runif(n, -1.5, 1.5)
  b <- rnorm(n_g, 0, sd_g)[as.integer(g)]
  eta <- b0 + b1 * x + b
  y <- switch(fam,
    gamma = rgamma(n, shape = 1/sig^2, rate = 1/sig^2/exp(eta)),
    beta = { mu <- plogis(eta); phi <- 1/sig^2; rbeta(n, mu*phi, (1-mu)*phi) },
    poisson = rpois(n, exp(eta)),
    nbinom2 = rnbinom(n, mu = exp(eta), size = 1/sig^2))
  data.frame(y = y, x = x, g = g)
}
cells <- list(
  gamma_s0003   = list("gamma", 24, 0.6, 0.003, 0.2, 0.5),
  beta_s0002    = list("beta", 20, 0.5, 0.002, 0.1, 0.8),
  gamma_sdg0    = list("gamma", 30, 0.02, 0.01, 0.5, 0.3),
  beta_sdg0     = list("beta", 30, 0.01, 0.02, -0.3, 0.4),
  poisson_sdg0  = list("poisson", 30, 0.03, NA, 1.0, 0.3),
  nbinom2_nearP = list("nbinom2", 30, 0.5, 0.03, 1.5, 0.3))
keep <- c(gamma_s0003 = "gamma_sigma0003.csv", nbinom2_nearP = "nbinom2_sigma003.csv")
for (cn in names(cells)) {
  a <- cells[[cn]]
  ms <- sample(4:13, a[[2]], replace = TRUE)
  d <- mk(a[[1]], a[[2]], ms, a[[3]], a[[4]], a[[5]], a[[6]])
  if (cn %in% names(keep))
    write.csv(d, file.path(fix_dir, keep[[cn]]), row.names = FALSE)
}

fits <- list(
  gamma_sigma0012  = list(Gamma(link = "log"), bf(y ~ x + z + f + (1 | g), sigma ~ 1)),
  beta_sigma0012   = list(beta_family(),       bf(y ~ x + z + f + (1 | g), sigma ~ 1)),
  gamma_sigma0003  = list(Gamma(link = "log"), bf(y ~ x + (1 | g), sigma ~ 1)),
  nbinom2_sigma003 = list(nbinom2(),           bf(y ~ x + (1 | g), sigma ~ 1)))
rows <- list()
for (cn in names(fits)) {
  d <- read.csv(file.path(fix_dir, paste0(cn, ".csv")))
  d$g <- factor(d$g)
  if ("f" %in% names(d)) d$f <- factor(d$f)
  fit <- drmTMB(fits[[cn]][[2]], family = fits[[cn]][[1]], data = d, engine = "tmb")
  par <- fit$opt$par; nm <- names(par)
  est <- c(par[nm == "beta_mu"], par[nm == "beta_sigma"], par[grepl("^log_sd", nm)])
  ll <- logLik(fit)
  rows[[cn]] <- data.frame(cell = cn, df = attr(ll, "df"),
                           logLik = sprintf("%.10f", as.numeric(ll)),
                           conv = fit$opt$convergence,
                           maxgrad = signif(max(abs(fit$obj$gr(par))), 3),
                           k = seq_along(est), estimate = sprintf("%.12f", unname(est)))
}
write.table(do.call(rbind, rows), file.path(out_dir, "small_sigma_native.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
