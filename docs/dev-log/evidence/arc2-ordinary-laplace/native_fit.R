# Arc 2 ordinary-Laplace receipt: native drmTMB (TMB Laplace) side.
#
# Simulates the fixtures, writes each data set to fixtures/<cell>.csv so the
# Julia side reads byte-identical data, fits engine = "tmb", and writes
# native.tsv (df, logLik, every free outer parameter on its working scale).
#
# Run from the DRModels.jl repo root:
#   DRMTMB_PATH=~/local-scratch/lanes/drmTMB-arc1-pr1304-fold \
#     Rscript docs/dev-log/evidence/arc2-ordinary-laplace/native_fit.R
suppressPackageStartupMessages({
  pkgload::load_all(Sys.getenv("DRMTMB_PATH"), quiet = TRUE)
})
out_dir <- "docs/dev-log/evidence/arc2-ordinary-laplace"
dir.create(file.path(out_dir, "fixtures"), showWarnings = FALSE)

sim <- function(family, seed, n_g = 30L, m = 10L, sd_g = 0.6) {
  set.seed(seed)
  n <- n_g * m
  g <- factor(sprintf("g%02d", rep(seq_len(n_g), each = m)))
  x <- rnorm(n)
  z <- rnorm(n)
  b <- rnorm(n_g, 0, sd_g)[as.integer(g)]
  y <- switch(family,
    poisson  = rpois(n, exp(0.4 + 0.5 * x + b)),
    nbinom2  = rnbinom(n, mu = exp(0.8 + 0.4 * x + b), size = 1 / 0.5^2),
    binomial = rbinom(n, 1, plogis(-0.2 + 0.8 * x + b)),
    gamma    = { mu <- exp(0.3 + 0.5 * x + b); rgamma(n, shape = 1 / 0.4^2, rate = 1 / 0.4^2 / mu) },
    beta     = { mu <- plogis(0.2 + 0.6 * x + b); phi <- 1 / 0.3^2; rbeta(n, mu * phi, (1 - mu) * phi) }
  )
  data.frame(y = y, x = x, z = z, g = g)
}

fam_obj <- function(family) switch(family,
  poisson = poisson(), nbinom2 = nbinom2(), binomial = binomial(),
  gamma = Gamma(link = "log"), beta = beta_family())

has_sigma <- function(family) family %in% c("nbinom2", "gamma", "beta")

cells <- expand.grid(family = c("poisson", "nbinom2", "binomial", "gamma", "beta"),
                     seed = c(20260924L, 20260925L), stringsAsFactors = FALSE)
rows <- list()
for (k in seq_len(nrow(cells))) {
  fam <- cells$family[k]; seed <- cells$seed[k]
  cell <- sprintf("%s_ri_s%d", fam, seed)
  dat <- sim(fam, seed)
  write.csv(dat, file.path(out_dir, "fixtures", paste0(cell, ".csv")), row.names = FALSE)
  form <- if (has_sigma(fam)) bf(y ~ x + (1 | g), sigma ~ 1) else bf(y ~ x + (1 | g))
  fit <- drmTMB(form, family = fam_obj(fam), data = dat, engine = "tmb")
  par <- fit$opt$par
  grad <- max(abs(fit$obj$gr(par)))
  ll <- as.numeric(logLik(fit))
  df <- attr(logLik(fit), "df")
  nm <- names(par)
  # working-scale outer parameters, in the order Julia reports them
  mu <- unname(par[nm == "beta_mu"])
  sig <- unname(par[nm == "beta_sigma"])
  lsd <- unname(par[grepl("^log_sd", nm)])
  est <- c(mu, sig, lsd)
  enames <- c("mu:(Intercept)", "mu:x",
              if (has_sigma(fam)) "sigma:(Intercept)",
              "resd:g(log_sd)")
  stopifnot(length(est) == length(enames))
  rows[[k]] <- data.frame(cell = cell, family = fam, seed = seed, engine = "tmb_laplace",
                          df = df, logLik = sprintf("%.10f", ll),
                          max_abs_grad = signif(grad, 3),
                          param = enames, estimate = sprintf("%.10f", est))
  cat(cell, "names:", paste(nm, collapse = ","), "logLik", ll, "\n")
}
out <- do.call(rbind, rows)
write.table(out, file.path(out_dir, "native.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("R", R.version.string, "| drmTMB", as.character(packageVersion("drmTMB")),
    "| TMB", as.character(packageVersion("TMB")), "\n")
