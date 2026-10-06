# Arc 2 sigma-RE Laplace receipt: native drmTMB (TMB Laplace) side.
#
# Simulates the fixtures, writes each data set to fixtures/<cell>.csv so the
# Julia side reads byte-identical data, fits engine = "tmb", and writes
# native.tsv (df, logLik, max |outer gradient|, every free outer parameter on
# its working scale, and its standard error from sdreport).
#
# Cells p1-p5 reuse the conductor's probe designs and seeds (sigre-ghq-probe.R),
# so their native logLik reproduces the probe. Cell c6 adds a covariate on
# sigma; cell c7 has unequal group sizes.
#
# Run from the DRModels.jl repo root:
#   DRMTMB_PATH=~/local-scratch/lanes/drmTMB-arc1-pr1304-fold \
#     Rscript docs/dev-log/evidence/arc2-sigma-re-laplace/native_fit.R
suppressPackageStartupMessages({
  pkgload::load_all(Sys.getenv("DRMTMB_PATH"), quiet = TRUE)
})
out_dir <- "docs/dev-log/evidence/arc2-sigma-re-laplace"
dir.create(file.path(out_dir, "fixtures"), showWarnings = FALSE)

# Probe DGP: log sigma = -0.3 + b_g, b_g ~ N(0, sdb^2); mu = 0.2 + 0.5 x.
sim_probe <- function(G, m, sdb, seed) {
  set.seed(seed)
  g <- factor(rep(seq_len(G), each = m)); b <- rnorm(G, sd = sdb)
  d <- data.frame(g = g, x = rnorm(G * m))
  d$y <- 0.2 + 0.5 * d$x + rnorm(G * m, sd = exp(-0.3 + b[as.integer(g)]))
  d
}
# Covariate on sigma: log sigma = -0.3 + 0.25 x + b_g.
sim_sigx <- function(G, m, sdb, seed) {
  set.seed(seed)
  g <- factor(rep(seq_len(G), each = m)); b <- rnorm(G, sd = sdb)
  d <- data.frame(g = g, x = rnorm(G * m))
  d$y <- 0.2 + 0.5 * d$x + rnorm(G * m, sd = exp(-0.3 + 0.25 * d$x + b[as.integer(g)]))
  d
}
# Unequal group sizes (3 to 120 rows per group).
sim_unequal <- function(G, sdb, seed) {
  set.seed(seed)
  sizes <- sample(3:120, G, replace = TRUE)
  g <- factor(rep(seq_len(G), times = sizes)); b <- rnorm(G, sd = sdb)
  n <- sum(sizes)
  d <- data.frame(g = g, x = rnorm(n))
  d$y <- 0.2 + 0.5 * d$x + rnorm(n, sd = exp(-0.3 + b[as.integer(g)]))
  d
}

cells <- list(
  list(cell = "p1_G15_m10_sdb040", sigma = "1", dat = function() sim_probe(15, 10, 0.40, 1)),
  list(cell = "p2_G10_m50_sdb040", sigma = "1", dat = function() sim_probe(10, 50, 0.40, 2)),
  list(cell = "p3_G8_m150_sdb040", sigma = "1", dat = function() sim_probe(8, 150, 0.40, 3)),
  list(cell = "p4_G8_m150_sdb015", sigma = "1", dat = function() sim_probe(8, 150, 0.15, 4)),
  list(cell = "p5_G6_m400_sdb030", sigma = "1", dat = function() sim_probe(6, 400, 0.30, 5)),
  list(cell = "c6_sigmax_G12_m60_sdb040", sigma = "x", dat = function() sim_sigx(12, 60, 0.40, 6)),
  list(cell = "c7_unequal_G20_sdb050", sigma = "1", dat = function() sim_unequal(20, 0.50, 7))
)

rows <- list()
for (k in seq_along(cells)) {
  cc <- cells[[k]]
  dat <- cc$dat()
  write.csv(dat, file.path(out_dir, "fixtures", paste0(cc$cell, ".csv")), row.names = FALSE)
  form <- if (cc$sigma == "x") bf(y ~ x, sigma ~ 1 + x + (1 | g)) else bf(y ~ x, sigma ~ 1 + (1 | g))
  fit <- drmTMB(form, family = gaussian(), data = dat, engine = "tmb")
  par <- fit$opt$par
  nm <- names(par)
  grad <- max(abs(fit$obj$gr(par)))
  ll <- as.numeric(logLik(fit))
  df <- attr(logLik(fit), "df")
  se <- sqrt(diag(fit$sdr$cov.fixed))
  stopifnot(identical(names(se), nm))
  keep <- nm %in% c("beta_mu", "beta_sigma", "log_sd_sigma")
  stopifnot(sum(keep) == length(par))   # no other free outer parameter
  enames <- c("mu:(Intercept)", "mu:x", "sigma:(Intercept)",
              if (cc$sigma == "x") "sigma:x", "resd:g_logsigma(log_sd)")
  ord <- c(which(nm == "beta_mu"), which(nm == "beta_sigma"), which(nm == "log_sd_sigma"))
  stopifnot(length(ord) == length(enames))
  rows[[k]] <- data.frame(cell = cc$cell, n = nrow(dat), G = nlevels(dat$g),
                          engine = "tmb_laplace", convergence = fit$opt$convergence,
                          df = df, logLik = sprintf("%.10f", ll),
                          max_abs_grad = signif(grad, 3),
                          param = enames, estimate = sprintf("%.10f", unname(par[ord])),
                          se = sprintf("%.10f", unname(se[ord])))
  cat(sprintf("%-26s n=%4d ll=%.6f grad=%.2e conv=%d\n", cc$cell, nrow(dat), ll, grad,
              fit$opt$convergence))
}
out <- do.call(rbind, rows)
write.table(out, file.path(out_dir, "native.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
drm_dir <- Sys.getenv("DRMTMB_PATH")
sha <- tryCatch(system2("git", c("-C", drm_dir, "rev-parse", "HEAD"), stdout = TRUE),
                error = function(e) NA_character_)
cat("R", R.version.string, "| drmTMB", as.character(packageVersion("drmTMB")), "at", sha,
    "| TMB", as.character(packageVersion("TMB")), "\n")
