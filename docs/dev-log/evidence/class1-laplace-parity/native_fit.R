# Class-1 twin-parity receipt: native drmTMB (TMB Laplace) side.
#
# Regenerates the issue cells (#709 Poisson, #710 Beta, #713 binary / bacteria /
# cbind binomial, #714 Student, #715 NB2, #716 Gamma), writes each data set to
# fixtures/<cell>.csv so the Julia side reads byte-identical data, fits
# engine = "tmb", and writes native.tsv (df, logLik, every free outer parameter
# on its working scale).
#
# The burn-cell CSVs of the issues (/workspace/drm-twin-grid-burn) are not
# available here. Where the issue gives the exact simulator (#713 binary) it is
# used verbatim; elsewhere the design (n, groups, formula, seed) is the issue's
# and the DGP constants (RE SD, coefficients, dispersion) are ours. Two extra
# larger Student cells (student_big_*) probe #714 beyond the single small cell.
#
# Run from the DRModels.jl repo root (drmTMB 0.7.1 installed):
#   Rscript docs/dev-log/evidence/class1-laplace-parity/native_fit.R
suppressPackageStartupMessages(library(drmTMB))
out_dir <- "docs/dev-log/evidence/class1-laplace-parity"
dir.create(file.path(out_dir, "fixtures"), showWarnings = FALSE)

grp <- function(G, m) factor(sprintf("g%02d", rep(seq_len(G), each = m)))
cells <- list()
add <- function(name, family, formula, data, enames) {
  write.csv(data, file.path(out_dir, "fixtures", paste0(name, ".csv")), row.names = FALSE)
  cells[[name]] <<- list(family = family, formula = formula, data = data, enames = enames)
}

# #709 Poisson, n = 60 (10 x 6), y ~ 1 + (1|g)
set.seed(20266905L); g <- grp(10, 6); b <- rnorm(10, 0, 0.55)[as.integer(g)]
add("poisson_709", poisson(), bf(y ~ 1 + (1 | g)),
    data.frame(y = rpois(60, exp(0.45 + b)), g = g), c("mu:(Intercept)", "resd:g(log_sd)"))

# #710 Beta, n = 60 (12 x 5), y ~ 1 + x + (1|g), sigma ~ 1
set.seed(20272905L); g <- grp(12, 5); b <- rnorm(12, 0, 0.3)[as.integer(g)]; x <- rnorm(60)
mu <- plogis(0.15 + 0.5 * x + b); phi <- 1 / 0.2^2
add("beta_710", beta_family(), bf(y ~ 1 + x + (1 | g), sigma ~ 1),
    data.frame(y = rbeta(60, mu * phi, (1 - mu) * phi), x = x, g = g),
    c("mu:(Intercept)", "mu:x", "sigma:(Intercept)", "resd:g(log_sd)"))

# #713 binary (exact simulator from the issue)
set.seed(20285905L)
g <- factor(rep(seq_len(12L), each = 8L)); b <- rnorm(12L, 0, 0.7); x <- rnorm(length(g))
y <- rbinom(length(g), 1L, plogis(-0.2 + 0.5 * x + b[as.integer(g)]))
add("binary_713", binomial(), bf(y ~ 1 + x + (1 | g)), data.frame(y, x, g),
    c("mu:(Intercept)", "mu:x", "resd:g(log_sd)"))
# #713 MASS::bacteria
bac <- MASS::bacteria
add("bacteria_713", binomial(), bf(y ~ 1 + week + (1 | g)),
    data.frame(y = as.integer(bac$y == "y"), week = bac$week, g = as.character(bac$ID)),
    c("mu:(Intercept)", "mu:week", "resd:g(log_sd)"))
# #713 cbind, 8 trials
set.seed(20286905L); g <- grp(12, 8); b <- rnorm(12, 0, 0.5)[as.integer(g)]; x <- rnorm(96)
succ <- rbinom(96, 8, plogis(-0.1 + 0.4 * x + b))
add("cbind_713", binomial(), bf(cbind(succ, fail) ~ 1 + x + (1 | g)),
    data.frame(succ = succ, fail = 8L - succ, x = x, g = g),
    c("mu:(Intercept)", "mu:x", "resd:g(log_sd)"))

# #714 Student, y ~ 1 + (1|g), sigma ~ 1, nu ~ 1
st <- function(seed, G, m, sdb, sig, df) {
  set.seed(seed); g <- grp(G, m); b <- rnorm(G, 0, sdb)[as.integer(g)]
  data.frame(y = 0.3 + b + sig * rt(G * m, df), g = g)
}
enm <- c("mu:(Intercept)", "sigma:(Intercept)", "nu:(Intercept)", "resd:g(log_sd)")
add("student_714", student(), bf(y ~ 1 + (1 | g), sigma ~ 1), st(20275905L, 12, 8, 0.5, 0.5, 5), enm)
add("student_big_a", student(), bf(y ~ 1 + (1 | g), sigma ~ 1), st(20275906L, 30, 10, 0.5, 0.5, 5), enm)
add("student_big_b", student(), bf(y ~ 1 + (1 | g), sigma ~ 1), st(20275907L, 30, 10, 0.8, 0.4, 8), enm)

# #715 NB2, y ~ 1 + (1|g): small group SD (issue: sd_mu ~ 0.07)
set.seed(20278905L); g <- grp(12, 8); b <- rnorm(12, 0, 0.12)[as.integer(g)]
add("nbinom2_715", nbinom2(), bf(y ~ 1 + (1 | g)),
    data.frame(y = rnbinom(96, mu = exp(1.0 + b), size = 4), g = g),
    c("mu:(Intercept)", "sigma:(Intercept)", "resd:g(log_sd)"))
# #716 Gamma, y ~ 1 + (1|g), sigma ~ 1
set.seed(20276905L); g <- grp(12, 8); b <- rnorm(12, 0, 0.35)[as.integer(g)]
m <- exp(0.5 + b); sh <- 1 / 0.4^2
add("gamma_716", Gamma(link = "log"), bf(y ~ 1 + (1 | g), sigma ~ 1),
    data.frame(y = rgamma(96, shape = sh, rate = sh / m), g = g),
    c("mu:(Intercept)", "sigma:(Intercept)", "resd:g(log_sd)"))

rows <- list()
for (cell in names(cells)) {
  cc <- cells[[cell]]
  fit <- drmTMB(cc$formula, family = cc$family, data = cc$data, engine = "tmb")
  par <- fit$opt$par; nm <- names(par)
  est <- unname(c(par[nm == "beta_mu"], par[nm == "beta_sigma"], par[nm == "beta_nu"],
                  par[grepl("^log_sd", nm)]))
  stopifnot(length(est) == length(cc$enames))
  ll <- as.numeric(logLik(fit))
  rows[[cell]] <- data.frame(cell = cell, engine = "tmb_laplace",
                             df = attr(logLik(fit), "df"), logLik = sprintf("%.10f", ll),
                             conv = fit$opt$convergence,
                             max_abs_grad = signif(max(abs(fit$obj$gr(par))), 3),
                             param = cc$enames, estimate = sprintf("%.10f", est))
  cat(cell, "logLik", ll, "conv", fit$opt$convergence, "\n")
}
write.table(do.call(rbind, rows), file.path(out_dir, "native.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)
cat("R", R.version.string, "| drmTMB", as.character(packageVersion("drmTMB")),
    "| TMB", as.character(packageVersion("TMB")), "\n")
