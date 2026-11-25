## gen_gaussian_locscale_re.R -- generate the drmTMB reference fixture for the
## Gaussian location-scale model with a mean random intercept:
##     y ~ x + (1 | g),  sigma ~ x        (family = gaussian())
## Maintainer-only; never run by the tests. Writes generated data + numbers
## only (no drmTMB source). Pinned release: drmTMB 0.7.0 (CRAN).
##   Rscript --vanilla test/parity/gen_gaussian_locscale_re.R
suppressPackageStartupMessages(library(drmTMB))
stopifnot(as.character(packageVersion("drmTMB")) == "0.7.0")

out <- file.path("test", "parity", "fixtures_re", "gaussian-locscale-re")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
num <- function(x) format(as.numeric(x), digits = 17, scientific = TRUE, trim = TRUE)

seed <- 20261031L
set.seed(seed)
G <- 40; ng <- 8
g <- factor(rep(sprintf("g%02d", 1:G), each = ng))
x <- rnorm(G * ng)
u <- rnorm(G, 0, 0.7)[g]
y <- 1 + 0.5 * x + u + rnorm(G * ng, 0, exp(-0.3 + 0.4 * x))
d <- data.frame(y = y, x = x, g = as.character(g))
write.csv(d, file.path(out, "data.csv"), row.names = FALSE, quote = FALSE)

fit <- drmTMB(drm_formula(y ~ x + (1 | g), sigma ~ x),
              family = gaussian(), data = d)
stopifnot(fit$opt$convergence == 0)
s <- summary(fit)
cf <- s$coefficients
rn <- sub(":", "_", rownames(cf), fixed = TRUE)
sdrow <- s$parameters["sd:mu:(1 | g)", ]
V <- vcov(fit)
ll <- as.numeric(logLik(fit)); df <- attr(logLik(fit), "df")

L <- c("[fit]",
  'family = "gaussian"',
  'formula = "y ~ x + (1 | g); sigma ~ x"',
  paste0("loglik = ", num(ll)),
  paste0("aic = ", num(-2 * ll + 2 * df)),
  paste0("df = ", df),
  paste0("n = ", nrow(d)),
  "", "[coef]",
  paste0('"', rn, '" = ', num(cf[, "estimate"])),
  paste0('"sd_mu_(1 | g)" = ', num(sdrow$estimate)),
  "", "[se]",
  paste0('"', rn, '" = ', num(cf[, "std_error"])),
  paste0('"sd_mu_(1 | g)" = ', num(sdrow$std_error)),
  "", "[vcov]",
  paste0("order = [", paste0('"', sub(":", "_", rownames(V), fixed = TRUE), '"', collapse = ", "), "]"),
  paste0("data = [", paste(apply(V, 1, function(r) paste0("[", paste(num(r), collapse = ", "), "]")), collapse = ", "), "]"))
writeLines(L, file.path(out, "expected.toml"))
writeLines(c(
  'drmtmb_version = "0.7.0"',
  'drmtmb_source = "CRAN release drmTMB 0.7.0 (pinned; no git commit hash available: github.com/itchyshin/drmTMB unreachable from the generating sandbox)"',
  paste0('generated_on = "', Sys.Date(), '"'),
  'r_call = "drmTMB(drm_formula(y ~ x + (1 | g), sigma ~ x), family = gaussian(), data = d)"',
  paste0("seed = ", seed),
  'note = "Generated outputs only; no drmTMB source vendored (MIT-clean). Simulated: G=40, n_g=8, beta_mu=(1,0.5), sd_u=0.7, log sigma=-0.3+0.4x. vcov is the fixed-effect (mu, sigma) block; sd_mu_(1 | g) is the natural-scale RE SD."',
  paste0('R_version = "', R.version.string, '"')),
  file.path(out, "expected.meta.toml"))
