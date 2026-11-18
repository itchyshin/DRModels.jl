# gen_poly_parity.R -- native drmTMB 0.8.1 reference for `poly(x, 2)` (orthogonal polynomial) terms
# on the mean of eight univariate families. Julia's @formula macro used to fail on `poly` with an
# UndefVarError before DRModels.jl saw the model; `poly(x, d)` is now a formula term.
#
#   Rscript --vanilla test/parity/gen_poly_parity.R
#
# Writes test/fixtures/poly_parity/data.csv and ref.toml (one table per family: opt$par, sdreport
# SEs, logLik, max |gradient|; plus drmTMB predictions at four new x values). Numbers and
# simulated data only; no drmTMB source is copied. Fixed seed. A family whose R fit is not
# converged (max|grad| >= 1e-6 or a non-finite SE) stops the script: it is not a valid target.
suppressPackageStartupMessages(library(drmTMB))
stopifnot(as.character(utils::packageVersion("drmTMB")) == "0.8.1")
out_dir <- file.path("test", "fixtures", "poly_parity")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
fmt <- function(x) paste0("[", paste(formatC(x, digits = 17, format = "g"), collapse = ", "), "]")
qs  <- function(x) paste0("[", paste(sprintf("\"%s\"", x), collapse = ", "), "]")

set.seed(20261111)
n <- 200
x1 <- stats::rnorm(n)
eta <- 0.3 + 0.5 * x1 - 0.4 * x1^2
d <- data.frame(x1 = x1)
d$y_g    <- eta + stats::rnorm(n, 0, 0.6)
d$y_pois <- stats::rpois(n, exp(0.4 + 0.3 * x1 - 0.2 * x1^2))
d$y_bin  <- stats::rbinom(n, 1, stats::plogis(-0.2 + 0.6 * x1 - 0.5 * x1^2))
d$y_gam  <- stats::rgamma(n, shape = 5, rate = 5 / exp(0.5 + 0.3 * x1 - 0.2 * x1^2))
mu_b <- stats::plogis(0.1 + 0.5 * x1 - 0.4 * x1^2)
d$y_beta <- stats::rbeta(n, mu_b * 10, (1 - mu_b) * 10)
d$y_nb   <- stats::rnbinom(n, mu = exp(0.8 + 0.3 * x1 - 0.2 * x1^2), size = 3)
d$y_ln   <- exp(0.2 + 0.3 * x1 - 0.2 * x1^2 + stats::rnorm(n, 0, 0.4))
d$y_st   <- eta + 0.5 * stats::rt(n, df = 5)
# a missing response and a missing predictor: R builds the basis on the complete rows only
d$y_gm <- d$y_g; d$y_gm[c(5, 50, 111)] <- NA
d$x1m  <- d$x1;  d$x1m[c(20, 77, 150)] <- NA
utils::write.csv(d, file.path(out_dir, "data.csv"), row.names = FALSE)
nd <- data.frame(x1 = c(-1, 0, 0.5, 2))

cells <- list(
  gaussian  = list(f = bf(y_g ~ poly(x1, 2), sigma ~ 1), fam = gaussian()),
  gaussian_reml = list(f = bf(y_g ~ poly(x1, 2), sigma ~ 1), fam = gaussian(), reml = TRUE),
  gaussian_na = list(f = bf(y_gm ~ poly(x1m, 2), sigma ~ 1), fam = gaussian()),
  poisson   = list(f = bf(y_pois ~ poly(x1, 2)), fam = stats::poisson(link = "log")),
  binomial  = list(f = bf(y_bin ~ poly(x1, 2)), fam = stats::binomial(link = "logit")),
  gamma     = list(f = bf(y_gam ~ poly(x1, 2), sigma ~ 1), fam = stats::Gamma(link = "log")),
  beta      = list(f = bf(y_beta ~ poly(x1, 2), sigma ~ 1), fam = beta_family()),
  nb2       = list(f = bf(y_nb ~ poly(x1, 2), sigma ~ 1), fam = nbinom2()),
  lognormal = list(f = bf(y_ln ~ poly(x1, 2), sigma ~ 1), fam = lognormal()),
  student   = list(f = bf(y_st ~ poly(x1, 2), sigma ~ 1), fam = student())
)
lines <- c(sprintf("drmTMB_version = \"%s\"", as.character(utils::packageVersion("drmTMB"))),
           sprintf("R_version = \"%s\"", paste(R.version$major, R.version$minor, sep = ".")),
           sprintf("newdata_x1 = %s", fmt(nd$x1)))
for (nm in names(cells)) {
  fit <- drmTMB(cells[[nm]]$f, family = cells[[nm]]$fam, data = d, REML = isTRUE(cells[[nm]]$reml))
  reml <- isTRUE(cells[[nm]]$reml)
  cf <- coef(fit)
  jn <- unlist(lapply(names(cf), function(p) paste0(p, "_", names(cf[[p]]))), use.names = FALSE)
  gr <- as.numeric(fit$obj$gr(fit$opt$par))
  if (reml) {
    # REML integrates the mean coefficients out of opt$par: report coef(fit) and vcov(fit) instead
    par <- unlist(cf)
    se <- tryCatch(sqrt(diag(as.matrix(vcov(fit)))), error = function(e) rep(NA_real_, length(par)))
    if (length(se) != length(par)) se <- rep(NA_real_, length(par))
  } else {
    par <- fit$opt$par
    se <- tryCatch(sqrt(diag(fit$sdr$cov.fixed)), error = function(e) rep(NA_real_, length(par)))
    stopifnot(isTRUE(all.equal(as.numeric(unlist(cf)), as.numeric(par), tolerance = 1e-10)))
  }
  cat(sprintf("%-10s max|grad|=%.2e allSEfinite=%s ll=%.10f\n  names: %s\n  par: %s\n",
              nm, max(abs(gr)), all(is.finite(se)), as.numeric(logLik(fit)),
              paste(jn, collapse = " | "), paste(signif(par, 6), collapse = " ")))
  stopifnot(length(jn) == length(par), max(abs(gr)) < 1e-6, all(is.finite(se)))
  pr <- if (nm %in% c("gaussian", "poisson", "binomial"))
    as.numeric(predict(fit, newdata = nd, type = "response")) else numeric(0)
  lines <- c(lines, "", sprintf("[%s]", nm),
             sprintf("names = %s", qs(jn)),
             sprintf("nobs = %d", as.integer(stats::nobs(fit))),
             sprintf("loglik = %s", formatC(as.numeric(logLik(fit)), digits = 17, format = "g")),
             sprintf("max_grad = %s", formatC(max(abs(gr)), digits = 6, format = "e")),
             sprintf("theta = %s", fmt(as.numeric(par))),
             sprintf("se = %s", fmt(se)),
             sprintf("predict_response = %s", fmt(pr)))
}
writeLines(lines, file.path(out_dir, "ref.toml"))
cat("written", file.path(out_dir, "ref.toml"), "\n")
