## gen_fixtures.R -- generate committed drmTMB numeric parity fixtures.
##
## This script is for maintainer machines with local R + drmTMB. It is never
## required at test time; `runparity.jl` consumes only the generated CSV/TOML
## fixtures. License boundary: write generated data/numbers only, never source.

suppressPackageStartupMessages(library(drmTMB))

repo_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    return(normalizePath(file.path(dirname(sub("^--file=", "", file_arg[[1]])), "..", "..")))
  }
  normalizePath(getwd())
}

toml_string <- function(x) {
  paste0('"', gsub('"', '\\"', as.character(x), fixed = TRUE), '"')
}

toml_num <- function(x) {
  if (!is.finite(x)) stop("cannot write non-finite TOML number")
  format(as.numeric(x), digits = 17, scientific = TRUE, trim = TRUE)
}

toml_array <- function(xs) {
  paste0("[", paste(vapply(xs, toml_string, character(1)), collapse = ", "), "]")
}

toml_matrix <- function(M) {
  rows <- apply(M, 1, function(row) {
    paste0("[", paste(vapply(row, toml_num, character(1)), collapse = ", "), "]")
  })
  paste0("[", paste(rows, collapse = ", "), "]")
}

flat_coef <- function(fit) {
  cf <- coef(fit)
  out <- numeric()
  for (param in names(cf)) {
    vals <- as.numeric(cf[[param]])
    names(vals) <- paste0(param, "_", names(cf[[param]]))
    out <- c(out, vals)
  }
  out
}

vcov_order <- function(fit) {
  sub(":", "_", rownames(vcov(fit)), fixed = TRUE)
}

transform_expected <- function(case, coefs, V, order) {
  deriv <- rep(1.0, length(order))
  names(deriv) <- order

  ## NB2 sigma and Student nu are on the same working scales in drmTMB and
  ## DRModels.jl now (NB2: log(sigma) with size = exp(-2*sigma); Student:
  ## log(nu - 2) with nu = 2 + exp(eta)). The former reparameterisation
  ## transforms are the identity — keep this hook for any future scale drift.

  D <- diag(deriv[order], nrow = length(order))
  list(coef = coefs, vcov = D %*% V %*% D)
}

## Extract hand-added [tol] overrides from an existing fixture so a regeneration
## does not destroy them.
##
## `write_case()` deletes the whole fixture directory before rewriting it, and
## `write_expected()` emits a FIXED [tol] block (atol_loglik/atol_aic, plus the
## ranef pair). Three fixtures -- meta-analysis-V, nbinom2-dispersion and
## robust-student -- also carry hand-added atol_coef/atol_vcov overrides
## documenting known soft-optimizer deltas vs drmTMB (#370, #392), together with
## the comments explaining WHY each bar was widened. Before 2026-08-24 any
## regeneration silently destroyed all of it: no error, no warning, and the
## reasoning simply gone. The next person to regenerate would have been left with
## fixtures failing for reasons nobody could reconstruct.
##
## Must be called BEFORE the unlink() in write_case(); afterwards there is
## nothing left to read.
read_preserved_tol <- function(path) {
  if (!file.exists(path)) return(character(0))
  old <- readLines(path, warn = FALSE)
  i <- which(trimws(old) == "[tol]")
  if (length(i) != 1L) return(character(0))
  j <- i + 1L
  block <- character(0)
  while (j <= length(old) && !grepl("^\\s*\\[", old[j])) {
    block <- c(block, old[j]); j <- j + 1L
  }
  generated <- c("atol_loglik", "atol_aic", "rtol_ranef", "atol_ranef")
  keep <- vapply(block, function(ln) {
    t <- trimws(ln)
    if (!nzchar(t)) return(FALSE)
    if (startsWith(t, "#")) return(TRUE)   # the reasoning is the point
    !(trimws(sub("=.*$", "", t)) %in% generated)
  }, logical(1), USE.NAMES = FALSE)
  sub("\\s+$", "", block[keep])
}

write_expected <- function(dir, family, formula, fit, case, ranef = NULL,
                           preserved_tol = character(0)) {
  coefs <- flat_coef(fit)
  V <- vcov(fit)
  order <- vcov_order(fit)
  names(coefs) <- sub(":", "_", names(coefs), fixed = TRUE)

  transformed <- transform_expected(case, coefs, V, order)
  coefs <- transformed$coef
  V <- transformed$vcov

  ll <- as.numeric(logLik(fit))
  df <- as.integer(attr(logLik(fit), "df"))
  n <- as.integer(attr(logLik(fit), "nobs"))
  if (is.na(n)) n <- nobs(fit)

  path <- file.path(dir, "expected.toml")
  con <- file(path, "w")
  on.exit(close(con), add = TRUE)
  writeLines(c(
    "[fit]",
    paste0("family = ", toml_string(family)),
    paste0("formula = ", toml_string(formula)),
    paste0("loglik = ", toml_num(ll)),
    paste0("aic = ", toml_num(AIC(fit))),
    paste0("df = ", df),
    paste0("n = ", n),
    "",
    "[coef]"
  ), con)
  for (name in sort(names(coefs))) {
    writeLines(paste0(toml_string(name), " = ", toml_num(coefs[[name]])), con)
  }

  ## Optional [se] block: Wald SEs, sqrt(diag(V)), same flat naming as [coef].
  ## Free to derive from the vcov already computed above -- no extra R call.
  se <- sqrt(diag(V))
  names(se) <- order
  se_lines <- c("", "[se]")
  for (name in sort(names(se))) {
    se_lines <- c(se_lines, paste0(toml_string(name), " = ", toml_num(se[[name]])))
  }
  writeLines(se_lines, con)

  tol_lines <- c(
    "",
    "[vcov]",
    paste0("order = ", toml_array(order)),
    paste0("data = ", toml_matrix(V)),
    "",
    "[tol]",
    "atol_loglik = 1e-3",
    "atol_aic = 1e-3"
  )
  if (!is.null(ranef)) tol_lines <- c(tol_lines, "rtol_ranef = 5e-2", "atol_ranef = 5e-2")
  tol_lines <- c(tol_lines, preserved_tol)
  writeLines(tol_lines, con)
  if (!is.null(ranef)) {
    writeLines(c(
      "",
      "[ranef]",
      paste0("group = ", toml_string(ranef$group)),
      paste0("sd_mu = ", toml_num(ranef$sd_mu)),
      paste0("sd_sigma = ", toml_num(ranef$sd_sigma)),
      paste0("cor = ", toml_num(ranef$cor))
    ), con)
  }
}

write_meta <- function(dir, r_call, seed, note_extra = "") {
  note <- "Generated outputs only; no drmTMB source vendored."
  if (nzchar(note_extra)) note <- paste(note, note_extra)
  lines <- c(
    paste0("drmtmb_version = ", toml_string(as.character(utils::packageVersion("drmTMB")))),
    paste0("generated_on = ", toml_string(as.character(Sys.Date()))),
    paste0("r_call = ", toml_string(r_call)),
    paste0("seed = ", as.integer(seed)),
    paste0("note = ", toml_string(note))
  )
  writeLines(lines, file.path(dir, "expected.meta.toml"))
}

write_case <- function(slug, seed, data, formula, family_label, r_call, fit,
                       note_extra = "", ranef = NULL) {
  dir <- file.path(repo_root(), "test", "parity", "fixtures", slug)
  ## Read hand-added [tol] overrides BEFORE the unlink below wipes the directory.
  preserved_tol <- read_preserved_tol(file.path(dir, "expected.toml"))
  if (dir.exists(dir)) unlink(dir, recursive = TRUE)
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(data, file.path(dir, "data.csv"), row.names = FALSE)
  write_expected(dir, family_label, formula, fit, slug, ranef = ranef,
                 preserved_tol = preserved_tol)
  write_meta(dir, r_call, seed, note_extra)
}

generate_gaussian <- function() {
  seed <- 20260604
  set.seed(seed)
  n <- 180
  dat <- data.frame(x = rnorm(n))
  dat$y <- 0.4 - 0.6 * dat$x + exp(-0.2 + 0.3 * dat$x) * rnorm(n)
  fit <- drmTMB(drm_formula(y ~ x, sigma ~ x), family = gaussian(), data = dat)
  write_case(
    "gaussian-locscale", seed, dat, "y ~ x; sigma ~ x", "gaussian",
    "drmTMB(drm_formula(y ~ x, sigma ~ x), family = gaussian(), data = dat)",
    fit
  )
}

generate_bivariate <- function() {
  seed <- 20260605
  set.seed(seed)
  n <- 180
  x <- rnorm(n)
  z1 <- rnorm(n)
  z2 <- rnorm(n)
  mu1 <- 0.2 + 0.4 * x
  mu2 <- -0.1 + 0.3 * x
  sigma1 <- exp(-0.1)
  sigma2 <- exp(0.15)
  rho <- tanh(0.35)
  dat <- data.frame(
    y1 = mu1 + sigma1 * z1,
    y2 = mu2 + sigma2 * (rho * z1 + sqrt(1 - rho^2) * z2),
    x = x
  )
  fit <- drmTMB(
    bf(mu1 = y1 ~ x, mu2 = y2 ~ x,
       sigma1 = sigma1 ~ 1, sigma2 = sigma2 ~ 1, rho12 = rho12 ~ 1),
    family = biv_gaussian(), data = dat
  )
  write_case(
    "gaussian-bivariate-rho12", seed, dat,
    paste(
      "mu1 = y1 ~ x", "mu2 = y2 ~ x", "sigma1 = sigma1 ~ 1",
      "sigma2 = sigma2 ~ 1", "rho12 = rho12 ~ 1", sep = "; "
    ),
    "biv_gaussian",
    paste(
      "drmTMB(bf(mu1 = y1 ~ x, mu2 = y2 ~ x,",
      "sigma1 = sigma1 ~ 1, sigma2 = sigma2 ~ 1, rho12 = rho12 ~ 1),",
      "family = biv_gaussian(), data = dat)"
    ),
    fit
  )
}

generate_meta <- function() {
  seed <- 20260606
  set.seed(seed)
  n <- 160
  x <- rnorm(n)
  v <- (0.12 + 0.25 * runif(n))^2
  y <- 0.2 + 0.5 * x + sqrt(v + 0.35^2) * rnorm(n)
  dat <- data.frame(y = y, x = x, v = v)
  fit <- drmTMB(drm_formula(y ~ x + meta_V(V = v), sigma ~ 1),
                family = gaussian(), data = dat)
  write_case(
    "meta-analysis-V", seed, dat, "y ~ x + meta_V(v); sigma ~ 1", "gaussian",
    "drmTMB(drm_formula(y ~ x + meta_V(V = v), sigma ~ 1), family = gaussian(), data = dat)",
    fit,
    "R drmTMB uses named meta_V(V = v); DRModels.jl runner uses meta_V(v)."
  )
}

generate_student <- function() {
  seed <- 20260607
  set.seed(seed)
  n <- 180
  x <- rnorm(n)
  y <- 0.2 + 0.4 * x + 0.7 * rt(n, df = 6)
  dat <- data.frame(y = y, x = x)
  fit <- drmTMB(drm_formula(y ~ x, sigma ~ 1, nu ~ 1),
                family = student(), data = dat)
  write_case(
    "robust-student", seed, dat, "y ~ x; sigma ~ 1; nu ~ 1", "student",
    "drmTMB(drm_formula(y ~ x, sigma ~ 1, nu ~ 1), family = student(), data = dat)",
    fit,
    "Student nu on shared log(nu - 2) scale (nu = 2 + exp(eta)); no transform."
  )
}

generate_nbinom2 <- function() {
  seed <- 20260608
  set.seed(seed)
  n <- 180
  x <- rnorm(n)
  mu <- exp(0.3 + 0.45 * x)
  theta <- 2.5
  y <- rnbinom(n, size = theta, mu = mu)
  dat <- data.frame(y = y, x = x)
  fit <- drmTMB(drm_formula(y ~ x, sigma ~ 1),
                family = nbinom2(), data = dat)
  write_case(
    "count-nbinom2", seed, dat, "y ~ x; sigma ~ 1", "nbinom2",
    "drmTMB(drm_formula(y ~ x, sigma ~ 1), family = nbinom2(), data = dat)",
    fit,
    "NB2 sigma is on the log(sigma) scale in both drmTMB and DRModels.jl; size = exp(-2*sigma)."
  )
}

generate_beta <- function() {
  seed <- 20260609
  set.seed(seed)
  n <- 180
  x <- rnorm(n)
  eta <- -0.2 + 0.5 * x
  p <- plogis(eta)
  phi <- 12
  y <- rbeta(n, p * phi, (1 - p) * phi)
  dat <- data.frame(y = y, x = x)
  fit <- drmTMB(drm_formula(y ~ x, sigma ~ 1),
                family = beta(), data = dat)
  write_case(
    "proportion-beta", seed, dat, "y ~ x; sigma ~ 1", "beta",
    "drmTMB(drm_formula(y ~ x, sigma ~ 1), family = beta(), data = dat)",
    fit
  )
}

## NB2 LOCATION–SCALE with a correlated species effect on BOTH the mean and the
## dispersion axis: bf(y ~ x + (1|p|species), sigma ~ x + (1|p|species)). This is
## the model DRModels.jl's augmented-state engine fits. The fixed sigma coefs now share
## the log(sigma) scale (DRModels.jl and drmTMB agree; size = exp(-2*sigma)), so no fixed-
## coef transform is applied. The group covariance on the sigma axis still differs:
##   * a^psi_DRM = -2 * a^sigma_drmTMB, hence sd_sigma_DRM = 2 * sd_sigma_drmTMB and
##     cor(mu,sigma) flips sign.
generate_nbinom2_locscale <- function() {
  seed <- 20260610
  set.seed(seed)
  G <- 40; m <- 30; n <- G * m
  species <- factor(rep(seq_len(G), each = m))
  x <- rnorm(n)
  # True correlated species effects in drmTMB's parameterisation (mu axis,
  # drmTMB sigma axis). drmTMB NB2 sigma is a sqrt-dispersion: theta = exp(-2*eta_sigma).
  sd_mu <- 0.5; sd_sig <- 0.2; rho <- 0.3
  S <- matrix(c(sd_mu^2, rho * sd_mu * sd_sig,
                rho * sd_mu * sd_sig, sd_sig^2), 2, 2)
  A <- matrix(rnorm(G * 2), G, 2) %*% chol(S)        # G x 2: [a_mu, a_sigma_drm]
  eta_mu  <- 0.3 + 0.45 * x + A[species, 1]
  eta_sig <- -0.10 + 0.20 * x + A[species, 2]
  theta_i <- exp(-2 * eta_sig)
  y <- rnbinom(n, size = theta_i, mu = exp(eta_mu))
  dat <- data.frame(y = y, x = x, species = species)

  fit <- drmTMB(drm_formula(y ~ x + (1 | p | species), sigma ~ x + (1 | p | species)),
                family = nbinom2(), data = dat)

  ## Extract the fitted group covariance. NOTE: adjust this accessor to drmTMB's
  ## actual VarCorr layout if it differs (glmmTMB-style assumed: a 2x2 covariance
  ## of [mu_(Intercept), sigma_(Intercept)] for the `species` grouping).
  vcc <- VarCorr(fit)
  Sigma <- vcc$cond$species
  sd_mu_hat  <- sqrt(Sigma[1, 1])
  sd_sig_hat <- sqrt(Sigma[2, 2])
  rho_hat    <- Sigma[1, 2] / (sd_mu_hat * sd_sig_hat)
  ranef <- list(group = "species",
                sd_mu = sd_mu_hat,
                sd_sigma = 2 * sd_sig_hat,   # ψ_DRM = -2·σ_drmTMB ⇒ SD ×2
                cor = -rho_hat)              # correlation flips sign under the -2 reparam

  write_case(
    "nbinom2-locscale", seed, dat,
    "y ~ x + (1|p|species); sigma ~ x + (1|p|species)", "nbinom2",
    paste("drmTMB(drm_formula(y ~ x + (1|p|species), sigma ~ x + (1|p|species)),",
          "family = nbinom2(), data = dat)"),
    fit,
    note_extra = paste("Location-scale NB2: sigma fixed coefs share the log(sigma)",
                       "scale (no transform); only the sigma-axis group covariance is",
                       "reparam'd (sd ×2, cor sign-flip) to DRModels.jl's psi convention."),
    ranef = ranef
  )
}

dir.create(file.path(repo_root(), "test", "parity", "fixtures"),
           recursive = TRUE, showWarnings = FALSE)

## NB2 with a covariate on the dispersion axis (sigma ~ x), FIXED effects — a
## location-scale-style model drmTMB DOES support today. Validates the shared
## convention (both drmTMB and DRModels.jl carry log(sigma); size = exp(-2*sigma))
## with a covariate, not just an intercept.
generate_nbinom2_dispersion <- function() {
  seed <- 20260611
  set.seed(seed)
  n <- 200
  x <- rnorm(n)
  mu <- exp(0.3 + 0.45 * x)
  eta_sigma <- -0.10 + 0.25 * x            # drmTMB sigma axis; theta = exp(-2*eta_sigma)
  y <- rnbinom(n, size = exp(-2 * eta_sigma), mu = mu)
  dat <- data.frame(y = y, x = x)
  fit <- drmTMB(drm_formula(y ~ x, sigma ~ x), family = nbinom2(), data = dat)
  write_case(
    "nbinom2-dispersion", seed, dat, "y ~ x; sigma ~ x", "nbinom2",
    "drmTMB(drm_formula(y ~ x, sigma ~ x), family = nbinom2(), data = dat)",
    fit,
    "NB2 varying dispersion; sigma coefs on log(sigma) in both (no transform)."
  )
}

## +4 FE cohort (#383): families already in drm_bridge `_bridge_family` but
## previously absent from committed fixtures / `_BRIDGE_PARITY_COHORT`.
generate_poisson <- function() {
  seed <- 20260803
  set.seed(seed)
  n <- 180
  x <- rnorm(n)
  y <- rpois(n, lambda = exp(0.3 + 0.45 * x))
  dat <- data.frame(y = y, x = x)
  fit <- drmTMB(drm_formula(y ~ x), family = poisson(), data = dat)
  write_case(
    "count-poisson", seed, dat, "y ~ x", "poisson",
    "drmTMB(drm_formula(y ~ x), family = poisson(), data = dat)",
    fit,
    "Poisson mean-only (log link); no sigma slot."
  )
}

generate_gamma <- function() {
  seed <- 20260804
  set.seed(seed)
  n <- 180
  x <- rnorm(n)
  mu <- exp(0.1 + 0.3 * x)
  shape <- 4
  y <- rgamma(n, shape = shape, scale = mu / shape)
  dat <- data.frame(y = y, x = x)
  ## drmTMB requires Gamma(link = "log"); base::gamma() is the math function.
  fit <- drmTMB(drm_formula(y ~ x, sigma ~ 1),
                family = Gamma(link = "log"), data = dat)
  write_case(
    "positive-gamma", seed, dat, "y ~ x; sigma ~ 1", "gamma",
    "drmTMB(drm_formula(y ~ x, sigma ~ 1), family = Gamma(link = \"log\"), data = dat)",
    fit,
    "Gamma: log(mu), log(sigma) with sigma = CV (alpha = 1/sigma^2); shared scale."
  )
}

generate_binomial <- function() {
  seed <- 20260805
  set.seed(seed)
  n <- 180
  x <- rnorm(n)
  trials <- rep(10L, n)
  p <- plogis(-0.2 + 0.5 * x)
  successes <- rbinom(n, size = trials, prob = p)
  failures <- as.integer(trials - successes)
  dat <- data.frame(successes = successes, failures = failures, x = x)
  fit <- drmTMB(drm_formula(cbind(successes, failures) ~ x),
                family = binomial(), data = dat)
  write_case(
    "binomial-trials", seed, dat, "cbind(successes, failures) ~ x", "binomial",
    "drmTMB(drm_formula(cbind(successes, failures) ~ x), family = binomial(), data = dat)",
    fit,
    "Binomial mean-only (logit link); cbind(successes, failures) response."
  )
}

generate_lognormal <- function() {
  seed <- 20260806
  set.seed(seed)
  n <- 180
  x <- rnorm(n)
  y <- exp(0.2 + 0.4 * x + 0.5 * rnorm(n))
  dat <- data.frame(y = y, x = x)
  fit <- drmTMB(drm_formula(y ~ x, sigma ~ 1),
                family = lognormal(), data = dat)
  write_case(
    "positive-lognormal", seed, dat, "y ~ x; sigma ~ 1", "lognormal",
    "drmTMB(drm_formula(y ~ x, sigma ~ 1), family = lognormal(), data = dat)",
    fit,
    "LogNormal: mu = E[log y], sigma = SD(log y) on log link; shared scale."
  )
}

## Optional filter: DRM_PARITY_ONLY=count-poisson,positive-gamma regenerates
## only those slugs (comma-separated). Empty = all default FE cases below.
.parity_only <- strsplit(Sys.getenv("DRM_PARITY_ONLY", unset = ""), ",", fixed = TRUE)[[1]]
.parity_only <- trimws(.parity_only[.parity_only != ""])
.run_parity <- function(slug, fn) {
  if (length(.parity_only) == 0L || slug %in% .parity_only) fn()
}

.run_parity("gaussian-locscale", generate_gaussian)
.run_parity("gaussian-bivariate-rho12", generate_bivariate)
.run_parity("meta-analysis-V", generate_meta)
.run_parity("robust-student", generate_student)
.run_parity("count-nbinom2", generate_nbinom2)
.run_parity("proportion-beta", generate_beta)
.run_parity("nbinom2-dispersion", generate_nbinom2_dispersion)
.run_parity("count-poisson", generate_poisson)
.run_parity("positive-gamma", generate_gamma)
.run_parity("binomial-trials", generate_binomial)
.run_parity("positive-lognormal", generate_lognormal)

## The coupled (1|p|species) mu/sigma correlated random effect is NOT yet
## supported by drmTMB ("planned for a later non-Gaussian random-effect gate"),
## so this case is guarded: it activates automatically once drmTMB implements it.
## DRModels.jl already fits this model; until drmTMB catches up it is validated
## internally (marginal vs Gauss–Hermite, exact gradient vs finite differences,
## recovery, stationarity), not against drmTMB.
if (length(.parity_only) == 0L || "nbinom2-locscale" %in% .parity_only) {
  tryCatch(generate_nbinom2_locscale(),
           error = function(e) message(
               "Skipping `nbinom2-locscale`: drmTMB does not yet support the coupled ",
               "(1|p|species) mu/sigma random effect for nbinom2 — ", conditionMessage(e)))
}

message("Generated drmTMB parity fixtures under test/parity/fixtures/")
