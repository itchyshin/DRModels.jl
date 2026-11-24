# gen_beta_binomial_family_parity.R -- R oracle for the Beta / Binomial /
# BetaBinomial parity tests (fixed effects + (1 | g)) in
# test/test_parity_beta_binomial_family.jl.
#
#   Rscript test/parity/gen_beta_binomial_family_parity.R
#
# Writes data.csv under test/parity/fixtures/<slug>/ and prints the drmTMB
# numbers that the Julia test hard-codes. Generated outputs only; no drmTMB
# source is vendored. Run with the drmTMB version recorded in the test header.
suppressMessages(library(drmTMB))

root <- file.path("test", "parity", "fixtures")
inv_logit <- function(z) 1 / (1 + exp(-z))

report <- function(slug, fit, sd_name = NULL) {
  cat("\n==", slug, "==\n")
  cat("convergence:", fit$opt$convergence, " pdHess:", fit$sdr$pdHess, "\n")
  for (p in c("mu", "sigma")) {
    cc <- tryCatch(coef(fit, p), error = function(e) NULL)
    if (!is.null(cc) && length(cc)) cat("coef", p, ":", sprintf("%.10g", cc), "\n")
  }
  if (!is.null(sd_name))
    cat("sd mu:", sprintf("%.10g", unname(fit$sdpars$mu[[sd_name]])), "\n")
  cat("logLik:", sprintf("%.10g", as.numeric(logLik(fit))), "\n")
}
save_data <- function(slug, dat) {
  d <- file.path(root, slug)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  write.csv(dat, file.path(d, "data.csv"), row.names = FALSE)
}

## ---- Beta, mu ~ x + (1 | g), sigma ~ 1 ------------------------------------
set.seed(20261031)
G <- 40; m <- 8; n <- G * m
g <- factor(rep(seq_len(G), each = m)); x <- rnorm(n)
b <- rnorm(G, sd = 0.5)
mu <- inv_logit(-0.2 + 0.5 * x + b[g]); phi <- 20
dat <- data.frame(g = g, x = x, y = rbeta(n, mu * phi, (1 - mu) * phi))
save_data("beta-mu-ranef", dat)
fit <- drmTMB(bf(y ~ x + (1 | g), sigma ~ 1), family = beta(), data = dat)
report("beta-mu-ranef", fit, "(1 | g)")

## ---- Binomial, cbind(s, f) ~ x + (1 | g) ----------------------------------
set.seed(20261032)
g <- factor(rep(seq_len(G), each = m)); x <- rnorm(n)
b <- rnorm(G, sd = 0.6)
p <- inv_logit(-0.2 + 0.5 * x + b[g]); ntr <- 20
s <- rbinom(n, ntr, p)
dat <- data.frame(g = g, x = x, successes = s, failures = ntr - s)
save_data("binomial-mu-ranef", dat)
fit <- drmTMB(bf(cbind(successes, failures) ~ x + (1 | g)),
              family = binomial(), data = dat)
report("binomial-mu-ranef", fit, "(1 | g)")

## ---- BetaBinomial, fixed effects ------------------------------------------
set.seed(20261033)
x <- rnorm(n)
p <- inv_logit(-0.2 + 0.5 * x); phi <- 12; ntr <- 20
pp <- rbeta(n, p * phi, (1 - p) * phi)
s <- rbinom(n, ntr, pp)
dat <- data.frame(x = x, successes = s, failures = ntr - s)
save_data("betabinomial-fixed", dat)
fit <- drmTMB(bf(cbind(successes, failures) ~ x, sigma ~ 1),
              family = beta_binomial(), data = dat)
report("betabinomial-fixed", fit)

## ---- BetaBinomial, mu ~ x + (1 | g) ---------------------------------------
set.seed(20261034)
g <- factor(rep(seq_len(G), each = m)); x <- rnorm(n)
b <- rnorm(G, sd = 0.5)
p <- inv_logit(-0.2 + 0.5 * x + b[g])
pp <- rbeta(n, p * phi, (1 - p) * phi)
s <- rbinom(n, ntr, pp)
dat <- data.frame(g = g, x = x, successes = s, failures = ntr - s)
save_data("betabinomial-mu-ranef", dat)
fit <- drmTMB(bf(cbind(successes, failures) ~ x + (1 | g), sigma ~ 1),
              family = beta_binomial(), data = dat)
report("betabinomial-mu-ranef", fit, "(1 | g)")
