# R reference for test/test_nongauss_corr_blocks.jl: drmTMB 0.8.1 ML, the ten cells of the
# wave-10 gap map that this change closes or classifies.
# Usage (Totoro, library ~/drm_campaign_20261108/Rlib):
#   DRMTMB_LIB=~/drm_campaign_20261108/Rlib Rscript run_r_nongauss_corr.R <data dir> <out csv>
# <data dir> holds data_<family>_ordinary.csv (this folder, and ../binomial_indep_restarts/
# g12_binomial.csv for the binomial file, which is byte-identical to data_binomial_ordinary.csv).
args <- commandArgs(TRUE)
DD <- args[1]; out <- args[2]
if (nzchar(Sys.getenv("DRMTMB_LIB"))) .libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
cat("drmTMB", as.character(packageVersion("drmTMB")), "\n")

read_d <- function(fam) {
  f <- file.path(DD, sprintf("data_%s_ordinary.csv", fam))
  if (!file.exists(f) && fam == "binomial") f <- file.path(DD, "g12_binomial.csv")
  d <- read.csv(f); d$g <- factor(d$g, levels = unique(d$g)); d
}
fam_of <- function(f) switch(f, poisson = poisson(), nb2 = nbinom2(), lognormal = lognormal(),
                             binomial = binomial(link = "logit"))
mu_of <- function(q) {
  bar <- paste(c("1", "x1", "x2", "x3")[seq_len(q)], collapse = " + ")
  as.formula(sprintf("y ~ x1 + x2 + x3 + (%s | g)", bar))
}
sd_of <- function(which) switch(which,
  first = list(sd(g, coef = "(Intercept)") ~ z), last = list(sd(g, coef = "x1") ~ z),
  both_one = list(sd(g, coef = "(Intercept)") ~ 1, sd(g, coef = "x1") ~ 1),
  both_z = list(sd(g, coef = "(Intercept)") ~ z, sd(g, coef = "x1") ~ z), none = list())

cells <- list(
  list("poisson", 2, "first"), list("poisson", 2, "last"),
  list("nb2", 2, "first"), list("nb2", 2, "last"),
  list("lognormal", 2, "both_one"), list("lognormal", 2, "both_z"),
  list("lognormal", 2, "first"), list("lognormal", 2, "last"),
  list("binomial", 3, "none"), list("binomial", 4, "none"))

rows <- list()
for (cl in cells) {
  fam <- cl[[1]]; q <- cl[[2]]; which <- cl[[3]]
  id <- sprintf("%s|ordinary|q%d|sd-%s|ML", fam, q, which)
  d <- read_d(fam)
  parts <- c(list(mu_of(q)), if (!(fam %in% c("poisson", "binomial"))) list(sigma = ~ 1), sd_of(which))
  fit <- tryCatch(drmTMB(do.call(bf, parts), family = fam_of(fam), data = d), error = function(e) e)
  if (inherits(fit, "error")) {
    rows[[id]] <- data.frame(cell = id, what = "error", name = "", value = NA, msg = conditionMessage(fit)); next
  }
  ll <- as.numeric(logLik(fit))
  pd <- isTRUE(fit$sdr$pdHess); isc <- isTRUE(drmTMB::is_converged(fit))
  par <- fit$opt$par
  rows[[id]] <- rbind(
    data.frame(cell = id, what = "loglik", name = "logLik", value = ll, msg = ""),
    data.frame(cell = id, what = "flag", name = "pdHess", value = as.numeric(pd), msg = ""),
    data.frame(cell = id, what = "flag", name = "is_converged", value = as.numeric(isc), msg = ""),
    data.frame(cell = id, what = "par", name = names(par), value = as.numeric(par), msg = ""))
}
write.csv(do.call(rbind, rows), out, row.names = FALSE)
