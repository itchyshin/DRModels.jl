# gen_data.R — R oracle for the cumulative_logit() mu random-intercept parity
# fixture (#563 S8). Run with drmTMB 0.7.0:
#
#   Rscript test/parity/fixtures/cumlogit-mu-ranef/gen_data.R
#
# DGP and seed copied verbatim from drmTMB's own Arc 2a sentinel
# (tests/testthat/test-arc2a-mu-random-intercept.R, the
# "cumulative_logit mu supports an ordinary random intercept" test) so the
# ground truth is drmTMB's own proven-recovering DGP, not a re-derived one.

suppressMessages(library(drmTMB))

set.seed(9)
n_id <- 45L; n_each <- 18L
id <- factor(rep(seq_len(n_id), each = n_each)); n <- length(id)
dat <- data.frame(id = id, x = stats::rnorm(n))
sd_id <- 0.7; u_id <- stats::rnorm(n_id, sd = sd_id); u_id <- u_id - mean(u_id)
cut <- c(-1, 0, 1)
lat <- 0.8 * dat$x + u_id[id] + stats::rlogis(n)
dat$y <- ordered(findInterval(lat, cut) + 1L, levels = 1:4)

fit <- drmTMB(bf(y ~ x + (1 | id)), family = cumulative_logit(), data = dat)

cat("convergence:", fit$opt$convergence, "pdHess:", fit$sdr$pdHess, "\n")
cat("coef mu (slope):", coef(fit, "mu"), "\n")
cat("cutpoints:", unname(fit$ordinal$cutpoints), "\n")
cat("sd_id estimate:", unname(fit$sdpars$mu[["(1 | id)"]]), "\n")
cat("loglik:", as.numeric(logLik(fit)), "\n")

# The correlated random slope `(1 + x | id)` stays rejected by drmTMB 0.7.0
# (`validate_cumulative_logit_mu_random_terms()`): only independent `(1 | id)`
# intercepts and `(0 + x | id)` slopes are implemented. DRModels.jl matches that
# same-target scope this slice.
res <- tryCatch(
  drmTMB(bf(y ~ x + (1 + x | id)), family = cumulative_logit(), data = dat),
  error = function(e) conditionMessage(e)
)
cat("correlated-slope-fit error:", res, "\n")

# integer-coded y (1..K) for DRModels.jl, which does not use R's `ordered` class
dat$y_int <- as.integer(dat$y)
write.csv(dat[, c("id", "x", "y_int")], "test/parity/fixtures/cumlogit-mu-ranef/data.csv",
          row.names = FALSE)
