# gen_cumlogit_laplace_fitted.R -- drmTMB 0.8.1 `fitted()` for the four CumulativeLogit (1 | g) cells of
# test/test_cumlogit_ordinary_laplace_parity.jl.
#
#   DRMTMB_LIB=<library with drmTMB 0.8.1> Rscript test/parity/gen_cumlogit_laplace_fitted.R
#
# Writes test/fixtures/cumlogit_laplace_parity/<cell>/fitted_r.csv with, per row of data.csv, the drmTMB
# `fitted()` value (`cond`: the expected category score with the random-effect modes included) and the
# expected category score with the random effect at zero (`pop`, from the population category
# probabilities already in ref.toml's recipe: P(y = k) = logistic(cut_k - x beta) - logistic(cut_(k-1) - x beta)).
if (nzchar(Sys.getenv("DRMTMB_LIB"))) .libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
stopifnot(as.character(utils::packageVersion("drmTMB")) == "0.8.1")
for (route in c("cl_1", "cl_2", "cl_3", "cl_4")) {
  dir <- file.path("test", "fixtures", "cumlogit_laplace_parity", route)
  d <- read.csv(file.path(dir, "data.csv"), stringsAsFactors = FALSE)
  K <- max(d$y)
  d$g <- factor(d$g)
  d$y_ord <- ordered(d$y, levels = seq_len(K))
  f <- drmTMB(bf(y_ord ~ x + (1 | g)), family = cumulative_logit(), data = d)
  par <- f$opt$par
  cuts <- unname(f$ordinal$cutpoints)
  eta <- unname(par[1]) * d$x
  cdf <- cbind(0, vapply(cuts, function(ct) stats::plogis(ct - eta), numeric(nrow(d))), 1)
  prob <- cdf[, -1, drop = FALSE] - cdf[, -ncol(cdf), drop = FALSE]
  out <- data.frame(row = seq_len(nrow(d)), cond = as.numeric(fitted(f)), pop = as.numeric(prob %*% seq_len(K)))
  write.csv(out, file.path(dir, "fitted_r.csv"), row.names = FALSE)
  cat(sprintf("%s rows %d K %d logLik %.10f max|cond-pop| %.4f\n", route, nrow(out), K,
              as.numeric(logLik(f)), max(abs(out$cond - out$pop))))
}
