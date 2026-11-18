# R reference value for the Binomial `(1 | g)` cell of test/test_fitted_conditional.jl (review of #1324).
#
#   DRMTMB_LIB=~/drm_campaign_20261108/Rlib Rscript p17_fitted_binomial_r.R data.csv p17_fitted_binomial_reference.csv
#
# Same layout as p17_fitted_review_r.R: per used row, the drmTMB 0.8.1 `fitted()` value (`cond`, random-effect
# modes included) and `pop`, the population value (every random effect at zero).
args <- commandArgs(TRUE)
.libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
d <- read.csv(args[1], stringsAsFactors = FALSE, na.strings = "NA")
d$g <- factor(d$g)
fit <- suppressWarnings(drmTMB(bf(y_bin ~ x1 + (1 | g)), family = binomial(), data = d))
cond <- as.numeric(fitted(fit))
pop <- as.numeric(predict(fit, newdata = d, dpar = "mu", type = "response"))
cat("OK W|binomial|(1|g)", length(cond), "logLik", as.numeric(logLik(fit)), "\n")
write.csv(data.frame(cell = "W|binomial|(1|g)", row = seq_along(cond), cond = cond, pop = pop),
          args[2], row.names = FALSE)
