# Reference for the Poisson correlated block with a mean offset under `marginal = :Laplace`:
#   c ~ x + offset(logb) + (1 + x | g), data_poisson_offset_corr.csv (made by gen_data_offset_corr.jl).
# drmTMB 0.8.1, engine = "tmb", library ~/drm_campaign_20261108/Rlib on Totoro. Usage: Rscript gen_ref_offset_corr.R r_reference_offset_corr.csv
.libPaths(c("~/drm_campaign_20261108/Rlib", .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
d <- read.csv("data_poisson_offset_corr.csv"); d$g <- factor(d$g)
fit <- drmTMB(bf(c ~ x + offset(logb) + (1 + x | g)), family = poisson(), data = d)
s <- summary(fit)
co <- as.data.frame(s$coefficients); sp <- as.data.frame(s$parameters)
out <- rbind(data.frame(term = rownames(co), estimate = co$estimate, se = co$std_error),
             data.frame(term = sp$parm, estimate = sp$estimate, se = sp$std_error))
out$loglik <- as.numeric(logLik(fit)); out$conv <- fit$opt$convergence
out$pdhess <- isTRUE(fit$sdr$pdHess); out$is_converged <- isTRUE(drmTMB::is_converged(fit))
print(out); write.csv(out, commandArgs(TRUE)[1], row.names = FALSE)
