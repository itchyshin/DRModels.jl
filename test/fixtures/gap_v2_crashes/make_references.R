# R references for the gap-map v2 crash fixes (docs/dev-log/2026-11-09-julia-crashes.md).
# One single-process call; output: r_reference.csv (long format: id, key, value).
# Run from this directory:
#   DRMTMB_LIB=~/drm_campaign_20261108/Rlib nice -n 10 Rscript make_references.R
# drmTMB 0.8.1. Data: data_v2.csv (160 rows, md5 ae47edf77c31bcd8c3b4c933e9201239), shared with Julia.
if (nzchar(Sys.getenv("DRMTMB_LIB"))) .libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
stopifnot(as.character(packageVersion("drmTMB")) == "0.8.1")
d <- read.csv("data_v2.csv", stringsAsFactors = FALSE, na.strings = "NA")
for (v in c("g", "g2", "h", "f", "f2", "gna", "gu", "y_cat")) d[[v]] <- factor(d[[v]])
d$y_ordf <- factor(d$y_ordf, levels = c("o1", "o2", "o3", "o4"), ordered = TRUE)

out <- list()
add <- function(id, fit) {
  out[[length(out) + 1]] <<- data.frame(id = id, key = "loglik", value = as.numeric(logLik(fit)))
  out[[length(out) + 1]] <<- data.frame(id = id, key = "nobs", value = as.numeric(nobs(fit)))
  cf <- unlist(coef(fit))
  out[[length(out) + 1]] <<- data.frame(id = id, key = paste0("coef:", names(cf)), value = as.numeric(cf))
}
gauss <- gaussian()
add("reml_fx1_sigma_x1f", drmTMB(bf(y_g ~ f * x1, sigma ~ x1 + f), family = gauss, data = d, REML = TRUE))
add("cumul_label", drmTMB(bf(y_ordf ~ x1), family = cumulative_logit(), data = d))
add("weights_na_response", drmTMB(bf(y_gm ~ x1, sigma ~ 1), family = gauss, data = d, weights = w))
add("dbar_gaussian", drmTMB(bf(y_g ~ x1 + (1 + x1 || g), sigma ~ 1), family = gauss, data = d))
add("indep_gaussian", drmTMB(bf(y_g ~ x1 + (1 | g) + (0 + x1 | g), sigma ~ 1), family = gauss, data = d))
add("poly2_gaussian", drmTMB(bf(y_g ~ poly(x1, 2), sigma ~ 1), family = gauss, data = d))
add("na_group_gaussian", drmTMB(bf(y_g ~ x1 + (1 | gna), sigma ~ 1), family = gauss, data = d))
res <- do.call(rbind, out)
write.csv(res, "r_reference.csv", row.names = FALSE)
print(res[res$key %in% c("loglik", "nobs"), ])
