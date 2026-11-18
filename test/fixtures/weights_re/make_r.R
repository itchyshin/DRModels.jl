# R reference for prior `weights` with random effects, meta_V and bivariate Gaussian (drmTMB 0.8.1).
# Writes data_weights.csv (a subset of the gap-map v2 grid data, 160 rows) and
# r_reference.tsv with columns model, key, estimate, se  (keys: ll, conv, pdhess, then every row of
# summary()$coefficients and summary()$parameters).
# Usage: Rscript make_r.R <data_v2.csv> <outdir>      (R with drmTMB 0.8.1)
args <- commandArgs(TRUE)
suppressMessages(library(drmTMB))
d <- read.csv(args[1], stringsAsFactors = FALSE)
out <- args[2]
d$g <- factor(d$g)
d$wz <- d$w * (d$id %% 7 != 0)       # zero weights on every seventh row
keep <- c("id", "g", "x1", "x2", "w", "wi", "wz", "y_g", "y_pois", "y1", "y2", "vi", "yi")
write.csv(d[, keep], file.path(out, "data_weights.csv"), row.names = FALSE)
fits <- list(
  gauss_re_ml    = function() drmTMB(bf(y_g ~ x1 + (1 | g), sigma ~ 1), family = gaussian(), data = d, weights = w),
  gauss_re_reml  = function() drmTMB(bf(y_g ~ x1 + (1 | g), sigma ~ 1), family = gaussian(), data = d, weights = w, REML = TRUE),
  gauss_re_int   = function() drmTMB(bf(y_g ~ x1 + (1 | g), sigma ~ 1), family = gaussian(), data = d, weights = wi),
  gauss_re_zero  = function() drmTMB(bf(y_g ~ x1 + (1 | g), sigma ~ 1), family = gaussian(), data = d, weights = wz),
  gauss_re_sig   = function() drmTMB(bf(y_g ~ x1 + (1 | g), sigma ~ x2), family = gaussian(), data = d, weights = w),
  pois_re        = function() drmTMB(bf(y_pois ~ x1 + (1 | g)), family = stats::poisson(link = "log"), data = d, weights = w),
  pois_re_int    = function() drmTMB(bf(y_pois ~ x1 + (1 | g)), family = stats::poisson(link = "log"), data = d, weights = wi),
  pois_re_zero   = function() drmTMB(bf(y_pois ~ x1 + (1 | g)), family = stats::poisson(link = "log"), data = d, weights = wz),
  meta           = function() drmTMB(bf(yi ~ x1 + meta_V(V = vi), sigma ~ 1), family = gaussian(), data = d, weights = w),
  meta_int       = function() drmTMB(bf(yi ~ x1 + meta_V(V = vi), sigma ~ 1), family = gaussian(), data = d, weights = wi),
  meta_zero      = function() drmTMB(bf(yi ~ x1 + meta_V(V = vi), sigma ~ 1), family = gaussian(), data = d, weights = wz),
  biv            = function() drmTMB(bf(mu1 = y1 ~ x1, mu2 = y2 ~ x2, sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1), family = biv_gaussian(), data = d, weights = w),
  biv_zero       = function() drmTMB(bf(mu1 = y1 ~ x1, mu2 = y2 ~ x2, sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1), family = biv_gaussian(), data = d, weights = wz),
  biv_re         = function() drmTMB(bf(mu1 = y1 ~ x1 + (1 | p | g), mu2 = y2 ~ x2 + (1 | p | g), sigma1 = ~ 1, sigma2 = ~ 1, rho12 = ~ 1), family = biv_gaussian(), data = d, weights = w)
)
rows <- list()
add <- function(m, key, est, se = NA_real_) rows[[length(rows) + 1L]] <<- data.frame(model = m, key = key, estimate = est, se = se)
for (m in names(fits)) {
  f <- tryCatch(fits[[m]](), error = function(e) e)
  if (inherits(f, "error")) { add(m, "error", NA); message(m, ": ", conditionMessage(f)); next }
  add(m, "ll", as.numeric(logLik(f)))
  add(m, "conv", as.numeric(isTRUE(is_converged(f))))
  add(m, "pdhess", as.numeric(isTRUE(f$sdr$pdHess)))
  sm <- summary(f)
  for (tb in c("coefficients", "parameters")) {
    cf <- sm[[tb]]
    if (is.null(cf)) next
    cn <- colnames(cf)
    for (i in seq_len(nrow(cf))) add(m, paste0(tb, ":", rownames(cf)[i]), cf[i, "estimate"], if ("std_error" %in% cn) cf[i, "std_error"] else NA)
  }
}
res <- do.call(rbind, rows)
write.table(res, file.path(out, "r_reference.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
