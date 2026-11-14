# Reference fits for the offset cells (drmTMB 0.8.1, ML; meta_V + offset under REML = TRUE is refused in Julia, see the test), written to reference_drmtmb.csv.
# Usage: DRMTMB_LIB=<lib> Rscript ref_offset_routes.R <data_v2.csv> <data_offset_routes.csv> <out.csv>
# <data_v2.csv> is the gap-map v2 data; the script writes the committed column subset to arg 2 and fits from it.
args <- commandArgs(TRUE)
if (nzchar(Sys.getenv("DRMTMB_LIB"))) .libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
stopifnot(as.character(packageVersion("drmTMB")) == "0.8.1")
full <- read.csv(args[1])
d <- full[, c("id", "x1", "x2", "off", "w", "y_pois", "y_zip", "vi", "yi")]
write.csv(d, args[2], row.names = FALSE)
d <- read.csv(args[2])
fits <- list(
  pois_one = drmTMB(bf(y_pois ~ x1 + offset(off)), family = stats::poisson(link = "log"), data = d),
  pois_two = drmTMB(bf(y_pois ~ x1 + offset(off) + offset(log(w))), family = stats::poisson(link = "log"), data = d),
  zip_off  = drmTMB(bf(y_zip ~ x1 + offset(off), zi ~ x2), family = stats::poisson(link = "log"), data = d),
  meta_off = drmTMB(bf(yi ~ x1 + offset(off) + meta_V(V = vi), sigma ~ 1), family = gaussian(), data = d))
out <- do.call(rbind, lapply(names(fits), function(k) {
  f <- fits[[k]]; est <- as.numeric(f$opt$par); v <- vcov(f); se <- sqrt(diag(v))
  stopifnot(length(est) == length(se))
  data.frame(cell = k, name = names(se), estimate = est, se = se, loglik = as.numeric(logLik(f)),
             is_converged = isTRUE(drmTMB::is_converged(f)), pdhess = isTRUE(f$sdr$pdHess))
}))
write.csv(out, args[3], row.names = FALSE)
print(out, digits = 12)
print(lapply(fits, function(f) names(f$opt$par)))
