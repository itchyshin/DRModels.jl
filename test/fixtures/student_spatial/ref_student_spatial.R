# Reference fits for Student `spatial(1 | g, coords)` and `spatial(1 + x1 | g, coords)` (drmTMB 0.8.1, ML),
# written to reference_drmtmb.csv.
# Usage: DRMTMB_LIB=<lib> Rscript ref_student_spatial.R <data_student_spatial.csv> <coords.csv> <out.csv>
# coords.csv is the committed ../struct_sd_counts/coords.csv.
args <- commandArgs(TRUE)
if (nzchar(Sys.getenv("DRMTMB_LIB"))) .libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
d <- read.csv(args[1]); d$g <- factor(d$g, levels = unique(d$g))
cc <- read.csv(args[2]); coords <- as.matrix(cc[, c("easting", "northing")]); rownames(coords) <- cc$id
rows <- list()
for (cell in c("q1", "q2")) {
  bar <- if (cell == "q1") "1" else "1 + x1"
  mu <- as.formula(paste("y ~ x1 + x2 + x3 + spatial(", bar, "| g, coords = coords)")); environment(mu) <- environment()
  fit <- drmTMB(bf(mu, sigma = ~ 1), family = student(), data = d)
  est <- as.numeric(fit$opt$par); nm <- names(fit$opt$par)
  se <- sqrt(diag(vcov(fit)))
  cat(cell, "opt par names:", nm, "\n"); cat(cell, "vcov names:", names(se), "\n")
  rows[[cell]] <- data.frame(cell = cell, idx = seq_along(est), name = nm, estimate = est,
                             se = c(unname(se), rep(NA, length(est) - length(se))),
                             loglik = as.numeric(logLik(fit)),
                             is_converged = isTRUE(drmTMB::is_converged(fit)),
                             pdHess = isTRUE(fit$sdr$pdHess))
}
out <- do.call(rbind, rows)
write.csv(out, args[3], row.names = FALSE)
print(out, digits = 12)
