# Regenerates data_meta_reml.csv and ref_meta_reml.csv (drmTMB 0.8.1, REML = TRUE, meta_V).
# Run: Rscript make_reference.R   (needs drmTMB installed)
suppressMessages(library(drmTMB))
set.seed(20261110)
k <- 40
d <- data.frame(g = factor(sample(sprintf("s%02d", 1:8), k, replace = TRUE)),
                x1 = rnorm(k), vi = runif(k, 0.05, 0.4))
u <- rnorm(8, 0, 0.5); names(u) <- sprintf("s%02d", 1:8)
d$yi <- 0.3 + 0.6 * d$x1 + u[as.character(d$g)] + rnorm(k, 0, 0.35) + rnorm(k, 0, sqrt(d$vi))
write.csv(d, "data_meta_reml.csv", row.names = FALSE)
rows <- list()
add <- function(cell, fit) {
  co <- summary(fit)$coefficients
  rows[[length(rows) + 1]] <<- data.frame(cell = cell, key = "logLik", est = as.numeric(logLik(fit)), se = NA)
  rows[[length(rows) + 1]] <<- data.frame(cell = cell, key = rownames(co), est = co[, "estimate"], se = co[, "std_error"])
  sdp <- summary(fit)$sdpars
  if (!is.null(sdp$mu)) rows[[length(rows) + 1]] <<- data.frame(cell = cell, key = "sd_g", est = unname(sdp$mu[["(1 | g)"]]), se = NA)
}
f1 <- drmTMB(bf(yi ~ x1 + meta_V(V = vi), sigma ~ 1), family = gaussian(), data = d, REML = TRUE)
f2 <- drmTMB(bf(yi ~ x1 + (1 | g) + meta_V(V = vi), sigma ~ 1), family = gaussian(), data = d, REML = TRUE)
add("meta_reml", f1); add("meta_re_reml", f2)
ref <- do.call(rbind, rows); rownames(ref) <- NULL
write.csv(ref, "ref_meta_reml.csv", row.names = FALSE)
print(ref, digits = 12)
