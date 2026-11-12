# Laplace-parity reference: fit the single-term `ordinary` cells `(1 | g)` and the correlated `corr` blocks `(1 + x1 | g)` (and any other cell kind of
# ../nongauss_indep/gen_ref.R) with R drmTMB (ML) on the committed gap-map data in ../nongauss_indep,
# and write estimates, standard errors and logLik. drmTMB is only RUN, never vendored.
# Usage (from this directory):  Rscript gen_ref.R r_reference.csv [cells.tsv]
#   cells_crossed.tsv + data_<family>_crossed.csv: the q = 1 crossed-intercept cells (`(1 | g) + (1 | g2)`), r_reference_crossed.csv
#   data_<family>.csv  400 rows, 39 groups g01..g39 (the wave-8 gap-map data, `ordinary` shape)
#   cells.tsv          one row per cell: family, kind (indep | slope), q, pat (none | one | z | first | last)
# Reference: drmTMB 0.8.1, engine = "tmb", library ~/drm_campaign_20261108/Rlib on Totoro.
.libPaths(c("~/drm_campaign_20261108/Rlib", .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
out <- commandArgs(TRUE)[1]
cells_file <- if (length(commandArgs(TRUE)) >= 2) commandArgs(TRUE)[2] else "cells.tsv"
cells <- read.delim(cells_file, stringsAsFactors = FALSE)
fam_of <- function(f) switch(f,
  poisson = poisson(), nb2 = nbinom2(), gamma = Gamma(link = "log"),
  binomial = binomial(link = "logit"), beta = beta_family(), lognormal = lognormal(),
  student = student(), tweedie = tweedie())
has_sigma <- function(f) !(f %in% c("poisson", "binomial"))
cname_r <- c("(Intercept)", "x1", "x2", "x3")

build <- function(cl) {
  q <- cl$q; kind <- cl$kind
  re <- switch(kind,
    ordinary = "(1 | g)",
    corr = paste0("(", paste(c("1", "x1", "x2", "x3")[seq_len(q)], collapse = " + "), " | g)"),
    crossed = "(1 | g) + (1 | g2)",
    slope = "(0 + x1 | g)",
    indep = paste0("(1 | g)", paste0(" + (0 + ", c("x1", "x2", "x3")[seq_len(q - 1)], " | g)", collapse = "")))
  mu <- as.formula(paste("y ~ x1 + x2 + x3 +", re)); environment(mu) <- environment()
  sel <- switch(cl$pat, none = integer(0), one = seq_len(q), z = seq_len(q), first = 1L, last = q)
  rhs <- if (cl$pat == "one") "1" else "z"
  sds <- lapply(sel, function(j) {
    cf <- if (kind == "slope") ', coef = "x1"' else if (kind == "crossed") "" else sprintf(', coef = "%s"', cname_r[j])
    as.formula(sprintf("sd(g%s) ~ %s", cf, rhs))
  })
  if (kind == "crossed" && cl$pat != "none") {
    sds <- c(sds, list(as.formula(sprintf("sd(g2) ~ %s", if (cl$pat == "one") "1" else "z2"))))
  }
  parts <- list(mu)
  if (has_sigma(cl$family)) parts <- c(parts, list(sigma = as.formula("~ 1")))
  do.call(bf, c(parts, sds))
}

rows <- list(); status <- list()
for (i in seq_len(nrow(cells))) {
  cl <- cells[i, ]
  d <- read.csv(file.path("..", "nongauss_indep", sprintf(if (cl$kind == "crossed") "data_%s_crossed.csv" else "data_%s.csv", cl$family)))
  d$g <- factor(d$g, levels = unique(d$g)); d$g2 <- factor(d$g2)
  res <- tryCatch({
    fit <- drmTMB(build(cl), family = fam_of(cl$family), data = d)
    s <- summary(fit)
    co <- as.data.frame(s$coefficients); sp <- as.data.frame(s$parameters)
    ll <- as.numeric(logLik(fit))
    list(ok = TRUE,
         rows = rbind(data.frame(term = rownames(co), estimate = co$estimate, se = co$std_error),
                      data.frame(term = sp$parm, estimate = sp$estimate, se = sp$std_error)),
         ll = ll, conv = fit$opt$convergence, pd = isTRUE(fit$sdr$pdHess),
         isc = isTRUE(drmTMB::is_converged(fit)))
  }, error = function(e) list(ok = FALSE, msg = substr(gsub("[\t\r\n]+", " ", conditionMessage(e)), 1, 300)))
  if (res$ok) {
    rows[[length(rows) + 1]] <- cbind(cell = cl$cell_id, res$rows, loglik = res$ll, conv = res$conv,
                                      pdhess = res$pd, is_converged = res$isc)
    status[[i]] <- data.frame(cell = cl$cell_id, status = "FITS", msg = "")
  } else {
    status[[i]] <- data.frame(cell = cl$cell_id, status = "ERROR", msg = res$msg)
  }
  cat(sprintf("%d/%d %s %s\n", i, nrow(cells), cl$cell_id, status[[i]]$status))
}
write.csv(do.call(rbind, rows), out, row.names = FALSE)
write.csv(do.call(rbind, status), sub("\\.csv$", "_status.csv", out), row.names = FALSE)
