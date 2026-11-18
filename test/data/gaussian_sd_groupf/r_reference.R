# R reference for the Gaussian SD "Group F" closure: drmTMB (engine = "tmb") fits of the models below on the
# three committed datasets gsf_s*.csv, written as a long table  dataset, model, key, estimate, se.
# Keys: ll, conv, pdhess; mu:<coef>, sigma:<coef>, sdf:<target>:<coef> (SD-formula coefficients, with se);
#       sd:<dpar>:<term> (natural-scale SD of a term with no SD formula, se on the log scale);
#       cor:<dpar>:<name> (correlation, no se); error (the fit failed in R).
# Per-group fitted SDs of an SD-formula target are not stored (the formula coefficients carry them).
# Run (R with drmTMB 0.8.1; .libPaths() set to a library holding it):  Rscript r_reference.R <dir with gsf_s*.csv> <out.tsv>
args <- commandArgs(TRUE)
suppressMessages(library(drmTMB))
dir <- args[1]; out <- args[2]
fx <- "y ~ x1 + x2 + x3"
G <- function(re, sd = character()) list(resp = "y", mu = paste(fx, "+", re), sigma = "~ 1", sd = sd)
models <- list(
  g_blk_slope       = G("(1 + x1 | g) + (0 + x2 | g2)"),
  g_blk_slope_sd    = G("(1 + x1 | g) + (0 + x2 | g2)", 'sd(g2, coef = "x2") ~ z2'),
  g_blk_indep       = G("(1 + x1 | g) + (0 + x3 | g)"),
  g_blk_indep_sd    = G("(1 + x1 | g) + (0 + x3 | g)", c('sd(g, coef = "x1") ~ z', 'sd(g, coef = "x3") ~ z')),
  g_blk_indep_sd2   = G("(1 + x1 | g) + (0 + x3 | g)", 'sd(g, coef = "x3") ~ z'),
  g_two_blk         = G("(1 + x1 | g) + (1 + x2 | g2)"),
  g_two_blk_sd      = G("(1 + x1 | g) + (1 + x2 | g2)", c('sd(g, coef = "x1") ~ z', 'sd(g2, coef = "x2") ~ z2')),
  g_blk_int_slope   = G("(1 + x1 | g) + (1 | g2) + (0 + x2 | g2)"),
  g_blk_int_slope_sd = G("(1 + x1 | g) + (1 | g2) + (0 + x2 | g2)", c('sd(g2, coef = "(Intercept)") ~ z2', 'sd(g2, coef = "x2") ~ z2')),
  g_q3_slope        = G("(1 + x1 + x2 | g) + (0 + x3 | g2)"),
  f_corr  = list(resp = "yf", mu = "yf ~ x1 + (1 | p | g)", sigma = "~ 1 + (1 | p | g)", sd = "sd(g) ~ z"),
  f_indep = list(resp = "yf", mu = "yf ~ x1 + (1 | g)",     sigma = "~ 1 + (1 | g)",     sd = "sd(g) ~ z"),
  f_cross = list(resp = "yf", mu = "yf ~ x1 + (1 | g)",     sigma = "~ 1 + (1 | g2)",    sd = "sd(g) ~ z")
)
rows <- list()
add <- function(ds, m, key, est, se = NA_real_) {
  rows[[length(rows) + 1L]] <<- data.frame(dataset = ds, model = m, key = key, estimate = est, se = se)
}
for (s in 1:3) {
  ds <- sprintf("gsf_s%d", s)
  d <- read.csv(file.path(dir, paste0(ds, ".csv")))
  for (m in names(models)) {
    sp <- models[[m]]
    parts <- c(list(as.formula(sp$mu), sigma = as.formula(sp$sigma)), lapply(sp$sd, as.formula))
    f <- tryCatch(do.call(drmTMB, list(do.call(bf, parts), data = d, family = gaussian(), engine = "tmb")),
                  error = function(e) e)
    if (inherits(f, "error")) { add(ds, m, "error", NA); message(m, " ", s, ": ", conditionMessage(f)); next }
    add(ds, m, "ll", as.numeric(logLik(f)))
    add(ds, m, "conv", as.numeric(isTRUE(is_converged(f))))
    add(ds, m, "pdhess", as.numeric(isTRUE(f$sdr$pdHess)))
    sm <- summary(f)
    cf <- sm$coefficients
    for (i in seq_len(nrow(cf))) {
      nm <- rownames(cf)[i]
      key <- if (grepl("^mu:", nm) || grepl("^sigma:", nm)) nm else paste0("sdf:", nm)
      add(ds, m, key, cf[i, "estimate"], cf[i, "std_error"])
    }
    pt <- sm$parameters
    for (dp in names(f$sdpars)) if (!startsWith(dp, "sd(")) for (nm in names(f$sdpars[[dp]])) {
      est <- f$sdpars[[dp]][[nm]]
      rn <- paste0("sd:", dp, ":", nm)
      se_nat <- if (rn %in% rownames(pt)) pt[rn, "std_error"] else NA_real_
      add(ds, m, paste0("sd:", dp, ":", nm), est, se_nat / est)
    }
    for (cp in names(f$corpars)) for (nm in names(f$corpars[[cp]])) add(ds, m, paste0("cor:", cp, ":", nm), f$corpars[[cp]][[nm]])
  }
}
res <- do.call(rbind, rows)
write.table(res, out, sep = "\t", quote = FALSE, row.names = FALSE)
