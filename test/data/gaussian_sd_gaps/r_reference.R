# R reference for the Gaussian SD-gap closure: drmTMB (engine = "tmb") fits of 15 models on the three
# committed datasets, written as a long table  dataset, model, key, estimate, se.
# Keys: ll, conv, pdhess; mu:<coef>, sigma:<coef>, sdf:<target>:<coef> (SD-formula coefficients, with se);
#       sd:<term> (natural-scale SD of a coefficient with no SD formula, se on the log scale);
#       cor:<name> (correlation, no se).
# Run (R with drmTMB 0.8.1):  Rscript r_reference.R <dir with gsd_s*.csv> <out.tsv>
args <- commandArgs(TRUE)
suppressMessages(library(drmTMB))
dir <- args[1]; out <- args[2]
fx <- "y ~ x1 + x2 + x3"
models <- list(
  ord_q2_lab   = list(re = "(1 + x1 | p | g)", sd = character()),
  ord_q3       = list(re = "(1 + x1 + x2 | g)", sd = character()),
  ord_q4_lab   = list(re = "(1 + x1 + x2 + x3 | p | g)", sd = character()),
  crs_q2       = list(re = "(1 + x1 | g) + (1 | g2)", sd = character()),
  crs_q2_lab   = list(re = "(1 + x1 | p | g) + (1 | g2)", sd = character()),
  crs_q3       = list(re = "(1 + x1 + x2 | g) + (1 | g2)", sd = character()),
  crs_q4_lab   = list(re = "(1 + x1 + x2 + x3 | p | g) + (1 | g2)", sd = character()),
  crs_q2_first = list(re = "(1 + x1 | g) + (1 | g2)", sd = c('sd(g, coef = "(Intercept)") ~ z', "sd(g2) ~ z2")),
  crs_q2_last  = list(re = "(1 + x1 | g) + (1 | g2)", sd = c('sd(g, coef = "x1") ~ z')),
  crs_q2_z     = list(re = "(1 + x1 | g) + (1 | g2)", sd = c('sd(g, coef = "(Intercept)") ~ z', 'sd(g, coef = "x1") ~ z', "sd(g2) ~ z2")),
  ind_q2_first = list(re = "(1 | g) + (0 + x1 | g)", sd = c('sd(g, coef = "(Intercept)") ~ z')),
  ind_q2_last  = list(re = "(1 | g) + (0 + x1 | g)", sd = c('sd(g, coef = "x1") ~ z')),
  ind_q2_z     = list(re = "(1 | g) + (0 + x1 | g)", sd = c('sd(g, coef = "(Intercept)") ~ z', 'sd(g, coef = "x1") ~ z')),
  ind_q2_one   = list(re = "(1 | g) + (0 + x1 | g)", sd = c('sd(g, coef = "(Intercept)") ~ 1', 'sd(g, coef = "x1") ~ 1')),
  ind_q3_z     = list(re = "(1 | g) + (0 + x1 | g) + (0 + x2 | g)",
                      sd = c('sd(g, coef = "(Intercept)") ~ z', 'sd(g, coef = "x1") ~ z', 'sd(g, coef = "x2") ~ z'))
)
rows <- list()
add <- function(ds, m, key, est, se = NA_real_) {
  rows[[length(rows) + 1L]] <<- data.frame(dataset = ds, model = m, key = key, estimate = est, se = se)
}
for (s in 1:3) {
  ds <- sprintf("gsd_s%d", s)
  d <- read.csv(file.path(dir, paste0(ds, ".csv")))
  for (m in names(models)) {
    sp <- models[[m]]
    mu <- as.formula(paste(fx, "+", sp$re))
    parts <- c(list(mu, sigma = ~ 1), lapply(sp$sd, as.formula))
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
      key <- if (grepl("^mu:", nm)) nm else if (grepl("^sigma:", nm)) nm else paste0("sdf:", nm)
      add(ds, m, key, cf[i, "estimate"], cf[i, "std_error"])
    }
    pt <- sm$parameters
    for (nm in names(f$sdpars$mu)) {
      se_nat <- pt[paste0("sd:mu:", nm), "std_error"]
      est <- f$sdpars$mu[[nm]]
      add(ds, m, paste0("sd:", nm), est, se_nat / est)
    }
    if (!is.null(f$corpars$re_cov)) for (nm in names(f$corpars$re_cov)) add(ds, m, paste0("cor:", nm), f$corpars$re_cov[[nm]])
  }
}
res <- do.call(rbind, rows)
write.table(res, out, sep = "\t", quote = FALSE, row.names = FALSE)
