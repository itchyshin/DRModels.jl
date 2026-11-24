# gen_count_ranef_parity.R -- native drmTMB reference for the Poisson / NB2 `(1 | g)` parity
# test (refs #1073, #1072).
#
#   Rscript --vanilla test/parity/gen_count_ranef_parity.R
#
# One folder per route under test/fixtures/count_parity/<route>/ holding the simulated data
# (data.csv) and ref.toml with the drmTMB numbers:
#   names     DRModels.jl flat coefficient names (`mu_(Intercept)`, `mu_x`, `sigma_(Intercept)`,
#             `resd_g` = log SD of the g intercept), in drmTMB's opt$par order
#   theta     drmTMB opt$par
#   se        sqrt(diag(sdreport cov.fixed)) at opt$par
#   max_grad  max |gradient of the TMB marginal objective at opt$par| (must be < 1e-6)
#   loglik    drmTMB logLik (ML, Laplace)
# License boundary: numbers and simulated data only. No drmTMB source is copied (see
# test/parity/GENERATING.md). The data come from a fixed seed, so the fixtures regenerate
# bit-for-bit given the same drmTMB / R versions (recorded in ref.toml).
#
# The designs have an interior optimum (no variance at a boundary): a boundary SE is not an
# estimand.

suppressPackageStartupMessages(library(drmTMB))

out_root <- file.path("test", "fixtures", "count_parity")
fmt <- function(x) paste0("[", paste(formatC(x, digits = 17, format = "g"), collapse = ", "), "]")
qs  <- function(x) paste0("[", paste(sprintf("\"%s\"", x), collapse = ", "), "]")

write_ref <- function(route, fit, names, data, note) {
  dir <- file.path(out_root, route)
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(data, file.path(dir, "data.csv"), row.names = FALSE)
  par <- fit$opt$par
  gr <- as.numeric(fit$obj$gr(par))
  se <- sqrt(diag(fit$sdr$cov.fixed))
  stopifnot(length(names) == length(par), max(abs(gr)) < 1e-6, all(is.finite(se)))
  writeLines(c(
    sprintf("route = \"%s\"", route),
    sprintf("drmTMB_version = \"%s\"", as.character(utils::packageVersion("drmTMB"))),
    sprintf("R_version = \"%s\"", paste(R.version$major, R.version$minor, sep = ".")),
    sprintf("note = \"%s\"", note),
    sprintf("tmb_names = %s", qs(names(par))),
    sprintf("names = %s", qs(names)),
    sprintf("loglik = %s", formatC(as.numeric(logLik(fit)), digits = 17, format = "g")),
    sprintf("max_grad = %s", formatC(max(abs(gr)), digits = 6, format = "e")),
    sprintf("theta = %s", fmt(as.numeric(par))),
    sprintf("se = %s", fmt(se))
  ), file.path(dir, "ref.toml"))
  cat(sprintf("%-12s n=%d max|grad|=%.2e\n  tmb: %s\n  par: %s\n  se : %s\n", route, nrow(data),
              max(abs(gr)), paste(names(par), collapse = " "),
              paste(signif(par, 5), collapse = " "), paste(signif(se, 4), collapse = " ")))
}

group_data <- function(seed, G, m, sd_g) {
  set.seed(seed)
  g <- factor(rep(seq_len(G), each = m))
  x <- stats::rnorm(G * m)
  list(d = data.frame(g = paste0("g", g), x = x),
       eta = 0.3 + 0.4 * x + stats::rnorm(G, 0, sd_g)[g], n = G * m)
}

# Poisson, y ~ x + (1 | g)
s <- group_data(20261101, 30, 8, 0.6)
s$d$y <- stats::rpois(s$n, exp(s$eta))
f <- drmTMB(drm_formula(y ~ x + (1 | g)), family = poisson(), data = s$d)
write_ref("poisson_g", f, c("mu_(Intercept)", "mu_x", "resd_g"), s$d,
          "Poisson, mean (1 | g); generated outputs only")

# NB2, y ~ x + (1 | g), sigma ~ 1   (sigma on the log scale in both packages)
s <- group_data(20261102, 30, 10, 0.6)
s$d$y <- stats::rnbinom(s$n, mu = exp(s$eta), size = 4)
f <- drmTMB(bf(y ~ x + (1 | g), sigma ~ 1), family = nbinom2(), data = s$d)
write_ref("nb2_g", f, c("mu_(Intercept)", "mu_x", "sigma_(Intercept)", "resd_g"), s$d,
          "NB2, mean (1 | g), constant sigma; generated outputs only")
