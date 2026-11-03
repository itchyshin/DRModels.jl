# gen_1088_nb2_runoff_reference.R -- native drmTMB reference for the #1088 NB2 run-off sets.
#
#   julia --project=. test/parity/gen_1088_nb2_runoff_data.jl         # writes the data.csv files
#   DRMTMB_COMMIT=<sha> Rscript --vanilla test/parity/gen_1088_nb2_runoff_reference.R
#
# Reads test/fixtures/j3_se/nb2_runoff_*/data.csv and writes ref.toml next to it, in the
# schema of the #1073 (lane J3) SE ledger (`test/j3_se_harness.jl` on the J3e branch), so
# each set is one #1073 comparison row:
#   drmTMB_commit, opt_convergence (fit$opt$convergence), pd_hess (fit$sdr$pdHess),
#   comparable (convergence == 0 && pdHess && finite SEs: the SE comparison is valid),
#   point_comparable (convergence == 0: drmTMB's optimum is a valid reference), theta (opt$par, in
#   DRModels.jl's theta order [beta_mu; log sigma; log SD g; log SD h]), se (sdreport),
#   loglik, max_grad (max |TMB gradient| at opt$par), and log_dispersion (= theta[3], the
#   fitted NB2 log sigma, size = exp(-2 log sigma)).
# License boundary: numbers only. No drmTMB source is copied (test/parity/GENERATING.md).

suppressPackageStartupMessages(library(drmTMB))

out_root <- file.path("test", "fixtures", "j3_se")
routes <- c("nb2_runoff_s7_n1500", "nb2_runoff_s7_n800", "nb2_runoff_s8_n1500",
            "nb2_disp3to4_s1_n5000")
drmtmb_commit <- Sys.getenv("DRMTMB_COMMIT", unset = NA_character_)
num <- function(x) ifelse(is.finite(x), formatC(x, digits = 17, format = "g"), "nan")   # TOML nan
fmt <- function(x) paste0("[", paste(num(x), collapse = ", "), "]")

for (route in routes) {
  dir <- file.path(out_root, route)
  d <- utils::read.csv(file.path(dir, "data.csv"))
  d$g <- factor(d$g); d$h <- factor(d$h)
  fit <- drmTMB(bf(y ~ 1 + x + (1 | g) + (1 | h), sigma ~ 1), family = nbinom2(), data = d)
  par <- fit$opt$par
  gr <- as.numeric(fit$obj$gr(par))
  se <- suppressWarnings(sqrt(diag(fit$sdr$cov.fixed)))
  conv_ok <- identical(as.integer(fit$opt$convergence), 0L)
  pd_ok <- isTRUE(fit$sdr$pdHess)
  comparable <- conv_ok && pd_ok && all(is.finite(se))
  lines <- c(
    sprintf("route = \"%s\"", route),
    "issue = 1088",
    sprintf("drmTMB_version = \"%s\"", as.character(utils::packageVersion("drmTMB"))),
    sprintf("drmTMB_commit = \"%s\"", if (is.na(drmtmb_commit)) "unrecorded" else drmtmb_commit),
    sprintf("R_version = \"%s\"", paste(R.version$major, R.version$minor, sep = ".")),
    sprintf("TMB_version = \"%s\"", as.character(utils::packageVersion("TMB"))),
    sprintf("opt_convergence = %d", as.integer(fit$opt$convergence)),
    sprintf("opt_message = \"%s\"", gsub("\"", "'", as.character(fit$opt$message))),
    sprintf("pd_hess = %s", tolower(as.character(pd_ok))),
    sprintf("comparable = %s", tolower(as.character(comparable))),
    sprintf("point_comparable = %s", tolower(as.character(conv_ok))),
    sprintf("tmb_names = %s", paste0("[", paste(sprintf("\"%s\"", names(par)), collapse = ", "), "]")),
    sprintf("loglik = %s", formatC(as.numeric(logLik(fit)), digits = 17, format = "g")),
    sprintf("max_grad = %s", formatC(max(abs(gr)), digits = 6, format = "e")),
    sprintf("log_dispersion = %s", formatC(as.numeric(par[3]), digits = 17, format = "g")),
    sprintf("theta = %s", fmt(as.numeric(par))),
    sprintf("se = %s", fmt(se))
  )
  writeLines(lines, file.path(dir, "ref.toml"))
  cat(sprintf("%-20s n=%d conv=%d (%s) pdHess=%s max|grad|=%.2e logLik=%.6f\n  par=%s\n  se=%s\n",
              route, nrow(d), as.integer(fit$opt$convergence), fit$opt$message, pd_ok, max(abs(gr)),
              as.numeric(logLik(fit)), paste(signif(par, 6), collapse = " "),
              paste(signif(se, 4), collapse = " ")))
}
