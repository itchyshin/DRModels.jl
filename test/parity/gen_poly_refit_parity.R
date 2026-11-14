# gen_poly_refit_parity.R -- native drmTMB 0.8.1 reference for `poly(x, 2)` on SHIFTED data
# (x + 1000). R recomputes the basis on every fit, so these numbers are what a second fit with a
# reused Julia `bf(...)` object on shifted data must reproduce.
#
#   Rscript --vanilla test/parity/gen_poly_refit_parity.R     (from the repository root)
#
# Reads test/fixtures/poly_parity/data.csv (written by gen_poly_parity.R) and writes
# test/fixtures/poly_parity/ref_shift.toml: opt$par, sdreport SEs, logLik and max |gradient|, plus
# predictions at four new x values. Numbers only.
suppressPackageStartupMessages(library(drmTMB))
stopifnot(as.character(utils::packageVersion("drmTMB")) == "0.8.1")
dir <- file.path("test", "fixtures", "poly_parity")
d <- utils::read.csv(file.path(dir, "data.csv"))
d$x1s <- d$x1 + 1000
nd <- data.frame(x1s = 1000 + c(-1, 0, 0.5, 2))
fmt <- function(x) paste0("[", paste(formatC(x, digits = 17, format = "g"), collapse = ", "), "]")
qs  <- function(x) paste0("[", paste(sprintf("\"%s\"", x), collapse = ", "), "]")
cells <- list(
  gaussian_shift      = list(f = bf(y_g ~ poly(x1s, 2), sigma ~ 1), fam = gaussian()),
  gaussian_reml_shift = list(f = bf(y_g ~ poly(x1s, 2), sigma ~ 1), fam = gaussian(), reml = TRUE),
  poisson_shift       = list(f = bf(y_pois ~ poly(x1s, 2)), fam = stats::poisson(link = "log"))
)
lines <- c(sprintf("drmTMB_version = \"%s\"", as.character(utils::packageVersion("drmTMB"))),
           sprintf("newdata_x1s = %s", fmt(nd$x1s)))
for (nm in names(cells)) {
  reml <- isTRUE(cells[[nm]]$reml)
  fit <- drmTMB(cells[[nm]]$f, family = cells[[nm]]$fam, data = d, REML = reml)
  cf <- coef(fit)
  jn <- unlist(lapply(names(cf), function(p) paste0(p, "_", names(cf[[p]]))), use.names = FALSE)
  gr <- as.numeric(fit$obj$gr(fit$opt$par))
  if (reml) {
    par <- unlist(cf)
    se <- sqrt(diag(as.matrix(vcov(fit))))
  } else {
    par <- fit$opt$par
    se <- sqrt(diag(fit$sdr$cov.fixed))
  }
  stopifnot(length(jn) == length(par), max(abs(gr)) < 1e-6, all(is.finite(se)))
  pr <- as.numeric(predict(fit, newdata = nd, type = "response"))
  cat(sprintf("%-20s max|grad|=%.2e ll=%.10f\n", nm, max(abs(gr)), as.numeric(logLik(fit))))
  lines <- c(lines, "", sprintf("[%s]", nm),
             sprintf("names = %s", qs(jn)),
             sprintf("loglik = %s", formatC(as.numeric(logLik(fit)), digits = 17, format = "g")),
             sprintf("max_grad = %s", formatC(max(abs(gr)), digits = 6, format = "e")),
             sprintf("theta = %s", fmt(as.numeric(par))),
             sprintf("se = %s", fmt(se)),
             sprintf("predict_response = %s", fmt(pr)))
}
writeLines(lines, file.path(dir, "ref_shift.toml"))
cat("written", file.path(dir, "ref_shift.toml"), "\n")
