# gen_xfam_external.R — generate the EXTERNAL reference for DRModels.jl's cross-family
# latent correlation, against the `gllvm` package (an independent codebase).
#
# Why gllvm: DRModels's `fit_mixed_family` reports a shared-latent correlation
#     rho = lambda1 lambda2 / sqrt((lambda1^2 + v1) (lambda2^2 + v2)),
# which is the SAME estimand as the trait-trait residual correlation of a
# 2-response, 1-factor GLLVM (`gllvm(..., num.lv = 1)`), reported by
# gllvm::getResidualCor. The standardisation v_k = link_residual(family_k, .)
# (Nakagawa & Schielzeth 2010) is the one gllvm itself uses on its diagonal
# (verified: getResidualCov adds phi^2 for gaussian, pi^2/3 for binomial-logit).
#
# CAVEAT (documented limitation, drives the test design): stock `gllvm` 2.0.5
# fits ONE family for the whole response matrix — it does NOT accept a per-column
# family vector (`gllvm(family = c("gaussian","poisson"))` errors). So gllvm can
# give an external reference for the SAME-family case (Gaussian x Gaussian: an
# identical-estimand, fully-independent-engine cross-check) but cannot fit a
# literal mixed Gaussian x Poisson. The genuinely cross-family number is therefore
# validated in Julia against a large-N Monte-Carlo of the true latent model.
#
# Output (generated NUMERIC outputs only — gllvm is GPL, its fitted numbers are
# data, not source; mirrors the drmTMB parity-fixture contract, AGENTS.md s.3):
#   fixtures/xfam-external-gllvm/data.csv            simulated y1,y2,x (DRModels refits this)
#   fixtures/xfam-external-gllvm/expected.toml       gllvm rho + loadings + phi + logLik
#   fixtures/xfam-external-gllvm/expected.meta.toml  provenance (gllvm version, seed)
#
# License: this script is fresh MIT code. It only *calls* gllvm; it vendors no
# gllvm/drmTMB source. Run: Rscript test/parity/gen_xfam_external.R

suppressMessages(library(gllvm))

set.seed(20260610)
# Resolve the output directory robustly under Rscript (--file=...) or interactive.
args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
script_dir <- if (length(file_arg)) dirname(normalizePath(file_arg)) else "test/parity"
outdir <- file.path(script_dir, "fixtures", "xfam-external-gllvm")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ---- True Gaussian x Gaussian shared-latent DGP -----------------------------
# y_k = X beta_k + lambda_k u + sigma_k eps,  u, eps ~ N(0,1).
n   <- 2000
b1  <- c(0.40,  0.70)        # mu1 intercept, slope
b2  <- c(-0.20, 0.50)        # mu2 intercept, slope
lam1 <- 0.90; lam2 <- 0.70   # loadings
s1   <- 0.50; s2   <- 0.60   # residual SDs

x  <- rnorm(n)
u  <- rnorm(n)
y1 <- b1[1] + b1[2] * x + lam1 * u + s1 * rnorm(n)
y2 <- b2[1] + b2[2] * x + lam2 * u + s2 * rnorm(n)

rho_true <- (lam1 * lam2) / sqrt((lam1^2 + s1^2) * (lam2^2 + s2^2))

# ---- Fit the 2-response 1-factor GLLVM (gaussian) ---------------------------
Y <- cbind(y1 = y1, y2 = y2)
Xdf <- data.frame(x = x)
m <- gllvm(y = Y, X = Xdf, formula = ~ x, family = "gaussian", num.lv = 1,
           sd.errors = FALSE, seed = 1, control = list(trace = FALSE))

# getResidualCor IS the latent-scale trait correlation (gllvm's own pipeline).
rc <- getResidualCor(m)
rho_gllvm <- rc[1, 2]

# Effective per-trait loadings are theta scaled by sigma.lv (gllvm fixes the
# first loading to 1 and carries the scale in sigma.lv). Stored for traceability.
eff_load <- as.numeric(m$params$theta[, 1] * m$params$sigma.lv[1])
phi      <- as.numeric(m$params$phi)          # gaussian residual SDs per trait
ll_gllvm <- as.numeric(logLik(m))

# Sanity: reconstruct getResidualCor from the loadings + phi via the SAME formula
# DRModels uses, to confirm we are reading gllvm's estimand correctly (not a coincidence).
rho_manual <- (eff_load[1] * eff_load[2]) /
  sqrt((eff_load[1]^2 + phi[1]^2) * (eff_load[2]^2 + phi[2]^2))
stopifnot(abs(rho_manual - rho_gllvm) < 1e-4)

# ---- Write data.csv (DRModels refits the IDENTICAL data) -------------------------
write.csv(data.frame(y1 = y1, y2 = y2, x = x),
          file = file.path(outdir, "data.csv"), row.names = FALSE)

# ---- Write expected.toml ----------------------------------------------------
fmtg <- function(v) formatC(v, format = "e", digits = 16)
exp_lines <- c(
  "[fit]",
  'family = "gllvm_gaussian_gaussian_num.lv1"',
  'estimand = "shared-latent trait correlation (getResidualCor)"',
  sprintf("rho_gllvm = %s",  fmtg(rho_gllvm)),
  sprintf("rho_true  = %s",  fmtg(rho_true)),
  sprintf("loglik    = %s",  fmtg(ll_gllvm)),
  sprintf("n = %d", n),
  "",
  "[loadings]",
  sprintf("eff1 = %s", fmtg(eff_load[1])),
  sprintf("eff2 = %s", fmtg(eff_load[2])),
  sprintf("phi1 = %s", fmtg(phi[1])),
  sprintf("phi2 = %s", fmtg(phi[2])),
  "",
  "[tol]",
  "# cross-package: DRModels (GHQ) vs gllvm (VA) fit the SAME data; the estimators",
  "# agree far tighter than sampling error. 0.02 is a conservative engine-vs-",
  "# engine band (on this data the achieved |DRModels-gllvm| was 1.2e-4).",
  "atol_rho_xpackage = 2e-2"
)
writeLines(exp_lines, file.path(outdir, "expected.toml"))

# ---- Write expected.meta.toml (provenance) ----------------------------------
meta_lines <- c(
  sprintf('gllvm_version = "%s"', as.character(packageVersion("gllvm"))),
  sprintf('generated_on = "%s"', Sys.Date()),
  'r_call = "gllvm(y = cbind(y1,y2), X = data.frame(x), formula = ~x, family = \\"gaussian\\", num.lv = 1)"',
  "seed = 20260610",
  'estimand = "getResidualCor[1,2] == lambda1 lambda2 / sqrt((lambda1^2+v1)(lambda2^2+v2))"',
  'note = "External reference for DRModels cross-family rho. gllvm is GPL; only its fitted NUMBERS are stored (data, not source). gllvm fits ONE family per matrix, so this validates the SAME-family (Gaussian x Gaussian) identical-estimand case; the genuinely mixed Gaussian x Poisson rho is validated against a large-N Monte-Carlo in Julia."'
)
writeLines(meta_lines, file.path(outdir, "expected.meta.toml"))

cat(sprintf("WROTE fixture: rho_true=%.6f rho_gllvm=%.6f rho_manual=%.6f logLik=%.3f\n",
            rho_true, rho_gllvm, rho_manual, ll_gllvm))
cat(sprintf("eff loadings = (%.5f, %.5f)  phi = (%.5f, %.5f)\n",
            eff_load[1], eff_load[2], phi[1], phi[2]))
