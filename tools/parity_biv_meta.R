# parity_biv_meta.R — bivariate meta-analysis with known sampling covariance:
# native drmTMB `meta_V(V = meta_vcov_bivariate(...))` vs DRModels.jl's
# `drm(...; V = meta_vcov_bivariate(...))`, on identical data.
#
# This is the comparator for the A8 slice (the engine path that unblocked the
# `meta_vcov_bivariate` port, previously refused as "output with no consumer").
# Compared per cell: mu1/mu2 coefficients, the marginal logLik, and the fitted
# heterogeneity components (sigma1, sigma2, rho12) — the quantities the model
# exists to separate from the known sampling covariance.
#
# Data cross the R/Julia boundary BY FILE so both engines fit byte-identical
# inputs (JuliaCall in-process precompile clashes with the lane's environment).
#
#   DRM_JL_PATH=/path/to/DRModels.jl Rscript tools/parity_biv_meta.R

suppressMessages(library(drmTMB))

tol <- 1e-4
out_path <- "docs/dev-log/evidence/parity-biv-meta.tsv"
jl <- Sys.getenv("DRM_JL_PATH", unset = ".")
tmp <- file.path(tempdir(), "biv-meta-parity")
dir.create(tmp, showWarnings = FALSE, recursive = TRUE)

make_cell <- function(seed, n, residual_rho, sampling_cor) {
  set.seed(seed)
  x <- rnorm(n)
  mu1 <- 0.2 + 0.5 * x
  mu2 <- -0.1 - 0.35 * x
  sigma1 <- 0.45; sigma2 <- 0.55
  v1 <- runif(n, 0.01, 0.04)
  v2 <- runif(n, 0.01, 0.05)
  y1 <- numeric(n); y2 <- numeric(n)
  for (i in seq_len(n)) {
    c12 <- sampling_cor * sqrt(v1[i] * v2[i]) + residual_rho * sigma1 * sigma2
    S <- matrix(c(v1[i] + sigma1^2, c12, c12, v2[i] + sigma2^2), 2)
    z <- as.vector(t(chol(S)) %*% rnorm(2))
    y1[i] <- mu1[i] + z[1]; y2[i] <- mu2[i] + z[2]
  }
  list(data = data.frame(x = x, y1 = y1, y2 = y2, v1 = v1, v2 = v2),
       sampling_cor = sampling_cor)
}

cells <- list(
  list(id = "biv_meta_indep",   seed = 601L, n = 120, rho = -0.35, scor = 0.0),
  list(id = "biv_meta_poscor",  seed = 602L, n = 120, rho = -0.35, scor = 0.6),
  list(id = "biv_meta_negcor",  seed = 603L, n = 150, rho = 0.25,  scor = -0.4)
)

rows <- list()
for (cell in cells) {
  cl <- make_cell(cell$seed, cell$n, cell$rho, cell$scor)
  d <- cl$data
  V <- meta_vcov_bivariate(v1 = d$v1, v2 = d$v2, cor12 = cl$sampling_cor)

  res <- list(cell_id = cell$id, status = NA_character_,
              max_abs_coef_diff = NA_real_, loglik_diff = NA_real_,
              het_diff = NA_real_, tolerance = tol, note = "")

  rv <- tryCatch({
    fit <- drmTMB(
      bf(mu1 = y1 ~ x + meta_V(V = V), mu2 = y2 ~ x,
         sigma1 = ~1, sigma2 = ~1, rho12 = ~1),
      family = c(gaussian(), gaussian()), data = d)
    s1 <- as.numeric(predict(fit, dpar = "sigma1"))[1]
    s2 <- as.numeric(predict(fit, dpar = "sigma2"))[1]
    rr <- as.numeric(predict(fit, dpar = "rho12"))[1]
    c(as.numeric(fit$coefficients$mu1), as.numeric(fit$coefficients$mu2),
      as.numeric(fit$logLik), s1, s2, rr)
  }, error = function(e) { res$note <<- paste("native:", conditionMessage(e)); NULL })

  # hand the identical data to DRModels.jl by file
  write.csv(d, file.path(tmp, paste0(cell$id, ".csv")), row.names = FALSE)
  writeLines(sprintf("%.17g", cl$sampling_cor), file.path(tmp, paste0(cell$id, "_scor.txt")))

  jv <- tryCatch({
    script <- sprintf('
using DRModels, DelimitedFiles
raw, hdr = readdlm("%s", \',\', header=true)
x = Float64.(raw[:,1]); y1 = Float64.(raw[:,2]); y2 = Float64.(raw[:,3])
v1 = Float64.(raw[:,4]); v2 = Float64.(raw[:,5])
scor = parse(Float64, readline("%s"))
V = meta_vcov_bivariate(v1, v2; cor12 = scor)
f = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
           sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
           rho12 = @formula(rho12 ~ 1)),
        Gaussian(); data = (; x, y1, y2), V = V)
out = vcat(coef(f, :mu1), coef(f, :mu2), loglik(f),
           f.scales[:sigma1][1], f.scales[:sigma2][1], f.scales[:rho12][1])
println(join([string(v) for v in out], ","))
', file.path(tmp, paste0(cell$id, ".csv")), file.path(tmp, paste0(cell$id, "_scor.txt")))
    sf <- file.path(tmp, paste0(cell$id, ".jl"))
    writeLines(script, sf)
    out <- system2("julia", c(shQuote(paste0("--project=", jl)), "--startup-file=no", shQuote(sf)),
                   stdout = TRUE, stderr = FALSE)
    as.numeric(strsplit(tail(out, 1), ",")[[1]])
  }, error = function(e) { res$note <<- paste(res$note, "julia:", conditionMessage(e)); NULL })

  if (!is.null(rv) && !is.null(jv) && length(rv) == length(jv) &&
      all(is.finite(rv)) && all(is.finite(jv))) {
    res$max_abs_coef_diff <- max(abs(rv[1:4] - jv[1:4]))
    res$loglik_diff <- abs(rv[5] - jv[5])
    res$het_diff <- max(abs(rv[6:8] - jv[6:8]))
    res$status <- if (max(res$max_abs_coef_diff, res$loglik_diff, res$het_diff) < tol)
      "PARITY_PASS" else "PARITY_FAIL"
  } else {
    res$status <- if (is.null(rv)) "NATIVE_FAILED" else "JULIA_FAILED"
  }
  rows[[length(rows) + 1L]] <- as.data.frame(res, stringsAsFactors = FALSE)
  cat(sprintf("%-16s %-13s coef|d|=%-10.3e ll|d|=%-10.3e het|d|=%-10.3e\n",
              res$cell_id, res$status, res$max_abs_coef_diff, res$loglik_diff, res$het_diff))
  if (nzchar(res$note)) cat("    note: ", substr(res$note, 1, 140), "\n", sep = "")
}

tab <- do.call(rbind, rows)
dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write.table(tab, out_path, sep = "\t", row.names = FALSE, quote = FALSE)
cat("\nwrote ", out_path, "\n", sep = "")
cat("OVERALL: ", if (all(tab$status == "PARITY_PASS")) "ALL CELLS PASS" else "SOME CELLS DID NOT PASS",
    "\n", sep = "")
