# Regenerates reml_nlme_reference.csv for test/test_biv_re_include_nlme.jl.
#
#   Rscript reml_nlme.R <this folder> reml_nlme_reference.csv
#
# REML with partly observed rows has no drmTMB reference (drmTMB 0.8.1 refuses it), so the reference is
# an independent REML fit of the same model by nlme::lme (R 4.6.1, nlme shipped with R) in long format,
# one row per OBSERVED entry: random effects (r1, r2) with an unstructured 2 x 2 covariance per group,
# residual standard deviation per response (varIdent) and a residual correlation between the two entries of
# one data row (corSymm over group/row). The data sets reml_data1.csv to reml_data3.csv are generated
# data (reml_data3 has two 0/1 covariates, xf2 and xf3, in mu1).
suppressMessages(library(nlme))
args <- commandArgs(TRUE)
out <- NULL
for (k in 1:3) {
  d <- read.csv(file.path(args[1], sprintf("reml_data%d.csv", k)), na.strings = "NA")
  d$i <- seq_len(nrow(d))
  L <- rbind(data.frame(g = d$g, i = d$i, k = 1L, y = d$y1, x1 = d$x1, x2 = 0, xf2 = d$xf2, xf3 = d$xf3),
             data.frame(g = d$g, i = d$i, k = 2L, y = d$y2, x1 = 0, x2 = d$x2, xf2 = 0, xf3 = 0))
  L <- L[!is.na(L$y), ]
  L <- L[order(L$g, L$i, L$k), ]
  L$r1 <- as.numeric(L$k == 1); L$r2 <- as.numeric(L$k == 2)
  L$g <- factor(L$g); L$resp <- factor(L$k)
  fx <- if (k == 3) y ~ 0 + r1 + x1 + xf2 + xf3 + r2 + x2 else y ~ 0 + r1 + x1 + r2 + x2
  m <- lme(fx, data = L, random = list(g = pdSymm(~ 0 + r1 + r2)),
           correlation = corSymm(form = ~ k | g/i), weights = varIdent(form = ~ 1 | resp),
           method = "REML", control = lmeControl(maxIter = 500, msMaxIter = 500, opt = "optim", niterEM = 100))
  vc <- getVarCov(m)
  ratio <- coef(m$modelStruct$varStruct, unconstrained = FALSE)
  rho <- coef(m$modelStruct$corStruct, unconstrained = FALSE)
  v <- c(ll = as.numeric(logLik(m)), N = nrow(L), fixef(m),
         Sigma11 = vc[1, 1], Sigma12 = vc[1, 2], Sigma22 = vc[2, 2],
         sigma1 = m$sigma, sigma2 = m$sigma * ratio[[1]], rho = rho[[1]])
  out <- rbind(out, data.frame(dataset = k, name = names(v), value = sprintf("%.10f", v)))
}
write.csv(out, args[2], row.names = FALSE, quote = FALSE)
