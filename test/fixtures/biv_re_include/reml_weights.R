# Regenerates reml_weights_data.csv and reml_weights_reference.csv for test/test_biv_re_include_weights.jl.
#
#   Rscript reml_weights.R <this folder>
#
# Bivariate Gaussian with a correlated random intercept per group, REML, rows with one or both responses
# missing kept, and prior weights. drmTMB 0.8.1 refuses REML with kept missing rows, so the references are
# independent of drmTMB and of the Julia code:
#   dataset 1: nlme::lme REML in long format (one row per observed entry), with every data row repeated
#              w_int times (a row of weight 0 is removed). A weight w on a row multiplies its whole log-density,
#              which for integer w is the same as w copies of the row sharing the group effect.
#   dataset 2: the dense restricted likelihood below, with the integer weights w_int.
#   dataset 3: the dense restricted likelihood below, with non-integer weights w_real (no nlme equivalent).
# Dense restricted likelihood (written from the formula, optimised by optim): for group g with observed
# entries r = y - X b, W = blockdiag(w_i * S_i^-1) over its rows (S_i the residual covariance cut to the
# observed coordinates), Z the entry-to-random-effect incidence (n_g x 2) and Sigma the group covariance,
#   log-likelihood contribution of g (given b) =
#       -1/2 sum_i w_i log|S_i| - 1/2 log|Sigma| - 1/2 log|Sigma^-1 + Z'WZ| - 1/2 r'(W - WZ(Sigma^-1 + Z'WZ)^-1 Z'W) r,
#   REML:  l_R = sum_g contributions(b = GLS) - (N_w - p)/2 log(2 pi) - 1/2 log|X' M X|,
# with M the group-wise W - WZ(...)^-1 Z'W and N_w = sum_i w_i * (number of observed entries of row i).
suppressMessages(library(nlme))
args <- commandArgs(TRUE)
d <- read.csv(file.path(args[1], "reml_data1.csv"), na.strings = "NA")
set.seed(2211)
n <- nrow(d)
d$w_int <- sample(0:3, n, replace = TRUE, prob = c(0.12, 0.4, 0.3, 0.18))
d$w_real <- round(runif(n, 0.3, 2.5), 3)
d$w_real[sample(n, 8)] <- 0
write.csv(d, file.path(args[1], "reml_weights_data.csv"), row.names = FALSE, quote = FALSE)

# ---- dense restricted likelihood -------------------------------------------------------------
# Parameters: log sigma1, log sigma2, atanh rho, then Sigma = L L' with L = [e^a 0; b e^c] (a, b, c).
dense_reml <- function(d, w) {
  gs <- split(seq_len(nrow(d)), d$g)
  S_of <- function(par) {
    s1 <- exp(par[1]); s2 <- exp(par[2]); rho <- tanh(par[3])
    matrix(c(s1^2, rho * s1 * s2, rho * s1 * s2, s2^2), 2, 2)
  }
  Sg_of <- function(par) { L <- matrix(c(exp(par[4]), par[5], 0, exp(par[6])), 2, 2, byrow = TRUE); L %*% t(L) }
  pieces <- function(par) {
    S <- S_of(par); Sg <- Sg_of(par)
    Q <- matrix(0, 4, 4); cc <- numeric(4); yMy <- 0; ld <- 0; Nw <- 0
    for (g in gs) {
      Wb <- list(); X <- NULL; y <- NULL; Zr <- NULL
      for (i in g) {
        o <- c(!is.na(d$y1[i]), !is.na(d$y2[i]))
        if (!any(o)) next
        So <- S[o, o, drop = FALSE]
        Wb[[length(Wb) + 1]] <- w[i] * solve(So)
        ld <- ld + w[i] * as.numeric(determinant(So)$modulus)
        Nw <- Nw + w[i] * sum(o)
        if (o[1]) { X <- rbind(X, c(1, d$x1[i], 0, 0)); y <- c(y, d$y1[i]); Zr <- rbind(Zr, c(1, 0)) }
        if (o[2]) { X <- rbind(X, c(0, 0, 1, d$x2[i])); y <- c(y, d$y2[i]); Zr <- rbind(Zr, c(0, 1)) }
      }
      if (is.null(y)) next
      W <- as.matrix(Matrix::bdiag(Wb))
      Om <- solve(Sg) + t(Zr) %*% W %*% Zr
      M <- W - W %*% Zr %*% solve(Om) %*% t(Zr) %*% W
      ld <- ld + as.numeric(determinant(Sg)$modulus) + as.numeric(determinant(Om)$modulus)
      Q <- Q + t(X) %*% M %*% X; cc <- cc + as.numeric(t(X) %*% M %*% y); yMy <- yMy + as.numeric(t(y) %*% M %*% y)
    }
    list(Q = Q, cc = cc, yMy = yMy, ld = ld, Nw = Nw)
  }
  negll <- function(par) tryCatch({
    if (any(abs(par[1:3]) > 8) || any(abs(par[4:6]) > 8)) return(1e10)
    pc <- pieces(par)
    b <- solve(pc$Q, pc$cc)
    q <- pc$yMy - sum(b * pc$cc)                    # r'Mr at the GLS b
    ld <- pc$ld + as.numeric(determinant(pc$Q)$modulus)
    0.5 * (ld + q) + 0.5 * (pc$Nw - 4) * log(2 * pi)
  }, error = function(e) 1e10)
  par0 <- c(0, 0, 0, -0.5, 0, -0.5)
  o <- optim(par0, negll, method = "Nelder-Mead", control = list(maxit = 4000, reltol = 1e-12))
  o <- optim(o$par, negll, method = "BFGS", control = list(maxit = 2000, reltol = 1e-15))
  o <- optim(o$par, negll, method = "Nelder-Mead", control = list(maxit = 4000, reltol = 1e-15))
  o <- optim(o$par, negll, method = "BFGS", control = list(maxit = 2000, reltol = 1e-15))
  pc <- pieces(o$par); b <- solve(pc$Q, pc$cc)
  Sg <- Sg_of(o$par); S <- S_of(o$par)
  c(ll = -o$value, N = pc$Nw, r1 = b[1], x1 = b[2], r2 = b[3], x2 = b[4],
    Sigma11 = Sg[1, 1], Sigma12 = Sg[1, 2], Sigma22 = Sg[2, 2],
    sigma1 = sqrt(S[1, 1]), sigma2 = sqrt(S[2, 2]), rho = S[1, 2] / sqrt(S[1, 1] * S[2, 2]))
}

# ---- nlme on the repeated rows ----------------------------------------------------------------
nlme_reml <- function(d, w) {
  rows <- rep(seq_len(nrow(d)), times = w)               # a row of weight 0 disappears
  e <- d[rows, ]
  e$i <- seq_len(nrow(e))
  L <- rbind(data.frame(g = e$g, i = e$i, k = 1L, y = e$y1, x1 = e$x1, x2 = 0),
             data.frame(g = e$g, i = e$i, k = 2L, y = e$y2, x1 = 0, x2 = e$x2))
  L <- L[!is.na(L$y), ]
  L <- L[order(L$g, L$i, L$k), ]
  L$r1 <- as.numeric(L$k == 1); L$r2 <- as.numeric(L$k == 2)
  L$g <- factor(L$g); L$resp <- factor(L$k)
  m <- lme(y ~ 0 + r1 + x1 + r2 + x2, data = L, random = list(g = pdSymm(~ 0 + r1 + r2)),
           correlation = corSymm(form = ~ k | g/i), weights = varIdent(form = ~ 1 | resp),
           method = "REML", control = lmeControl(maxIter = 500, msMaxIter = 500, opt = "optim", niterEM = 100))
  vc <- getVarCov(m)
  ratio <- coef(m$modelStruct$varStruct, unconstrained = FALSE)
  rho <- coef(m$modelStruct$corStruct, unconstrained = FALSE)
  c(ll = as.numeric(logLik(m)), N = nrow(L), fixef(m),
    Sigma11 = vc[1, 1], Sigma12 = vc[1, 2], Sigma22 = vc[2, 2],
    sigma1 = m$sigma, sigma2 = m$sigma * ratio[[1]], rho = rho[[1]])
}

res <- list(nlme_reml(d, d$w_int), dense_reml(d, d$w_int), dense_reml(d, d$w_real))
out <- do.call(rbind, lapply(1:3, function(k) data.frame(dataset = k, name = names(res[[k]]), value = sprintf("%.10f", res[[k]]))))
write.csv(out, file.path(args[1], "reml_weights_reference.csv"), row.names = FALSE, quote = FALSE)
