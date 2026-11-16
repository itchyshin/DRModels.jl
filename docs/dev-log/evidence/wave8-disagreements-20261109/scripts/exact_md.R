# Exact (adaptive tensor Gauss-Hermite) marginal log-likelihood of an ordinary / indep random-effect block at R's estimate.
IDS <- commandArgs(TRUE); suppressPackageStartupMessages({library(drmTMB); library(statmod)}); source("r_cell.R")
for (id in IDS) {
  f <- strsplit(id, "|", fixed = TRUE)[[1]]; fam <- f[1]; kind <- f[2]; q <- as.integer(sub("q", "", f[4]))
  fit <- fit_cell(id); llR <- as.numeric(logLik(fit))
  d <- read.csv(sprintf("data/data_%s_ordinary.csv", fam)); d$g <- factor(d$g, levels = unique(d$g))
  cn <- c("(Intercept)", "x1", "x2", "x3")[1:q]; Zfull <- cbind(1, d$x1, d$x2, d$x3)[, 1:q, drop = FALSE]
  X <- cbind(1, d$x1, d$x2, d$x3); p <- fit$opt$par; th <- unname(p)
  eta0 <- as.numeric(X %*% th[1:4]); y <- d$y
  nsig <- if (fam %in% c("poisson", "binomial")) 0 else 1
  s <- if (nsig) th[5] else NA
  logf <- function(eta) switch(fam,
    poisson = dpois(y, exp(eta), log = TRUE), binomial = dbinom(y, 1, plogis(eta), log = TRUE),
    nb2 = dnbinom(y, mu = exp(eta), size = exp(-2 * s), log = TRUE))
  groups <- levels(d$g)
  sdm <- sapply(cn, function(cc) { v <- fit$sdpars[[sprintf('sd(g, coef = "%s")', cc)]]; if (is.null(v)) rep(NA_real_, length(groups)) else unname(v[paste0(sprintf('sd(g, coef = "%s"):', cc), groups)]) })
  if (is.null(dim(sdm))) sdm <- matrix(sdm, nrow = length(groups))
  lsd <- th[names(p) == "log_sd_mu"]; k <- 0
  for (j in seq_len(q)) if (all(is.na(sdm[, j]))) { k <- k + 1; sdm[, j] <- exp(lsd[k]) }
  C <- diag(q)
  cp <- fit$corpars$mu
  if (kind == "ordinary" && q >= 2 && length(cp)) for (nm in names(cp)) { ab <- strsplit(sub(" \\| g\\)$", "", sub("^cor\\(", "", nm)), ",")[[1]]; i <- match(ab[1], cn); j <- match(ab[2], cn); C[i, j] <- C[j, i] <- cp[[nm]] }
  m <- c(`1` = 60, `2` = 25, `3` = 13, `4` = 9)[as.character(q)]
  gq <- gauss.quad(m, "hermite"); grid <- as.matrix(expand.grid(rep(list(seq_len(m)), q)))
  T <- matrix(gq$nodes[grid], ncol = q); lwk <- rowSums(matrix(log(gq$weights)[grid], ncol = q))
  tot <- 0
  for (gi in seq_along(groups)) {
    idx <- which(d$g == groups[gi]); Zg <- Zfull[idx, , drop = FALSE]
    D <- diag(sdm[gi, ], q); Sig <- D %*% C %*% D; Pm <- solve(Sig); ldS <- as.numeric(determinant(Sig)$modulus)
    lj <- function(b) { eta <- eta0[idx] + as.numeric(Zg %*% b); sum(logf(eta)[idx] * 0 + { e <- eta0; e[idx] <- eta; logf(e)[idx] }) - 0.5 * as.numeric(t(b) %*% Pm %*% b) }
    op <- optim(rep(0, q), function(b) -lj(b), method = "BFGS", control = list(reltol = 1e-14, maxit = 500), hessian = TRUE)
    bh <- op$par; H <- op$hessian; Lc <- t(chol(solve(H)))
    lv <- apply(T, 1, function(t) lj(bh + sqrt(2) * as.numeric(Lc %*% t)))
    la <- lv + rowSums(T^2) + q * 0.5 * log(2) + as.numeric(determinant(Lc)$modulus) - 0.5 * q * log(2 * pi) - 0.5 * ldS + lwk
    mx <- max(la); tot <- tot + mx + log(sum(exp(la - mx)))
  }
  cat(sprintf("EXACT\t%s\t%.10f\t%.10f\n", id, llR, tot))
}
