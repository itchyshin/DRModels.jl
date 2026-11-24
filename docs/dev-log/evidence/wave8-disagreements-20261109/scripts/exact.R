# Exact (high-node, non-adaptive Gauss-Hermite) marginal log-likelihood of 1-D random-effect cells at a given theta.
args <- commandArgs(TRUE); id <- args[1]; n <- args[2]; N <- as.integer(args[3])
suppressPackageStartupMessages(library(statmod))
f <- strsplit(id, "|", fixed = TRUE)[[1]]; fam <- f[1]; kind <- f[2]; pat <- sub("sd-", "", f[5])
d <- read.csv(sprintf("data/data_%s_ordinary.csv", fam)); d$g <- factor(d$g, levels = unique(d$g))
X <- cbind(1, d$x1, d$x2, d$x3); y <- d$y
logf <- function(eta, th) {
  s <- th[5]; nu <- th[6]
  switch(fam,
    poisson = dpois(y, exp(eta), log = TRUE),
    binomial = dbinom(y, 1, plogis(eta), log = TRUE),
    nb2 = dnbinom(y, mu = exp(eta), size = exp(-2 * th[5]), log = TRUE),
    gamma = dgamma(y, shape = exp(-2 * th[5]), scale = exp(eta) * exp(2 * th[5]), log = TRUE),
    beta = { phi <- exp(-2 * th[5]); mu <- plogis(eta); dbeta(y, mu * phi, (1 - mu) * phi, log = TRUE) },
    student = dt((y - eta) / exp(th[5]), df = 2 + exp(th[6]), log = TRUE) - th[5],
    tweedie = log(tweedie::dtweedie(y, power = 1 + plogis(th[6]), mu = exp(eta), phi = exp(2 * th[5]))))
}
exact_ll <- function(th, N) {
  nsig <- if (fam %in% c("poisson", "binomial")) 0 else if (fam %in% c("student", "tweedie")) 2 else 1
  th2 <- c(th[1:4], if (nsig >= 1) th[5] else NA, if (nsig == 2) th[6] else NA)
  sd_par <- th[(4 + nsig + 1):length(th)]
  eta0 <- as.numeric(X %*% th[1:4])
  gq <- gauss.quad(N, "hermite"); tt <- gq$nodes; lw <- log(gq$weights)
  tot <- 0
  for (g in levels(d$g)) {
    idx <- which(d$g == g)
    zg <- d$z[idx[1]]
    sdg <- if (pat == "none" || pat == "one") exp(sd_par[1]) else exp(sd_par[1] + sd_par[2] * zg)
    # adaptive Gauss-Hermite: centre at the conditional mode, scale by the curvature there
    lj <- function(b) {
      eta <- eta0; eta[idx] <- eta0[idx] + if (kind == "slope") b * d$x1[idx] else b
      sum(logf(eta, th2)[idx]) - 0.5 * (b / sdg)^2
    }
    op <- optimize(function(b) -lj(b), interval = c(-12 * sdg, 12 * sdg), tol = 1e-10)
    bh <- op$minimum; h <- 1e-4 * max(sdg, 1e-3)
    cur <- -(lj(bh + h) - 2 * lj(bh) + lj(bh - h)) / h^2
    sc <- 1 / sqrt(max(cur, 1e-8))
    lv <- numeric(N)
    for (k in seq_len(N)) {
      b <- bh + sqrt(2) * sc * tt[k]
      lv[k] <- lj(b) + tt[k]^2 + log(sqrt(2) * sc) - 0.5 * log(2 * pi) - log(sdg)
    }
    m <- max(lv + lw); tot <- tot + m + log(sum(exp(lv + lw - m)))
  }
  tot
}
rl <- strsplit(readLines(file.path("../out/xr", paste0(n, ".tsv"))), "\t")[[1]]
jl <- strsplit(readLines(file.path("../out/xj", paste0(n, ".tsv"))), "\t")[[1]]
thR <- as.numeric(strsplit(rl[5], ";")[[1]]); thJ <- as.numeric(strsplit(jl[5], ";")[[1]])
res <- c(exact_R = exact_ll(thR, N), exact_J = exact_ll(thJ, N), exact_J_half = exact_ll(thJ, max(20, N %/% 2)))
dir.create("../out/xe", showWarnings = FALSE)
writeLines(paste(id, paste(sprintf("%.8f", res), collapse = "\t"), sep = "\t"), file.path("../out/xe", paste0(n, ".tsv")))
