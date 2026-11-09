IDS <- commandArgs(TRUE); suppressPackageStartupMessages(library(drmTMB)); source("r_cell.R")
for (fam in c("nb2","gamma","beta","student","tweedie")) {
  d <- read.csv(sprintf("data/data_%s_ordinary.csv", fam))
  fit <- drmTMB(bf(y ~ x1+x2+x3, sigma ~ 1), family = fam_of(fam), data = d)
  p <- fit$opt$par; eta <- as.numeric(cbind(1,d$x1,d$x2,d$x3) %*% p[1:4]); y <- d$y
  ll <- as.numeric(logLik(fit)); s <- p[5]; nu <- if (length(p) > 5) p[6] else NA
  cands <- list()
  if (fam == "nb2") { mu <- exp(eta); cands <- list(size_1_over_sig = sum(dnbinom(y, mu=mu, size=1/exp(s), log=TRUE)), size_sig = sum(dnbinom(y, mu=mu, size=exp(s), log=TRUE)), size_1_over_sig2 = sum(dnbinom(y, mu=mu, size=1/exp(s)^2, log=TRUE))) }
  if (fam == "gamma") { mu <- exp(eta); sg <- exp(s); cands <- list(shape_1_over_sig2 = sum(dgamma(y, shape=1/sg^2, scale=mu*sg^2, log=TRUE)), shape_1_over_sig = sum(dgamma(y, shape=1/sg, scale=mu*sg, log=TRUE)), shape_sig = sum(dgamma(y, shape=sg, scale=mu/sg, log=TRUE))) }
  if (fam == "beta") { mu <- plogis(eta); sg <- exp(s); cands <- list(phi_1_over_sig2 = sum(dbeta(y, mu/sg^2, (1-mu)/sg^2, log=TRUE)), phi_sig = sum(dbeta(y, mu*sg, (1-mu)*sg, log=TRUE)), phi_1_over_sig = sum(dbeta(y, mu/sg, (1-mu)/sg, log=TRUE))) }
  if (fam == "student") { sg <- exp(s); cands <- list(nu_exp = sum(dt((y-eta)/sg, df=exp(nu), log=TRUE) - log(sg)), nu_1p_exp = sum(dt((y-eta)/sg, df=1+exp(nu), log=TRUE) - log(sg)), nu_raw = sum(dt((y-eta)/sg, df=nu, log=TRUE) - log(sg))) }
  if (fam == "tweedie") { mu <- exp(eta); sg <- exp(s)
    for (pw in c(exp(nu), 1+plogis(nu), 1+exp(nu)/(1+exp(nu)), 1.01+0.98*plogis(nu))) cands[[sprintf("p=%.4f phi=sig", pw)]] <- sum(log(tweedie::dtweedie(y, power=pw, mu=mu, phi=sg)))
    for (pw in c(1+plogis(nu))) cands[[sprintf("p=%.4f phi=sig^2", pw)]] <- sum(log(tweedie::dtweedie(y, power=pw, mu=mu, phi=sg^2))) }
  cat(fam, "R ll =", ll, "\n"); print(unlist(cands))
}
