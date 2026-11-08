# How the committed q4*/r_tight_ll.txt values were made: refit each q=4 dataset with
# r_gen_fit.R, then re-optimise from the final estimate with nlminb (rel.tol = x.tol = xf.tol = 1e-15,
# six restarts) and BFGS (reltol 1e-15); the loglik stops moving (max change 3e-7), i.e. drmTMB's default
# fit is already at its optimum. Values stored: the nlminb loglik.
suppressMessages(library(drmTMB))
O <- Sys.getenv("OUT_DIR", "out")
for (tag in c("q4a","q4b","q4c")) {
  d <- read.csv(file.path(O,tag,"data.csv")); cc <- read.csv(file.path(O,tag,"coords.csv")); coords <- cc[,c("x","y")]; rownames(coords) <- cc$site
  fm <- bf(mu1 = y1 ~ x1 + spatial(1|p|site, coords = coords), mu2 = y2 ~ x1 + spatial(1|p|site, coords = coords),
           sigma1 = ~spatial(1|p|site, coords = coords), sigma2 = ~spatial(1|p|site, coords = coords), rho12 = ~1)
  fit <- suppressWarnings(drmTMB(fm, family = biv_gaussian(), data = d))
  ll0 <- as.numeric(logLik(fit)); obj <- fit$obj; par <- fit$opt$par
  g0 <- max(abs(obj$gr(par)))
  res <- list(); cur <- par
  for (k in 1:6) { o <- nlminb(cur, obj$fn, obj$gr, control = list(rel.tol = 1e-15, x.tol = 1e-15, xf.tol=1e-15, iter.max = 5000, eval.max = 10000)); cur <- o$par
    res[[k]] <- c(k, -o$objective, max(abs(obj$gr(cur)))) }
  o2 <- optim(cur, obj$fn, obj$gr, method = "BFGS", control = list(maxit = 5000, reltol = 1e-15)); cur2 <- o2$par
  cat(sprintf("%s default ll=%.8f maxgrad=%.2e | nlminb restarts: %s | BFGS ll=%.8f maxgrad=%.2e\n", tag, ll0, g0,
      paste(sprintf("%.8f(g=%.1e)", sapply(res,`[`,2), sapply(res,`[`,3)), collapse=" "), -o2$value, max(abs(obj$gr(cur2)))))
  write.csv(data.frame(par = cur2), file.path(O, tag, "r_tight_par.csv"), row.names = FALSE)
}
