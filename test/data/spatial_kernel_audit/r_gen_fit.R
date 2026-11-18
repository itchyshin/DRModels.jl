suppressMessages(library(drmTMB))
O <- Sys.getenv("OUT_DIR", "out")
mk_coords <- function(layout, G) {
  if (layout == "spiral") { th <- seq(0, 1.5*pi, length.out = G); data.frame(x = cos(th) + seq_len(G)/(3*G), y = sin(th)) }
  else if (layout == "uniform") data.frame(x = runif(G), y = runif(G))
  else { k <- rep(1:2, length.out = G); data.frame(x = c(0, 4)[k] + 0.3*rnorm(G), y = c(0, 1)[k] + 0.3*rnorm(G)) }
}
gen <- function(tag, route, layout, G, nrep, seed) {
  set.seed(seed); coords <- mk_coords(layout, G); rownames(coords) <- paste0("s", 1:G)
  D <- as.matrix(dist(coords)); L <- t(chol(exp(-D/(0.7*mean(D[D > 0]))) + 1e-8*diag(G)))
  site <- rep(rownames(coords), each = nrep); n <- length(site); ix <- match(site, rownames(coords))
  Z <- sapply(1:4, function(k) drop(L %*% rnorm(G)))
  x1 <- rnorm(n); x2 <- rnorm(n); x2 <- x1;  # Julia biv structured routes need one shared design
   e1 <- rnorm(n); e2 <- -0.2*e1 + sqrt(0.96)*rnorm(n)
  if (route == "q2") {
    a1 <- 0.7*Z[,1]; a2 <- 0.6*(0.4*Z[,1] + sqrt(0.84)*Z[,2])
    d <- data.frame(y1 = .3 + .5*x1 + a1[ix] + .3*e1, y2 = -.2 - .25*x2 + a2[ix] + .35*e2, x1 = x1, x2 = x2, site = site)
  } else {
    d <- data.frame(y1 = .3 + .5*x1 + .8*Z[ix,1] + exp(-1 + .5*Z[ix,3])*e1,
                    y2 = -.2 + .4*x2 + .8*Z[ix,2] + exp(-1 + .5*Z[ix,4])*e2, x1 = x1, x2 = x2, site = site)
  }
  dir.create(file.path(O, tag), recursive = TRUE, showWarnings = FALSE)
  write.csv(d, file.path(O, tag, "data.csv"), row.names = FALSE)
  write.csv(cbind(site = rownames(coords), coords), file.path(O, tag, "coords.csv"), row.names = FALSE)
  sp <- function() quote(spatial(1 | p | site, coords = coords))
  fm <- if (route == "q2") bf(mu1 = y1 ~ x1 + spatial(1|p|site, coords = coords), mu2 = y2 ~ x1 + spatial(1|p|site, coords = coords),
                              sigma1 = ~1, sigma2 = ~1, rho12 = ~1)
        else bf(mu1 = y1 ~ x1 + spatial(1|p|site, coords = coords), mu2 = y2 ~ x1 + spatial(1|p|site, coords = coords),
                sigma1 = ~spatial(1|p|site, coords = coords), sigma2 = ~spatial(1|p|site, coords = coords), rho12 = ~1)
  fit <- suppressWarnings(drmTMB(fm, family = biv_gaussian(), data = d))
  s <- summary(fit$sdreport, "fixed")
  cp <- corpairs(fit); cp <- cp[cp$level == "spatial", ]
  r <- data.frame(name = rownames(s), est = s[,1], se = s[,2])
  write.csv(r, file.path(O, tag, "r_fixed.csv"), row.names = FALSE)
  write.csv(cp[, c("from_dpar", "to_dpar", "estimate")], file.path(O, tag, "r_cor.csv"), row.names = FALSE)
  writeLines(c(sprintf("%.10f", as.numeric(logLik(fit))), as.character(fit$opt$convergence)), file.path(O, tag, "r_ll.txt"))
  cat(tag, "conv", fit$opt$convergence, "ll", as.numeric(logLik(fit)), "maxcor", max(abs(cp$estimate)), "\n")
}
gen("q2a", "q2", "spiral", 14, 4, 11); gen("q2b", "q2", "uniform", 12, 5, 12); gen("q2c", "q2", "cluster", 24, 4, 13)
gen("q4a", "q4", "uniform", 40, 6, 21); gen("q4b", "q4", "spiral", 36, 6, 24); gen("q4c", "q4", "cluster", 40, 6, 23)
