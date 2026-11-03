suppressMessages({library(ape); library(drmTMB)})
d <- read.csv("carni70.csv")
for (lab in c("raw", "unit")) {
  tree <- read.tree(if (lab == "raw") "carni70.nwk" else "carni70_unit.nwk")
  f <- drmTMB(drm_formula(lsize ~ lrange + phylo(1 | species, tree = tree),
                          sigma ~ 1 + phylo(1 | species, tree = tree)),
              family = gaussian(), data = d)
  obj <- f$obj; p <- f$opt$par; k <- which(names(p) == "eta_cor_phylo")
  fn <- function(q) { pp <- p; pp[-k] <- q; pp[k] <- 0; obj$fn(pp) }
  gr <- function(q) { pp <- p; pp[-k] <- q; pp[k] <- 0; obj$gr(pp)[-k] }
  best <- NULL
  for (s in list(p[-k], c(p[-k][1:3], log(0.3), log(0.3)))) {
    o <- nlminb(s, fn, gr, control = list(eval.max = 1e4, iter.max = 1e4, rel.tol = 1e-14))
    if (is.null(best) || o$objective < best$objective) best <- o
  }
  g <- gr(best$par)
  cat(sprintf("%s drmTMB cor-fixed-0 (separate): logLik = %.8f  max|grad| = %.2e  conv = %d\n", lab, -best$objective, max(abs(g)), best$convergence))
  print(setNames(best$par, names(p)[-k]), digits = 10)
}
