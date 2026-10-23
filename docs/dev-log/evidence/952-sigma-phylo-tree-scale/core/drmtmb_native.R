suppressMessages({library(ade4); library(ape); library(drmTMB)})
data(carni70)
tr <- read.tree(text = carni70$tre)
stopifnot(is.ultrametric(tr))
tab <- carni70$tab
sp <- gsub("_", ".", rownames(tab))
d <- data.frame(species = sp, lsize = log(tab$size), lrange = log(tab$range + 1))
d$lrange <- d$lrange - mean(d$lrange)
stopifnot(setequal(d$species, tr$tip.label))
write.csv(d, "carni70.csv", row.names = FALSE)
writeLines(carni70$tre, "carni70.nwk")
H <- max(node.depth.edgelength(tr)); cat("tree height", H, "\n")
tru <- tr; tru$edge.length <- tru$edge.length / H
write.tree(tru, "carni70_unit.nwk")
fitone <- function(tree, lab) {
  t0 <- Sys.time()
  f <- drmTMB(drm_formula(lsize ~ lrange + phylo(1 | species, tree = tree),
                          sigma ~ 1 + phylo(1 | species, tree = tree)),
              family = gaussian(), data = d)
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  ll <- logLik(f)
  cat(sprintf("%s: logLik = %.8f df = %d conv = %s  (%.1fs)\n", lab, as.numeric(ll), attr(ll, "df"), is_converged(f), el))
  print(f$opt$par, digits = 10)
  rep <- f$obj$report(); 
  invisible(f)
}
f_raw <- fitone(tr, "raw")
f_unit <- fitone(tru, "unit")
saveRDS(list(raw = f_raw$opt$par, unit = f_unit$opt$par), "drmTMB_par.rds")
