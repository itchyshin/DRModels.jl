source("ep_cell_r.R")
args <- commandArgs(TRUE); ids_all <- readLines(args[1])[-1]; con <- file(args[2], "w")
for (id in ids_all) {
  cl <- cells[id, ]; kk <- if (cl$kind %in% c("indep", "slope")) "ordinary" else cl$kind
  d <- read.csv(file.path(DD, sprintf("data_%s_%s.csv", cl$family, kk))); d$g <- factor(d$g, levels = unique(d$g)); d$g2 <- factor(d$g2)
  fit <- drmTMB(build(cl), family = fam_of(cl$family), data = d)
  for (s in 1:6) { sm <- as.matrix(simulate(fit, nsim = 1000, seed = s)); cat(sprintf("%s\tseed\t%d\t%.6g\t%.6g\n", id, s, var(as.numeric(sm)), mean(sm)), file = con); flush(con) }
}
