IDS <- commandArgs(TRUE); suppressPackageStartupMessages(library(drmTMB)); source("r_cell.R")
for (fam in c("nb2","gamma","beta","student","tweedie")) {
  d <- read.csv(sprintf("data/data_%s_ordinary.csv", fam))
  fit <- drmTMB(bf(y ~ x1+x2+x3, sigma ~ 1), family = fam_of(fam), data = d)
  cat("R", fam, "ll=", as.numeric(logLik(fit)), "par=", round(fit$opt$par, 5), "names=", names(fit$opt$par), "\n")
}
