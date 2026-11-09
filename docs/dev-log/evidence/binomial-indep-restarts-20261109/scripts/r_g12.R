# R side of the G12 cells: drmTMB fit of the two binomial independent sd-z cells on the committed CSV.
suppressPackageStartupMessages(library(drmTMB))
cat("drmTMB", as.character(packageVersion("drmTMB")), "\n")
d <- read.csv(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "..", "..", "..", "..", "..", "test", "fixtures", "binomial_indep_restarts", "g12_binomial.csv"))
d$g <- factor(d$g, levels = unique(d$g))
for (q in 2:3) {
  re <- c("(1 | g) + (0 + x1 | g)", "(1 | g) + (0 + x1 | g) + (0 + x2 | g)")[q - 1]
  mu <- as.formula(paste("y ~ x1 + x2 + x3 +", re))
  sds <- list(sd(g, coef = "(Intercept)") ~ z, sd(g, coef = "x1") ~ z, sd(g, coef = "x2") ~ z)[seq_len(q)]
  fm <- do.call(bf, c(list(mu), sds))
  f <- drmTMB(fm, family = binomial(link = "logit"), data = d)
  cat(sprintf("R q=%d logLik=%.4f converged=%s\n", q, as.numeric(logLik(f)), is_converged(f)))
}
