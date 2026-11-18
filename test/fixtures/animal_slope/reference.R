# drmTMB 0.8.1 reference for the animal/relmat intercept-plus-slope cells.
# Usage: Rscript reference.R <outfile.tsv>   (run in this directory; DRMTMB_LIB = library holding drmTMB 0.8.1)
args <- commandArgs(TRUE); out <- args[1]
if (nzchar(Sys.getenv("DRMTMB_LIB"))) .libPaths(c(Sys.getenv("DRMTMB_LIB"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
ids <- read.csv("A_ids.csv")$id
A <- as.matrix(read.csv("A.csv", header = FALSE)); dimnames(A) <- list(ids, ids)
ld <- function(f) { d <- read.csv(sprintf("data_%s.csv", f)); d$g <- factor(d$g, levels = unique(d$g)); d }
sdf <- function(level, lab) {
  list(as.formula(sprintf('sd(g, level = "%s", coef = "(Intercept)") ~ 1', level)),
       as.formula(sprintf('sd(g, level = "%s", coef = "x1") ~ 1', level)))
}
cells <- list(
  beta_animal_bare   = list(d = "beta_animal", fam = beta_family(), reml = FALSE,
    f = bf(y ~ x1 + x2 + x3 + animal(1 + x1 | g, A = A), sigma = ~ 1)),
  beta_relmat_bare   = list(d = "beta_relmat", fam = beta_family(), reml = FALSE,
    f = bf(y ~ x1 + x2 + x3 + relmat(1 + x1 | g, K = A), sigma = ~ 1)),
  gauss_animal_reml  = list(d = "gaussian_animal", fam = gaussian(), reml = TRUE,
    f = bf(y ~ x1 + x2 + x3 + animal(1 + x1 | g, A = A), sigma = ~ 1)),
  gauss_relmat_reml  = list(d = "gaussian_relmat", fam = gaussian(), reml = TRUE,
    f = bf(y ~ x1 + x2 + x3 + relmat(1 + x1 | g, K = A), sigma = ~ 1)),
  binom_animal_indep_one = list(d = "binomial_animal", fam = binomial(link = "logit"), reml = FALSE,
    f = do.call(bf, c(list(y ~ x1 + x2 + x3 + animal(1 + x1 | g, A = A)), sdf("animal", 0)))),
  binom_animal_lab_one   = list(d = "binomial_animal", fam = binomial(link = "logit"), reml = FALSE,
    f = do.call(bf, c(list(y ~ x1 + x2 + x3 + animal(1 + x1 | p | g, A = A)), sdf("animal", 1)))))
zsd <- function(level) list(as.formula(sprintf('sd(g, level = "%s", coef = "(Intercept)") ~ z', level)),
                            as.formula(sprintf('sd(g, level = "%s", coef = "x1") ~ z', level)))
cells$binom_animal_indep_z <- list(d = "binomial_animal", fam = binomial(link = "logit"), reml = FALSE,
    f = do.call(bf, c(list(y ~ x1 + x2 + x3 + animal(1 + x1 | g, A = A)), zsd("animal"))))
cells$binom_animal_lab_z <- list(d = "binomial_animal", fam = binomial(link = "logit"), reml = FALSE,
    f = do.call(bf, c(list(y ~ x1 + x2 + x3 + animal(1 + x1 | p | g, A = A)), zsd("animal"))))
rows <- list()
for (nm in names(cells)) {
  cl <- cells[[nm]]
  fit <- drmTMB(cl$f, family = cl$fam, data = ld(cl$d), REML = cl$reml)
  s <- summary(fit$sdr, "fixed")
  rows[[nm]] <- data.frame(cell = nm, idx = seq_len(nrow(s)), par = rownames(s), estimate = s[, 1], se = s[, 2],
    loglik = as.numeric(logLik(fit)), converged = isTRUE(drmTMB::is_converged(fit)), pdhess = isTRUE(fit$sdr$pdHess))
  cat(nm, as.numeric(logLik(fit)), "conv", drmTMB::is_converged(fit), "\n")
}
write.table(do.call(rbind, rows), out, sep = "\t", quote = FALSE, row.names = FALSE)
