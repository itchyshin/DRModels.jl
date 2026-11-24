# Regenerates r_reference.csv: drmTMB fits of the seven committed Beta structured-SD
# datasets (test/fixtures/beta_structured_sd/*.tsv), the independent R reference for
# test/test_beta_structured_sd.jl.
#
# Regenerate (from the repository root; needs R with ape and a built drmTMB >= 0.8.1):
#   Rscript test/fixtures/beta_structured_sd/r_make_reference.R [path/to/drmTMB/library]
# Reference committed here was made with drmTMB 0.8.1 (git tag v0.8.1, 6fc0910) built from source
# on R 4.6.0 / macOS arm64. Output columns:
#   shape, term, estimate, se, loglik, converged, drmTMB_version
# `se` is NA where drmTMB reports no finite Wald SE. Kernels mirror the Julia test
# (_bss_kernels): 16 levels per structure; relmat K and animal A are RAW covariances rho^|i-j| scaled by a
# linear s_i s_j; phylo is the tip correlation of the balanced ultrametric tree below;
# spatial is exp(-d / median pairwise distance) + 1e-6 I on a 4 x 4 unit grid.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1L) .libPaths(c(args[[1L]], .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
here <- tryCatch(dirname(normalizePath(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]]))), error = function(e) ".")

G <- 16L
ids <- sprintf("t%02d", seq_len(G))
raw_cov <- function(rho, lo, hi, ids) {
  R <- rho^abs(outer(seq_len(G), seq_len(G), "-"))
  s <- seq(lo, hi, length.out = G)
  K <- outer(s, s) * R
  dimnames(K) <- list(ids, ids)
  K
}
# Level labels in the committed fixtures: relmat k01.., animal a01.., phylo / spatial t01..
Kmat <- raw_cov(0.5, 0.7, 1.4, sprintf("k%02d", seq_len(G)))
Amat <- raw_cov(0.4, 0.8, 1.3, sprintf("a%02d", seq_len(G)))
nw <- function(ix, root) {
  if (length(ix) == 1L) return(sprintf("%s:1.5", ids[[ix]]))
  h <- length(ix) %/% 2L
  paste0("(", nw(ix[seq_len(h)], FALSE), ",", nw(ix[-seq_len(h)], FALSE), ")",
         if (root) ";" else ":0.1")
}
tree <- ape::read.tree(text = nw(seq_len(G), TRUE))
side <- 4L
coords <- cbind(x = rep(0:(side - 1L), times = side), y = rep(0:(side - 1L), each = side))
storage.mode(coords) <- "double"
rownames(coords) <- ids

shapes <- list(
  relmat_q1  = list(kind = "relmat",  q = 1L), spatial_q1 = list(kind = "spatial", q = 1L),
  phylo_q1   = list(kind = "phylo",   q = 1L), relmat_q2  = list(kind = "relmat",  q = 2L),
  spatial_q2 = list(kind = "spatial", q = 2L), phylo_q2   = list(kind = "phylo",   q = 2L),
  animal_q2  = list(kind = "animal",  q = 2L))
lvl <- c(relmat = "relmat", spatial = "spatial", phylo = "phylogenetic", animal = "animal")

make_formula <- function(kind, q) {
  lv <- lvl[[kind]]
  rhs <- if (q == 1L) "1" else "1 + x"
  prov <- switch(kind, relmat = "K = Kmat", animal = "A = Amat", phylo = "tree = tree",
                 spatial = "coords = coords")
  mu <- as.formula(sprintf("y ~ x + %s(%s | id, %s)", kind, rhs, prov), env = environment())
  sdf <- if (q == 1L) list(as.formula(sprintf('sd(id, level = "%s") ~ z0', lv), env = environment())) else
    list(as.formula(sprintf('sd(id, level = "%s", coef = "(Intercept)") ~ z0', lv), env = environment()),
         as.formula(sprintf('sd(id, level = "%s", coef = "x") ~ z1', lv), env = environment()))
  do.call(bf, c(list(mu, sigma ~ 1), sdf))
}

out <- list()
for (nm in names(shapes)) {
  d <- read.delim(file.path(here, paste0(nm, ".tsv")), stringsAsFactors = FALSE)
  d$id <- factor(d$id, levels = sort(unique(d$id)))
  fit <- drmTMB(make_formula(shapes[[nm]]$kind, shapes[[nm]]$q), family = beta_family(), data = d)
  tab <- summary(fit)$coefficients
  stopifnot(is.matrix(tab) || is.data.frame(tab))
  ll <- as.numeric(logLik(fit))
  conv <- tryCatch(as.logical(is_converged(fit)), error = function(e) NA)
  se <- tab[, grep("std", colnames(tab), ignore.case = TRUE)[[1L]]]
  se[!is.finite(se)] <- NA_real_
  # Terms are written without commas or quotes so the Julia test can split lines on ",".
  term <- gsub('"', "", gsub(",", ";", rownames(tab), fixed = TRUE), fixed = TRUE)
  out[[nm]] <- data.frame(shape = nm, term = term, estimate = tab[, 1L], se = se,
                          loglik = ll, converged = conv, drmTMB_version = as.character(packageVersion("drmTMB")),
                          stringsAsFactors = FALSE)
  cat(sprintf("%-11s ll=%.8f  converged=%s\n", nm, ll, conv))
}
res <- do.call(rbind, out)
write.csv(res, file.path(here, "r_reference.csv"), row.names = FALSE, quote = FALSE)
print(res, digits = 10)
