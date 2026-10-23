suppressMessages(pkgload::load_all("~/local-scratch/lanes/drmTMB-arc1-pr1304-fold", quiet = TRUE))
rmvn <- function(S, sd) as.numeric(t(chol(S)) %*% stats::rnorm(nrow(S))) * sd
# Merge-time check (#814 + #815): species NOT in tip order, one tip absent.
# Run from this directory: Rscript native.R; then
#   julia --project=<DRModels.jl>/test julia.jl
out <- character()
## Phylo: 22-tip tree, data uses 21 tips (one absent), species NOT in tip order.
set.seed(20260925)
tree <- ape::rcoal(22, tip.label = sample(paste0("sp", sprintf("%02d", 1:22))))
C <- ape::vcv(tree, corr = TRUE)
absent <- tree$tip.label[5]
used <- setdiff(tree$tip.label, absent)
used <- rev(sort(used))                       # reverse-alphabetic, unrelated to tip order
dat <- expand.grid(sp = used, h = paste0("h", 1:5), stringsAsFactors = FALSE)
dat <- dat[sample.int(nrow(dat)), ]
a <- rmvn(C, 0.9); names(a) <- rownames(C)
b <- stats::rnorm(5, 0, 0.7); names(b) <- paste0("h", 1:5)
dat$x <- stats::rnorm(nrow(dat))
dat$y <- 1 + 0.5 * dat$x + a[dat$sp] + b[dat$h] + stats::rnorm(nrow(dat), 0, 0.6)
fit <- drmTMB(drmTMB::bf(y ~ x + phylo(1 | sp, tree = tree) + (1 | h), sigma ~ 1),
              family = gaussian(), data = dat, engine = "tmb")
stopifnot(fit$opt$convergence == 0)
write.table(dat, "phylo.data.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
ape::write.tree(tree, "phylo.tree.nwk")
cat("tip order:", tree$tip.label[1:6], "... absent:", absent, "; first-seen data:", unique(dat$sp)[1:6], "\n")
out <- c(out, sprintf("phylo\tdf\t%d", attr(logLik(fit), "df")),
         sprintf("phylo\tlogLik\t%.10f", as.numeric(logLik(fit))))
print(fit$sdpars$mu); print(fit$coefficients)
## Relmat: AR1 K over 20 ids, data first-seen order permuted.
set.seed(424242)
lev <- paste0("i", sprintf("%02d", 1:20))
K <- 0.6^abs(outer(1:20, 1:20, "-")); dimnames(K) <- list(lev, lev)
d2 <- expand.grid(id = lev, h = paste0("h", 1:5), stringsAsFactors = FALSE)
d2 <- d2[sample.int(nrow(d2)), ]
a <- rmvn(K, 0.8); names(a) <- lev
b <- stats::rnorm(5, 0, 0.7); names(b) <- paste0("h", 1:5)
d2$x <- stats::rnorm(nrow(d2))
d2$y <- 1 + 0.5 * d2$x + a[d2$id] + b[d2$h] + stats::rnorm(nrow(d2), 0, 0.6)
fit2 <- drmTMB(drmTMB::bf(y ~ x + relmat(1 | id, K = K) + (1 | h), sigma ~ 1),
               family = gaussian(), data = d2, engine = "tmb")
stopifnot(fit2$opt$convergence == 0)
write.table(d2, "relmat.data.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
ord <- unique(d2$id)
write.table(K[ord, ord], "relmat.K_firstseen.tsv", sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
out <- c(out, sprintf("relmat\tdf\t%d", attr(logLik(fit2), "df")),
         sprintf("relmat\tlogLik\t%.10f", as.numeric(logLik(fit2))))
writeLines(out, "native.tsv"); cat(out, sep = "\n")
