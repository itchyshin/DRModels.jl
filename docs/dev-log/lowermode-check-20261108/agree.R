# R side of the wave-9 agreement study (native drmTMB, engine = "tmb"). Usage: Rscript agree.R <jobs.tsv> <k> <N> <out.tsv>
args <- commandArgs(TRUE)
jobs <- read.delim(args[1], header = FALSE, stringsAsFactors = FALSE, colClasses = c("character","character","integer"))
kk <- as.integer(args[2]); NN <- as.integer(args[3]); outfile <- args[4]
DD <- Sys.getenv("W9_DATA", "data")
.libPaths(c(path.expand("~/drm_campaign_20261108/Rlib"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
tree <- ape::read.tree(file.path(DD, "tree.nwk"))
cc <- read.csv(file.path(DD, "coords.csv")); coords <- as.matrix(cc[, c("easting", "northing")]); rownames(coords) <- cc$id
ids <- read.csv(file.path(DD, "A_ids.csv"))$id
A <- as.matrix(read.csv(file.path(DD, "A.csv"), header = FALSE)); dimnames(A) <- list(ids, ids); K <- A
fam_of <- function(f) switch(f, gaussian = gaussian(), poisson = poisson(), nb2 = nbinom2(), gamma = Gamma(link = "log"),
                              binomial = binomial(link = "logit"))
has_sigma <- function(f) !(f %in% c("poisson", "binomial"))
cname_r <- c("(Intercept)", "x1", "x2", "x3")
level_of <- c(animal = "animal", relmat = "relmat", phylo = "phylogenetic", spatial = "spatial")
build <- function(kind, lab, q, pat, fam) {
  terms <- c("1", "x1", "x2", "x3")[seq_len(q)]
  inner <- paste0(paste(terms, collapse = " + "), if (lab == 1) " | p | " else " | ", "g")
  re <- switch(kind, animal = paste0("animal(", inner, ", A = A)"), relmat = paste0("relmat(", inner, ", K = K)"),
               phylo = paste0("phylo(", inner, ", tree = tree)"), spatial = paste0("spatial(", inner, ", coords = coords)"))
  mu <- as.formula(paste("y ~ x1 + x2 + x3 +", re)); environment(mu) <- environment()
  sel <- switch(pat, none = integer(0), one = seq_len(q), z = seq_len(q))
  rhs <- if (pat == "one") "1" else "z"
  sds <- lapply(sel, function(j) as.formula(sprintf('sd(g, level = "%s", coef = "%s") ~ %s', level_of[[kind]], cname_r[j], rhs)))
  parts <- list(mu); if (has_sigma(fam)) parts <- c(parts, list(sigma = as.formula("~ 1")))
  do.call(bf, c(parts, sds))
}
fmt <- function(v) paste(sprintf("%.8g", v), collapse = ",")
con <- file(outfile, "a")
for (i in seq(kk, nrow(jobs), by = NN)) {
  sid <- jobs[i, 1]; cell <- jobs[i, 2]; seed <- jobs[i, 3]
  p <- strsplit(cell, "|", fixed = TRUE)[[1]]
  fam <- p[1]; kind <- p[2]; lab <- as.integer(substring(p[3], 4)); q <- as.integer(substring(p[4], 2)); pat <- substring(p[5], 4)
  file <- if (seed == 0) sprintf("data_%s_%s.csv", fam, kind) else sprintf("data_%s_%s_s%d.csv", fam, kind, seed)
  d <- read.csv(file.path(DD, file)); d$g <- factor(d$g, levels = unique(d$g))
  warn <- character(0); t0 <- Sys.time()
  res <- tryCatch(withCallingHandlers(list(ok = TRUE, fit = drmTMB(build(kind, lab, q, pat, fam), family = fam_of(fam), data = d)),
        warning = function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") }),
        error = function(e) list(ok = FALSE, e = e))
  secs <- as.numeric(Sys.time() - t0, units = "secs")
  if (!res$ok) { cat(paste(sid, seed, "ERROR", "", "", "", "", "", "", sprintf("%.1f", secs), gsub("[\t\n]", " ", substr(conditionMessage(res$e), 1, 300)), sep = "\t"), "\n", file = con, sep = ""); next }
  fit <- res$fit
  ll <- as.numeric(logLik(fit)); conv <- fit$opt$convergence
  pd <- isTRUE(fit$sdr$pdHess); isc <- tryCatch(isTRUE(drmTMB::is_converged(fit)), error = function(e) NA)
  s <- summary(fit$sdr, "fixed")
  cat(paste(sid, seed, "OK", sprintf("%.10f", ll), as.character(conv), fmt(fit$opt$par), fmt(s[, 2]),
            paste(names(fit$opt$par), collapse = ";"), sprintf("pdHess=%s;is_converged=%s", pd, isc), sprintf("%.1f", secs),
            gsub("[\t\n]", " ", substr(paste(warn, collapse = " || "), 1, 200)), sep = "\t"), "\n", file = con, sep = "")
}
close(con)
