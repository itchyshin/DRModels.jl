DD <- Sys.getenv("FUZZ_DATA", "data"); .libPaths(c(path.expand("~/drm_campaign_20261108/Rlib"), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
cells_file <- "cells_all.tsv"; cells <- read.delim(cells_file, stringsAsFactors = FALSE)
rownames(cells) <- cells$cell_id

tree <- ape::read.tree(file.path(DD, "tree.nwk"))
cc <- read.csv(file.path(DD, "coords.csv")); coords <- as.matrix(cc[, c("easting", "northing")]); rownames(coords) <- cc$id
ids <- read.csv(file.path(DD, "A_ids.csv"))$id
A <- as.matrix(read.csv(file.path(DD, "A.csv"), header = FALSE)); dimnames(A) <- list(ids, ids); K <- A

fam_of <- function(f) switch(f,
  gaussian = gaussian(), poisson = poisson(), nb2 = nbinom2(), gamma = Gamma(link = "log"),
  binomial = binomial(link = "logit"), beta = beta_family(), lognormal = lognormal(),
  student = student(), tweedie = tweedie())
has_sigma <- function(f) !(f %in% c("poisson", "binomial"))
cname_r <- c("(Intercept)", "x1", "x2", "x3")
level_of <- c(animal = "animal", relmat = "relmat", phylo = "phylogenetic", spatial = "spatial")

build <- function(cl) {
  q <- cl$q; kind <- cl$kind
  terms <- c("1", "x1", "x2", "x3")[seq_len(q)]
  bar <- paste(terms, collapse = " + ")
  lab <- if (cl$lab == 1 || cl$cm == 1) " | p | " else " | "
  inner <- paste0(bar, lab, "g")
  re <- switch(kind,
    ordinary = paste0("(", inner, ")"),
    slope = "(0 + x1 | g)",
    indep = paste0("(1 | g)", paste0(" + (0 + ", c("x1", "x2", "x3")[seq_len(q - 1)], " | g)", collapse = "")),
    crossed = paste0("(", inner, ") + (1 | g2)"),
    animal = paste0("animal(", inner, ", A = A)"),
    relmat = paste0("relmat(", inner, ", K = K)"),
    phylo = paste0("phylo(", inner, ", tree = tree)"),
    spatial = paste0("spatial(", inner, ", coords = coords)"))
  mu <- as.formula(paste("y ~ x1 + x2 + x3 +", re)); environment(mu) <- environment()
  sel <- switch(cl$pat, none = integer(0), one = seq_len(q), z = seq_len(q), first = 1L, last = q)
  rhs <- if (cl$pat == "one") "1" else "z"
  lvl <- if (kind %in% names(level_of)) sprintf(', level = "%s"', level_of[[kind]]) else ""
  sds <- lapply(sel, function(j) {
    cf <- if (kind == "slope") ', coef = "x1"' else if (q == 1) "" else sprintf(', coef = "%s"', cname_r[j])
    as.formula(sprintf("sd(g%s%s) ~ %s", lvl, cf, rhs))
  })
  if (kind == "crossed" && cl$pat != "none" && cl$pat != "last") {
    sds <- c(sds, list(as.formula(sprintf("sd(g2) ~ %s", if (cl$pat == "one") "1" else "z2"))))
  }
  parts <- list(mu)
  if (has_sigma(cl$family)) {
    sg <- if (cl$cm == 1) as.formula("~ 1 + (1 | p | g)") else as.formula("~ 1")
    parts <- c(parts, list(sigma = sg))
  }
  do.call(bf, c(parts, sds))
}

