# Dump full fitted/predict vectors from drmTMB for a list of gap-map cell ids. Usage: Rscript dump_r.R cells.tsv ids.txt outdir
args <- commandArgs(TRUE)
.libPaths(c(Sys.getenv("DRMTMB_LIB", path.expand("~/drm_campaign_20261108/Rlib")), .libPaths()))
suppressPackageStartupMessages(library(drmTMB))
cells <- read.delim(args[1], stringsAsFactors = FALSE, quote = "", comment.char = ""); rownames(cells) <- cells$id
ids <- readLines(args[2]); out <- args[3]; dir.create(out, showWarnings = FALSE)
d <- read.csv(file.path(Sys.getenv("FUZZ_DATA", "data"), "data_v2.csv"), stringsAsFactors = FALSE, na.strings = "NA")
for (v in c("g", "g2", "h", "f", "f2", "gna", "gu", "y_cat")) d[[v]] <- factor(d[[v]])
d$y_ordf <- factor(d$y_ordf, levels = c("o1", "o2", "o3", "o4"), ordered = TRUE)
for (k in seq_along(ids)) {
  id <- ids[k]
  fit <- tryCatch(suppressWarnings(eval(parse(text = cells[id, "r"]))), error = function(e) NULL)
  if (is.null(fit)) next
  fv <- tryCatch(as.matrix(fitted(fit)), error = function(e) NULL)
  pv <- tryCatch(as.matrix(predict(fit)), error = function(e) NULL)
  w <- function(m, nm) if (!is.null(m)) write.table(cbind(m), file.path(out, sprintf("%03d_%s.csv", k, nm)), sep = ",", row.names = FALSE, col.names = FALSE)
  w(fv, "fitted"); w(pv, "predict")
  cat(sprintf("%03d\t%s\tnobs=%s\tfitted=%s\tpredict=%s\n", k, id, tryCatch(nobs(fit), error = function(e) NA),
      if (is.null(fv)) "NULL" else paste(dim(fv), collapse = "x"), if (is.null(pv)) "NULL" else paste(dim(pv), collapse = "x")))
}
