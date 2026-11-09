source("ep_cell_r.R")
args <- commandArgs(TRUE); ids_all <- readLines(args[1])[-1]; outfile <- args[2]
con <- file(outfile, "w")
emit <- function(...) { cat(paste(c(...), collapse = "\t"), "\n", file = con, sep = ""); flush(con) }
for (id in ids_all) {
  cl <- cells[id, ]; kk <- if (cl$kind %in% c("indep", "slope")) "ordinary" else cl$kind
  d <- read.csv(file.path(DD, sprintf("data_%s_%s.csv", cl$family, kk))); d$g <- factor(d$g, levels = unique(d$g)); d$g2 <- factor(d$g2)
  fit <- tryCatch(drmTMB(build(cl), family = fam_of(cl$family), data = d, REML = (cl$est == "REML")), error = function(e) NULL)
  if (is.null(fit)) { emit(id, "prof", "FIT_ERROR"); next }
  t0 <- Sys.time()
  pn <- rownames(summary(fit)$coefficients); ci <- tryCatch(suppressWarnings(confint(fit, parm = pn, method = "profile")), error = function(e) { emit(id, "prof", "ERROR", gsub("[\t\n]+", " ", substr(conditionMessage(e), 1, 250))); NULL })
  if (!is.null(ci)) for (i in seq_len(nrow(ci))) emit(id, "prof", paste(ci$parm[i], ci$scale[i], sep = "|"), sprintf("%.10g", ci$lower[i]), sprintf("%.10g", ci$upper[i]), paste(ci$conf.status[i], ci$method[i]))
  emit(id, "prof", "secs", sprintf("%.1f", as.numeric(Sys.time() - t0, units = "secs")))
}
close(con)
