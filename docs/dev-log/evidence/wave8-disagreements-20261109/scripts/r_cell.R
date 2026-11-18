# Source run_r.R's builders without running its main loop (cwd must be harness/).
src <- readLines("run_r.R")
cut <- grep("^con <- file\\(outfile", src)[1]
commandArgs <- function(...) c("cells_run.tsv", "ids54.txt", "1", "/dev/null")
eval(parse(text = src[seq_len(cut - 1)]), envir = globalenv())
fit_cell <- function(id) {
  cl <- cells[id, ]
  d <- read.csv(file.path(DD, sprintf("data_%s_%s.csv", cl$family, if (cl$kind %in% c("indep", "slope")) "ordinary" else cl$kind)))
  d$g <- factor(d$g, levels = unique(d$g)); d$g2 <- factor(d$g2)
  fm <- build(cl)
  drmTMB(fm, family = fam_of(cl$family), data = d, control = drm_control(keep_tmb_object = TRUE))
}
