## compare_phylo_nb2.R — aggregate Julia vs drmTMB NB2 phylo timings.
##
## Run from repo root after both fit runners:
##   Rscript bench/R/compare_phylo_nb2.R

suppressPackageStartupMessages(library(jsonlite))

`%||%` <- function(a, b) if (is.null(a)) b else a

here <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script_path <- if (length(file_arg)) {
    normalizePath(sub("^--file=", "", file_arg[[1]]))
  } else {
    normalizePath(sys.frames()[[1]]$ofile %||% ".")
  }
  dirname(dirname(script_path))
}

bench_root <- tryCatch(here(), error = function(e) file.path(getwd(), "bench"))
results_dir <- file.path(bench_root, "results", "phylo_nb2")
report_dir <- file.path(dirname(bench_root), "report")
dir.create(report_dir, showWarnings = FALSE, recursive = TRUE)

r_path <- file.path(results_dir, "r_phylo_nb2.json")
j_path <- file.path(results_dir, "julia_phylo_nb2.json")
if (!file.exists(r_path)) stop("R results missing: ", r_path)
if (!file.exists(j_path)) stop("Julia results missing: ", j_path)

r_res <- jsonlite::read_json(r_path, simplifyVector = FALSE)
j_res <- jsonlite::read_json(j_path, simplifyVector = FALSE)

idx <- function(x) setNames(x, vapply(x, function(z) as.character(z$cell_id), character(1)))
r_by <- idx(r_res)
j_by <- idx(j_res)
cell_ids <- names(r_by)
missing_j <- setdiff(cell_ids, names(j_by))
if (length(missing_j)) stop("Julia results missing cells: ", paste(missing_j, collapse = ", "))

num <- function(x) suppressWarnings(as.numeric(x %||% NA_real_))
vec <- function(x) as.numeric(unlist(x %||% numeric()))

rows <- lapply(cell_ids, function(id) {
  r <- r_by[[id]]
  j <- j_by[[id]]
  rb <- vec(r$beta_mu)
  jb <- vec(j$beta_mu)
  d_beta <- if (length(rb) == length(jb) && length(rb)) max(abs(rb - jb)) else NA_real_
  rt <- num(r$time_s_med)
  jt <- num(j$time_s_med)
  data.frame(
    cell_id = id,
    p = as.integer(r$p %||% j$p %||% NA_integer_),
    n = as.integer(r$n %||% j$n %||% NA_integer_),
    r_time = rt,
    julia_time = jt,
    speedup = if (is.finite(rt) && is.finite(jt) && jt > 0) rt / jt else NA_real_,
    dlogLik = abs(num(r$logLik) - num(j$logLik)),
    max_d_beta = d_beta,
    d_theta = abs(num(r$theta_nb2) - num(j$theta_nb2)),
    d_sd = abs(num(r$sd_phylo) - num(j$sd_phylo)),
    r_converged = isTRUE(r$converged),
    julia_converged = isTRUE(j$converged),
    stringsAsFactors = FALSE
  )
})
tbl <- do.call(rbind, rows)

fmt <- function(x, f = "%.4f") if (is.finite(x)) sprintf(f, x) else "-"
fmtx <- function(x) if (is.finite(x)) sprintf("%.2fx", x) else "-"
line <- function(row) {
  sprintf("| %-11s | %5d | %5d | %9s | %9s | %8s | %9s | %9s | %8s | %8s | %s/%s |",
          row$cell_id, row$p, row$n, fmt(row$r_time), fmt(row$julia_time),
          fmtx(row$speedup), fmt(row$dlogLik, "%.3e"), fmt(row$max_d_beta, "%.3e"),
          fmt(row$d_theta, "%.3f"), fmt(row$d_sd, "%.3f"),
          row$r_converged, row$julia_converged)
}

speedups <- tbl$speedup[is.finite(tbl$speedup)]
headline <- if (length(speedups)) {
  sprintf("Measured median speedup R/Julia: %.2fx (min %.2fx, max %.2fx)",
          stats::median(speedups), min(speedups), max(speedups))
} else {
  "Measured median speedup R/Julia: unavailable"
}

conv_gate <- all(tbl$r_converged & tbl$julia_converged)
speed_gate <- length(speedups) && all(speedups > 1)
large_gate <- {
  large <- tbl[tbl$p >= 1000, , drop = FALSE]
  nrow(large) > 0 && all(large$speedup >= 2, na.rm = TRUE)
}
parity_gate <- all(tbl$dlogLik < 1e-3, na.rm = TRUE) &&
  all(tbl$max_d_beta < 1e-4, na.rm = TRUE) &&
  all(tbl$d_theta < 1e-3, na.rm = TRUE) &&
  all(tbl$d_sd < 1e-3, na.rm = TRUE)

md <- c(
  "# NB2 phylo sparse-Laplace benchmark",
  "",
  headline,
  "",
  "| cell | p | n | R med/s | Julia med/s | speedup | |dLL| | max|dβ| | |dTheta| | |dSD| | conv R/J |",
  "|:-----|--:|--:|--------:|------------:|--------:|------:|--------:|--------:|------:|:---------|",
  vapply(seq_len(nrow(tbl)), function(i) line(tbl[i, ]), character(1)),
  "",
  "## Gates",
  "",
  sprintf("- Both engines converge in every measured cell: %s", if (conv_gate) "PASS" else "FAIL"),
  sprintf("- Julia faster than drmTMB in every measured cell: %s", if (speed_gate) "PASS" else "FAIL"),
  sprintf("- p >= 1000 cells at least 2x faster: %s", if (large_gate) "PASS" else "FAIL"),
  sprintf("- Likelihood/estimate parity on reported metrics: %s", if (parity_gate) "PASS" else "FAIL"),
  "- CPU-aware timing: both engines are run without post-fit SE/sdreport; Julia pins BLAS to one thread and uses `g_tol = 1e-6` for the benchmark fits.",
  "- Scale note: drmTMB reports NB2 public `sigma`; this report converts it to size `theta = 1 / sigma^2` before comparing with DRModels.jl's current NB2 `sigma` slot.",
  "- Scope: NB2 mean model with `phylo(1 | species)` and `sigma ~ 1` only; no `zi`/`hu`, structured `sigma`, or q>1 non-Gaussian phylo model is claimed here.",
  "- All speedups above are measured from the JSON result files, not extrapolated.",
  ""
)

out <- file.path(report_dir, "phylo-nb2-benchmark.md")
writeLines(md, out)
cat(paste(md, collapse = "\n"), "\n")
