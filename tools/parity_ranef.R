# parity_ranef.R -- native-vs-Julia same-target comparison for the FIVE
# drmTMB-native random-effect capabilities whose bridge axis reads UNCITED on
# `docs/design/parity-scoreboard.md` (drmTMB): no receipt and no cited refusal.
#
# WHY THIS FILE EXISTS. Ordinary-random-effect parity HAD been measured before
# (drmTMB `inst/extdata/julia-capabilities.tsv` rows
# `gaussian_random_intercept_mu`, `gaussian_random_slope_mu`,
# `gaussian_sigma_random_intercept`, `gaussian_phylo_mean` all quote numbers),
# but the numbers were written into
# `docs/dev-log/evidence/julia-r-parity/ordinary-re-census/`, a table the
# scoreboard generator does not join. Evidence that exists and lifts nothing is
# still UNCITED. These rows land in `parity-classc.tsv`, which IS joined by
# `capability_id`, so a reader -- and the generator -- can reach them.
#
# WHY IT APPENDS INTO parity-classc.tsv RATHER THAN OPENING A NEW TABLE.
# drmTMB's `tools/write-parity-scoreboard.R` joins exactly four evidence tables
# (`parity-se`, `parity-fixtures`, `parity-classc`, `parity-phylo-nongaussian`).
# A fifth table would be invisible to it. The append-or-replace-by-`cell_id`
# contract used here is the one `tools/parity_classc_largep.R` already
# established on this same file: existing rows are preserved byte-for-byte and
# only this script's own `cell_id`s are rewritten.
#
# WHAT EACH CELL IS. The same target fitted twice through drmTMB,
# `engine = "tmb"` vs `engine = "julia"`, comparing coefficients, logLik, and
# fixed-effect Wald SEs. `status` is decided on coefficients + logLik at
# `tol = 1e-4` (the convention `tools/parity_classc.R` uses); the SE verdict is
# reported SEPARATELY in `note` at the 1e-3 relative bar `tools/parity_se.R`
# uses, because a coefficient/logLik agreement is not an SE claim and must not
# be read as one.
#
# TWO CELLS ARE REFUSALS, NOT FITS. `tweedie_random_intercept_mu` and
# `gaussian_phylo_slope_two_sd_mu` are recorded with status
# `REFUSED_BY_DRMTMB`: drmTMB's own bridge refuses the route before Julia
# starts, and the verbatim message is the evidence. They are cited negative
# controls, not passes, and the scoreboard reads them as such.
#
# D-139: the whole script is five small fits (n = 150; phylo p = 64, m = 3) plus
# two pure-R refusal probes. Measured wall clock on an M-series laptop, cold
# Julia: about 8 minutes, dominated by the first `engine = "julia"` boot.
#
#   DRM_JL_PATH=$PWD DRM_JL_PHYLO_PATH=$PWD OPENBLAS_NUM_THREADS=1 \
#     DRMTMB_JULIA_TESTS=true NOT_CRAN=true Rscript tools/parity_ranef.R

suppressMessages(library(drmTMB))
stopifnot(requireNamespace("ape", quietly = TRUE))

tol <- 1e-4          # coefficient / logLik bar (parity_classc.R convention)
se_rel_tol <- 1e-3   # SE bar (parity_se.R convention), reported separately
sd_floor <- 1e-6
out_path <- "docs/dev-log/evidence/parity-classc.tsv"

# One TSV cell per message. cli::cli_abort() wraps onto several lines and
# decorates bullets with non-ASCII glyphs; both are folded away so the note
# column stays one ASCII line (tools/parity_crosscheck.py and every sibling
# evidence TSV assume one row per cell, and an ASCII column travels).
one_line <- function(x) {
  txt <- trimws(paste(x, collapse = " "))
  txt <- gsub("\u2716", "x", txt)
  txt <- gsub("\u2139", "i", txt)
  txt <- iconv(txt, from = "UTF-8", to = "ASCII", sub = " ")
  gsub("\\s+", " ", trimws(txt))
}

# The single named random-effect SD on `dpar`. NA when the engine does not
# report one for that shape -- the Julia bridge returns the correlated
# `(1 + x | g)` block as raw log-Cholesky coefficients and leaves `sdpars`
# empty, a REPORTING gap that is recorded here, not silently filled in.
re_sd <- function(fit, dpar = "mu") {
  v <- tryCatch(fit$sdpars[[dpar]], error = function(e) NULL)
  if (!is.null(v) && length(v)) return(unname(v[[1L]]))
  NA_real_
}

# Fixed-effect Wald SEs in the order the fit returns them (the positional
# convention parity_classc.R uses). NULL, not an error, when vcov() fails on
# one side: an SE is a second-order quantity and its absence does not retract
# the coefficient/logLik comparison already made.
se_of <- function(fit) {
  tryCatch({
    v <- diag(as.matrix(vcov(fit)))
    ifelse(v > 0, sqrt(v), NA_real_)
  }, error = function(e) NULL)
}
fmt_se <- function(x) if (is.null(x)) "NA" else paste(sprintf("%.6g", x), collapse = ";")

empty_row <- function(capability_id, cell_id, label) {
  list(
    capability_id = capability_id, cell_id = cell_id, label = label,
    status = NA_character_, max_abs_coef_diff = NA_real_,
    loglik_tmb = NA_real_, loglik_julia = NA_real_, loglik_diff = NA_real_,
    max_abs_se_diff = NA_real_, max_rel_se_diff = NA_real_,
    se_tmb = "NA", se_julia = "NA", tolerance = tol, note = ""
  )
}

report <- function(res) {
  cat(sprintf("%-34s %-26s %-20s coef=%.3e ll=%.3e\n", res$capability_id,
              res$cell_id, res$status, res$max_abs_coef_diff, res$loglik_diff))
  if (nzchar(res$note)) cat("    note: ", substr(res$note, 1, 320), "\n", sep = "")
  as.data.frame(res, stringsAsFactors = FALSE)
}

run_cell <- function(capability_id, cell_id, label, native_expr, julia_expr,
                     sd_dpar = "mu") {
  res <- empty_row(capability_id, cell_id, label)
  ft <- tryCatch(eval(native_expr), error = function(e) {
    res$note <<- one_line(paste0("native: ", conditionMessage(e))); NULL })
  fj <- tryCatch(eval(julia_expr), error = function(e) {
    res$note <<- one_line(paste(res$note, paste0("julia: ", conditionMessage(e)))); NULL })

  sd_t <- if (!is.null(ft)) re_sd(ft, sd_dpar) else NA_real_
  sd_j <- if (!is.null(fj)) re_sd(fj, sd_dpar) else NA_real_

  if (is.null(ft)) {
    res$status <- "NATIVE_FAILED"
    return(report(res))
  }
  if (is.null(fj)) {
    res$status <- "JULIA_FAILED"
    return(report(res))
  }

  ct <- unlist(fixef(ft)); cj <- unlist(fixef(fj))
  k <- min(length(ct), length(cj))
  res$max_abs_coef_diff <- max(abs(ct[seq_len(k)] - cj[seq_len(k)]))
  res$loglik_tmb <- as.numeric(logLik(ft))
  res$loglik_julia <- as.numeric(logLik(fj))
  res$loglik_diff <- abs(res$loglik_tmb - res$loglik_julia)

  st <- se_of(ft); sj <- se_of(fj)
  res$se_tmb <- fmt_se(st); res$se_julia <- fmt_se(sj)
  se_verdict <- "SE_NOT_COMPARED (vcov unavailable on one engine)"
  if (!is.null(st) && !is.null(sj)) {
    ks <- min(length(st), length(sj))
    d <- abs(st[seq_len(ks)] - sj[seq_len(ks)])
    res$max_abs_se_diff <- max(d)
    res$max_rel_se_diff <- max(d / pmax(abs(st[seq_len(ks)]), abs(sj[seq_len(ks)])))
    se_verdict <- sprintf("%s (%d fixed-effect SE(s), max rel %.3e vs bar %.0e)",
                          if (res$max_rel_se_diff < se_rel_tol) "SE_PASS" else "SE_FAIL",
                          ks, res$max_rel_se_diff, se_rel_tol)
  }

  near_floor <- (!is.na(sd_j) && sd_j < 5 * sd_floor) || (!is.na(sd_t) && sd_t < 5 * sd_floor)
  if (near_floor) {
    res$status <- "BOUNDARY_NOT_COMPARABLE"
    res$note <- sprintf("fitted RE SD tmb=%.3e julia=%.3e (floor=%.0e); %s",
                        sd_t, sd_j, sd_floor, se_verdict)
  } else {
    agree <- res$max_abs_coef_diff < tol && res$loglik_diff < tol
    res$status <- if (agree) "PARITY_PASS" else "PARITY_FAIL"
    res$note <- sprintf("RE SD tmb=%s julia=%s; %d coefficient(s) compared; %s",
                        format(sd_t, digits = 6), format(sd_j, digits = 6), k, se_verdict)
  }
  report(res)
}

# A route drmTMB's own bridge refuses before Julia starts. The evidence is the
# verbatim message plus a POSITIVE CONTROL on the same engine call: a shape the
# fence must NOT catch has to get further than the refusal, otherwise a broken
# environment is indistinguishable from a fence.
run_refusal_cell <- function(capability_id, cell_id, label, refused_expr,
                             control_expr, control_desc) {
  res <- empty_row(capability_id, cell_id, label)
  got <- tryCatch({ eval(refused_expr); NULL },
                  error = function(e) one_line(conditionMessage(e)))
  ctl <- tryCatch({ eval(control_expr); "control fitted" },
                  error = function(e) one_line(conditionMessage(e)))
  if (is.null(got)) {
    res$status <- "NOT_REFUSED"
    res$note <- "the route was expected to be refused before Julia starts and was not"
  } else {
    res$status <- "REFUSED_BY_DRMTMB"
    res$note <- sprintf("refusal: %s || positive control (%s): %s",
                        substr(got, 1, 420), control_desc, substr(ctl, 1, 220))
  }
  report(res)
}

rows <- list()

## ---------------------------------------------------------------------
## (a) ordinary Gaussian random effects -- the census design (n = 150,
##     15 groups x 10, seed 20260904) so these numbers sit beside the ones
##     the drmTMB ledger rows already quote.
## ---------------------------------------------------------------------
set.seed(20260904)
ng <- 15L; per <- 10L; n <- ng * per
g <- factor(rep(paste0("g", seq_len(ng)), each = per))
x <- stats::rnorm(n)
u <- stats::rnorm(ng, 0, 0.6); ux <- stats::rnorm(ng, 0, 0.4)
dat_ri <- data.frame(y = 0.5 + 0.8 * x + u[g] + stats::rnorm(n, 0, 0.7), x = x, g = g)
dat_rs <- data.frame(y = 0.5 + 0.8 * x + u[g] + ux[g] * x + stats::rnorm(n, 0, 0.7), x = x, g = g)
dat_sg <- data.frame(y = 0.5 + 0.8 * x + stats::rnorm(n, 0, exp(-0.3 + u[g])), x = x, g = g)

dat <- dat_ri
rows[[length(rows) + 1L]] <- run_cell(
  "gaussian_random_intercept_mu", "gaussian_ri_mu_ml",
  "Gaussian, ordinary (1 | g) mean random intercept, sigma ~ 1, ML",
  quote(drmTMB(bf(y ~ x + (1 | g), sigma ~ 1), family = stats::gaussian(),
               data = dat, engine = "tmb")),
  quote(drmTMB(bf(y ~ x + (1 | g), sigma ~ 1), family = stats::gaussian(),
               data = dat, engine = "julia")))

rows[[length(rows) + 1L]] <- run_cell(
  "gaussian_random_intercept_mu", "gaussian_ri_mu_reml",
  "Gaussian, ordinary (1 | g) mean random intercept, sigma ~ 1, REML",
  quote(drmTMB(bf(y ~ x + (1 | g), sigma ~ 1), family = stats::gaussian(),
               data = dat, engine = "tmb", REML = TRUE)),
  quote(drmTMB(bf(y ~ x + (1 | g), sigma ~ 1), family = stats::gaussian(),
               data = dat, engine = "julia", REML = TRUE)))

dat <- dat_rs
rows[[length(rows) + 1L]] <- run_cell(
  "gaussian_random_slope_mu", "gaussian_rs_mu_ml",
  "Gaussian, correlated (1 + x | g) mean intercept+slope block, sigma ~ 1, ML",
  quote(drmTMB(bf(y ~ x + (1 + x | g), sigma ~ 1), family = stats::gaussian(),
               data = dat, engine = "tmb")),
  quote(drmTMB(bf(y ~ x + (1 + x | g), sigma ~ 1), family = stats::gaussian(),
               data = dat, engine = "julia")))

# The one cell that does NOT pass, and the reason it must not be promoted:
# drmTMB integrates the sigma-side random intercept by Laplace, DRModels.jl by
# 32-node Gauss-Hermite quadrature (src/gaussian_ranef.jl). Both converge; the
# gap is the approximation, not a wrong answer. It doubles as this table's
# negative control -- proof the harness can report a failure.
dat <- dat_sg
rows[[length(rows) + 1L]] <- run_cell(
  "gaussian_sigma_random_intercept", "gaussian_sigma_ri_ml",
  "Gaussian, fixed-effect mean, sigma ~ (1 | g) random intercept, ML (Laplace vs 32-node GHQ)",
  quote(drmTMB(bf(y ~ x, sigma ~ (1 | g)), family = stats::gaussian(),
               data = dat, engine = "tmb")),
  quote(drmTMB(bf(y ~ x, sigma ~ (1 | g)), family = stats::gaussian(),
               data = dat, engine = "julia")),
  sd_dpar = "sigma")

## ---------------------------------------------------------------------
## (b) Gaussian phylogenetic mean intercept. Seed 404 and a generating SD of
##     0.7 keep the phylogenetic SD well clear of DRModels.jl's
##     `_LAPLACE_LOG_SD_FLOOR` and of the unidentifiable draw DRModels.jl#483
##     found on the earlier seed.
## ---------------------------------------------------------------------
set.seed(404)
p <- 64L; m <- 3L
tree <- ape::rcoal(p); tree$tip.label <- paste0("sp_", seq_len(p))
A <- ape::vcv(tree, corr = TRUE)
uu <- as.vector(t(chol(A)) %*% stats::rnorm(p)) * 0.7
species <- factor(rep(tree$tip.label, each = m), levels = tree$tip.label)
idx <- rep(seq_len(p), each = m); nn <- p * m
xp <- stats::rnorm(nn)
dat <- data.frame(y = 0.4 + 0.6 * xp + uu[idx] + stats::rnorm(nn, 0, 0.5),
                  x = xp, species = species)
rows[[length(rows) + 1L]] <- run_cell(
  "gaussian_phylo_mean", "gaussian_phylo_mu_ml",
  "Gaussian, phylo(1 | species) mean intercept, sigma ~ 1, ML",
  quote(drmTMB(bf(y ~ x + phylo(1 | species, tree = tree), sigma ~ 1),
               family = stats::gaussian(), data = dat, engine = "tmb")),
  quote(drmTMB(bf(y ~ x + phylo(1 | species, tree = tree), sigma ~ 1),
               family = stats::gaussian(), data = dat, engine = "julia")))

## ---------------------------------------------------------------------
## (c) the two routes drmTMB REFUSES before Julia starts. Recorded so the
##     capabilities stop reading "no receipt and no cited refusal".
## ---------------------------------------------------------------------
set.seed(20260905)
nt <- 120L
id_tw <- factor(rep(paste0("g", seq_len(12L)), each = 10L))
u_tw <- stats::rnorm(12L, 0, 0.4); x_tw <- stats::rnorm(nt)
mu_tw <- exp(0.5 + 0.3 * x_tw + u_tw[id_tw])
dat_tw <- data.frame(
  y = drmTMB:::rtweedie_compound(nt, mu = mu_tw, phi = 1.2, power = 1.5),
  x = x_tw, id = id_tw
)
rows[[length(rows) + 1L]] <- run_refusal_cell(
  "tweedie_random_intercept_mu", "tweedie_ri_mu_fence",
  "Tweedie, ordinary (1 | id) mean random intercept -- refused by drmTMB's fe-only fence",
  quote(drmTMB(bf(y ~ x + (1 | id), nu ~ 1), family = tweedie(),
               data = dat_tw, engine = "julia")),
  quote(drmTMB(bf(y ~ x, nu ~ 1), family = tweedie(),
               data = dat_tw, engine = "julia")),
  "the same tweedie() call with NO random-effect bar")

set.seed(20260905)
ps <- 40L
tr <- ape::rcoal(ps); tr$tip.label <- paste0("sp_", seq_len(ps))
As <- ape::vcv(tr, corr = TRUE)
a_s <- as.vector(t(chol(As)) %*% stats::rnorm(ps)) * 0.6
b_s <- as.vector(t(chol(As)) %*% stats::rnorm(ps)) * 0.3
xs <- stats::rnorm(ps)
dat_ps <- data.frame(species = tr$tip.label, x = xs,
                     y = 0.4 + 0.5 * xs + a_s + b_s * xs + stats::rnorm(ps, 0, 0.4))
rows[[length(rows) + 1L]] <- run_refusal_cell(
  "gaussian_phylo_slope_two_sd_mu", "gaussian_phylo_slope_fence",
  "Gaussian, phylo(1 + x | species) two-SD mean intercept+slope -- refused by drmTMB's marker-slope guard",
  quote(drmTMB(bf(y ~ x + phylo(1 + x | species, tree = tr), sigma ~ 1),
               family = stats::gaussian(), data = dat_ps, engine = "julia")),
  quote(drmTMB(bf(y ~ x + phylo(1 | species, tree = tr), sigma ~ 1),
               family = stats::gaussian(), data = dat_ps, engine = "julia")),
  "the same call with an INTERCEPT-ONLY phylo marker")

## ---------------------------------------------------------------------
## merge into parity-classc.tsv, replacing only this script's own cell_ids
## (the contract tools/parity_classc_largep.R already uses on this file).
## ---------------------------------------------------------------------
new_tab <- do.call(rbind, rows)

.tools_dir <- tryCatch({
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(f)) dirname(sub("^--file=", "", f[1])) else "tools"
}, error = function(e) "tools")
source(file.path(.tools_dir, "drmtmb_provenance_lib.R"))
new_tab$drmtmb_code_hash <- drmtmb_code_hash()

base_order <- c("capability_id", "cell_id", "label", "status", "max_abs_coef_diff",
                "loglik_tmb", "loglik_julia", "loglik_diff",
                "max_abs_se_diff", "max_rel_se_diff", "se_tmb", "se_julia",
                "tolerance", "note", "drmtmb_code_hash")
old <- read.delim(out_path, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE,
                  colClasses = "character", na.strings = character(0), quote = "")
for (col in setdiff(base_order, names(old))) old[[col]] <- NA_character_
old <- old[, base_order]

new_chr <- as.data.frame(lapply(new_tab, as.character), stringsAsFactors = FALSE)
new_chr <- new_chr[, base_order]
for (i in seq_len(nrow(new_chr))) {
  hit <- which(old$cell_id == new_chr$cell_id[i])
  if (length(hit) == 1L) old[hit, ] <- new_chr[i, ] else old <- rbind(old, new_chr[i, ])
}

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write.table(old, out_path, sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
cat("\nwrote ", out_path, "\n", sep = "")
passing <- sum(new_tab$status == "PARITY_PASS")
cat(sprintf("OVERALL: %d of %d cells PARITY_PASS; %d refused by drmTMB; %d PARITY_FAIL\n",
            passing, nrow(new_tab), sum(new_tab$status == "REFUSED_BY_DRMTMB"),
            sum(new_tab$status == "PARITY_FAIL")))
