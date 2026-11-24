## gen_biv_q4_phylo_reml.R -- generate the biv_q4_phylo_reml same-target fixture.
##
## Maintainer machine with local R + installed drmTMB. Writes generated
## data/numbers only into test/parity/q4-reml/biv-q4-phylo-reml/.
## Never edits the shared drmTMB checkout. Never vendors drmTMB source.
## Do not edit gen_fixtures.R / runparity.jl.

suppressPackageStartupMessages({
  library(drmTMB)
  library(ape)
})

repo_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    return(normalizePath(file.path(dirname(sub("^--file=", "", file_arg[[1]])), "..", "..")))
  }
  normalizePath(getwd())
}

toml_string <- function(x) {
  paste0('"', gsub('"', '\\"', as.character(x), fixed = TRUE), '"')
}

toml_num <- function(x) {
  if (!is.finite(x)) stop("cannot write non-finite TOML number: ", x)
  format(as.numeric(x), digits = 17, scientific = TRUE, trim = TRUE)
}

toml_bool <- function(x) {
  if (isTRUE(x)) return("true")
  if (identical(x, FALSE)) return("false")
  "\"NA\""
}

flat_coef <- function(fit) {
  cf <- coef(fit)
  out <- numeric()
  for (param in names(cf)) {
    vals <- as.numeric(cf[[param]])
    names(vals) <- paste0(param, "_", names(cf[[param]]))
    out <- c(out, vals)
  }
  names(out) <- sub(":", "_", names(out), fixed = TRUE)
  out
}

extract_pdHess <- function(fit) {
  tryCatch({
    if (!is.null(fit$sdr) && !is.null(fit$sdr$pdHess)) return(isTRUE(fit$sdr$pdHess))
    if (!is.null(fit$sdreport) && !is.null(fit$sdreport$pdHess)) return(isTRUE(fit$sdreport$pdHess))
    NA
  }, error = function(e) NA)
}

extract_interval_status <- function(fit) {
  ci <- tryCatch(confint(fit, method = "wald"), error = function(e) e)
  if (inherits(ci, "error")) {
    return(paste0("wald_unavailable:", conditionMessage(ci)))
  }
  if (is.null(ci)) return("wald_unavailable")
  nums <- suppressWarnings(as.numeric(as.matrix(ci)))
  if (length(nums) && all(is.finite(nums))) return("wald_finite")
  "wald_unavailable"
}

# Evidence-only helper; this file imports no package-engine source.
generate_bivtree <- function(seed, n_tip) {
  set.seed(seed)
  tree <- ape::rcoal(n_tip)
  tree$tip.label <- paste0("sp_", seq_len(n_tip))
  K <- ape::vcv.phylo(tree, corr = TRUE)
  factor <- t(chol(K))
  list(tree = tree, factor = factor)
}

simulate_cell <- function(seed, n_tip, n_each, with_x = TRUE) {
  fixture <- generate_bivtree(seed, n_tip)
  tree <- fixture$tree
  L <- fixture$factor
  Lam <- matrix(c(
    0.25, 0.10, 0.05, 0.00,
    0.10, 0.25, 0.00, 0.04,
    0.05, 0.00, 0.16, 0.02,
    0.00, 0.04, 0.02, 0.16
  ), 4, 4, byrow = TRUE)
  U <- L %*% matrix(stats::rnorm(n_tip * 4), n_tip, 4) %*% t(chol(Lam))
  tip <- rep(seq_len(n_tip), each = n_each)
  n <- length(tip)
  x <- if (with_x) stats::rnorm(n) else rep(0, n)
  dat <- data.frame(
    y1 = 0.5 + 0.3 * x + U[tip, 1] + exp(-0.6 + U[tip, 3]) * stats::rnorm(n),
    y2 = -0.2 + 0.4 * x + U[tip, 2] + exp(-0.6 + U[tip, 4]) * stats::rnorm(n),
    x = x,
    species = factor(tree$tip.label[tip], levels = tree$tip.label)
  )
  list(data = dat, tree = tree, seed = seed, n_tip = n_tip, n_each = n_each, with_x = with_x)
}

fit_cell <- function(cell) {
  tree <- cell$tree
  dat <- cell$data
  form <- if (cell$with_x) {
    bf(
      mu1 = y1 ~ x + phylo(1 | p | species, tree = tree),
      mu2 = y2 ~ x + phylo(1 | p | species, tree = tree),
      sigma1 = sigma1 ~ 1 + phylo(1 | p | species, tree = tree),
      sigma2 = sigma2 ~ 1 + phylo(1 | p | species, tree = tree),
      rho12 = rho12 ~ 1
    )
  } else {
    bf(
      mu1 = y1 ~ 1 + phylo(1 | p | species, tree = tree),
      mu2 = y2 ~ 1 + phylo(1 | p | species, tree = tree),
      sigma1 = sigma1 ~ 1 + phylo(1 | p | species, tree = tree),
      sigma2 = sigma2 ~ 1 + phylo(1 | p | species, tree = tree),
      rho12 = rho12 ~ 1
    )
  }
  drmTMB(
    form,
    family = biv_gaussian(),
    data = dat,
    REML = TRUE,
    engine = "tmb",
    control = drm_control(optimizer_preset = "robust")
  )
}

formula_text <- function(with_x) {
  mu <- if (with_x) "y1 ~ x + phylo(1 | species)" else "y1 ~ 1 + phylo(1 | species)"
  mu2 <- if (with_x) "y2 ~ x + phylo(1 | species)" else "y2 ~ 1 + phylo(1 | species)"
  paste0(
    "mu1 = ", mu, "; mu2 = ", mu2,
    "; sigma1 = sigma1 ~ 1 + phylo(1 | species); ",
    "sigma2 = sigma2 ~ 1 + phylo(1 | species); rho12 = rho12 ~ 1"
  )
}

r_call_text <- function(with_x) {
  mu <- if (with_x) "y1 ~ x + phylo(1 | p | species, tree = tree)" else "y1 ~ 1 + phylo(1 | p | species, tree = tree)"
  mu2 <- if (with_x) "y2 ~ x + phylo(1 | p | species, tree = tree)" else "y2 ~ 1 + phylo(1 | p | species, tree = tree)"
  paste0(
    "drmTMB(bf(mu1 = ", mu, ", mu2 = ", mu2,
    ", sigma1 = sigma1 ~ 1 + phylo(1 | p | species, tree = tree), ",
    "sigma2 = sigma2 ~ 1 + phylo(1 | p | species, tree = tree), rho12 = rho12 ~ 1), ",
    "family = biv_gaussian(), data = dat, REML = TRUE, engine = \"tmb\", ",
    "control = drm_control(optimizer_preset = \"robust\"))"
  )
}

write_fixture <- function(cell, fit, out_dir) {
  if (dir.exists(out_dir)) unlink(out_dir, recursive = TRUE)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  utils::write.csv(cell$data, file.path(out_dir, "data.csv"), row.names = FALSE)
  ape::write.tree(cell$tree, file.path(out_dir, "tree.newick"))

  coefs <- flat_coef(fit)
  ll <- as.numeric(stats::logLik(fit))
  n <- nrow(cell$data)
  converged <- isTRUE(identical(fit$opt$convergence, 0L) || identical(fit$opt$convergence, 0))
  pd <- extract_pdHess(fit)
  interval_status <- extract_interval_status(fit)
  estimator <- if (!is.null(fit$estimator)) as.character(fit$estimator) else "REML"

  con <- file(file.path(out_dir, "expected.toml"), "w")
  on.exit(close(con), add = TRUE)
  writeLines(c(
    "[fit]",
    paste0("family = ", toml_string("biv_gaussian")),
    paste0("formula = ", toml_string(formula_text(cell$with_x))),
    paste0("method = ", toml_string("REML")),
    paste0("engine = ", toml_string("tmb")),
    paste0("estimator = ", toml_string(estimator)),
    paste0("loglik = ", toml_num(ll)),
    paste0("n = ", as.integer(n)),
    "",
    "[coef]"
  ), con)
  for (name in sort(names(coefs))) {
    writeLines(paste0(toml_string(name), " = ", toml_num(coefs[[name]])), con)
  }
  writeLines(c(
    "",
    "[status]",
    paste0("converged = ", toml_bool(converged)),
    paste0("pdHess = ", toml_bool(pd)),
    paste0("interval_status = ", toml_string(interval_status)),
    "",
    "[tol]",
    "atol_loglik = 1e-3",
    "atol_coef = 1e-3",
    "rtol_coef = 1e-3"
  ), con)

  note <- paste(
    "Generated outputs only; no drmTMB source vendored.",
    "Workflow G fixtures remain 0.6.0 / ML / no tree.",
    "This cell is 0.7.0 REML + tree, outside the fixtures/ glob.",
    "Not a TSV supported flip. Not interval coverage or AI-REML."
  )
  writeLines(c(
    paste0("drmtmb_version = ", toml_string(as.character(utils::packageVersion("drmTMB")))),
    paste0("generated_on = ", toml_string(as.character(Sys.Date()))),
    paste0("r_call = ", toml_string(r_call_text(cell$with_x))),
    paste0("seed = ", as.integer(cell$seed)),
    paste0("n_tip = ", as.integer(cell$n_tip)),
    paste0("n_each = ", as.integer(cell$n_each)),
    paste0("with_x = ", toml_bool(cell$with_x)),
    paste0("note = ", toml_string(note))
  ), file.path(out_dir, "expected.meta.toml"))
}

attempts <- list(
  list(seed = 20260816L, n_tip = 16L, n_each = 5L, with_x = TRUE),
  list(seed = 20260817L, n_tip = 16L, n_each = 5L, with_x = TRUE),
  list(seed = 20260820L, n_tip = 16L, n_each = 5L, with_x = TRUE),
  list(seed = 20260821L, n_tip = 16L, n_each = 5L, with_x = TRUE),
  list(seed = 20260818L, n_tip = 16L, n_each = 8L, with_x = TRUE),
  list(seed = 20260822L, n_tip = 16L, n_each = 8L, with_x = TRUE),
  list(seed = 20260823L, n_tip = 12L, n_each = 5L, with_x = TRUE),
  list(seed = 20260819L, n_tip = 16L, n_each = 5L, with_x = FALSE),
  list(seed = 20260824L, n_tip = 12L, n_each = 8L, with_x = FALSE)
)

out_dir <- file.path(repo_root(), "test", "parity", "q4-reml", "biv-q4-phylo-reml")
ok <- FALSE
log_lines <- character()

for (att in attempts) {
  msg <- sprintf("attempt seed=%s n_tip=%s n_each=%s with_x=%s",
                 att$seed, att$n_tip, att$n_each, att$with_x)
  message(msg)
  log_lines <- c(log_lines, msg)
  cell <- simulate_cell(att$seed, att$n_tip, att$n_each, att$with_x)
  fit <- tryCatch(fit_cell(cell), error = function(e) e)
  if (inherits(fit, "error")) {
    log_lines <- c(log_lines, paste("  error:", conditionMessage(fit)))
    next
  }
  conv <- tryCatch(fit$opt$convergence, error = function(e) NA)
  est <- tryCatch(as.character(fit$estimator), error = function(e) NA_character_)
  ll <- tryCatch(as.numeric(stats::logLik(fit)), error = function(e) NA_real_)
  line <- sprintf("  estimator=%s convergence=%s logLik=%s", est, conv, ll)
  message(line)
  log_lines <- c(log_lines, line)
  if (identical(est, "REML") && (identical(conv, 0L) || identical(conv, 0))) {
    write_fixture(cell, fit, out_dir)
    ok <- TRUE
    log_lines <- c(log_lines, paste("  wrote", out_dir))
    break
  }
}

cat(paste(log_lines, collapse = "\n"), "\n")
if (!ok) {
  stop("HANDS TO Codex: no converged native-TMB REML cell on Mac-small attempts. Do not escalate to Totoro.")
}
invisible(NULL)
