#!/usr/bin/env Rscript
# Bounded Gaussian mean-intercept bridge payload validation, not full parity.
args <- commandArgs(TRUE)
if (length(args) != 3L) stop('usage: parity_conditional_prediction.R R_CHECKOUT JULIA_CHECKOUT OUTPUT_JSON')
pkg <- normalizePath(args[[1L]], mustWork = TRUE)
jl <- normalizePath(args[[2L]], mustWork = TRUE)
out <- args[[3L]]
if (file.exists(out)) stop('refusing stale output')
Sys.setenv(DRM_JL_PATH = jl, DRMTMB_JULIA_TESTS = 'true', JULIA_NUM_THREADS = '1', OPENBLAS_NUM_THREADS = '1')
pkgload::load_all(pkg, quiet = TRUE, recompile = FALSE)
sha <- function(p) digest::digest(file = p, algo = 'sha256')
source_manifest <- function(root) {
  files <- sort(list.files(file.path(root, 'src'), recursive = TRUE))
  as.list(setNames(vapply(files, function(f) sha(file.path(root, 'src', f)), character(1L)), files))
}
set.seed(202608302L)
labels <- c('z', 'a', 'm', 'b', 'x', 'c', 'w', 'd', 'v', 'e', 'u', 'f')
g <- factor(rep(labels, each = 12L), levels = c('unused', rev(sort(labels))))
n <- length(g)
x <- runif(n, -1, 1)
b <- setNames(rnorm(length(labels), sd = 0.8), labels)
y <- 0.3 + 0.6*x + b[as.character(g)] + rnorm(n, sd = exp(-0.5 + 0.15*x))
dat <- data.frame(y, x, g)[sample.int(n), , drop = FALSE]
rownames(dat) <- NULL
grid <- data.frame(x = c(-0.7, 0, 0.8)) # zeroRE newdata needs no grouping column
forms <- list(constant_scale = bf(y ~ x + (1 | g), sigma ~ 1), varying_scale = bf(y ~ x + (1 | g), sigma ~ x))
forms$numeric_group <- bf(y ~ x + g + (1 | g), sigma ~ 1)
numeric_dat <- dat
numeric_dat$g <- as.numeric(match(as.character(dat$g), labels))
datasets <- list(constant_scale = dat, varying_scale = dat, numeric_group = numeric_dat)
grids <- list(constant_scale = grid, varying_scale = grid, numeric_group = transform(grid, g = c(1, 5, 12)))
receipt <- list(scope = 'ML Gaussian one ordinary mean random intercept, stored conditional and newdata zeroRE prediction',
  seed = 202608302L, data = dat, factor_levels = levels(dat$g), newdata = grid,
  independent_fit_tolerance = 4e-6, adapter_tolerance = 1e-10, expected_predictions = 24L,
  R_checkout = pkg, Julia_checkout = jl, Julia_source_sha256 = source_manifest(jl),
  R_bridge_sha256 = sha(file.path(pkg, 'R/julia-bridge.R')),
  runner_sha256 = sha('tools/parity_conditional_prediction.R'),
  native_fit_source_sha256 = sha(file.path(pkg, 'R/drmTMB.R')),
  native_methods_sha256 = sha(file.path(pkg, 'R/methods.R')), cases = list())
save_receipt <- function() jsonlite::write_json(receipt, out, pretty = TRUE, auto_unbox = TRUE, digits = 17, null = 'null')
start <- proc.time()[['elapsed']]
drmTMB:::drm_julia_setup()
JuliaCall::julia_command('using LinearAlgebra; LinearAlgebra.BLAS.set_num_threads(1)')
receipt$Julia_runtime <- list(version = JuliaCall::julia_eval('string(VERSION)'), threads = JuliaCall::julia_eval('Threads.nthreads()'), blas = JuliaCall::julia_eval('LinearAlgebra.BLAS.get_num_threads()'), loaded_source = JuliaCall::julia_eval('pathof(DRModels)'))
stopifnot(normalizePath(receipt$Julia_runtime$loaded_source) == normalizePath(file.path(jl, 'src/DRModels.jl')))
dll <- getLoadedDLLs()[['drmTMB']][['path']]
receipt$native_DLL <- list(path = dll, sha256 = sha(dll))
for (nm in names(forms)) {
  dat <- datasets[[nm]]
  grid <- grids[[nm]]
  receipt$cases[[nm]] <- tryCatch({
    fr <- drmTMB(forms[[nm]], data = dat, engine = 'tmb')
    fj <- drmTMB(forms[[nm]], data = dat, engine = 'julia')
    X <- model.matrix(if (nm == 'numeric_group') ~ x + g else ~ x, dat)
    S <- model.matrix(if (nm == 'varying_scale') ~ x else ~ 1, dat)
    # Dense Gaussian conditioning is independent of Julia's grouped sum code.
    group_factor <- factor(as.character(dat$g), levels = unique(as.character(dat$g)))
    Z <- model.matrix(~ 0 + group_factor)
    raw_names <- as.character(unlist(fj$bridge$coef_names))
    raw_values <- as.numeric(unlist(fj$bridge$coefficients))
    stopifnot(identical(names(fj$coefficients$mu), colnames(X)),
      identical(names(fj$coefficients$sigma), colnames(S)),
      length(raw_names) == length(raw_values), sum(raw_names == 'resd_g') == 1L)
    fixed <- drop(X %*% fj$coefficients$mu)
    log_sigma <- drop(S %*% fj$coefficients$sigma)
    sb2 <- exp(2*raw_values[raw_names == 'resd_g'])
    V <- diag(exp(2*log_sigma)) + sb2*tcrossprod(Z)
    modes <- drop(sb2 * crossprod(Z, solve(V, dat$y - fixed)))
    conditional <- fixed + drop(Z %*% modes)
    Xnew <- model.matrix(if (nm == 'numeric_group') ~ x + g else ~ x, grid)
    Snew <- model.matrix(if (nm == 'varying_scale') ~ x else ~ 1, grid)
    observations <- list()
    for (where in c('stored', 'newdata')) for (dpar in c('mu', 'sigma')) for (type in c('link', 'response')) {
      key <- paste(where, dpar, type, sep = '/')
      observations[[key]] <- tryCatch({
        nd <- if (where == 'stored') NULL else grid
        a <- predict(fr, newdata = nd, dpar = dpar, type = type)
        p <- predict(fj, newdata = nd, dpar = dpar, type = type)
        oracle <- if (dpar == 'mu') {
          if (where == 'stored') conditional else drop(Xnew %*% fj$coefficients$mu)
        } else {
          eta <- if (where == 'stored') log_sigma else drop(Snew %*% fj$coefficients$sigma)
          if (type == 'link') eta else exp(eta)
        }
        expected_n <- if (where == 'stored') n else nrow(grid)
        valid <- length(a) == expected_n && length(p) == expected_n && length(oracle) == expected_n && all(is.finite(c(a, p, oracle)))
        delta <- if (valid) max(abs(a-p)) else Inf
        adapter_delta <- if (valid) max(abs(p-oracle)) else Inf
        list(status = if (valid && delta < receipt$independent_fit_tolerance) 'PASS' else 'FAIL',
          adapter_status = if (valid && adapter_delta < receipt$adapter_tolerance) 'PASS' else 'FAIL',
          native = a, bridge = p, dense_oracle = oracle, max_abs_diff = delta, adapter_max_abs_diff = adapter_delta)
      }, error = function(e) list(status = 'ERROR', adapter_status = 'ERROR', error = conditionMessage(e)))
    }
    list(status = if (fr$opt$convergence == 0L && fj$opt$convergence == 0L && all(vapply(observations, function(z) identical(z$status, 'PASS'), logical(1)))) 'PASS' else 'FAIL',
      convergence = c(native = fr$opt$convergence, Julia = fj$opt$convergence),
      data = dat, newdata = grid,
      coefficients = lapply(fj$coefficients, function(z) list(names = names(z), values = unname(z))),
      raw_coefficients = list(names = raw_names, values = raw_values),
      observations = observations)
  }, error = function(e) list(status = 'ERROR', error = conditionMessage(e)))
  save_receipt()
  cat(nm, receipt$cases[[nm]]$status, '\n')
}
receipt$seconds <- proc.time()[['elapsed']] - start
observations <- unlist(lapply(receipt$cases, function(z) z$observations), recursive = FALSE)
receipt$completed_predictions <- length(observations)
receipt$adapter_status <- if (length(observations) == receipt$expected_predictions && all(vapply(observations, function(z) identical(z$adapter_status, 'PASS'), logical(1)))) 'PASS' else 'FAIL'
receipt$status <- if (receipt$adapter_status == 'PASS' && all(vapply(receipt$cases, function(z) identical(z$status, 'PASS'), logical(1)))) 'PASS' else 'FAIL'
receipt$bridge_unchanged_during_run <- identical(receipt$R_bridge_sha256, sha(file.path(pkg, 'R/julia-bridge.R')))
receipt$Julia_source_unchanged_during_run <- identical(receipt$Julia_source_sha256, source_manifest(jl))
if (!receipt$bridge_unchanged_during_run || !receipt$Julia_source_unchanged_during_run) receipt$status <- 'FAIL'
save_receipt()
cat('CONDITIONAL_ADAPTER_', receipt$adapter_status, ' predictions=', length(observations), '\n', sep = '')
cat('CONDITIONAL_FIT_PARITY_', receipt$status, '\n', sep = '')
if (receipt$status != 'PASS') quit(status = 1L)
