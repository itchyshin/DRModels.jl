#!/usr/bin/env Rscript
# Bounded native/bridge prediction contract; no installed-package substitution.
args <- commandArgs(TRUE)
if (length(args) != 3L) stop('usage: parity_prediction.R R_CHECKOUT JULIA_CHECKOUT OUTPUT_JSON')
pkg <- normalizePath(args[[1L]], mustWork = TRUE)
jl <- normalizePath(args[[2L]], mustWork = TRUE)
out <- args[[3L]]
if (file.exists(out)) stop('refusing stale output')
Sys.setenv(DRM_JL_PATH = jl, DRMTMB_JULIA_TESTS = 'true', JULIA_NUM_THREADS = '1', OPENBLAS_NUM_THREADS = '1')
pkgload::load_all(pkg, quiet = TRUE, recompile = FALSE)
sha <- function(p) digest::digest(file = p, algo = 'sha256')
git <- function(p) {
  value <- suppressWarnings(system2('git', c('-C', shQuote(p), 'rev-parse', 'HEAD'), stdout = TRUE, stderr = TRUE))
  if (!is.null(attr(value, 'status'))) return(NULL)
  value
}
set.seed(202608301L)
n <- 160L
x <- runif(n, -1, 1)
g <- factor(rep(c('a', 'b'), length.out = n))
y <- 0.3 + 0.6*x + 0.25*(g == 'b') + rnorm(n, sd = exp(-0.5 + 0.2*x))
dat <- data.frame(y, x, g, v = rep(0.02, n))
grid <- data.frame(x = c(-0.7, 0, 0.8), g = factor(c('a', 'b', 'a'), levels = levels(g)))
forms <- list(numeric = bf(y ~ x, sigma ~ x), factors = bf(y ~ x + g, sigma ~ x + g), default_scale = bf(y ~ x), known_variance = bf(y ~ x + meta_V(V = v), sigma ~ 1))
expected_predictions <- length(forms) * 8L
receipt <- list(scope = 'Gaussian fixed-effect native/bridge prediction contract only; not full parity or performance evidence',
  fixture_version = 2L, expected_predictions = expected_predictions,
  seed = 202608301L, data = dat, newdata = grid, tolerance = 4e-6,
  R = R.version.string, package_version = as.character(packageVersion('drmTMB')),
  R_checkout = pkg, R_head = git(pkg), Julia_checkout = jl, Julia_head = git(jl),
  adapter_oracle_scope = 'native prediction evaluated at the same Julia coefficients, tolerance1e-12; diagnostic only, not a replacement baseline',
  R_bridge_sha256 = sha(file.path(pkg, 'R/julia-bridge.R')),
  native_methods_sha256 = sha(file.path(pkg, 'R/methods.R')),
  runner_sha256 = sha('tools/parity_prediction.R'), cases = list())
save_receipt <- function() jsonlite::write_json(receipt, out, pretty = TRUE, auto_unbox = TRUE, digits = 17, null = 'null')
start <- proc.time()[['elapsed']]
drmTMB:::drm_julia_setup()
JuliaCall::julia_command('using LinearAlgebra; LinearAlgebra.BLAS.set_num_threads(1)')
receipt$Julia_runtime <- list(version = JuliaCall::julia_eval('string(VERSION)'), threads = JuliaCall::julia_eval('Threads.nthreads()'), blas = JuliaCall::julia_eval('LinearAlgebra.BLAS.get_num_threads()'), loaded_source = JuliaCall::julia_eval('pathof(DRModels)'))
if (normalizePath(receipt$Julia_runtime$loaded_source) != normalizePath(file.path(jl, 'src/DRModels.jl'))) stop('wrong Julia source loaded')
dll <- getLoadedDLLs()[['drmTMB']][['path']]
receipt$native_DLL <- list(path = dll, sha256 = sha(dll))
for (nm in names(forms)) {
  receipt$cases[[nm]] <- tryCatch({
    f <- forms[[nm]]
    fr <- drmTMB(f, data = dat, engine = 'tmb')
    fj <- drmTMB(f, data = dat, engine = 'julia')
    # Separate adapter oracle: native prediction with Julia's coefficients.
    # This isolates conversion/design from optimizer drift; it NEVER replaces
    # the original independently fitted native-R parity verdict below.
    same_parameters <- fr
    for (p in names(fr$coefficients)) {
      b <- fj$coefficients[[p]]
      names(b) <- gsub(': ', '', gsub(' & ', ':', names(b), fixed = TRUE), fixed = TRUE)
      expected <- names(fr$coefficients[[p]])
      if (anyDuplicated(names(b)) || !setequal(names(b), expected)) stop('adapter oracle coefficient names differ')
      same_parameters$coefficients[[p]] <- b[expected]
    }
    observations <- list()
    for (where in c('stored', 'newdata')) for (dpar in c('mu', 'sigma')) for (type in c('link', 'response')) {
      key <- paste(where, dpar, type, sep = '/')
      nd <- if (where == 'stored') NULL else grid
      observations[[key]] <- tryCatch({
        a <- predict(fr, newdata = nd, dpar = dpar, type = type)
        b <- predict(fj, newdata = nd, dpar = dpar, type = type)
        oracle <- predict(same_parameters, newdata = nd, dpar = dpar, type = type)
        expected_n <- if (where == 'stored') n else nrow(grid)
        same_length <- length(a) == expected_n && length(b) == expected_n
        difference <- if (same_length) max(abs(a - b)) else Inf
        list(status = if (same_length && is.finite(difference) && difference < receipt$tolerance) 'PASS' else 'FAIL', native = a, bridge = b, max_abs_diff = difference, adapter_oracle = oracle, adapter_max_abs_diff = max(abs(oracle - b)), adapter_status = if (length(oracle) == expected_n && length(b) == expected_n && all(is.finite(oracle - b)) && max(abs(oracle - b)) < 1e-12) 'PASS' else 'FAIL')
      }, error = function(e) list(status = 'ERROR', error = conditionMessage(e)))
    }
    fit_ok <- fr$opt$convergence == 0L && fj$opt$convergence == 0L
    list(status = if (fit_ok && all(vapply(observations, function(z) identical(z$status, 'PASS'), logical(1)))) 'PASS' else 'FAIL', formula = deparse(f), convergence = c(native = fr$opt$convergence, bridge = fj$opt$convergence), native_coefficients = coef(fr), bridge_coefficients = coef(fj), native_gradient_diagnostic = tryCatch(list(max_abs = max(abs(fr$obj$gr(fr$opt$par)))), error = function(e) list(unavailable = conditionMessage(e))), observations = observations)
  }, error = function(e) list(status = 'ERROR', error = conditionMessage(e)))
  save_receipt()
  cat(nm, receipt$cases[[nm]]$status, '\n')
}
receipt$seconds <- proc.time()[['elapsed']] - start
receipt$bridge_unchanged_during_run <- identical(receipt$R_bridge_sha256, sha(file.path(pkg, 'R/julia-bridge.R')))
observations <- unlist(lapply(receipt$cases, function(z) z$observations), recursive = FALSE)
receipt$completed_predictions <- length(observations)
receipt$adapter_status <- if (receipt$completed_predictions == expected_predictions && all(vapply(observations, function(z) identical(z$adapter_status, 'PASS'), logical(1)))) 'PASS' else 'FAIL'
receipt$status <- if (receipt$bridge_unchanged_during_run && identical(receipt$adapter_status, 'PASS') && length(receipt$cases) == length(forms) && all(vapply(receipt$cases, function(z) identical(z$status, 'PASS'), logical(1)))) 'PASS' else 'FAIL'
save_receipt()
cat('PREDICTION_CONTRACT_', receipt$status, ' cases=', length(receipt$cases), ' predictions=', receipt$completed_predictions, '\n', sep = '')
if (!identical(receipt$status, 'PASS')) quit(status = 1L)
