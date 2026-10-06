#!/usr/bin/env Rscript
# Independent observation-space Gaussian oracle for existing component fitters.
args <- commandArgs(TRUE)
if (length(args) != 3L) stop('usage: parity_conditional_components.R R_CHECKOUT JULIA_CHECKOUT OUTPUT_JSON')
pkg <- normalizePath(args[[1L]], mustWork = TRUE)
jl <- normalizePath(args[[2L]], mustWork = TRUE)
out <- args[[3L]]
if (file.exists(out)) stop('refusing stale output')
Sys.setenv(DRM_JL_PATH = jl, DRMTMB_JULIA_TESTS = 'true', JULIA_NUM_THREADS = '1', OPENBLAS_NUM_THREADS = '1')
pkgload::load_all(pkg, quiet = TRUE, recompile = FALSE)
sha <- function(p) digest::digest(file = p, algo = 'sha256')
manifest <- function() {
  files <- sort(list.files(file.path(jl, 'src'), recursive = TRUE))
  as.list(setNames(vapply(files, function(f) sha(file.path(jl, 'src', f)), character(1)), files))
}
set.seed(202608303L)
labels <- c('z','a','m','b','x','c','w','d','v','e','u','f')
g <- factor(rep(labels, each = 12L), levels = c('unused', rev(sort(labels))))
h <- rep(c(8,2,7,3,6,4), length.out = length(g))
x <- runif(length(g), -1, 1)
bg <- setNames(rnorm(12, sd = .8), labels)
bs <- setNames(.3*bg + rnorm(12, sd = .5), labels)
bh <- setNames(rnorm(6, sd = .6), as.character(unique(h)))
noise <- rnorm(length(g))
base <- .3 + .6*x
datasets <- list(
  independent_slope = data.frame(y = base + bs[as.character(g)]*x + exp(-.5+.15*x)*noise, x, g, h),
  correlated_slope = data.frame(y = base + bg[as.character(g)] + bs[as.character(g)]*x + exp(-.5)*noise, x, g, h),
  crossed_slope = data.frame(y = base + bg[as.character(g)] + bh[as.character(h)]*x + exp(-.5+.15*x)*noise, x, g, h),
  crossed_intercepts = data.frame(y = base + bg[as.character(g)] + bh[as.character(h)] + exp(-.5)*noise, x, g, h))
order <- sample.int(length(g))
datasets <- lapply(datasets, function(d) { d <- d[order, , drop=FALSE]; rownames(d) <- NULL; d })
forms <- list(independent_slope = bf(y ~ x + (0+x|g), sigma ~ x),
  correlated_slope = bf(y ~ x + (1+x|g), sigma ~ 1),
  crossed_slope = bf(y ~ x + (1|g) + (0+x|h), sigma ~ x),
  crossed_intercepts = bf(y ~ x + (1|g) + (1|h), sigma ~ 1))
grid <- data.frame(x = c(-.7, 0, .8))
receipt <- list(scope='Four additional ordinary Gaussian ML conditional component cases; not full parity',
  seed=202608303L, expected_predictions=32L, adapter_tolerance=1e-10, fit_tolerance=4e-6, likelihood_tolerance=1e-8,
  R_checkout=pkg, Julia_checkout=jl, Julia_source_sha256=manifest(),
  R_bridge_sha256=sha(file.path(pkg,'R/julia-bridge.R')), native_fit_sha256=sha(file.path(pkg,'R/drmTMB.R')),
  native_methods_sha256=sha(file.path(pkg,'R/methods.R')), runner_sha256=sha('tools/parity_conditional_components.R'), cases=list())
save_receipt <- function() jsonlite::write_json(receipt,out,pretty=TRUE,auto_unbox=TRUE,digits=17,null='null')
start <- proc.time()[['elapsed']]
drmTMB:::drm_julia_setup()
JuliaCall::julia_command('using LinearAlgebra; LinearAlgebra.BLAS.set_num_threads(1)')
receipt$Julia_runtime <- list(version=JuliaCall::julia_eval('string(VERSION)'), threads=JuliaCall::julia_eval('Threads.nthreads()'),
  blas=JuliaCall::julia_eval('LinearAlgebra.BLAS.get_num_threads()'), loaded_source=JuliaCall::julia_eval('pathof(DRModels)'))
stopifnot(normalizePath(receipt$Julia_runtime$loaded_source)==normalizePath(file.path(jl,'src/DRModels.jl')),
  receipt$Julia_runtime$threads==1, receipt$Julia_runtime$blas==1)
dll <- getLoadedDLLs()[['drmTMB']][['path']]
receipt$native_DLL <- list(path=dll,sha256=sha(dll))
for (nm in names(forms)) {
  dat <- datasets[[nm]]
  receipt$cases[[nm]] <- tryCatch({
    fr <- drmTMB(forms[[nm]],data=dat,engine='tmb')
    fj <- drmTMB(forms[[nm]],data=dat,engine='julia')
    varying <- nm %in% c('independent_slope','crossed_slope')
    multi <- startsWith(nm,'crossed_')
    X <- model.matrix(~x,dat); S <- model.matrix(if(varying) ~x else ~1,dat)
    Xnew <- model.matrix(~x,grid); Snew <- model.matrix(if(varying) ~x else ~1,grid)
    stopifnot(identical(names(fj$coefficients$mu),colnames(X)),identical(names(fj$coefficients$sigma),colnames(S)))
    raw_names <- as.character(unlist(fj$bridge$coef_names)); raw_values <- as.numeric(unlist(fj$bridge$coefficients))
    value <- function(key) { stopifnot(sum(raw_names==key)==1); raw_values[raw_names==key] }
    indicator <- function(v) { f <- factor(as.character(v),levels=unique(as.character(v))); model.matrix(~0+f) }
    A <- indicator(dat$g); W <- A*dat$x
    clamp <- function(v) pmax(-30,pmin(30,v))
    eta <- drop(S %*% fj$coefficients$sigma)
    clamp_active <- multi && any(abs(eta)>=30)
    variance <- exp(2*if(multi) clamp(eta) else eta)
    if (nm=='correlated_slope') {
      L <- matrix(c(exp(value('recov_g:L11')),value('recov_g:L21'),0,exp(value('recov_g:L22'))),2,2)
      C <- tcrossprod(L)
      K <- C[1,1]*tcrossprod(A)+C[1,2]*(tcrossprod(A,W)+tcrossprod(W,A))+C[2,2]*tcrossprod(W)
    } else {
      logsd <- value('resd_g'); clamp_active <- clamp_active || (multi && abs(logsd)>=30)
      sd2 <- exp(2*if(multi) clamp(logsd) else logsd)
      K <- sd2*tcrossprod(if(nm=='independent_slope') W else A)
      if(multi) {
        H <- indicator(dat$h); if(nm=='crossed_slope') H <- H*dat$x
        logh <- value('resd_h'); clamp_active <- clamp_active || abs(logh)>=30
        K <- K + exp(2*clamp(logh))*tcrossprod(H)
      }
    }
    fixed <- drop(X %*% fj$coefficients$mu); r <- dat$y-fixed
    V <- diag(variance)+K; alpha <- solve(V,r)
    conditional <- fixed+drop(K %*% alpha)
    oracle_loglik <- -.5*(length(r)*log(2*pi)+2*sum(log(diag(chol(V))))+sum(r*alpha))
    ll_error <- abs(oracle_loglik-fj$logLik)
    observations <- list()
    for(where in c('stored','newdata')) for(dpar in c('mu','sigma')) for(type in c('link','response')) {
      key <- paste(where,dpar,type,sep='/')
      observations[[key]] <- tryCatch({
        nd <- if(where=='stored') NULL else grid
        native <- as.numeric(predict(fr,newdata=nd,dpar=dpar,type=type)); bridge <- as.numeric(predict(fj,newdata=nd,dpar=dpar,type=type))
        oracle <- if(dpar=='mu') { if(where=='stored') conditional else drop(Xnew%*%fj$coefficients$mu) } else {
          z <- if(where=='stored') eta else drop(Snew%*%fj$coefficients$sigma)
          # Existing Julia multi-component engine clamps stored residual scales.
          if(multi && where=='stored') z <- clamp(z)
          if(type=='link') z else exp(z)
        }
        expected <- if(where=='stored') nrow(dat) else nrow(grid)
        valid <- length(native)==expected && length(bridge)==expected && length(oracle)==expected && all(is.finite(c(native,bridge,oracle)))
        a <- if(valid) max(abs(bridge-oracle)) else Inf; b <- if(valid) max(abs(native-bridge)) else Inf
        list(status=if(valid&&b<receipt$fit_tolerance)'PASS' else 'FAIL',adapter_status=if(valid&&a<receipt$adapter_tolerance)'PASS' else 'FAIL',native=native,bridge=bridge,dense_oracle=oracle,max_abs_diff=b,adapter_max_abs_diff=a)
      },error=function(e)list(status='ERROR',adapter_status='ERROR',error=conditionMessage(e)))
    }
    list(status=if(fr$opt$convergence==0&&fj$opt$convergence==0&&all(vapply(observations,function(x)identical(x$status,'PASS'),logical(1))))'PASS' else 'FAIL',
      convergence=c(native=fr$opt$convergence,Julia=fj$opt$convergence), data=dat,newdata=grid,
      raw_coefficients=list(names=raw_names,values=raw_values), clamp_active=clamp_active,
      likelihood=list(dense=oracle_loglik,Julia=fj$logLik,native=as.numeric(logLik(fr)),engine_oracle_error=ll_error,status=if(is.finite(ll_error)&&ll_error<receipt$likelihood_tolerance)'PASS' else 'FAIL'), observations=observations)
  },error=function(e)list(status='ERROR',error=conditionMessage(e)))
  save_receipt();cat(nm,receipt$cases[[nm]]$status,'\n')
}
receipt$seconds <- proc.time()[['elapsed']]-start
obs <- unlist(lapply(receipt$cases,function(x)x$observations),recursive=FALSE)
receipt$completed_predictions <- length(obs)
receipt$source_unchanged <- identical(receipt$Julia_source_sha256,manifest())&&identical(receipt$R_bridge_sha256,sha(file.path(pkg,'R/julia-bridge.R')))&&identical(receipt$native_fit_sha256,sha(file.path(pkg,'R/drmTMB.R')))
receipt$adapter_status <- if(receipt$source_unchanged&&length(obs)==32&&all(vapply(obs,function(x)identical(x$adapter_status,'PASS'),logical(1))))'PASS' else 'FAIL'
receipt$likelihood_status <- if(all(vapply(receipt$cases,function(x)identical(x$likelihood$status,'PASS'),logical(1))))'PASS' else 'FAIL'
receipt$status <- if(receipt$adapter_status=='PASS'&&receipt$likelihood_status=='PASS'&&all(vapply(receipt$cases,function(x)identical(x$status,'PASS'),logical(1))))'PASS' else 'FAIL'
save_receipt()
cat('COMPONENT_ADAPTER_',receipt$adapter_status,'; LIKELIHOOD_',receipt$likelihood_status,'; FIT_PARITY_',receipt$status,'\n',sep='')
if(receipt$status!='PASS')quit(status=1L)
