# engine="julia" usability + threaded bootstrap — live demo of record

*2026-08-28, owner-requested. Script: the fenced R below; run against DRM.jl main (post-#517) and drmTMB main (post-#1098). Verbatim output follows.*

```r
# The owner's question, answered live: fit with engine="julia" IN R, use the
# result as an ordinary R object, and run the bootstrap THROUGH Julia — timed
# against the native TMB bootstrap on the same model and data.
suppressMessages(library(drmTMB)); suppressMessages(library(ape))

set.seed(404)
ntip <- 32; m <- 4
tree <- rcoal(ntip); tree$tip.label <- paste0("sp_", seq_len(ntip))
h <- max(diag(vcv(tree))); tree$edge.length <- tree$edge.length / h
A <- vcv(tree, corr = TRUE)
u <- as.vector(t(chol(A)) %*% rnorm(ntip)) * 0.6
species <- factor(rep(tree$tip.label, each = m), levels = tree$tip.label)
n <- ntip * m; x <- rnorm(n)
dat <- data.frame(y = 0.3 + 0.5 * x + u[as.integer(species)] + 0.4 * rnorm(n),
                  x = x, species = species)

cat("== fit in R with engine = 'julia' ==\n")
t_fit <- system.time(
  fj <- drmTMB(bf(y ~ x + phylo(1 | species, tree = tree), sigma ~ 1),
               family = gaussian(), data = dat, engine = "julia"))[["elapsed"]]
cat(sprintf("fit wall (incl. first-call startup): %.1f s\n\n", t_fit))

cat("== the result is an ordinary R object ==\n")
print(fj)
cat("\ncoef():   "); print(round(coef(fj, "mu"), 4))
cat("fixef():  "); print(round(unlist(fixef(fj)), 4))
cat(sprintf("logLik(): %.4f   nobs(): %d   sigma(): %.4f   converged: %s\n",
            as.numeric(logLik(fj)), nobs(fj), sigma(fj), is_converged(fj)))
cat("vcov() [mu block diag]: "); print(signif(diag(as.matrix(vcov(fj)))[1:2], 4))
nd <- data.frame(x = c(-1, 0, 1), species = rep(tree$tip.label[1], 3))
cat("predict(newdata):       "); print(round(as.numeric(predict(fj, newdata = nd)), 4))
cat("residuals()[1:4]:       "); print(round(residuals(fj)[1:4], 4))
cat("\nsummary():\n"); print(summary(fj))

cat("\n== intervals: wald / profile / BOOTSTRAP (through Julia) ==\n")
cat("wald:\n"); print(confint(fj, method = "wald"))
t_bj1 <- system.time(bj1 <- confint(fj, parm = "fixef:mu:x", method = "bootstrap", R = 199L, seed = 1L))[["elapsed"]]
cat(sprintf("\njulia bootstrap R=199, single-thread: %.1f s\n", t_bj1)); print(bj1)
t_bjt <- system.time(bjt <- confint(fj, parm = "fixef:mu:x", method = "bootstrap", R = 199L, seed = 1L,
                                    threads = TRUE))[["elapsed"]]
cat(sprintf("\njulia bootstrap R=199, threads=TRUE: %.1f s\n", t_bjt))

cat("\n== the same bootstrap natively in TMB, same model+data ==\n")
ft <- drmTMB(bf(y ~ x + phylo(1 | species, tree = tree), sigma ~ 1),
             family = gaussian(), data = dat)
t_bt <- tryCatch(system.time(bt <- confint(ft, parm = "mu:x", method = "boot", R = 199L, seed = 1L))[["elapsed"]],
                 error = function(e) { cat("native boot:", conditionMessage(e), "\n"); NA })
if (is.finite(t_bt)) {
  cat(sprintf("tmb bootstrap R=199: %.1f s  ->  julia speedup: %.1fx (single) / %.1fx (threads)\n",
              t_bt, t_bt / t_bj1, t_bt / t_bjt))
  print(bt)
}
```

## Output (verbatim, trimmed of loader noise)

```
== fit in R with engine = 'julia' ==
fit wall (incl. first-call startup): 15.6 s
== the result is an ordinary R object ==
<drmTMB Julia-engine fit>
observations: 128
estimator: ML
logLik: -72.43
convergence: 0
coef():   (Intercept)           x 
     0.3087      0.4631 
fixef():     mu.(Intercept)              mu.x sigma.(Intercept) 
           0.3087            0.4631           -0.9367 
logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
 logLik(): -72.4340   nobs(): 128   sigma(): 0.3919   converged: TRUE
vcov() [mu block diag]: mu_(Intercept)           mu_x 
      0.077190       0.001381 
predict(newdata):       [1] -0.1545  0.3087  0.7718
residuals()[1:4]:       [1] -0.0036 -0.2941  0.2122 -0.4521
summary():
<drmTMB Julia-engine fit summary>
observations: 128
logLik: -72.43
convergence: converged
uncertainty: partial
Fixed effects (link scale):
  dpar        term   estimate  std.error statistic      p.value
    mu (Intercept)  0.3086540 0.27782606  1.110961 2.665850e-01
    mu           x  0.4631103 0.03715795 12.463291 1.183822e-35
 sigma (Intercept) -0.9367030         NA        NA           NA
Random effects (SD, response scale):
 dpar               term        sd
   mu phylo(1 | species) 0.4318211
== intervals: wald / profile / BOOTSTRAP (through Julia) ==
wald:
                     parm level      lower     upper    scale   transformation
1    fixef:mu:(Intercept)  0.95 -0.2358751 0.8531831     link linear_predictor
2              fixef:mu:x  0.95  0.3902821 0.5359385     link linear_predictor
3 fixef:sigma:(Intercept)  0.95         NA        NA     link linear_predictor
4                   sigma  0.95         NA        NA response              exp
      tmb_parameter index method profile.engine      conf.status
1    mu_(Intercept)     1   wald           <NA>             wald
2              mu_x     2   wald           <NA>             wald
3 sigma_(Intercept)     1   wald           <NA> wald_unavailable
4 sigma_(Intercept)     1   wald           <NA> wald_unavailable
  profile.boundary profile.message
1               NA            <NA>
2               NA            <NA>
3               NA            <NA>
4               NA            <NA>
julia bootstrap R=199, single-thread: 5.1 s
        parm level     lower     upper scale   transformation tmb_parameter
1 fixef:mu:x  0.95 0.3903631 0.5288836  link linear_predictor          mu_x
  index    method profile.engine conf.status profile.boundary
1     2 bootstrap           <NA>   bootstrap            FALSE
            profile.message julia.threaded julia.workers julia.threads
1 199/199 successful refits          FALSE             1             1
  julia.blas_threads julia.elapsed bootstrap.n bootstrap.failed
1                 16      1.389054         199                0
  bootstrap.parallel bootstrap.workers
1               none                 1
julia bootstrap R=199, threads=TRUE: 1.4 s
== the same bootstrap natively in TMB, same model+data ==
tmb bootstrap R=199: 13.7 s  ->  julia speedup: 2.7x (single) / 10.0x (threads)
        parm level     lower     upper scale   transformation tmb_parameter
1 fixef:mu:x  0.95 0.3836529 0.5392479  link linear_predictor       beta_mu
  index    method profile.engine conf.status profile.boundary
1     2 bootstrap           <NA>   bootstrap               NA
            profile.message bootstrap.n bootstrap.failed bootstrap.parallel
1 199/199 successful refits         199                0               none
  bootstrap.workers
1                 1
Julia exit.
```

## Reading

- The engine="julia" fit is a first-class R object: print/summary/coef/fixef/vcov/logLik/nobs/sigma/residuals/predict(newdata) all exercised above.
- Known wrinkle, stated: the sigma-block Wald SE is NA on this phylo route (summary says 'uncertainty: partial'); that block's uncertainty comes via method="profile".
- Bootstrap R=199, same model+data: TMB 13.7 s · Julia single 5.1 s (2.7x) · Julia threads=TRUE 1.4 s (10x); 199/199 refits both engines; intervals agree to bootstrap MC variation.
- Correction (2026-10-03, #689): this session ran Julia with one thread (`julia.threads` = 1 above), so the `threads = TRUE` call ran its refits serially. The 10x is warm serial Julia against TMB, not a threading gain; `threads = TRUE` parallelises only when `JULIA_NUM_THREADS` > 1 is set before Julia starts.
- The Julia bootstrap is per-target by design (#460): parm = "fixef:mu:x".
