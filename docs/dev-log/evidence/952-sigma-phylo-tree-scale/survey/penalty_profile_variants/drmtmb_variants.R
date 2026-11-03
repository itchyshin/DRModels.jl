# drmTMB (ca3ddef) reference fits for the penalty / profile_ci sibling survey (#952).
suppressMessages({library(ape); library(drmTMB)})
D <- "/tmp/claude-0/-home-user-DRModels-jl/bf8c2b27-03aa-5044-87f6-e06cfadb9c02/scratchpad/n5/"
d <- read.csv(paste0(D, "carni70.csv"))
fmt <- function(x) paste(formatC(x, digits = 10, format = "g"), collapse = ", ")

fit_one <- function(tree, pen) {
  t0 <- Sys.time()
  f <- tryCatch(
    drmTMB(drm_formula(lsize ~ lrange + phylo(1 | species, tree = tree),
                       sigma ~ 1 + phylo(1 | species, tree = tree)),
           family = gaussian(), data = d, penalty = pen),
    error = function(e) e)
  attr(f, "time") <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  f
}

# The separate-block twin: drmTMB's own objective with eta_cor_phylo fixed at 0.
sep_twin <- function(f) {
  obj <- f$obj; p <- f$opt$par; k <- which(names(p) == "eta_cor_phylo")
  fn <- function(q) { pp <- p; pp[-k] <- q; pp[k] <- 0; obj$fn(pp) }
  gr <- function(q) { pp <- p; pp[-k] <- q; pp[k] <- 0; obj$gr(pp)[-k] }
  best <- NULL
  for (s in list(p[-k], c(p[-k][1:3], log(0.3), log(0.3)))) {
    o <- nlminb(s, fn, gr, control = list(eval.max = 1e4, iter.max = 1e4, rel.tol = 1e-14))
    if (is.null(best) || o$objective < best$objective) best <- o
  }
  pp <- p; pp[-k] <- best$par; pp[k] <- 0
  obj$fn(pp)
  rep <- obj$report(obj$env$last.par)
  pen <- if (is.null(rep$phylo_penalty)) 0 else rep$phylo_penalty
  list(par = setNames(best$par, names(p)[-k]), objective = best$objective, pen = pen,
       loglik = -best$objective + pen, maxgrad = max(abs(gr(best$par))),
       conv = best$convergence, fn = fn, gr = gr)
}

# Profile-likelihood CI (95%) for one log SD of the separate twin, on the SD scale.
prof_ci <- function(tw, idx) {
  thr <- qchisq(0.95, 1) / 2
  vhat <- tw$par[idx]; free <- setdiff(seq_along(tw$par), idx)
  prof <- function(v) {
    fn2 <- function(z) { q <- tw$par; q[idx] <- v; q[free] <- z; tw$fn(q) }
    gr2 <- function(z) { q <- tw$par; q[idx] <- v; q[free] <- z; tw$gr(q)[free] }
    ctl <- list(eval.max = 1e4, iter.max = 1e4, rel.tol = 1e-12)
    o <- tryCatch(nlminb(tw$par[free], fn2, gr2, control = ctl), error = function(e) NULL)
    if (is.null(o) || !is.finite(o$objective))
      o <- tryCatch(nlminb(tw$par[free], fn2, control = ctl), error = function(e) NULL)
    if (is.null(o) || !is.finite(o$objective)) return(NA_real_)
    (o$objective - tw$objective) - thr
  }
  # Bisection (24 halvings over an 8-unit log-SD bracket), NA-aware: an NA at the
  # midpoint is reported as an unresolved endpoint (NaN) rather than guessed.
  ends <- c(NA, NA)
  for (j in 1:2) {
    cap <- vhat + c(-8, 8)[j]
    dc <- prof(cap)
    if (is.na(dc)) { ends[j] <- NaN; next }
    if (dc <= 0) { ends[j] <- if (j == 1) -Inf else Inf; next }
    a <- vhat; b <- cap; ok <- TRUE
    for (it in 1:40) {
      m <- 0.5 * (a + b); dm <- prof(m)
      if (is.na(dm)) { ok <- FALSE; break }
      if (dm > 0) b <- m else a <- m
    }
    ends[j] <- if (ok) 0.5 * (a + b) else NaN
  }
  exp(ends)
}

for (lab in c("unit", "raw")) {
  tree <- read.tree(paste0(D, if (lab == "raw") "carni70.nwk" else "carni70_unit.nwk"))
  specs <- list(
    pen_sdu1 = drm_phylo_penalty(sd_u = 1, sd_alpha = 0.05),
    pen_sdu1_cor1 = drm_phylo_penalty(sd_u = 1, sd_alpha = 0.05, cor_sd = 1),
    ML = NULL)
  specs["ML"] <- list(NULL)
  for (nm in names(specs)) {
    f <- fit_one(tree, specs[[nm]])
    if (inherits(f, "error")) {
      cat(sprintf("R|%s|coupled_%s|ERROR|%s\n", lab, nm, conditionMessage(f))); next
    }
    ll <- logLik(f)
    cat(sprintf("R|%s|coupled_%s|loglik=%.10g|df=%d|is_converged=%s|time=%.1f|phylo_penalty=%s|par=[%s]|names=%s\n",
                lab, nm, as.numeric(ll), attr(ll, "df"), is_converged(f), attr(f, "time"),
                if (is.null(f$phylo_penalty)) "NA" else formatC(f$phylo_penalty, digits = 10, format = "g"),
                fmt(f$opt$par), paste(names(f$opt$par), collapse = ",")))
    if (nm %in% c("ML", "pen_sdu1")) {
      tw <- sep_twin(f)
      cat(sprintf("R|%s|separate_twin_%s|loglik=%.10g|objective=%.10g|phylo_penalty=%.10g|maxgrad=%.2e|nlminb_conv=%d|par=[%s]\n",
                  lab, nm, tw$loglik, tw$objective, tw$pen, tw$maxgrad, tw$conv, fmt(tw$par)))
      if (nm == "ML") {
        isd <- which(names(tw$par) == "log_sd_phylo")
        ci_mu <- prof_ci(tw, isd[1]); ci_sig <- prof_ci(tw, isd[2])
        cat(sprintf("R|%s|separate_twin_ML_profile|sd_mu=%.10g|ci_sd_mu=[%s]|sd_sigma=%.10g|ci_sd_sigma=[%s]\n",
                    lab, exp(tw$par[isd[1]]), fmt(ci_mu), exp(tw$par[isd[2]]), fmt(ci_sig)))
      }
    }
    flush(stdout())
  }
}
cat("# done R\n")
