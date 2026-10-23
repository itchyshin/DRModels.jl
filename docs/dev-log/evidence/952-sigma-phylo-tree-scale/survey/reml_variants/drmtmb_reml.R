suppressPackageStartupMessages({ library(drmTMB); library(ape) })
D <- "/tmp/claude-0/-home-user-DRModels-jl/bf8c2b27-03aa-5044-87f6-e06cfadb9c02/scratchpad/n5"
d <- read.csv(file.path(D, "carni70.csv"))
trees <- list(raw = ape::read.tree(file.path(D, "carni70.nwk")),
              unit = ape::read.tree(file.path(D, "carni70_unit.nwk")))
cat("drmTMB version:", as.character(packageVersion("drmTMB")), "\n")
for (tn in names(trees)) cat("TREE", tn, "height =", format(max(ape::node.depth.edgelength(trees[[tn]])), digits = 12),
                             "ultrametric =", ape::is.ultrametric(trees[[tn]]), "\n")
res <- list()
for (vn in c("b_coupled", "c_asymmetric")) for (tn in names(trees)) {
  tree <- trees[[tn]]
  frm <- if (vn == "b_coupled") {
    drmTMB::drm_formula(lsize ~ lrange + phylo(1 | species, tree = tree),
                        sigma ~ 1 + phylo(1 | species, tree = tree))
  } else {
    drmTMB::drm_formula(lsize ~ lrange, sigma ~ 1 + phylo(1 | species, tree = tree))
  }
  t0 <- proc.time()[["elapsed"]]
  out <- tryCatch({
    f <- drmTMB::drmTMB(frm, family = gaussian(), data = d, REML = TRUE)
    dt <- proc.time()[["elapsed"]] - t0
    ll <- logLik(f)
    rnd <- tryCatch(paste(unique(names(f$obj$env$par[f$obj$env$random])), collapse = ","), error = function(e) "?")
    cat("RFIT", vn, tn, "logLik=", format(as.numeric(ll), digits = 10), "df=", attr(ll, "df"),
        "converged=", drmTMB::is_converged(f), "conv_code=", f$opt$convergence,
        "time_s=", sprintf("%.1f", dt), "\n  par:", paste(names(f$opt$par), format(f$opt$par, digits = 10), sep = "=", collapse = " "),
        "\n  random:", rnd, "\n")
    if (vn == "b_coupled") {
      # Supplementary (NOT a native drmTMB route): the separate block (a) as the
      # coupled REML objective with eta_cor_phylo held at 0, re-optimised over the rest.
      obj <- f$obj; p <- f$opt$par; k <- which(names(p) == "eta_cor_phylo")
      fn <- function(q) { pp <- p; pp[-k] <- q; pp[k] <- 0; obj$fn(pp) }
      gr <- function(q) { pp <- p; pp[-k] <- q; pp[k] <- 0; obj$gr(pp)[-k] }
      best <- NULL
      for (s in list(p[-k], replace(p[-k], grepl("log_sd|sd", names(p)[-k]), log(0.3)))) {
        o <- tryCatch(nlminb(s, fn, gr, control = list(eval.max = 1e4, iter.max = 1e4, rel.tol = 1e-14)),
                      error = function(e) NULL)
        if (!is.null(o) && (is.null(best) || o$objective < best$objective)) best <- o
      }
      if (!is.null(best)) {
        cat("RSEP0", tn, "logLik(cor fixed 0)=", format(-best$objective, digits = 10),
            "max|grad|=", format(max(abs(gr(best$par))), digits = 3), "nlminb_conv=", best$convergence,
            "\n  par:", paste(names(p)[-k], format(best$par, digits = 10), sep = "=", collapse = " "), "\n")
      } else cat("RSEP0", tn, "ERROR all starts failed\n")
    }
    as.numeric(ll)
  }, error = function(e) {
    cat("RFIT", vn, tn, "ERROR", conditionMessage(e), "\n"); NA_real_
  })
  res[[paste(vn, tn)]] <- out
}
for (vn in c("b_coupled", "c_asymmetric"))
  cat("RDELTA", vn, "raw-unit =", format(res[[paste(vn, "raw")]] - res[[paste(vn, "unit")]], digits = 6), "\n")
