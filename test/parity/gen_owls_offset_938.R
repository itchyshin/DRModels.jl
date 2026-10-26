# Owls offset parity row for #938 (owner decision, 25 Oct 2026).
#
# Writes the prepared glmmTMB::Owls table to a CSV OUTSIDE the repository (the
# data ship with glmmTMB, AGPL; they are not vendored here) and prints drmTMB's
# fits of the issue's models, each with an `offset(lbs)` held at coefficient 1.
# test/test_offset_owls_parity_938.jl reads that CSV (DRM_OWLS_CSV) and checks
# DRModels.jl against the numbers recorded there. drmTMB is only run, never
# vendored.
#
#   Rscript test/parity/gen_owls_offset_938.R /path/outside/repo/owls.csv
#
# Load drmTMB from a source tree with DRMTMB_SRC=/path/to/drmTMB (pkgload),
# otherwise the installed package is used.
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args)) args[[1]] else "owls_938.csv"
src <- Sys.getenv("DRMTMB_SRC")
if (nzchar(src)) {
  pkgload::load_all(src, quiet = TRUE)
  cat("drmTMB source", src, "\n")
} else {
  suppressMessages(library(drmTMB))
}
cat("drmTMB", as.character(utils::packageVersion("drmTMB")), "\n")
data("Owls", package = "glmmTMB")
d <- data.frame(
  NCalls = Owls$SiblingNegotiation,
  arr = Owls$ArrivalTime - 24,
  lbs = log(Owls$BroodSize),
  Food = factor(Owls$FoodTreatment),
  Sex = factor(Owls$SexParent),
  Nest = factor(Owls$Nest)
)
write.csv(transform(d, Food = as.character(Food), Sex = as.character(Sex),
                    Nest = as.character(Nest)),
          out, row.names = FALSE, quote = FALSE)
show <- function(tag, f) {
  cat(tag, "logLik", format(as.numeric(logLik(f)), digits = 12),
      "conv", f$opt$convergence, "\n")
  print(unlist(coef(f)), digits = 10)
  sp <- tryCatch(summary(f)$sdpars, error = function(e) NULL)
  if (!is.null(sp)) print(sp, digits = 10)
}
# O1: NB2, fixed effects, sigma ~ 1 (the issue's first repro)
show("O1_NB2_FE", drmTMB(drm_formula(NCalls ~ Food + offset(lbs), sigma ~ 1),
                         family = nbinom2(), data = d))
# L2: NB2, fixed effects, sigma ~ Food
show("L2_NB2_FE_SIGMA", drmTMB(drm_formula(NCalls ~ Food * Sex + arr + offset(lbs),
                                           sigma ~ Food), family = nbinom2(), data = d))
# L1: NB2 + (1 | Nest), sigma ~ 1 (DRModels.jl `marginal = :Laplace`)
show("L1_NB2_RE", drmTMB(drm_formula(NCalls ~ Food * Sex + arr + offset(lbs) + (1 | Nest),
                                     sigma ~ 1), family = nbinom2(), data = d))
