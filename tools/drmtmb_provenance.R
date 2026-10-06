#!/usr/bin/env Rscript
# Identify WHICH drmTMB build is installed — not just its version string.
#
# Why this exists
# ---------------
# Every parity fixture in test/parity/ records `drmtmb_version = "0.7.0"`. That
# string does not identify a build. Measured 2026-08-24: the installed drmTMB was
# built 2026-08-15, while drmTMB origin/main had SIXTEEN further commits to
# shipped files (R/, src/, NAMESPACE) — including one that changed which formulas
# are admitted (d30841491, "binomial responses accept a phylogenetic random
# effect", 2026-08-17). Both builds call themselves 0.7.0.
#
# The consequence is not academic. Reinstalling drmTMB silently changes the
# comparator under every banked number at once. A later re-run that disagrees
# would look exactly like a DRModels.jl regression, and the version string offers no
# way to tell the two apart. AGENTS.md already says to "re-anchor fixtures when
# regenerating against a new installed version" — this makes "which version"
# answerable.
#
# What it prints
# --------------
#   version    the DESCRIPTION Version field (ambiguous on its own — that is the point)
#   built      the DESCRIPTION Built timestamp (the discriminator that actually moved)
#   code_hash  sha256 over the installed namespace's function sources, sorted by
#              name. Independent of build timestamps and file mtimes, so two
#              installs of identical source hash identically.
#   n_objects  how many exported+internal objects went into the hash
#
# Usage:
#   Rscript tools/drmtmb_provenance.R            # human-readable
#   Rscript tools/drmtmb_provenance.R --toml     # paste into expected.meta.toml
#   Rscript tools/drmtmb_provenance.R --check <code_hash>   # exit 1 on mismatch

# The fingerprint itself lives in drmtmb_provenance_lib.R so the parity harnesses
# can stamp it into their evidence without a second copy of the hashing logic.
# Two copies could drift, and a stamp that no longer matches what `--check`
# computes would be a false reassurance rather than provenance.
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])),
                 "drmtmb_provenance_lib.R"))

prov <- drmtmb_provenance()
if (is.null(prov)) {
  cat("ERROR: drmTMB is not installed in this R library.\n", file = stderr())
  quit(status = 2)
}
version   <- prov$version
built     <- prov$built
code_hash <- prov$code_hash
nms       <- seq_len(prov$n_objects)   # only its length is used below

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 2 && args[1] == "--check") {
  expected <- args[2]
  if (identical(expected, code_hash)) {
    cat(sprintf("OK — installed drmTMB matches the recorded build (%s).\n", substr(code_hash, 1, 16)))
    quit(status = 0)
  }
  cat(sprintf(
    paste0(
      "COMPARATOR CHANGED — the installed drmTMB is not the build these numbers were measured against.\n",
      "  recorded  %s\n  installed %s  (version %s, built %s)\n\n",
      "Any disagreement you are about to see may be the COMPARATOR moving, not DRModels.jl.\n",
      "Re-anchor the fixtures deliberately (AGENTS.md) rather than reading this as a regression.\n"
    ),
    substr(expected, 1, 32), substr(code_hash, 1, 32), version, built
  ), file = stderr())
  quit(status = 1)
}

if (length(args) && args[1] == "--toml") {
  cat(sprintf('drmtmb_version = "%s"\n', version))
  cat(sprintf('drmtmb_built = "%s"\n', built))
  cat(sprintf('drmtmb_code_hash = "%s"\n', code_hash))
  quit(status = 0)
}

cat("drmTMB installed-build provenance\n")
cat(sprintf("  version    %s   <- ambiguous on its own; several builds share it\n", version))
cat(sprintf("  built      %s\n", built))
cat(sprintf("  code_hash  %s\n", code_hash))
cat(sprintf("  n_objects  %d\n", length(nms)))
cat(sprintf("  libpath    %s\n", find.package("drmTMB")))
