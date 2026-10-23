#!/usr/bin/env python3
"""Reconcile the DRModels.jl public surface against drmTMB's, at a named git ref.

The catch-up campaign's countdown. Reads three drmTMB artifacts -- `NAMESPACE`,
`inst/extdata/julia-capabilities.tsv`, `inst/extdata/julia-gates.tsv` -- plus
DRModels.jl's own export block, and reports what the R bridge admits versus what the
Julia engine can actually fit. Reports BOTH directions: what drmTMB exports have
no DRModels.jl twin (the forward "genuinely owed" pass) and what DRModels.jl exports have
no drmTMB twin (the reverse "genuinely ahead" pass, #481) -- so DRModels.jl growing a
capability drmTMB lacks (e.g. #471's structured markers on bivariate LogNormal)
is a written decision instead of unnoticed drift in either direction.

Always reads drmTMB through `git show <ref>:<path>` rather than the working
tree. A working checkout can sit hundreds of commits behind `origin/main`
(2026-08-14: it was 987 behind, and reading it produced a gap list that was
wrong in both directions).

    python3 tools/parity_ledger.py --drmtmb ../drmTMB --ref origin/main
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# drmTMB R name -> DRModels.jl symbol, where the twin exists under a different name.
ALIASES = {
    "nbinom2": "NegBinomial2",
    "truncated_nbinom2": "TruncatedNegBinomial2",
    "truncated_poisson": "TruncatedPoisson",
    "biv_gaussian": "cbind",           # bivariate is cbind()/mvbind() in DRModels.jl
    "drmTMB": "drm",                   # the fitting verb
    "phylo_interaction": "fit_phylo_interaction",
}

# Exports with no meaningful Julia counterpart -- R-idiom helpers, not capability.
NOT_CAPABILITY = {"gr", "drm_control", "meta_known_V"}

# A4e. The raw "exports with no twin" count MIXES three different things, and
# reporting their sum as one countdown overstates the work: it counts a
# capability DRModels.jl already has under another spelling, and work that correctly
# lives in R, as though both were missing engine features.
#
# Each name below therefore carries a WRITTEN REASON, and the countdown reports
# the classes separately. Adding a name here is a claim -- it must be true, and
# it must say why.
DELIBERATELY_NOT_PORTED = {
    # 2026-09-02 (drmTMB #1114): cross-engine objective at a supplied point.
    # DRModels.jl's counterparts are `reml_objective_at` (primitive, #589) and the
    # supported bridge entry `drm_bridge_objective_at` (#590); the name differs
    # by design because the Julia entry takes the bridge payload, not a fit.
    "objective_at": "counterpart is drm_bridge_objective_at / reml_objective_at (DRModels.jl #589/#590)",
    # 2026-09-02 (drmTMB #1114): build provenance baked at install via configure.
    # DRModels.jl records the comparator build with tools/drmtmb_provenance.R --toml
    # (DRModels.jl#473) rather than a runtime accessor; a Julia twin is not owed.
    "drm_provenance": "DRModels.jl stamps comparator provenance via tools/drmtmb_provenance.R --toml (#473)",
    # --- delivered, but spelled differently: a family plus a bivariate formula
    "biv_lognormal": "delivered as LogNormal() with a bivariate formula (A3a, parity-verified)",
    "biv_student": "delivered as Student() with a bivariate formula (A3b, parity-verified)",
    "biv_associate": "delivered as associate_pairs() -- the staged frozen-margin route (A3c)",
    # --- delivered through the BRIDGE PAYLOAD; the R function correctly stays in R.
    #     A2a established that these five collapse to ONE contract: per-dpar
    #     response-scale columns. They are R post-fit functions that CONSUME the
    #     Julia payload, not engine features DRModels.jl is missing.
    "fitted_distribution": "R post-fit function fed by the drm_bridge dpars payload (A2a)",
    "qq_plot": "R post-fit function fed by the drm_bridge dpars payload (A2a)",
    "worm_plot": "R post-fit function fed by the drm_bridge dpars payload (A2a)",
    "centile_chart": "R post-fit function fed by the drm_bridge dpars payload (A2a)",
    "exceedance": "R post-fit function fed by the drm_bridge dpars payload (A2a)",
    # --- delivered as a field/accessor rather than an export
    "rho_latent": "delivered as fit.rho_latent, surfaced through mf_summary()",
    # --- R-side preparation that never reaches the engine (owner-confirmed 2026-08-15)
    "make_mesh": "R-side geospatial prep (sf, CRS validation) before any fit -- owner-confirmed",
    "spatial_coords": "R-side geospatial prep before any fit -- owner-confirmed",
    # --- parked behind an owner fence (#49 missing data)
    "categorical": "an imputation family (drm_impute_family), not a response family -- #49 PARKED",
    "mi": "missing-data surface -- #49 PARKED",
    "impute_model": "missing-data surface -- #49 PARKED",
    "imputed": "missing-data surface -- #49 PARKED",
    "miss_control": "missing-data surface -- #49 PARKED",
    # --- genuinely absent, and blocked for a stated structural reason (A4d)
    "corpair": "BLOCKED: StatsModels' @formula cannot express keyword args or string "
               "literals, so drmTMB's syntax is not representable; and the fitted route "
               "needs the labelled covariance-block grammar (1|p|id), absent in DRModels.jl",
}

# #481. The forward pass above answers "what does drmTMB have that DRModels.jl doesn't".
# It never asks the reverse question, so DRModels.jl can grow capabilities drmTMB has no
# analogue for -- e.g. #471's structured markers on bivariate LogNormal, which
# drmTMB's own source refuses outright (R/drmTMB.R:8998, :9097) -- and the ledger
# stays silent, because silence is all it can express in that direction.
#
# NOTE on scope: this reverse pass is a NAME-level diff, exactly like the forward
# one. It cannot see a capability difference hiding under a NAME that already
# matches -- `LogNormal` already matches R's `lognormal` export directly, so the
# #471 structured-marker advantage sits underneath an already-matched name and
# does not appear as a row below. That is a real limit of a name-keyed instrument,
# not a bug in this pass; it is recorded here rather than faked into a row.
#
# Each name below is DRModels.jl exporting something drmTMB's NAMESPACE has no matching
# symbol for. Most are legitimate and not a capability gap in DRModels.jl's favour: a
# Julia-idiomatic generic name drmTMB reaches via S3method() on an already-global
# base/stats generic (never its own export()), a struct/exception type backing
# what R represents as an implicit S3 class tag, sparse-Laplace engine internals
# never meant to have an R-facing name, or an accessor function for a value R
# exposes as a raw fitted-object field or printed summary() text instead of a
# queryable function. Each entry is a WRITTEN CLAIM -- it must be true.
AHEAD_ACCOUNTED = {
    # --- mirrors a base R / stats generic drmTMB reaches via S3method(), never
    #     its own NAMESPACE export() (verified against drmTMB's S3method() list)
    "aic": "mirrors stats::AIC; drmTMB reaches it via S3method(AIC, drmTMB), never export()",
    "anova": "mirrors stats::anova; drmTMB reaches it via S3method(anova, drmTMB), never export()",
    "bic": "mirrors stats::BIC; drmTMB reaches it via S3method(BIC, drmTMB), never export()",
    "coef": "mirrors stats::coef; drmTMB reaches it via S3method(coef, drmTMB/_julia/_julia_xfam)",
    "confint": "mirrors stats::confint; drmTMB reaches it via S3method(confint, drmTMB/_julia/_julia_xfam)",
    "deviance": "mirrors stats::deviance; drmTMB reaches it via S3method(deviance, drmTMB/_julia)",
    "dof_residual": "StatsAPI naming for stats::df.residual; drmTMB reaches it via S3method(df.residual, ...)",
    "fitted": "mirrors stats::fitted; drmTMB reaches it via S3method(fitted, drmTMB/_julia/_julia_xfam)",
    "loglik": "StatsAPI naming for stats::logLik; drmTMB reaches it via S3method(logLik, drmTMB/_julia/_julia_xfam)",
    "nobs": "mirrors stats::nobs; drmTMB reaches it via S3method(nobs, drmTMB/_julia/_julia_xfam)",
    "predict": "mirrors stats::predict; drmTMB reaches it via S3method(predict, drmTMB/_julia/_julia_xfam)",
    "residuals": "mirrors stats::residuals; drmTMB reaches it via S3method(residuals, drmTMB/_julia)",
    "sigma": "mirrors stats::sigma; drmTMB reaches it via S3method(sigma, drmTMB/_julia)",
    "simulate": "mirrors stats::simulate; drmTMB reaches it via S3method(simulate, drmTMB)",
    "update": "mirrors stats::update; drmTMB reaches it via S3method(update, drm_pair_association)",
    "vcov": "mirrors stats::vcov; drmTMB reaches it via S3method(vcov, drmTMB/_julia/_julia_xfam)",
    "weights": "mirrors stats::weights; drmTMB reaches it via S3method(weights, drmTMB)",
    # --- base-R family constructor, reused directly by drmTMB and never
    #     re-exported (verified: drmTMB.R passes `family = gaussian()/poisson()/
    #     binomial()` straight through to base stats:: constructors)
    "Binomial": "R passes family = stats::binomial() straight through; drmTMB never re-exports it",
    "Gamma": "R passes family = stats::Gamma() straight through; drmTMB never re-exports it",
    "Gaussian": "R passes family = stats::gaussian() straight through; drmTMB never re-exports it",
    "Poisson": "R passes family = stats::poisson() straight through; drmTMB never re-exports it",
    # --- Julia formula-literal / struct / exception type backing what R
    #     represents as a bare `~` literal or an implicit S3 class tag, not an
    #     exported symbol
    "@formula": "StatsModels macro mirroring R's built-in `~` formula literal; base R syntax needs no export",
    "BivariateDrmFormula": "struct backing a bivariate `bf()` formula; R represents the same as an S3 'drm_formula' tag",
    "DrmFit": "struct backing a fitted model; R represents the same as an S3 'drmTMB'/'drmTMB_julia' tag",
    "PairAssociation": "struct backing associate_pairs(); R represents the same as an S3 'drm_pair_association' tag",
    "PhyloCorPenaltyNeedsTwoSD": "Exception type for drm_phylo_penalty()'s two-SD precondition; R signals the same via cli_abort(), not a typed condition export",
    "PhyloPenalty": "struct backing drm_phylo_penalty()'s return object; R represents the same as an S3-classed list",
    # --- sparse augmented-state Laplace / q4 Fisher-z / general-q coevolution
    #     ENGINE INTERNALS: problem construction, packing, and inner-loop math
    #     reached only through the compiled bridge call (or, for the general-q
    #     coevolution block, not reached from R at all -- see the coevolution_*
    #     rows below) -- never meant to carry an R-facing name
    "AugProblem": "sparse augmented-state Laplace problem struct; internal, reached only via drm_bridge",
    "CoevoProblem": "general-q coevolution problem struct (#188); internal, no R bridge call reaches it",
    "aug_prior_grad!": "augmented-state prior-gradient inner-loop helper; engine internal",
    "augmented_phy": "phylogenetic augmented-state constructor; engine internal",
    "augmented_tree_precision": "sparse tree-precision assembly; engine internal",
    "build_Huu": "augmented-state Hessian block assembly; engine internal",
    "coevo_marginal": "general-q coevolution marginal likelihood (#188); internal, no R bridge call reaches it",
    "coevo_marginal_cov": "general-q coevolution marginal covariance (#188); internal, no R bridge call reaches it",
    "coevo_pack": "general-q coevolution parameter packing (#188); internal utility",
    "coevo_theta_len": "general-q coevolution parameter-vector length (#188); internal utility",
    "coevo_unpack": "general-q coevolution parameter unpacking (#188); internal utility",
    "cov_to_lc": "covariance -> log-Cholesky parameterization utility; engine internal",
    "estep_mode": "sparse-EM E-step mode-finder; engine internal",
    "fit_q4_sparse_fisherz": "q4 Fisher-z reparameterized fitter; internal, reached only via the verified fit_q4_sparse_tmb route",
    "fit_q4_sparse_tmb": "verified q4 sparse-Laplace fitter; internal, reached only via drm_bridge, never called by name from R",
    "fz_DRD": "Fisher-z D*R*D covariance reconstruction; engine internal",
    "fz_R": "Fisher-z spherical-correlation Cholesky; engine internal",
    "fz_correlations": "Fisher-z pairwise correlation extraction; engine internal",
    "fz_init_from_Sigma": "Fisher-z parameter initialization from Sigma; engine internal",
    "fz_marginal_and_grad": "Fisher-z marginal likelihood + gradient; engine internal",
    "fz_phi_to_lc": "Fisher-z -> log-Cholesky parameter map; engine internal",
    "gaussian_locscale_phylo_sds": "accessor over DRModels.jl's SEPARATE-vs-COUPLED block representation for sigma~phylo fits; an internal representation detail, not a modelling capability",
    "joint_grad": "augmented-state joint gradient; engine internal",
    "joint_nll": "augmented-state joint negative log-likelihood; engine internal",
    "lc_len": "log-Cholesky parameter-vector length; engine internal",
    "lc_metric": "Fisher/observed-information metric on log-Cholesky params (#13 S1b infra); engine internal",
    "lc_to_cov": "log-Cholesky -> covariance parameterization utility; engine internal",
    "lc_to_Λ": "log-Cholesky -> precision-matrix map; engine internal",
    "Λ_to_lc": "precision-matrix -> log-Cholesky map (inverse of lc_to_Λ); engine internal",
    "make_coevo_problem": "general-q coevolution problem constructor (#188); internal, no R bridge call reaches it",
    "make_coevo_problem_from_covariance": "general-q coevolution problem constructor (#188); internal, no R bridge call reaches it",
    "make_coevo_problem_from_precision": "general-q coevolution problem constructor (#188); internal, no R bridge call reaches it",
    "make_problem": "sparse augmented-state problem constructor; internal, reached only via drm_bridge",
    "marginal_and_exact_grad": "exact O(p) marginal likelihood + gradient; internal, reached only via drm_bridge",
    "marginal_nll": "marginal negative log-likelihood; engine internal",
    "pack_theta": "augmented-state parameter packing; engine internal",
    "phylo_correlation": "tree branch-length -> correlation matrix utility; engine internal (adjacent to sigma_phy_dense)",
    "phylo_interaction_nll": "likelihood piece backing fit_phylo_interaction (already matched via ALIASES); engine internal",
    "phylo_tree_height": "tree-utility accessor; engine internal",
    "prior_precision": "augmented-state prior precision assembly; engine internal",
    "random_balanced_tree": "synthetic phylogenetic tree generator for tests/benchmarks; engine internal",
    "random_caterpillar_tree": "synthetic phylogenetic tree generator for tests/benchmarks; engine internal",
    "sigma_phy_dense": "dense phylogenetic covariance construction (small-tree/testing path); engine internal",
    "takahashi_selinv": "Takahashi selected-inverse sparse linear algebra; engine internal",
    "unpack_theta": "augmented-state parameter unpacking; engine internal",
    # --- reached dynamically by R's own bridge, not through a matching R-level
    #     export name (verified: R/julia-bridge.R calls these via
    #     JuliaCall::julia_call("drmTMB_drm_bridge"[_inference], ...))
    "drm_bridge": "the function R's engine=\"julia\" bridge calls via JuliaCall::julia_call(\"drmTMB_drm_bridge\", ...) -- this IS the bridge, reached dynamically rather than by a matching R export",
    "drm_bridge_inference": "the function R's confint()/profile() bridge calls via JuliaCall::julia_call(\"drmTMB_drm_bridge_inference\", ...) -- reached dynamically, not by a matching R export",
    # --- cross-family (`fit_mixed_family`) post-fit accessors: R dispatches the
    #     SAME base generics (coef/AIC/BIC/fitted/summary) via the
    #     drmTMB_julia_xfam S3 class; Julia's method-per-name convention needs a
    #     distinct symbol per generic for the same underlying values
    "mf_aic": "R reaches this via S3method(AIC, drmTMB_julia_xfam); Julia needs a distinct name for the same generic",
    "mf_bic": "R reaches this via S3method(BIC, drmTMB_julia_xfam); Julia needs a distinct name for the same generic",
    "mf_coef": "R reaches this via S3method(coef, drmTMB_julia_xfam); Julia needs a distinct name for the same generic",
    "mf_fitted": "R reaches this via S3method(fitted, drmTMB_julia_xfam); Julia needs a distinct name for the same generic",
    "mf_summary": "R reaches this via S3method(summary, drmTMB_julia_xfam); Julia needs a distinct name for the same generic",
    # --- plotting / raw-data feeders for the Makie backend, paralleling an
    #     already-ported R plot_*() consumer or S3-dispatched plot() method
    "corpairs_data": "data feeder for the Makie corpairs plot; R's plot_corpairs() (already ported) is a ggplot2 consumer of the same values",
    "drm_figure": "Makie plotting dispatcher (#336); R's ggplot2 equivalents are the already-ported plot_corpairs()/plot_parameter_surface() directly",
    "parameter_surface": "convenience wrapper combining prediction_grid() + predict_parameters() (both already ported) into one call",
    "plot_profile": "Makie wrapper mirroring R's S3method(plot, profile.drmTMB); R dispatches via base plot(), not a separate name",
    "profile_curve": "named accessor for what R's profile.drmTMB() computes internally via the private drm_profile_curve() helper and returns as part of the S3 object",
    "profile_result": "named accessor for the object R's profile() generic (S3method(profile, drmTMB)) already returns",
    # --- StatsAPI accessor function for a value R exposes as a raw fitted-object
    #     field or as printed summary()/coeftable() text, not a separate function
    "coeftable": "StatsAPI generic for the coefficient table; R prints the same via summary.drmTMB/print.drmTMB, no separate accessor",
    "dof": "StatsAPI generic for fixed-effect parameter count; drmTMB computes this internally for AIC/BIC without exposing an accessor",
    "estimation_method": "accessor for what R exposes as the raw fit$REML/fit$estimator field",
    "family": "StatsAPI accessor for the fitted family object; R never queries it back -- family is a formula-time argument only",
    "lrtest": "convenience wrapper equivalent to R's anova(fit1, fit2); drmTMB has no separately named lrtest()",
    "ml_loglik": "accessor for what R exposes as the raw fit$REML field alongside logLik.drmTMB's single value",
    "niterations": "StatsAPI generic for optimizer iteration count; R's TMB-based fits keep this as an internal fit$opt$iterations field, not an accessor",
    "re_sd": "random-effect SD accessor; the same values are in drmTMB's printed summary() random-effects block, not a queryable function",
    "reml_loglik": "accessor for what R exposes as the raw fit$REML field alongside logLik.drmTMB's single value",
    "stderror": "StatsAPI generic for per-parameter SEs; R surfaces the same values as a std.error column in coeftable()/predict_parameters() output",
    "vc": "variance-components accessor; drmTMB surfaces the same values through printed summary() output, not a queryable function",
    # --- drmTMB keeps this as a field on its own fitted object, not a function
    "integration_diagnostics": "drmTMB retains the same quadrature abs.error as a field on its own association object, not a separate accessor function",
    # --- exists BECAUSE an R capability stays parked, not ahead of it
    "drm_listwise": "listwise-deletion preprocessing stand-in while the full drmTMB missing-data surface (#49: categorical/mi/impute_model/imputed/miss_control) stays parked -- a stopgap, not an advantage",
}


def git_show(repo: Path, ref: str, path: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        sys.exit(f"cannot read {path} at {ref} in {repo}: {out.stderr.strip()}")
    return out.stdout


def norm(s: str) -> str:
    return re.sub(r"[_.]", "", s).lower()


def drmtmb_exports(repo: Path, ref: str) -> list[str]:
    ns = git_show(repo, ref, "NAMESPACE")
    return sorted(re.findall(r"^export\((.+?)\)", ns, re.M))


def drmjl_exports(root: Path) -> list[str]:
    src = (root / "src" / "DRModels.jl").read_text()
    names: list[str] = []
    for block in re.findall(r"^export\s+(.+?)(?=\n\s*\n|\nexport|\Z)", src, re.M | re.S):
        block = re.sub(r"#.*", "", block)
        names += [n.strip() for n in block.split(",") if n.strip()]
    return sorted(set(names))


def tsv_rows(repo: Path, ref: str, path: str) -> list[dict]:
    text = git_show(repo, ref, path)
    lines = [l for l in text.splitlines() if l.strip()]
    header = lines[0].split("\t")
    return [dict(zip(header, l.split("\t"))) for l in lines[1:]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drmtmb", required=True, type=Path, help="path to the drmTMB repo")
    ap.add_argument("--ref", default="origin/main", help="git ref to read (default origin/main)")
    ap.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    args = ap.parse_args()

    sha = subprocess.run(
        ["git", "-C", str(args.drmtmb), "rev-parse", args.ref],
        capture_output=True, text=True,
    ).stdout.strip()
    version = re.search(
        r"^Version:\s*(\S+)", git_show(args.drmtmb, args.ref, "DESCRIPTION"), re.M
    ).group(1)

    r_exports = drmtmb_exports(args.drmtmb, args.ref)
    j_exports = drmjl_exports(args.root)
    j_norm = {norm(x) for x in j_exports}

    def has_twin(name: str) -> bool:
        if norm(name) in j_norm:
            return True
        alias = ALIASES.get(name)
        return bool(alias and norm(alias) in j_norm)

    unmatched = [x for x in r_exports if not has_twin(x) and x not in NOT_CAPABILITY]
    # A4e: split the raw count into "actually owed" vs "accounted for in writing".
    missing = [x for x in unmatched if x not in DELIBERATELY_NOT_PORTED]
    accounted = [x for x in unmatched if x in DELIBERATELY_NOT_PORTED]

    # #481: the reverse pass. Same has_twin() logic, direction flipped -- reuses
    # ALIASES (r_name -> j_name) by checking the ALIASES *values*, so a DRModels.jl
    # export spelled to match a known R alias (e.g. NegBinomial2 for nbinom2)
    # counts as twinned without duplicating the map.
    r_norm = {norm(x) for x in r_exports}
    alias_j_norm = {norm(v) for v in ALIASES.values()}

    def has_r_twin(name: str) -> bool:
        n = norm(name)
        return n in r_norm or n in alias_j_norm

    ahead_unmatched = [x for x in j_exports if not has_r_twin(x)]
    ahead_missing = [x for x in ahead_unmatched if x not in AHEAD_ACCOUNTED]
    ahead_accounted = [x for x in ahead_unmatched if x in AHEAD_ACCOUNTED]

    caps = tsv_rows(args.drmtmb, args.ref, "inst/extdata/julia-capabilities.tsv")
    gates = tsv_rows(args.drmtmb, args.ref, "inst/extdata/julia-gates.tsv")

    print(f"drmTMB {version} @ {args.ref} ({sha[:9]})")
    print(f"  exports: {len(r_exports)}   DRModels.jl exports: {len(j_exports)}")
    print()

    print(f"BRIDGE CAPABILITY ROWS ({len(caps)}) -- the campaign's countdown")
    by_claim: dict[str, list[str]] = {}
    for row in caps:
        by_claim.setdefault(row.get("claim_status", "?"), []).append(row["capability_id"])
    for claim, ids in sorted(by_claim.items()):
        print(f"  {claim:<14} {len(ids):>2}  {', '.join(ids)}")
    print()

    blocked = [g for g in gates if g.get("r_bridge_status") == "intentional_error"]
    print(f"GATES CLOSED BY INTENTIONAL ERROR ({len(blocked)})")
    for g in blocked:
        print(f"  {g['gate_id']:<34} {g.get('syntax','')[:70]}")
    print()

    print(f"ACCOUNTED FOR IN WRITING ({len(accounted)}) -- not owed, and why")
    for name in accounted:
        print(f"  {name:<22} {DELIBERATELY_NOT_PORTED[name]}")
    print()

    print(f"drmTMB EXPORTS WITH NO DRModels.jl TWIN ({len(missing)}) -- genuinely owed")
    for name in missing:
        print(f"  {name}")
    print()

    # #481: the reverse direction. Same shape as the forward blocks above --
    # DRModels.jl exports with no drmTMB twin, split into what's accounted for in
    # writing versus what's a genuine, unaccounted divergence.
    print(f"AHEAD OF drmTMB, ACCOUNTED FOR IN WRITING ({len(ahead_accounted)}) -- not a gap, and why")
    for name in ahead_accounted:
        print(f"  {name:<28} {AHEAD_ACCOUNTED[name]}")
    print()

    print(f"DRModels.jl EXPORTS WITH NO drmTMB TWIN ({len(ahead_missing)}) -- genuinely ahead")
    for name in ahead_missing:
        print(f"  {name}")
    print()
    # Report the real distribution against the GOVERNING vocabulary, which
    # drmTMB's docs/design/168-r-julia-finish-capability-matrix.md defines as:
    #   covered > partial > experimental > planned > unsupported
    #
    # This line used to read `sum(... claim_status != 'supported')` and print it as
    # "N unsupported capability rows". `supported` is not a value in that
    # vocabulary and never has been, so the condition was true for every row: the
    # number was len(caps) by construction and could never move, no matter how much
    # evidence landed. It also labelled `partial` and `experimental` rows
    # "unsupported", a word the vocabulary reserves for "deliberately rejected or
    # out of scope" -- which describes exactly one row (engine_control_surface).
    # That phrasing had propagated into the handover, the coordination board and
    # Mission Control as though it were a measurement.
    order = ["covered", "partial", "experimental", "planned", "unsupported"]
    tally = {k: 0 for k in order}
    other: dict[str, int] = {}
    for c in caps:
        st = (c.get("claim_status") or "").strip()
        if st in tally:
            tally[st] += 1
        else:
            other[st or "<blank>"] = other.get(st or "<blank>", 0) + 1
    dist = " · ".join(f"{tally[k]} {k}" for k in order if tally[k])
    if other:
        dist += " · " + " · ".join(f"{v} {k}?" for k, v in sorted(other.items()))

    print(f"COUNTDOWN: {len(missing)} export gaps ({len(unmatched)} raw, "
          f"{len(accounted)} accounted for) · "
          f"{len(ahead_missing)} ahead-of-drmTMB ({len(ahead_accounted)} accounted for) · "
          f"{len(caps)} capability rows [{dist}] · "
          f"{len(blocked)} closed gates")

    # Closure invariant: every row is either `covered` (the top rung, whose
    # evidence is audited separately) or says IN WRITING why it is not. Asserting
    # this by hand rots the moment someone adds a row, so check it and fail loudly.
    #
    # Note this is UNCHANGED in strictness: the old code exempted `supported`,
    # a value that never occurs, so the exemption never fired and every row needed
    # a claim_boundary. No row is `covered` today either, so the same rows are
    # checked as before -- but the exemption now names a status that can actually
    # be reached.
    print()
    unclosed: list[str] = []
    for row in caps:
        if row.get("claim_status") == "covered":
            continue                      # fixture evidence is audited separately
        if not row.get("claim_boundary", "").strip():
            unclosed.append(f"capability {row['capability_id']}: no claim_boundary")
    for g in blocked:
        if not g.get("evidence", "").strip():
            unclosed.append(f"gate {g['gate_id']}: no evidence")
        if not g.get("review_due", "").strip():
            unclosed.append(f"gate {g['gate_id']}: no review_due")

    if unclosed:
        print(f"CLOSURE: FAIL — {len(unclosed)} row(s) neither `covered` nor bounded")
        for u in unclosed:
            print(f"  {u}")
        return 1

    print(f"CLOSURE: PASS — every one of {len(caps)} capability rows is `covered` "
          f"or carries a written claim_boundary; all {len(blocked)} closed gates "
          f"carry evidence + review_due")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
