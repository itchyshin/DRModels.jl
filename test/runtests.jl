using DRModels
using Test, LinearAlgebra, SparseArrays, Random
# --- Deterministic Julia-suite sharding (DRM_TEST_SHARD="k/N") -------------
# Splits the auto-discovered, sharded include calls below across N shards by
# file position: shard k (1-based) gets file i when (i - 1) % N == k - 1.
# Unset or empty means "run everything", so a local `julia test/runtests.jl`
# is unchanged. Helpers live in shard_util.jl and are exercised standalone in
# test_shard_selection.jl, which proves the N shards partition the file list
# exactly -- disjoint AND complete, so sharding cannot silently drop a file.
#
# Deliberately NOT sharded: the JET gate (which has an else-branch) and the
# three parity suites behind DRM_PARITY_TESTS. Those are guarded by their own
# conditions; putting a shard test in front of a conditional would change
# what the guard means.
#
# CONFLICT-FREE REGISTRATION: this file used to hand-list every test_*.jl via
# a literal `_shard_include("...")` call, so almost every PR touched the same
# few lines here (and the matching NEWS.md line -- see news/README.md). Two
# PRs landing close together flipped every OTHER open PR to CONFLICTING on
# GitHub even though `.gitattributes` marks both files `merge=union` --
# GitHub's mergeability check does not honor local/custom merge drivers.
# Adding a new test_*.jl file now needs NO edit to this file at all: it is
# picked up by readdir() below. Only two lists stay hand-maintained, both
# short and rarely touched: _TEST_ORDER (files that must run before the
# general suite) and _TEST_EXCLUDE (files intentionally not run through the
# discovery loop, e.g. because they are included some other way).
include("shard_util.jl")

_shard_spec = get(ENV, "DRM_TEST_SHARD", "")
_SHARD = isempty(_shard_spec) ? nothing : _parse_shard_spec(_shard_spec)

# Dry-run mode: print the shard plan (which file lands in which shard) without
# loading or running anything. `DRM_TEST_LIST_ONLY=1 julia --project=test -e
# 'include("test/runtests.jl")'` (optionally with DRM_TEST_SHARD set too).
const _LIST_ONLY = get(ENV, "DRM_TEST_LIST_ONLY", "0") == "1"

# Files that must run before the general suite (packaging/shape guards) --
# kept explicit and small; everything else is discovered.
const _TEST_ORDER = [
    "test_shard_selection.jl",         # sharding helper self-test
    "test_runtests_include_list.jl",   # this file's own shape (see below)
    "test_load_contract.jl",
    "test_aqua.jl",                    # General-registry hygiene; early so packaging regressions surface first
]

# Files present in test/ that match the test_*.jl discovery pattern but are
# deliberately NOT run through the discovery loop.
const _TEST_EXCLUDE = [
    "test_qgate_jet.jl",           # included manually below, gated on the JET package being present
    "test_analytic_grad.jl",       # #465: investigated, NOT wired -- superseded by test_qgate_fd_gradient.jl
    "test_q4_laplace.jl",          # #465: investigated, NOT wired -- superseded by an obsolete bench POC
    "test_corr_locscale_equiv.jl", # deferred cluster-① follow-up (Laplace-vs-GHQ optimum gap; not a σ-phylo blocker)
]

_discovered = sort(filter(f -> occursin(r"^test_.*\.jl$", f) &&
                                !(f in _TEST_ORDER) && !(f in _TEST_EXCLUDE),
                           readdir(@__DIR__)))

# --- BLAS thread count: pin once, then guard it per file --------------------
# The suite's numerical pins were all measured at one BLAS thread (the repo's
# stated invariant), and ten test files assert or set BLAS=1 themselves. CI
# does not set OPENBLAS_NUM_THREADS, so without this pin a shard starts at 2
# and silently switches to 1 partway through (the first test_joint_missing_*
# file sets it and never restores). Pin it here so every file, on every
# platform, runs at the same count.
_LIST_ONLY || BLAS.set_num_threads(1)

# Test-isolation guard: the BLAS thread count is process-global, so every
# file must leave `BLAS.get_num_threads()` where the suite set it and leave
# no `_with_pinned_blas` scope open. A different count changes the summation
# order inside dense kernels, which perturbs every tight-tolerance numerical
# test that runs after the leak. Checked per file so a failure names the
# leaking file, not a victim far downstream. Measured 2026-09-19 (bisect over
# the full include order): no file leaks today. The in-suite failures of
# test_q4_perf_identities.jl's rtol=1e-8 vcov pins that motivated this guard
# were NOT a leak: `Pkg.test()` runs with `--check-bounds=yes`, and that
# codegen change alone reproduces them bit-for-bit in a pristine session.
const _BLAS_THREADS_AT_START = _LIST_ONLY ? 1 : BLAS.get_num_threads()
function _check_blas_restored(path::AbstractString)
    nt = BLAS.get_num_threads()
    scopes = DRModels._blas_pin_scopes[]
    if nt != _BLAS_THREADS_AT_START || scopes != 0
        @testset "BLAS state restored after $path" begin
            @test nt == _BLAS_THREADS_AT_START
            @test scopes == 0
        end
    end
end

_shard_pos = Ref(0)

function _shard_include(path::AbstractString)
    _shard_pos[] += 1
    _in_shard = _SHARD === nothing || (_shard_pos[] - 1) % _SHARD[2] == _SHARD[1] - 1
    if _LIST_ONLY
        if _in_shard
            shard_label = _SHARD === nothing ? "-" : "$(_SHARD[1])/$(_SHARD[2])"
            println("  [shard $shard_label] $path")
        end
        return
    end
    if _in_shard
        include(path)
        _check_blas_restored(path)
    end
end

_n_test_files = length(_TEST_ORDER) + length(_discovered)
_n_selected = _SHARD === nothing ? _n_test_files :
    length(_shard_indices(_n_test_files, _SHARD[1], _SHARD[2]))
println(_SHARD === nothing ?
    "DRModels tests: all $_n_test_files files (unsharded)" :
    "DRModels tests: shard $(_SHARD[1])/$(_SHARD[2]) - $_n_selected of $_n_test_files files")

if !_LIST_ONLY
    @testset "DRModels.jl — engine loads + phylo foundation" begin
        @testset "public API present" begin
            @test DRModels.DRM === DRModels
            for f in (:fit_q4_sparse_tmb, :marginal_and_exact_grad, :make_problem,
                      :estep_mode, :prior_precision, :augmented_phy,
                      :random_balanced_tree, :sigma_phy_dense, :takahashi_selinv,
                      :lc_metric)
                @test isdefined(DRModels, f)
            end
        end

        @testset "sparse augmented phylo precision (p=8)" begin
            Random.seed!(1); p = 8
            phy = random_balanced_tree(p; branch_length = 0.2)
            Σ = sigma_phy_dense(phy; σ²_phy = 1.0)          # dense leaf covariance
            @test size(Σ) == (p, p)
            @test isposdef(Symmetric(Σ))                    # well-conditioned tree cov
            # kron(Q_cond, Λ⁻¹) prior precision is sparse + PD (the O(p) engine core)
            Λ = Matrix(Symmetric(0.3I(4) + 0.02 * (ones(4, 4) - I(4))))
            keep = setdiff(1:phy.n_total, [phy.root_index])
            P = prior_precision(phy.Q_topology[keep, keep], inv(Λ))
            @test issparse(P)
            @test isposdef(Symmetric(Matrix(P)))
        end
    end
end

# Julia General-registry hygiene (Aqua.jl) + the shape guards run first, so
# packaging regressions surface before the numerical suite.
for f in _TEST_ORDER
    _shard_include(f)
end

# Standing Workflow Q JET gate (Karpinski): type-stability of hot lc↔Λ kernels.
# JET lives in test/Project.toml — skip gracefully when absent (bare
# `julia --project=. test/runtests.jl`). Macro body is in a separate file so it
# is only parsed when JET is present (same pattern as GLLVM.jl). Deliberately
# not part of the discovery loop above (excluded via _TEST_EXCLUDE): it needs
# its own else-branch instead of a plain shard slot.
if _LIST_ONLY
    println("  [unsharded, gated on JET] test_qgate_jet.jl")
else
    const _HAS_JET = Base.find_package("JET") !== nothing
    @testset "Q-gate: JET type-stability (lc_to_Λ / Λ_to_lc)" begin
        if _HAS_JET
            @eval using JET
            include("test_qgate_jet.jl")
        else
            @info "JET not in this environment — run `Pkg.test()` for the Workflow Q JET gate"
            @test_skip false
        end
    end
end

# Every other test_*.jl file in test/, sorted for a deterministic and
# reviewable shard assignment, not hand-listed (see the CONFLICT-FREE
# REGISTRATION note above).
for f in _discovered
    _shard_include(f)
end

# Gated real-parity suite vs committed drmTMB fixtures (off by default).
# Native `drm()` path (#17) plus `drm_bridge` marshalling path (#370).
# Deliberately not part of the discovery loop above: these three live under
# test/parity/ (not top-level test/), so readdir(@__DIR__) never finds them,
# and they are guarded behind DRM_PARITY_TESTS rather than sharded.
if _LIST_ONLY
    println("  [unsharded, gated on DRM_PARITY_TESTS] parity/runparity.jl, parity/runparity_bridge.jl, parity/runparity_bridge_formula.jl")
elseif get(ENV, "DRM_PARITY_TESTS", "0") == "1"
    @testset "R-parity vs drmTMB 0.6.0" begin
        include("parity/runparity.jl")
    end
    @testset "R-parity via drm_bridge vs drmTMB 0.6.0" begin
        include("parity/runparity_bridge.jl")
    end
    @testset "R-parity via drm_bridge R-formula constructs vs drmTMB 0.7.0 (#467)" begin
        include("parity/runparity_bridge_formula.jl")
    end
else
    @info "R-parity suite skipped (set DRM_PARITY_TESTS=1 to run)"
end

# Model comparison + accessor parity (lrtest / anova / aicc / weights / update).
_shard_include("test_comparison.jl")

# Sentinel-loglik fit-level guards (_nondegenerate_fit, lrtest, aic/bic/aicc, lrt_boundary).
_shard_include("test_sentinel_fit_level.jl")

# Chi-bar-square boundary-corrected p-values for variance-component LR tests.
_shard_include("test_chibar.jl")

# #304: lrtest/anova warn on a boundary variance-component drop (naive χ² invalid).
_shard_include("test_lrtest_boundary_warn.jl")

# #639: lrtest refuses REML-vs-ML / mismatched-fixed-effect REML pairs, and the
# REML guard + boundary variance-component label work on bivariate fits.
_shard_include("test_twin_gap_639.jl")

# #320 / #323.2: coeftable/show suppress z/p for non-location blocks and Inf-SE rows.
_shard_include("test_summary_zp_suppress.jl")
_shard_include("test_r2_constant_sigma.jl")

# #325.3: bootstrap summary indexes coefficients by stored block range, not a counter.
_shard_include("test_bootstrap_block_index.jl")

# #313: heritability :profile is a TRUE profile (re-optimises nuisance), not ELR.
_shard_include("test_heritability_true_profile.jl")
_shard_include("test_sentinel_heritability.jl")

# #310: REML-reported Wald vcov includes the restricted-penalty curvature.
_shard_include("test_reml_vcov_curvature.jl")

# Randomized quantile residuals (DHARMa/glmmTMB style) — feat-quantile-residuals.
_shard_include("test_quantile_residuals.jl")
_shard_include("test_twin_gap_760.jl")

# TruncatedNegBinomial2 quantile residuals: no NaN at extreme dispersion / μ.
_shard_include("test_qres_trunc_nan.jl")

# S3: cross-family bivariate (shared-latent GHQ) + link-residual standardization.
_shard_include("test_mixed_family.jl")
# aghq = true: per-observation adaptive GHQ for the shared latent (#719/#834).
_shard_include("test_mixed_family_aghq.jl")
# Post-fit accessors (coef/aic/bic/fitted/summary) for the cross-family fit.
_shard_include("test_mixed_family_postfit.jl")
# Sentinel (1e10) guards: profile CI, bootstrap, AIC/BIC.
_shard_include("test_sentinel_mixed_family.jl")

# Independent validation of the cross-family latent correlation against EXTERNAL
# references: gllvm (Gaussian × Gaussian, identical estimand; guarded — skips if
# the fixture is absent) + an independent Monte-Carlo population reference for the
# genuinely mixed Gaussian × Poisson case + the Gaussian × Gaussian closed form.
_shard_include("test_xfam_external_validation.jl")

# Shared prepared joint missing-predictor likelihood and conditional moments.
_shard_include("test_joint_missing_predictor.jl")
_shard_include("test_joint_missing_two_predictor.jl")
_shard_include("test_joint_missing_finite.jl")
_shard_include("test_joint_missing_uncertainty.jl")
_shard_include("test_joint_missing_frontend.jl")
_shard_include("test_joint_missing_two_frontend.jl")
_shard_include("test_joint_missing_finite_frontend.jl")
_shard_include("test_joint_missing_finite_factor_coding.jl")
_shard_include("test_joint_missing_finite_prediction.jl")
_shard_include("test_joint_missing_bridge.jl")
_shard_include("test_joint_missing_two_bridge.jl")
_shard_include("test_joint_missing_finite_bridge.jl")

# Issue #577: prior_precision dropped exact zeros, so at an exactly diagonal Lambda
# the cross-axis entries of H_uu were structurally absent at non-leaf nodes and the
# Takahashi selected inverse could not supply the logdet-H traces. Guards the root
# fix (structurally full axis block) and the ML exact gradient it silently broke.
_shard_include("test_577_ml_structural_zeros.jl")
_shard_include("test_609_varying_scale.jl")

# Issues #746/#747: Gaussian (1 | g) + sigma ~ x returned logLik ~ +1e44..+1e125
# from catastrophic cancellation in the Woodbury quadratic q1 - q2. Pins the
# cancellation-free `_re_quad_stable` and drmTMB-twinned fits on formerly failing seeds.
_shard_include("test_twin_gap_747.jl")

# #848 (follow-up to #835): data-driven starts + one deterministic restart so
# Gaussian (1 | g) + sigma ~ x reaches drmTMB's interior optimum on the RUNAWAY seeds.
_shard_include("test_twin_gap_747_starts.jl")

# #835 docs regression: `_re_quad_stable` (shared by `_fit_ranef_gaussian` and
# `_fit_ranef_gaussian_lss`) hit `0.0 * Inf == NaN` when a line-search probe drove
# a `sd(g) ~ z` group's sigma_b,k -> 0 exactly, failing LineSearches' finiteness
# assertion on the location-scale-scale REML tutorial. Pins the fix against main.
_shard_include("test_twin_gap_747_lss_reml.jl")

# The same lss REML model on literal data: Julia 1.10 vs 1.13 "different optima"
# were different MersenneTwister draws; the real defect was a -Inf REML objective
# from Woodbury-subtraction garbage in Xmu'V^-1Xmu at a line-search probe (Dual
# Cholesky accepted a zero pivot). Pins the PSD `_re_xtvinvx_stable` form and #835's guard.
_shard_include("test_lss_reml_falseconv.jl")

# Issue #646: on the missing-response Gaussian route `is_converged` read false on a
# genuinely converged fit (the degeneracy bar took std() of a NaN-carrying response,
# and every > against NaN is false), and every bootstrap replicate threw
# DimensionMismatch because simulate drew fit.nobs values against full-design
# means. Guards both, plus the iteration count the full-row rebuild dropped.
_shard_include("test_bridge_response_mask_inference.jl")

# Issue #758: the compact 2-arg `show(io, fit)` hardcoded every family as
# "Gaussian location–scale"; only the MIME"text/plain" method named the real
# family. Guards that the compact form now matches.
_shard_include("test_twin_display_758.jl")
# Issue #668: StatsBase.loglikelihood(::DrmFit) was missing (DRM.loglik worked).
_shard_include("test_twin_display_668.jl")
# Issue #708 / #763: re_sd(fit) returned an empty Dict for a correlated
# random-effect block (1 + x | g); only vc(fit) exposed the SDs/correlation.
_shard_include("test_twin_display_708.jl")
# Issue #759: ranef(fit) silently returned an empty Dict for a non-Gaussian
# GLMM random-intercept fit; the docstring's #73 pointer was also stale.
_shard_include("test_twin_display_759.jl")
# Issues #762/#707: Gaussian correlated (1 + x | g) threw DomainError (log of a
# negative capacitance "determinant") with an uncentred covariate or rho -> +-1, and
# AssertionError in the line search. Pins the stable whitened objective, the
# centred/QR-preconditioned optimisation and drmTMB-twinned H0 refits.
_shard_include("test_twin_gap_762.jl")
# Issue #739 (twin drmTMB #1281): ZeroOneBeta() had no `tree=`/`K=` phylo/relmat
# method. Adds a `Val(:zeroonebeta_fixed)` sparse-Laplace kernel (additive to
# sparse_laplace_glmm.jl) and the phylo/relmat fitters in zeroonebeta.jl.
_shard_include("test_twin_gap_739.jl")
# #857 site K: coevolution prior in whitened coordinates (no Λ⁻¹ near singular Λ);
# also pins the q=4 engine's same-construction defect as @test_broken.
_shard_include("test_q4_prior_whitening.jl")
# Cancellation sweep: multi-RE (1|g)+(1|h), spatial(1|site) and the sparse
# two-structured Gaussian marginals vs a BigFloat dense reassembly as σ_e → 0.
_shard_include("test_cancellation_sweep.jl")
# Issue #732 (twin drmTMB#1272): `re_sd(fit)` on a `phylo(1 | g)` grouping is on
# the raw branch-length scale; drmTMB reports the tip-correlation scale. Guards
# the new `re_sd(fit; scale = :drmtmb, tree = ...)` conversion option (default
# unchanged) against the same two R-oracle fixtures already used elsewhere.
_shard_include("test_twin_gap_732.jl")
# Permanent property test for the "spuriously high logLik at extreme θ" bug
# class: for a discrete response every probability mass is ≤ 1, so a marginal
# logLik must be ≤ 0 too, and no route's objective may ever return NaN or
# -Inf-as-nll. Sweeps every discrete route's fitted θ across an extreme grid.
_shard_include("test_ll_sanity_fuzzer.jl")
# Issue #706 (test part): classic lme4 twins - cbpp (binomial herd RE, integrator-matched
# to glmer nAGQ = 1 / 3 / 25) and sleepstudy (Gaussian random slope, ML vs lmer; REML pinned broken).
_shard_include("test_lme4_twins.jl")

# reml_q4.jl's six naive-inv(Λ) sites (lines 309/437/505-506/516/599/999):
# extreme-regime REML objective vs a 256-bit reference, plus a normal-regime
# identity check against a frozen pre-whitening copy of the same code path.
_shard_include("test_reml_q4_chol.jl")
