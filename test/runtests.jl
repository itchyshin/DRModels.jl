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

# Julia General-registry hygiene (Aqua.jl): deps_compat, stale deps, exports,
# project-extras, unbound args, piracy. Runs early so packaging regressions
# surface before the numerical suite.
_shard_include("test_shard_selection.jl")
_shard_include("test_runtests_include_list.jl")  # this file's own shape: no duplicate or plain includes (see MAINTENANCE NOTE)
_shard_include("test_load_contract.jl")
_shard_include("test_aqua.jl")

# Gaussian location–scale front end (drm/bf public API).
_shard_include("test_gaussian_core.jl")
# REML estimation (opt-in, fixed-effect Gaussian location–scale) + the
# model-selection guard for the classic REML trap (issue #11). Placed early so it
# runs near the core Gaussian tests.
_shard_include("test_reml.jl")
_shard_include("test_reml_ordinary_ranef.jl")  # #445 Option A: #439/#440 Gaussian mean (1 | g) REML
_shard_include("test_niterations.jl")  # #466: fit.niterations wired into every iterative family fitter
_shard_include("test_bf_grammar.jl")
_shard_include("test_gaussian_bivariate.jl")
_shard_include("test_bivariate_lognormal.jl")
_shard_include("test_bivariate_student.jl")
_shard_include("test_associate_pairs.jl")
_shard_include("test_gaussian_bivariate_phylo.jl")
_shard_include("test_gaussian_bivariate_q4_structured.jl")
_shard_include("test_missing_response_bivariate.jl")   # #19: per-cell missing on the bivariate q=4 phylo engine
_shard_include("test_missing_response.jl")
_shard_include("test_missing_response_nongaussian.jl")
_shard_include("test_lowbatch_j7.jl")                  # #322–#326 low-severity twin-review fixes
_shard_include("test_sigma_phylo_missing_guard.jl")
_shard_include("test_coevo_accessors.jl")
# General-q coevolution block (#188): the among-trait covariance Λ generalised
# from the q=4 PLSM to q=6/q=8 on the same sparse kron(Q_tree, Λ⁻¹) precision.
_shard_include("test_coevo_q6.jl")
_shard_include("test_corpairs.jl")
_shard_include("test_gaussian_ranef.jl")
_shard_include("test_lss_group.jl")   # #544 location-scale-scale sd(g) ~ x
_shard_include("test_lss_phylo.jl")   # #545 sd_phylo + #548 cancellation regression
_shard_include("test_lss_tip_identity.jl") # Named tree-tip mapping under shuffled rows
_shard_include("test_lss_bootstrap_contract.jl") # Marginal components, masks and REML refits
_shard_include("test_bootstrap_thread_flags.jl") # Independent storage for parallel status flags
_shard_include("test_lss_sparse.jl")  # #551 O(p) sparse exact marginal LSS engine
_shard_include("test_lss_sparse_gradient_scaling.jl")  # #627 O(p) gradient + profile endpoint invariance
_shard_include("test_lss_sparse_multi.jl")  # #563 S7b.1 sparse multi-component block assembly + objective
_shard_include("test_lss_sparse_multi_gradient.jl")  # #563 S7b.2/S7b.2b sparse multi-component exact gradient
_shard_include("test_lss_sparse_multi_reml.jl")  # #563 S7b.3 sparse multi-component REML correction + gradient
_shard_include("test_lss_sparse_multi_public.jl")  # #563 S7b.4 public drm() route wiring (D-206 router)
_shard_include("test_sparse_precision_storage.jl")  # S5a (#563): no dense copy of the augmented tree precision
_shard_include("test_lsss_multi.jl")  # #555 multi-component sd() (lsss)
_shard_include("test_lss_reml.jl")    # #558 location-scale-scale REML
_shard_include("test_lss_missing_response.jl") # #559 location-scale-scale missing responses
_shard_include("test_inference.jl")
_shard_include("test_inference_blas_pinning.jl") # Nested and overlapping global-BLAS restoration
_shard_include("test_profile_ci.jl")
_shard_include("test_profile_nuisance_status.jl")
_shard_include("test_profile_acceptance_oracles.jl")
_shard_include("test_bridge_profile_status.jl")
_shard_include("test_profile_infinite_bound.jl")  # #631 no infinite bound from a failed endpoint
_shard_include("test_check_drm.jl")
_shard_include("test_bias_correct.jl")
_shard_include("test_visualization.jl")
_shard_include("test_makie_ext_stub.jl")   # #336: DRModelsMakieExt method-less stub (Makie OUT of CI)
_shard_include("test_postfit.jl")
_shard_include("test_meta.jl")
_shard_include("test_simulate.jl")
_shard_include("test_simulate_scale_conventions.jl") # NB2 size, Gamma slot conventions, owned auxiliary draws
_shard_include("test_locscale_bootstrap_simulator.jl") # coupled marginal draws, precision and family contracts
_shard_include("test_locscale_bootstrap_refit.jl") # same-seed public Gamma refits, serial/threaded
_shard_include("test_gaussian_structured.jl")
_shard_include("test_gaussian_phylo_mean_missing_response.jl")  # #482: species-subset (drop) + include refusal
# Silent-data-loss fix: `phylo(<not 1> | group)` refused on univariate routes
# instead of silently fitting the intercept-only model.
_shard_include("test_phylo_slope_refusal.jl")
_shard_include("test_phylo_slope_two_sd.jl")
_shard_include("test_phylo_interaction.jl")
_shard_include("test_two_structured_gaussian.jl")
_shard_include("test_two_structured_gaussian_sparse.jl")
_shard_include("test_heritability.jl")
_shard_include("test_conjugate_em.jl")
_shard_include("test_location_only_reml_mme.jl")
_shard_include("test_bootstrap.jl")
_shard_include("test_bootstrap_marginal.jl")   # #459: bootstrap must redraw random effects
_shard_include("test_gaussian_spatial.jl")
_shard_include("test_predict.jl")
_shard_include("test_predict_response.jl")
_shard_include("test_ranef.jl")
_shard_include("test_ranef_varying_scale_convergence.jl")  # #609: `converged` on the varying-scale ranef route is the GRADIENT criterion
_shard_include("test_correlated_re.jl")
_shard_include("test_multi_re.jl")
_shard_include("test_sigma_re.jl")
_shard_include("test_sigma.jl")
_shard_include("test_student.jl")
_shard_include("test_skewnormal.jl")
_shard_include("test_poisson.jl")
_shard_include("test_nbinom2.jl")
_shard_include("test_nb2_dispersion_seed.jl")
_shard_include("test_beta.jl")
_shard_include("test_gamma.jl")
# eta-clamp twin parity (#324): the NB2/Gamma/Beta density guards now use the
# ported soft-clamp (identity in the band, smooth beyond) so DRModels.jl agrees with
# drmTMB instead of hard-clamping the mean/scale predictors.
_shard_include("test_eta_clamp_parity.jl")
_shard_include("test_zi.jl")
# The `zi`/`hu` count mixtures through the drm_bridge MARSHALLING boundary --
# the route drmTMB's engine = "julia" crosses, and the one its banked
# zi_poisson parity receipt rests on. `test_zi.jl` above covers native drm only.
_shard_include("test_bridge_zi.jl")
_shard_include("test_lognormal.jl")
_shard_include("test_hurdle.jl")
_shard_include("test_truncated_nb.jl")
_shard_include("test_betabinomial.jl")
_shard_include("test_zeroonebeta.jl")
_shard_include("test_tweedie.jl")
_shard_include("test_tweedie_ranef.jl")
_shard_include("test_cumulative.jl")
_shard_include("test_cumlogit_ranef.jl")
_shard_include("test_cumlogit_phylo.jl")
_shard_include("test_poisson_re.jl")
_shard_include("test_poisson_slope_re.jl")
_shard_include("test_poisson_crossed_laplace.jl")
_shard_include("test_poisson_phylo_laplace.jl")
_shard_include("test_relmat_counts.jl")
_shard_include("test_relmat_counts_nb2.jl")
_shard_include("test_relmat_counts_beta.jl")
_shard_include("test_spatial_coord_poisson.jl")
_shard_include("test_nb2_phylo_laplace.jl")
_shard_include("test_gamma_beta_phylo_laplace.jl")
_shard_include("test_binomial_phylo_laplace.jl")
_shard_include("test_betabinomial_phylo_laplace.jl")
_shard_include("test_crossed_laplace_generic.jl")
_shard_include("test_betabinomial_crossed_laplace.jl")
_shard_include("test_crossed_selected_inverse.jl")
_shard_include("test_locscale_kernels.jl")
_shard_include("test_locscale_inner.jl")
_shard_include("test_locscale_inner_status.jl")
_shard_include("test_locscale_marginal.jl")
_shard_include("test_locscale_fit.jl")
_shard_include("test_locscale_grad.jl")
_shard_include("test_locscale_precision_derivatives.jl")
_shard_include("test_locscale_compensated_gradient.jl")
_shard_include("test_locscale_whitened.jl")
_shard_include("test_locscale_infer.jl")
_shard_include("test_locscale_profile.jl")
_shard_include("test_locscale_profile_status.jl")
_shard_include("test_locscale_profile_threads.jl") # Finite canonical intervals and owned coefficient jobs
# Optimizer / mode-accuracy robustness for the module-wired locscale + sparse-aug
# paths: NaN (not zero) outer gradient on inner-mode failure (#314), tightened
# fast E-step acceptance gate (#317), and Zη/Zψ threading through the profiler
# (#325.4).
_shard_include("test_optimizer_robustness.jl")
_shard_include("test_locscale_gamma_e2e.jl")
_shard_include("test_locscale_phylo_e2e.jl")
_shard_include("test_locscale_frontend.jl")
# cluster ① (correlated/independent slopes rerouted onto the q2 locscale core) +
# the structured-slope locscale path.
_shard_include("test_locscale_structured.jl")
# NOTE (cluster-① follow-up): test_corr_locscale_equiv.jl (Laplace≈GHQ cross-engine
# check) is deferred from this σ-phylo landing. For Poisson (1+x|g) the two engines
# AGREE in loglik to 0.02% (Laplace −4651.4 vs GHQ −4652.4) but the fixed-effect
# coefficients + RE covariance differ beyond the branch's rtol — the GHQ-vs-Laplace
# optimum gap on a flat Poisson surface, against merged-main's GHQ reference (which
# differs from the branch's). Re-anchoring that reference is a cluster-① task, not a
# σ-phylo blocker. The capability itself ships (test_locscale_structured passes).
# include("test_corr_locscale_equiv.jl")
# cluster ② standalone non-Gaussian σ-axis RE (sigma ~ 1 + (1|g)) via
# _fit_sigma_axis_re — guards the inner grad! keyword contract + Gamma recovery.
_shard_include("test_sigma_axis_re.jl")
# Non-constant dispersion (sigma ~ x) SIMULTANEOUSLY with a random effect for
# non-Gaussian families (#164): recovery of the dispersion slope + σ-axis RE
# covariance via the location–scale engine, an FD-vs-exact gradient gate, and a
# guard pinning the still-open mean-RE-only Laplace sub-case.
_shard_include("test_nonconst_sigma_re.jl")
_shard_include("test_nbinom2_slope_re.jl")
_shard_include("test_beta_slope_re.jl")
_shard_include("test_gamma_slope_re.jl")
_shard_include("test_nbinom2_re.jl")
_shard_include("test_beta_re.jl")
_shard_include("test_gamma_re.jl")
_shard_include("test_student_re.jl")
_shard_include("test_student_slope_re.jl")
_shard_include("test_lognormal_re.jl")
_shard_include("test_lognormal_slope_re.jl")
_shard_include("test_lognormal_structured_mean.jl")
_shard_include("test_betabinomial_re.jl")
_shard_include("test_betabinomial_slope_re.jl")
_shard_include("test_binomial.jl")
_shard_include("test_binomial_re.jl")
_shard_include("test_summary.jl")
_shard_include("test_bootstrap_nongaussian.jl")
_shard_include("test_bootstrap_nongaussian_structured.jl")   # #479: K/A/tree threaded through refit
_shard_include("test_bootstrap_formula_structured.jl")   # #480: same fix on the formula-based surface
_shard_include("test_aic_bic.jl")
_shard_include("test_vcov_guard.jl")
_shard_include("test_variational.jl")
_shard_include("test_va_poisson_elbo.jl")
_shard_include("test_va_frontend_poisson.jl")
_shard_include("test_va_frontend_families.jl")
_shard_include("test_variational_binomial.jl")
_shard_include("test_variational_nb2.jl")
_shard_include("test_variational_gamma.jl")
_shard_include("test_aghq_1d.jl")                  # #448: 1-D Liu–Pierce AGHQ (lever 2)
# Numerical-stability guards from the twin code-review pass (#303/#308/#311/#312/
# #319/#321/#324.6/#324.7): SD-collapse, coincident coords, VA inner damping,
# scale-aware FD Hessian, and PD-prior Cholesky barriers.
_shard_include("test_numerical_guards.jl")
_shard_include("test_family_accessor.jl")
_shard_include("test_parity_accessors.jl")
_shard_include("test_coverage_accessors.jl")  # anchors fixef + engine joint_nll/joint_grad/build_Huu/unpack_theta
_shard_include("test_rho12_accessor.jl")
_shard_include("test_summary_method.jl")
_shard_include("test_predict_parameters.jl")
_shard_include("test_prediction_grid.jl")
_shard_include("test_bridge.jl")
_shard_include("test_bridge_option_passthrough.jl")  # #527-adjacent: control-option forwarding + gradient exposure
_shard_include("test_bridge_bootstrap_tree.jl") # same-tree non-Gaussian fixed-effect bootstrap
_shard_include("test_bootstrap_provider_forwarding.jl") # K/A/tree/coords survive bootstrap refits
_shard_include("test_bridge_profile_target.jl")
_shard_include("test_bridge_q2_direct_export.jl")
_shard_include("test_bridge_q4_direct_export.jl")
# Missing-data handling (#49): documents that raw missing/NaN responses ERROR
# (no silent garbage), and anchors the listwise-deletion preprocessing path
# (drm_listwise) + MAR recovery. FIML / imputation remain follow-up (#49).
_shard_include("test_missing_listwise.jl")

# Deepened coverage of genuinely-untested exported engine entry points
# (fit_q4_sparse_tmb end-to-end; marginal_nll / marginal_and_exact_grad return
# contract + cross-consistency) and the bivariate bf() meta_V/relmat/animal
# constructor guard rails.
_shard_include("test_coverage_engine.jl")
_shard_include("test_q4_objective_diagnostic.jl")
_shard_include("test_bridge_formula_translation.jl")
_shard_include("test_bridge_materialization_collision.jl")
_shard_include("test_bridge_formula_labels.jl")
_shard_include("test_bridge_base_r_names.jl")  # #563/#467: the ten design-258 constructs render base-R names
_shard_include("test_bridge_coef_labels_echo.jl")  # #563: options["coef_labels"] echo (design 258 §7.1-7.3)
_shard_include("test_bridge_formula_constructs.jl")  # #467/#609 A6: R-contrast fidelity of the coef_labels echo
_shard_include("test_bridge_zi_marginal_mean.jl")  # bridge fitted/residuals = drmTMB's unconditional mean for zi count fits
_shard_include("test_bridge_lss_labels.jl")
_shard_include("test_bridge_lss_routes.jl")  # #563 S6: bridge-vs-direct parity across every LSS route
# Bridge inference for the bivariate q4 σ-phylo fit: among-axis SD CIs via bootstrap
# (multi-row payload) + the profile→bootstrap redirect (Ayumi #2 uncertainty-via-R).
_shard_include("test_bridge_bivariate_inference.jl")

# Profile/bootstrap intervals on a FIXED-EFFECT target of the residual-only
# bivariate Gaussian fit (drmTMB A8b): the q4 file above covers the among-axis
# SDs of the STRUCTURED bivariate route; this one covers the residual route's
# mu1/mu2/sigma/rho12 coefficients, which had no interval target at all.
_shard_include("test_bridge_biv_inference.jl")

# Poc-migrated foundation checks (#465): the poc's script-style includes and
# hardcoded paths were fixed to run in-suite. `test_step1_sparse` cross-checks
# the ported sparse Newick/Takahashi infra against real R (ape::vcv) fixtures;
# `test_sparse_aug` is Checkpoint 3 (augmented sparse Laplace == dense leaf-only
# oracle); `test_lambda_direction` checks the sparse-EM Λ M-step direction
# ascends the true marginal (mstep_Lambda/fit_em_aug back the sparse_em_fit.jl
# demos, off the public `drm()` path, and had no other coverage).
_shard_include("test_step1_sparse.jl")
_shard_include("test_takahashi_selinv_identity_pilot.jl")  # lane C: H² SIMD selinv identity, one banded cell
_shard_include("test_sparse_aug.jl")
_shard_include("test_lambda_direction.jl")

# Julia-side standing gate for the phylo_count_large_p capability row: the row's
# own boundary named the absence of one as its limitation (the harness was R-side
# only). Needs no R and no fixtures. Also round-trips re_sd across tree heights,
# which is the raw-vs-normalised covariance trap.
_shard_include("test_phylo_count_largep_gate.jl")

# NOTE (HANDOVER step, #465 remainder): test_analytic_grad.jl and
# test_q4_laplace.jl were investigated and NOT wired — see the #465 after-task
# note for why (superseded by test_qgate_fd_gradient.jl / obsolete bench POC).
# test_lambda_p100.jl was WIRED as a #472 characterisation of a measured defect
# (mstep_Lambda descends the true marginal at p=100) so that a future repair
# would trip it loudly and have to revisit the fence. On 2026-09-02 it tripped:
# the descent was the #577 sparsity-pattern artefact. The file now asserts the
# repaired ASCENT and stays a tripwire in the other direction. #472 itself stays
# OPEN — the synthetic pure-noise p=100 case still descends, unexplained.
_shard_include("test_lambda_p100.jl")

# Always-on R-parity HARNESS smoke test (machinery only, no R, no fixtures).
# Placed at the END to avoid colliding with other in-flight branches' includes.
# API freeze gate, v0.7 line (Wave A, D-181): total classification of every export.
_shard_include("test_api_stability.jl")
_shard_include("test_parity_harness.jl")
_shard_include("test_parity_biv_q4_phylo_reml.jl")       # #445 Option A: #433/#434 same-target fixture
_shard_include("test_parity_gaussian_phylo_mean.jl")     # #445 Option A: #437/#438 Route A fixture
_shard_include("test_q4_reml_warm_restart.jl")           # #484: public drm() converges the q4 phylo REML cell
_shard_include("test_reml_objective_at.jl")              # #575: objective-at-point diagnostic primitive
_shard_include("test_bridge_objective_at.jl")             # #563: drm_bridge_objective_at (bridge-boundary reml_objective_at)
_shard_include("test_575_exact_reml_gradient.jl")        # #575: exact REML gradient vs a tight central difference
_shard_include("test_575_q4_optimum.jl")                 # #575: the q4 REML cold-start route reaches its own optimum
_shard_include("test_reml_q4_missing_response.jl")       # #578: _reml_border_blocks mask consistency, missing responses
_shard_include("test_q4_reml_vcov.jl")                   # #563 S11: pin q4 REML vcov() native/bridge behaviour
_shard_include("test_reml_prior_precision_collapse.jl")  # #563: _reml_prior_precision collapsed into prior_precision after #577
_shard_include("test_reml_surface_contract.jl")          # #624: bridge estim_method/loglik honesty + refusal message surface
_shard_include("test_reml_reml_biv_residual.jl")        # #624: REML on the residual-only bivariate Gaussian route

# Delta-method prediction standard errors (feat-predict-se).
_shard_include("test_predict_se.jl")

# Standing Q-gate (issue #14): FD-vs-exact gradient check ≤ 1e-6 for the verified
# q4 sparse-Laplace engine (Workflow Q).
_shard_include("test_qgate_fd_gradient.jl")

# Standing engine-quality Q-gate (issue #15): zero-allocation gate on the inner
# Newton mode-finder's pure-Julia arithmetic (the CHOLMOD factor is excluded as
# out-of-Julia-control). Cheap → per-PR. (Workflow Q.)
_shard_include("test_qgate_alloc_inner.jl")

# leaf-S5 (Julia speed lane, 2026-09-19): identity gates for the q=4 ML phylo
# route's cholesky!-reuse / warm-u0-vcov performance changes. Pins numbers
# measured on ORIGINAL origin/main so each change is a provable identity.
_shard_include("test_q4_perf_identities.jl")
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
# Deliberately not part of the discovery loop above: these four live under
# test/parity/ (not top-level test/), so readdir(@__DIR__) never finds them,
# and they are guarded behind DRM_PARITY_TESTS rather than sharded.
if _LIST_ONLY
    println("  [unsharded, gated on DRM_PARITY_TESTS] parity/runparity.jl, parity/runparity_bridge.jl, parity/runparity_bridge_formula.jl, parity/runparity_temporal.jl")
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
    @testset "R-parity temporal AR1/OU vs drmTMB 0.7.1 (D-310)" begin
        include("parity/runparity_temporal.jl")
    end
else
    @info "R-parity suite skipped (set DRM_PARITY_TESTS=1 to run)"
end
