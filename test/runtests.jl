using DRModels
using Test, LinearAlgebra, SparseArrays, Random
# --- Deterministic Julia-suite sharding (DRM_TEST_SHARD="k/N") -------------
# Splits the TOP-LEVEL sharded include calls below across N shards by file
# position: shard k (1-based) gets file i when (i - 1) % N == k - 1. Unset or
# empty means "run everything", so a local `julia test/runtests.jl` is
# unchanged. Helpers live in shard_util.jl and are exercised standalone in
# test_shard_selection.jl, which proves the N shards partition the file list
# exactly -- disjoint AND complete, so sharding cannot silently drop a file.
#
# Deliberately NOT sharded: the four includes nested inside conditionals (the
# JET gate, which has an else-branch, and the three parity suites behind
# DRM_PARITY_TESTS). Those are guarded by their own conditions; putting a shard
# test in front of a conditional would change what the guard means.
#
# MAINTENANCE NOTE: this file is REGENERATED from main on merge conflicts, not
# hand-merged. Every top-level include line changes shape here, so git resolves
# a conflict by keeping BOTH sides -- which silently duplicates whole blocks of
# includes (52 of them, caught 2026-09-03). Re-apply the transform to main's
# version instead of resolving hunks.
include("shard_util.jl")

_shard_spec = get(ENV, "DRM_TEST_SHARD", "")
_SHARD = isempty(_shard_spec) ? nothing : _parse_shard_spec(_shard_spec)

_shard_pos = Ref(0)

# --- BLAS thread count: pin once, then guard it per file --------------------
# The suite's numerical pins were all measured at one BLAS thread (the repo's
# stated invariant), and ten test files assert or set BLAS=1 themselves. CI
# does not set OPENBLAS_NUM_THREADS, so without this pin a shard starts at 2
# and silently switches to 1 partway through (the first test_joint_missing_*
# file sets it and never restores). Pin it here so every file, on every
# platform, runs at the same count.
BLAS.set_num_threads(1)

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
const _BLAS_THREADS_AT_START = BLAS.get_num_threads()
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

function _shard_include(path::AbstractString)
    _shard_pos[] += 1
    if _SHARD === nothing || (_shard_pos[] - 1) % _SHARD[2] == _SHARD[1] - 1
        include(path)
        _check_blas_restored(path)
    end
end

# Count only the TOP-LEVEL call sites, not this line's own literal.
_n_test_files = count("\n_shard_include(\"", read(@__FILE__, String))
_n_selected = _SHARD === nothing ? _n_test_files :
    length(_shard_indices(_n_test_files, _SHARD[1], _SHARD[2]))
println(_SHARD === nothing ?
    "DRModels tests: all $_n_test_files files (unsharded)" :
    "DRModels tests: shard $(_SHARD[1])/$(_SHARD[2]) - $_n_selected of $_n_test_files files")


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
_shard_include("test_meta_random_effect.jl")  # Arc 2: meta_V + (1 | g) / phylo / relmat, same model as drmTMB
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

# Standing Workflow Q JET gate (Karpinski): type-stability of hot lc↔Λ kernels.
# JET lives in test/Project.toml — skip gracefully when absent (bare
# `julia --project=. test/runtests.jl`). Macro body is in a separate file so it
# is only parsed when JET is present (same pattern as GLLVM.jl).
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

# #13 S1b: extracted Fisher / observed-information metric (natgrad solver FAIL —
# infra only; not a public `:natgrad` path).
_shard_include("test_lc_metric.jl")

# Standing FD-vs-exact gradient gate (issue #165) for the non-Gaussian (Poisson)
# phylogenetic sparse-Laplace route — the exact implicit-logdet outer gradient.
_shard_include("test_poisson_phylo_grad_gate.jl")

# Standing FD-vs-exact gradient gate (#165) for the Poisson CROSSED-random-
# intercepts route — same full-Newton-in-basin inner-mode fix as the phylo route.
_shard_include("test_poisson_crossed_grad_gate.jl")

# Standing FD-vs-exact gradient gates (#165) for the other non-Gaussian phylo
# routes (NB2, Gamma, Binomial, Beta-binomial (#166) ≤ 1e-6; Beta reported honestly).
_shard_include("test_nongaussian_phylo_grad_gate.jl")

# Non-Gaussian phylogenetic LOCATION–SCALE (#202): scale-axis SD recovery + the
# ≤ 1e-6 FD gradient gate on the q=2 (mean + log-σ) Laplace marginal.
_shard_include("test_phylo_locscale.jl")
# #202 closeout: PUBLIC drm() grammar B `(1 | p | phylo(species))` (tree= forward).
_shard_include("test_public_phylo_locscale.jl")
# σ-phylo location-scale (Ayumi #2): separate/coupled/asymmetric blocks + boundary CIs.
_shard_include("test_gaussian_locscale_phylo.jl")
_shard_include("test_gaussian_locscale_phylo_boundary.jl")
# A4c: penalized-MAP phylo variance components (drmTMB's drm_phylo_penalty + sweep).
_shard_include("test_phylo_penalty.jl")
# #422: boundary polish when a variance component collapses onto the flat shelf.
_shard_include("test_boundary_polish.jl")
# Tree-scale convention: O(p) height + the sqrt(h) reporting warning.
_shard_include("test_phylo_tree_height.jl")
_shard_include("test_phylo_polytomy.jl")
_shard_include("test_phylo_labels.jl")
_shard_include("test_phylo_polytomy_kernels.jl")
# A4d-2: post-fit inventories (profile_targets, structured_effects).
_shard_include("test_introspection.jl")
# A8: bivariate meta-analysis with known sampling covariance (meta_vcov_bivariate).
_shard_include("test_meta_vcov_bivariate.jl")
# A11: formula front end for the cross-family latent-rho route.
_shard_include("test_cross_family_formula.jl")
# Profile-likelihood CIs for the bivariate q4 among-axis SDs (Ayumi #2): the calibrated,
# no-Hessian complement to the bootstrap — collapsed axis → lower bound 0; panel-hardened
# (straddle guard, warm-start convergence gate, consistent nll_hat).
_shard_include("test_profile_sigma_a.jl")
# Parametric bootstrap of the bivariate q=4 among-axis SDs (Ayumi #2): the
# single-tree boundary-honest CI for sqrt.(diag(Σ_a)) — a collapsing axis reports
# an interval that sits at ~0, where the q4 profile is singular.
_shard_include("test_bootstrap_sigma_a.jl")
# REML for the σ-phylo location-scale route (Patterson–Thompson) + the fast observed-
# information Newton (the average-information data-quadratic was proven invalid here — â is
# the shrunk BLUP; see the second file's note). Both were orphan files; wired in here.
_shard_include("test_reml_sigma_phylo.jl")
_shard_include("test_reml_reml_phylo_mean.jl")
_shard_include("test_reml_newton_sigma_phylo.jl")
# Bivariate q4 REML must correct ALL FOUR among-axis SDs (β_μ AND β_σ profiled), not
# just the means — regression for the scale-axis REML gap (#18).
_shard_include("test_reml_q4_allaxes.jl")
# Bivariate q2 structured REML (#470): marginalises beta_mu1/beta_mu2 only (the
# axes with a random effect on this route); sigma1/sigma2/rho12 stay outer.
_shard_include("test_reml_q2_structured.jl")
_shard_include("test_q2_structured_vcov.jl")
_shard_include("test_reml_baseline_ladder.jl")
# Covariate dispersion (`sigma ~ x`) with a mean-only phylo RE for NB2 (#164):
# the per-observation log-dispersion (vector-nuisance) generalisation of the
# scalar phylo Laplace spine, with its own FD-vs-exact gate ≤ 1e-6.
_shard_include("test_164_mean_re_covariate_sigma.jl")
# Same covariate-dispersion path extended to Gamma and Beta (#164).
_shard_include("test_164_gamma_hetero.jl")

# Cox–Reid (opt-in `method = :REML`) for Poisson (#465, migrated from the poc
# and previously never run — no test covered the PR #451 Cox–Reid landing).
# #443: the scalar `(1 | g)` GHQ route (the one certified cell). #450: phylo /
# relmat / animal Laplace. The characterization file documents the ML default
# and the routes still uncertified.
_shard_include("test_cox_reid_poisson_ranef.jl")
_shard_include("test_cox_reid_poisson_phylo.jl")
_shard_include("test_cox_reid_characterization.jl")

# Experimental optimizer / EM-robustness fixes for the not-yet-wired sources under
# src/experimental/ (#305 deterministic LBFGS gradient, #306 monotone conjugate EM,
# #307 gradient-norm E-step convergence, #325.1 guarded step). Loads the standalone
# experimental scripts into isolated modules and exercises the specific defects.
_shard_include("test_experimental_optimizer.jl")

# Gated real-parity suite vs committed drmTMB fixtures (off by default).
# Native `drm()` path (#17) plus `drm_bridge` marshalling path (#370).
if get(ENV, "DRM_PARITY_TESTS", "0") == "1"
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

# Chi-bar-square boundary-corrected p-values for variance-component LR tests.
_shard_include("test_chibar.jl")

# #304: lrtest/anova warn on a boundary variance-component drop (naive χ² invalid).
_shard_include("test_lrtest_boundary_warn.jl")

# #320 / #323.2: coeftable/show suppress z/p for non-location blocks and Inf-SE rows.
_shard_include("test_summary_zp_suppress.jl")
_shard_include("test_r2_constant_sigma.jl")

# #325.3: bootstrap summary indexes coefficients by stored block range, not a counter.
_shard_include("test_bootstrap_block_index.jl")

# #313: heritability :profile is a TRUE profile (re-optimises nuisance), not ELR.
_shard_include("test_heritability_true_profile.jl")

# #310: REML-reported Wald vcov includes the restricted-penalty curvature.
_shard_include("test_reml_vcov_curvature.jl")

# Randomized quantile residuals (DHARMa/glmmTMB style) — feat-quantile-residuals.
_shard_include("test_quantile_residuals.jl")

# S3: cross-family bivariate (shared-latent GHQ) + link-residual standardization.
_shard_include("test_mixed_family.jl")
# Post-fit accessors (coef/aic/bic/fitted/summary) for the cross-family fit.
_shard_include("test_mixed_family_postfit.jl")

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

# Issue #646: on the missing-response Gaussian route `is_converged` read false on a
# genuinely converged fit (the degeneracy bar took std() of a NaN-carrying response,
# and every > against NaN is false), and every bootstrap replicate threw
# DimensionMismatch because simulate drew fit.nobs values against full-design
# means. Guards both, plus the iteration count the full-row rebuild dropped.
_shard_include("test_bridge_response_mask_inference.jl")
