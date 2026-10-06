using DRModels
using Test, Random, LinearAlgebra, Statistics

@testset "Location-only Gaussian phylo: supplied-variance REML and traces" begin
    Random.seed!(20260621)
    G = 10
    phy = random_balanced_tree(G; branch_length = 0.25)
    species = repeat(1:G, inner = 2)
    n = length(species)
    x = range(-1.0, 1.0; length = n)
    X = hcat(ones(n), collect(x))
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    σ = 0.45
    σ_phy = 0.7
    u = σ_phy .* (cholesky(Symmetric(C)).L * randn(G))
    y = X * [0.25, -0.4] .+ u[species] .+ σ .* randn(n)
    prob = DRModels.make_loc_problem(phy, y, X; species = species)
    lσ = log(σ)
    lσ_phy = log(σ_phy)

    comp = DRModels._loconly_reml_components(prob, lσ, lσ_phy)
    @test comp.converged
    @test isfinite(comp.nll)
    @test isfinite(comp.ml_nll)
    @test isfinite(comp.penalty)
    @test length(comp.beta) == size(X, 2)
    dense_comp = DRModels._loconly_dense_reml_components(prob, lσ, lσ_phy)
    cmp_diag = DRModels._loconly_dense_comparator_diagnostic(prob, lσ, lσ_phy)
    score_diag = DRModels._loconly_reml_dense_score_diagnostic(prob, lσ, lσ_phy)
    sparse_score_diag = DRModels._loconly_reml_sparse_score_diagnostic(prob, lσ, lσ_phy)
    boundary = DRModels._loconly_reml_boundary_status(prob, lσ, lσ_phy)

    V = σ^2 .* I(n) + σ_phy^2 .* C[species, species]
    chV = cholesky(Symmetric(Matrix(V)))
    VinvX = chV \ X
    Vinvy = chV \ y
    info_dense = X' * VinvX
    beta_dense = info_dense \ (X' * Vinvy)
    r = y .- X * beta_dense
    ml_nll_dense = 0.5 * (n * log(2π) + logdet(chV) + dot(r, chV \ r))
    restricted_penalty_dense =
        sum(log, LinearAlgebra.diag(cholesky(Symmetric(info_dense)).U))
    reml_constant_offset = 0.5 * size(X, 2) * log(2π)
    reml_nll_dense = ml_nll_dense + restricted_penalty_dense - reml_constant_offset

    @test comp.beta ≈ beta_dense rtol = 1e-8 atol = 1e-8
    @test comp.ml_nll ≈ ml_nll_dense rtol = 1e-8 atol = 1e-8
    @test comp.penalty ≈ restricted_penalty_dense rtol = 1e-8 atol = 1e-8
    @test comp.info ≈ info_dense rtol = 1e-8 atol = 1e-8
    @test comp.nll ≈ reml_nll_dense rtol = 1e-8 atol = 1e-8
    @test DRModels._loconly_reml_nll(prob, lσ, lσ_phy) ≈ reml_nll_dense rtol = 1e-8 atol = 1e-8
    @test dense_comp.converged
    @test dense_comp.matrix_mode === :dense_developer
    @test dense_comp.beta ≈ beta_dense rtol = 1e-8 atol = 1e-8
    @test dense_comp.info ≈ info_dense rtol = 1e-8 atol = 1e-8
    @test dense_comp.nll ≈ reml_nll_dense rtol = 1e-8 atol = 1e-8
    @test cmp_diag.comparator === :dense_same_estimand_oracle
    @test cmp_diag.finite
    @test cmp_diag.nll_absdiff < 1e-8
    @test cmp_diag.beta_absdiff < 1e-8
    @test cmp_diag.info_absdiff < 1e-8
    @test score_diag.target === :gaussian_loconly_reml
    @test score_diag.parameterization === :log_sd
    @test score_diag.matrix_mode === :dense_developer
    @test score_diag.finite
    @test score_diag.score ≈ score_diag.fd_score rtol = 1e-6 atol = 1e-6
    @test score_diag.max_absdiff < 1e-6
    @test sparse_score_diag.target === :gaussian_loconly_reml
    @test sparse_score_diag.parameterization === :log_sd
    @test sparse_score_diag.matrix_mode === :sparse_woodbury_developer
    @test sparse_score_diag.finite
    @test sparse_score_diag.score ≈ score_diag.score rtol = 1e-8 atol = 1e-8
    @test sparse_score_diag.score ≈ sparse_score_diag.fd_score rtol = 1e-6 atol = 1e-6
    @test sparse_score_diag.max_absdiff_dense < 1e-8
    @test sparse_score_diag.max_absdiff_fd < 1e-6
    @test all(isfinite, sparse_score_diag.trace_terms)
    @test all(isfinite, sparse_score_diag.quadratic_terms)
    @test all(isfinite, sparse_score_diag.correction_terms)
    @test boundary.boundary_status === :interior
    @test boundary.finite
    @test boundary.converged

    _, M, chM, _ = DRModels.build_M(prob, σ_phy^2, σ^2)
    tr_QM, tr_SMS = DRModels.exact_traces(prob, chM)
    Minv = inv(Matrix(M))
    tr_QM_dense = tr(Matrix(prob.Q_cond) * Minv)
    tr_SMS_dense = sum(prob.STS_diag .* LinearAlgebra.diag(Minv))
    diag = DRModels._loconly_takahashi_trace_diagnostic(prob, lσ, lσ_phy)
    pev = DRModels._loconly_takahashi_pev_diagnostic(prob, lσ, lσ_phy)

    @test diag.trace_mode === :takahashi_selinv
    @test diag.finite
    @test diag.tr_QM ≈ tr_QM_dense rtol = 1e-8 atol = 1e-8
    @test diag.tr_SMS ≈ tr_SMS_dense rtol = 1e-8 atol = 1e-8
    @test tr_QM ≈ tr_QM_dense rtol = 1e-8 atol = 1e-8
    @test tr_SMS ≈ tr_SMS_dense rtol = 1e-8 atol = 1e-8
    @test pev.trace_mode === :takahashi_selinv
    @test pev.finite
    @test pev.n_keep == prob.n_keep
    @test pev.n_leaves == G
    @test pev.posterior_variance ≈ LinearAlgebra.diag(Minv) rtol = 1e-8 atol = 1e-8
    @test pev.leaf_posterior_variance ≈ LinearAlgebra.diag(Minv)[prob.leaf_pos] rtol = 1e-8 atol = 1e-8
    @test all(>=(0), pev.leaf_posterior_variance)
    @test pev.posterior_variance_min >= 0
    @test pev.posterior_variance_max >= pev.posterior_variance_min
    @test pev.leaf_posterior_variance_mean ≈ mean(pev.leaf_posterior_variance) rtol = 1e-12 atol = 1e-12
    @test pev.weighted_leaf_posterior_trace ≈ tr_SMS_dense rtol = 1e-8 atol = 1e-8

    info_diag = DRModels._loconly_ai_information_diagnostic(prob, lσ, lσ_phy)
    sparse_info_diag = DRModels._loconly_reml_sparse_ai_information_diagnostic(prob, lσ, lσ_phy)
    @test info_diag.target === :gaussian_loconly_reml
    @test info_diag.parameterization === :log_sd
    @test info_diag.matrix_mode === :dense_developer
    @test info_diag.finite
    @test size(info_diag.ai) == (2, 2)
    @test size(info_diag.observed) == (2, 2)
    @test info_diag.ai ≈ info_diag.ai' atol = 1e-10
    @test info_diag.observed ≈ info_diag.observed' atol = 1e-8
    @test info_diag.relative_error < 0.1
    @test sparse_info_diag.target === :gaussian_loconly_reml
    @test sparse_info_diag.parameterization === :log_sd
    @test sparse_info_diag.matrix_mode === :sparse_woodbury_developer
    @test sparse_info_diag.finite
    @test sparse_info_diag.ai ≈ sparse_info_diag.ai' atol = 1e-10
    @test sparse_info_diag.ai ≈ info_diag.ai rtol = 1e-8 atol = 1e-8
    @test sparse_info_diag.max_absdiff_dense < 1e-8
    @test sparse_info_diag.relative_error_observed < 0.1

    opt_diag = DRModels._loconly_reml_optimizer_diagnostic(
        prob; starts = [[lσ, lσ_phy], [log(σ * 1.2), log(σ_phy * 0.8)]],
        iterations = 80,
    )
    @test opt_diag.target === :gaussian_loconly_reml
    @test opt_diag.estimator === :fd_reml_optimizer_experiment
    @test opt_diag.parameterization === :log_sd
    @test opt_diag.optimizer === :lbfgs_fd_gradient
    @test opt_diag.claim_status === :optimizer_experiment
    @test !opt_diag.ai_reml_ready
    @test occursin("finite-difference", opt_diag.reason_not_ai_reml)
    @test opt_diag.finite
    @test length(opt_diag.records) == 2
    @test opt_diag.best_nll <= reml_nll_dense + 1e-5
    @test DRModels._loconly_reml_nll(prob, opt_diag.best_minimizer[1], opt_diag.best_minimizer[2]) ≈ opt_diag.best_nll rtol = 1e-8 atol = 1e-8
    @test opt_diag.dense_comparator.finite
    @test opt_diag.dense_comparator.nll_absdiff < 1e-7
    @test opt_diag.observed_hessian ≈ opt_diag.observed_hessian' atol = 1e-8
    @test opt_diag.observed_hessian_pd
    @test opt_diag.boundary_status === :interior
    @test opt_diag.local_profile.finite
    @test opt_diag.local_profile.center_is_axis_min
    @test opt_diag.fd_stability.finite
    @test opt_diag.fd_stability.gradient_max_absdiff < 1e-3
    @test opt_diag.fd_stability.hessian_max_absdiff < 1e-2
    @test opt_diag.dense_score.finite
    @test opt_diag.dense_score.max_absdiff < 1e-6
    @test opt_diag.best_score_norm < 1e-3
    @test opt_diag.n_starts == 2
    @test opt_diag.n_finite_records == 2
    @test opt_diag.n_accepted_records >= 1
    @test opt_diag.best_improvement >= -1e-8

    score_opt = DRModels._loconly_reml_dense_score_optimizer_diagnostic(
        prob; starts = [[lσ, lσ_phy], [log(σ * 1.2), log(σ_phy * 0.8)]],
        iterations = 80,
    )
    @test score_opt.target === :gaussian_loconly_reml
    @test score_opt.estimator === :dense_score_reml_optimizer_experiment
    @test score_opt.optimizer === :lbfgs_dense_reml_score
    @test score_opt.claim_status === :optimizer_experiment
    @test !score_opt.ai_reml_ready
    @test occursin("dense analytic score", score_opt.reason_not_ai_reml)
    @test score_opt.finite
    @test score_opt.accepted
    @test score_opt.dense_comparator.finite
    @test score_opt.boundary_status === :interior
    @test score_opt.n_starts == 2
    @test score_opt.n_finite_records == 2
    @test score_opt.best_score_norm < 1e-3
    @test score_opt.best_nll ≈ opt_diag.best_nll rtol = 1e-6 atol = 1e-6

    sparse_score_opt = DRModels._loconly_reml_sparse_score_optimizer_diagnostic(
        prob; starts = [[lσ, lσ_phy], [log(σ * 1.2), log(σ_phy * 0.8)]],
        iterations = 80,
    )
    @test sparse_score_opt.target === :gaussian_loconly_reml
    @test sparse_score_opt.estimator === :sparse_score_reml_optimizer_experiment
    @test sparse_score_opt.optimizer === :lbfgs_sparse_woodbury_reml_score
    @test sparse_score_opt.claim_status === :optimizer_experiment
    @test !sparse_score_opt.ai_reml_ready
    @test occursin("sparse Woodbury score", sparse_score_opt.reason_not_ai_reml)
    @test sparse_score_opt.finite
    @test sparse_score_opt.accepted
    @test sparse_score_opt.dense_comparator.finite
    @test sparse_score_opt.boundary_status === :interior
    @test sparse_score_opt.sparse_score.finite
    @test sparse_score_opt.best_max_absdiff_dense < 1e-8
    @test sparse_score_opt.n_starts == 2
    @test sparse_score_opt.n_finite_records == 2
    @test sparse_score_opt.best_score_norm < 1e-3
    @test sparse_score_opt.best_nll ≈ score_opt.best_nll rtol = 1e-6 atol = 1e-6

    ai_update_opt = DRModels._loconly_reml_ai_update_optimizer_diagnostic(
        prob; starts = [[lσ, lσ_phy], [log(σ * 1.2), log(σ_phy * 0.8)]],
        iterations = 30,
    )
    @test ai_update_opt.target === :gaussian_loconly_reml
    @test ai_update_opt.estimator === :guarded_ai_update_reml_optimizer_experiment
    @test ai_update_opt.optimizer === :guarded_sparse_average_information_update
    @test ai_update_opt.claim_status === :optimizer_experiment
    @test !ai_update_opt.ai_reml_ready
    @test occursin("guarded average-information", ai_update_opt.reason_not_ai_reml)
    @test ai_update_opt.finite
    @test ai_update_opt.accepted
    @test ai_update_opt.dense_comparator.finite
    @test ai_update_opt.boundary_status === :interior
    @test ai_update_opt.sparse_score.finite
    @test ai_update_opt.sparse_information.finite
    @test ai_update_opt.n_starts == 2
    @test ai_update_opt.n_finite_records == 2
    @test ai_update_opt.n_accepted_records >= 1
    @test ai_update_opt.best_score_norm < 1e-3
    @test ai_update_opt.best_nll ≈ opt_diag.best_nll rtol = 1e-5 atol = 1e-5
    @test ai_update_opt.best_nll ≈ score_opt.best_nll rtol = 1e-5 atol = 1e-5
    @test ai_update_opt.best_nll ≈ sparse_score_opt.best_nll rtol = 1e-5 atol = 1e-5
    @test any(r -> any(t -> t.status === :accepted_step, r.trace), ai_update_opt.records)
    @test all(r -> all(t -> t.halvings >= 0, r.trace), ai_update_opt.records)

    payload = DRModels._loconly_reml_diagnostic_payload(prob, lσ, lσ_phy)
    @test payload.target === :gaussian_loconly_phylo_reml
    @test payload.estimator === :supplied_variance_reml
    @test payload.boundary.boundary_status === :interior
    @test payload.dense_comparator.finite
    @test payload.score.finite
    @test payload.sparse_score.finite
    @test payload.sparse_score.max_absdiff_dense < 1e-8
    @test payload.trace.finite
    @test payload.pev.finite
    @test payload.information.finite
    @test payload.sparse_information.finite
    @test payload.sparse_information.max_absdiff_dense < 1e-8
    @test payload.fd_stability.finite
    @test payload.local_profile.finite
    @test payload.validation_status.claim_status === :internal_diagnostic
    @test payload.bridge_schema.r_bridge_status == "planned"
    @test payload.claim_status === :internal_diagnostic

    recovery = DRModels._loconly_reml_recovery_grid_diagnostic(
        ; reps = 3, G = 10, n_per_species = 3, sigma = σ, sigma_phy = σ_phy,
        seed = 20260624, iterations = 30,
    )
    @test recovery.target === :gaussian_loconly_phylo_reml
    @test recovery.estimator === :guarded_ai_update_reml_optimizer_experiment
    @test recovery.design === :tiny_deterministic_recovery_grid
    @test recovery.claim_status === :simulation_diagnostic
    @test recovery.coverage_status === :not_evaluated
    @test !recovery.ai_reml_ready
    @test recovery.conditions.reps == 3
    @test recovery.conditions.G == 10
    @test recovery.conditions.n_per_species == 3
    @test recovery.n_reps == 3
    @test recovery.n_accepted == 3
    @test recovery.convergence_rate == 1.0
    @test recovery.boundary_counts.interior == 3
    @test all(r -> r.accepted && r.finite, recovery.records)
    @test all(r -> r.score_norm < 1e-3, recovery.records)
    @test isfinite(recovery.bias_sigma)
    @test isfinite(recovery.bias_sigma_phy)
    @test isfinite(recovery.rmse_sigma)
    @test isfinite(recovery.rmse_sigma_phy)
    @test isfinite(recovery.mcse_bias_sigma)
    @test isfinite(recovery.mcse_bias_sigma_phy)
    # This tiny deterministic grid is a diagnostic row-contract gate, not a
    # promotion-grade recovery study; keep point-bias bounds wide enough for
    # CI RNG and Julia-version drift while the status fields above stay strict.
    @test abs(recovery.bias_sigma) < 0.08
    @test abs(recovery.bias_sigma_phy) < 0.30

    condition_grid = DRModels._loconly_reml_recovery_condition_grid_diagnostic(
        ; reps = 2, iterations = 30,
    )
    @test condition_grid.target === :gaussian_loconly_phylo_reml
    @test condition_grid.estimator === :guarded_ai_update_reml_optimizer_experiment
    @test condition_grid.design === :tiny_condition_recovery_grid
    @test condition_grid.claim_status === :simulation_diagnostic
    @test condition_grid.coverage_status === :not_evaluated
    @test !condition_grid.ai_reml_ready
    @test condition_grid.n_cells == 2
    @test length(condition_grid.rows) == 2
    @test condition_grid.min_convergence_rate == 1.0
    @test condition_grid.all_cells_accepted
    @test Set(r.cell for r in condition_grid.rows) ==
        Set((:baseline_interior, :higher_phylo_interior))
    @test all(r -> r.n_reps == 2 && r.n_accepted == 2, condition_grid.rows)
    @test all(r -> r.boundary_counts.interior == 2, condition_grid.rows)
    @test all(r -> isfinite(r.bias_sigma) && isfinite(r.bias_sigma_phy), condition_grid.rows)
    @test all(r -> r.diagnostic.claim_status === :simulation_diagnostic, condition_grid.rows)
end

@testset "Location-only Gaussian phylo: boundary diagnostics are finite or explicit" begin
    Random.seed!(20260622)
    G = 8
    species = repeat(1:G, inner = 2)
    n = length(species)
    x = range(-1.0, 1.0; length = n)
    X = hcat(ones(n), collect(x))

    phy_weak = random_balanced_tree(G; branch_length = 0.25)
    y_weak = X * [0.2, 0.1] .+ 0.4 .* randn(n)
    prob_weak = DRModels.make_loc_problem(phy_weak, y_weak, X; species = species)
    comp_zero = DRModels._loconly_reml_components(prob_weak, log(0.4), log(1e-8))
    trace_zero = DRModels._loconly_takahashi_trace_diagnostic(prob_weak, log(0.4), log(1e-8))
    score_zero = DRModels._loconly_reml_sparse_score_diagnostic(prob_weak, log(0.4), log(1e-8))
    boundary_zero = DRModels._loconly_reml_boundary_status(prob_weak, log(0.4), log(1e-8))
    @test comp_zero.converged
    @test isfinite(comp_zero.nll)
    @test trace_zero.finite
    @test trace_zero.tr_QM >= 0
    @test trace_zero.tr_SMS >= 0
    @test score_zero.finite
    @test score_zero.max_absdiff_dense < 1e-8
    @test boundary_zero.boundary_status === :near_zero_variance

    phy_near = random_balanced_tree(G; branch_length = 1e-5)
    C_near = sigma_phy_dense(phy_near; σ²_phy = 1.0)
    u_near = 0.1 .* (cholesky(Symmetric(C_near)).L * randn(G))
    y_near = X * [0.2, 0.1] .+ u_near[species] .+ 0.4 .* randn(n)
    prob_near = DRModels.make_loc_problem(phy_near, y_near, X; species = species)
    comp_near = DRModels._loconly_reml_components(prob_near, log(0.4), log(0.1))
    trace_near = DRModels._loconly_takahashi_trace_diagnostic(prob_near, log(0.4), log(0.1))
    @test comp_near.converged
    @test isfinite(comp_near.nll)
    @test trace_near.finite

    X_bad = hcat(ones(n), ones(n))
    prob_bad = DRModels.make_loc_problem(phy_weak, y_weak, X_bad; species = species)
    comp_bad = DRModels._loconly_reml_components(prob_bad, log(0.4), log(0.2))
    info_bad = DRModels._loconly_ai_information_diagnostic(prob_bad, log(0.4), log(0.2))
    sparse_info_bad = DRModels._loconly_reml_sparse_ai_information_diagnostic(prob_bad, log(0.4), log(0.2))
    score_bad = DRModels._loconly_reml_sparse_score_diagnostic(prob_bad, log(0.4), log(0.2))
    ai_update_bad = DRModels._loconly_reml_ai_update_optimizer_diagnostic(
        prob_bad; starts = [[log(0.4), log(0.2)]], iterations = 5,
    )
    boundary_bad = DRModels._loconly_reml_boundary_status(prob_bad, log(0.4), log(0.2))
    boundary_invalid = DRModels._loconly_reml_boundary_status(prob_bad, Inf, log(0.2))
    @test !comp_bad.converged
    @test comp_bad.nll == DRModels._LOCONLY_PENALTY
    @test !info_bad.finite
    @test !sparse_info_bad.finite
    @test !score_bad.finite
    @test !ai_update_bad.finite
    @test !ai_update_bad.accepted
    @test ai_update_bad.n_starts == 1
    @test boundary_bad.boundary_status === :singular_fixed_effect_information
    @test boundary_invalid.boundary_status === :nonfinite_objective

    weak_probe = DRModels._loconly_reml_weak_signal_recovery_probe(
        ; reps = 2, seed = 20260630, iterations = 25,
    )
    @test weak_probe.target === :gaussian_loconly_phylo_reml
    @test weak_probe.design === :weak_signal_boundary_probe
    @test weak_probe.expected_behavior === :boundary_states_allowed
    @test weak_probe.claim_status === :simulation_diagnostic
    @test weak_probe.coverage_status === :not_evaluated
    @test !weak_probe.ai_reml_ready
    @test weak_probe.diagnostic.n_reps == 2
    # The weak-signal probe permits boundary states, but tiny deterministic
    # draws can land fully interior on some Julia/RNG combinations.
    @test 0 <= weak_probe.boundary_reps <= weak_probe.diagnostic.n_reps
    @test 0 <= weak_probe.boundary_rate <= 1
    @test weak_probe.boundary_rate == weak_probe.boundary_reps / weak_probe.diagnostic.n_reps
    @test weak_probe.convergence_rate <= 1.0

    sim_status = DRModels._loconly_reml_simulation_status()
    @test sim_status.target === :gaussian_loconly_phylo_reml
    @test sim_status.estimator === :guarded_ai_update_reml_optimizer_experiment
    @test sim_status.claim_status === :simulation_diagnostic
    @test sim_status.coverage_status === :not_evaluated
    @test !sim_status.ai_reml_ready
    @test sim_status.n_rows == 4
    @test length(sim_status.rows) == 4
    @test Tuple(r.row_id for r in sim_status.rows) == (
        :stable_recovery,
        :condition_grid,
        :weak_signal_boundary_probe,
        :larger_interior_stress,
    )
    schema = DRModels._loconly_reml_simulation_status_schema()
    expected_schema = (
        :row_id, :target, :estimator, :design, :claim_status, :coverage_status,
        :expected_behavior, :n_reps, :n_accepted, :convergence_rate,
        :boundary_rate, :failure_reason_counts, :bias_sigma, :bias_sigma_phy,
        :rmse_sigma, :rmse_sigma_phy, :mcse_bias_sigma, :mcse_bias_sigma_phy,
        :mcse_status, :runtime_seconds, :runtime_budget_seconds, :seed,
        :seed_registry, :next_gate, :evidence,
    )
    @test schema == expected_schema
    @test :expected_behavior in schema
    @test :failure_reason_counts in schema
    @test :runtime_budget_seconds in schema
    @test :seed_registry in schema
    @test all(r -> all(field -> field in propertynames(r), schema), sim_status.rows)
    @test all(r -> r.target === :gaussian_loconly_phylo_reml, sim_status.rows)
    @test all(r -> r.estimator === :guarded_ai_update_reml_optimizer_experiment, sim_status.rows)
    @test all(r -> r.claim_status === :simulation_diagnostic, sim_status.rows)
    @test all(r -> r.coverage_status === :not_evaluated, sim_status.rows)
    @test all(r -> r.expected_behavior in (
        :stable_interior_recovery,
        :row_separated_stable_recovery,
        :boundary_states_allowed,
        :stress_smoke,
    ), sim_status.rows)
    @test all(r -> r.n_reps >= 2, sim_status.rows)
    @test all(r -> r.n_accepted <= r.n_reps, sim_status.rows)
    @test all(r -> r.boundary_rate >= 0 && r.boundary_rate <= 1, sim_status.rows)
    @test all(r -> sum(values(r.failure_reason_counts)) <= r.n_reps, sim_status.rows)
    @test all(r -> r.mcse_status === :diagnostic_only, sim_status.rows)
    @test all(r -> r.runtime_seconds >= 0, sim_status.rows)
    @test all(r -> r.runtime_budget_seconds > 0, sim_status.rows)
    @test all(r -> !isempty(string(r.seed)), sim_status.rows)
    @test all(r -> r.seed_registry.deterministic, sim_status.rows)
    @test all(r -> !isempty(r.evidence), sim_status.rows)
    @test all(r -> r.next_gate in (
        :broader_recovery_grid,
        :broader_condition_grid,
        :boundary_diagnostics,
        :optional_runtime_stress,
    ), sim_status.rows)
    weak_row = only(filter(r -> r.row_id === :weak_signal_boundary_probe, sim_status.rows))
    @test weak_row.expected_behavior === :boundary_states_allowed
    @test 0 <= weak_row.boundary_rate <= 1
    stress_row = only(filter(r -> r.row_id === :larger_interior_stress, sim_status.rows))
    @test stress_row.expected_behavior === :stress_smoke
    @test stress_row.n_reps == 2
    @test stress_row.n_accepted == 2

    validation = DRModels._loconly_reml_validate_simulation_status(sim_status)
    @test validation.ok
    @test isempty(validation.errors)
    @test validation.required_fields == schema
    @test validation.row_order == Tuple(r.row_id for r in sim_status.rows)
    @test validation.coverage_status === :not_evaluated
    @test !validation.ai_reml_ready

    bad_row = merge(first(sim_status.rows), (coverage_status = :covered,))
    bad_status = merge(sim_status, (rows = (bad_row,), n_rows = 1))
    bad_validation = DRModels._loconly_reml_validate_simulation_status(bad_status)
    @test !bad_validation.ok
    @test any(err -> occursin("evaluated coverage", err), bad_validation.errors)

    mktempdir() do dir
        path = joinpath(dir, "loconly-status.tsv")
        write_result = DRModels._loconly_reml_write_simulation_status_tsv(path; status = sim_status)
        @test write_result.path == path
        @test write_result.n_rows == sim_status.n_rows
        @test write_result.schema == schema
        @test write_result.validation.ok
        lines = readlines(path)
        @test String.(split(lines[1], '\t')) == collect(string.(schema))
        @test length(lines) == sim_status.n_rows + 1
        @test first(split(lines[2], '\t')) == "stable_recovery"

        bad_path = joinpath(dir, "bad-status.tsv")
        @test_throws ErrorException DRModels._loconly_reml_write_simulation_status_tsv(
            bad_path; status = bad_status,
        )
        @test !isfile(bad_path)
    end

    medium_status = DRModels._loconly_reml_simulation_status(
        ; include_medium_stress = true,
        medium_stress_reps = 1,
    )
    @test medium_status.n_rows == 5
    @test Tuple(r.row_id for r in medium_status.rows)[1:4] ==
        Tuple(r.row_id for r in sim_status.rows)
    medium_row = only(filter(r -> r.row_id === :medium_interior_stress,
                             medium_status.rows))
    @test medium_row.expected_behavior === :stress_smoke
    @test medium_row.n_reps == 1
    @test medium_row.runtime_budget_seconds == 15.0
    @test DRModels._loconly_reml_validate_simulation_status(medium_status).ok

    large_status = DRModels._loconly_reml_simulation_status(; include_large_stress = true)
    @test large_status.n_rows == 5
    large_row = only(filter(r -> r.row_id === :large_interior_stress_skipped,
                            large_status.rows))
    @test large_row.expected_behavior === :skipped_runtime_guard
    @test large_row.n_reps == 0
    @test large_row.n_accepted == 0
    @test large_row.runtime_seconds == 0.0
    @test large_row.runtime_budget_seconds == 0.0
    @test large_row.next_gate === :runtime_budget_review
    @test DRModels._loconly_reml_validate_simulation_status(large_status).ok

    provenance = DRModels._loconly_reml_simulation_status_provenance(sim_status)
    @test provenance.target === sim_status.target
    @test provenance.estimator === sim_status.estimator
    @test provenance.n_rows == sim_status.n_rows
    @test !provenance.ai_reml_ready
    @test provenance.coverage_status === :not_evaluated
    @test Tuple(r.row_id for r in provenance.rows) ==
        Tuple(r.row_id for r in sim_status.rows)
    @test all(r -> !isempty(r.helper), provenance.rows)
    @test all(r -> r.test == sim_status.evidence, provenance.rows)
    @test all(r -> occursin("ai_reml_ready=false", r.claim_boundary),
              provenance.rows)

    broader = DRModels._loconly_reml_broader_recovery_grid_diagnostic(
        ; reps = 1,
        iterations = 25,
    )
    @test broader.target === :gaussian_loconly_phylo_reml
    @test broader.design === :broader_recovery_grid
    @test broader.n_cells == 3
    @test broader.expected_behavior === :stable_or_stress_recovery
    @test broader.claim_status === :simulation_diagnostic
    @test broader.coverage_status === :not_evaluated
    @test !broader.ai_reml_ready
    @test Set(r.cell for r in broader.rows) == Set((
        :baseline_interior,
        :higher_phylo_interior,
        :medium_interior_stress,
    ))
    @test all(r -> r.n_reps == 1, broader.rows)

    weak_grid = DRModels._loconly_reml_weak_signal_condition_grid_diagnostic(
        ; reps = 1,
        iterations = 20,
    )
    @test weak_grid.target === :gaussian_loconly_phylo_reml
    @test weak_grid.design === :weak_signal_condition_grid
    @test weak_grid.n_cells == 2
    @test weak_grid.expected_behavior === :boundary_states_allowed
    @test weak_grid.claim_status === :simulation_diagnostic
    @test weak_grid.coverage_status === :not_evaluated
    @test !weak_grid.ai_reml_ready
    @test Set(r.cell for r in weak_grid.rows) ==
        Set((:low_phylo_signal, :near_zero_phylo_signal))
    @test all(r -> r.diagnostic.coverage_status === :not_evaluated, weak_grid.rows)

    boundary_grid = DRModels._loconly_reml_boundary_grid_status(
        ; reps = 1, iterations = 20,
    )
    boundary_grid_schema = DRModels._loconly_reml_boundary_grid_status_schema()
    @test boundary_grid_schema == (
        :row_id, :target, :design, :condition, :n_reps, :n_accepted,
        :convergence_rate, :boundary_rate, :near_zero_variance,
        :nonfinite_objective, :singular_fixed_effect_information,
        :expected_behavior, :claim_status, :coverage_status, :ai_reml_ready,
        :next_gate,
    )
    @test boundary_grid.target === :gaussian_loconly_phylo_reml
    @test boundary_grid.design === :near_zero_variance_boundary_grid
    @test boundary_grid.n_rows == 2
    @test boundary_grid.expected_behavior === :boundary_states_allowed
    @test boundary_grid.claim_status === :simulation_diagnostic
    @test boundary_grid.coverage_status === :not_evaluated
    @test !boundary_grid.ai_reml_ready
    boundary_grid_validation =
        DRModels._loconly_reml_validate_boundary_grid_status(boundary_grid)
    @test boundary_grid_validation.ok
    @test boundary_grid_validation.required_fields == boundary_grid_schema
    @test Set(r.row_id for r in boundary_grid.rows) ==
        Set((:low_phylo_signal, :near_zero_phylo_signal))
    @test all(r -> r.n_reps == 1, boundary_grid.rows)
    @test all(r -> 0 <= r.boundary_rate <= 1, boundary_grid.rows)
    @test all(r -> r.expected_behavior === :boundary_states_allowed,
              boundary_grid.rows)
    @test all(r -> r.claim_status === :simulation_diagnostic,
              boundary_grid.rows)
    @test all(r -> r.coverage_status === :not_evaluated,
              boundary_grid.rows)
    @test all(r -> !r.ai_reml_ready, boundary_grid.rows)
    bad_boundary_row = merge(first(boundary_grid.rows), (boundary_rate = 1.5,))
    bad_boundary_status =
        merge(boundary_grid, (rows = (bad_boundary_row,), n_rows = 1))
    bad_boundary_validation =
        DRModels._loconly_reml_validate_boundary_grid_status(bad_boundary_status)
    @test !bad_boundary_validation.ok
    @test any(err -> occursin("invalid boundary_rate", err),
              bad_boundary_validation.errors)
end

@testset "Location-only Gaussian phylo: status schema and scaling smoke" begin
    status = DRModels._loconly_reml_validation_status()
    @test status.target === :gaussian_loconly_phylo_reml
    @test status.estimator === :supplied_variance_reml
    @test status.source_status === :partial
    @test status.tests_status === :partial
    @test status.comparator_status === :dense_same_estimand_oracle
    @test status.external_comparator_status === :planned
    @test status.optimizer_status === :experiment_only
    @test status.r_bridge_status === :planned
    @test status.claim_status === :internal_diagnostic
    @test status.q4_status === :excluded

    schema = DRModels._loconly_reml_bridge_payload_schema()
    @test schema.target == "gaussian_loconly_phylo_reml"
    @test schema.estimator == "supplied_variance_reml"
    @test schema.effective_REML === true
    @test schema.trace_mode == "takahashi_selinv"
    @test schema.score_mode == "dense_or_sparse_woodbury_diagnostic"
    @test schema.information_mode == "ai_vs_observed_diagnostic"
    @test schema.claim_status == "internal_diagnostic"
    @test schema.r_bridge_status == "planned"
    @test "near_zero_variance" in schema.boundary_status_levels

    comparator_status = DRModels._loconly_reml_external_comparator_status()
    comparator_schema = DRModels._loconly_reml_external_comparator_schema()
    @test comparator_schema == (
        :comparator_id, :target, :comparator, :same_estimand_status,
        :dependency_status, :artifact_status, :decision, :reason, :next_gate,
    )
    @test comparator_status.target === :gaussian_loconly_phylo_reml
    @test comparator_status.external_comparator_status === :planned
    @test comparator_status.dependency_status === :not_added
    @test comparator_status.claim_status === :internal_diagnostic
    @test comparator_status.coverage_status === :not_evaluated
    @test !comparator_status.ai_reml_ready
    @test comparator_status.fixture_status === :versioned_fixture_defined
    @test comparator_status.validation.ok
    @test comparator_status.validation.required_fields == comparator_schema
    @test comparator_status.validation.n_rows == length(comparator_status.rows)
    @test comparator_status.fixture_validation.ok
    @test comparator_status.fixture_validation.required_fields ==
        comparator_status.fixture_schema
    @test comparator_status.version_probe_status === :defined_not_run
    @test comparator_status.version_probe_schema == (
        :comparator_id, :target, :candidate_package, :candidate_version,
        :fixture_id, :fixture_version, :same_estimand_status,
        :dependency_status, :artifact_status, :decision, :blocked_by,
        :required_evidence, :next_gate,
    )
    @test comparator_status.version_probe_validation.ok
    @test comparator_status.version_probe_validation.required_fields ==
        comparator_status.version_probe_schema
    @test comparator_status.version_probe_validation.dependency_status === :not_added
    @test comparator_status.version_probe_validation.n_rows ==
        length(comparator_status.version_probe_rows)
    @test length(comparator_status.version_probe_rows) == 1
    version_probe = only(comparator_status.version_probe_rows)
    @test version_probe.comparator_id === :phylolm_or_equivalent_reml
    @test version_probe.target === :gaussian_loconly_phylo_reml
    @test version_probe.candidate_package === :unselected_phylolm_style_reml
    @test version_probe.candidate_version === :unprobed
    @test version_probe.fixture_id === comparator_status.fixture.fixture_id
    @test version_probe.fixture_version == comparator_status.fixture.version
    @test version_probe.same_estimand_status === :requires_fixture_reproduction
    @test version_probe.dependency_status === :not_added
    @test version_probe.artifact_status === :probe_plan_defined
    @test version_probe.blocked_by === :external_package_version_not_probed
    @test :same_restricted_likelihood_target in version_probe.required_evidence
    @test :same_covariance_target in version_probe.required_evidence
    @test version_probe.next_gate === :optional_developer_comparator_script
    bad_version_probe = merge(version_probe, (dependency_status = :added,))
    bad_version_probe_validation =
        DRModels._loconly_reml_validate_external_comparator_version_probe(
            (bad_version_probe,), comparator_status.fixture)
    @test !bad_version_probe_validation.ok
    @test any(err -> occursin("must not add a dependency", err),
              bad_version_probe_validation.errors)
    probe_status = DRModels._loconly_reml_external_comparator_probe_status(
        ; candidate_package = "phylolm",
        candidate_version = "unavailable",
        package_available = false,
        rscript_status = :package_not_installed,
        evidence = "test/test_location_only_reml_mme.jl",
    )
    probe_schema = DRModels._loconly_reml_external_comparator_probe_result_schema()
    @test probe_schema == (
        :probe_id, :target, :fixture_id, :fixture_version, :comparator_id,
        :candidate_package, :candidate_version, :package_available,
        :rscript_status, :fit_status, :same_estimand_status,
        :dependency_status, :artifact_status, :claim_status,
        :coverage_status, :ai_reml_ready, :next_gate, :evidence,
    )
    @test probe_status.target === :gaussian_loconly_phylo_reml
    @test probe_status.fixture_id === comparator_status.fixture.fixture_id
    @test probe_status.fixture_version == comparator_status.fixture.version
    @test probe_status.n_rows == 1
    @test probe_status.claim_status === :internal_diagnostic
    @test probe_status.coverage_status === :not_evaluated
    @test !probe_status.ai_reml_ready
    probe_validation =
        DRModels._loconly_reml_validate_external_comparator_probe_status(probe_status)
    @test probe_validation.ok
    @test probe_validation.required_fields == probe_schema
    @test probe_validation.coverage_status === :not_evaluated
    @test !probe_validation.ai_reml_ready
    probe_row = only(probe_status.rows)
    @test probe_row.comparator_id === :phylolm_or_equivalent_reml
    @test probe_row.candidate_package == "phylolm"
    @test probe_row.candidate_version == "unavailable"
    @test probe_row.package_available === false
    @test probe_row.rscript_status === :package_not_installed
    @test probe_row.fit_status === :not_run
    @test probe_row.dependency_status === :not_added
    @test probe_row.artifact_status === :optional_developer_probe_written
    @test probe_row.next_gate === :external_fixture_fit_comparison
    bad_probe_row = merge(probe_row, (fit_status = :fit_run,))
    bad_probe_status = merge(probe_status, (rows = (bad_probe_row,),))
    bad_probe_validation =
        DRModels._loconly_reml_validate_external_comparator_probe_status(
            bad_probe_status)
    @test !bad_probe_validation.ok
    @test any(err -> occursin("must not mark a fit as run", err),
              bad_probe_validation.errors)
    mktempdir() do dir
        path = joinpath(dir, "external-comparator-probe.tsv")
        write_result = DRModels._loconly_reml_write_external_comparator_probe_tsv(
            path;
            candidate_package = "phylolm",
            candidate_version = "unavailable",
            package_available = false,
            rscript_status = :package_not_installed,
            evidence = "test/test_location_only_reml_mme.jl",
        )
        @test write_result.path == path
        @test write_result.n_rows == 1
        @test write_result.schema == probe_schema
        @test write_result.validation.ok
        lines = readlines(path)
        @test String.(split(lines[1], '\t')) == collect(string.(probe_schema))
        @test length(lines) == 2
        @test first(split(lines[2], '\t')) == "phylolm_style_version_probe"
    end
    fit_feasibility =
        DRModels._loconly_reml_external_comparator_fit_feasibility_status(
            ; candidate_package = "phylolm",
            candidate_version = "2.6.5",
        )
    fit_feasibility_schema =
        DRModels._loconly_reml_external_comparator_fit_feasibility_schema()
    @test fit_feasibility_schema == (
        :check_id, :target, :fixture_id, :fixture_version, :comparator_id,
        :candidate_package, :candidate_version, :n_obs, :n_species,
        :n_per_species, :within_species_x_varies, :fit_status,
        :same_estimand_status, :dependency_status, :blocked_by, :reason,
        :claim_status, :coverage_status, :ai_reml_ready, :next_gate,
    )
    @test fit_feasibility.target === :gaussian_loconly_phylo_reml
    @test fit_feasibility.n_rows == 1
    @test fit_feasibility.claim_status === :internal_diagnostic
    @test fit_feasibility.coverage_status === :not_evaluated
    @test !fit_feasibility.ai_reml_ready
    fit_feasibility_validation =
        DRModels._loconly_reml_validate_external_comparator_fit_feasibility(
            fit_feasibility)
    @test fit_feasibility_validation.ok
    @test fit_feasibility_validation.required_fields == fit_feasibility_schema
    fit_feasibility_row = only(fit_feasibility.rows)
    @test fit_feasibility_row.check_id === :phylolm_fixture_fit_feasibility
    @test fit_feasibility_row.candidate_package == "phylolm"
    @test fit_feasibility_row.candidate_version == "2.6.5"
    @test fit_feasibility_row.n_obs == comparator_status.fixture.n_obs
    @test fit_feasibility_row.n_species == comparator_status.fixture.n_species
    @test fit_feasibility_row.n_per_species ==
        comparator_status.fixture.n_per_species
    @test fit_feasibility_row.n_obs > fit_feasibility_row.n_species
    @test fit_feasibility_row.within_species_x_varies
    @test fit_feasibility_row.fit_status === :not_run
    @test fit_feasibility_row.same_estimand_status ===
        :not_same_estimand_current_fixture
    @test fit_feasibility_row.dependency_status === :not_added
    @test fit_feasibility_row.blocked_by ===
        :repeated_observation_fixture_not_tip_level
    @test occursin("replicated species rows", fit_feasibility_row.reason)
    @test fit_feasibility_row.next_gate ===
        :choose_replicate_capable_comparator_or_tip_level_fixture
    bad_feasibility_row = merge(fit_feasibility_row, (fit_status = :fit_run,))
    bad_feasibility_status =
        merge(fit_feasibility, (rows = (bad_feasibility_row,),))
    bad_feasibility_validation =
        DRModels._loconly_reml_validate_external_comparator_fit_feasibility(
            bad_feasibility_status)
    @test !bad_feasibility_validation.ok
    @test any(err -> occursin("must not mark a fit as run", err),
              bad_feasibility_validation.errors)
    derivative_fd = DRModels._loconly_reml_derivative_fd_status()
    derivative_schema = DRModels._loconly_reml_derivative_fd_status_schema()
    @test derivative_schema == (
        :check_id, :target, :fixture_id, :fixture_version,
        :parameterization, :matrix_mode, :h, :score_max_absdiff_fd,
        :dense_sparse_max_absdiff, :finite, :claim_status,
        :coverage_status, :ai_reml_ready, :next_gate,
    )
    @test derivative_fd.target === :gaussian_loconly_phylo_reml
    @test derivative_fd.fixture_id === comparator_status.fixture.fixture_id
    @test derivative_fd.fixture_version == comparator_status.fixture.version
    @test derivative_fd.n_rows == 2
    @test derivative_fd.claim_status === :internal_diagnostic
    @test derivative_fd.coverage_status === :not_evaluated
    @test !derivative_fd.ai_reml_ready
    derivative_validation =
        DRModels._loconly_reml_validate_derivative_fd_status(derivative_fd)
    @test derivative_validation.ok
    @test derivative_validation.required_fields == derivative_schema
    @test derivative_validation.coverage_status === :not_evaluated
    @test !derivative_validation.ai_reml_ready
    @test Tuple(r.check_id for r in derivative_fd.rows) ==
        (:dense_score_fd_fixture, :sparse_score_fd_fixture)
    @test all(r -> r.target === :gaussian_loconly_phylo_reml,
              derivative_fd.rows)
    @test all(r -> r.parameterization === :log_sd, derivative_fd.rows)
    @test all(r -> r.finite, derivative_fd.rows)
    @test all(r -> r.claim_status === :internal_diagnostic,
              derivative_fd.rows)
    @test all(r -> r.coverage_status === :not_evaluated,
              derivative_fd.rows)
    @test all(r -> !r.ai_reml_ready, derivative_fd.rows)
    @test all(r -> r.score_max_absdiff_fd < 1e-5, derivative_fd.rows)
    @test last(derivative_fd.rows).dense_sparse_max_absdiff < 1e-7
    bad_derivative_row =
        merge(first(derivative_fd.rows), (score_max_absdiff_fd = 1.0,))
    bad_derivative_status =
        merge(derivative_fd, (rows = (bad_derivative_row,), n_rows = 1))
    bad_derivative_validation =
        DRModels._loconly_reml_validate_derivative_fd_status(
            bad_derivative_status)
    @test !bad_derivative_validation.ok
    @test any(err -> occursin("exceeds FD tolerance", err),
              bad_derivative_validation.errors)
    line_search = DRModels._loconly_reml_line_search_status()
    line_search_schema = DRModels._loconly_reml_line_search_status_schema()
    @test line_search_schema == (
        :check_id, :target, :fixture_id, :fixture_version, :estimator,
        :optimizer, :n_starts, :n_records, :n_accepted_records,
        :accepted, :best_score_norm, :boundary_status, :trace_status,
        :claim_status, :coverage_status, :ai_reml_ready,
        :reason_not_ai_reml, :next_gate,
    )
    @test line_search.target === :gaussian_loconly_phylo_reml
    @test line_search.fixture_id === comparator_status.fixture.fixture_id
    @test line_search.fixture_version == comparator_status.fixture.version
    @test line_search.n_rows == 1
    @test line_search.claim_status === :optimizer_experiment
    @test line_search.coverage_status === :not_evaluated
    @test !line_search.ai_reml_ready
    line_search_validation =
        DRModels._loconly_reml_validate_line_search_status(line_search)
    @test line_search_validation.ok
    @test line_search_validation.required_fields == line_search_schema
    line_search_row = only(line_search.rows)
    @test line_search_row.check_id === :guarded_ai_update_line_search_fixture
    @test line_search_row.estimator ===
        :guarded_ai_update_reml_optimizer_experiment
    @test line_search_row.optimizer ===
        :guarded_sparse_average_information_update
    @test line_search_row.n_records == line_search_row.n_starts
    @test line_search_row.n_accepted_records >= 1
    @test line_search_row.accepted
    @test line_search_row.best_score_norm < 1e-3
    @test line_search_row.boundary_status === :interior
    @test line_search_row.trace_status === :accepted_guarded_line_search
    @test line_search_row.claim_status === :optimizer_experiment
    @test line_search_row.coverage_status === :not_evaluated
    @test !line_search_row.ai_reml_ready
    @test occursin("no simulation, bridge, or coverage gate",
                   line_search_row.reason_not_ai_reml)
    bad_line_search_row = merge(
        line_search_row, NamedTuple{(:ai_reml_ready,)}((true,)),
    )
    bad_line_search_status =
        merge(line_search, (rows = (bad_line_search_row,),))
    bad_line_search_validation =
        DRModels._loconly_reml_validate_line_search_status(
            bad_line_search_status)
    @test !bad_line_search_validation.ok
    @test any(err -> occursin("must not mark ai_reml_ready", err),
              bad_line_search_validation.errors)
    profile_status = DRModels._loconly_reml_profile_status()
    profile_schema = DRModels._loconly_reml_profile_status_schema()
    @test profile_schema == (
        :row_id, :target, :fixture_id, :fixture_version, :parameterization,
        :axis, :step, :finite, :center_is_axis_min, :center_nll, :left_nll,
        :right_nll, :claim_status, :coverage_status, :ai_reml_ready,
        :next_gate,
    )
    @test profile_status.target === :gaussian_loconly_phylo_reml
    @test profile_status.fixture_id === comparator_status.fixture.fixture_id
    @test profile_status.fixture_version == comparator_status.fixture.version
    @test profile_status.n_rows == 2
    @test profile_status.claim_status === :internal_diagnostic
    @test profile_status.coverage_status === :not_evaluated
    @test !profile_status.ai_reml_ready
    profile_validation =
        DRModels._loconly_reml_validate_profile_status(profile_status)
    @test profile_validation.ok
    @test profile_validation.required_fields == profile_schema
    @test Tuple(r.row_id for r in profile_status.rows) ==
        (:residual_logsd_profile_axis, :phylogenetic_logsd_profile_axis)
    @test Set(r.axis for r in profile_status.rows) ==
        Set((:log_sigma, :log_sigma_phy))
    @test all(r -> r.target === :gaussian_loconly_phylo_reml,
              profile_status.rows)
    @test all(r -> r.parameterization === :log_sd, profile_status.rows)
    @test all(r -> r.finite, profile_status.rows)
    @test all(r -> r.center_is_axis_min, profile_status.rows)
    @test all(r -> r.center_nll <= min(r.left_nll, r.right_nll) + 1e-8,
              profile_status.rows)
    @test all(r -> r.claim_status === :internal_diagnostic,
              profile_status.rows)
    @test all(r -> r.coverage_status === :not_evaluated,
              profile_status.rows)
    @test all(r -> !r.ai_reml_ready, profile_status.rows)
    bad_profile_row =
        merge(first(profile_status.rows), (center_is_axis_min = false,))
    bad_profile_status =
        merge(profile_status, (rows = (bad_profile_row,), n_rows = 1))
    bad_profile_validation =
        DRModels._loconly_reml_validate_profile_status(bad_profile_status)
    @test !bad_profile_validation.ok
    @test any(err -> occursin("center is not axis minimum", err),
              bad_profile_validation.errors)
    variance_status = DRModels._loconly_reml_variance_component_status()
    variance_schema = DRModels._loconly_reml_variance_component_status_schema()
    @test variance_schema == (
        :row_id, :target, :fixture_id, :fixture_version, :estimator,
        :optimizer, :component, :logsd_parameter, :logsd_estimate,
        :sd_estimate, :variance_estimate, :point_status, :interval_status,
        :boundary_status, :claim_status, :coverage_status, :ai_reml_ready,
        :next_gate,
    )
    @test variance_status.target === :gaussian_loconly_phylo_reml
    @test variance_status.fixture_id === comparator_status.fixture.fixture_id
    @test variance_status.fixture_version == comparator_status.fixture.version
    @test variance_status.n_rows == 2
    @test variance_status.point_status === :finite_optimizer_diagnostic
    @test variance_status.interval_status === :not_evaluated
    @test variance_status.claim_status === :internal_diagnostic
    @test variance_status.coverage_status === :not_evaluated
    @test !variance_status.ai_reml_ready
    variance_validation =
        DRModels._loconly_reml_validate_variance_component_status(variance_status)
    @test variance_validation.ok
    @test variance_validation.required_fields == variance_schema
    @test Tuple(r.row_id for r in variance_status.rows) ==
        (:residual_variance_component, :phylogenetic_variance_component)
    @test Set(r.component for r in variance_status.rows) ==
        Set((:residual, :phylogenetic))
    @test Set(r.logsd_parameter for r in variance_status.rows) ==
        Set((:log_sigma, :log_sigma_phy))
    @test all(r -> r.estimator ===
              :guarded_ai_update_reml_optimizer_experiment,
              variance_status.rows)
    @test all(r -> r.optimizer ===
              :guarded_sparse_average_information_update,
              variance_status.rows)
    @test all(r -> r.point_status === :finite_optimizer_diagnostic,
              variance_status.rows)
    @test all(r -> r.interval_status === :not_evaluated,
              variance_status.rows)
    @test all(r -> r.boundary_status === :interior, variance_status.rows)
    @test all(r -> isfinite(r.logsd_estimate), variance_status.rows)
    @test all(r -> r.sd_estimate > 0, variance_status.rows)
    @test all(r -> r.variance_estimate > 0, variance_status.rows)
    @test all(r -> r.variance_estimate ≈ abs2(r.sd_estimate),
              variance_status.rows)
    @test all(r -> r.coverage_status === :not_evaluated,
              variance_status.rows)
    @test all(r -> !r.ai_reml_ready, variance_status.rows)
    bad_variance_row =
        merge(first(variance_status.rows), (interval_status = :profile_interval,))
    bad_variance_status =
        merge(variance_status, (rows = (bad_variance_row,), n_rows = 1))
    bad_variance_validation =
        DRModels._loconly_reml_validate_variance_component_status(
            bad_variance_status)
    @test !bad_variance_validation.ok
    @test any(err -> occursin("evaluated intervals", err),
              bad_variance_validation.errors)
    @test all(r -> r.target === :gaussian_loconly_phylo_reml,
              comparator_status.rows)
    @test any(r -> r.comparator_id === :internal_dense_gls_oracle &&
                   r.same_estimand_status === :same_estimand_internal,
              comparator_status.rows)
    @test any(r -> r.comparator_id === :phylolm_or_equivalent_reml &&
                   r.artifact_status === :fixture_defined &&
                   r.next_gate === :external_package_version_probe,
              comparator_status.rows)
    @test any(r -> r.dependency_status === :not_added,
              comparator_status.rows)
    bad_comparator = merge(first(comparator_status.rows), (target = :q4_phylo,))
    bad_comparator_validation =
        DRModels._loconly_reml_validate_external_comparator_rows((bad_comparator,))
    @test !bad_comparator_validation.ok
    @test any(err -> occursin("wrong target", err),
              bad_comparator_validation.errors)

    fixture = comparator_status.fixture
    @test fixture.target === :gaussian_loconly_phylo_reml
    @test fixture.estimator === :supplied_variance_reml
    @test fixture.parameterization === :log_sd
    @test fixture.n_obs == length(fixture.y) == length(fixture.species) == size(fixture.X, 1)
    @test fixture.n_species == size(fixture.Sigma_phy, 1) == size(fixture.Sigma_phy, 2)
    @test fixture.seed_registry.deterministic
    @test !fixture.seed_registry.rng_used
    @test fixture.coverage_status === :not_evaluated
    @test !fixture.ai_reml_ready
    sp = collect(fixture.species)
    X_fixture = fixture.X
    y_fixture = collect(fixture.y)
    V_fixture = fixture.known_sigma^2 .* I(fixture.n_obs) .+
        fixture.known_sigma_phy^2 .* fixture.Sigma_phy[sp, sp]
    ch_fixture = cholesky(Symmetric(Matrix(V_fixture)))
    info_fixture = X_fixture' * (ch_fixture \ X_fixture)
    beta_fixture = info_fixture \ (X_fixture' * (ch_fixture \ y_fixture))
    resid_fixture = y_fixture .- X_fixture * beta_fixture
    ml_fixture = 0.5 * (
        fixture.n_obs * log(2π) +
        logdet(ch_fixture) +
        dot(resid_fixture, ch_fixture \ resid_fixture)
    )
    restricted_penalty_fixture =
        sum(log, LinearAlgebra.diag(cholesky(Symmetric(info_fixture)).U))
    reml_constant_offset_fixture = 0.5 * size(X_fixture, 2) * log(2π)
    reml_fixture = ml_fixture +
        restricted_penalty_fixture - reml_constant_offset_fixture
    @test beta_fixture ≈ collect(fixture.reference.beta_hat) rtol = 1e-8 atol = 1e-8
    @test info_fixture ≈ fixture.reference.fixed_effect_information rtol = 1e-8 atol = 1e-8
    @test ml_fixture ≈ fixture.reference.ml_nll rtol = 1e-8 atol = 1e-8
    @test reml_fixture ≈ fixture.reference.reml_nll rtol = 1e-8 atol = 1e-8
    @test restricted_penalty_fixture ≈ fixture.reference.restricted_penalty rtol = 1e-8 atol = 1e-8
    @test fixture.reference.boundary_status === :interior
    bad_fixture = merge(fixture, (target = :q4_phylo,))
    bad_fixture_validation =
        DRModels._loconly_reml_validate_external_comparator_fixture(bad_fixture)
    @test !bad_fixture_validation.ok
    @test any(err -> occursin("wrong target", err),
              bad_fixture_validation.errors)

    Random.seed!(20260623)
    elapsed = Float64[]
    for G in (8, 16, 32)
        phy = random_balanced_tree(G; branch_length = 0.2)
        species = collect(1:G)
        X = hcat(ones(G), range(-0.5, 0.5; length = G))
        y = X * [0.1, 0.3] .+ 0.35 .* randn(G)
        prob = DRModels.make_loc_problem(phy, y, X; species = species)
        push!(elapsed, @elapsed begin
            comp = DRModels._loconly_reml_components(prob, log(0.35), log(0.2))
            trace_diag = DRModels._loconly_takahashi_trace_diagnostic(prob, log(0.35), log(0.2))
            pev_diag = DRModels._loconly_takahashi_pev_diagnostic(prob, log(0.35), log(0.2))
            @test comp.converged
            @test isfinite(comp.nll)
            @test trace_diag.finite
            @test pev_diag.finite
            @test length(pev_diag.leaf_posterior_variance) == G
        end)
    end
    @test all(isfinite, elapsed)
    @test all(>=(0), elapsed)

    phy = random_balanced_tree(10; branch_length = 0.25)
    X1 = hcat(ones(10), range(-0.5, 0.5; length = 10))
    y1 = X1 * [0.1, 0.2]
    prob_single = DRModels.make_loc_problem(phy, y1, X1; species = collect(1:10))
    species_double = repeat(1:10, inner = 2)
    X2 = X1[species_double, :]
    y2 = y1[species_double]
    prob_double = DRModels.make_loc_problem(phy, y2, X2; species = species_double)
    pev_single = DRModels._loconly_takahashi_pev_diagnostic(prob_single, log(0.4), log(0.3))
    pev_double = DRModels._loconly_takahashi_pev_diagnostic(prob_double, log(0.4), log(0.3))
    @test pev_single.finite
    @test pev_double.finite
    @test all(pev_double.leaf_posterior_variance .<= pev_single.leaf_posterior_variance .+ 1e-12)
end
