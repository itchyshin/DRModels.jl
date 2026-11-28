using DRModels
using StatsModels
using Test
using Random
using LinearAlgebra

@testset "prior-fit convergence metadata preserves the ML fit contract" begin
    x = repeat([-1.0, 0.0, 1.0, 2.0], 10)
    y = 0.4 .+ 0.7 .* x .+ repeat([-0.2, 0.1, 0.0, 0.1], 10)
    dat = (; y, x)
    form = bf(@formula(y ~ 1 + x))

    plain = drm(form, Gaussian(); data=dat)
    plain_report = check_drm(plain)
    prior = drm(form, Gaussian(); data=dat,
        priors=(prior=["normal(0,2)"], class=["fixef"], coef=["x"]))
    prior_report = check_drm(prior)

    # The prior path retains the data likelihood separately from the MAP
    # objective. The prior-only fields are absent from no-prior output.
    @test !(:penalised in keys(plain_report))
    @test !(:objective in keys(plain_report))
    @test :penalised in keys(prior_report)
    @test prior_report.penalised
    @test prior_report.logLik ≈ loglik(prior) atol=1e-10
    @test prior_report.objective ≈ prior_report.logLik + prior_report.log_prior atol=1e-10
    @test prior_report.objective ≈ -prior.penalty.objective atol=1e-10
    @test prior_report.se_type === :penalised
    @test prior_report.pd_hessian_unpenalised
    @test isempty(prior_report.prior_held_params)
    @test prior_report.convergence_status === :converged
end

@testset "legacy phylogenetic MAP fits expose penalized and data-only diagnostics" begin
    rng = MersenneTwister(824)
    G, m = 8, 4
    tree = random_balanced_tree(G; branch_length=0.3)
    C = sigma_phy_dense(tree; σ²_phy=1.0)
    d = sqrt.(diag(C)); K = C ./ (d * d')
    species = repeat(1:G, inner=m)
    x = randn(rng, G * m)
    u = 0.7 .* (cholesky(Symmetric(K)).L * randn(rng, G))
    y = 0.2 .+ 0.4 .* x .+ u[species] .+ 0.5 .* randn(rng, G * m)
    form = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1))
    fit = drm(form, Gaussian(); data=(; y, x, species), tree=tree,
              penalty=drm_phylo_penalty(sd_u=0.6, sd_alpha=0.05))
    report = @test_logs (:warn, r"AIC/BIC.*unpenalized refits") match_mode = :any check_drm(fit)

    @test report.penalised
    @test report.logLik ≈ fit.loglik atol=1e-10
    @test report.objective ≈ fit.loglik - fit.phylo_penalty atol=1e-8
    @test report.objective ≈ report.logLik + report.log_prior atol=1e-8
    @test report.se_type === :penalised
    @test report.pd_hessian_unpenalised isa Bool
    @test hasproperty(report, :prior_held_params)
    @test_throws ArgumentError aic(fit)
    @test_throws ArgumentError bic(fit)
end

@testset "SD-only and mixed prior fits expose penalized diagnostics" begin
    x = repeat([-1.0, 0.0, 1.0], 4)
    y = [0.2, 0.8, 0.5, 1.4, 1.0, 1.6, 2.0, 1.3, 2.2, 1.8, 2.4, 2.1]
    site = repeat(["a", "b", "c", "d"], inner=3)
    data = (; y, x, site)
    form = bf(@formula(y ~ 1 + x + (1 | site)))
    sd_only = (prior=["gamma(0.6666666666666666,2)"],
               class=["ranef_sd"], coef=["(Intercept)"],
               dpar=["mu"], group=["site"], scale=["natural_sd"])
    residual_only = (prior=["gamma(0.6666666666666666,2)"],
                     class=["residual_sd"], coef=["(Intercept)"],
                     dpar=["sigma"], scale=["natural_sd"])
    mixed = (prior=["normal(0,2)", "gamma(0.6666666666666666,2)",
                    "gamma(0.6666666666666666,2)"],
             class=["fixef", "residual_sd", "ranef_sd"],
             coef=["x", "(Intercept)", "(Intercept)"],
             dpar=["mu", "sigma", "mu"],
             group=["", "", "site"],
             scale=["link", "natural_sd", "natural_sd"])

    for (model, table) in ((form, sd_only),
                           (bf(@formula(y ~ 1 + x)), residual_only),
                           (form, mixed))
        fit = drm(model, Gaussian(); data=data, priors=table)
        report = check_drm(fit)
        @test report.penalised
        @test report.objective ≈ -fit.nll(fit.theta) atol=1e-10
        @test report.logLik ≈ loglik(fit) atol=1e-10
        @test report.objective ≈ report.logLik + report.log_prior atol=1e-10
        @test fit.penalty.log_prior ≈ report.log_prior atol=1e-10
        @test report.se_type === :penalised
        @test report.pd_hessian_unpenalised isa Union{Bool,Missing}
        @test hasproperty(report, :prior_held_params)
    end
end

@testset "a prior-held alias remains converged and is named" begin
    x = repeat([-1.0, 0.0, 1.0, 2.0], 10)
    orthogonal = repeat([1.0, -1.0, -1.0, 1.0], 10)
    # Keep the design full rank so prior resolution accepts it, but make the
    # data-only curvature effectively singular along the x/z contrast.
    z = 2 .* x .+ 1e-6 .* orthogonal
    y = 0.4 .+ 0.7 .* x .+ repeat([-0.2, 0.1, 0.0, 0.1], 10)
    dat = (; y, x, z)
    form = bf(@formula(y ~ 1 + x + z))
    priors = (prior=["normal(0,2)", "normal(0,2)"],
              class=["fixef", "fixef"], coef=["x", "z"])

    fit = drm(form, Gaussian(); data=dat, priors=priors)
    report = check_drm(fit)

    @test report.penalised
    @test report.convergence_status === :converged
    @test report.vcov_posdef
    @test !report.pd_hessian_unpenalised
    @test Set(string(p.name) for p in report.prior_held_params) == Set(["x", "z"])
    @test all(p -> !isempty(string(p.kind)), report.prior_held_params)
    @test report.se_type === :penalised
end

@testset "prior-held correlations identify their working coordinate" begin
    random_corr = (class=:random_effect_correlation, dpar=:mu, group="site",
                   from_coef="(Intercept)", to_coef="x", index=1)
    random_held = DRModels._prior_held_parameters(
        [2.0 0 0; 0 1 0; 0 0 0], (correlation_row=random_corr,))
    @test length(random_held) == 1
    @test only(random_held).name === Symbol("rho[site|mu|(Intercept)|x]")
    @test only(random_held).kind === :random_effect_correlation

    residual_corr = (class=:residual_correlation, dpar=:rho12,
                     coef="(Intercept)", index=2)
    residual_held = DRModels._prior_held_parameters(
        [1.0 0 0; 0 0 0; 0 0 2], (correlation_row=residual_corr,))
    @test length(residual_held) == 1
    @test only(residual_held).name === :rho12
    @test only(residual_held).kind === :residual_correlation
end

@testset "four-state convergence checks optimizer, gradient, boundary and Hessian" begin
    good = (
        optimizer_ok=true,
        objective_ok=true,
        relative_gradient=0.0,
        geometry_ok=true,
        at_boundary=false,
        hessian_pd=true,
        standard_errors_ok=true,
        inflated_se=false,
    )
    classify(x) = DRModels._fit_convergence_status(; x...)
    @test classify(good) === :converged
    @test classify(merge(good, (optimizer_ok=false,))) === :not_converged
    @test classify(merge(good, (relative_gradient=Inf,))) === :not_converged
    @test classify(merge(good, (relative_gradient=2e-4,))) === :not_converged
    @test classify(merge(good, (geometry_ok=false,))) === :degenerate
    @test classify(merge(good, (at_boundary=true, boundary_kkt_valid=true,))) === :boundary
    @test classify(merge(good, (at_boundary=true, boundary_kkt_valid=true,
                                relative_gradient=2e-4,))) === :not_converged
    @test classify(merge(good, (at_boundary=true, boundary_kkt_valid=false,))) === :not_converged
    @test classify(merge(good, (at_boundary=true, boundary_kkt_valid=nothing,))) === :degenerate
    @test classify(merge(good, (at_boundary=true, optimizer_ok=false,))) === :not_converged
    @test classify(merge(good, (at_boundary=true, objective_ok=false,))) === :not_converged
    @test classify(merge(good, (hessian_pd=false,))) === :degenerate
    @test classify(merge(good, (standard_errors_ok=false,))) === :degenerate
end

@testset "projected boundary scores use lower and upper KKT signs" begin
    lower_ok = DRModels._projected_kkt_gradient([2.0, 3.0], [1], Int[])
    @test lower_ok == [0.0, 3.0]
    @test DRModels._projected_kkt_gradient([-2.0, 3.0], [1], Int[]) === false

    upper_ok = DRModels._projected_kkt_gradient([-2.0, 3.0], Int[], [1])
    @test upper_ok == [0.0, 3.0]
    @test DRModels._projected_kkt_gradient([2.0, 3.0], Int[], [1]) === false
    @test DRModels._projected_kkt_gradient([0.0], [1], [1]) === nothing
    @test DRModels._projected_kkt_gradient([NaN], [1], Int[]) === nothing
end

@testset "near-boundary transformed correlations require full stationarity" begin
    good = (optimizer_ok=true, objective_ok=true, relative_gradient=0.0,
            geometry_ok=true, at_boundary=true, boundary_kkt_valid=true,
            hessian_pd=true, standard_errors_ok=true, inflated_se=false)
    classify(x) = DRModels._fit_convergence_status(; x...)
    # A near-boundary correlation is a threshold diagnostic on an atanh/logit
    # transformed parameter, not an optimizer box constraint to project away.
    @test classify(good) === :boundary
    @test classify(merge(good, (relative_gradient=2e-4,))) === :not_converged
    @test DRModels._projected_kkt_gradient([2.0], Int[], Int[]) == [2.0]
end

@testset "conditional log-SD walls map only to actual optimizer-bound coordinates" begin
    x = repeat([-1.0, 0.0, 1.0], 4)
    y = [0.2, 0.8, 0.5, 1.4, 1.0, 1.6, 2.0, 1.3, 2.2, 1.8, 2.4, 2.1]
    site = repeat(["a", "b", "c", "d"], inner=3)
    fit = drm(bf(@formula(y ~ 1 + x + (1 | site))), Gaussian(); data=(; y, x, site))
    fit.scales[:cond_logsd_wall] = [-8.0]
    unmapped = DRModels._boundary_constraint_indices(fit)
    @test !unmapped.known

    idx = first(DRModels._block_range(fit, :resd))
    fit.theta[idx] = -8.0
    fit.vcov[idx, :] .= NaN
    fit.vcov[:, idx] .= NaN
    # The independent inner-mode wall remains unmapped even when an outer
    # variance coordinate happens to sit at its own optimizer floor.
    @test !DRModels._boundary_constraint_indices(fit).known
    delete!(fit.scales, :cond_logsd_wall)
    fit.scales[:optimizer_logsd_bound] = [-8.0]
    mapped = DRModels._boundary_constraint_indices(fit)
    @test mapped.known
    @test mapped.active
    @test idx in mapped.lower

    fit.theta[idx] = 3.0
    upper_mapped = DRModels._boundary_constraint_indices(fit)
    @test upper_mapped.known
    @test upper_mapped.active
    @test idx in upper_mapped.upper
end

@testset "skipping the covariance does not pass the Hessian convergence gate" begin
    x = repeat([-1.0, 0.0, 1.0], 4)
    y = [0.2, 0.8, 0.5, 1.4, 1.0, 1.6, 2.0, 1.3, 2.2, 1.8, 2.4, 2.1]
    fit = drm(bf(@formula(y ~ 1 + x)), Gaussian(); data=(; y, x),
              control=DRModels.DRMControl(se=false))
    @test fit.control_info.se === :skipped
    @test convergence_status(fit) === :degenerate
    @test !is_converged(fit)
end
