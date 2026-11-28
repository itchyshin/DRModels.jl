using DRModels
using ForwardDiff
using Random
using Test

@testset "Gamma prior on natural residual SD" begin
    exact = drm_prior_residual_sd_gamma(shape=3.3, rate=0.1)
    @test exact.shape === 3.3
    @test exact.rate === 0.1
    @test exact.mean ≈ 33.0
    shape, rate = 2.0, 3.0
    penalty(logsd) = rate * exp(logsd) - (shape - 1) * logsd -
                     shape * log(rate) + DRModels.loggamma(shape)

    # Independent hand calculation at an interior SD, including the natural-
    # SD normalizer and the absence of a log-SD Jacobian.
    eta = log(0.8)
    @test penalty(eta) ≈ 0.425918974 atol=1e-9
    @test penalty(log(1 / 3)) ≈ minimum(penalty.(-8:0.001:3)) atol=2e-5
    @test ForwardDiff.derivative(penalty, eta) ≈ rate * exp(eta) - (shape - 1) atol=1e-12
    @test ForwardDiff.derivative(x -> ForwardDiff.derivative(penalty, x), eta) ≈ rate * exp(eta) atol=1e-12

    row = (kind=:residual_sd_gamma, index=1, shape=shape, rate=rate)
    theta = [eta]
    @test DRModels._fixed_prior_value([row], theta) ≈ penalty(eta) atol=1e-12
    @test ForwardDiff.gradient(t -> DRModels._fixed_prior_value([row], t), theta)[1] ≈
          ForwardDiff.derivative(penalty, eta) atol=1e-12
    @test ForwardDiff.hessian(t -> DRModels._fixed_prior_value([row], t), theta)[1, 1] ≈
          ForwardDiff.derivative(x -> ForwardDiff.derivative(penalty, x), eta) atol=1e-12

    # A log-SD density would move this mode to 2/3 and is deliberately rejected.
    natural_eta_mode = log((shape - 1) / rate)
    jacobian_eta_mode = log(shape / rate)
    @test exp(natural_eta_mode) == 1 / 3
    @test penalty(natural_eta_mode) < penalty(jacobian_eta_mode)
    @test penalty(jacobian_eta_mode) - jacobian_eta_mode <
          penalty(natural_eta_mode) - natural_eta_mode

    data = (y=[1.0, 2.2, 2.9, 4.1, 4.8, 6.3], x=collect(0.0:5.0))
    form = bf(@formula(y ~ x))
    table = (prior=["gamma(0.6666666666666666,2)"],
             class=["residual_sd"], coef=["(Intercept)"],
             dpar=["sigma"], scale=["natural_sd"])
    fit = drm(form, Gaussian(); data=data, priors=table)
    fitted_row = only(fit.penalty.rows)
    fitted_eta = fit.theta[fitted_row.index]
    @test fitted_row.class === :residual_sd
    @test DRModels._prior_kind_label(fit) == "SD-prior"
    @test fit.penalty.fixed_penalty ≈ penalty(fitted_eta) atol=1e-10
    @test fit.penalty.objective ≈ fit.penalty.data_nll(fit.theta) + penalty(fitted_eta) atol=1e-9
    @test fit.loglik ≈ -fit.penalty.data_nll(fit.theta) atol=1e-9
    replay = update(fit, form; data=data)
    @test only(replay.penalty.specs) isa DRModels.ResidualSDGammaPrior
    @test only(replay.penalty.specs).shape == shape
    @test only(replay.penalty.specs).rate == rate
    @test replay.theta ≈ fit.theta atol=1e-6
    bootstrap_kwargs = DRModels._bootstrap_refit_kwargs(fit)
    @test bootstrap_kwargs.priors == fit.penalty.specs
    bootstrap_refit = drm(form, Gaussian(); data=data, bootstrap_kwargs...)
    retained_bootstrap_refit = DRModels._bootstrap_retain_prior_table(fit, bootstrap_refit)
    @test retained_bootstrap_refit.penalty.input_table == table
    replay_row = only(filter(r -> r.class === :residual_sd,
        DRModels._resolve_priors_native(form, Gaussian(); data=data,
            priors=retained_bootstrap_refit.penalty.specs)))
    @test (replay_row.dpar, replay_row.coef, replay_row.shape, replay_row.rate) ==
          (fitted_row.dpar, fitted_row.coef, fitted_row.shape, fitted_row.rate)
    @test bootstrap_refit.penalty.log_prior ≈ -bootstrap_refit.penalty.fixed_penalty atol=1e-10
    boot = bootstrap_result(fit; data=data, B=2, rng=MersenneTwister(71),
                            check_converged=false, failures=:skip)
    @test boot.estimation_method === :MAP
    @test boot.refit_estimation_method === :MAP
    @test boot.attempted == 2 && boot.used == 2
    @test boot.penalised && all(row -> row.penalised, boot.summary)
    @test_throws ArgumentError aic(fit)
end
