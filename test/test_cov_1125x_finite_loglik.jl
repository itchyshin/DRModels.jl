# test_cov_1125x_finite_loglik.jl — finite-input and loglik sentinel helpers (src/finite_inputs.jl).
using DRModels
using Test
using Tables

_msg(f) = try
    f()
    ""
catch e
    e isa ArgumentError || rethrow()
    e.msg
end

function _with_loglik(fit::DrmFit, ll::Real)
    return DrmFit(fit.family, fit.blocks, fit.coefnames, fit.theta, fit.vcov, Float64(ll),
                  fit.nobs, fit.converged, fit.means, fit.obs, fit.scales, fit.formula,
                  fit.nll, fit.nllgrad, fit.ranef, fit.estim_method, fit.reml_loglik,
                  fit.ml_loglik, fit.marginal, fit.phylo_penalty, fit.penalty,
                  fit.iterations, fit.phylo_scale, fit.control_info, fit.uncertainty,
                  fit.geometry)
end

@testset "finite_inputs: predictors respect observed mask (#1045)" begin
    X = [1.0 2.0; 3.0 NaN; 5.0 6.0]
    observed = [true, false, true]
    @test DRModels._require_finite_inputs(; predictors = (X, ["x1", "x2"]),
                                          observed = observed) === nothing
    msg = _msg(() -> DRModels._require_finite_inputs(; predictors = (X, ["x1", "x2"]),
                                                     observed = [true, true, true]))
    @test occursin("x2", msg) && occursin("row 2", msg)
end

@testset "finite_inputs: table-shaped arrays" begin
    tbl = (a = [1.0, 2.0], b = [3.0, 4.0])
    @test DRModels._require_finite_inputs(; weights = tbl) === nothing
    bad = (a = [1.0, NaN], b = [3.0, 4.0])
    msg = _msg(() -> DRModels._require_finite_inputs(; weights = bad))
    @test occursin("weights", msg)
end

@testset "finite_inputs: loglik sentinels and display" begin
    @test DRModels._loglik_is_sentinel(-1e18)
    @test DRModels._loglik_is_sentinel(-floatmax(Float64))
    @test !DRModels._loglik_is_sentinel(-12.5)
    @test DRModels._loglik_is_failed(-1e18, true)
    @test DRModels._loglik_is_failed(1e20, false)
    @test !DRModels._loglik_is_failed(-4.0, true)
    @test !DRModels._coefficients_finite([0.1, NaN])
    y = [0.2, -0.1]
    fit = DrmFit(Gaussian(), [:mu => 1:1], [:mu => ["(Intercept)"]],
                 [0.3], [1.0;;], -1e18, 2, true,
                 Dict(:mu => y), Dict(:mu => y), Dict(:sigma => ones(2)))
    @test isnan(DRModels._display_loglik(fit))
    @test DRModels._display_loglik(_with_loglik(fit, -8.5)) == -8.5
end