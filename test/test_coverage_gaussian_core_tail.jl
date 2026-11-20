# DJ1: uncovered branches of src/gaussian_core.jl that open drafts are not
# rewriting. Predict, `_drm_gaussian_fit`, `_fit_for_simulate_newdata`, and the
# coefnames field are left to the drafts and to batch DJ-R.
using DRModels, Test, Random, LinearAlgebra, Statistics
import Distributions

struct _DJ1Cols
    y::Int
end

function _bare(fam, means, scales)
    n = length(means[:mu])
    return DrmFit(fam,
                  [:mu => 1:1],
                  [:mu => ["(Intercept)"]],
                  [0.0],
                  reshape([NaN], 1, 1),
                  0.0, n, true,
                  means,
                  Dict(:mu => zeros(n)),
                  scales)
end

@testset "table-column lookup and missing-column classification" begin
    nt = (y = [1.0, 2.0], x = [3.0, 4.0])
    @test DRModels._table_column(nt, :y) == [1.0, 2.0]
    @test DRModels._table_column(Dict(:y => [5.0]), :y) == [5.0]
    @test DRModels._table_column(Dict("y" => [6.0]), :y) == [6.0]
    holder = (z = 7.0,)
    # A NamedTuple miss falls through to `getproperty`. Julia 1.12+ raises
    # `FieldError`; 1.10 raises an `ErrorException` (see `_missing_column_lookup`).
    @test try
        DRModels._table_column(holder, :missing_col)
        false
    catch err
        if isdefined(Base, :FieldError)
            err isa Base.FieldError && err.field === :missing_col && err.type == typeof(holder)
        else
            err isa ErrorException && occursin("no field", sprint(showerror, err))
        end
    end
    @test DRModels._table_column(_DJ1Cols(4), :y) == 4

    @test DRModels._missing_column_lookup(KeyError(:y))
    @test DRModels._missing_column_lookup(ArgumentError("no such column"))
    @test DRModels._missing_column_lookup(ErrorException("column y not found"))
    @test DRModels._missing_column_lookup(ErrorException("type has no field y"))
    @test !DRModels._missing_column_lookup(ErrorException("domain error"))
    @test !DRModels._missing_column_lookup(DimensionMismatch("sizes"))
    field_err = try
        _DJ1Cols(1).nope
    catch e
        e
    end
    if isdefined(Base, :FieldError)
        @test field_err isa Base.FieldError
    else
        @test field_err isa ErrorException
        @test occursin("no field", sprint(showerror, field_err))
    end
    @test DRModels._missing_column_lookup(field_err)
end

@testset "rho12 inverse link and its derivative" begin
    η = [-1.5, 0.0, 0.8]
    ρ = DRModels._param_response(DRModels.Gaussian(), :rho12, η)
    @test ρ ≈ DRModels.RHO_GUARD .* tanh.(η) atol = 1e-12
    dρ = DRModels._link_deriv(DRModels.Gaussian(), :rho12, η)
    # d/dη [RHO_GUARD * tanh(η)] = RHO_GUARD * sech(η)^2.
    @test dρ ≈ DRModels.RHO_GUARD .* (1 .- tanh.(η) .^ 2) atol = 1e-12
    # Finite difference of the inverse link, so the two helpers cannot drift apart.
    h = 1e-6
    fd = (DRModels._param_response(DRModels.Gaussian(), :rho12, η .+ h) .-
          DRModels._param_response(DRModels.Gaussian(), :rho12, η .- h)) ./ (2h)
    @test dρ ≈ fd atol = 1e-8
    @test_throws ArgumentError DRModels._param_response(DRModels.Poisson(), :nu, [0.0])
    @test_throws ArgumentError DRModels._param_response(DRModels.Gaussian(), :not_a_parameter, [0.0])
    @test_throws ArgumentError DRModels._link_deriv(DRModels.Poisson(), :nu, [0.0])
    @test_throws ArgumentError DRModels._link_deriv(DRModels.Gaussian(), :not_a_parameter, [0.0])
end

@testset "zero-inflated, hurdle, and zero-one-beta draws" begin
    μ = [1.5, 2.0, 0.5, 3.0]
    zi = ones(4)
    fit = _bare(DRModels.Poisson(), Dict(:mu => μ), Dict(:zi => zi))
    draw = DRModels._simulate_once(fit, MersenneTwister(1))
    @test draw == zeros(4)
    fit_h = _bare(DRModels.Poisson(), Dict(:mu => μ), Dict(:hu => ones(4)))
    @test DRModels._simulate_once(fit_h, MersenneTwister(2)) == zeros(4)

    # Structural-zero probability 0: the count is Poisson(μ). 4 Monte Carlo
    # standard errors of the mean, seed fixed.
    n = 4000
    μ0 = 2.5
    fit0 = _bare(DRModels.Poisson(), Dict(:mu => fill(μ0, n)), Dict(:zi => zeros(n)))
    draws = DRModels._simulate_once(fit0, MersenneTwister(3))
    se = sqrt(μ0 / n)
    @test abs(mean(draws) - μ0) < 4 * se
    @test all(draws .>= 0)

    σ = fill(0.8, n)
    θ = 1 / 0.8^2
    fit_nb = _bare(DRModels.NegBinomial2(), Dict(:mu => fill(μ0, n)),
                   Dict(:sigma => σ, :zi => ones(n)))
    @test DRModels._simulate_once(fit_nb, MersenneTwister(4)) == zeros(n)
    fit_nb0 = _bare(DRModels.NegBinomial2(), Dict(:mu => fill(μ0, n)),
                    Dict(:sigma => σ, :zi => zeros(n)))
    nb = DRModels._simulate_once(fit_nb0, MersenneTwister(5))
    # NB2 mean is μ; variance is μ + μ^2/θ.
    var_nb = μ0 + μ0^2 / θ
    @test abs(mean(nb) - μ0) < 4 * sqrt(var_nb / n)

    λ = 1.5
    tp = [DRModels._rand_positive_poisson(MersenneTwister(1000 + i), λ) for i in 1:n]
    @test all(tp .> 0)
    trunc_mean = λ / (1 - exp(-λ))
    trunc_var = trunc_mean * (1 + λ - trunc_mean)  # Var(Poisson | > 0)
    @test abs(mean(tp) - trunc_mean) < 4 * sqrt(trunc_var / n)

    φ = 12.0
    μb = 0.4
    zob1 = _bare(DRModels.ZeroOneBeta(),
                 Dict(:mu => fill(μb, 20)),
                 Dict(:beta_mu => fill(μb, 20), :sigma => fill(1 / sqrt(φ), 20),
                      :zoi => ones(20), :coi => ones(20)))
    @test DRModels._simulate_once(zob1, MersenneTwister(6)) == ones(20)
    zob0 = _bare(DRModels.ZeroOneBeta(),
                 Dict(:mu => fill(μb, 20)),
                 Dict(:beta_mu => fill(μb, 20), :sigma => fill(1 / sqrt(φ), 20),
                      :zoi => ones(20), :coi => zeros(20)))
    @test DRModels._simulate_once(zob0, MersenneTwister(7)) == zeros(20)
    m = 3000
    zob = _bare(DRModels.ZeroOneBeta(),
                Dict(:mu => fill(μb, m)),
                Dict(:beta_mu => fill(μb, m), :sigma => fill(1 / sqrt(φ), m),
                     :zoi => zeros(m), :coi => fill(0.5, m)))
    zb = DRModels._simulate_once(zob, MersenneTwister(8))
    @test all(0 .< zb .< 1)
    vbeta = μb * (1 - μb) / (φ + 1)
    @test abs(mean(zb) - μb) < 4 * sqrt(vbeta / m)
end
