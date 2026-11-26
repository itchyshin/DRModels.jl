# refs #1542, #1543 — finite loglik / AIC / BIC with a warning when !is_converged
# (Shinichi, 26 Nov 2026): sentinel loglik still returns NaN without this warn path.
using DRModels
using DRModels: Poisson, Student
using Test, Random, Logging, Distributions

@testset "unconverged infocrit warnings (#1542 / #1543)" begin
  ############################################################ ZIP RE, :degenerate (#1542)
    Random.seed!(112613)
    n = 100
    x = randn(n)
    g = repeat(1:5, inner=20)
    μ = exp.(0.2 .+ 0.5 .* x)
    y = [rand() < 0.25 ? 0 : rand(Distributions.Poisson(μ[i])) for i in 1:n]
    data = (; y, x, g)
    zip_fit = with_logger(NullLogger()) do
        drm(bf(@formula(y ~ x + (1 | g)), @formula(zi ~ 1)), Poisson(); data = data, se = false)
    end
    @test convergence_status(zip_fit) === :degenerate
    @test !is_converged(zip_fit)
    @test isfinite(loglik(zip_fit))
    aic_val = @test_logs match_mode = :any (:warn, r"not converged") aic(zip_fit)
    @test isfinite(aic_val)
    bic_val = @test_logs match_mode = :any (:warn, r"not converged") bic(zip_fit)
    @test isfinite(bic_val)
    ll = @test_logs match_mode = :any (:warn, r"not converged") loglik(zip_fit)
    @test isfinite(ll)

    ############################################################ non-sentinel degenerate (#1543)
    hand = DrmFit(Poisson(), [:mu => 1:2], [:mu => ["(Intercept)", "x"]], [0.1, 0.2],
        [1.0 1.0; 1.0 1.0], -223.42, 80, true, Dict(:mu => ones(80)),
        Dict(:mu => Float64.(0:79)), Dict{Symbol,Vector{Float64}}())
    @test convergence_status(hand) === :degenerate
    @test isfinite(@test_logs match_mode = :any (:warn, r"not converged") aic(hand))
    @test isfinite(@test_logs match_mode = :any (:warn, r"not converged") bic(hand))

    ############################################################ Wald confint on singular Hessian
    bound_hand = DrmFit(Poisson(), [:mu => 1:2], [:mu => ["(Intercept)", "x"]], [0.0, 0.2],
        [0.01 0.0; 0.0 0.0], -10.0, 10, true, Dict(:mu => ones(10)),
        Dict(:mu => Float64.(0:9)), Dict{Symbol,Vector{Float64}}())
    @test_logs match_mode = :any (:warn, r"unbounded interval endpoint") confint(bound_hand)

    ############################################################ healthy fit stays quiet
    yg = 0.2 .+ 0.5 .* x[1:60] .+ 0.3 .* randn(60)
    good = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = (; y = yg, x = x[1:60]))
    @test is_converged(good)
    @test_logs aic(good)
    @test_logs bic(good)
    @test_logs loglik(good)
end

@testset "temporal interpretability boundary in convergence_status" begin
    L = readlines(joinpath(@__DIR__, "fixtures", "temporal", "ou_boundary_panel.csv"))
    h = split(L[1], ","); r = split.(L[2:end], ",")
    col(n) = getindex.(r, findfirst(==(n), h))
    d = (id = String.(col("id")), t = parse.(Float64, col("t")),
         x = parse.(Float64, col("x")), y = parse.(Float64, col("y")))
    fOU = bf(@formula(y ~ x + temporal(1 | id, t, ou)), @formula(sigma ~ 1))
    fit = with_logger(NullLogger()) do; drm(fOU, Gaussian(); data = d); end
    @test convergence_status(fit) in (:boundary, :degenerate)
    @test !is_converged(fit)
    @test isfinite(@test_logs (:warn, r"not converged") aic(fit))
end