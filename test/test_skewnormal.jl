# Skew-normal family: location–scale–shape regression with an asymmetric
# Gaussian. A formula per parameter — μ (mean, identity), σ (SD, log link), ν
# (slant α, identity). The public parameterisation is the moment form (μ = mean,
# σ = SD); data are simulated from a skew-normal with a known mean/SD/slant and
# the fit recovers all three. Fixed effects, maximum likelihood. Mirrors drmTMB's
# `skew_normal`.
using DRModels
using Test, Random, Statistics
using Distributions: SkewNormal as DSkewNormal

# Map the public moment params (mean μ, SD σ, slant α) to Azzalini's internal
# (ξ, ω, α), so we can draw from Distributions.SkewNormal(ξ, ω, α) and know the
# true mean/SD exactly.
function _moment_to_internal(μ, σ, α)
    δ = α / sqrt(1 + α^2)
    ω = σ / sqrt(1 - 2 * δ^2 / π)
    ξ = μ - ω * δ * sqrt(2 / π)
    return ξ, ω
end

@testset "Skew-normal location–scale–shape — recovery" begin
    Random.seed!(20260610)
    n = 800
    x = randn(n)
    β = [1.0, -0.6]; σ = 1.2; α = 4.0          # mean intercept/slope, SD, slant
    μ = β[1] .+ β[2] .* x                       # per-observation mean
    y = similar(μ)
    for i in 1:n
        ξ, ω = _moment_to_internal(μ[i], σ, α)
        y[i] = rand(DSkewNormal(ξ, ω, α))       # mean = μ[i], SD = σ, slant = α
    end
    data = (; y, x)

    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), SkewNormal(); data = data)

    @test is_converged(fit)
    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.12        # mean intercept
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.10        # mean slope
    @test exp(coef(fit, :sigma)[1]) ≈ σ atol = 0.15   # SD
    @test coef(fit, :nu)[1] ≈ α atol = 2.0            # slant (weakly identified — loose)
    @test sign(coef(fit, :nu)[1]) == sign(α)          # at least gets the skew direction right
    @test isfinite(loglik(fit))
end
# --- appended to test/test_skewnormal.jl -------------------------------------
# The skew-normal simulator: `_simulate_once` had no `SkewNormal` branch, so it
# fell through to `error("simulate: not yet supported for SkewNormal.")` and
# every parametric-bootstrap replicate died at step one. Two things are asserted
# here: that `simulate` RUNS, and that it uses the SAME moment→internal mapping
# as the likelihood — a simulator parameterised in (ξ, ω) instead of the public
# (μ = mean, σ = SD) would run fine and make the bootstrap quietly wrong.
@testset "Skew-normal — simulate and parametric bootstrap" begin
    Random.seed!(20260824)
    n = 400
    x = randn(n)
    β = [1.0, -0.6]; σ = 1.2; α = 4.0
    μ = β[1] .+ β[2] .* x
    y = similar(μ)
    for i in 1:n
        ξ, ω = _moment_to_internal(μ[i], σ, α)
        y[i] = rand(DSkewNormal(ξ, ω, α))
    end
    data = (; y, x)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)),
              SkewNormal(); data = data)
    @test is_converged(fit)

    # (1) simulate must not reach the "not yet supported" fallback.
    rep = simulate(fit; rng = MersenneTwister(11))
    @test rep isa Vector{Float64}
    @test length(rep) == n
    @test all(isfinite, rep)

    # (2) PARAMETERISATION LOCK. Averaged over many replicate draws at the fitted
    # parameters, each row's mean must equal the fitted PUBLIC μ and its SD the
    # fitted PUBLIC σ, with skew in the direction of the fitted slant ν. Drawing
    # in the internal (ξ, ω) parameterisation shifts every row mean by
    # ω·δ·√(2/π) ≈ 1.4 here and shrinks the SD by √(1 − 2δ²/π) ≈ 0.63, so this
    # is the assertion that keeps simulator and likelihood from drifting apart.
    S = 1000
    Y = simulate(fit; nsim = S, rng = MersenneTwister(12))
    @test size(Y) == (n, S)
    μ̂, σ̂, ν̂ = fit.means[:mu], fit.scales[:sigma], fit.scales[:nu]
    rowmean = vec(mean(Y; dims = 2))
    rowsd   = vec(std(Y; dims = 2))
    rowskew = vec(mean(((Y .- rowmean) ./ rowsd) .^ 3; dims = 2))
    @test maximum(abs.(rowmean .- μ̂)) < 0.25      # mean of the draw == public μ
    @test maximum(abs.(rowsd   .- σ̂)) < 0.25      # SD   of the draw == public σ
    @test all(sign.(rowskew) .== sign(ν̂[1]))      # skewed the way ν says

    # (3) End to end: simulate-then-refit now completes every replicate.
    res = bootstrap_result(fit; data = data, B = 40, rng = MersenneTwister(20260824),
                           failures = :skip, check_converged = true)
    @test res.failed == 0
    @test res.used == 40
    row = only(filter(r -> r.param === :mu && r.coef == "x", res.summary))
    @test isfinite(row.lower) && isfinite(row.upper)
    @test row.lower < row.upper
    @test row.lower <= coef(fit, :mu)[2] <= row.upper
end
