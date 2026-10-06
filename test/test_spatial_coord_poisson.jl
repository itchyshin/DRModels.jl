# test_spatial_coord_poisson.jl — coordinate-based exponential-kernel spatial
# covariance for POISSON counts with the range ρ ESTIMATED JOINTLY (#167/#270).
#
# Today `spatial(1 | site)` for counts needs a PRECOMPUTED covariance via `K = …`;
# only the Gaussian path turned site coordinates into a kernel and estimated the
# range. This exercises the new coordinate-spatial Poisson path
# (`_fit_poisson_spatial_coord`): coords → C(ρ) = exp(-d/ρ), with the random
# intercept b ~ N(0, σ_b² C(ρ)) on log λ and θ = [β_μ; log σ_b; log ρ], so ρ is a
# genuine jointly-estimated hyperparameter.
#
# Three checks:
#   1. Recovery — fixed effects, the spatial SD σ_b, and (weakly) the range ρ are
#      recovered from data simulated with a KNOWN exponential-kernel covariance.
#   2. FD gate — the ForwardDiff outer gradient of the marginal NLL matches a
#      central finite-difference gradient to ≤ 1e-6 at a θ OFF the optimum (so the
#      implicit db̂/dθ correction, including the ρ-channel, is exercised — a
#      frozen-mode gradient would fail this). This is what certifies the gradient
#      that drives the joint estimation, including the ρ-derivative.
#   3. Likelihood-scale cross-check — at a FIXED range, the coordinate-spatial
#      marginal agrees with the verified `K = C(ρ)` sparse general-covariance path,
#      confirming the new marginal is on the same scale as the engine it mirrors.
using DRModels
using Test, Random, LinearAlgebra
import ForwardDiff
import Distributions

# Genuine exponential-kernel spatial covariance among G sites at random positions.
function _spatial_cov(rng, G; range, jitter = 1e-8)
    pos = rand(rng, G, 2) .* 10.0
    D = [sqrt(sum(abs2, pos[k, :] .- pos[l, :])) for k in 1:G, l in 1:G]
    C = exp.(-D ./ range) + jitter * I
    return pos, Symmetric(C)
end

@testset "Poisson spatial(1|site, coords) — joint range estimation + recovery" begin
    rng = MersenneTwister(20260610)
    G = 90
    m = 6
    ρtrue = 2.5
    σb = 0.55
    coords, C = _spatial_cov(rng, G; range = ρtrue)
    site = repeat(1:G, inner = m)
    n = length(site)
    x = randn(rng, n)
    β = [0.20, 0.40]
    u = σb .* (cholesky(C).L * randn(rng, G))
    λ = exp.(β[1] .+ β[2] .* x .+ u[site])
    y = Float64.([rand(rng, Distributions.Poisson(λi)) for λi in λ])
    data = (; y, x, site)

    fit = drm(bf(@formula(y ~ x + spatial(1 | site))), Poisson();
              data = data, coords = coords)

    @test fit.converged
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.15          # slope (RE-independent)
    @test re_sd(fit)[:site] ≈ σb atol = 0.20            # spatial SD recovered
    # Range is only weakly identified from one realization; assert it is finite,
    # positive, and in a sane band around the truth rather than a tight equality.
    ρ̂ = exp(coef(fit, :range)[1])
    @test isfinite(ρ̂) && ρ̂ > 0
    @test 0.3 * ρtrue < ρ̂ < 6.0 * ρtrue                 # jointly-estimated, sane band
    @test isfinite(loglik(fit))
    @test all(fitted(fit) .> 0)
end

@testset "Poisson spatial(1|site, coords) — FD gate: ForwardDiff vs central FD ≤ 1e-6" begin
    rng = MersenneTwister(20260611)
    G = 14
    m = 5
    coords, C = _spatial_cov(rng, G; range = 1.8)
    site = repeat(1:G, inner = m)
    n = length(site)
    x = randn(rng, n)
    σb = 0.45
    u = σb .* (cholesky(C).L * randn(rng, G))
    λ = exp.(0.2 .+ 0.30 .* x .+ u[site])
    y = Float64.([rand(rng, Distributions.Poisson(λi)) for λi in λ])

    Xμ = hcat(ones(n), x)
    gidx, Gn = DRModels._group_index(site)
    @test Gn == G
    Ddist = [sqrt(sum(abs2, coords[k, :] .- coords[l, :])) for k in 1:G, l in 1:G]
    lf = [DRModels._logfactorial(round(Int, yi)) for yi in y]

    # Fresh warm-start buffer; θ deliberately OFF the optimum so the implicit
    # db̂/dθ terms (β, log σ_b, AND log ρ channels) are all nonzero.
    bref = zeros(G)
    mnll(θ) = DRModels._poisson_spatial_marginal(θ, y, Xμ, gidx, Ddist, lf, bref)

    θ = [0.10, 0.42, log(0.60), log(2.2)]               # [β(2); log σ_b; log ρ], off-optimum
    v0 = mnll(θ)
    @test isfinite(v0) && v0 < 1e17                     # genuine marginal, never the sentinel

    g_ad = ForwardDiff.gradient(mnll, θ)

    h = 1e-5
    g_fd = similar(g_ad)
    for k in eachindex(θ)
        tp = copy(θ); tp[k] += h
        tm = copy(θ); tm[k] -= h
        vp = mnll(tp); vm = mnll(tm)
        @assert isfinite(vp) && isfinite(vm) && vp < 1e17 && vm < 1e17
        g_fd[k] = (vp - vm) / (2h)
    end

    max_abs_diff = maximum(abs, g_ad .- g_fd)
    @info "Poisson coord-spatial gradient gate" max_abs_diff g_ad g_fd
    @test max_abs_diff ≤ 1e-6
end

@testset "Poisson spatial coords — fixed-range marginal matches the K = C path" begin
    # At a FIXED range, the coordinate-spatial marginal must agree (up to a constant
    # log y! offset convention, which both include) with the verified sparse
    # general-covariance fit through `K = C(ρ)`. This pins the new marginal to the
    # same likelihood scale as the engine it mirrors.
    rng = MersenneTwister(20260612)
    G = 25
    m = 5
    ρfix = 1.5
    coords, C = _spatial_cov(rng, G; range = ρfix)
    site = repeat(1:G, inner = m)
    n = length(site)
    x = randn(rng, n)
    u = 0.5 .* (cholesky(C).L * randn(rng, G))
    y = Float64.([rand(rng, Distributions.Poisson(exp(0.2 + 0.35x[i] + u[site[i]]))) for i in 1:n])

    Xμ = hcat(ones(n), x)
    gidx, _ = DRModels._group_index(site)
    Ddist = [sqrt(sum(abs2, coords[k, :] .- coords[l, :])) for k in 1:G, l in 1:G]
    lf = [DRModels._logfactorial(round(Int, yi)) for yi in y]

    # Reference fit via the verified K = C(ρfix) sparse path (range NOT estimated).
    Cfix = exp.(-Ddist ./ ρfix) + 1e-8 * I
    fit_K = drm(bf(@formula(y ~ x + spatial(1 | site))), Poisson();
                data = (; y, x, site), K = Matrix(Cfix), se = false)
    β_K = coef(fit_K, :mu)
    logσ_K = coef(fit_K, :resd)[1]

    # Coordinate-spatial marginal at the SAME (β, σ_b) and the fixed range: its NLL
    # must equal the reference fit's −loglik (both are the Poisson Laplace marginal
    # with C unit-diagonal, so identical up to floating point).
    bref = zeros(G)
    θ = vcat(β_K, [logσ_K, log(ρfix)])
    nll_coord = DRModels._poisson_spatial_marginal(θ, y, Xμ, gidx, Ddist, lf, bref)
    @test nll_coord ≈ -loglik(fit_K) rtol = 1e-6
end
