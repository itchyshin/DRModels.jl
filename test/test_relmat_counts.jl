# test_relmat_counts.jl — general user-supplied PD-covariance random effect
# (relatedness / animal model / precomputed spatial) for a COUNT family (#167).
#
# DRModels.jl's phylogenetic sparse-Laplace engine is fully general in the prior
# precision Q: nothing in the inner mode-finder, the Takahashi selected-inversion
# log-det derivatives, or the exact O(p) outer gradient requires Q to come from a
# tree. This test exercises the Poisson route with an ARBITRARY PD covariance C
# supplied by the user via `relmat(1 | id)` + `K = C` (and the `animal(1 | id)` /
# `spatial(1 | id)` aliases), confirming (a) parameter recovery of the fixed
# effects and the variance component, and (b) that the exact analytic gradient
# matches central finite differences at the fitted θ (the FD gate the gradient
# path must pass, mirroring test_poisson_phylo_laplace.jl).
using DRModels
using Test, Random, LinearAlgebra
import Distributions

# Build a genuine, well-conditioned PD correlation among `G` groups from an
# exponential kernel over random latent positions (a relatedness/spatial-style
# matrix that is NOT a tree), rescaled to unit diagonal so the recovered `:resd`
# block is the random-effect SD σ_b directly.
function _random_corr(rng, G; range = 0.8, jitter = 1e-8)
    pos = rand(rng, G, 2) .* 6.0
    D = [sqrt(sum(abs2, pos[k, :] .- pos[l, :])) for k in 1:G, l in 1:G]
    C = exp.(-D ./ range) + jitter * I
    d = sqrt.(diag(C))
    return Symmetric(C ./ (d * d'))
end

@testset "Poisson relmat(1|id) — general-covariance sparse Laplace recovery" begin
    # G groups with a smooth exponential-kernel relatedness; G is large enough and
    # the kernel local enough that the realized RE mean ≈ 0, so even the intercept
    # is recoverable here (in general only the slope and σ_b are RE-independent).
    rng = MersenneTwister(20260610)
    G = 80
    m = 8
    C = _random_corr(rng, G; range = 0.8)
    id = repeat(1:G, inner = m)
    n = length(id)
    x = randn(rng, n)
    β = [0.20, 0.40]
    σphy = 0.50
    u = σphy .* (cholesky(C).L * randn(rng, G))
    λ = exp.(β[1] .+ β[2] .* x .+ u[id])
    y = Float64.([rand(rng, Distributions.Poisson(λi)) for λi in λ])

    fit = drm(bf(@formula(y ~ x + relmat(1 | id))), Poisson();
              data = (; y, x, id), K = Matrix(C), se = false)

    @test fit.converged
    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.20       # intercept (mean(u) ≈ 0 here)
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.15       # slope (independent of the RE)
    @test re_sd(fit)[:id] ≈ σphy atol = 0.15         # variance component recovered
    @test isfinite(loglik(fit))
    @test all(fitted(fit) .> 0)
end

@testset "Poisson relmat(1|id) — exact gradient vs finite differences (FD gate)" begin
    rng = MersenneTwister(20260611)
    G = 12
    m = 6
    C = _random_corr(rng, G; range = 1.2)
    id = repeat(1:G, inner = m)
    n = length(id)
    x = randn(rng, n)
    σphy = 0.40
    u = σphy .* (cholesky(C).L * randn(rng, G))
    λ = exp.(0.2 .+ 0.30 .* x .+ u[id])
    y = Float64.([rand(rng, Distributions.Poisson(λi)) for λi in λ])

    fit = drm(bf(@formula(y ~ x + relmat(1 | id))), Poisson();
              data = (; y, x, id), K = Matrix(C), se = false)
    θ = coef(fit)
    g = zeros(length(θ))
    fit.nllgrad(g, θ)

    h = 1e-4
    fd = similar(g)
    for k in eachindex(θ)
        e = zeros(length(θ))
        e[k] = h
        fd[k] = (fit.nll(θ .+ e) - fit.nll(θ .- e)) / (2h)
    end
    @test g ≈ fd rtol = 2e-3 atol = 2e-3
end

@testset "Poisson animal/spatial aliases + routing errors" begin
    rng = MersenneTwister(20260612)
    G = 30
    m = 6
    C = _random_corr(rng, G; range = 1.5)
    id = repeat(1:G, inner = m)
    n = length(id)
    x = randn(rng, n)
    u = 0.45 .* (cholesky(C).L * randn(rng, G))
    y = Float64.([rand(rng, Distributions.Poisson(exp(0.2 + 0.35x[i] + u[id[i]]))) for i in 1:n])
    Cm = Matrix(C)

    # animal(1|id) takes the relatedness matrix via `A = …`
    fit_a = drm(bf(@formula(y ~ x + animal(1 | id))), Poisson();
                data = (; y, x, id), A = Cm, se = false)
    @test fit_a.converged
    @test haskey(re_sd(fit_a), :id)
    @test coef(fit_a, :mu)[2] ≈ 0.35 atol = 0.2

    # spatial(1|id) accepts a precomputed spatial covariance via `K = …`
    fit_s = drm(bf(@formula(y ~ x + spatial(1 | id))), Poisson();
                data = (; y, x, id), K = Cm, se = false)
    @test fit_s.converged
    @test haskey(re_sd(fit_s), :id)

    # spatial(1|id) with site coords → coordinate-based exponential-kernel covariance
    # with the range estimated jointly (#270). Here just confirm coords-spatial is now
    # a supported fit that returns the right blocks; convergence + recovery on a
    # properly-simulated spatial dataset is asserted in test_spatial_coord_poisson.jl.
    # (These coords are independent of the data-generating covariance, so there is no
    # spatial signal and the range is unidentified — σ_b → 0 — which is correct.)
    coords = rand(G, 2) .* 5.0
    fit_c = drm(bf(@formula(y ~ x + spatial(1 | id))), Poisson();
                data = (; y, x, id), coords = coords, se = false)
    @test :range in first.(fit_c.blocks)
    @test haskey(re_sd(fit_c), :id)
    @test isfinite(loglik(fit_c))
    @test isfinite(exp(coef(fit_c, :range)[1]))

    # relmat/animal without their matrix → clear error; spatial with neither K nor
    # coords → clear error.
    @test_throws ErrorException drm(bf(@formula(y ~ x + relmat(1 | id))), Poisson();
                                    data = (; y, x, id), se = false)
    @test_throws ErrorException drm(bf(@formula(y ~ x + animal(1 | id))), Poisson();
                                    data = (; y, x, id), se = false)
    @test_throws ErrorException drm(bf(@formula(y ~ x + spatial(1 | id))), Poisson();
                                    data = (; y, x, id), se = false)
end
