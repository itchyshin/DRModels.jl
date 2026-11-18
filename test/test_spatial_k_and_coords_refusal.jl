using DRModels, Test, Random, LinearAlgebra, Statistics
import Distributions as Dist

# True when the call raised the new refusal (any other outcome, including an error
# about the unused `coords`, is a different matter).
occurs_both(thunk) = try; thunk(); false; catch e
    e isa ArgumentError && occursin("both `K = …` and `coords = …`", e.msg)
end

@testset "spatial() with both K and coords is refused by name" begin
    rng = MersenneTwister(20261113)
    G, m = 20, 5
    coords = rand(rng, G, 2) .* 10
    Dm = [norm(coords[i, :] .- coords[j, :]) for i in 1:G, j in 1:G]
    K = exp.(-Dm ./ (0.4 * median(filter(>(0), vec(Dm))))) + 1e-6I
    site = repeat(1:G, inner = m)
    n = G * m
    x = randn(rng, n)
    eta = 0.2 .+ 0.3 .* x .+ 0.4 .* (cholesky(Symmetric(K)).L * randn(rng, G))[site]
    z = [zeros(G ÷ 2); ones(G - G ÷ 2)][site]
    mu01 = 1 ./ (1 .+ exp.(-eta))
    data = (; x, site, z,
            ycnt = Float64.(rand.(rng, Dist.Poisson.(exp.(eta)))),
            ygau = eta .+ 0.5 .* randn(rng, n),
            ygam = rand.(rng, Dist.Gamma.(4.0, exp.(eta) ./ 4.0)),
            ybeta = rand.(rng, Dist.Beta.(8 .* mu01, 8 .* (1 .- mu01))),
            ybin = Float64.(rand.(rng, Dist.Bernoulli.(mu01))),
            y2 = eta .+ 0.5 .* randn(rng, n))
    both = r"both `K = …` and `coords = …`"

    cases = [
        ("Poisson", Poisson(), bf(@formula(ycnt ~ x + spatial(1 | site)))),
        ("Poisson sd", Poisson(), bf(@formula(ycnt ~ x + spatial(1 | site)), @formula(sd(site, spatial) ~ z))),
        ("NB2", NegBinomial2(), bf(@formula(ycnt ~ x + spatial(1 | site)), @formula(sigma ~ 1))),
        ("Beta", DRModels.Beta(), bf(@formula(ybeta ~ x + spatial(1 | site)), @formula(sigma ~ 1))),
        ("Gamma", DRModels.Gamma(), bf(@formula(ygam ~ x + spatial(1 | site)), @formula(sigma ~ 1))),
        ("Gaussian", Gaussian(), bf(@formula(ygau ~ x + spatial(1 | site)), @formula(sigma ~ 1))),
        ("Binomial", Binomial(), bf(@formula(ybin ~ x + spatial(1 | site)))),
    ]
    for (name, fam, f) in cases
        @testset "$name" begin
            @test_throws both drm(f, fam; data, K, coords)
        end
    end

    @testset "bivariate Gaussian" begin
        fb = bf(mu1 = @formula(ygau ~ x + spatial(1 | site)), mu2 = @formula(y2 ~ x + spatial(1 | site)))
        @test_throws both drm(fb, Gaussian(); data, K, coords)
    end

    @testset "the message names the single argument to give" begin
        f = bf(@formula(ycnt ~ x + spatial(1 | site)))
        err = try; drm(f, Poisson(); data, K, coords); catch e; e; end
        @test err isa ArgumentError
        @test occursin("either `coords = …`", err.msg) && occursin("`K = …`", err.msg)
        fg = bf(@formula(ygau ~ x + spatial(1 | site)), @formula(sigma ~ 1))
        err = try; drm(fg, Gaussian(); data, K, coords); catch e; e; end
        @test occursin("Give `coords = …` only", err.msg)
    end

    @testset "one argument still fits" begin
        f = bf(@formula(ycnt ~ x + spatial(1 | site)))
        @test drm(f, Poisson(); data, K, se = false) isa DRModels.DrmFit
        @test drm(f, Poisson(); data, coords, se = false) isa DRModels.DrmFit
        fbeta = bf(@formula(ybeta ~ x + spatial(1 | site)), @formula(sigma ~ 1))
        @test drm(fbeta, DRModels.Beta(); data, K, se = false) isa DRModels.DrmFit
        # relmat(1 | g) with K and an unrelated coords is not a spatial() term: untouched here.
        frel = bf(@formula(ycnt ~ x + relmat(1 | site)))
        @test !occurs_both(() -> drm(frel, Poisson(); data, K, coords, se = false))
    end
end
