# A second structured marker (`phylo`, `relmat`, `animal`, `spatial`) on a non-Gaussian route used to be
# neither stored by `_split_ranef` (which returns only the first) nor refused: the fit used the first term
# alone and the second was dropped silently (Poisson, NB2, Gamma, Beta, ZeroOneBeta, Binomial, BetaBinomial,
# CumulativeLogit). `_require_one_structured_marker` (src/input_validation.jl, run by `_drm_entry_guard`)
# now refuses it, naming both terms. Gaussian, whose mean fits two structured terms, and LogNormal, which
# hands its markers to that router on log(y), must keep fitting both.
using DRModels, Test, Random, LinearAlgebra
import Distributions as Dst

_sm_corr(M) = (d = sqrt.(diag(M)); M ./ (d * d'))

# Message of the error a fit raises, or `nothing` when it returns a fit.
function _sm_refusal(thunk)
    try
        thunk()
        return nothing
    catch e
        return e isa ArgumentError ? e.msg : rethrow()
    end
end

@testset "second structured marker is refused, not dropped" begin
    Random.seed!(20261110)
    G = 12; m = 5; n = G * m
    phy = random_balanced_tree(G; branch_length = 0.4)
    Kmat = _sm_corr(sigma_phy_dense(phy; σ²_phy = 1.0))
    g = repeat(1:G, inner = m); h = repeat(1:G, outer = m)
    x = randn(n)
    pts = rand(G, 2) .* 5
    coords = Dict(i => pts[i, :] for i in 1:G)
    y_cnt = Float64.(rand.(Dst.Poisson.(exp.(0.5 .+ 0.3 .* x))))
    y_gam = exp.(0.5 .+ 0.3 .* x .+ 0.3 .* randn(n))
    y_b = clamp.(rand(n) .* 0.8 .+ 0.1, 0.05, 0.95)
    y_zob = rand(n)
    s = Float64.(rand.(Dst.Binomial.(10, fill(0.5, n)))); fail = 10.0 .- s
    k = rand(1:3, n)
    d = (; y_cnt, y_gam, y_b, y_zob, s, fail, k, x, g, h)

    function check(thunk, a::String, b::String)
        msg = _sm_refusal(thunk)
        @test msg !== nothing
        msg === nothing && return
        @test occursin("one structured term per formula", msg)
        @test occursin(a, msg) && occursin(b, msg)
        @test occursin("Gaussian()", msg)             # says what to try
    end

    @testset "Poisson: spatial + relmat, both orders, relmat + animal" begin
        check(() -> drm(bf(@formula(y_cnt ~ x + spatial(1 | g) + relmat(1 | h))), Poisson();
                        data = d, coords = coords, K = Kmat, se = false), "spatial(", "relmat(")
        check(() -> drm(bf(@formula(y_cnt ~ x + relmat(1 | g) + spatial(1 | h))), Poisson();
                        data = d, coords = coords, K = Kmat, se = false), "relmat(", "spatial(")
        check(() -> drm(bf(@formula(y_cnt ~ x + relmat(1 | g) + animal(1 | h))), Poisson();
                        data = d, K = Kmat, A = Kmat, se = false), "relmat(", "animal(")
        check(() -> drm(bf(@formula(y_cnt ~ x + phylo(1 | g) + spatial(1 | h))), Poisson();
                        data = d, tree = phy, coords = coords, se = false), "phylo(", "spatial(")
    end
    @testset "NB2: coordinate spatial + relmat, both orders; same kind twice" begin
        for sig in (true, false)
            f(mu) = sig ? bf(mu, @formula(sigma ~ 1)) : bf(mu)
            check(() -> drm(f(@formula(y_cnt ~ x + spatial(1 | g) + relmat(1 | h))), NegBinomial2();
                            data = d, coords = coords, K = Kmat, se = false), "spatial(", "relmat(")
            check(() -> drm(f(@formula(y_cnt ~ x + relmat(1 | g) + spatial(1 | h))), NegBinomial2();
                            data = d, coords = coords, K = Kmat, se = false), "relmat(", "spatial(")
        end
        check(() -> drm(bf(@formula(y_cnt ~ x + spatial(1 | g) + spatial(1 | h))), NegBinomial2();
                        data = d, coords = coords, se = false), "spatial(… | g)", "spatial(… | h)")
    end
    @testset "Gamma, Beta, ZeroOneBeta" begin
        check(() -> drm(bf(@formula(y_gam ~ x + phylo(1 | g) + relmat(1 | h)), @formula(sigma ~ 1)), Gamma();
                        data = d, tree = phy, K = Kmat, se = false), "phylo(", "relmat(")
        check(() -> drm(bf(@formula(y_b ~ x + relmat(1 | g) + relmat(1 | h)), @formula(sigma ~ 1)), Beta();
                        data = d, K = Kmat, se = false), "relmat(… | g)", "relmat(… | h)")
        check(() -> drm(bf(@formula(y_zob ~ x + phylo(1 | g) + relmat(1 | h))), ZeroOneBeta();
                        data = d, tree = phy, K = Kmat, se = false), "phylo(", "relmat(")
    end
    @testset "Binomial, BetaBinomial, CumulativeLogit" begin
        check(() -> drm(bf(@formula(cbind(s, fail) ~ x + phylo(1 | g) + relmat(1 | h))), Binomial();
                        data = d, tree = phy, K = Kmat, se = false), "phylo(", "relmat(")
        check(() -> drm(bf(@formula(cbind(s, fail) ~ x + phylo(1 | g) + phylo(1 | h)), @formula(sigma ~ 1)),
                        BetaBinomial(); data = d, tree = phy, se = false), "phylo(… | g)", "phylo(… | h)")
        check(() -> drm(bf(@formula(k ~ x + phylo(1 | g) + phylo(1 | h))), CumulativeLogit();
                        data = d, tree = phy, se = false), "phylo(… | g)", "phylo(… | h)")
    end
    @testset "Student, SkewNormal, Tweedie, truncated: two markers on mu (no keywords, so the marker check speaks)" begin
        y_pos = y_cnt .+ 1
        check(() -> drm(bf(@formula(y_gam ~ x + relmat(1 | g) + relmat(1 | h)), @formula(sigma ~ 1)), Student();
                        data = d), "relmat(… | g)", "relmat(… | h)")
        check(() -> drm(bf(@formula(y_gam ~ x + phylo(1 | g) + relmat(1 | h)), @formula(sigma ~ 1)), SkewNormal();
                        data = d), "phylo(", "relmat(")
        check(() -> drm(bf(@formula(y_gam ~ x + relmat(1 | g) + relmat(1 | h))), Tweedie();
                        data = d), "relmat(… | g)", "relmat(… | h)")
        check(() -> drm(bf(@formula(y_pos ~ x + relmat(1 | g) + relmat(1 | h))), TruncatedPoisson();
                        data = (; d..., y_pos)), "relmat(… | g)", "relmat(… | h)")
    end
    @testset "labelled term names the group, not the label" begin
        msg = _sm_refusal(() -> drm(bf(@formula(y_cnt ~ x + relmat(1 + x | p | g) + relmat(1 | h))), Poisson();
                                    data = d, K = Kmat, se = false))
        @test msg !== nothing
        msg === nothing || (@test occursin("(… | g)", msg) && !occursin("| p)", msg))
    end
    @testset "one structured term still fits" begin
        fit = drm(bf(@formula(y_cnt ~ x + relmat(1 | g))), Poisson(); data = d, K = Kmat, se = false)
        @test fit.converged
        @test length(fit.theta) == 3                  # intercept, slope, log SD of the one structured term
        @test haskey(re_sd(fit), :g)
    end
end

# Ported from the earlier Poisson and Gamma guard (maintainer decision, 13 Nov 2026): the two extra cases.
# (1) All 16 ordered pairs of `spatial`, `animal`, `relmat` and `phylo` (same kind twice included) are
# refused by name on Poisson and on Gamma, whichever term comes first.
# (2) One structured spatial term still fits on both routes, so the guard refuses only the second term.
@testset "Poisson and Gamma: all 16 ordered pairs are refused; one spatial term still fits" begin
    rng = Random.MersenneTwister(20261113)
    G = 12; m = 10
    g = repeat(1:G, inner = m); n = length(g)
    h = [mod1(i * 5 + (i ÷ 7), G) for i in 1:n]
    x = randn(rng, n)
    pos = rand(rng, G, 2) .* 6
    Dm = [sqrt(sum(abs2, pos[k, :] .- pos[l, :])) for k in 1:G, l in 1:G]
    C = _sm_corr(exp.(-Dm ./ 2) + 1e-6 * I)
    tree = random_balanced_tree(G; branch_length = 0.2)
    mu = exp.(0.3 .+ 0.3 .* x .+ 0.4 .* (cholesky(Symmetric(C)).L * randn(rng, G))[g])
    kwfor(k) = k === :spatial ? (; coords = pos) : k === :animal ? (; A = C) :
               k === :relmat ? (; K = C) : (; tree = tree)
    kinds = (:spatial, :animal, :relmat, :phylo)
    for (fam, y) in ((Poisson(), Float64.([rand(rng, Dst.Poisson(mi)) for mi in mu])),
                     (Gamma(), Float64.([rand(rng, Dst.Gamma(4.0, mi / 4.0)) for mi in mu])))
        dat = (; y, x, g, h)
        fname = string(nameof(typeof(fam)))
        npairs = 0
        for k1 in kinds, k2 in kinds
            f = eval(Meta.parse("@formula(y ~ x + $k1(1 | g) + $k2(1 | h))"))
            bff = fam isa Gamma ? bf(f, @formula(sigma ~ 1)) : bf(f)
            msg = _sm_refusal(() -> drm(bff, fam; data = dat, se = false, merge(kwfor(k1), kwfor(k2))...))
            @test msg !== nothing
            if msg !== nothing
                @test occursin("the $fname route fits one structured term per formula", msg)
                @test occursin("`$k1(… | g)`", msg) && occursin("`$k2(… | h)`", msg)
            end
            npairs += 1
        end
        @test npairs == 16
        f1 = @formula(y ~ x + spatial(1 | g))
        fit1 = drm(fam isa Gamma ? bf(f1, @formula(sigma ~ 1)) : bf(f1), fam; data = dat, se = false, coords = pos)
        @test isfinite(loglikelihood(fit1))
        @test length(re_sd(fit1)) == 1
    end
end

@testset "two structured terms still fit where the route supports them" begin
    Random.seed!(20261111)
    G = 15; m = 6; n = G * m
    phy = random_balanced_tree(G; branch_length = 0.4)
    Cphy = _sm_corr(sigma_phy_dense(phy; σ²_phy = 1.0))
    M = randn(G, G); Canim = _sm_corr(M * M' / G + I)
    species = repeat(1:G, inner = m); id = repeat(1:G, inner = m)
    x = randn(n)
    a1 = 0.7 .* (cholesky(Symmetric(Cphy)).L * randn(G))
    a2 = 0.5 .* (cholesky(Symmetric(Canim)).L * randn(G))
    y = 0.3 .+ 0.5 .* x .+ a1[species] .+ a2[id] .+ 0.5 .* randn(n)
    data = (; y, x, species, id)
    fit = drm(bf(@formula(y ~ x + phylo(1 | species) + relmat(1 | id)), @formula(sigma ~ 1)),
              Gaussian(); data = data, tree = phy, K = Canim)
    @test isfinite(loglik(fit))
    @test haskey(re_sd(fit), :species) && haskey(re_sd(fit), :id)      # both structured SDs are in the fit

    # LogNormal passes both markers to the Gaussian router on log(y): np = 2 betas + log sigma + 2 SDs
    dl = (; y = exp.(y), x, species, id)
    fl = drm(bf(@formula(y ~ x + phylo(1 | species) + relmat(1 | id)), @formula(sigma ~ 1)),
             LogNormal(); data = dl, tree = phy, K = Canim)
    @test isfinite(loglik(fl))
    @test length(fl.theta) == 5
end

# One structured term on `sigma` and none on `mu` is a supported NB2 route; the guard must not touch it.
# The expected log-likelihood was taken from the parent commit (guard absent).
@testset "NB2 structured sigma (one marker, on sigma) is not refused" begin
    Random.seed!(20261112)
    G = 12; m = 8; n = G * m
    M = randn(G, G); K = _sm_corr(M * M' / G + I)
    id = repeat(1:G, inner = m); x = randn(n)
    u = 0.5 .* (cholesky(Symmetric(K)).L * randn(G))
    mu = exp.(0.5 .+ 0.3 .* x)
    y = Float64.([rand(Dst.NegativeBinomial(5.0 * exp(u[id[i]]), 5.0 * exp(u[id[i]]) / (5.0 * exp(u[id[i]]) + mu[i]))) for i in 1:n])
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + relmat(1 + x | id))), NegBinomial2();
              data = (; y, x, id), K = K, se = false)
    @test fit.converged
    @test loglikelihood(fit) ≈ -156.90824338672078 atol = 1e-8
end
