# Covariate-conditional repeatability for location-scale-scale fits (#694).
#
# With sd(id) ~ sex AND sigma ~ sex the between-individual SD sigma_b(z) and the
# residual SD sigma_e(z) both move with the covariate, so
#     R(z) = sigma_b(z)^2 / (sigma_b(z)^2 + sigma_e(z)^2) = logistic(2 (alpha'z - gamma'z))
# is a function of z: no single repeatability exists. `repeatability(fit)` /
# `icc(fit)` / `heritability(fit)` refuse such fits (and say why); the two-argument form
# `repeatability(fit, newdata)` returns R(z) with a Wald-on-logit interval.

using DRModels
using Test
using Random
using LinearAlgebra
using ForwardDiff
using Logging

@testset "LSS conditional repeatability (#694)" begin
    # the tutorial example (docs/src/tutorials/location-scale-scale.md)
    rng = Random.MersenneTwister(20260715)
    n_id, n_each = 80, 6
    sex = repeat([0.0, 1.0], inner = n_id ÷ 2)
    b = randn(rng, n_id) .* [0.65, 0.40][Int.(sex) .+ 1]
    id = repeat(1:n_id, inner = n_each)
    sexl = sex[id]
    y = [0.35, 0.70][Int.(sexl) .+ 1] .+ b[id] .+
        randn(rng, n_id * n_each) .* [0.35, 0.60][Int.(sexl) .+ 1]
    dat = (; y, sex = sexl, id)
    fit = Logging.with_logger(Logging.NullLogger()) do
        drm(bf(@formula(y ~ sex + (1 | id)), @formula(sigma ~ sex),
               @formula(sd(id) ~ sex)), Gaussian(); data = dat)
    end

    @testset "single-number extractors refuse and state the estimand" begin
        for f in (repeatability, icc, heritability)
            err = try f(fit); nothing catch e; e end
            @test err !== nothing
            msg = sprint(showerror, err)
            @test occursin("covariate-CONDITIONAL", msg)
            @test occursin("repeatability(fit, (; sex", msg)
        end
    end

    @testset "R(z) equals the closed form logistic(2(alpha'z - gamma'z))" begin
        nd = (; sex = [0.0, 1.0, 0.5])
        r = repeatability(fit, nd)
        α = coef(fit, :sd); γ = coef(fit, :sigma)
        for (i, s) in enumerate(nd.sex)
            σb = exp(α[1] + α[2] * s); σe = exp(γ[1] + γ[2] * s)
            @test r.estimate[i] ≈ σb^2 / (σb^2 + σe^2) rtol = 1e-12
        end
        @test r.method === :wald_logit && r.level == 0.95
        @test all(0 .< r.lower .< r.estimate .< r.upper .< 1)
        # truth: females 0.65 / 0.35 -> 0.775, males 0.40 / 0.60 -> 0.308 (sampling error at n = 80 x 6)
        @test r.lower[1] < 0.775 < r.upper[1]
        @test r.lower[2] < 0.308 < r.upper[2]
        # R is sex-specific: the two numbers differ by far more than their SEs, so any one
        # package-wide scalar would misreport at least one sex
        @test r.estimate[1] - r.estimate[2] > 0.4
        # icc is the same call
        @test icc(fit, nd).estimate == r.estimate
    end

    @testset "delta-method SE through the joint vcov (sd and sigma blocks)" begin
        nd = (; sex = [0.0, 1.0])
        r = repeatability(fit, nd)
        isd, isg = fit.blocks[3].second, fit.blocks[2].second
        @test fit.blocks[3].first === :sd && fit.blocks[2].first === :sigma
        V = fit.vcov
        for (i, s) in enumerate(nd.sex)
            g(θ) = 1 / (1 + exp(-2 * ((θ[isd[1]] + θ[isd[2]] * s) - (θ[isg[1]] + θ[isg[2]] * s))))
            ∇ = ForwardDiff.gradient(g, fit.theta)
            se_prob = sqrt(dot(∇, V * ∇))
            R = r.estimate[i]
            @test se_prob ≈ R * (1 - R) * r.se_logit[i] rtol = 1e-8   # same SE on the two scales
        end
        # a wider level widens the interval
        r99 = repeatability(fit, nd; level = 0.99)
        @test all(r99.lower .< r.lower) && all(r99.upper .> r.upper)
    end

    @testset "agrees with the constant-SD repeatability when sd and sigma do not vary" begin
        rng2 = Random.MersenneTwister(5)
        ng, ne = 60, 5
        id2 = repeat(1:ng, inner = ne)
        z = zeros(ng * ne)
        y2 = randn(rng2, ng)[id2] .* 0.8 .+ randn(rng2, ng * ne) .* 0.5
        d2 = (; y = y2, z, id = id2)
        fc = Logging.with_logger(Logging.NullLogger()) do
            drm(bf(@formula(y ~ 1 + (1 | id)), @formula(sigma ~ 1),
                   @formula(sd(id) ~ 1)), Gaussian(); data = d2)
        end
        rc = repeatability(fc, (; z = [0.0]))
        # closed form from the fitted coefficients (the constant-SD `icc` is a different route)
        σb = exp(coef(fc, :sd)[1]); σe = exp(coef(fc, :sigma)[1])
        @test rc.estimate[1] ≈ σb^2 / (σb^2 + σe^2) rtol = 1e-12
        @test 0.6 < rc.estimate[1] < 0.8               # truth 0.64 / (0.64 + 0.25) = 0.72
    end

    @testset "misuse is reported, not guessed" begin
        plain = Logging.with_logger(Logging.NullLogger()) do
            drm(bf(@formula(y ~ sex + (1 | id)), @formula(sigma ~ 1)), Gaussian(); data = dat)
        end
        @test_throws ErrorException repeatability(plain, (; sex = [0.0]))
        # `heritability(fit, newdata)` is for sd(g, phylogenetic); an iid-sd fit has no such block
        @test_throws ErrorException heritability(fit, (; sex = [0.0]))
        @test_throws ArgumentError repeatability(fit, (; sex = [0.0]); level = 1.2)
    end

    @testset "phylogenetic LSS: h2(z) = sigma_a(z)^2 / (sigma_a(z)^2 + sigma_e(z)^2)" begin
        # a 16-tip balanced tree built like the tutorial's 64-tip one (kept small for speed)
        function _baln(d)
            node(p, k) = k == 0 ? "$(p):$(1/d)" :
                (k == d ? "($(node(p*"a",k-1)),$(node(p*"b",k-1)));" :
                          "($(node(p*"a",k-1)),$(node(p*"b",k-1))):$(1/d)")
            node("t", d)
        end
        phy = DRModels.augmented_phy(_baln(4))
        G = phy.n_leaves
        K0 = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
        dK = sqrt.(diag(K0)); K = K0 ./ (dK * dK')
        rng2 = Random.MersenneTwister(11)
        x = randn(rng2, G)
        sda = exp.(-0.5 .+ 0.4 .* x); sde = exp.(-1.0 .- 0.3 .* x)
        a = Diagonal(sda) * (cholesky(Symmetric(K)).L * randn(rng2, G))
        yq = 1.0 .+ 0.5 .* x .+ a .+ sde .* randn(rng2, G)
        datq = (y = yq, x = x, species = String.(phy.leaf_names))
        fitq = Logging.with_logger(Logging.NullLogger()) do
            drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x),
                   @formula(sd(species, phylogenetic) ~ x)), Gaussian(); data = datq, tree = phy)
        end
        nd = (; x = [-1.0, 0.0, 1.0])
        h = heritability(fitq, nd)
        α = coef(fitq, :sd_phylo); γ = coef(fitq, :sigma)
        for (i, xv) in enumerate(nd.x)
            σa = exp(α[1] + α[2] * xv); σe = exp(γ[1] + γ[2] * xv)
            @test h.estimate[i] ≈ σa^2 / (σa^2 + σe^2) rtol = 1e-12
        end
        @test all(0 .< h.lower .< h.estimate .< h.upper .< 1)
        @test h.estimate[3] > h.estimate[1]            # phylogenetic share rises with the covariate (truth)
        # the iid form is not defined on a phylogenetic-sd fit
        @test_throws ErrorException repeatability(fitq, nd)
        err = try heritability(fitq); nothing catch e; e end
        @test occursin("h²(z)", sprint(showerror, err))
    end
end
