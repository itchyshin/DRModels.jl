# VA scaffold + deterministic anchors (#136 Rung 2).
# Method-selection surface stays; `_fit_va` still errors for unwired families.
# Anchors a/b/c are live on tip kernels (Poisson / NB2 `(1 | g)`) — no longer
# `@test_skip` placeholders. Fuller per-family coverage lives in
# `test_va_poisson_elbo.jl` / `test_variational_{binomial,nb2,gamma}.jl`.

using DRModels
using Test
using Random
import Distributions

const DVA = DRModels

@testset "VA marginal scaffold (#136)" begin
    @test DRModels.Laplace() isa DRModels.MarginalMethod
    @test DRModels.Variational() isa DRModels.MarginalMethod
    @test DRModels._marginal_method(:LA) == DRModels.Laplace()
    @test DRModels._marginal_method(:va) == DRModels.Variational()   # case-insensitive
    @test_throws ArgumentError DRModels._marginal_method(:nope)
    # generic `_fit_va` is still the unwired-family stub — must error, citing #136
    err = try; DRModels._fit_va(); catch e; e; end
    @test err isa ErrorException && occursin("136", err.msg)

    # ── Anchor (a): σ_RE → 0 ⇒ ELBO = independent Poisson log-likelihood ─────
    @testset "anchor (a): RE variance → 0 ⇒ ELBO = independent loglik" begin
        rng = MersenneTwister(136)
        ng, per = 6, 5
        gidx = repeat(1:ng, inner = per)
        n = ng * per
        x = randn(rng, n)
        Xμ = hcat(ones(n), x)
        βtrue = [0.4, -0.3]
        η = Xμ * βtrue
        y = Float64.([rand(rng, Distributions.Poisson(exp(ηi))) for ηi in η])
        fit_va = DVA._fit_poisson_ranef_va(DVA.Poisson(), y, Xμ, gidx, ng,
                                           ["(Intercept)", "x"], :g, 1e-8)
        nll_va = fit_va.nll
        lf = [DVA._logfactorial(round(Int, yi)) for yi in y]
        glm_nll(β) = -sum(y[i] * (Xμ * β)[i] - exp((Xμ * β)[i]) - lf[i] for i in 1:n)
        for β in (βtrue, [0.0, 0.0], [1.0, 0.5])
            @test isapprox(nll_va(vcat(β, -12.0)), glm_nll(β); atol = 1e-6)
        end
    end

    # ── Anchor (b): ELBO ≤ dense adaptive GHQ at low latent dimension ────────
    # Non-adaptive engine GHQ (nodes at 0) can sit *below* the ELBO; the bound
    # is vs the exact marginal, approximated here by mode-centered K=64 GHQ.
    @testset "anchor (b): ELBO ≤ dense adaptive GHQ (low latent dim)" begin
        rng = MersenneTwister(1361)
        ng, per = 4, 5
        gidx = repeat(1:ng, inner = per)
        n = ng * per
        x = randn(rng, n)
        Xμ = hcat(ones(n), x)
        β = [0.3, 0.4]; logσb = log(0.55)
        bg = exp(logσb) .* randn(rng, ng)
        y = Float64.([rand(rng, Distributions.Poisson(exp((Xμ * β)[i] + bg[gidx[i]])))
                      for i in 1:n])
        fit_va = DVA._fit_poisson_ranef_va(DVA.Poisson(), y, Xμ, gidx, ng,
                                           ["(Intercept)", "x"], :g, 1e-8)
        θ = vcat(β, logσb)
        elbo = -fit_va.nll(θ)
        members = [Int[] for _ in 1:ng]
        for i in 1:n
            push!(members[gidx[i]], i)
        end
        yint = round.(Int, y)
        σb = exp(logσb); η0 = Xμ * β
        z, w = DVA._gauss_hermite(64)
        ll = 0.0
        for idx in members
            isempty(idx) && continue
            bm = 0.0
            for _ in 1:60
                g = -bm / σb^2; h = -1 / σb^2
                for i in idx
                    μi = exp(η0[i] + bm)
                    g += yint[i] - μi
                    h += -μi
                end
                bm -= g / h
            end
            h = -1 / σb^2
            for i in idx
                h += -exp(η0[i] + bm)
            end
            sd = sqrt(-1 / h)
            terms = Float64[]
            for k in 1:64
                bk = bm + sqrt(2) * sd * z[k]
                lp = -0.5 * log(2π * σb^2) - bk^2 / (2σb^2)
                for i in idx
                    μi = exp(η0[i] + bk)
                    lp += yint[i] * log(μi) - μi - DVA._logfactorial(yint[i])
                end
                push!(terms, lp + log(w[k]) + z[k]^2 + log(sqrt(2) * sd))
            end
            mx = maximum(terms)
            ll += mx + log(sum(exp.(terms .- mx)))
        end
        @test elbo ≤ ll + 1e-6
    end

    # ── Anchor (c): NB2 size r → ∞ reproduces Poisson-VA on a shared fixture ─
    @testset "anchor (c): NB2 r → ∞ reproduces Poisson-VA" begin
        rng = MersenneTwister(1362)
        ng, per = 8, 6
        gidx = repeat(1:ng, inner = per)
        n = ng * per
        x = randn(rng, n)
        Xμ = hcat(ones(n), x)
        Xσ = ones(n, 1)
        β = [0.4, -0.3]; logσb = log(0.5)
        bg = exp(logσb) .* randn(rng, ng)
        y = Float64.([rand(rng, Distributions.Poisson(exp((Xμ * β)[i] + bg[gidx[i]])))
                      for i in 1:n])
        fit_p = DVA._fit_poisson_ranef_va(DVA.Poisson(), y, Xμ, gidx, ng,
                                          ["(Intercept)", "x"], :g, 1e-8)
        fit_nb = DVA._fit_nb2_ranef_va(DVA.NegBinomial2(), y, Xμ, Xσ, gidx, ng,
                                       ["(Intercept)", "x"], ["(Intercept)"], :g, 1e-8)
        r∞ = 1.0e6
        lψ = -0.5 * log(r∞)                          # size = exp(−2ψ) = r∞
        elbo_p = -fit_p.nll(vcat(β, logσb))
        elbo_nb = -fit_nb.nll(vcat(β, lψ, logσb))
        @test isapprox(elbo_nb, elbo_p; atol = 1e-4)
    end

    # ── Mixed-marginal AIC/LRT: aicc must not short-circuit past the VA guard ─
    # When n − k − 1 ≤ 0, `aicc` used to return Inf before calling `aic`, so a
    # tiny VA fit silently escaped the ELBO-is-not-a-likelihood error.
    @testset "aicc on a tiny VA fit errors (no Inf short-circuit)" begin
        tiny = DRModels._withmarginal(
            DRModels.DrmFit(Poisson(),
                       [:mu => 1:1, :resd => 2:2],
                       [:mu => ["(Intercept)"], :resd => ["g"]],
                       [0.0, -1.0], [1.0 0.0; 0.0 1.0], -10.0, 2, true,
                       Dict(:mu => [1.0, 1.0]), Dict(:mu => [1.0, 2.0]),
                       Dict{Symbol,Vector{Float64}}()),
            :VA)
        @test nobs(tiny) - dof(tiny) - 1 <= 0
        @test_throws ArgumentError aicc(tiny)
        @test_throws ArgumentError aic(tiny)
        @test_throws ArgumentError bic(tiny)
    end
end
