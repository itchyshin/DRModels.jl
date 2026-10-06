# VA/ELBO proof kernel (#136): mean-field Gaussian variational marginal for the
# Binomial/Bernoulli random-intercept (logistic GLMM) model. Unlike Poisson the
# logit link has no closed-form E_q[log p] (logistic-normal), so the per-group
# E_q[log-lik] is evaluated by Gauss–Hermite quadrature over the Gaussian q. We
# verify the kernel two ways, exactly as for the Poisson VA:
#
#   (1) Anchor (design note §5, anchor a): σ_RE → 0  ⇒  ELBO = fixed-effect
#       logistic log-likelihood (exact structural identity). With no latent
#       spread q collapses to a point mass at 0, KL → 0, E_q[log p] → log p(y|Xβ).
#
#   (2) Recovery vs the verified near-exact baseline: on a simulated Binomial
#       random-intercept dataset, the VA estimates of (β, σ_RE) sit close to the
#       32-node Gauss–Hermite Laplace MLE (`_fit_binomial_ranef`). VA is a lower
#       bound, so its ELBO ≤ the GHQ marginal log-likelihood — also checked.
#
# These need no external truth and no drmTMB call. The GHQ fit IS DRModels.jl's
# verified Binomial RE marginal; the VA fit is the new code under test.

using DRModels
using Test
using Random
using Statistics
import Distributions

const DB = DRModels   # internal kernels live under DRModels.*

@testset "Binomial random-intercept VA (ELBO) marginal (#136)" begin

    # ── Anchor (a): σ_RE → 0 ⇒ ELBO == fixed-effect logistic loglik ────────────
    # Build a tiny grouped Binomial dataset, evaluate the VA objective at a fixed
    # β with logσ driven low, and compare to the exact fixed-effect logistic nll.
    @testset "anchor: σ_RE → 0 ⇒ ELBO = fixed-effect loglik" begin
        rng = MersenneTwister(7)
        ng, per = 6, 5
        gidx = repeat(1:ng, inner = per)
        n = ng * per
        x = randn(rng, n)
        Xμ = hcat(ones(n), x)
        ntr = fill(10.0, n)                       # 10 trials each
        βtrue = [0.3, -0.6]
        η = Xμ * βtrue
        s = Float64.([rand(rng, Distributions.Binomial(10, DB._logistic(ηi))) for ηi in η])

        # The VA fit assembles the objective closure; pull it back out and probe it.
        fit_va = DB._fit_binomial_ranef_va(DB.Binomial(), s, ntr, Xμ, gidx, ng,
                                           ["(Intercept)", "x"], :g, 1e-8)
        nll_va = fit_va.nll

        # Independent fixed-effect logistic nll at the same β (no RE term at all).
        sint = round.(Int, s); nint = round.(Int, ntr)
        function glm_nll(β)
            ηβ = Xμ * β
            v = 0.0
            for i in 1:n
                μ = DB._logistic(ηβ[i])
                v -= Distributions.logpdf(Distributions.Binomial(nint[i], μ), sint[i])
            end
            return v
        end

        # At logσ = −12 the latent has essentially no spread: the VA objective must
        # equal the fixed-effect nll (q collapses to a point mass at 0, KL → 0).
        for β in (βtrue, [0.0, 0.0], [0.7, 0.4])
            θ = vcat(β, -12.0)                    # [β; logσ]
            @test isapprox(nll_va(θ), glm_nll(β); atol = 1e-5)
        end
    end

    # ── Recovery: VA estimates ≈ near-exact GHQ-Laplace MLE ───────────────────
    @testset "recovery vs 32-node GHQ Laplace (β, σ_RE)" begin
        rng = MersenneTwister(20240602)
        ng, per = 60, 12                 # 60 groups × 12 trials-bearing obs; identified
        gidx = repeat(1:ng, inner = per)
        n = ng * per
        x = randn(rng, n)
        Xμ = hcat(ones(n), x)
        ntr = fill(8.0, n)               # 8 binomial trials per observation
        β0, β1 = 0.4, -0.7
        σ_true = 0.8
        b = σ_true .* randn(rng, ng)     # group random intercepts on the logit scale
        η = β0 .+ β1 .* x .+ b[gidx]
        s = Float64.([rand(rng, Distributions.Binomial(8, DB._logistic(ηi))) for ηi in η])
        nm = ["(Intercept)", "x"]

        fit_la = DB._fit_binomial_ranef(DB.Binomial(), s, ntr, Xμ, gidx, ng, nm, :g, 1e-8)  # GHQ MLE
        fit_va = DB._fit_binomial_ranef_va(DB.Binomial(), s, ntr, Xμ, gidx, ng, nm, :g, 1e-8) # new VA

        @test DB.is_converged(fit_va)

        θla = coef(fit_la); θva = coef(fit_va)
        β_la = θla[1:2];      β_va = θva[1:2]
        σ_la = exp(θla[3]);   σ_va = exp(θva[3])

        # Fixed effects: VA should match GHQ closely (mean is the well-behaved axis).
        @test isapprox(β_va[1], β_la[1]; atol = 0.05)
        @test isapprox(β_va[2], β_la[2]; atol = 0.05)
        # RE sd: mean-field VA is known to under-shrink slightly vs the exact
        # marginal but must land in the same neighbourhood (and recover σ_true).
        @test isapprox(σ_va, σ_la; atol = 0.12)
        @test isapprox(σ_va, σ_true; atol = 0.25)

        # ── Lower-bound property of the ELBO ──────────────────────────────────
        # The ELBO is a lower bound on the EXACT marginal log-likelihood, not on
        # the engine's 32-node *non-adaptive* GHQ value (which centers nodes at
        # b=0, so for groups whose posterior mode is far from 0 it can sit BELOW
        # the VA ELBO — the VA quadrature is effectively mode-centered). We
        # therefore check the bound against a near-exact reference: high-order
        # ADAPTIVE Gauss–Hermite (mode-centered, 200 nodes) at the VA estimate.
        members = [Int[] for _ in 1:ng]
        for i in 1:n
            push!(members[gidx[i]], i)
        end
        sint = round.(Int, s); nint = round.(Int, ntr)
        function marg_ll_adaptive(θ, K)              # near-exact marginal log-lik
            βμ = θ[1:2]; σb = exp(θ[3]); η0 = Xμ * βμ
            z, w = DB._gauss_hermite(K); ll = 0.0
            for idx in members
                isempty(idx) && continue
                bm = 0.0                              # per-group posterior mode of b
                for _ in 1:50
                    g = -bm / σb^2; h = -1 / σb^2
                    for i in idx
                        μ = DB._logistic(η0[i] + bm)
                        g += sint[i] - nint[i] * μ
                        h += -nint[i] * μ * (1 - μ)
                    end
                    bm -= g / h
                end
                h = -1 / σb^2                          # curvature at the mode
                for i in idx
                    μ = DB._logistic(η0[i] + bm); h += -nint[i] * μ * (1 - μ)
                end
                sd = sqrt(-1 / h)                      # adaptive node width
                terms = Float64[]
                for k in 1:K                           # mode-centered adaptive GH
                    bk = bm + sqrt(2) * sd * z[k]
                    lp = -0.5 * log(2π * σb^2) - bk^2 / (2σb^2)
                    for i in idx
                        μ = DB._logistic(η0[i] + bk)
                        lp += Distributions.logpdf(Distributions.Binomial(nint[i], μ), sint[i])
                    end
                    push!(terms, lp + log(w[k]) + z[k]^2 + log(sqrt(2) * sd))
                end
                mx = maximum(terms)
                ll += mx + log(sum(exp.(terms .- mx)))
            end
            return ll
        end
        exact_va = marg_ll_adaptive(θva, 200)
        @test loglik(fit_va) ≤ exact_va + 1e-6        # ELBO ≤ exact marginal
    end
end
