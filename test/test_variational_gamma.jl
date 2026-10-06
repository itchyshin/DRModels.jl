# VA/ELBO proof kernel (#136): mean-field Gaussian variational marginal for the
# Gamma and Beta random-intercept models. Both have a non-identity link (log for
# Gamma, logit for Beta) so the latent integral has no closed-form E_q[log p]; the
# per-group E_q[log-density] is evaluated by Gauss–Hermite quadrature over the
# Gaussian q, with the dispersion (Gamma shape α / Beta precision φ) as a free
# outer parameter held fixed in the inner solve. We verify each kernel two ways,
# exactly as for the Poisson / Binomial / NB2 VA:
#
#   (1) Anchor (design note §5, anchor a): σ_RE → 0  ⇒  ELBO = fixed-effect
#       log-likelihood (exact structural identity). With no latent spread q
#       collapses to a point mass at b=0, KL → 0, E_q[log p] → log p(y|Xβ, disp).
#
#   (2) Recovery vs the verified near-exact baseline: on a simulated random-
#       intercept dataset, the VA estimates of (β, σ_RE, disp) sit close to the
#       32-node Gauss–Hermite Laplace MLE (`_fit_gamma_ranef` / `_fit_beta_ranef`).
#       VA is a lower bound, so its ELBO ≤ the exact marginal log-likelihood —
#       checked against a high-order ADAPTIVE (mode-centered) GH reference at the
#       VA fit.
#
# These need no external truth and no drmTMB call. The GHQ fit IS DRModels.jl's verified
# RE marginal for each family; the VA fit is the new code under test.

using DRModels
using Test
using Random
using Statistics
import Distributions
import ForwardDiff               # adaptive-GH reference for the Beta lower-bound check

const DGB = DRModels   # internal kernels live under DRModels.*

@testset "Gamma random-intercept VA (ELBO) marginal (#136)" begin

    # ── Anchor (a): σ_RE → 0 ⇒ ELBO == fixed-effect Gamma loglik ────────────────
    # Build a tiny grouped Gamma dataset, evaluate the VA objective at fixed
    # (β, logσ) with logσ_b driven low, and compare to the exact fixed-effect nll.
    @testset "anchor: σ_RE → 0 ⇒ ELBO = fixed-effect loglik" begin
        rng = MersenneTwister(13)
        ng, per = 20, 12                          # identified enough that the fit's
        gidx = repeat(1:ng, inner = per)          # final Hessian is non-singular
        n = ng * per
        x = randn(rng, n)
        Xμ = hcat(ones(n), x)
        Xσ = ones(n, 1)                           # sigma ~ 1 (intercept-only dispersion)
        βtrue = [0.5, -0.4]; logσ = -0.3          # α = exp(-2·logσ) ≈ 1.82
        η = Xμ * βtrue
        μ = exp.(η); α = exp(-2 * logσ)
        y = Float64.([rand(rng, Distributions.Gamma(α, μ[i] / α)) for i in 1:n])

        # The VA fit assembles the objective closure; pull it back out and probe it.
        fit_va = DGB._fit_gamma_ranef_va(DGB.Gamma(), y, Xμ, Xσ, gidx, ng,
                                         ["(Intercept)", "x"], ["(Intercept)"], :g, 1e-8)
        nll_va = fit_va.nll

        # Independent fixed-effect Gamma nll at the same (β, logσ): no RE term.
        function gamma_nll(βμ, lσ)
            ηβ = Xμ * βμ; a = exp(-2 * lσ)
            v = 0.0
            for i in 1:n
                μi = exp(ηβ[i])
                v -= Distributions.logpdf(Distributions.Gamma(a, μi / a), y[i])
            end
            return v
        end

        # At logσ_b = −12 the latent has essentially no spread: the VA objective
        # must equal the fixed-effect nll (q collapses to a point mass, KL → 0).
        for βμ in (βtrue, [0.0, 0.0], [0.8, 0.3]), lσ in (logσ, log(0.5), log(1.2))
            θ = vcat(βμ, lσ, -12.0)               # [β_μ; log σ; log σ_b]
            @test isapprox(nll_va(θ), gamma_nll(βμ, lσ); atol = 1e-5)
        end
    end

    # ── Recovery: VA estimates ≈ near-exact GHQ-Laplace MLE ───────────────────
    @testset "recovery vs 32-node GHQ Laplace (β, σ_RE, α)" begin
        rng = MersenneTwister(20260629)
        ng, per = 60, 15                 # 60 groups × 15 obs; well-identified
        gidx = repeat(1:ng, inner = per)
        n = ng * per
        x = randn(rng, n)
        Xμ = hcat(ones(n), x)
        Xσ = ones(n, 1)
        β0, β1 = 0.4, 0.5
        α_true = 4.0                     # Gamma shape (σ = 1/√α = 0.5)
        σ_true = 0.5
        b = σ_true .* randn(rng, ng)     # group random intercepts on the log scale
        μ = exp.(β0 .+ β1 .* x .+ b[gidx])
        y = Float64.([rand(rng, Distributions.Gamma(α_true, μ[i] / α_true)) for i in 1:n])
        nmμ = ["(Intercept)", "x"]; nmσ = ["(Intercept)"]

        fit_la = DGB._fit_gamma_ranef(DGB.Gamma(), y, Xμ, Xσ, gidx, ng, nmμ, nmσ, :g, 1e-8) # GHQ MLE
        fit_va = DGB._fit_gamma_ranef_va(DGB.Gamma(), y, Xμ, Xσ, gidx, ng, nmμ, nmσ, :g, 1e-8) # new VA

        @test DGB.is_converged(fit_va)

        # θ = [β0, β1, logσ (α = exp(-2logσ)), logσ_b] in both fits (same layout).
        θla = coef(fit_la); θva = coef(fit_va)
        β_la = θla[1:2];        β_va = θva[1:2]
        α_la = exp(-2 * θla[3]); α_va = exp(-2 * θva[3])
        σ_la = exp(θla[4]);     σ_va = exp(θva[4])

        # Fixed effects: VA should match GHQ closely (mean is the well-behaved axis).
        @test isapprox(β_va[1], β_la[1]; atol = 0.06)
        @test isapprox(β_va[2], β_la[2]; atol = 0.05)
        # Shape α: VA should land near the GHQ MLE and recover α_true.
        @test isapprox(α_va, α_la; atol = 0.6)
        @test isapprox(α_va, α_true; atol = 1.2)
        # RE sd: mean-field VA is known to under-shrink slightly vs the exact
        # marginal but must land in the same neighbourhood (and recover σ_true).
        @test isapprox(σ_va, σ_la; atol = 0.12)
        @test isapprox(σ_va, σ_true; atol = 0.20)

        # ── Lower-bound property of the ELBO ──────────────────────────────────
        # The ELBO is a lower bound on the EXACT marginal log-likelihood, not on
        # the engine's 32-node *non-adaptive* GHQ value (nodes centered at b=0).
        # We check the bound against a near-exact reference: high-order ADAPTIVE
        # (mode-centered) Gauss–Hermite at the VA estimate.
        members = [Int[] for _ in 1:ng]
        for i in 1:n
            push!(members[gidx[i]], i)
        end
        function marg_ll_adaptive(θ, K)              # near-exact marginal log-lik
            βμ = θ[1:2]; a = exp(-2 * θ[3]); σb = exp(θ[4]); η0 = Xμ * βμ
            z, w = DGB._gauss_hermite(K); ll = 0.0
            for idx in members
                isempty(idx) && continue
                bm = 0.0                              # per-group posterior mode of b
                for _ in 1:60
                    g = -bm / σb^2; h = -1 / σb^2
                    for i in idx
                        μi = exp(η0[i] + bm)
                        # d/db logGamma(α, μ/α) = α·(y/μ − 1); d²/db² = −α·y/μ
                        g += a * (y[i] / μi - 1)
                        h += -a * y[i] / μi
                    end
                    bm -= g / h
                end
                h = -1 / σb^2                          # curvature at the mode
                for i in idx
                    μi = exp(η0[i] + bm); h += -a * y[i] / μi
                end
                sd = sqrt(-1 / h)                      # adaptive node width
                terms = Float64[]
                for k in 1:K                           # mode-centered adaptive GH
                    bk = bm + sqrt(2) * sd * z[k]
                    lp = -0.5 * log(2π * σb^2) - bk^2 / (2σb^2)
                    for i in idx
                        μi = exp(η0[i] + bk)
                        lp += Distributions.logpdf(Distributions.Gamma(a, μi / a), y[i])
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

@testset "Beta random-intercept VA (ELBO) marginal (#136)" begin

    # ── Anchor (a): σ_RE → 0 ⇒ ELBO == fixed-effect Beta loglik ─────────────────
    @testset "anchor: σ_RE → 0 ⇒ ELBO = fixed-effect loglik" begin
        rng = MersenneTwister(17)
        ng, per = 20, 12
        gidx = repeat(1:ng, inner = per)
        n = ng * per
        x = randn(rng, n)
        Xμ = hcat(ones(n), x)
        Xσ = ones(n, 1)                           # sigma ~ 1 (intercept-only precision)
        βtrue = [0.3, -0.5]; logσ = -1.0          # φ = exp(-2·logσ) ≈ 7.39
        η = Xμ * βtrue
        μ = DGB._logistic.(η); φ = exp(-2 * logσ)
        y = Float64.([rand(rng, Distributions.Beta(μ[i] * φ, (1 - μ[i]) * φ)) for i in 1:n])

        fit_va = DGB._fit_beta_ranef_va(DGB.Beta(), y, Xμ, Xσ, gidx, ng,
                                        ["(Intercept)", "x"], ["(Intercept)"], :g, 1e-8)
        nll_va = fit_va.nll

        # Independent fixed-effect Beta nll at the same (β, logσ): no RE term.
        function beta_nll(βμ, lσ)
            ηβ = Xμ * βμ; f = exp(-2 * lσ)
            v = 0.0
            for i in 1:n
                μi = DGB._logistic(ηβ[i])
                v -= Distributions.logpdf(Distributions.Beta(μi * f, (1 - μi) * f), y[i])
            end
            return v
        end

        for βμ in (βtrue, [0.0, 0.0], [0.6, 0.2]), lσ in (logσ, log(0.3), log(0.6))
            θ = vcat(βμ, lσ, -12.0)               # [β_μ; log σ; log σ_b]
            @test isapprox(nll_va(θ), beta_nll(βμ, lσ); atol = 1e-5)
        end
    end

    # ── Recovery: VA estimates ≈ near-exact GHQ-Laplace MLE ───────────────────
    @testset "recovery vs 32-node GHQ Laplace (β, σ_RE, φ)" begin
        rng = MersenneTwister(20260630)
        ng, per = 60, 15
        gidx = repeat(1:ng, inner = per)
        n = ng * per
        x = randn(rng, n)
        Xμ = hcat(ones(n), x)
        Xσ = ones(n, 1)
        β0, β1 = 0.3, -0.6
        φ_true = 12.0                    # Beta precision (σ = 1/√φ ≈ 0.289)
        σ_true = 0.5
        b = σ_true .* randn(rng, ng)     # group random intercepts on the logit scale
        μ = DGB._logistic.(β0 .+ β1 .* x .+ b[gidx])
        y = Float64.([rand(rng, Distributions.Beta(μ[i] * φ_true, (1 - μ[i]) * φ_true)) for i in 1:n])
        nmμ = ["(Intercept)", "x"]; nmσ = ["(Intercept)"]

        fit_la = DGB._fit_beta_ranef(DGB.Beta(), y, Xμ, Xσ, gidx, ng, nmμ, nmσ, :g, 1e-8) # GHQ MLE
        fit_va = DGB._fit_beta_ranef_va(DGB.Beta(), y, Xμ, Xσ, gidx, ng, nmμ, nmσ, :g, 1e-8) # new VA

        @test DGB.is_converged(fit_va)

        θla = coef(fit_la); θva = coef(fit_va)
        β_la = θla[1:2];        β_va = θva[1:2]
        φ_la = exp(-2 * θla[3]); φ_va = exp(-2 * θva[3])
        σ_la = exp(θla[4]);     σ_va = exp(θva[4])

        # Fixed effects: VA should match GHQ closely (mean is the well-behaved axis).
        @test isapprox(β_va[1], β_la[1]; atol = 0.06)
        @test isapprox(β_va[2], β_la[2]; atol = 0.05)
        # Precision φ: VA should land near the GHQ MLE and recover φ_true.
        @test isapprox(φ_va, φ_la; atol = 2.0)
        @test isapprox(φ_va, φ_true; atol = 4.0)
        # RE sd: mean-field VA is known to under-shrink slightly vs the exact
        # marginal but must land in the same neighbourhood (and recover σ_true).
        @test isapprox(σ_va, σ_la; atol = 0.12)
        @test isapprox(σ_va, σ_true; atol = 0.20)

        # ── Lower-bound property of the ELBO ──────────────────────────────────
        members = [Int[] for _ in 1:ng]
        for i in 1:n
            push!(members[gidx[i]], i)
        end
        function marg_ll_adaptive(θ, K)              # near-exact marginal log-lik
            βμ = θ[1:2]; f = exp(-2 * θ[3]); σb = exp(θ[4]); η0 = Xμ * βμ
            z, w = DGB._gauss_hermite(K); ll = 0.0
            for idx in members
                isempty(idx) && continue
                bm = 0.0                              # per-group posterior mode of b
                # Newton on b: gradient/curvature of Σ logBeta + log prior. The Beta
                # log-density derivative wrt η (logit) has no elementary closed form
                # via digamma in b alone cleanly; use ForwardDiff for robustness.
                gfun = bb -> begin
                    s = -bb / σb^2
                    for i in idx
                        s += ForwardDiff.derivative(t -> begin
                            μi = DGB._logistic(η0[i] + t)
                            Distributions.logpdf(Distributions.Beta(μi * f, (1 - μi) * f), y[i])
                        end, bb)
                    end
                    s
                end
                for _ in 1:60
                    g = gfun(bm)
                    h = ForwardDiff.derivative(gfun, bm)
                    bm -= g / h
                end
                h = ForwardDiff.derivative(gfun, bm)   # curvature at the mode
                sd = sqrt(-1 / h)                      # adaptive node width
                terms = Float64[]
                for k in 1:K                           # mode-centered adaptive GH
                    bk = bm + sqrt(2) * sd * z[k]
                    lp = -0.5 * log(2π * σb^2) - bk^2 / (2σb^2)
                    for i in idx
                        μi = DGB._logistic(η0[i] + bk)
                        lp += Distributions.logpdf(Distributions.Beta(μi * f, (1 - μi) * f), y[i])
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
