# VA/ELBO proof kernel (#136): mean-field Gaussian variational marginal for the
# NegBinomial2 (NB2) random-intercept model. Like the Binomial case the log link
# has no closed-form E_q[log p], so the per-group E_q[NB2 log-pmf] is evaluated by
# Gauss–Hermite quadrature over the Gaussian q; the NB2 size/dispersion θ is a free
# outer parameter. We verify the kernel two ways, exactly as for the Poisson and
# Binomial VA:
#
#   (1) Anchor (design note §5, anchor a): σ_RE → 0  ⇒  ELBO = fixed-effect NB2
#       log-likelihood (exact structural identity). With no latent spread q
#       collapses to a point mass at b=0, KL → 0, E_q[log p] → log p(y|Xβ, θ).
#
#   (2) Recovery vs the verified near-exact baseline: on a simulated NB2 random-
#       intercept dataset, the VA estimates of (β, σ_RE, θ) sit close to the
#       32-node Gauss–Hermite Laplace MLE (`_fit_negbin2_ranef`). VA is a lower
#       bound, so its ELBO ≤ the exact marginal log-likelihood — also checked
#       against a high-order ADAPTIVE (mode-centered) GH reference at the VA fit.
#
# These need no external truth and no drmTMB call. The GHQ fit IS DRModels.jl's
# verified NB2 RE marginal; the VA fit is the new code under test.

using DRModels
using Test
using Random
using Statistics
import Distributions

const DBNB = DRModels   # internal kernels live under DRModels.*

@testset "NB2 random-intercept VA (ELBO) marginal (#136)" begin

    # ── Anchor (a): σ_RE → 0 ⇒ ELBO == fixed-effect NB2 loglik ──────────────────
    # Build a tiny grouped NB2 dataset, evaluate the VA objective at fixed (β, logθ)
    # with logσ driven low, and compare to the exact fixed-effect NB2 nll.
    @testset "anchor: σ_RE → 0 ⇒ ELBO = fixed-effect loglik" begin
        rng = MersenneTwister(11)
        ng, per = 20, 12                          # identified enough that the fit's
        gidx = repeat(1:ng, inner = per)          # final Hessian is non-singular
        n = ng * per
        x = randn(rng, n)
        Xμ = hcat(ones(n), x)
        Xσ = ones(n, 1)                           # sigma ~ 1 (intercept-only dispersion)
        βtrue = [0.5, -0.4]; logθ = log(3.0)
        η = Xμ * βtrue
        μ = exp.(η); θsz = exp(logθ)
        y = Float64.([rand(rng, Distributions.NegativeBinomial(θsz, θsz / (θsz + μ[i]))) for i in 1:n])

        # The VA fit assembles the objective closure; pull it back out and probe it.
        fit_va = DBNB._fit_nb2_ranef_va(DBNB.NegBinomial2(), y, Xμ, Xσ, gidx, ng,
                                        ["(Intercept)", "x"], ["(Intercept)"], :g, 1e-8)
        nll_va = fit_va.nll

        # Independent fixed-effect NB2 nll at the same (β, logθ): no RE term at all.
        yint = round.(Int, y)
        function nb2_nll(βμ, lθ)
            ηβ = Xμ * βμ; r = exp(-2 * lθ)   # ψ = log σ; NB2 size = 1/σ² = exp(−2ψ) (matches the VA fitter)
            v = 0.0
            for i in 1:n
                μi = exp(ηβ[i]); p = r / (r + μi)
                v -= Distributions.logpdf(Distributions.NegativeBinomial(r, p), yint[i])
            end
            return v
        end

        # At logσ = −12 the latent has essentially no spread: the VA objective must
        # equal the fixed-effect nll (q collapses to a point mass at 0, KL → 0).
        for βμ in (βtrue, [0.0, 0.0], [0.8, 0.3]), lθ in (logθ, log(1.5), log(6.0))
            θ = vcat(βμ, lθ, -12.0)               # [β_μ; log θ_size; log σ_b]
            @test isapprox(nll_va(θ), nb2_nll(βμ, lθ); atol = 1e-5)
        end
    end

    # ── Recovery: VA estimates ≈ near-exact GHQ-Laplace MLE ───────────────────
    @testset "recovery vs 32-node GHQ Laplace (β, σ_RE, θ)" begin
        rng = MersenneTwister(20260628)
        ng, per = 60, 15                 # 60 groups × 15 obs; well-identified
        gidx = repeat(1:ng, inner = per)
        n = ng * per
        x = randn(rng, n)
        Xμ = hcat(ones(n), x)
        Xσ = ones(n, 1)
        β0, β1 = 0.4, 0.5
        θ_true = 3.0
        σ_true = 0.5
        b = σ_true .* randn(rng, ng)     # group random intercepts on the log scale
        μ = exp.(β0 .+ β1 .* x .+ b[gidx])
        y = Float64.([rand(rng, Distributions.NegativeBinomial(θ_true, θ_true / (θ_true + μ[i]))) for i in 1:n])
        nmμ = ["(Intercept)", "x"]; nmσ = ["(Intercept)"]

        fit_la = DBNB._fit_negbin2_ranef(DBNB.NegBinomial2(), y, Xμ, Xσ, gidx, ng, nmμ, nmσ, :g, 1e-8) # GHQ MLE
        fit_va = DBNB._fit_nb2_ranef_va(DBNB.NegBinomial2(), y, Xμ, Xσ, gidx, ng, nmμ, nmσ, :g, 1e-8)   # new VA

        @test DBNB.is_converged(fit_va)

        # θ = [β0, β1, logθ_size, logσ_b] in both fits (same block layout).
        θla = coef(fit_la); θva = coef(fit_va)
        β_la = θla[1:2];        β_va = θva[1:2]
        θsz_la = exp(-2 * θla[3]);   θsz_va = exp(-2 * θva[3])   # coef is log σ now; size = exp(−2·coef)
        σ_la = exp(θla[4]);     σ_va = exp(θva[4])

        # Fixed effects: VA should match GHQ closely (mean is the well-behaved axis).
        @test isapprox(β_va[1], β_la[1]; atol = 0.06)
        @test isapprox(β_va[2], β_la[2]; atol = 0.05)
        # Dispersion θ: VA should land near the GHQ MLE and recover θ_true.
        @test isapprox(θsz_va, θsz_la; atol = 0.6)
        @test isapprox(θsz_va, θ_true; atol = 1.2)
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
        yint = round.(Int, y)
        function marg_ll_adaptive(θ, K)              # near-exact marginal log-lik
            βμ = θ[1:2]; rsz = exp(-2 * θ[3]); σb = exp(θ[4]); η0 = Xμ * βμ   # size = exp(−2·log σ)
            z, w = DBNB._gauss_hermite(K); ll = 0.0
            for idx in members
                isempty(idx) && continue
                bm = 0.0                              # per-group posterior mode of b
                for _ in 1:60
                    g = -bm / σb^2; h = -1 / σb^2
                    for i in idx
                        μi = exp(η0[i] + bm)
                        # d/db logNB = y − (y+r)·μ/(r+μ); d²/db² = −(y+r)·rμ/(r+μ)²
                        g += yint[i] - (yint[i] + rsz) * μi / (rsz + μi)
                        h += -(yint[i] + rsz) * rsz * μi / (rsz + μi)^2
                    end
                    bm -= g / h
                end
                h = -1 / σb^2                          # curvature at the mode
                for i in idx
                    μi = exp(η0[i] + bm); h += -(yint[i] + rsz) * rsz * μi / (rsz + μi)^2
                end
                sd = sqrt(-1 / h)                      # adaptive node width
                terms = Float64[]
                for k in 1:K                           # mode-centered adaptive GH
                    bk = bm + sqrt(2) * sd * z[k]
                    lp = -0.5 * log(2π * σb^2) - bk^2 / (2σb^2)
                    for i in idx
                        μi = exp(η0[i] + bk); p = rsz / (rsz + μi)
                        lp += Distributions.logpdf(Distributions.NegativeBinomial(rsz, p), yint[i])
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
