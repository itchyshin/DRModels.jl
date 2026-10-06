# 1-D Liu–Pierce AGHQ (#448, lever 2).
#
# TDD first: this file is the contract. k=1 is 1-point Laplace *plumbing*,
# not a quadrature or recovery headline. The k≈5 smoke is nll *agreement*
# with existing non-adaptive GHQ-32 on a tiny Poisson `(1 | g)` — not
# parameter recovery, not drmTMB −7.3/−5.0/−0.9, not a GLLVM Λ claim.
#
# Default `marginal = :LA` stays today's GHQ-32. Capability row stays missing.

using DRModels
using Test
using Random
using LinearAlgebra
import Distributions

const DA = DRModels

# ── S1 / S2 kernel ───────────────────────────────────────────────────────────

@testset "1-D Liu–Pierce AGHQ kernel (#448)" begin

    @testset "k=1 is the 1-point Laplace identity (plumbing, not quadrature)" begin
        # ∫ exp(−½ (x−μ)² / s²) dx = s √(2π). Liu–Pierce k=1 is Laplace:
        #   log ∫ ≈ logf(mode) + ½ log(2π) − ½ log(−hess).
        μ, s = 0.4, 1.7
        logf(x) = -0.5 * ((x - μ) / s)^2
        mode, hess = μ, -1 / s^2
        laplace = logf(mode) + 0.5 * log(2π) - 0.5 * log(-hess)
        aghq1 = DA._aghq_1d_logint(logf, mode, hess; k = 1)
        @test aghq1 ≈ laplace atol = 1e-14
        @test aghq1 ≈ log(s * sqrt(2π)) atol = 1e-12
    end

    @testset "fail-loud when the integral is not 1-D" begin
        logf(x) = -0.5 * sum(abs2, x)
        err = try
            DA._aghq_1d_logint(logf, [0.0, 0.0], -I(2); k = 5)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("1-D", msg) || occursin("dimension", lowercase(msg))
        @test occursin("448", msg) || occursin("AGHQ", msg)
        @test_throws ArgumentError DA._aghq_require_1d(2)
        @test DA._aghq_require_1d(1) === nothing
    end

    @testset "adaptive k=5 nll agrees with GHQ-32 on a tiny Poisson (1|g) — not recovery" begin
        rng = MersenneTwister(448)
        G = 8; per = 6; n = G * per
        g = repeat(1:G, inner = per)
        x = randn(rng, n)
        β = [0.25, 0.40]; σb = 0.55
        bg = σb .* randn(rng, G)
        λ = exp.(β[1] .+ β[2] .* x .+ bg[g])
        y = Float64.([rand(rng, Distributions.Poisson(λi)) for λi in λ])
        Xμ = hcat(ones(n), x)
        gidx, _ = DA._group_index(g)
        members = [Int[] for _ in 1:G]
        for i in 1:n
            push!(members[gidx[i]], i)
        end
        lf = [DA._logfactorial(round(Int, yi)) for yi in y]
        θ = [β[1], β[2], log(σb)]   # same θ for both integrals

        # Existing non-adaptive GHQ-32 (the production `:LA` rule on `(1|g)`).
        z32, w32 = DA._gauss_hermite(32)
        logw32 = log.(w32)
        rt2 = sqrt(2.0)
        lπ = log(π)
        function nll_ghq32(θ)
            βμ = θ[1:2]; σ = exp(θ[3]); η0 = Xμ * βμ
            s = 0.0
            for idx in members
                isempty(idx) && continue
                terms = Vector{Float64}(undef, 32)
                for k in 1:32
                    δ = rt2 * σ * z32[k]
                    gll = logw32[k]
                    for i in idx
                        η = η0[i] + δ
                        gll += y[i] * η - exp(η) - lf[i]
                    end
                    terms[k] = gll
                end
                mx = maximum(terms)
                s -= (-0.5 * lπ + mx + log(sum(exp.(terms .- mx))))
            end
            return s
        end

        function nll_aghq5(θ)
            βμ = θ[1:2]; σ = exp(θ[3]); η0 = Xμ * βμ
            s = 0.0
            for idx in members
                isempty(idx) && continue
                s -= DA._poisson_group_aghq_logint(y, η0, lf, idx, σ, 5)
            end
            return s
        end

        nll_a = nll_aghq5(θ)
        nll_g = nll_ghq32(θ)
        @test isfinite(nll_a) && isfinite(nll_g)
        # Agreement of the *integral*, not recovery of β / σ_b.
        @test nll_a ≈ nll_g atol = 5e-3 rtol = 0
        @info "AGHQ k=5 vs GHQ-32 nll agreement" nll_aghq5 = nll_a nll_ghq32 = nll_g absdiff = abs(nll_a - nll_g)
    end
end

# ── S3 public surface ────────────────────────────────────────────────────────

function _aghq_err(expr)
    try
        expr()
        nothing
    catch e
        e
    end
end

@testset "Poisson (1|g) public AGHQ surface (#448)" begin

    @testset "_marginal_method accepts :AGHQ beside :LA/:VA" begin
        @test DA.AGHQ() isa DA.MarginalMethod
        @test DA._marginal_method(:AGHQ) == DA.AGHQ()
        @test DA._marginal_method(:aghq) == DA.AGHQ()
        @test DA._marginal_method(:LA) == DA.Laplace()
        @test DA._marginal_method(:VA) == DA.Variational()
    end

    rng = MersenneTwister(44801)
    G = 12; per = 6; n = G * per
    g = repeat(1:G, inner = per); x = randn(rng, n)
    β = [0.20, 0.35]; σb = 0.50
    bg = σb .* randn(rng, G)
    λ = exp.(β[1] .+ β[2] .* x .+ bg[g])
    y = Float64.([rand(rng, Distributions.Poisson(λi)) for λi in λ])
    data = (; y, x, g)

    @testset "marginal=:AGHQ nAGQ=5 fits (1|g); default :LA is unchanged GHQ-32" begin
        fit_aghq = drm(bf(@formula(y ~ x + (1 | g))), Poisson();
                       data = data, marginal = :AGHQ, nAGQ = 5)
        fit_la = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = data)
        fit_default = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = data, marginal = :LA)
        @test is_converged(fit_aghq)
        @test fit_aghq.marginal === :AGHQ
        @test fit_la.marginal === :LA
        @test fit_default.marginal === :LA
        @test coef(fit_default) == coef(fit_la)
        @test loglik(fit_default) == loglik(fit_la)
        @test isfinite(loglik(fit_aghq)) && isfinite(loglik(fit_la))
        # Fitted nll after two independent opts — agreement, not recovery.
        # Kernel smoke above is the tight integral check (fixed θ, atol=5e-3).
        # CI Julia 1 measured |Δll|≈0.0569 (−105.64377 vs −105.58690) against
        # the old atol=0.05; Mac local saw ~2e-4 on a different draw. atol=0.1
        # is ~1.8× the CI gap — not so loose a broken AGHQ (O(1) nll) would pass.
        @test loglik(fit_aghq) ≈ loglik(fit_la) atol = 0.1
        @info "AGHQ vs LA fitted loglik" ll_aghq = loglik(fit_aghq) ll_la = loglik(fit_la) absdiff = abs(loglik(fit_aghq) - loglik(fit_la))
    end

    @testset "method=:AGHQ points at marginal; :REML×:AGHQ is rejected" begin
        e_method = _aghq_err(() -> drm(bf(@formula(y ~ x + (1 | g))), Poisson();
                                       data = data, method = :AGHQ))
        @test e_method isa ArgumentError
        @test occursin("marginal", sprint(showerror, e_method))
        e_reml = _aghq_err(() -> drm(bf(@formula(y ~ x + (1 | g))), Poisson();
                                     data = data, marginal = :AGHQ, method = :REML))
        @test e_reml isa ArgumentError
        msg = sprint(showerror, e_reml)
        @test occursin("REML", msg) && occursin("AGHQ", msg)
    end

    @testset "fail-loud: phylo / crossed / relmat / (1+x|g) / associate_pairs" begin
        h = repeat(1:G, inner = per)[randperm(rng, n)]
        data2 = (; y, x, g, h)

        e_corr = _aghq_err(() -> drm(bf(@formula(y ~ x + (1 + x | g))), Poisson();
                                     data = data, marginal = :AGHQ))
        e_cross = _aghq_err(() -> drm(bf(@formula(y ~ x + (1 | g) + (1 | h))), Poisson();
                                      data = data2, marginal = :AGHQ))
        e_fe = _aghq_err(() -> drm(bf(@formula(y ~ x)), Poisson();
                                   data = data, marginal = :AGHQ))

        phy = random_balanced_tree(8; branch_length = 0.2)
        species = repeat(1:8, inner = 3)
        yp = Float64.([rand(rng, Distributions.Poisson(exp(0.2))) for _ in species])
        e_phy = _aghq_err(() -> drm(bf(@formula(y ~ 1 + phylo(1 | species))), Poisson();
                                    data = (; y = yp, species), tree = phy, se = false,
                                    marginal = :AGHQ))

        id = repeat(1:6, inner = 3)
        yr = Float64.([rand(rng, Distributions.Poisson(exp(0.2))) for _ in id])
        C = Matrix{Float64}(I(6))
        e_rel = _aghq_err(() -> drm(bf(@formula(y ~ 1 + relmat(1 | id))), Poisson();
                                    data = (; y = yr, id), K = C, se = false,
                                    marginal = :AGHQ))

        for (e, tag) in ((e_corr, "corr"), (e_cross, "crossed"), (e_fe, "fe"),
                         (e_phy, "phylo"), (e_rel, "relmat"))
            @test e isa ArgumentError
            msg = sprint(showerror, e)
            @test occursin("AGHQ", msg)
            # crossed intercepts are now an AGHQ route (#761) that refuses only when
            # both groupings have more than 8 levels (this fixture: 12 and 12+)
            @test occursin(tag == "crossed" ? "761" : "448", msg)
        end

        # associate_pairs QuadGK is not AGHQ — reject the keyword rather than
        # silently treating a rectangle integral as Liu–Pierce.
        d_g = (y = randn(rng, 20), x = randn(rng, 20),
               z = Float64.(rand(rng, 20) .> 0.5))
        fg = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = d_g)
        fb = drm(bf(@formula(z ~ x)), Binomial(); data = d_g)
        e_ap = _aghq_err(() -> associate_pairs(fg, fb; kernel = latent_normal(),
                                               marginal = :AGHQ))
        @test e_ap isa ArgumentError
        @test occursin("AGHQ", sprint(showerror, e_ap))
        @test occursin("448", sprint(showerror, e_ap)) || occursin("QuadGK", sprint(showerror, e_ap))
    end

    @testset "Binomial (1|g) rejects :AGHQ (Poisson-only this slice)" begin
        z = Float64.([rand(rng) > 0.4 ? 1.0 : 0.0 for _ in 1:n])
        e_bin = _aghq_err(() -> drm(bf(@formula(z ~ x + (1 | g))), Binomial();
                                    data = (; z, x, g), marginal = :AGHQ))
        @test e_bin isa ArgumentError
        @test occursin("AGHQ", sprint(showerror, e_bin))
    end
end

