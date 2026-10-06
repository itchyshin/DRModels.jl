# test_reml_ordinary_ranef.jl — TDD S1 for #439 (ordinary Gaussian mean (1 | g) REML).
#
# Wired into test/runtests.jl by Option A (#445) after #423+#428 landed.
# Worked example (after S2 lands):
#
#   using DRModels
#   G, m = 12, 4; n = G * m
#   g = repeat(1:G, inner = m); x = randn(n)
#   y = 0.5 .+ 0.3 .* x .+ randn(G)[g] .+ 0.6 .* randn(n)
#   data = (; y, x, g)
#   fit = drm(bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1)),
#             Gaussian(); data, method = :REML)
#   estimation_method(fit) === :REML
#
# Fence: σ-RE, random slopes, multi-ranef, non-Gaussian REML, q4 are out of scope.
# ML remains the default. Not a drmTMB numeric-parity fixture.
using DRModels
using Test, Random, LinearAlgebra
import ForwardDiff

@testset "REML ordinary Gaussian mean (1 | g) — #439" begin
    Random.seed!(20260817)
    G = 12; m = 4; n = G * m
    g = repeat(1:G, inner = m)
    x = randn(n)
    b = 0.8 .* randn(G)
    y = 0.4 .+ 0.5 .* x .+ b[g] .+ 0.6 .* randn(n)
    data = (; y, x, g)
    fμ = @formula(y ~ 1 + x + (1 | g))
    fσ = @formula(sigma ~ 1)

    @testset "method omitted ≡ :ML (default unchanged)" begin
        f_default = drm(bf(fμ, fσ), Gaussian(); data)
        f_ml      = drm(bf(fμ, fσ), Gaussian(); data, method = :ML)
        @test coef(f_default) == coef(f_ml)
        @test loglik(f_default) == loglik(f_ml)
        @test estimation_method(f_default) === :ML
        @test isnan(reml_loglik(f_default))
        @test ml_loglik(f_default) == loglik(f_default)
    end

    # S1 red: method = :REML currently throws at gaussian_core.jl:413-423.
    # Catch so this is a Fail (not an Error); after S2 the else-branch is the gate.
    freml = nothing
    reml_err = nothing
    try
        freml = drm(bf(fμ, fσ), Gaussian(); data, method = :REML)
    catch e
        reml_err = e
    end

    @testset "method = :REML fits Gaussian mean (1 | g)" begin
        @test reml_err === nothing
        if reml_err !== nothing
            @test reml_err isa ArgumentError
            @test occursin("method = :REML", sprint(showerror, reml_err))
        else
            @test estimation_method(freml) === :REML
            @test isfinite(reml_loglik(freml))
            @test loglik(freml) == reml_loglik(freml)
            @test isfinite(ml_loglik(freml))
            @test freml.converged
        end
    end

    @testset "σ_b²_REML ≥ σ_b²_ML (skip while REML still throws)" begin
        if reml_err !== nothing
            @test_skip "σ_b²_REML ≥ σ_b²_ML — waiting for S2 Woodbury PT"
        else
            fml = drm(bf(fμ, fσ), Gaussian(); data, method = :ML)
            σb²_ml   = re_sd(fml)[:g]^2
            σb²_reml = re_sd(freml)[:g]^2
            @test σb²_reml ≥ σb²_ml - 1e-12
            σ_ml   = exp(coef(fml,   :sigma)[1])
            σ_reml = exp(coef(freml, :sigma)[1])
            @test σ_reml ≥ σ_ml - 1e-12
        end
    end

    @testset "FD ≤ 1e-6 on the restricted Woodbury objective" begin
        freml === nothing && return
        Xμ = hcat(ones(n), x)
        Xσ = reshape(ones(n), n, 1)
        gidx = Int.(g)
        pμ = size(Xμ, 2)
        const_2pi = 0.5 * n * log(2π)
        const_pμ = 0.5 * pμ * log(2π)
        function nll_reml(θ)
            βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+1]; lσb = θ[pμ+2]
            ημ = Xμ * βμ; ησ = Xσ * βσ
            σb² = exp(2lσb)
            T = eltype(θ)
            S = zeros(T, G); C = zeros(T, G)
            ZtDinvX = zeros(T, G, pμ)
            XtDinvX = zeros(T, pμ, pμ)
            q1 = zero(T); logdetD = zero(T)
            @inbounds for i in 1:n
                invD = exp(-2 * ησ[i])
                r = y[i] - ημ[i]
                a = r * invD
                k = gidx[i]
                S[k] += invD
                C[k] += a
                q1 += r * a
                logdetD += 2 * ησ[i]
                @inbounds for j in 1:pμ
                    xj = Xμ[i, j]
                    ZtDinvX[k, j] += invD * xj
                    @inbounds for l in 1:pμ
                        XtDinvX[j, l] += invD * xj * Xμ[i, l]
                    end
                end
            end
            q2 = zero(T); logdetCap = zero(T)
            XtVinvX = copy(XtDinvX)
            @inbounds for k in 1:G
                invMk = 1 / (1 / σb² + S[k])
                q2 += C[k]^2 * invMk
                logdetCap += log(1 + σb² * S[k])
                @inbounds for j in 1:pμ
                    zj = ZtDinvX[k, j]
                    @inbounds for l in 1:pμ
                        XtVinvX[j, l] -= zj * invMk * ZtDinvX[k, l]
                    end
                end
            end
            return 0.5 * (logdetD + logdetCap + q1 - q2) + const_2pi +
                   0.5 * logdet(Symmetric(XtVinvX)) - const_pμ
        end
        gθ = ForwardDiff.gradient(nll_reml, freml.theta)
        @test maximum(abs, gθ) ≤ 1e-6
    end

    @testset "σ-RE and random slopes still rejected under REML" begin
        @test_throws ArgumentError drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + (1 | g))),
                                       Gaussian(); data, method = :REML)
        @test_throws ArgumentError drm(bf(@formula(y ~ 1 + x + (0 + x | g)), @formula(sigma ~ 1)),
                                       Gaussian(); data, method = :REML)
    end

    @testset "model-selection guard still refuses different mean structures" begin
        full = drm(bf(fμ, fσ), Gaussian(); data, method = :REML)
        red  = drm(bf(@formula(y ~ 1 + (1 | g)), fσ), Gaussian(); data, method = :REML)
        @test_throws ArgumentError lrtest(red, full)
        @test isfinite(aic(full))
    end

    # ------------------------------------------------------------------ #499
    # Xμ′V⁻¹Xμ is PSD by construction but formed by Woodbury SUBTRACTION, so it
    # can round to a noise-level NEGATIVE determinant. `logdet` has no Symmetric
    # method and falls through to LU, where logabsdet returns sign -1 and
    # log(-1.0) throws DomainError.
    #
    # First seen as an INTERMITTENT CI failure (pass → fail → pass on an unchanged
    # commit) and assumed to be runner ULP noise. It is not. A mean covariate held
    # CONSTANT WITHIN GROUP is collinear with Z, so as σb² grows the group means
    # absorb the mean signal, Xμ′V⁻¹Xμ → 0, and rounding decides the sign. That
    # design reproduces it deterministically.
    #
    # Deliberately RNG-free: #498 showed MersenneTwister is not stream-stable
    # across Julia versions, and a regression test must not depend on that.
    @testset "REML restriction matrix stays PSD-safe when x is collinear with g (#499)" begin
        G = 8; per = 6
        gi = repeat(1:G, inner = per)
        xg = [-1.5, -0.9, -0.4, 0.1, 0.3, 0.8, 1.2, 1.7]   # one x per group
        bg = [ 3.1, -2.4,  1.8, -3.6, 2.9, -1.1, 0.7, -2.2] # wide group spread
        x  = xg[gi]
        y  = 1.0 .+ 0.5 .* x .+ bg[gi] .+
             0.01 .* [isodd(i) ? 1.0 : -1.0 for i in 1:(G * per)]
        data = (; y, x, g = gi)

        fμ = @formula(y ~ 1 + x + (1 | g))
        fσ = @formula(sigma ~ 1)

        # Before the fix this threw DomainError(-1.0, "log was called with a
        # negative real argument"). A non-PD restriction must reject the step,
        # never throw.
        #
        # This asserts ONLY that the fit completes with a finite criterion. The x
        # slope is NOT identified under this design (x is collinear with g, so the
        # random intercepts absorb it) and is not checked — a degenerate design is
        # exactly the point, and its estimate is not meaningful.
        fit = drm(bf(fμ, fσ), Gaussian(); data, method = :REML)
        @test fit.converged
        @test isfinite(reml_loglik(fit))
    end
end
