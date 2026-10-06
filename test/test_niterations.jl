# test_niterations.jl — #466: fit.niterations wired into every family fitter
# that actually runs an iterative optimiser, not just Gaussian / bivariate
# Gaussian. Reporting-only: no fitted value, loglik, or convergence flag is
# touched by this file or by the wiring it exercises — see docstring on
# `niterations` (src/gaussian_core.jl) for the full per-family coverage table.
#
# These are SMALL, targeted fits (niterations only needs `> 0`, not parameter
# recovery), so `n` is kept modest and no `atol` recovery checks are made here
# — those already live in each family's own test file.
using DRModels
using Test, Random
import Distributions

@testset "niterations — iterative fitters report a real count" begin
    Random.seed!(46600)

    @testset "Gaussian (ML and REML fixed-effects)" begin
        n = 500; x = randn(n)
        y = 0.5 .+ 0.8 .* x .+ 0.6 .* randn(n)
        data = (; y, x)
        fit_ml = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = data)
        @test niterations(fit_ml) > 0
        fit_reml = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian();
                        data = data, method = :REML)
        @test niterations(fit_reml) > 0
    end

    @testset "Student" begin
        n = 500; x = randn(n)
        y = 0.5 .+ 0.8 .* x .+ 0.6 .* rand(Distributions.TDist(6.0), n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Student();
                  data = (; y, x))
        @test niterations(fit) > 0
    end

    @testset "SkewNormal" begin
        n = 500; x = randn(n)
        y = 1.0 .+ 0.5 .* x .+ 0.8 .* randn(n)   # slant ≈ 0 is a legal skew-normal fit
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), SkewNormal();
                  data = (; y, x))
        @test niterations(fit) > 0
    end

    @testset "Poisson (fixed effects and (1|g))" begin
        n = 500; x = randn(n)
        y = Float64.(rand.(Distributions.Poisson.(exp.(0.3 .+ 0.4 .* x))))
        fit = drm(bf(@formula(y ~ x)), Poisson(); data = (; y, x))
        @test niterations(fit) > 0

        G, m = 20, 15
        g = repeat(1:G, inner = m)
        xg = randn(G * m)
        b = 0.4 .* randn(G)
        yg = Float64.([rand(Distributions.Poisson(exp(0.2 + 0.3 * xg[i] + b[g[i]]))) for i in eachindex(xg)])
        fit_re = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = (; y = yg, x = xg, g))
        @test niterations(fit_re) > 0
    end

    @testset "NegBinomial2 and TruncatedNegBinomial2" begin
        n = 500; x = randn(n)
        μ = exp.(0.6 .+ 0.3 .* x); θ = 3.0
        y = Float64.([rand(Distributions.NegativeBinomial(θ, θ / (θ + μi))) for μi in μ])
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), NegBinomial2(); data = (; y, x))
        @test niterations(fit) > 0

        rtnb(r, p) = (while true; k = rand(Distributions.NegativeBinomial(r, p)); k > 0 && return k; end)
        yt = Float64.([rtnb(θ, θ / (θ + μi)) for μi in μ])
        fitt = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), TruncatedNegBinomial2(); data = (; y = yt, x))
        @test niterations(fitt) > 0
    end

    @testset "Beta and BetaBinomial" begin
        n = 500; x = randn(n)
        μ = 1 ./ (1 .+ exp.(-(0.3 .+ 0.6 .* x))); φ = 12.0
        y = Float64.([rand(Distributions.Beta(μi * φ, (1 - μi) * φ)) for μi in μ])
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Beta(); data = (; y, x))
        @test niterations(fit) > 0

        ntr = fill(20, n)
        s = [rand(Distributions.BetaBinomial(ntr[i], μ[i] * φ, (1 - μ[i]) * φ)) for i in 1:n]
        fail = ntr .- s
        fitb = drm(bf(@formula(cbind(s, fail) ~ x), @formula(sigma ~ 1)), BetaBinomial();
                   data = (; s = Float64.(s), fail = Float64.(fail), x))
        @test niterations(fitb) > 0
    end

    @testset "Binomial" begin
        n = 500; x = randn(n)
        μ = 1 ./ (1 .+ exp.(-(-0.4 .+ 0.9 .* x)))
        ntr = fill(15, n)
        s = [rand(Distributions.Binomial(ntr[i], μ[i])) for i in 1:n]
        fail = ntr .- s
        fit = drm(bf(@formula(cbind(s, fail) ~ x)), Binomial();
                  data = (; s = Float64.(s), fail = Float64.(fail), x))
        @test niterations(fit) > 0
    end

    @testset "Gamma" begin
        n = 500; x = randn(n)
        μ = exp.(0.4 .+ 0.3 .* x); α = 6.0
        y = Float64.([rand(Distributions.Gamma(α, μi / α)) for μi in μ])
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gamma(); data = (; y, x))
        @test niterations(fit) > 0
    end

    @testset "LogNormal (univariate and bivariate)" begin
        n = 500; x = randn(n)
        y = exp.(0.4 .+ 0.5 .* x .+ 0.3 .* randn(n))
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), LogNormal(); data = (; y, x))
        @test niterations(fit) > 0

        nb = 400; xb = randn(nb)
        s1, s2, rho = 0.5, 0.8, 0.6
        z1 = randn(nb); z2 = rho .* z1 .+ sqrt(1 - rho^2) .* randn(nb)
        y1 = exp.(0.4 .+ 0.9 .* xb .+ s1 .* z1)
        y2 = exp.(-0.2 .+ 0.5 .* xb .+ s2 .* z2)
        fb = bf(mu1 = @formula(y1 ~ xb), mu2 = @formula(y2 ~ xb),
                sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                rho12 = @formula(rho12 ~ 1))
        fitbiv = drm(fb, LogNormal(); data = (; y1, y2, xb))
        @test niterations(fitbiv) > 0     # #466: borrowed from the Gaussian donor fit, not left at -1
    end

    @testset "ZeroOneBeta" begin
        n = 500; x = randn(n)
        β = [0.2, 0.6]; φ = 12.0; zoi = 0.25; coi = 0.4
        μ = 1 ./ (1 .+ exp.(-(β[1] .+ β[2] .* x)))
        y = Vector{Float64}(undef, n)
        for i in 1:n
            if rand() < zoi
                y[i] = rand() < coi ? 1.0 : 0.0
            else
                y[i] = rand(Distributions.Beta(μ[i] * φ, (1 - μ[i]) * φ))
            end
        end
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)),
                  ZeroOneBeta(); data = (; y, x))
        @test niterations(fit) > 0
    end

    @testset "Tweedie" begin
        function _rtweedie(μ, φ, p)
            λ = μ^(2 - p) / (φ * (2 - p)); γ = φ * (p - 1) * μ^(p - 1); sh = (2 - p) / (p - 1)
            Nn = rand(Distributions.Poisson(λ))
            Nn == 0 ? 0.0 : rand(Distributions.Gamma(Nn * sh, γ))
        end
        n = 500; x = randn(n)
        μ = exp.(0.5 .+ 0.3 .* x)
        y = [_rtweedie(μ[i], 2.0, 1.5) for i in 1:n]
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Tweedie();
                  data = (; y, x))
        @test niterations(fit) > 0
    end

    @testset "CumulativeLogit" begin
        n = 800; x = randn(n)
        βslope = 0.8; θtrue = [-1.0, 0.0, 1.2]; K = 4
        η = βslope .* x
        y = Vector{Int}(undef, n)
        for i in 1:n
            u = rand(); yi = K
            for k in 1:(K-1)
                if u < 1 / (1 + exp(-(θtrue[k] - η[i])))
                    yi = k; break
                end
            end
            y[i] = yi
        end
        fit = drm(bf(@formula(y ~ x)), CumulativeLogit(); data = (; y = Float64.(y), x))
        @test niterations(fit) > 0
    end

    @testset "Bivariate Student" begin
        n = 500; x = randn(n)
        s1, s2, rho, nu = 0.7, 1.1, 0.5, 6.0
        z1 = randn(n); z2 = rho .* z1 .+ sqrt(1 - rho^2) .* randn(n)
        chi = [sum(randn(6) .^ 2) for _ in 1:n]
        sh = sqrt.(nu ./ chi)
        y1 = 0.5 .+ 0.8 .* x .+ s1 .* z1 .* sh
        y2 = -0.3 .+ 0.4 .* x .+ s2 .* z2 .* sh
        f = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
               sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
               nu = @formula(nu ~ 1), rho12 = @formula(rho12 ~ 1))
        fit = drm(f, Student(); data = (; y1, y2, x))
        @test niterations(fit) > 0
    end

    @testset "-1 stays honest: Poisson Cox-Reid REML (1|g) (untracked secondary refit)" begin
        # _fit_poisson_ranef's `method = :REML` branch re-optimises via a separate
        # restricted refit (`_glsp_reml_refit_clean`) whose own iteration count is
        # not tracked, so `Optim.iterations(res)` from the seeding ML run would be
        # a MISMATCH, not a full count — niterations stays the honest default -1.
        G, m = 10, 6
        g = repeat(1:G, inner = m)
        x = randn(G * m)
        b = 0.6 .* randn(G)
        y = Float64.([rand(Distributions.Poisson(exp(0.3 + 0.2 * x[i] + b[g[i]]))) for i in eachindex(x)])
        fit = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = (; y, x, g), method = :REML)
        @test estimation_method(fit) === :REML
        @test niterations(fit) == -1
    end
end
