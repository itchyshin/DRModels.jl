# NB2 dispersion seed — scale regression (A-nb2).
#
# The method-of-moments initialiser computes the NB2 SIZE, r = m^2/(v - m), but
# the parameter it seeds is `eta_sigma = log(sigma)` with `r = exp(-2*eta_sigma)`.
# The conversion is therefore `eta_sigma = -0.5 * log(r)`. The -0.5 was missing at
# six sites, so every NB2 fitter started ~r^(3/2) away from the moment estimate —
# for the case below, a seed of r = 0.14 when the truth is r ~ 2.8.
#
# It mattered because the wrong seed did not always fail: LBFGS recovered on many
# datasets, so the suite stayed green while a whole region of dispersion space
# silently converged to the Poisson boundary (sigma-hat 5.6e-7, `converged=false`)
# and DRModels.jl reported a logLik 291 units worse than drmTMB on the same data.
# Found via the staged-association parity harness, not by a family test.

using DRModels
using Test
using Random
import Distributions

@testset "NB2 dispersion seed is on the log-sigma scale" begin

    @testset "recovers dispersion where the old seed collapsed to Poisson" begin
        rng = MersenneTwister(4)
        n = 1200
        x = randn(rng, n)
        sg_true = 0.6
        r = 1 / sg_true^2
        mu = exp.(1.4 .- 0.2 .* x)
        u = rand(rng, n)
        y = [Float64(Distributions.quantile(
                 Distributions.NegativeBinomial(r, r / (r + mu[i])), u[i])) for i in 1:n]
        d = (; y = y, x = x)

        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), NegBinomial2(); data = d)

        @test is_converged(fit)
        σ̂ = fit.scales[:sigma][1]
        # The failure mode was a collapse to the Poisson boundary, sigma -> 0.
        @test σ̂ > 0.1
        @test isapprox(σ̂, sg_true; atol = 0.15)
        @test isapprox(coef(fit)[1], 1.4; atol = 0.15)
        @test isapprox(coef(fit)[2], -0.2; atol = 0.15)
    end

    @testset "seed direction is right across a dispersion sweep" begin
        # The bug was a SCALE error, so it worsens as dispersion moves away from
        # sigma = 1 (where log r and -0.5 log r happen to agree in sign only).
        # Sweep both sides of 1 and require convergence with a sane sigma.
        for sg_true in (0.35, 0.6, 1.0, 1.6)
            rng = MersenneTwister(11)
            n = 900
            x = randn(rng, n)
            r = 1 / sg_true^2
            mu = exp.(1.2 .+ 0.3 .* x)
            u = rand(rng, n)
            y = [Float64(Distributions.quantile(
                     Distributions.NegativeBinomial(r, r / (r + mu[i])), u[i])) for i in 1:n]
            fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), NegBinomial2();
                      data = (; y = y, x = x))
            @test is_converged(fit)
            @test isapprox(fit.scales[:sigma][1], sg_true; rtol = 0.35)
        end
    end
end
