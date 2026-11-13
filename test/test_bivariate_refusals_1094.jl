# #1094: bivariate Gaussian with an ordinary correlated random effect on a
# sigma axis is not supported (drmTMB fits it; the mu1/mu2 form is now supported,
# see test_bivariate_ordinary_re.jl). The refusal must be an
# ArgumentError that names drmTMB, and `marginal =` on the bivariate method must
# be an ArgumentError, not a bare MethodError.

using DRModels
using Test
using Random

@testset "bivariate Gaussian refusals (#1094)" begin
    rng = MersenneTwister(1094)
    n = 60                                  # cabbages-style: 60 rows, 3 groups
    Cult = repeat(["c39", "c52"], inner = 30)
    Date = repeat(["d16", "d20", "d21"], outer = 20)
    y1 = randn(rng, n)
    y2 = randn(rng, n)
    data = (; y1, y2, Cult, Date)

    f_re = bf(mu1 = @formula(y1 ~ 1 + Cult + (1 | p | Date)),
              mu2 = @formula(y2 ~ 1 + Cult + (1 | p | Date)),
              sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
              rho12 = @formula(rho12 ~ 1))
    f_re_sigma = bf(mu1 = @formula(y1 ~ 1 + Cult), mu2 = @formula(y2 ~ 1 + Cult),
                    sigma1 = @formula(sigma1 ~ 1 + (1 | p | Date)),
                    sigma2 = @formula(sigma2 ~ 1 + (1 | p | Date)),
                    rho12 = @formula(rho12 ~ 1))
    f_fixed = bf(mu1 = @formula(y1 ~ 1 + Cult), mu2 = @formula(y2 ~ 1 + Cult),
                 sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                 rho12 = @formula(rho12 ~ 1))

    @testset "ordinary (1 | p | g) on sigma names drmTMB" begin
        err = try
            drm(f_re_sigma, Gaussian(); data = data)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = err === nothing ? "" : sprint(showerror, err)
        @test occursin("drmTMB", msg)
        @test occursin("bivariate q=4", msg)      # existing wording kept
    end

    @testset "ordinary (1 | p | g) on mu1 and mu2 now fits" begin
        fit = drm(f_re, Gaussian(); data = data)
        @test isfinite(DRModels.loglik(fit))
    end

    @testset "marginal= on the bivariate method is an ArgumentError" begin
        for m in (:Laplace, :VA, :AGHQ)
            err = try
                drm(f_fixed, Gaussian(); data = data, marginal = m)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test !(err isa MethodError)
            @test err === nothing || occursin("marginal", sprint(showerror, err))
        end
        # the default integrator is still accepted and the fixed-effect fit runs
        fit = drm(f_fixed, Gaussian(); data = data, marginal = :LA)
        @test isfinite(DRModels.loglik(fit))
    end
end
