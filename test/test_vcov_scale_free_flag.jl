# The boundary flag of the Hessian guard does not depend on units (DRModels.jl#1383).
#
# A coordinate is flagged only for a real boundary or non-identification, never because a
# covariate or the response is measured in large or small units. The criterion
# (`_flat_units_free`, src/vcov_guard.jl): a coordinate that carries the units of a
# covariate or of the response is not flagged for being small next to another
# coordinate; every other coordinate is flagged when its curvature is at most 3e-8 times
# the largest curvature among the dimensionless coordinates (the number of observations
# when there is only one).

using DRModels
using Test
using Random
using LinearAlgebra
using Logging
using StableRNGs
import Distributions

const _DVS = DRModels

_quiet(f) = Logging.with_logger(f, Logging.NullLogger())

@testset "scale-free boundary flag" begin
    @testset "unit: the criterion on matrices" begin
        unit = [true, true, false, false]
        # a covariate in units of 1e5 (curvature 1e13 next to 1e3): nothing is flagged
        H = Diagonal([4.0e3, 4.0e13, 480.0, 96.0])
        @test _DVS._flat_units_free(H, unit, 240) == Int[]
        # the same covariate in units of 1e-5 (curvature 1e-7): nothing is flagged
        H = Diagonal([4.0e3, 4.0e-7, 480.0, 96.0])
        @test _DVS._flat_units_free(H, unit, 240) == Int[]
        # a flat dimensionless coordinate is flagged, whatever the units of the others
        for u in (1e-5, 1.0, 1e5)
            H = Diagonal([4.0e3 * u^2, 4.0e3 / u^2, 480.0, 9.85e-16])
            @test _DVS._flat_units_free(H, unit, 240) == [4]
        end
        # a coordinate with exactly zero curvature is flagged whatever its kind
        @test _DVS._flat_units_free(Diagonal([4.0e3, 0.0, 480.0, 96.0]), unit, 240) == [2]
        @test _DVS._flat_units_free(zeros(4, 4), unit, 240) == [1, 2, 3, 4]
        # only one dimensionless coordinate: the number of observations is the reference
        unit1 = [true, true, true, false]
        @test _DVS._flat_units_free(Diagonal([1e13, 1e-9, 4e3, 480.0]), unit1, 240) == Int[]
        @test _DVS._flat_units_free(Diagonal([1e13, 1e-9, 4e3, 1e-9]), unit1, 240) == [4]
        # invariance: rescaling the unit-bearing coordinates (H -> S H S) never changes the flags
        rng = MersenneTwister(1)
        B = randn(rng, 4, 4); H = B * B' + 0.1I; H[4, :] .*= 1e-12; H[:, 4] .*= 1e-12; H[4, 4] = 1e-15
        base = _DVS._flat_units_free(H, unit, 240)
        @test base == [4]
        for s1 in (1e-5, 1e-2, 1.0, 1e3, 1e5), s2 in (1e-5, 1.0, 1e5)
            S = Diagonal([s1, s2, 1.0, 1.0])
            @test _DVS._flat_units_free(S * H * S, unit, 240) == base
        end
    end

    # Gaussian y ~ x + (1 | g), sigma ~ x: a well identified fit.
    function _sf_gaussian(; xscale = 1.0, yscale = 1.0)
        rng = MersenneTwister(1503)
        n = 240; G = 24
        g = repeat(1:G, inner = n ÷ G)
        x0 = randn(rng, n); b = 0.4 .* randn(rng, G)
        y = 0.3 .+ 0.5 .* x0 .+ b[g] .+ exp.(-0.3 .+ 0.15 .* x0) .* randn(rng, n)
        return _quiet() do
            drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x)), Gaussian();
                data = (; y = yscale .* y, x = xscale .* x0, g))
        end
    end

    @testset "invariance: a well identified Gaussian fit, covariate and response units" begin
        base = _sf_gaussian()
        se0 = stderror(base)
        @test isempty(bridge_diagnostics(base).boundary)
        @test is_converged(base)
        @test all(isfinite, se0)
        # a covariate in units of 1e5 and 1e-5: no flag, finite SEs that rescale as theory says
        for u in (1e-5, 1e5)
            fit = _sf_gaussian(xscale = u)
            se = stderror(fit)
            @test isempty(bridge_diagnostics(fit).boundary)
            @test all(isfinite, se) && all(isfinite, vcov(fit))
            @test se[2] * u ≈ se0[2] rtol = 2e-3          # slope of mu: per unit of x
            @test se[4] * u ≈ se0[4] rtol = 2e-3          # slope of log sigma
            @test se[[1, 3, 5]] ≈ se0[[1, 3, 5]] rtol = 2e-3
        end
        @test is_converged(_sf_gaussian(xscale = 1e5))
        # the response in units of 1e3: mu coefficients scale with it, log scale ones do not
        fit = _sf_gaussian(yscale = 1e3)
        se = stderror(fit)
        @test isempty(bridge_diagnostics(fit).boundary)
        @test all(isfinite, se)
        @test se[1:2] ≈ 1e3 .* se0[1:2] rtol = 2e-3
        @test se[3:5] ≈ se0[3:5] rtol = 2e-3
        @test is_converged(fit)
        # both at once
        fit = _sf_gaussian(xscale = 1e5, yscale = 1e3)
        @test isempty(bridge_diagnostics(fit).boundary) && all(isfinite, stderror(fit))
    end

    @testset "invariance: a Poisson random-intercept fit, covariate units" begin
        function pois(u)
            rng = MersenneTwister(77)
            G = 20; n = 200
            g = repeat(1:G, inner = n ÷ G)
            x0 = randn(rng, n); b = 0.5 .* randn(rng, G)
            y = Float64.([rand(rng, Distributions.Poisson(exp(0.3 + 0.4 * x0[i] + b[g[i]]))) for i in 1:n])
            return _quiet() do
                drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = (; y, x = u .* x0, g))
            end
        end
        base = pois(1.0)
        @test isempty(bridge_diagnostics(base).boundary)
        for u in (1e-5, 1e5)
            fit = pois(u)
            se = stderror(fit)
            @test isempty(bridge_diagnostics(fit).boundary)
            @test all(isfinite, se)
            @test se[2] * u ≈ stderror(base)[2] rtol = 5e-3
            @test se[[1, 3]] ≈ stderror(base)[[1, 3]] rtol = 5e-3
        end
    end

    @testset "true boundaries stay flagged at every covariate unit" begin
        # random-effect SD at zero: group means of the residual are exactly zero
        function re_zero(u)
            rng = MersenneTwister(5)
            G = 20; m = 8; n = G * m
            g = repeat(1:G, inner = m)
            x0 = randn(rng, n)
            e = 0.5 .* randn(rng, n)
            for i in 1:G
                r = g .== i
                e[r] .-= sum(e[r]) / m
            end
            return _quiet() do
                drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian();
                    data = (; y = 0.3 .+ 0.5 .* x0 .+ e, x = u .* x0, g))
            end
        end
        for u in (1e-5, 1.0, 1e5)
            fit = re_zero(u)
            @test bridge_diagnostics(fit).boundary == [4]
            @test isnan(stderror(fit)[4]) && all(isfinite, stderror(fit)[1:3])
            # the status also applies the rule on inflated SEs (units of 1e-5 give a slope
            # SE in the thousands, the same rule as drmTMB), so it is read at the other two
            u == 1e-5 || @test convergence_status(fit) === :boundary
        end

        # NB2 fitted to Poisson data: the size runs to infinity
        function nb2(u)
            rng = MersenneTwister(11)
            n = 150
            x0 = randn(rng, n)
            y = Float64.([rand(rng, Distributions.Poisson(exp(0.5 + 0.3 * x0[i]))) for i in 1:n])
            return _quiet() do
                drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), NegBinomial2(); data = (; y, x = u .* x0))
            end
        end
        for u in (1e-5, 1.0, 1e5)
            fit = nb2(u)
            @test bridge_diagnostics(fit).boundary == [3]
            @test isnan(stderror(fit)[3]) && all(isfinite, stderror(fit)[1:2])
        end

        # complete separation: flagged by the separation screen, whatever the units
        xs = collect(range(-2, 2; length = 40))
        ys = Float64.(xs .> 0)
        for u in (1e-5, 1.0, 1e5)
            fit = _quiet() do
                drm(bf(@formula(y ~ x)), Binomial(); data = (; y = ys, x = u .* xs))
            end
            @test bridge_diagnostics(fit).boundary == [1, 2]
            @test convergence_status(fit) === :degenerate
        end

        # an aliased column: NaN for the aliased coefficient, not converged
        for u in (1e-5, 1.0, 1e5)
            rng = MersenneTwister(3)
            x0 = randn(rng, 100)
            fit = _quiet() do
                drm(bf(@formula(y ~ x + z), @formula(sigma ~ 1)), Gaussian();
                    data = (; y = 0.2 .+ 0.4 .* x0 .+ 0.3 .* randn(rng, 100), x = u .* x0, z = 2 .* u .* x0 .+ 1))
            end
            @test !is_converged(fit)
            @test any(isnan, stderror(fit))
        end

        # a residual correlation at its boundary (atanh rho about 5.7): the Hessian stays
        # regular there, so no coordinate is flagged; the status says boundary
        for u in (1e-5, 1.0, 1e5)
            rng = MersenneTwister(9)
            n = 60
            x0 = randn(rng, n)
            y1 = 0.3 .+ 0.5 .* x0 .+ randn(rng, n)
            y2 = 1 .+ 0.8 .* y1 .+ 0.005 .* randn(rng, n)
            fit = _quiet() do
                drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x), sigma1 = @formula(sigma1 ~ 1),
                       sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1)), Gaussian();
                    data = (; x = u .* x0, y1, y2))
            end
            # nothing flagged: the correlation is at its boundary, the Hessian is not singular
            @test isempty(bridge_diagnostics(fit).boundary)
            @test all(isfinite, stderror(fit))
            u == 1.0 && @test convergence_status(fit) === :boundary
        end
    end

    @testset "the flag is read once, from the fit: stored all-flagged marker" begin
        # every coordinate flagged: stderror is NaN everywhere (not Inf), as vcov
        n = 200
        x = randn(MersenneTwister(3), n)
        fit = _quiet() do
            drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Tweedie();
                data = (; y = zeros(n), x))
        end
        @test bridge_diagnostics(fit).boundary == 1:4
        @test all(isnan, vcov(fit))
        @test all(isnan, stderror(fit))
        ci = confint(fit; method = :wald)
        @test all(r -> isnan(r.lower) && isnan(r.upper), ci)
        @test all(isnan, coeftable(fit).cols[2])
        # a covariance that was never computed (se = false) is not a boundary: Inf as before
        fit2 = _quiet() do
            drm(bf(@formula(y ~ x)), Poisson(); data = (; y = Float64.(rand(MersenneTwister(1), 0:3, n)), x), se = false)
        end
        @test all(isnan, vcov(fit2))
        @test all(==(Inf), stderror(fit2))
    end
end
