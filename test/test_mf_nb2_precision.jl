# NB2 observation log-density of the mixed-family fitter at tiny sigma.
# With size θ = 1/σ² near 1e13 (σ clamp), loggamma(y + θ) - loggamma(θ) cancels
# and the density was off by up to 0.05 nats per observation, so the Poisson
# limit was scored wrongly. Reference: exact Σ_{j<y} log(θ + j) in 512-bit BigFloat.
using DRModels, Test, ForwardDiff, SpecialFunctions

@testset "mixed-family NB2 density: precision at tiny sigma" begin
    fam = DRModels.NegBinomial2()
    f(η, y, d) = DRModels._mf_obs_ll(fam, η, Float64(y), 1.0, d)
    function ref(η, y, d)
        setprecision(BigFloat, 512) do
            μ = exp(BigFloat(η)); θ = inv(BigFloat(clamp(d, 3.1e-7, 3.3e6))^2)
            A = sum(log(θ + j) for j in 0:y-1; init = BigFloat(0))
            lf = sum(log(BigFloat(j)) for j in 2:y; init = BigFloat(0))
            l1p = log1p(μ / θ)
            Float64(A - lf - θ * l1p + y * log(μ) - y * log(θ) - y * l1p)
        end
    end
    etas = (log(0.5), 0.0, log(5.0), log(50.0))
    ys = (0, 1, 3, 10, 50, 200)

    @testset "absolute error against BigFloat, log sigma from -2 to -20" begin
        for ls in (-2, -3, -4, -5, -6, -8, -10, -12, -15, -20), η in etas, y in ys
            d = exp(Float64(ls))
            @test abs(f(η, y, d) - ref(η, y, d)) < 5e-12 * max(1, abs(ref(η, y, d)) / 100)
        end
    end

    @testset "ordinary sigma is unchanged (log sigma >= -3)" begin
        old(η, y, d) = (μ = exp(η); θ = inv(d^2);
                        loggamma(y + θ) - loggamma(θ) - loggamma(y + 1) +
                        θ * (log(θ) - log(θ + μ)) + y * (log(μ) - log(θ + μ)))
        for ls in -3:0.25:2, η in etas, y in ys
            d = exp(ls)
            @test abs(f(η, y, d) - old(η, y, Float64(d))) < 1e-12
        end
    end

    @testset "smooth Poisson limit" begin
        for η in etas, y in ys
            pois = y * η - exp(η) - loggamma(y + 1)
            @test abs(f(η, y, exp(-15.0)) - pois) < 1e-6
            @test abs(f(η, y, exp(-20.0)) - pois) < 1e-6
        end
    end

    @testset "ForwardDiff gradient is finite and matches a finite difference" begin
        for ls in (-2.0, -4.0, -6.0, -8.0), η in (log(0.5), log(5.0)), y in (0, 7, 200)
            gσ = ForwardDiff.derivative(x -> f(η, y, exp(x)), ls)
            gη = ForwardDiff.derivative(x -> f(x, y, exp(ls)), η)
            @test isfinite(gσ) && isfinite(gη)
            h = 1e-4
            fdσ = (f(η, y, exp(ls + h)) - f(η, y, exp(ls - h))) / (2h)
            @test isapprox(gσ, fdσ; atol = 1e-6, rtol = 1e-5)
            @test isapprox(gη, y - exp(η) * (y + exp(-2ls)) / (exp(-2ls) + exp(η)); atol = 1e-8, rtol = 1e-8)
        end
        for ls in (-10.0, -15.0, -20.0), y in (0, 7, 200)
            @test isfinite(ForwardDiff.derivative(x -> f(log(5.0), y, exp(x)), ls))
        end
    end
end
