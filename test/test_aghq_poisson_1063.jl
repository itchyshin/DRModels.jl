# Poisson (1 | g) `marginal = :AGHQ` must stay evaluable when the outer optimiser
# probes a wild θ (#1063).
#
# Before the fix the group mode was a fixed 12-step Newton unroll from b = 0.
# On the committed 30-group rate fixture the first L-BFGS line-search probe has
# a huge β_x (gradient ≈ [15, −404, −7]); η0 = Xβ then overflows exp(η0 + b),
# the curvature goes NaN and `_aghq_1d_logint` threw "AGHQ Hessian at the mode
# must be negative", aborting `drm` while the default `:LA` / `:Laplace` fit the
# same data. These tests pin: (i) the fit runs and agrees with `:LA`; (ii) the
# group integral is accurate (QuadGK oracle) and its ForwardDiff derivatives are
# exact; (iii) hostile θ gives a finite objective (so the optimiser backs off),
# not an exception.

using DRModels
using Test
using DelimitedFiles
using ForwardDiff
using QuadGK

const DA1063 = DRModels

function _rate_1063()
    m, _ = readdlm(joinpath(@__DIR__, "fixtures", "aghq_poisson_1063", "rate.csv"), ','; header = true)
    return (; y = Float64.(m[:, 1]), x = Float64.(m[:, 2]), logexp = Float64.(m[:, 3]),
            g = String.(m[:, 4]))
end

@testset "Poisson (1|g) :AGHQ — robust inner mode (#1063)" begin
    d = _rate_1063()
    n = length(d.y)
    ug = unique(d.g)
    members = [findall(==(s), d.g) for s in ug]
    X = hcat(ones(n), d.x)
    lf = [DA1063._logfactorial(round(Int, v)) for v in d.y]
    function nll(θ; k = 5)
        η0 = X * θ[1:2]; σb = exp(θ[3])
        s = zero(eltype(θ))
        for idx in members
            s -= DA1063._poisson_group_aghq_logint(d.y, η0, lf, idx, σb, k)
        end
        return s
    end

    @testset "the issue's reproducer fits under :AGHQ and matches :LA" begin
        fm = bf(@formula(y ~ 1 + x + (1 | g)))
        fla = drm(fm, Poisson(); data = d)
        faghq = drm(fm, Poisson(); data = d, marginal = :AGHQ)      # threw on the base commit
        @test faghq.converged
        @test faghq.loglik ≈ fla.loglik atol = 1e-8
        @test faghq.theta ≈ fla.theta atol = 1e-6
        @test all(isfinite, stderror(faghq))
        # nAGQ = 1 is the 1-point Laplace identity: same fit as :Laplace.
        f1 = drm(fm, Poisson(); data = d, marginal = :AGHQ, nAGQ = 1)
        flap = drm(fm, Poisson(); data = d, marginal = :Laplace)
        @test f1.loglik ≈ flap.loglik atol = 1e-8
        # The accurate integral (32 nodes) is the QuadGK-checked one below.
        f32 = drm(fm, Poisson(); data = d, marginal = :AGHQ, nAGQ = 32)
        @test f32.converged
        @test f32.loglik ≈ -774.5923963 atol = 1e-5
    end

    @testset "nll stays finite and differentiable at the optimiser's wild probes" begin
        θ0 = [log(sum(d.y) / n), 0.0, log(0.5)]
        gr = ForwardDiff.gradient(nll, θ0)
        @test all(isfinite, gr)
        for α in (1e-2, 1e-1, 1.0)               # α = 1 threw on the base commit
            θ = θ0 .- α .* gr
            @test isfinite(nll(θ))
            @test all(isfinite, ForwardDiff.gradient(nll, θ))
        end
        # Monotone in the probe distance: a bad θ is a bad (finite) objective, not NaN/garbage.
        @test nll(θ0 .- 1.0 .* gr) > nll(θ0 .- 1e-1 .* gr) > nll(θ0)
    end

    @testset "group integral matches a QuadGK oracle (k = 32), incl. large-count groups" begin
        θ = [1.0, 3.0, log(1.3)]                 # steep slope ⇒ some groups far from b = 0
        η0 = X * θ[1:2]; σb = exp(θ[3])
        for idx in members[1:6]
            f(b) = -0.5 * log(2π) - log(σb) - 0.5 * (b / σb)^2 +
                   sum(d.y[i] * (η0[i] + b) - exp(η0[i] + b) - lf[i] for i in idx)
            bs = range(-12σb, 12σb; length = 4001)
            _, jmax = findmax(f.(bs)); b̂ = bs[jmax]; fmax = f(b̂)
            ref = fmax + log(first(quadgk(b -> exp(f(b) - fmax), -40.0, b̂ - 1.0, b̂, b̂ + 1.0, 40.0;
                                          rtol = 1e-12)))
            @test DA1063._poisson_group_aghq_logint(d.y, η0, lf, idx, σb, 32) ≈ ref atol = 1e-8
        end
    end

    @testset "ForwardDiff gradient and Hessian of the objective are exact" begin
        θ = [1.7, 0.3, log(0.6)]
        g = ForwardDiff.gradient(nll, θ)
        H = ForwardDiff.hessian(nll, θ)
        h = 1e-5
        for j in 1:3
            e = zeros(3); e[j] = h
            @test g[j] ≈ (nll(θ .+ e) - nll(θ .- e)) / (2h) rtol = 1e-6
            @test vec(H[:, j]) ≈ (ForwardDiff.gradient(nll, θ .+ e) .-
                                  ForwardDiff.gradient(nll, θ .- e)) ./ (2h) rtol = 1e-5
        end
    end

    @testset "unevaluable groups return a finite penalty, not an exception" begin
        idx = members[1]
        pen = DA1063._AGHQ_PENALTY_LOGINT
        @test isfinite(pen) && pen < -1e6
        η_nan = fill(NaN, n)
        @test DA1063._poisson_group_aghq_logint(d.y, η_nan, lf, idx, 0.5, 5) == pen
        η_inf = fill(Inf, n)
        @test DA1063._poisson_group_aghq_logint(d.y, η_inf, lf, idx, 0.5, 5) == pen
        for η in (fill(1e3, n), fill(-1e3, n))     # overflowing / underflowing but defined
            v = DA1063._poisson_group_aghq_logint(d.y, η, lf, idx, 0.5, 5)
            @test isfinite(v)
        end
        # An all-zero group (log-posterior mode < 0, b_lik = -Inf) is handled.
        y0 = zeros(n)
        @test isfinite(DA1063._poisson_group_aghq_logint(y0, fill(0.5, n), zeros(n), idx, 0.7, 5))
    end
end
