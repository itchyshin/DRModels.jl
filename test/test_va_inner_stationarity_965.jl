# DRModels.jl #965: the `marginal = :VA` fitters profile each group's variational
# (m, log s) with an inner Newton solve, and the outer `converged` flag could not
# see whether that solve reached stationarity. Before the fix the five inner
# solves were fixed 25/30-step unrolls with a coordinate-wise step clamp and no
# stationarity check. Three failure mechanisms were measured on base f71b292:
#
#   (1) STALL. Clamping Δm and Δr separately can turn the damped Newton ascent
#       direction into a descent direction; `_va_backtrack` then rejects every
#       trial and the iterate never moves (2000 iterations change nothing).
#       All-zero / all-one groups under a very large random-effect SD.
#   (2) CAP. 25/30 steps of at most |Δm| = 4 cannot reach a far optimum. Also an
#       all-zero group with a huge Σe^{η0} (log S ≳ 36): the warm start puts s far
#       below its optimum, Hrr rounds to 0 or is tiny, and the old det-vs-entries
#       test damped every step so r crept up by ~1e-2 per step.
#   (3) ROUNDOFF (Binomial). `s·η − n·softplus(η)` cancels when η ≫ 0 and s ≈ n;
#       the ~1e-11 evaluation noise exceeds the line-search tolerance and freezes
#       the solve at |∇F| ≈ 2.5e-6. Reached by a realistic aggregated-binomial
#       fit that reported `converged = true` (fit-level testset below).
#
# The fix keeps the fixed unroll, caps the step along its own direction, damps a
# Hessian only when it is not usably negative definite, keeps stepping while the
# squared Newton decrement λ² = gᵀ(−H)⁻¹g exceeds max(1e-14, the rounding of the
# group ELBO) (the inner functions return ρ = λ² / that bound), writes the
# Binomial kernel and the Poisson Hrr without cancellation, re-measures every
# group's ρ at θ̂ and ANDs ρ ≤ 1 into `converged` (with a warning). Each testset
# below except the last fails on base.

using DRModels
using Test
using ForwardDiff
using StableRNGs
using Logging
import Distributions
import SpecialFunctions

const DVI = DRModels

# |∇F| of the Poisson group ELBO in (m, r = log s) at the returned (m, s), computed
# here in closed form, independently of the package's own λ².
function _pois_inner_gnorm(A, S, τ, m, s)
    E = exp(m + s / 2) * S
    return hypot(A - E - τ * m, (1 / s - τ - E) * (s / 2))
end
# |∇F| by ForwardDiff of a quadrature-family group ELBO `obj(u)`, u = [m, log s].
_ad_inner_gnorm(obj, m, s) = hypot(ForwardDiff.gradient(obj, [m, log(s)])...)

const _GH_Z, _GH_W = DVI._gauss_hermite(12)
const _GH_LWπ = log.(_GH_W) .- 0.5 * log(π)

@testset "#965 VA inner stationarity" begin

    @testset "Poisson inner: stall and cap cells reach stationarity" begin
        # (A, log S, log σ_b): all-zero groups under σ_b = e⁴ ≈ 55. On base the
        # first keeps |∇F| = 2.5 for any number of iterations (STALL); the second
        # has |∇F| = 1.66 after 25 steps but is solvable (CAP).
        # The last three are the scout cells at θ̂ of measured fits: R2 (huge S) at
        # an ordinary σ_b, and the CAP / STALL groups of two converged = true fits.
        for (A, lS, lσ) in ((0.0, -37.0, 4.0), (0.0, -19.0, 4.0), (0.0, -38.0, 4.5),
                            (10.0, 60.0, 0.0), (0.0, log(4) + 37.599058437158035, 3.6710579140424255),
                            (0.0, -61.242185737691955, 3.8612174228408462))
            S = exp(lS); τ = exp(-2lσ)
            out = DVI._poisson_va_inner(A, S, τ)
            @test _pois_inner_gnorm(A, S, τ, out[1], out[2]) < 1e-6
            @test length(out) == 3 && out[3] <= 1
        end
    end

    @testset "Binomial inner: stall, cap and roundoff cells reach stationarity" begin
        # (η0, successes, trials, log σ_b). On base: STALL |∇F| = 8.9e-2, CAP
        # |∇F| = 0.60 after 30 steps, ROUNDOFF |∇F| = 5.4e-6 for any iterations.
        cells = (([-15.0], [0], [1], 4.5),
                 ([-15.0], [0], [10], 4.0),
                 (fill(20.0, 5), fill(1000, 5), fill(1000, 5), 2.75))
        for (η0, s, nt, lσ) in cells
            τ = exp(-2lσ)
            obj = u -> DVI._binom_va_group_elbo(u[1], u[2], η0, s, nt, τ, _GH_Z, _GH_LWπ)
            out = DVI._binom_va_inner(η0, s, nt, τ, _GH_Z, _GH_LWπ)
            @test _ad_inner_gnorm(obj, out[1], out[2]) < 1e-6
            @test length(out) == 3 && out[3] <= 1
        end
    end

    @testset "Binomial kernel: same value as s·η − n·softplus(η), no cancellation" begin
        sp(η) = η > 0 ? η + log1p(exp(-η)) : log1p(exp(η))
        rng = StableRNG(965)
        for _ in 1:200
            k = rand(rng, 1:6); η0 = 3 .* randn(rng, k); nt = rand(rng, 1:20, k)
            s = [rand(rng, 0:n) for n in nt]; b = 2 * randn(rng)
            old = sum(s[j] * (η0[j] + b) - nt[j] * sp(η0[j] + b) for j in 1:k)
            @test DVI._binom_group_kernel(b, η0, s, nt) ≈ old rtol = 1e-12 atol = 1e-10
        end
        # The cancelling case: all successes, η = 30. Exact value −n·log1p(e^{−η}).
        @test DVI._binom_group_kernel(0.0, [30.0], [1000], [1000]) ≈ -1000 * log1p(exp(-30.0)) rtol = 1e-12
    end

    @testset "the profiled gradient is the envelope gradient at a former stall" begin
        # Through a stationary inner point, the derivative ForwardDiff takes through
        # the inner iterations must equal the partial derivative of F at fixed
        # (m, r) (envelope theorem). On base the stalled point breaks this.
        A = 0.0
        function profiled(p)                           # p = (log S, log σ_b)
            S = exp(p[1]); τ = exp(-2p[2])
            out = DVI._poisson_va_inner(oftype(S, A), S, τ)
            m, s = out[1], out[2]
            return A * m - exp(m + s / 2) * S - 0.5 * (s * τ + m^2 * τ - 1 - log(s * τ))
        end
        p0 = [-37.0, 4.0]
        out = DVI._poisson_va_inner(A, exp(p0[1]), exp(-2p0[2]))
        m̂, ŝ = out[1], out[2]
        partial = ForwardDiff.gradient(p -> begin
            S = exp(p[1]); τ = exp(-2p[2])
            A * m̂ - exp(m̂ + ŝ / 2) * S - 0.5 * (ŝ * τ + m̂^2 * τ - 1 - log(ŝ * τ))
        end, p0)
        @test ForwardDiff.gradient(profiled, p0) ≈ partial rtol = 1e-6 atol = 1e-9
    end

    @testset "damped step ascends; non-finite Hessian gives no step" begin
        # A positive-definite H passed the old det-only guard, so the "Newton step"
        # −H⁻¹g went downhill.
        Δm, Δr = DVI._va_damped_step(1.0, 0.0, 1.0, 1.0, 1.0)
        @test Δm * 1.0 + Δr * 1.0 > 0
        @test DVI._va_damped_step(-1.0, 0.0, -2.0, 1.0, 1.0) == (1.0, 0.5)   # ND: plain Newton
        @test DVI._va_damped_step(NaN, 0.0, -1.0, 1.0, 1.0) == (0.0, 0.0)
        # A negative-definite H with a tiny Hrr (s far below its optimum) gets the
        # plain Newton step; the old det > eps·(1 + |entries|) test damped it to
        # Δr ≈ 0.04.
        Δm, Δr = DVI._va_damped_step(-12.0, 0.0, -5e-26, 56.6, 0.5)
        @test Δm ≈ 56.6 / 12 && Δr ≈ 0.5 / 5e-26
        # The capped step keeps the Newton direction (the clamp did not).
        cm, cr = DVI._va_cap_step(-19.27, 0.52)
        @test cm ≈ -4.0 && cr / cm ≈ 0.52 / -19.27
        @test DVI._va_cap_step(0.5, -0.25) == (0.5, -0.25)
    end

    @testset "an inner solve stopped short is reported, not hidden" begin
        # One step from the warm start cannot reach the optimum of a former CAP cell.
        out = DVI._poisson_va_inner(0.0, exp(-19.0), exp(-8.0); iters = 1, maxiters = 1)
        @test out[3] > 1
        @test DVI._va_newton_decrement2(1.0, 0.0, 1.0, 0.0, -1.0) == Inf   # H not ND
        @test DVI._va_newton_decrement2(2.0, 0.0, -4.0, 0.0, -1.0) ≈ 1.0     # gᵀ(−H)⁻¹g
        # The bound is 1e-14 unless the group ELBO is large enough that 32·eps·|F|
        # (twice the line search's tie tolerance) exceeds it.
        @test DVI._va_inner_bound(0.0) == 1e-14
        @test DVI._va_inner_bound(-1e4) ≈ 32 * eps() * (1 + 1e4)
    end

    @testset "fit-level report: converged ANDs the inner check, with a warning" begin
        @test DVI._va_inner_converged([0.0, 1e-6, 1.0], "route") === true
        ok = @test_logs (:warn, r"inner solve did not reach stationarity for 2 of 3 groups.*#965") DVI._va_inner_converged([0.0, 1e3, NaN], "route")
        @test ok === false
        # Quiet inside bootstrap / profile refits (same switch as the clamp warnings),
        # but still not converged.
        task_local_storage(:drm_quiet_boundary, true) do
            quiet = @test_logs DVI._va_inner_converged([1e3], "route")
            @test quiet === false
        end
    end

    @testset "Binomial (1 | g) fit: converged = true now means every inner is stationary" begin
        # Aggregated binomial, 1000 trials per row, high success rates. On base this
        # fit reports converged = true while group 6 (5000/5000 successes) is frozen
        # at |∇F| = 2.5e-6 by the kernel's cancellation (mechanism 3).
        rng = StableRNG(2)
        G, ng, t = 10, 5, 1000
        g = repeat(1:G, inner = ng); x = randn(rng, G * ng)
        b = 2.0 .* randn(rng, G)
        μ = DVI._logistic.(8.0 .+ 0.5 .* x .+ b[g])
        s = Float64.([rand(rng, Distributions.Binomial(t, p)) for p in μ]); fail = t .- s
        fit = drm(bf(@formula(cbind(s, fail) ~ x + (1 | g))), Binomial();
                  data = (; s, fail, x, g), marginal = :VA)
        θ̂ = fit.theta
        η0 = θ̂[1] .+ θ̂[2] .* x; τ = exp(-2θ̂[3])
        worst = 0.0
        for k in 1:G
            idx = findall(==(k), g); e = η0[idx]; sk = round.(Int, s[idx]); nk = fill(t, ng)
            obj = u -> DVI._binom_va_group_elbo(u[1], u[2], e, sk, nk, τ, _GH_Z, _GH_LWπ)
            out = DVI._binom_va_inner(e, sk, nk, τ, _GH_Z, _GH_LWπ)
            worst = max(worst, _ad_inner_gnorm(obj, out[1], out[2]))
        end
        @test fit.converged
        @test worst < 1e-6
        # The fit's own objective records the same per-group λ² it ANDs into converged.
        dec = fill(NaN, G)
        fit.nll(θ̂, dec)
        @test all(d -> d <= 1, dec)
    end

    @testset "Poisson (1 | g) fit: no converged = true at a non-stationary inner" begin
        # Four groups, one all-zero. On base the fit stops at θ̂ = [37.60, 3.67]
        # (σ_b ≈ 39) with converged = true, while group 1's inner (A = 0,
        # log S ≈ 39) is capped at |∇F| = 0.20; the profiled ELBO there is 7.5 nats
        # below the VA optimum near θ = [2.84, 1.43] (the default :LA fit is
        # [2.81, 1.45]).
        y = [0.0, 0.0, 0.0, 0.0, 12.0, 147.0, 1778.0]
        g = [1, 1, 1, 1, 2, 3, 4]
        fit = drm(bf(@formula(y ~ 1 + (1 | g))), Poisson(); data = (; y, g), marginal = :VA)
        θ̂ = fit.theta; τ = exp(-2θ̂[end])
        worst = maximum(1:4) do k
            idx = findall(==(k), g); A = sum(y[idx]); S = length(idx) * exp(θ̂[1])
            out = DVI._poisson_va_inner(A, S, τ)
            _pois_inner_gnorm(A, S, τ, out[1], out[2])
        end
        @test fit.converged
        @test worst < 1e-6
        @test θ̂ ≈ [2.8367, 1.4298] atol = 1e-3
        @test loglik(fit) ≈ -24.8805 atol = 1e-3
    end

    @testset "NB2 kernel: no cancellation, no false alarm at a large size" begin
        # The old kernel built each node from terms of size r·log r that cancel; at
        # NB2 size ≈ 430 its rounding noise in the group ELBO stalled the inner solve
        # at λ² ≈ 1e-11 (ρ ≈ 17 with the stationarity check), so this ordinary fit
        # was flagged. The b-dependent part is now y·η − (y + r)·log1p(μ/r) and the
        # b-free part a sum of log1p(k/r).
        for (y, r) in (([1, 3, 0, 2], 430.0), ([0, 5, 12], 3.6e8), ([2, 0], exp(40.0)))
            exact = sum(Float64(SpecialFunctions.loggamma(big(k) + big(r)) - SpecialFunctions.loggamma(big(r)) -
                                k * log(big(r))) for k in y)
            @test DVI._nb2_group_const(y, r) ≈ exact rtol = 1e-12 atol = 1e-14
        end
        η0 = [0.5, -1.0, 2.0]; yk = [1, 0, 7]
        for r in (0.3, 430.0, 1e9), b in (-3.0, 0.1, 4.0)
            μ = exp.(big.(η0) .+ b); R = big(r)
            exact = Float64(sum(yk[j] * (η0[j] + b) - (yk[j] + R) * log1p(μ[j] / R) for j in 1:3))
            @test DVI._nb2_group_kernel(b, η0, yk, r) ≈ exact rtol = 1e-12 atol = 1e-12
        end
        rng = StableRNG(7_000_000 + 10000 * 3 + 100 * 2 + 15 + 4)
        G, ng = 20, 4; n = G * ng
        g = repeat(1:G, inner = ng); x = randn(rng, n)
        bg = 1.5 .* randn(rng, G)
        η = -0.5 .+ 0.5 .* x .+ bg[g]
        y = Float64.([rand(rng, Distributions.NegativeBinomial(2.0, 2.0 / (2.0 + exp(e)))) for e in η])
        gidx, _ = DVI._group_index(g)
        fit = @test_logs min_level = Logging.Warn DVI._fit_nb2_ranef_va(DVI.NegBinomial2(), y,
                  hcat(ones(n), x), ones(n, 1), gidx, G, ["(Intercept)", "x"], ["(Intercept)"], :g, 1e-8)
        @test fit.converged
        dec = fill(NaN, G); fit.nll(fit.theta, dec)
        @test all(d -> d <= 1, dec)
    end
end
