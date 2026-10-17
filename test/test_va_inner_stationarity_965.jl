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
#   (2) CAP. The solve keeps moving but cannot get there in 25/30 steps: the
#       coordinate clamp leaves a direction along which F barely rises, so the line
#       search accepts only 2⁻⁴ to 2⁻⁹ of each step (the Poisson and Binomial cells);
#       elsewhere |Δr| ≤ 2 or Newton's own +1 step sets the pace (Gamma). Also a
#       group with a huge Σe^{η0}: the warm start puts r = log s near −log S, far
#       below its optimum. Under σ_b ≈ 39 (log S ≳ 37) r climbs only at the
#       |Δr| = 2 cap; at σ_b ≈ 1 (log S ≳ 40) Hrr is tiny, the old det-vs-entries
#       test damped every step, and r crept up by ~1e-2 per step.
#   (3) ROUNDOFF (Binomial). `s·η − n·softplus(η)` cancels when η ≫ 0 and s ≈ n;
#       the ~1e-11 evaluation noise exceeds the line-search tolerance and freezes
#       the solve short of stationarity (|∇F| = 5.4e-6 in the unit cell, 2.5e-6
#       for group 6 of a realistic aggregated-binomial fit that reported
#       `converged = true`; fit-level testset below).
#
# (Further mechanisms, found later, are pinned below: the old det-only damping test
# let a positive-definite H through (Beta); the NB2, Gamma and Beta kernels cancelled
# terms of size r·log r, α·log α or φ·log φ; the Gamma warm start sat far below its
# optimum, where Newton creeps +1 per step.)
#
# The fix keeps the fixed unroll, caps the step along its own direction, damps a
# Hessian only when it is not usably negative definite, takes full Newton steps
# within λ² ≤ 1e-8 of the maximum, and keeps stepping until the squared Newton
# decrement λ² = gᵀ(−H)⁻¹g is at most 1e-14. A group the solve cannot take that far
# is judged by ρ = λ² / max(1e-14, 32·eps·(1 + |F|)), which the inner functions
# return. The Binomial, NB2 and Gamma kernels, the Beta kernel above φ = 50 and the
# Poisson Hrr are written without cancellation, and every group's ρ is re-measured
# at θ̂ and ρ ≤ 1 ANDed into `converged` (with a warning).
#
# Every testset fails on base. These fail on an assertion about base's behaviour:
# the Poisson and Binomial inner cells; the Binomial kernel; the envelope gradient;
# the damped step; the inner cells of NB2, Gamma and Beta; the Binomial, Poisson,
# both Gamma, the Beta and the positive-ELBO NB2 fits. The rest fail on base because
# they call new code: the reporter, the stopped-short ρ with `maxiters` and its
# bound, the Newton zone, the kernel constants, the `nll(θ, ρ)` recording and the
# fitters' `va_inner` keyword; the NB2 size ≈ 430 fit is a no-false-alarm guard.

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

# Data for the Gamma end-to-end testset, as literals so the test does not depend on
# a sampler stream: 20 groups of 4 drawn from the model (β0 = 0.5, β1 = 8, shape 2,
# σ_b = 1, StableRNG(1)) plus group 21, two observations with EQUAL y = e^0.5 at
# x = ±4.5, a high-leverage group whose responses do not follow x.
const _GAMMA965_Y = [
    0.006634014162002215, 0.29188841667879223, 333.72974234699007,
    0.0008420943669918487, 2.7923279661918185e-7, 1.600881735405424, 34104.70902594438,
    645.2180645515004, 2.898162500721262, 13.066392314764828, 1.017796423885318e-5,
    2.5084014885619004, 5.888201134435547e-7, 0.17618092691864048, 8.99838669646414,
    24.152237047955445, 14.636206494982517, 5.5842578509061664e-5, 7823.306587610906,
    0.002078251200184794, 167.68225765737125, 258986.68590687535, 0.023082936320790014,
    0.002669944723382729, 139.6470415185482, 24.862163817896167, 102.97837394418167,
    7.795826012491152e-5, 0.009391351441623723, 0.0035721640149102766,
    13.165218866837689, 11.565158054785801, 0.17588760576688028, 22.033074438747857,
    1706.4393659369232, 237862.82934969713, 68.33908848583, 0.0015211362928358573,
    3474.193604166241, 1.2653787145019604, 1.063863456538067, 0.000142714302763192,
    66.21905825847, 1.6926749139184738, 0.00021870164639301723, 4351.267313644083,
    6.99924343085768e-7, 3.737334814543326e8, 1.0856497455701273,
    0.0021649566708228703, 55.29975440956056, 12.7938620406297, 42.82857464726425,
    11.741084282467543, 0.013379116815300518, 0.035762160708531214,
    0.42100927432386576, 436.973855807545, 0.1259237240309934, 0.09000184053009064,
    67.30961243107349, 3.998648533188118e6, 0.0018846862758439646, 0.8453187675564455,
    1.0706110455715134e-8, 20389.02463209108, 1477.9557106187506, 99.16103473844238,
    6.578054962265386e-5, 1305.4117538885598, 102.09796785406505, 139.92794806361448,
    146.9491898020529, 2.1688328296438604e-7, 0.35824097342151473,
    2.605716037556776e-11, 75441.48722620151, 200.11119828218, 0.001082083062783665,
    0.3089005331872578, 1.6487212707001282, 1.6487212707001282]
const _GAMMA965_X = [
    -0.5325200748641231, 0.098465514284785, 0.7528865221245234, -0.8435374038742546,
    -2.0138293556311067, -0.26980764119712713, 1.0322934723039892, 0.6755660991191906,
    0.08216082754396425, 0.24692664035915107, -1.5005291473189497, 0.32863200921212465,
    -1.8531987065432076, -0.33719082654695276, 0.03566251637914036,
    0.23214355889044092, 0.556345368618102, -1.0278210092684674, 1.206403573149015,
    -0.6205921994412764, 0.6123082058797865, 1.4913944293489885, -0.6757996689733272,
    -0.7512237101939829, 0.49395595649913915, 0.4136706876598097, 0.5566484200778432,
    -1.2580215560803685, -0.5406150525395599, -0.7281156252321607, 0.3213483657325198,
    0.6882831873751686, -0.3899398874160884, 0.05438833466615296, 0.7283022866605751,
    1.331489923091551, 0.4986114118380903, -0.8584115280138382, 0.9910360582065267,
    0.03813525217366319, -0.001413359889278404, -0.8526640041940637,
    0.6751427795073505, 0.31584286771756176, -0.9460270317435727, 1.1048619354568248,
    -1.7254776725212033, 2.6333449175920385, 0.049062337276717195, -0.5403822275366147,
    0.2896304349055892, 0.14525905655767543, 0.5908630515307562, 0.2874973018846024,
    -0.3571020878849498, -0.3574789623630043, 0.13695883374649123, 1.0419815412419215,
    0.052755040558498466, -0.2461920035269255, 0.8457263707411052, 1.88049002193508,
    -0.7759877870846973, 0.1443425453284156, -2.1996744272017246, 1.3443365695149092,
    0.9360605441917709, 0.597384740176566, -1.2363025017625762, 0.9808989479195825,
    0.6666812074800155, 0.9377943118424776, 0.5404062939314883, -1.8489132707132137,
    0.020810881369009222, -2.9322666947228786, 1.532931723220232, 0.799768011679008,
    -0.8791568057139619, -0.08685885565775965, -4.5, 4.5]

const _GH_Z, _GH_W = DVI._gauss_hermite(12)
const _GH_LWπ = log.(_GH_W) .- 0.5 * log(π)

@testset "#965 VA inner stationarity" begin

    @testset "Poisson inner: stall and cap cells reach stationarity" begin
        # (A, log S, log σ_b): all-zero groups under σ_b = e⁴ ≈ 55. On base the
        # first keeps |∇F| = 2.5 for any number of iterations (STALL); the second
        # has |∇F| = 1.66 after 25 steps but is solvable (CAP); the third, under
        # σ_b = e^4.5 ≈ 90, is a STALL at |∇F| = 0.62.
        # The last three: a synthetic huge-S group (A = 10, log S = 60, σ_b = 1)
        # whose r creeps ~7e-3 per step on base, and the CAP / STALL groups at θ̂
        # of two converged = true fits.
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

    @testset "reporter: a non-stationary group returns false, with a warning" begin
        @test DVI._va_inner_converged([0.0, 1e-6, 1.0], "route") === true
        ok = @test_logs (:warn, r"inner solve did not reach stationarity for 2 of 3 groups.*#965") DVI._va_inner_converged([0.0, 1e3, NaN], "route")
        @test ok === false
        # Quiet inside bootstrap refits (the switch `_without_boundary_warnings` sets,
        # as for the clamp warnings), but still not converged.
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
        # The fit's own objective records the per-group ρ (λ² over its bound) that the
        # fitter ANDs into converged.
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
        # The b-dependent part is −y·log1p(r/μ) − r·log1p(μ/r), a sum of non-positive
        # terms; the b-free Σ loggamma(y+r) − loggamma(r) is built without cancelling
        # its r·log r terms (exactly for y ≤ 1000, by the Stirling remainder above).
        # The old kernel's cancellation is what gave the Poisson-limit fits ELBOs of
        # +2577 and +3539 on base (test_vcov_guard.jl's fixture is one of them).
        lgb(x) = SpecialFunctions.loggamma(x)
        for (y, r) in (([1, 3, 0, 2], 430.0), ([0, 5, 12], 3.6e8), ([2, 0], exp(40.0)),
                       ([1001, 3], 3.6e8), ([3000], exp(40.0)), ([100_000], 2.5))
            R = big(r)
            exact = Float64(sum(lgb(big(k) + R) - lgb(R) for k in y))
            @test DVI._nb2_group_const(y, r) ≈ exact rtol = 1e-14 atol = 1e-12
        end
        η0 = [0.5, -1.0, 2.0]; yk = [1, 0, 1000]
        for r in (0.3, 1.0, 430.0, 1e9), b in (-3.0, 0.1, 4.0)
            μ = exp.(big.(η0) .+ b); R = big(r)
            exact = Float64(sum(-yk[j] * log1p(R / μ[j]) - R * log1p(μ[j] / R) for j in 1:3))
            @test DVI._nb2_group_kernel(b, η0, yk, r) ≈ exact rtol = 1e-14 atol = 1e-12
        end
        # No false alarm at NB2 size ≈ 430 on ordinary data. (Base has no check; with
        # the old kernel, drivers without the Newton zone left one or two groups here
        # at λ² = 2e-14 to 5.5e-13 and flagged the fit.)
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

    @testset "Gamma and Beta kernels: no cancellation at a large shape / precision" begin
        # At the largest shape and precision (α, φ = 1e6) the test also evaluates the
        # OLD (pre-#965) kernel in Float64 and checks that it misses the BigFloat
        # value by more than the tolerance the new one meets.
        lgb(x) = SpecialFunctions.loggamma(x)
        for x in (1e-3, 0.5, 14.9, 15.0, 1e3, 1e8)
            X = big(x)
            exact = Float64(lgb(X) - (X - 0.5) * log(X) + X - 0.5 * log(2 * big(π)))
            @test DVI._lgamma_stirling_rem(x) ≈ exact atol = 5e-14
        end
        gamma_old(b, η0, y, ly, α) = sum((α - 1) * ly[j] - α * y[j] / exp(η0[j] + b) - α * (η0[j] + b) +
                                         α * log(α) - lgb(α) for j in eachindex(η0))
        for α in (1e4, 1e6), b in (-0.3, 0.2)
            η0 = [0.1, 1.5]; y = exp.(η0 .+ b .+ [0.01, -0.02]); ly = log.(y)
            A = big(α)
            exact = Float64(sum((A - 1) * log(big(y[j])) - A * big(y[j]) / exp(big(η0[j]) + b) - A * (big(η0[j]) + b) +
                                A * log(A) - lgb(A) for j in 1:2))
            got = DVI._gamma_group_kernel(b, η0, y, ly, α) + DVI._gamma_group_const(ly, α)
            tol = 1e-13 * (1 + abs(exact))
            @test abs(got - exact) <= tol
            α == 1e6 && @test abs(gamma_old(b, η0, y, ly, α) - exact) > tol
        end
        beta_old(b, η0, ly, l1my, φ) = sum((μ = DVI._logistic(η0[j] + b); lgb(φ) - lgb(μ * φ) - lgb((1 - μ) * φ) +
                                            (μ * φ - 1) * ly[j] + ((1 - μ) * φ - 1) * l1my[j]) for j in eachindex(η0))
        for φ in (3.0, 1e4, 1e6), b in (-0.4, 0.3)
            η0 = [0.2, -1.0]; y = DVI._logistic.(η0 .+ b) .+ [0.003, -0.002]; ly = log.(y); l1my = log1p.(-y)
            P = big(φ)
            exact = Float64(sum((μ = 1 / (1 + exp(-(big(η0[j]) + b))); lgb(P) - lgb(μ * P) - lgb((1 - μ) * P) +
                                (μ * P - 1) * log(big(y[j])) + ((1 - μ) * P - 1) * log1p(-big(y[j]))) for j in 1:2))
            got = DVI._beta_group_kernel(b, η0, ly, l1my, φ) + DVI._beta_group_const(ly, l1my, φ)
            tol = 5e-12 * (1 + abs(exact))
            @test abs(got - exact) <= tol
            φ == 1e6 && @test abs(beta_old(b, η0, ly, l1my, φ) - exact) > tol
        end
    end

    @testset "Newton zone: full steps where the line search cannot resolve the gain" begin
        # A concave quadratic whose VALUE carries 1e-9 of deterministic rounding-like
        # noise while its gradient and Hessian are exact. From a start inside the zone
        # the gain on offer (~1e-12) is below the noise: 200 damped Newton steps that
        # all go through the line search freeze short of the optimum (asserted), while
        # the full Newton steps taken in the zone converge.
        noise(m, r) = 1e-9 * sin(1e7 * m + 3e6 * r)
        F = (m, r) -> -(m - 1)^2 - 2 * (r + 2)^2 + 0.5 * (m - 1) * (r + 2) + noise(m, r)
        gH = (m, r) -> (-2 * (m - 1) + 0.5 * (r + 2), -4 * (r + 2) + 0.5 * (m - 1), -2.0, 0.5, -4.0)
        m0, r0 = 1 + 1e-6, -2 + 1e-6
        @test DVI._va_newton_decrement2(gH(m0, r0)...) <= DVI._VA_NEWTON_ZONE
        mb, rb = m0, r0
        for _ in 1:200
            gm, gr, Hmm, Hmr, Hrr = gH(mb, rb)
            mb, rb = DVI._va_backtrack(F, mb, rb, DVI._va_damped_step(Hmm, Hmr, Hrr, gm, gr)...)
        end
        @test abs(mb - 1) > 1e-8                       # the line search alone stalls
        m, r, ρ = DVI._va_inner_solve(gH, F, m0, r0, 30, 200)
        @test abs(m - 1) < 1e-12 && abs(r + 2) < 1e-12
        @test ρ <= 1
    end

    @testset "Gamma (1 | g) fit at CV ≈ 1%: every inner solved" begin
        # Shape ≈ 1e4. On base the kernel's rounding noise stalls the inner solves:
        # converged = true with the worst group at λ² = 9.4e-13, |∇F| = 9.7e-7.
        rng = StableRNG(100 * 2 + 4)
        G, ng = 20, 5; n = G * ng
        g = repeat(1:G, inner = ng); x = randn(rng, n); bg = 0.5 .* randn(rng, G)
        η = 0.3 .+ 0.5 .* x .+ bg[g]
        y = [rand(rng, Distributions.Gamma(1e4, exp(e) / 1e4)) for e in η]
        gidx, _ = DVI._group_index(g)
        fit = @test_logs min_level = Logging.Warn DVI._fit_gamma_ranef_va(DVI.Gamma(), y, hcat(ones(n), x),
                  ones(n, 1), gidx, G, ["(Intercept)", "x"], ["(Intercept)"], :g, 1e-8)
        @test fit.converged
        θ̂ = fit.theta; τ = exp(-2θ̂[end]); α = exp(-2clamp(θ̂[3], -15, 15)); η0 = θ̂[1] .+ θ̂[2] .* x
        worst = maximum(1:G) do k
            idx = findall(==(k), g); e = η0[idx]; yk = y[idx]; lyk = log.(yk)
            out = DVI._gamma_va_inner(e, yk, lyk, α, τ, _GH_Z, _GH_LWπ)
            obj = u -> DVI._gamma_va_group_elbo(u[1], u[2], e, yk, lyk, α, τ, _GH_Z, _GH_LWπ)
            u = [out[1], log(out[2])]
            gv = ForwardDiff.gradient(obj, u)
            -(gv' * (ForwardDiff.hessian(obj, u) \ gv))          # λ² = gᵀ(−H)⁻¹g
        end
        @test worst <= 1e-14
        dec = fill(NaN, G); fit.nll(fit.theta, dec)
        @test all(d -> d <= 1, dec)
    end

    @testset "inner cells of NB2, Gamma and Beta reach stationarity" begin
        # Unit-level cells per family, each failing on base (measured there):
        #   NB2 all-zero group, σ_b = e^4.25: STALL, |∇F| = 0.17 for any iterations;
        #   Gamma warm start ~26 below its optimum (m₀ = 0, m* = 26.3): CAP,
        #     |∇F| = 0.15 after 30 steps (the Newton step on this kernel is +1 per
        #     step from below, to ~1e-5);
        #   Gamma, shape 0.01, σ_b = e^5: STALL, |∇F| = 1.47;
        #   Beta, a positive-definite H that the old det-only test let through: STALL,
        #     |∇F| = 2.9 (the maximum is at m ≈ 2.314, log s ≈ −1.774);
        #   Beta, top quadrature node past μ == 1.0 (old 1 − logistic(η)): F = −Inf
        #     cliff, |∇F| = 12.3.
        obj_nb = u -> DVI._nb2_va_group_elbo(u[1], u[2], [0.0], [0], 1.0, exp(-8.5), _GH_Z, _GH_LWπ)
        out = DVI._nb2_va_inner([0.0], [0], 1.0, exp(-8.5), _GH_Z, _GH_LWπ)
        @test _ad_inner_gnorm(obj_nb, out[1], out[2]) < 1e-6 && out[3] <= 1
        for (η0, y, α, τ) in (([-29.0, 29.0], [1.0, 1.0], 2.0, 1.0), ([0.0], [1.0], 0.01, exp(-10.0)))
            ly = log.(y)
            obj = u -> DVI._gamma_va_group_elbo(u[1], u[2], η0, y, ly, α, τ, _GH_Z, _GH_LWπ)
            out = DVI._gamma_va_inner(η0, y, ly, α, τ, _GH_Z, _GH_LWπ)
            @test _ad_inner_gnorm(obj, out[1], out[2]) < 1e-6 && out[3] <= 1
        end
        for (η0, y, φ, τ) in (([0.0, 0.0], [0.1, 0.999], 30.0, 0.01), ([20.0], [0.9], 100.0, 1.0))
            ly = log.(y); l1my = log1p.(-y)
            obj = u -> DVI._beta_va_group_elbo(u[1], u[2], η0, ly, l1my, φ, τ, _GH_Z, _GH_LWπ)
            out = DVI._beta_va_inner(η0, ly, l1my, φ, τ, sum(y) / length(y), _GH_Z, _GH_LWπ)
            @test _ad_inner_gnorm(obj, out[1], out[2]) < 1e-6 && out[3] <= 1
        end
        out = DVI._beta_va_inner([0.0, 0.0], log.([0.1, 0.999]), log1p.(-[0.1, 0.999]), 30.0, 0.01,
                                 0.5495, _GH_Z, _GH_LWπ)
        @test out[1] ≈ 2.3137446 atol = 1e-6
        @test log(out[2]) ≈ -1.7738484 atol = 1e-6
    end

    @testset "Gamma (1 | g) fit: a high-leverage group no longer derails the fit" begin
        # On base this fit reports converged = true at θ̂ = [2.084, 7.034, 0.457, 1.870]
        # with ELBO −230.69: group 21's inner starts ~31 below its optimum, creeps +1
        # per Newton step and is left at |∇F| = 3.2 after 30. The VA optimum (every
        # inner stationary) is θ = [1.948, 7.702, 0.379, 1.991], 5.45 nats better.
        y = _GAMMA965_Y; x = _GAMMA965_X
        g = vcat(repeat(1:20, inner = 4), [21, 21])
        fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gamma();
                  data = (; y, x, g), marginal = :VA)
        θ̂ = fit.theta; τ = exp(-2θ̂[end]); α = exp(-2clamp(θ̂[3], -15, 15)); η0 = θ̂[1] .+ θ̂[2] .* x
        worst = maximum(1:21) do k
            idx = findall(==(k), g); e = η0[idx]; yk = y[idx]; lyk = log.(yk)
            out = DVI._gamma_va_inner(e, yk, lyk, α, τ, _GH_Z, _GH_LWπ)
            obj = u -> DVI._gamma_va_group_elbo(u[1], u[2], e, yk, lyk, α, τ, _GH_Z, _GH_LWπ)
            _ad_inner_gnorm(obj, out[1], out[2])
        end
        @test fit.converged
        @test worst < 1e-6
        @test θ̂ ≈ [1.948032, 7.701650, 0.378781, 1.991434] atol = 1e-4
        @test loglik(fit) ≈ -225.239306 atol = 1e-4
    end

    @testset "Beta (1 | g) fit at precision ≈ 1e6: every inner solved" begin
        # Small μ, precision ≈ 1e6. On base this fit reports converged = true while
        # kernel rounding (terms of size φ·log φ) freezes three groups at |∇F| = 1.3e-5
        # to 4.1e-5 (λ² up to 6.7e-10). Here θ̂ itself is right to ~1e-9: the base
        # certificate was wrong, the estimate was not.
        rng = StableRNG(5)
        G, per = 10, 10; n = G * per
        g = repeat(1:G, inner = per); x = randn(rng, n); bg = randn(rng, G)
        μ = DVI._logistic.(-6.0 .+ 0.5 .* x .+ bg[g])
        y = [clamp(rand(rng, Distributions.Beta(m * 1e6, (1 - m) * 1e6)), 1e-300, prevfloat(1.0)) for m in μ]
        fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Beta();
                  data = (; y, x, g), marginal = :VA)
        θ̂ = fit.theta; τ = exp(-2θ̂[end]); φ = exp(-2clamp(θ̂[3], -15, 15)); η0 = θ̂[1] .+ θ̂[2] .* x
        worst = maximum(1:G) do k
            idx = findall(==(k), g); e = η0[idx]; ly = log.(y[idx]); l1my = log.(1 .- y[idx])
            out = DVI._beta_va_inner(e, ly, l1my, φ, τ, sum(y[idx]) / per, _GH_Z, _GH_LWπ)
            obj = u -> DVI._beta_va_group_elbo(u[1], u[2], e, ly, l1my, φ, τ, _GH_Z, _GH_LWπ)
            _ad_inner_gnorm(obj, out[1], out[2])
        end
        @test fit.converged
        @test worst < 1e-6
        dec = fill(NaN, G); fit.nll(θ̂, dec)
        @test all(d -> d <= 1, dec)
    end

    @testset "NB2 (1 | g) fit near the Poisson limit: no positive ELBO" begin
        # Large RE SD, NB2 size → ∞. On base the old kernel's cancellation of r·log r
        # terms gives this fit an ELBO of +2577.42, impossible for a count likelihood,
        # with converged = true. The fix: ELBO −68.774, every inner ρ ≤ 1.
        rng = StableRNG(7_031_127)
        G, ng = 20, 2; n = G * ng
        g = repeat(1:G, inner = ng); x = randn(rng, n); bg = 2.5 .* randn(rng, G)
        η = -0.5 .+ 0.5 .* x .+ bg[g]
        y = Float64.([rand(rng, Distributions.NegativeBinomial(2.0, 2.0 / (2.0 + exp(e)))) for e in η])
        gidx, _ = DVI._group_index(g)
        fit = DVI._fit_nb2_ranef_va(DVI.NegBinomial2(), y, hcat(ones(n), x), ones(n, 1), gidx, G,
                                    ["(Intercept)", "x"], ["(Intercept)"], :g, 1e-8)
        @test loglik(fit) < 0
        @test loglik(fit) ≈ -68.7741 atol = 1e-3
        @test fit.converged
        dec = fill(NaN, G); fit.nll(fit.theta, dec)
        @test all(d -> d <= 1, dec)
    end

    @testset "fitters: a forced non-stationary inner gives converged = false, with a warning" begin
        # `va_inner = (iters = 1, maxiters = 1)` stops every inner solve after one
        # Newton step. Each of the five fitters must then report converged = false and
        # warn, although its outer optimiser stops (g_tol = 1e-3); with the default
        # inner the same data converge.
        rng = StableRNG(965)
        G, ng = 6, 4; n = G * ng
        g = repeat(1:G, inner = ng); x = randn(rng, n); b = randn(rng, G)
        η = 0.3 .+ 0.5 .* x .+ b[g]; gidx, _ = DVI._group_index(g); X = hcat(ones(n), x)
        nm = ["(Intercept)", "x"]; Xs = ones(n, 1); nms = ["(Intercept)"]
        yp = Float64.([rand(rng, Distributions.Poisson(exp(e))) for e in η])
        sb = Float64.([rand(rng, Distributions.Binomial(6, DVI._logistic(e))) for e in η])
        yn = Float64.([rand(rng, Distributions.NegativeBinomial(3.0, 3.0 / (3.0 + exp(e)))) for e in η])
        yg = [rand(rng, Distributions.Gamma(4.0, exp(e) / 4.0)) for e in η]
        yb = [rand(rng, Distributions.Beta(DVI._logistic(e) * 10, (1 - DVI._logistic(e)) * 10)) for e in η]
        fitters = (
            kw -> DVI._fit_poisson_ranef_va(DVI.Poisson(), yp, X, gidx, G, nm, :g, 1e-3; kw...),
            kw -> DVI._fit_binomial_ranef_va(DVI.Binomial(), sb, fill(6.0, n), X, gidx, G, nm, :g, 1e-3; kw...),
            kw -> DVI._fit_nb2_ranef_va(DVI.NegBinomial2(), yn, X, Xs, gidx, G, nm, nms, :g, 1e-3; kw...),
            kw -> DVI._fit_gamma_ranef_va(DVI.Gamma(), yg, X, Xs, gidx, G, nm, nms, :g, 1e-3; kw...),
            kw -> DVI._fit_beta_ranef_va(DVI.Beta(), yb, X, Xs, gidx, G, nm, nms, :g, 1e-3; kw...))
        one_step = (va_inner = (iters = 1, maxiters = 1),)
        for f in fitters
            @test f((;)).converged
            fit = @test_logs (:warn, r"did not reach stationarity.*#965") match_mode = :any f(one_step)
            @test fit.converged === false
            @test fit.loglik ≈ -fit.nll(fit.theta)
            dec = fill(NaN, G); fit.nll(fit.theta, dec)
            @test maximum(dec) > 1
        end
        # Quiet inside bootstrap refits, and still not converged.
        quiet = @test_logs min_level = Logging.Warn DVI._without_boundary_warnings(() -> fitters[1](one_step))
        @test quiet.converged === false
    end
end
