# test_cumlogit_nan.jl — regression test for a NaN nll in the CumulativeLogit
# (ordinal) `(1 | id)` random-intercept route (src/cumulative.jl), found by the
# likelihood-sanity fuzzer (draft PR #866, test/test_ll_sanity_fuzzer.jl,
# `sweep_route("CumulativeLogit (1|id)", ...)`). Root causes (see cumulative.jl
# for the fix): (1) the interior-category probability
# `logistic(cuts[k]-η) - logistic(cuts[k-1]-η)` was computed as a raw
# difference of two independently-rounded probabilities, so near/at cutpoint
# collapse it can go to exact 0 (fine, -Inf) but the group's 32-node
# Gauss-Hermite quadrature accumulator did `mx = maximum(terms)` then
# `terms .- mx`: when EVERY node's log-likelihood for a group is exactly -Inf
# (a deterministic cutpoint- or fixed-effect-driven "impossible category" that
# does not depend on the per-node random effect draw), `mx == -Inf` and
# `-Inf - (-Inf) == NaN`, poisoning the whole nll. This file reproduces the
# EXACT dataset/formula/seed the fuzzer used for that route (n=48, G=6, m=8,
# `y ~ x + (1|id)`, CumulativeLogit()) — since the fuzzer file itself must not
# be edited — and pins down the specific (coordinate, grid value) cells the
# fuzzer flagged as OPEN/NaN on origin/main.
using DRModels
using Test, Random

const _ordinalize_nan(η; cuts = (-0.5, 0.6)) = η <= cuts[1] ? 1.0 : (η <= cuts[2] ? 2.0 : 3.0)

# Same grid as the fuzzer (test/test_ll_sanity_fuzzer.jl `_GRID`).
const _GRID_NAN = (-50.0, -30.0, -20.0, -15.0, -5.0, 0.0, 3.0, 8.0, 20.0, 50.0)
const _ALLEXTREME_NAN = (-30.0, -5.0, 8.0)

# `(coordinate index, grid value)` cells the fuzzer found NaN on origin/main
# for "CumulativeLogit (1|id)" (idx == 0 marks an all-coordinates-set-to-`v`
# constant vector). θ = [β(x), δ1(cuts[1]), δ2(cuts[2] increment), logσb].
const _FAILING_CELLS = [
    (1, -50.0),
    (3, -50.0), (3, 8.0), (3, 20.0), (3, 50.0),
    (4, 8.0), (4, 20.0), (4, 50.0),
    (0, -30.0), (0, 8.0),
]

_safe_nll(nllfun, θ) = try
    nllfun(θ)
catch
    NaN
end

# The TRUE marginal nll at the origin/main θ̂ below: an exact per-group QuadGK
# integral (rtol 1e-12), 21.300119400926. This file originally pinned the old
# 32-node prior-scale GHQ value there, 21.304898931486296 (4.8e-3 nat off the
# exact integral); the routes now use per-group adaptive GHQ
# (test/test_cumlogit_aghq.jl), which reproduces the exact value to ~1e-11.
const _CUMLOGIT_NAN_BASELINE_NLL = 21.300119400926

@testset "CumulativeLogit (1|id) — no NaN nll at extreme θ (#866 fuzzer finding)" begin
    Random.seed!(24); G = 6; m = 8; n = G * m
    id = repeat(1:G, inner = m); x = randn(n)
    bg = 0.5 .* randn(G)
    η = 0.5 .* x .+ bg[id] .+ 0.2 .* randn(n)
    y = _ordinalize_nan.(η)
    fit = drm(bf(@formula(y ~ x + (1 | id))), CumulativeLogit(); data = (; y, x, id), se = false)
    θ0 = fit.theta
    @test length(θ0) == 4   # β(x), δ1, δ2, logσb — pins the θ-index comments above

    # The θ̂ the SAME fit call converged to on origin/main before the fix
    # (recorded once, see the PR body for the reproduction script/output).
    # Fixing the NaN removes a spurious "cliff" that had been silently
    # stopping LBFGS short (a NaN/-Inf function value looks like a failed
    # line-search step), so a FRESH fit on the fixed code legitimately
    # converges further, to a materially different (lower-nll) θ̂ — that is
    # the fix working as intended, not a regression. The invariant this
    # checks is narrower and more precise: the nll FUNCTION VALUE at this
    # fixed, ordinary (non-adversarial) θ point must not move.
    const_θ_baseline = [3.855077419302269, -0.8384295206269546,
                        1.8967583900573493, 1.214621917300004]

    @testset "ordinary θ (baseline θ̂) unchanged" begin
        v0 = fit.nll(const_θ_baseline)
        @test isfinite(v0)
        @test -v0 <= 1e-8                       # discrete data: logLik ≤ 0
        # The NaN fix only touches extreme-θ cancellation paths, so at this
        # ordinary θ the nll must equal the exact marginal (integration
        # tolerance 1e-6; see `_CUMLOGIT_NAN_BASELINE_NLL`).
        @test isapprox(v0, _CUMLOGIT_NAN_BASELINE_NLL; atol = 1e-6)
    end

    @testset "previously-NaN cells are now finite-or-+Inf, discrete logLik ≤ 0" begin
        p = length(θ0)
        for (i, g) in _FAILING_CELLS
            θ = i == 0 ? fill(g, p) : (θc = copy(θ0); θc[i] = g; θc)
            v = _safe_nll(fit.nll, θ)
            @test !isnan(v)
            @test v != -Inf
            @test -v <= 1e-8
        end
    end

    @testset "full grid sweep stays NaN-free (no new violation introduced)" begin
        p = length(θ0)
        for i in 1:p, g in _GRID_NAN
            θ = copy(θ0); θ[i] = g
            v = _safe_nll(fit.nll, θ)
            @test !isnan(v)
        end
        for g in _ALLEXTREME_NAN
            θ = fill(g, p)
            v = _safe_nll(fit.nll, θ)
            @test !isnan(v)
        end
    end
end
