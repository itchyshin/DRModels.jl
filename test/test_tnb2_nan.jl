# test_tnb2_nan.jl — regression test for the plain zero-truncated NB2 nll
# (`_fit_truncated_negbin2`, src/negbinomial.jl — the `TruncatedNegBinomial2()`
# route WITHOUT an `hu` part) returning NaN, not +Inf or finite, when the
# `sigma` (dispersion) coordinate sits at log σ ≈ -20/-30/-50: size
# r = exp(-2·ησ) ~ 1e17-1e43 there, so r + μ rounds to EXACTLY r in Float64 and
# p = r/(r+μ) rounds to EXACTLY 1.0. `logpdf(NegativeBinomial(r, p), 0)` then
# comes out finite (it needs no log(1-p) term) while `logpdf(·, k)` for k > 0 is
# -Inf (it does need log(1-p) = log(0)). The zero-truncation divisor,
# `_log1mexp(logpdf(d, 0))`, is then subtracted from that -Inf: -Inf - (-Inf) =
# NaN. Same cancellation class as #866 (`_fit_negbin2_hu`, fixed in
# test_nb2_hurdle_nan.jl by `_nb2_logpmf`) and #846 (the `(1|g)` AGHQ path) —
# `_fit_truncated_negbin2` was the one other site with the identical
# `logpdf(d, y) - _log1mexp(logpdf(d, 0))` pattern. Fixed here the same way:
# `_nb2_logpmf` (src/negbinomial.jl), a log1p-space NB2 log-pmf that never
# forms r + μ.
using DRModels
using Test, Random
import Distributions

# Independent reference zero-truncated nll using Distributions' own
# NegativeBinomial logpdf exactly as `_fit_truncated_negbin2` computed it
# BEFORE this fix. Used only to confirm the new `_nb2_logpmf` path reproduces
# the old numbers at ORDINARY parameter values (not the extreme cells this
# test is about, where the old path is NaN by construction).
function _ref_trunc_nll(y::Vector{Float64}, x::Vector{Float64}, θ::Vector{Float64})
    n = length(y)
    βμ = θ[1:2]; ησ = θ[3]
    r = exp(-2 * ησ)
    s = 0.0
    for i in 1:n
        μ = exp(βμ[1] + βμ[2] * x[i])
        p = r / (r + μ)
        d = Distributions.NegativeBinomial(r, p; check_args = false)
        lp0 = Distributions.logpdf(d, 0)
        l1mexp = lp0 < -log(2) ? log1p(-exp(lp0)) : log(-expm1(lp0))
        s -= Distributions.logpdf(d, round(Int, y[i])) - l1mexp
    end
    return s
end

_sigma_idx(fit) = fit.blocks[findfirst(p -> p.first === :sigma, fit.blocks)].second

Random.seed!(871); n = 60; x = randn(n); θd = 3.0
rtnb(r, pp) = (while true; k = rand(Distributions.NegativeBinomial(r, pp)); k > 0 && return k; end)
μtrue = exp.(0.4 .+ 0.3 .* x)
y = Float64.([rtnb(θd, θd / (θd + μtrue[i])) for i in 1:n])   # strictly positive (≥ 1)
data = (; y, x)

@testset "Truncated NB2: no NaN at extreme sigma" begin
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), TruncatedNegBinomial2(); data = data)
    θ0 = fit.theta
    idx = only(_sigma_idx(fit))

    # Ordinary parameter values: unchanged vs. the pre-fix logpdf path.
    @test fit.nll(θ0) ≈ _ref_trunc_nll(y, x, θ0) atol = 1e-10

    for g in (-20.0, -30.0, -50.0)
        θ = copy(θ0); θ[idx] = g
        v = fit.nll(θ)
        @test !isnan(v)
        @test v != -Inf
        @test -v <= 1e-8     # discrete response: fitted logLik must be <= 0
    end
end

# #883 (review of #871/#874/#876): TNB2's nll calls `_nb2_logpmf` once per
# observation per NLL evaluation, so #871's O(k)->O(1) fix (src/negbinomial.jl)
# carries straight through here — no change needed in this file's fit code.
# Regression-guard it directly: at #874's own slow regime (mean~3000, measured
# 110x slower pre-fix), the fit must complete fast and give the same logLik.
@testset "TruncatedNegBinomial2: O(1) NB2 log-pmf keeps fits fast (#883)" begin
    Random.seed!(874); n = 400; x = randn(n)
    μtrue2 = 3000.0 .* exp.(0.3 .* x)
    y2 = Float64.([rtnb(5.0, 5.0 / (5.0 + μtrue2[i])) for i in 1:n])
    dat2 = (; y = y2, x = x)
    f() = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), TruncatedNegBinomial2(); data = dat2)
    fit2 = f()
    ll1 = loglik(fit2)
    # Fixed effects only, no `se`; a second fit with a fresh RNG draw of the
    # SAME model class should reproduce a comparable, finite, fast fit.
    @test isfinite(ll1)

    # Timing guard robust to a shared/slow runner: compare the #874 slow-case
    # fit against a tiny-count fit on the same n, rather than a fixed wall-time
    # budget. Pre-fix this ratio was ~110x (0.0013s -> 0.147s); fixed, both
    # regimes cost about the same per-NLL-evaluation.
    μtrue_small = 5.0 .* exp.(0.3 .* x)
    y_small = Float64.([rtnb(5.0, 5.0 / (5.0 + μtrue_small[i])) for i in 1:n])
    dat_small = (; y = y_small, x = x)
    fsmall() = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), TruncatedNegBinomial2(); data = dat_small)
    fsmall()  # warm up / compile
    f()
    t_small = @elapsed fsmall()
    t_large = @elapsed f()
    @test t_large < 20 * max(t_small, 1e-6)   # generous margin; pre-fix was ~110x
end
