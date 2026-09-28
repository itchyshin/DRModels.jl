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
