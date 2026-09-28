# test_nb2_hurdle_nan.jl — regression test for #866: the hurdle NB2 fixed-effects
# nll (`_fit_negbin2_hu`, src/negbinomial.jl — shared by `NegBinomial2() + hu~1`
# and, via delegation, `TruncatedNegBinomial2() + hu~1`) returned NaN, not +Inf or
# finite, when the `sigma` (dispersion) coordinate sat at log σ ≈ -20/-30/-50:
# size r = exp(-2·ησ) ~ 1e17-1e43 there, so r + μ rounds to EXACTLY r in Float64
# and p = r/(r+μ) rounds to EXACTLY 1.0. `logpdf(NegativeBinomial(r, p), 0)` then
# comes out finite (it needs no log(1-p) term) while `logpdf(·, k)` for k > 0 is
# -Inf (it does need log(1-p) = log(0)). The hurdle's zero-truncation term,
# `_log1mexp(logpdf(d, 0))`, is then subtracted from that -Inf: -Inf - (-Inf) =
# NaN. Found by the likelihood sanity fuzzer (#866, draft PR on branch
# claude/ll-sanity-fuzzer, test/test_ll_sanity_fuzzer.jl). Fixed here by
# `_nb2_logpmf` (src/negbinomial.jl), a log1p-space NB2 log-pmf that never forms
# r + μ — the same cancellation class #846 fixed on the `(1|g)` AGHQ path.
using DRModels
using Test, Random
import Distributions

# Independent reference hurdle nll using Distributions' own NegativeBinomial
# logpdf exactly as `_fit_negbin2_hu` computed it BEFORE this fix. Used only to
# confirm the new `_nb2_logpmf` path reproduces the old numbers at ORDINARY
# parameter values (not the extreme cells this test is about, where the old
# path is NaN by construction).
function _ref_hurdle_nll(y::Vector{Float64}, x::Vector{Float64}, θ::Vector{Float64})
    n = length(y)
    βμ = θ[1:2]; ησ = θ[3]; ηh = θ[4]
    lπ = -log1p(exp(-ηh)); l1mπ = -log1p(exp(ηh))
    r = exp(-2 * ησ)
    s = 0.0
    for i in 1:n
        if y[i] == 0
            s -= lπ
        else
            μ = exp(βμ[1] + βμ[2] * x[i])
            p = r / (r + μ)
            d = Distributions.NegativeBinomial(r, p; check_args = false)
            lp0 = Distributions.logpdf(d, 0)
            l1mexp = lp0 < -log(2) ? log1p(-exp(lp0)) : log(-expm1(lp0))
            s -= l1mπ + Distributions.logpdf(d, round(Int, y[i])) - l1mexp
        end
    end
    return s
end

_sigma_idx(fit) = fit.blocks[findfirst(p -> p.first === :sigma, fit.blocks)].second

Random.seed!(866); n = 60; x = randn(n); θd = 3.0; πz = 0.35
rtnb(r, pp) = (while true; k = rand(Distributions.NegativeBinomial(r, pp)); k > 0 && return k; end)
μtrue = exp.(0.4 .+ 0.3 .* x)
y = Float64.([rand() < πz ? 0 : rtnb(θd, θd / (θd + μtrue[i])) for i in 1:n])
data = (; y, x)

@testset "NB2 hurdle: no NaN at extreme sigma (#866)" begin
    @testset "NegBinomial2 + hu~1" begin
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(hu ~ 1)), NegBinomial2();
                  data = data, se = false)
        θ0 = fit.theta
        idx = only(_sigma_idx(fit))

        # Ordinary parameter values: unchanged vs. the pre-fix logpdf path.
        @test fit.nll(θ0) ≈ _ref_hurdle_nll(y, x, θ0) atol = 1e-10

        for g in (-20.0, -30.0, -50.0)
            θ = copy(θ0); θ[idx] = g
            v = fit.nll(θ)
            @test !isnan(v)
            @test v != -Inf
            @test -v <= 1e-8     # discrete response: fitted logLik must be <= 0
        end
    end

    @testset "TruncatedNegBinomial2 + hu~1" begin
        # No `se` kwarg on this constructor's `drm` method (fixed-effects-only
        # route; matches test_hurdle.jl's own call and the fuzzer's).
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(hu ~ 1)), TruncatedNegBinomial2();
                  data = data)
        θ0 = fit.theta
        idx = only(_sigma_idx(fit))

        @test fit.nll(θ0) ≈ _ref_hurdle_nll(y, x, θ0) atol = 1e-10

        for g in (-20.0, -30.0, -50.0)
            θ = copy(θ0); θ[idx] = g
            v = fit.nll(θ)
            @test !isnan(v)
            @test v != -Inf
            @test -v <= 1e-8
        end
    end
end
