# test_offset_no_silent_route_938.jl: #938 sibling-route regression guard.
#
# #938 was a SILENT wrong model: `offset(z)` reached a design matrix and got a
# free coefficient. #1211 applies the offset (coefficient fixed at 1) on the
# fixed-effects mean of every univariate family and on a few `(1 | g)` routes,
# and refuses it by name everywhere else. test_offset_fixed_coef_938.jl covers
# the fixed-effects routes; this file pins the invariant on the sibling routes
# (random effects per family and marginal, `zi`/`hu`, bivariate and mixed-family
# fits): every route either throws or applies the offset exactly. A route that
# fits must (a) have no `offset(...)` coefficient and (b) move the intercept by
# exactly -c for a constant offset c with the log-likelihood unchanged, so an
# offset that is stripped from the formula but never applied is caught too.

using DRModels
using Test
using StableRNGs
import Distributions

const _NSR_C = log(2.0)
_nsr_logistic(x) = 1 / (1 + exp(-x))
_nsr_names(fit) = reduce(vcat, [collect(String, v) for (_, v) in fit.coefnames])

function _nsr_data(tag::Symbol; n = 600, seed = 9380)
    rng = StableRNG(seed)
    x = randn(rng, n)
    g = repeat(string.("g", 1:30), outer = cld(n, 30))[1:n]
    h = repeat(string.("h", 1:7), outer = cld(n, 7))[1:n]
    u = Dict(l => 0.3 * randn(rng) for l in unique(g))
    η = 0.3 .+ 0.5 .* x .+ [u[l] for l in g]
    c = fill(_NSR_C, n)
    y = if tag in (:nb2, :poisson)
        [Float64(rand(rng, Distributions.NegativeBinomial(4.0, 4.0 / (4.0 + exp(e))))) for e in η]
    elseif tag in (:tnb2, :tpois)
        [Float64(1 + rand(rng, Distributions.Poisson(exp(e)))) for e in η]
    elseif tag === :gamma
        [rand(rng, Distributions.Gamma(6.0, exp(e) / 6.0)) for e in η]
    elseif tag === :lognormal
        exp.(η .+ 0.4 .* randn(rng, n))
    elseif tag in (:gaussian, :student, :skewnormal)
        η .+ 0.5 .* randn(rng, n)
    elseif tag in (:beta, :zoib)
        [rand(rng, Distributions.Beta(20 * _nsr_logistic(e), 20 * (1 - _nsr_logistic(e)))) for e in η]
    elseif tag in (:binomial, :betabinomial)
        s = [Float64(rand(rng, Distributions.Binomial(12, _nsr_logistic(e)))) for e in η]
        return (; s = s, f = 12 .- s, x, c, g, h)
    elseif tag === :tweedie
        [rand(rng) < 0.2 ? 0.0 : rand(rng, Distributions.Gamma(2.0, exp(e) / 2.0)) for e in η]
    elseif tag === :cumulative
        [Float64(1 + (e + randn(rng) > 0) + (e + randn(rng) > 1)) for e in η]
    else
        error("unknown family $tag")
    end
    return (; y, x, c, g, h)
end

_nsr_lhs(tag) = tag in (:binomial, :betabinomial) ? "cbind(s, f)" : "y"

# :threw, :applied, or a string naming the silent failure.
function _nsr_outcome(tag, fam, rhs0, rhsc; extra = (), kw...)
    mk(r) = bf(eval(Meta.parse("@formula($(_nsr_lhs(tag)) ~ $r)")), extra...)
    d = _nsr_data(tag)
    fc = try
        drm(mk(rhsc), fam; data = d, kw...)
    catch e
        e isa Exception || rethrow()
        return :threw
    end
    any(n -> occursin("offset", n), _nsr_names(fc)) && return "free offset coefficient"
    f0 = drm(mk(rhs0), fam; data = d, kw...)
    abs(loglik(fc) - loglik(f0)) < 1e-4 || return "logLik moved by a constant offset"
    if tag === :cumulative
        all(abs.(fc.scales[:ordinal_cuts] .- f0.scales[:ordinal_cuts] .- _NSR_C) .< 1e-3) ||
            return "cutpoints did not absorb the offset"
    else
        abs(coef(fc, :mu)[1] - coef(f0, :mu)[1] + _NSR_C) < 1e-3 ||
            return "intercept did not move by -c (offset dropped?)"
    end
    return :applied
end

const _NSR_FAMILIES = [
    (:nb2, NegBinomial2()), (:poisson, Poisson()), (:tnb2, TruncatedNegBinomial2()),
    (:tpois, TruncatedPoisson()), (:gamma, DRModels.Gamma()), (:lognormal, DRModels.LogNormal()),
    (:gaussian, Gaussian()), (:student, Student()), (:skewnormal, SkewNormal()),
    (:beta, DRModels.Beta()), (:zoib, ZeroOneBeta()), (:binomial, DRModels.Binomial()),
    (:betabinomial, DRModels.BetaBinomial()), (:tweedie, Tweedie()), (:cumulative, CumulativeLogit()),
]

const _NSR_ROUTES = [
    ("(1|g)",           "1 + x + (1 | g)",           "1 + x + offset(c) + (1 | g)",           (;)),
    ("(1|g) :Laplace",  "1 + x + (1 | g)",           "1 + x + offset(c) + (1 | g)",           (; marginal = :Laplace)),
    ("(1|g) :VA",       "1 + x + (1 | g)",           "1 + x + offset(c) + (1 | g)",           (; marginal = :VA)),
    ("(1+x|g)",         "1 + x + (1 + x | g)",       "1 + x + offset(c) + (1 + x | g)",       (;)),
    ("(1|g) + (1|h)",   "1 + x + (1 | g) + (1 | h)", "1 + x + offset(c) + (1 | g) + (1 | h)", (;)),
]

# Routes #1211 documents as applying the offset; they must not regress to a refusal.
const _NSR_MUST_APPLY = Set([(:nb2, "(1|g)"), (:nb2, "(1|g) :Laplace"), (:poisson, "(1|g)"),
                             (:poisson, "(1|g) :Laplace"), (:gamma, "(1|g) :Laplace"),
                             (:beta, "(1|g) :Laplace")])

@testset "#938 no route estimates or drops offset(): random-effect routes" begin
    for (tag, fam) in _NSR_FAMILIES, (rname, r0, rc, kw) in _NSR_ROUTES
        @testset "$tag $rname" begin
            out = _nsr_outcome(tag, fam, r0, rc; kw...)
            @test out === :applied || out === :threw
            (tag, rname) in _NSR_MUST_APPLY && @test out === :applied
        end
    end
end

@testset "#938 no route estimates or drops offset(): zi / hu" begin
    for (tag, fam) in [(:nb2, NegBinomial2()), (:poisson, Poisson())], p in (:zi, :hu)
        ex = (eval(Meta.parse("@formula($p ~ 1)")),)
        out = _nsr_outcome(tag, fam, "1 + x", "1 + x + offset(c)"; extra = ex)
        @test out === :applied || out === :threw
        tag === :nb2 && @test out === :applied
        out_re = _nsr_outcome(tag, fam, "1 + x + (1 | g)", "1 + x + offset(c) + (1 | g)"; extra = ex)
        @test out_re === :applied || out_re === :threw
    end
end

@testset "#938 bivariate and mixed-family fits refuse offset() in mu1" begin
    rng = StableRNG(9383); n = 300
    x = randn(rng, n); c = fill(_NSR_C, n)
    y1 = 1.0 .+ 0.4 .* x .+ 0.4 .* randn(rng, n); y2 = -0.4 .+ 0.3 .* x .+ 0.4 .* randn(rng, n)
    dat = (; y1, y2, y1p = exp.(y1), y2p = exp.(y2), x, c)
    biv(r1, r2) = bf(mu1 = eval(Meta.parse("@formula($r1 ~ 1 + x + offset(c))")),
                     mu2 = eval(Meta.parse("@formula($r2 ~ 1 + x)")),
                     sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                     rho12 = @formula(rho12 ~ 1))
    @test_throws ArgumentError drm(biv(:y1, :y2), Gaussian(); data = dat)
    @test_throws ArgumentError drm(biv(:y1, :y2), Student(); data = dat)
    @test_throws ArgumentError drm(biv(:y1p, :y2p), DRModels.LogNormal(); data = dat)
    @test_throws ArgumentError drm(bf(mu1 = @formula(y1 ~ 1 + x + offset(c)), mu2 = @formula(y2 ~ 1 + x)),
                                   (Gaussian(), Gaussian()); data = dat)
    @test_throws ArgumentError drm(bf(mu1 = @formula(y1 ~ 1 + x + offset(c)), mu2 = @formula(y2p ~ 1 + x)),
                                   (Gaussian(), DRModels.LogNormal()); data = dat)
end
