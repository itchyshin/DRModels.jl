# Hurdle (`hu`) modifier for count families: a two-part model. A logit "hurdle"
# decides zero vs positive (π = P(y=0)); positive counts follow the ZERO-TRUNCATED
# count distribution. Unlike `zi`, all zeros are structural. Mirrors drmTMB's `hu`.
using DRModels
using Test, Random
import Distributions

_logis(η) = 1 / (1 + exp(-η))
rtpois(λ) = (while true; k = rand(Distributions.Poisson(λ)); k > 0 && return k; end)
rtnb(r, p) = (while true; k = rand(Distributions.NegativeBinomial(r, p)); k > 0 && return k; end)

@testset "Hurdle Poisson: y ~ x, hu ~ 1 — recovery" begin
    Random.seed!(20260620)
    n = 4000; x = randn(n)
    βμ = [0.6, 0.4]; πz = _logis(-0.4)                  # π ≈ 0.40 structural zeros
    λ = exp.(βμ[1] .+ βμ[2] .* x)
    y = Float64.([rand() < πz ? 0 : rtpois(λ[i]) for i in 1:n])

    fit = drm(bf(@formula(y ~ x), @formula(hu ~ 1)), Poisson(); data = (; y, x))

    @test coef(fit, :mu)[1] ≈ βμ[1] atol = 0.08          # log-λ intercept (positive part)
    @test coef(fit, :mu)[2] ≈ βμ[2] atol = 0.08          # log-λ slope
    @test _logis(coef(fit, :hu)[1]) ≈ πz atol = 0.05     # hurdle (zero) probability
    @test isfinite(loglik(fit))
end

# Hurdle Poisson correctness pin (issue #726). The hurdle Poisson is accepted here
# but (as of drmTMB 0.7.0) refused by drmTMB, so it needs its own independent anchor:
#   (1) the log-likelihood equals a hand computation (zero part Bernoulli(hu);
#       positive part zero-truncated Poisson) at arbitrary parameter values;
#   (2) the fit equals an R reference: logistic `glm(y == 0 ~ w)` for the zero part
#       and `VGAM::vglm(y ~ x, pospoisson)` for the positive part -- exactly the
#       decomposition `pscl::hurdle(dist = "poisson", zero.dist = "binomial")` uses.
#       Constants are GENERATED outputs (R 4.x, VGAM, set.seed(7262)), not drmTMB source.
@testset "Hurdle Poisson: log-likelihood and fit pinned to independent computations" begin
    y = Float64.([0, 1, 3, 0, 1, 0, 3, 0, 0, 2, 0, 3, 4, 0, 4, 7, 0, 0, 0, 0, 0, 1, 6, 0, 0, 0, 0, 0, 1, 0, 2, 3, 2, 2, 0, 2, 0, 1, 0, 1, 0, 2, 1, 3, 3, 0, 5, 1, 0, 1, 1, 0, 5, 0, 3, 0, 3, 3, 0, 0, 0, 2, 0, 5, 0, 0, 3, 0, 4, 3, 2, 0, 0, 0, 1, 0, 0, 2, 4, 0])
    x = [0.093, -1.593, 2.332, 0.18, -2.515, 0.532, -0.098, -1.276, 1.172, -0.325, -0.562, -1.153, -0.578, -0.747, 0.152, 0.714, 1.328, -1.004, 0.866, 0.702, -0.487, 0.497, 1.432, -0.51, -0.528, -1.254, -0.302, -0.041, 0.566, -0.791, -0.78, 0.211, 0.682, 0.471, 0.276, 0.889, 0.377, 0.532, 0.697, 0.72, -1.778, -0.207, -0.506, -0.424, -1.042, 0.138, 0.578, -0.328, 0.375, 0.487, -0.371, 0.731, -0.019, 0.999, 0.423, -1.13, -0.311, 0.208, 0.623, 0.657, -0.355, -0.829, 1.072, 0.106, 0.227, -1.101, -1.207, -0.081, 1.924, -0.483, -0.468, -1.379, 0.676, -1.296, -1.075, 0.09, 0.027, -0.225, -0.152, -0.771]
    w = [1.419, 1.539, -0.011, -1.59, -0.287, 0.604, -0.752, 1.101, 1.052, 0.905, 1.031, -2.129, 0.101, 0.78, 0.206, 0.446, -0.483, 0.475, 0.827, -0.047, -2.175, -1.492, 0.514, -0.807, 0.392, 0.347, 0.522, 0.555, 0.399, -0.378, 1.576, 1.319, -2.051, 0.469, 0.875, -0.823, -0.236, 0.608, 0.908, 1.991, -0.02, 0.032, 0.832, -0.93, -0.685, -0.25, 0.587, 0.433, -0.004, 1.198, -1.407, 1.465, 0.227, 0.2, -0.088, 0.588, 0.408, 0.584, 0.536, 0.511, 0.03, -0.921, -0.732, 0.791, -0.167, -0.139, 0.434, -0.28, 0.681, -1.012, 0.795, -0.101, 2.34, 0.383, 1.67, 0.449, 0.038, -1.527, 0.65, 0.005]
    data = (; y, x, w)
    hand(θ) = sum(eachindex(y)) do i
        π0 = _logis(θ[3] + θ[4] * w[i]); λ = exp(θ[1] + θ[2] * x[i])
        y[i] == 0 ? log(π0) :
            log1p(-π0) + Distributions.logpdf(Distributions.Poisson(λ), Int(y[i])) - log(-expm1(-λ))
    end
    fit = drm(bf(@formula(y ~ x), @formula(hu ~ w)), Poisson(); data = data)
    @test fit.converged
    # (1) likelihood = hand computation, at the optimum and at off-optimum points
    @test loglik(fit) ≈ hand(fit.theta) atol = 1e-10
    rng = Random.MersenneTwister(726)
    for _ in 1:5
        θ = fit.theta .+ 0.7 .* randn(rng, 4)
        @test -fit.nll(θ) ≈ hand(θ) atol = 1e-10
    end
    # (2) fit = R reference (generated constants)
    @test coef(fit, :mu) ≈ [0.8576962504, 0.2575260765] atol = 1e-6
    @test coef(fit, :hu) ≈ [-0.0279126001, 0.1453952764] atol = 1e-6
    @test loglik(fit) ≈ -120.5723428573 atol = 1e-6
end

@testset "Hurdle negative-binomial: y ~ x, hu ~ 1 — recovery" begin
    Random.seed!(20260621)
    n = 5000; x = randn(n)
    βμ = [0.7, 0.3]; θ = 3.0; πz = _logis(-0.3)          # π ≈ 0.43
    μ = exp.(βμ[1] .+ βμ[2] .* x)
    y = Float64.([rand() < πz ? 0 : rtnb(θ, θ / (θ + μ[i])) for i in 1:n])

    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(hu ~ 1)), NegBinomial2(); data = (; y, x))

    @test coef(fit, :mu)[1] ≈ βμ[1] atol = 0.12
    @test coef(fit, :mu)[2] ≈ βμ[2] atol = 0.10
    @test _logis(coef(fit, :hu)[1]) ≈ πz atol = 0.06
    @test isfinite(loglik(fit))
end

# drmTMB spells the hurdle NB2 as `truncated_nbinom2()` + `hu ~ ...` — it has no
# `hurdle_nbinom2()` constructor, and the positive component of a hurdle IS the
# zero-truncated count distribution. Accepting that spelling here is what lets a
# drmTMB user flip `engine =` on ONE call. It must fit the SAME likelihood as the
# `NegBinomial2()` + `hu` spelling, not merely a similar one.
@testset "Hurdle NB2 — TruncatedNegBinomial2() + hu is the same fit as NegBinomial2() + hu" begin
    Random.seed!(20260905)
    n = 1200; x = randn(n); w = randn(n)
    βμ = [0.5, 0.35]; θ = exp(2 * 0.30)                   # σ = exp(-0.30) ⇒ θ = 1/σ²
    μ = exp.(βμ[1] .+ βμ[2] .* x)
    πz = _logis.(-0.6 .+ 0.5 .* w)
    y = Float64.([rand() < πz[i] ? 0 : rtnb(θ, θ / (θ + μ[i])) for i in 1:n])
    data = (; y, x, w)
    @test any(y .== 0) && any(y .> 0)                     # a real hurdle fixture

    form = bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(hu ~ w))
    fit_t = drm(form, TruncatedNegBinomial2(); data = data)
    fit_n = drm(form, NegBinomial2(); data = data)

    for p in (:mu, :sigma, :hu)
        @test coef(fit_t, p) == coef(fit_n, p)            # same code path, bit-for-bit
    end
    @test loglik(fit_t) == loglik(fit_n)
    @test isfinite(loglik(fit_t))
    # The delegation is declared, not hidden: the fit reports the family whose
    # `_fit_negbin2_hu` kernel actually ran.
    @test fit_t.family isa NegBinomial2
    # and it is the HURDLE, not the plain zero-truncated model: the hu block
    # exists and both its intercept and slope are recovered
    @test _logis(coef(fit_t, :hu)[1]) ≈ _logis(-0.6) atol = 0.06
    @test coef(fit_t, :hu)[2] ≈ 0.5 atol = 0.15
    @test coef(fit_t, :mu)[1] ≈ βμ[1] atol = 0.15
    @test coef(fit_t, :mu)[2] ≈ βμ[2] atol = 0.10
    # a zero-truncated response check would have rejected this data outright
    @test count(iszero, y) > 100
end

# A formula part the family does not consume must be an ERROR, never a silent
# drop. Before this guard, `zi` was quietly deleted from the likelihood and
# `TruncatedNegBinomial2()` fitted the plain zero-truncated model instead — a
# different model than the caller wrote, with no diagnostic.
@testset "TruncatedNegBinomial2 refuses a formula part it does not consume" begin
    Random.seed!(20260905)
    n = 300; x = randn(n); w = randn(n)
    y = Float64.([rtnb(3.0, 3.0 / (3.0 + exp(0.5 + 0.3 * x[i]))) for i in 1:n])
    data = (; y, x, w)
    @test all(y .>= 1)

    err = try
        drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(zi ~ w)),
            TruncatedNegBinomial2(); data = data)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("unsupported formula part `zi`", err)
    @test occursin("belongs to NegBinomial2()", err)

    # the plain zero-truncated fit still works, unchanged
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), TruncatedNegBinomial2(); data = data)
    @test isfinite(loglik(fit))
    @test length(coef(fit, :mu)) == 2
end
