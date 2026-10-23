# TruncatedPoisson(): zero-truncated Poisson, and the shared hurdle-Poisson
# spelling (`TruncatedPoisson()` + `hu`, drmTMB's `truncated_poisson()` + `hu`;
# issue #726, owner decision 16a). Mirrors TruncatedNegBinomial2 (test_hurdle.jl).
using DRModels
using Test, Random
import Distributions

_rtpois_tp(λ) = (while true; k = rand(Distributions.Poisson(λ)); k > 0 && return k; end)

# The #726 / #906 fixture (test_hurdle.jl "Hurdle Poisson: log-likelihood and fit pinned").
const _TP_Y = Float64.([0, 1, 3, 0, 1, 0, 3, 0, 0, 2, 0, 3, 4, 0, 4, 7, 0, 0, 0, 0, 0, 1, 6, 0, 0, 0, 0, 0, 1, 0, 2, 3, 2, 2, 0, 2, 0, 1, 0, 1, 0, 2, 1, 3, 3, 0, 5, 1, 0, 1, 1, 0, 5, 0, 3, 0, 3, 3, 0, 0, 0, 2, 0, 5, 0, 0, 3, 0, 4, 3, 2, 0, 0, 0, 1, 0, 0, 2, 4, 0])
const _TP_X = [0.093, -1.593, 2.332, 0.18, -2.515, 0.532, -0.098, -1.276, 1.172, -0.325, -0.562, -1.153, -0.578, -0.747, 0.152, 0.714, 1.328, -1.004, 0.866, 0.702, -0.487, 0.497, 1.432, -0.51, -0.528, -1.254, -0.302, -0.041, 0.566, -0.791, -0.78, 0.211, 0.682, 0.471, 0.276, 0.889, 0.377, 0.532, 0.697, 0.72, -1.778, -0.207, -0.506, -0.424, -1.042, 0.138, 0.578, -0.328, 0.375, 0.487, -0.371, 0.731, -0.019, 0.999, 0.423, -1.13, -0.311, 0.208, 0.623, 0.657, -0.355, -0.829, 1.072, 0.106, 0.227, -1.101, -1.207, -0.081, 1.924, -0.483, -0.468, -1.379, 0.676, -1.296, -1.075, 0.09, 0.027, -0.225, -0.152, -0.771]
const _TP_W = [1.419, 1.539, -0.011, -1.59, -0.287, 0.604, -0.752, 1.101, 1.052, 0.905, 1.031, -2.129, 0.101, 0.78, 0.206, 0.446, -0.483, 0.475, 0.827, -0.047, -2.175, -1.492, 0.514, -0.807, 0.392, 0.347, 0.522, 0.555, 0.399, -0.378, 1.576, 1.319, -2.051, 0.469, 0.875, -0.823, -0.236, 0.608, 0.908, 1.991, -0.02, 0.032, 0.832, -0.93, -0.685, -0.25, 0.587, 0.433, -0.004, 1.198, -1.407, 1.465, 0.227, 0.2, -0.088, 0.588, 0.408, 0.584, 0.536, 0.511, 0.03, -0.921, -0.732, 0.791, -0.167, -0.139, 0.434, -0.28, 0.681, -1.012, 0.795, -0.101, 2.34, 0.383, 1.67, 0.449, 0.038, -1.527, 0.65, 0.005]

@testset "TruncatedPoisson() + hu is the same fit as Poisson() + hu (bit-identical)" begin
    data = (; y = _TP_Y, x = _TP_X, w = _TP_W)
    form = bf(@formula(y ~ x), @formula(hu ~ w))
    fit_t = drm(form, TruncatedPoisson(); data = data)
    fit_p = drm(form, Poisson(); data = data)
    for p in (:mu, :hu)
        @test coef(fit_t, p) == coef(fit_p, p)
    end
    @test loglik(fit_t) == loglik(fit_p)
    @test fit_t.theta == fit_p.theta
    @test fit_t.family isa Poisson          # delegation is declared: the Poisson kernel ran
    # the #906 pinned values, reproduced under the new spelling
    @test fit_t.converged
    @test coef(fit_t, :mu) ≈ [0.8576962504, 0.2575260765] atol = 1e-6
    @test coef(fit_t, :hu) ≈ [-0.0279126001, 0.1453952764] atol = 1e-6
    @test loglik(fit_t) ≈ -120.5723428573 atol = 1e-6
    @test isfinite(loglik(fit_p))           # Poisson() + hu still works
end

@testset "TruncatedPoisson() without hu: zero-truncated Poisson likelihood" begin
    Random.seed!(20261001)
    n = 600; x = randn(n)
    βμ = [0.7, 0.4]
    y = Float64.([_rtpois_tp(exp(βμ[1] + βμ[2] * x[i])) for i in 1:n])
    data = (; y, x)
    @test all(y .>= 1)
    fit = drm(bf(@formula(y ~ x)), TruncatedPoisson(); data = data)
    @test fit.converged
    @test fit.family isa TruncatedPoisson
    @test coef(fit, :mu) ≈ βμ atol = 0.12
    # log-likelihood = hand computation with Distributions, at the optimum and off it
    hand(β) = sum(eachindex(y)) do i
        λ = exp(β[1] + β[2] * x[i]); d = Distributions.Poisson(λ)
        Distributions.logpdf(d, Int(y[i])) - log1p(-Distributions.pdf(d, 0))
    end
    @test loglik(fit) ≈ hand(fit.theta) atol = 1e-10
    rng = Random.MersenneTwister(1001)
    for _ in 1:5
        β = fit.theta .+ 0.5 .* randn(rng, 2)
        @test -fit.nll(β) ≈ hand(β) atol = 1e-10
    end
    # sanity: the truncated pmf sums to 1 over k ≥ 1
    d0 = Distributions.Poisson(1.7)
    @test sum(Distributions.pdf(d0, k) / (1 - Distributions.pdf(d0, 0)) for k in 1:200) ≈ 1 atol = 1e-12
    # downstream accessors work on a no-hu fit
    @test all(isfinite, predict(fit, data))
    ysim = simulate(fit; rng = Random.MersenneTwister(2))
    @test length(ysim) == n && all(ysim .>= 1)
    @test all(isfinite, residuals(fit; type = :quantile))
end

@testset "TruncatedPoisson() refusals mirror TruncatedNegBinomial2()" begin
    Random.seed!(20261001)
    n = 200; x = randn(n); w = randn(n)
    y = Float64.([_rtpois_tp(exp(0.6 + 0.3 * x[i])) for i in 1:n])
    data = (; y, x, w)
    msg(f) = try; f(); ""; catch e; sprint(showerror, e); end
    err = msg(() -> drm(bf(@formula(y ~ x), @formula(zi ~ w)), TruncatedPoisson(); data = data))
    @test occursin("unsupported formula part `zi`", err)
    @test occursin("belongs to Poisson()", err)
    err = msg(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ w)), TruncatedPoisson(); data = data))
    @test occursin("unsupported formula part `sigma`", err)
    y0 = copy(y); y0[1] = 0.0                       # zeros need a hurdle part
    err = msg(() -> drm(bf(@formula(y ~ x)), TruncatedPoisson(); data = (; y = y0, x, w)))
    @test occursin("requires positive integer counts", err)
    g = repeat(1:10, inner = 20)
    err = msg(() -> drm(bf(@formula(y ~ x + (1 | g))), TruncatedPoisson(); data = (; y, x, g)))
    @test occursin("currently supports fixed effects only", err)
end

@testset "TruncatedPoisson bridge name" begin
    @test DRModels._bridge_family("truncated_poisson") isa TruncatedPoisson
end
