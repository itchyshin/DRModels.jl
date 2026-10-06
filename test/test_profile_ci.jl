# Profile-likelihood confidence intervals: confint(fit; method = :profile).
# Mirrors drmTMB's confint(fit, method = "profile"). For each parameter the
# interval endpoints are the values where 2(ℓ̂ − ℓ_profile) = χ²₁(level), with
# the nuisance parameters re-optimised at each fixed value. For a Gaussian mean
# coefficient the log-likelihood is near-quadratic, so the profile interval
# closely matches the Wald interval — that is the cross-check here.
using DRModels
using Test, Random
import Distributions

@testset "Profile-likelihood CI: confint(fit; method=:profile)" begin
    Random.seed!(20260612)
    n = 500
    x = randn(n); z = randn(n)
    β = [0.4, 0.8, -0.5]; σ = 0.5
    y = β[1] .+ β[2] .* x .+ β[3] .* z .+ σ .* randn(n)
    data = (; y, x, z)
    fit = drm(bf(@formula(y ~ x + z), @formula(sigma ~ 1)), Gaussian(); data = data)

    wald = confint(fit; method = :wald)
    prof = confint(fit; method = :profile)

    @test length(prof) == length(wald)
    @test [r.param for r in prof] == [r.param for r in wald]   # same rows / order
    @test [r.coef for r in prof] == [r.coef for r in wald]
    @test all(r.estimate ≈ w.estimate for (r, w) in zip(prof, wald))

    @test all(r.lower < r.estimate < r.upper for r in prof)    # every CI brackets est

    # mean coefficients: near-quadratic likelihood ⇒ profile ≈ Wald
    for (r, w) in zip(prof, wald)
        r.param === :mu || continue
        @test r.lower ≈ w.lower atol = 0.03
        @test r.upper ≈ w.upper atol = 0.03
    end

    @test confint(fit) == wald                                 # default stays Wald
    @test [r.param for r in confint(fit; method = :wald, parm = :mu)] == fill(:mu, 3)

    pres = profile_result(fit)
    @test pres.ci == prof
    @test pres.attempted == pres.used == length(prof)
    @test pres.failed == 0
    @test pres.threaded == false
    @test pres.worker_threads == 1
    @test pres.julia_threads == Threads.nthreads()
    @test pres.blas_threads >= 1
    @test pres.elapsed >= 0
    @test pres.autodiff in (:stored, :forward, :finite)
    @test length(pres.stats) == length(prof)
    @test all(s.evaluations > 0 for s in pres.stats)

    sigma_serial = profile_result(fit; parm = :sigma)
    sigma_threaded = profile_result(fit; parm = :sigma, threads = true)
    @test sigma_threaded.ci == sigma_serial.ci
    @test sigma_threaded.threaded == (Threads.nthreads() > 1)
    @test sigma_threaded.worker_threads == (Threads.nthreads() > 1 ? min(2, Threads.nthreads()) : 1)
end

@testset "Gaussian crossed profile uses stored gradient" begin
    rng = MersenneTwister(20260614)
    G = 7
    H = 6
    n = 260
    x = randn(rng, n)
    gid = rand(rng, 1:G, n)
    hid = rand(rng, 1:H, n)
    g = Symbol.("g", gid)
    h = Symbol.("h", hid)
    bg = 0.45 .* randn(rng, G)
    bh = 0.35 .* randn(rng, H)
    y = Float64[
        0.25 + 0.55 * x[i] + bg[gid[i]] + bh[hid[i]] + 0.45 * randn(rng)
        for i in 1:n
    ]

    fit = drm(
        bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1)),
        Gaussian();
        data = (; y, x, g, h),
    )
    @test fit.nllgrad !== nothing

    θ = fit.theta .+ [0.01, -0.02, 0.015, 0.01, -0.012]
    gstored = zeros(length(θ))
    fit.nllgrad(gstored, θ)
    hfd = 1e-5
    gfd = similar(gstored)
    for j in eachindex(θ)
        step = zeros(length(θ))
        step[j] = hfd
        gfd[j] = (fit.nll(θ .+ step) - fit.nll(θ .- step)) / (2hfd)
    end
    @test maximum(abs.(gstored .- gfd)) < 1e-5

    resd = confint(fit; method = :profile, parm = :resd)
    @test [r.coef for r in resd] == ["g", "h"]
    @test all(isfinite(r.lower) && isfinite(r.upper) for r in resd)

    one_serial = profile_result(fit; parm = :resd)
    one_threaded = profile_result(fit; parm = :resd, threads = true)
    @test one_threaded.ci == one_serial.ci
    @test one_threaded.stats[1].evaluations == one_serial.stats[1].evaluations
    @test one_threaded.threaded == (Threads.nthreads() > 1)
    @test one_threaded.worker_threads == (Threads.nthreads() > 1 ? min(2, Threads.nthreads()) : 1)
    @test one_threaded.blas_oversubscribed == (one_threaded.threaded && one_threaded.blas_threads > 1)
end

@testset "Profile CI on crossed Poisson random-effect SDs" begin
    rng = MersenneTwister(20260613)
    G = 8
    H = 7
    n = 420
    x = randn(rng, n)
    gid = rand(rng, 1:G, n)
    hid = rand(rng, 1:H, n)
    g = Symbol.("g", gid)
    h = Symbol.("h", hid)
    bg = 0.55 .* randn(rng, G)
    bh = 0.40 .* randn(rng, H)
    η = @. 0.2 + 0.45 * x + bg[gid] + bh[hid]
    y = Float64[rand(rng, Distributions.Poisson(exp(η[i]))) for i in 1:n]

    fit = drm(bf(@formula(y ~ x + (1 | g) + (1 | h))), Poisson(); data = (; y, x, g, h))
    @test fit.nllgrad !== nothing
    resd = confint(fit; method = :profile, parm = :resd)

    @test [r.coef for r in resd] == ["g", "h"]
    @test all(isfinite(r.lower) && isfinite(r.upper) for r in resd)
    @test all(r.lower < r.estimate < r.upper for r in resd)
    @test all(exp(r.lower) < exp(r.estimate) < exp(r.upper) for r in resd)

    curve = profile_curve(fit, fit.blocks[2].second[1]; npoints = 9)
    @test curve.param === :resd
    @test curve.coef == "g"
    @test length(curve.x) == length(curve.deviance) == 9
    @test all(isfinite, curve.deviance)
end

# The fast endpoint finder (warm-start continuation + guarded-Newton using the
# envelope-theorem slope) must stay correct on an ASYMMETRIC profile — the scale
# parameter, where the log-σ likelihood is not symmetric, so bisection-vs-Newton
# and the bracket guard are genuinely exercised (the mean case above is near
# quadratic). Endpoints must be finite, bracket the estimate, and (since the LR
# and Wald agree to first order) sit within a sane neighbourhood of the Wald CI.
@testset "Profile CI on the scale parameter (asymmetric)" begin
    Random.seed!(20260601)
    n = 400
    x = randn(n)
    y = 0.5 .+ 0.7 .* x .+ exp.(0.2 .* x) .* randn(n)     # heteroscedastic → σ ~ x
    data = (; y, x)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = data)

    prof = confint(fit; method = :profile)
    wald = confint(fit; method = :wald)
    @test [r.param for r in prof] == [r.param for r in wald]
    @test all(isfinite(r.lower) && isfinite(r.upper) for r in prof)
    @test all(r.lower < r.estimate < r.upper for r in prof)

    # profile endpoints should land near Wald (same first-order curvature), but
    # need not be symmetric — a loose neighbourhood check, not equality.
    for (r, w) in zip(prof, wald)
        @test r.lower ≈ w.lower atol = 0.15
        @test r.upper ≈ w.upper atol = 0.15
    end
end
