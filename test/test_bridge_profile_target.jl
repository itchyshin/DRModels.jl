using DRModels, Test, Random, LinearAlgebra
using Distributions: Chisq, quantile

@testset "bridge profiles only the requested coefficient" begin
    rng = MersenneTwister(56329)
    n = 80
    x = randn(rng, n)
    z = randn(rng, n)
    y = 0.4 .+ 0.7x .- 0.2z .+ 0.6randn(rng, n)
    data = (; y, x, z)
    X = hcat(ones(n), x, z)
    beta = X \ y
    sse = sum(abs2, y - X * beta)
    # Independent exact Gaussian ML profile: RSS(v) = RSSmin +
    # (v-beta[k])^2 / inv(X'X)[k,k], and LR = n*log(RSS(v)/RSSmin).
    q = quantile(Chisq(1), 0.9)
    for (k, name) in ((1, "(Intercept)"), (2, "x"), (3, "z"))
        halfwidth = sqrt(expm1(q / n) * sse * inv(X'X)[k, k])
        result = drm_bridge_inference(; formula="y ~ x + z; sigma ~ 1",
            family="gaussian", data, method="profile", level=0.9,
            parm="fixef:mu:$name", threads=false)
        @test result["attempted"] == 1
        @test result["used"] == 1
        @test result["failed"] == 0
        @test result["coef"] == name
        @test result["estimate"] ≈ beta[k] atol=1e-6
        @test result["lower"] ≈ beta[k]-halfwidth atol=1e-5
        @test result["upper"] ≈ beta[k]+halfwidth atol=1e-5
        threaded = drm_bridge_inference(; formula="y ~ x + z; sigma ~ 1",
            family="gaussian", data, method="profile", level=0.9,
            parm="fixef:mu:$name", threads=true)
        @test threaded["attempted"] == 1
        @test threaded["failed"] == 0
        @test threaded["worker_threads"] <= 2
        @test threaded["lower"] ≈ result["lower"] atol=1e-8
        @test threaded["upper"] ≈ result["upper"] atol=1e-8
    end
end
