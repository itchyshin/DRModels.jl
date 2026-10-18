# #1089: a `missing`/NaN predictor must not crash `drm` (KeyError: key missing not
# found); incomplete rows are dropped like R's na.omit, with a warning giving the count.
using Test, Random, DRModels

@testset "#1089 missing predictors: drop incomplete rows with a counted warning" begin
    Random.seed!(20281017801)
    G, n = 20, 400
    g = repeat(1:G, inner = n ÷ G)
    x = Vector{Union{Missing,Float64}}(randn(n))
    z = randn(n)
    u = randn(G)
    y = Vector{Union{Missing,Float64}}(1 .+ 0.5 .* x .- 0.3 .* z .+ u[g] .+ randn(n))
    ymiss = randperm(n)[1:30]
    xmiss = randperm(n)[1:20]
    y[ymiss] .= missing
    x[xmiss] .= missing
    complete = [!ismissing(y[i]) && !ismissing(x[i]) for i in 1:n]
    ncomplete = count(complete)
    f = bf(@formula(y ~ 1 + x + z + (1 | g)), @formula(sigma ~ 1))

    fit = @test_logs (:warn, r"\b" * string(n - ncomplete) * r" of " * string(n) * r" rows") match_mode = :any drm(f, Gaussian(); data = (; y, x, z, g))
    ref = drm(f, Gaussian(); data = (; y = Float64.(y[complete]), x = Float64.(x[complete]),
                                     z = z[complete], g = g[complete]))
    @test nobs(fit) == ncomplete
    @test loglikelihood(fit) ≈ loglikelihood(ref) atol = 1e-8

    # fixed-effects-only Gaussian and a non-Gaussian family take the same path
    fx = bf(@formula(y ~ x + z), @formula(sigma ~ 1))
    fitx = @test_logs (:warn,) match_mode = :any drm(fx, Gaussian(); data = (; y, x, z, g))
    @test nobs(fitx) == ncomplete
    cnt = Vector{Union{Missing,Float64}}(rand(0:5, n)); cnt[ymiss] .= missing
    fitp = @test_logs (:warn,) match_mode = :any drm(bf(@formula(cnt ~ x + z)), Poisson(); data = (; cnt, x, z, g))
    @test nobs(fitp) == count([!ismissing(cnt[i]) && !ismissing(x[i]) for i in 1:n])

    # NaN predictor is treated like missing
    xn = Float64.(coalesce.(x, NaN))
    fitn = @test_logs (:warn,) match_mode = :any drm(fx, Gaussian(); data = (; y, x = xn, z, g))
    @test nobs(fitn) == ncomplete

    # no missing: no warning, unchanged
    @test_logs drm(fx, Gaussian(); data = (; y = randn(n), x = randn(n), z, g))
end
