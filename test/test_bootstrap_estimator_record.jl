# DRModels.jl#1025: `bootstrap_result` must repeat the seed fit's estimator in
# every replicate, and must say which estimator it used. The fit-based method
# forwards REML, a non-default marginal and a MAP penalty (checked on stub fits
# in test_sweep_blocked_bootstrap.jl). This file checks the public result:
# the seed estimator and the replicates' estimator are recorded on it, and the
# formula-based method can request REML.

using DRModels, Test, Random

function _est_data()
    rng = MersenneTwister(2501)
    g = repeat(1:12, inner = 5)
    u = randn(rng, 12)
    n = length(g)
    return (y = 1.0 .+ 0.5 .* randn(rng, n) .+ u[g], x = randn(rng, n),
            g = string.(g))
end

@testset "bootstrap_result records the estimator (#1025)" begin
    dat = _est_data()
    form = bf(@formula(y ~ x + (1 | g)))

    reml = drm(form, Gaussian(); data = dat, method = :REML)
    r = bootstrap_result(reml; data = dat, B = 3, rng = MersenneTwister(1))
    @test r.estimation_method === :REML
    @test r.refit_estimation_method === :REML
    @test r.marginal === :LA
    @test r.refit_marginal === :LA

    ml = drm(form, Gaussian(); data = dat)
    r0 = bootstrap_result(ml; data = dat, B = 3, rng = MersenneTwister(1))
    @test r0.estimation_method === :ML
    @test r0.refit_estimation_method === :ML
end

@testset "formula-based bootstrap_result can request REML (#1025)" begin
    dat = _est_data()
    form = bf(@formula(y ~ x + (1 | g)))

    r = bootstrap_result(form, Gaussian(); data = dat, B = 3,
                         rng = MersenneTwister(1), method = :REML)
    @test r.estimation_method === :REML
    @test r.refit_estimation_method === :REML
    reml = drm(form, Gaussian(); data = dat, method = :REML)
    @test [row.estimate for row in r.summary] ≈ [row.estimate for row in
        bootstrap_result(reml; data = dat, B = 3, rng = MersenneTwister(1)).summary]

    @test_throws ArgumentError bootstrap_result(form, Gaussian(); data = dat,
        B = 3, method = :MAP)
end
