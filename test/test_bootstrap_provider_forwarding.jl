using DRModels
using Test, Random, LinearAlgebra

@testset "coordinate provider survives every bootstrap surface" begin
    rng = MersenneTwister(20260831)
    G = 8
    m = 4
    coords = rand(rng, G, 2) .* 3
    D = [sqrt(sum(abs2, coords[i, :] .- coords[j, :])) for i in 1:G, j in 1:G]
    C = exp.(-D ./ 0.8) + 1e-8I
    site = repeat(1:G, inner=m)
    x = randn(rng, G*m)
    u = 0.45 .* (cholesky(Symmetric(C)).L * randn(rng, G))
    y = 0.2 .+ 0.35 .* x .+ u[site] .+ 0.3 .* randn(rng, G*m)
    data = (; y, x, site)
    form = bf(@formula(y ~ x + spatial(1 | site)), @formula(sigma ~ 1))
    fit = drm(form, Gaussian(); data, coords, g_tol=1e-6)
    @test fit.converged

    # Independent small-model oracle: the marginal bootstrap draw must use the
    # fitted coordinate-spatial range, not an identity matrix or a fixed K.
    ρfit = exp(only(coef(fit, :range)))
    Cfit = exp.(-D ./ ρfit) + 1e-8I
    seed = 699
    oracle_rng = MersenneTwister(seed)
    ustar = only(values(re_sd(fit))) .* (cholesky(Symmetric(Cfit)).L * randn(oracle_rng, G))
    expected = predict(fit, data) .+ ustar[site] .+
               DRModels._scale_vector(fit, :sigma) .* randn(oracle_rng, G*m)
    simulator = DRModels._marginal_simulator(fit, data; coords)
    @test simulator(MersenneTwister(seed)) ≈ expected atol=1e-12 rtol=1e-12

    fit_result = bootstrap_result(fit; data, coords, B=2,
        rng=MersenneTwister(701), failures=:skip, check_converged=false,
        g_tol=1e-6)
    threaded_result = bootstrap_result(fit; data, coords, B=2,
        rng=MersenneTwister(701), failures=:skip, check_converged=false,
        threads=true, g_tol=1e-6)
    formula_result = bootstrap_result(form, Gaussian(); data, coords, B=2,
        rng=MersenneTwister(701), failures=:skip, check_converged=false,
        g_tol=1e-6)
    @test fit_result.attempted == formula_result.attempted == 2
    @test fit_result.used == formula_result.used == 2
    @test fit_result.seeds == formula_result.seeds
    @test threaded_result.seeds == fit_result.seeds
    @test threaded_result.summary == fit_result.summary
    @test threaded_result.threaded == (Threads.nthreads() > 1)

    rows_summary = bootstrap_summary(fit; data, coords, B=2,
        rng=MersenneTwister(701), failures=:skip, check_converged=false,
        g_tol=1e-6)
    rows_ci = bootstrap_ci(fit; data, coords, B=2,
        rng=MersenneTwister(701), failures=:skip, check_converged=false,
        g_tol=1e-6)
    @test [(r.param, r.coef, r.lower, r.upper) for r in rows_summary] ==
          [(r.param, r.coef, r.lower, r.upper) for r in fit_result.summary]
    @test rows_ci == [(param=r.param, coef=r.coef, estimate=r.estimate,
                      lower=r.lower, upper=r.upper)
                     for r in fit_result.summary]

    bridge = drm_bridge_inference(
        formula=Dict("mu" => "y ~ x + spatial(1 | site)",
                     "sigma" => "sigma ~ 1"),
        family="gaussian", data=data, coords=coords,
        options=Dict("g_tol" => 1e-6), method="bootstrap", level=0.95,
        B=2, seed=701, parm="fixef:mu:x")
    direct_x = only(r for r in fit_result.summary if r.param === :mu && r.coef == "x")
    @test bridge["attempted"] == bridge["used"] == 2
    @test bridge["estimate"] ≈ direct_x.estimate atol=1e-12 rtol=1e-12
    @test bridge["lower"] ≈ direct_x.lower atol=1e-12 rtol=1e-12
    @test bridge["upper"] ≈ direct_x.upper atol=1e-12 rtol=1e-12

    for (marker, provider) in (("relmat", (; K=Matrix(C))),
                               ("animal", (; A=Matrix(C))))
        result = drm_bridge_inference(
            formula=Dict("mu" => "y ~ x + $marker(1 | site)",
                         "sigma" => "sigma ~ 1"),
            family="gaussian", data=data,
            K=get(provider, :K, nothing), A=get(provider, :A, nothing),
            options=Dict("g_tol" => 1e-6), method="bootstrap", level=0.95,
            B=2, seed=702, parm="fixef:mu:x")
        @test result["attempted"] == result["used"] == 2
        @test all(isfinite, (result["estimate"], result["lower"], result["upper"]))
    end

    @test_throws ErrorException bootstrap_result(fit; data, B=2,
        rng=MersenneTwister(701), failures=:skip, check_converged=false,
        g_tol=1e-6)
end
