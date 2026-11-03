# test_1142_crossed_failed_fit_warning.jl -- #1142: a crossed nuisance fit that ends on the
# failed-inner-solve sentinel (objective 1e18) must say so.
#
# On the Julia 1.10 seed-3 Beta draws of #1142 the start point is already the sentinel: nothing
# runs past a clamp (so `_warn_nuisance_runoff` is silent, and the box re-fit does not fire), the
# optimiser never moves, and the only message was the NaN-vcov one. Those draws are not
# committed and differ by Julia version, so the sentinel is forced here: the Beta inner solve is
# made to fail at every point by a precision that is NaN.
using DRModels
using Test, Random, Logging

@testset "#1142: crossed Beta fit ending on the 1e18 sentinel warns and is not converged" begin
    rng = MersenneTwister(3)
    G, H, n = 6, 5, 60
    gidx = rand(rng, 1:G, n); hidx = rand(rng, 1:H, n)
    x = randn(rng, n)
    y = clamp.(1 ./ (1 .+ exp.(-(0.2 .+ 0.3 .* x .+ 0.5 .* randn(rng, G)[gidx] .+
                                 0.4 .* randn(rng, H)[hidx] .+ 0.3 .* randn(rng, n)))),
               1e-6, 1 - 1e-6)
    X = hcat(ones(n), x)
    ylogit = log.(y) .- log1p.(-y)
    # The failed inner solve: a NaN precision makes every Laplace mode solve fail.
    aux_from(logsigma) = (y = y, precision = NaN, ylogit = ylogit,
                          lgammaφ = NaN, digammaφ = NaN)
    run_fit(; kw...) = DRModels._fit_crossed_mean_laplace_nuisance(
        DRModels.Beta(), Val(:beta_fixed), aux_from, n, X, gidx, G, hidx, H,
        ["(Intercept)", "x"], ["(Intercept)"], ["g", "h"], 1e-8;
        θβ0 = [0.0, 0.0], θσ0 = -0.5 * log(0.5), sigma_scale = exp, se = true, kw...)

    # The forcing really lands on the sentinel at the start point.
    θ0 = [0.0, 0.0, -0.5 * log(0.5), log(0.4), log(0.4)]
    @test DRModels._crossed_mean_laplace_nuisance_fg(
        Val(:beta_fixed), aux_from, n, X, gidx, G, hidx, H, θ0; grad = false,
        b0 = zeros(G + H))[3] == false

    logger = Test.TestLogger(min_level = Logging.Warn)
    fit = Logging.with_logger(logger) do
        run_fit()
    end
    msgs = [string(r.message) for r in logger.logs]
    @test fit.loglik == -1.0e18
    @test !is_converged(fit)
    # A dedicated failed-fit warning, distinct from the runoff and the NaN-vcov ones.
    @test count(m -> occursin("the fit failed", m) && occursin("sentinel", m), msgs) == 1
    @test !any(m -> occursin("ended past an upper clamp", m), msgs)
    @test any(m -> occursin("NaN vcov", m), msgs)

    # Quiet inside bootstrap / profile refits, like the other boundary warnings.
    qlogger = Test.TestLogger(min_level = Logging.Warn)
    Logging.with_logger(qlogger) do
        task_local_storage(:drm_quiet_boundary, true) do
            run_fit()
        end
    end
    @test !any(r -> occursin("the fit failed", string(r.message)), qlogger.logs)
end

@testset "#1142: the committed Julia 1.10 seed-3 Beta draws end on the sentinel and warn" begin
    # Frozen Julia 1.10.12 draws of `gen_beta_runoff(3, 300)` (test/parity/gen_1088_nb2_runoff_data.jl).
    # MersenneTwister draws differ from Julia 1.12, whose draws (fixture beta_runoff_s3_n300) run
    # off and are recovered. Here the start point is already the failed-inner-solve sentinel:
    # nothing runs past a clamp, the box re-fit does not fire, log sigma stays at its start value.
    lines = readlines(joinpath(@__DIR__, "fixtures", "j3_se", "beta_runoff_s3_n300_jl110", "data.csv"))
    rows = [split(l, ',') for l in lines[2:end]]
    d = (; y = [parse(Float64, r[1]) for r in rows], x = [parse(Float64, r[2]) for r in rows],
         g = String[r[3] for r in rows], h = String[r[4] for r in rows])
    DRModels._CROSSED_NUISANCE_BOX_REFITS[] = 0
    logger = Test.TestLogger(min_level = Logging.Warn)
    fit = Logging.with_logger(logger) do
        drm(bf(@formula(y ~ 1 + x + (1 | g) + (1 | h)), @formula(sigma ~ 1)), DRModels.Beta(); data = d)
    end
    msgs = [string(r.message) for r in logger.logs]
    @test length(d.y) == 300
    @test loglik(fit) == -1.0e18
    @test !is_converged(fit)
    @test fit.theta[3] ≈ 0.34657359027997264 atol = 1e-9   # log of the Method-of-Moments cap: never moved
    @test DRModels._CROSSED_NUISANCE_BOX_REFITS[] == 0
    @test count(m -> occursin("the fit failed", m) && occursin("sentinel", m), msgs) == 1
    @test !any(m -> occursin("ended past an upper clamp", m), msgs)
    @test any(m -> occursin("NaN vcov", m), msgs)
end

@testset "#1142: a healthy crossed Beta fit does not emit the failed-fit warning" begin
    rng = MersenneTwister(11)
    G, H, n = 8, 6, 200
    gidx = rand(rng, 1:G, n); hidx = rand(rng, 1:H, n)
    x = randn(rng, n)
    mu = 1 ./ (1 .+ exp.(-(0.2 .+ 0.3 .* x .+ 0.4 .* randn(rng, G)[gidx] .+
                           0.3 .* randn(rng, H)[hidx])))
    y = clamp.(mu .+ 0.05 .* randn(rng, n), 0.02, 0.98)
    comps = [(ones(n), gidx, G, "g"), (ones(n), hidx, H, "h")]
    logger = Test.TestLogger(min_level = Logging.Warn)
    f = Logging.with_logger(logger) do
        DRModels._fit_beta_crossed_laplace(DRModels.Beta(), y, hcat(ones(n), x), ones(n, 1),
                                           comps, ["(Intercept)", "x"], ["(Intercept)"], 1e-8;
                                           se = false)
    end
    @test f.loglik > -1e17
    @test !any(r -> occursin("the fit failed", string(r.message)), logger.logs)
end
