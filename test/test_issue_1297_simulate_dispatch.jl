# refs #1297 — simulation for LogNormal / Student / CumulativeLogit uses typed methods.
module TestIssue1297SimulateDispatch

using DRModels, Random, Test

@testset "#1297 typed _simulate_once dispatch" begin
    Random.seed!(1297)
    x = sin.(1:50)
    y_ln = [exp(0.2 + 0.4 * x[i]) for i in 1:50]
    fit_ln = drm(bf(@formula(y_ln ~ x), @formula(sigma ~ 1)), LogNormal(); data = (; y_ln, x))
    sim_ln = simulate(fit_ln)
    @test length(sim_ln) == 50
    @test all(isfinite, sim_ln)
    @test all(>(0), sim_ln)

    y_ord = repeat(1:3; inner = 17)[1:50]
    fit_cl = drm(bf(@formula(y_ord ~ x), @formula(sigma ~ 1)), CumulativeLogit();
                 data = (; y_ord, x))
    sim_cl = simulate(fit_cl)
    @test length(sim_cl) == 50
    @test all(i -> i in 1:3, sim_cl)
end

end