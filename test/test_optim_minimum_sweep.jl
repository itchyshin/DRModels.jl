# test_optim_minimum_sweep.jl — sweep follow-up to test_optim_minimum_contract.jl.
# The boundary-restart comparison in src/gaussian_ranef.jl used
# `Optim.minimum(res2) < Optim.minimum(res)`; after a failed line search either
# side can be stale, so the WRONG restart could be kept. It now goes through
# `DRModels._better_restart`, which compares f(minimizer) freshly.
#
#   julia --project=. -e 'using DRModels, Test; include("test/test_optim_minimum_sweep.jl")'

module TestOptimMinimumSweep

using DRModels
using Test
using DRModels: Optim

const SENTINEL = 1e18
f(x) = x[1] > 5.0 ? SENTINEL : (x[1] - 3.0)^2
function g!(G, x)
    G[1] = x[1] > 5.0 ? 0.0 : 2 * (x[1] - 3.0)
end

# A run whose line search fails: minimum(res) == SENTINEL, but f(minimizer) == 9.
function stale_run()
    ls = Optim.LineSearches.HagerZhang(linesearchmax = 2)
    method = Optim.LBFGS(alphaguess = Optim.LineSearches.InitialStatic(alpha = 1e7),
                         linesearch = ls)
    Optim.optimize(f, g!, [0.0], method,
                   Optim.Options(iterations = 50, allow_f_increases = true))
end
# An honest, non-stale run: zero iterations, so minimum(res) == f(minimizer).
honest_run(x0) = Optim.optimize(f, g!, [x0], Optim.LBFGS(), Optim.Options(iterations = 0))

@testset "_better_restart: stale Optim.minimum cannot pick the wrong restart" begin
    stale = stale_run()
    @test !Optim.converged(stale)
    @test Optim.minimum(stale) == SENTINEL           # premise: stale value
    @test f(Optim.minimizer(stale)) ≈ 9.0 atol = 1e-8

    # Case 1: the incumbent is stale-but-good (true 9); the restart is honestly
    # worse (true 16). Old rule: minimum(res2)=16 < 1e18 -> kept the WORSE restart.
    worse = honest_run(-1.0)
    @test Optim.minimum(worse) ≈ 16.0
    @test Optim.minimum(worse) < Optim.minimum(stale)   # the buggy comparison's verdict
    @test DRModels._better_restart(f, stale, worse) === stale

    # Case 2: the restart is stale-but-good (true 9); the incumbent is honestly
    # worse (true 16). Old rule: 1e18 < 16 is false -> DISCARDED the better restart.
    @test !(Optim.minimum(stale) < Optim.minimum(worse))
    @test DRModels._better_restart(f, worse, stale) === stale

    # Converged/honest case: identical to the old rule, and ties keep the incumbent.
    better = honest_run(2.0)                              # true value 1
    @test DRModels._better_restart(f, worse, better) === better
    @test DRModels._better_restart(f, better, worse) === better
    @test DRModels._better_restart(f, better, better) === better
end

@testset "src/gaussian_ranef.jl no longer compares bare Optim.minimum" begin
    src = read(joinpath(pkgdir(DRModels), "src", "gaussian_ranef.jl"), String)
    @test !occursin("Optim.minimum(res2)", src)
    @test occursin("_better_restart(nllc, res, res2)", src)
end

end # module
