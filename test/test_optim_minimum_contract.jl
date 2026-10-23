# test_optim_minimum_contract.jl — contract test for the "Optim.minimum after a
# failed line search" bug class (see src/optim_minimum_guard.jl).
#
# CONFIRMED against Optim v1.13.3 (this repo's Manifest, checked directly, not
# inferred): when Optim.jl's `perform_linesearch!` catches a LineSearchException,
# it still applies the failed trial's alpha to `state.x` (moving the minimizer)
# but the outer loop breaks BEFORE refreshing the cached objective at the new
# `state.x`. So `Optim.minimum(res)` can hold an earlier, REJECTED trial's
# value while `Optim.minimizer(res)` has already moved past it.
#
#   julia --project=. -e 'using DRModels, Test; include("test/test_optim_minimum_contract.jl")'

module TestOptimMinimumContract

using DRModels
using Test
using DRModels: Optim   # Optim is a DRModels dependency, NOT in test/Project.toml

@testset "Optim.jl: minimum(res) can disagree with f(minimizer(res)) after a failed line search" begin
    # A barrier objective: SENTINEL outside a safe region, with a genuinely
    # zero (flat) gradient there so a line search that overshoots into it can
    # never satisfy the Wolfe curvature condition and must fail.
    SENTINEL = 1e18
    f(x) = x[1] > 5.0 ? SENTINEL : (x[1] - 3.0)^2
    function g!(G, x)
        G[1] = x[1] > 5.0 ? 0.0 : 2 * (x[1] - 3.0)
    end

    x0 = [0.0]
    ls = Optim.LineSearches.HagerZhang(linesearchmax = 2)
    method = Optim.LBFGS(alphaguess = Optim.LineSearches.InitialStatic(alpha = 1e7),
                          linesearch = ls)
    res = Optim.optimize(f, g!, x0, method,
                          Optim.Options(iterations = 50, allow_f_increases = true))

    # The reproduction is itself the contract: confirm the failure mode fires
    # (a green run here without it would mean Optim.jl's internals changed and
    # this whole guard's premise needs re-checking, not that the bug is gone).
    @test !Optim.converged(res)
    @test Optim.minimum(res) == SENTINEL
    @test f(Optim.minimizer(res)) != Optim.minimum(res)
    @test isapprox(f(Optim.minimizer(res)), (0.0 - 3.0)^2; atol = 1e-8)

    # DRModels.jl's own helper (src/optim_minimum_guard.jl) must report the
    # true value at the minimizer, not the stale `Optim.minimum(res)`.
    @test DRModels._objective_at_minimizer(f, res) == f(Optim.minimizer(res))

    # only_fg!-style closures (fg!(F, G, x)) hit the same failure mode; the
    # `_objective_at_minimizer_fg` helper must recover the same true value.
    fg! = function (F, G, x)
        G !== nothing && g!(G, x)
        return f(x)
    end
    od = Optim.NLSolversBase.only_fg!(fg!)
    res_fg = Optim.optimize(od, x0, method,
                             Optim.Options(iterations = 50, allow_f_increases = true))
    @test !Optim.converged(res_fg)
    @test DRModels._objective_at_minimizer_fg(fg!, res_fg) == f(Optim.minimizer(res_fg))
end

@testset "_objective_at_minimizer(_fg): identical to Optim.minimum in the converged case" begin
    # Sanity check that the helpers change NOTHING when the line search
    # actually succeeds -- the normal, converged path must be numerically
    # unaffected by this fix.
    f(x) = (x[1] - 3.0)^2 + (x[2] + 1.0)^2
    function g!(G, x)
        G[1] = 2 * (x[1] - 3.0)
        G[2] = 2 * (x[2] + 1.0)
    end
    res = Optim.optimize(f, g!, [0.0, 0.0], Optim.LBFGS(), Optim.Options(g_tol = 1e-10))
    @test Optim.converged(res)
    @test DRModels._objective_at_minimizer(f, res) == Optim.minimum(res)

    fg! = function (F, G, x)
        G !== nothing && g!(G, x)
        return f(x)
    end
    od = Optim.NLSolversBase.only_fg!(fg!)
    res_fg = Optim.optimize(od, [0.0, 0.0], Optim.LBFGS(), Optim.Options(g_tol = 1e-10))
    @test Optim.converged(res_fg)
    @test DRModels._objective_at_minimizer_fg(fg!, res_fg) == Optim.minimum(res_fg)
end

end # module
