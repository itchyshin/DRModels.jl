# DRModels.jl#944: Optim.converged is the OR of the x, f, and g criteria.
# With default f/x tolerances of 0, a numerical plateau sets f_converged or
# x_converged while the gradient criterion is still false. The reported flag
# must require the gradient criterion. Routes that deliberately stop on
# f_reltol (#946) are not covered here.

using DRModels, Optim, Test

function _plateau_result(res)
    stopped = merge(res.stopped_by, (x_converged = true, f_converged = true, g_converged = false))
    res.stopped_by = stopped
    return res
end

@testset "drm_optim_converged requires the gradient criterion (#944)" begin
    good = Optim.optimize(x -> (x[1] - 0.3)^2, [1.0], Optim.LBFGS(),
                          Optim.Options(g_tol = 1e-8))
    @test Optim.converged(good)
    @test Optim.g_converged(good)
    @test DRModels.drm_optim_converged(good)

    plateau = _plateau_result(deepcopy(good))
    @test Optim.converged(plateau)
    @test !Optim.g_converged(plateau)
    @test !DRModels.drm_optim_converged(plateau)
end
