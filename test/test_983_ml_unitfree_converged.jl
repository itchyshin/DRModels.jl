# #983: the default ML Gaussian location-scale route reported `converged = false`
# for fits at the optimum when the residual SD is tiny (cell 15: sd about 1e-4).
# Three unit-dependent vetoes caused it: `Optim.converged` (a line-search stall at
# the noise floor), the LBFGS iteration cap reached at the optimum, and the
# raw-eigenvalue singularity test inside `_unitfree_converged` (a Hessian whose
# blocks differ by 1e8 only because of the units of y).
using DRModels
using Test, Random, LinearAlgebra, Logging

function _gen_983(rep, sd; n = 100)
    Random.seed!(20260927 + 15000 + rep)
    x = randn(n)
    return (; y = 1 .+ 0.5 .* x .+ sd .* randn(n), x)
end

@testset "#983 ML Gaussian unit-free converged flag" begin
    f = bf(@formula(y ~ x), @formula(sigma ~ 1))

    # Every rep of the filed cell is at the optimum, and the verdict does not move
    # when y is rescaled. The cap still fails the flag (kept as in #1170): on these
    # seeds 48 of 50 converge and the other two stop on the 1000-iteration cap
    # (Julia 1.12.6, macOS). Main gave 19 of 50.
    nconv = 0
    for rep in 1:50
        d = _gen_983(rep, exp(-9.2))
        fit = drm(f, Gaussian(); data = d)
        fit4 = drm(f, Gaussian(); data = (; y = d.y .* 1e4, x = d.x))
        @test isapprox(fit.theta[2], 0.5; atol = 1e-3)
        @test isapprox(fit.loglik, fit4.loglik + 100 * log(1e4); rtol = 1e-10)
        nconv += fit.converged
        if max(fit.iterations, fit4.iterations) < 1000
            @test fit.converged == fit4.converged
        end
    end
    @test nconv >= 47

    # A truncated fit still reports unconverged: `g_tol = 1e6` stops the
    # optimiser at its starting point, far from the ML optimum of log sigma.
    Random.seed!(1)
    x = randn(100)
    y = 5 .+ 3 .* randn(100)
    @test drm(f, Gaussian(); data = (; y, x)).converged
    @test !drm(f, Gaussian(); data = (; y, x), g_tol = 1e6).converged
end

@testset "#983 _unitfree_converged is invariant to parameter units" begin
    # Well conditioned in correlation form, but the raw eigenvalue ratio is 1e-10.
    H = [1e10 0.0; 0.0 1.0]
    g = [1e-2, 1e-9]                     # Newton step 1e-12 and 1e-9; SEs 1e-5 and 1
    @test DRModels._unitfree_converged(H, g)
    D = Diagonal([1e-5, 1e3])            # reparameterise θ = D φ
    @test DRModels._unitfree_converged(D * H * D, D * g)
    # A step that is not small in SE units is still rejected after rescaling.
    @test !DRModels._unitfree_converged(H, [1e3, 0.0])
    # A genuinely singular Hessian (collinear in correlation form) still falls
    # back to the absolute rule.
    Hs = [1.0 1.0; 1.0 1.0]
    @test DRModels._unitfree_converged(Hs, [1e-6, 1e-6])
    @test !DRModels._unitfree_converged(Hs, [1.0, 1.0])
    # A saddle is not a minimum.
    @test !DRModels._unitfree_converged([1.0 0.0; 0.0 -1.0], [0.0, 0.0])
    # A non-finite (or all-zero, flat-plateau) Hessian falls back to the absolute
    # gradient rule, so the helper alone does not reject it. The routes AND a
    # finite-Hessian check, and the DrmFit constructor clears `converged` on a
    # failed-objective sentinel (nll = 1e18, so loglik = -1e18).
    @test DRModels._unitfree_converged([NaN 0.0; 0.0 1.0], [0.0, 0.0])
    @test !DRModels._unitfree_converged([NaN 0.0; 0.0 1.0], [1.0, 0.0])
    @test DRModels._unitfree_converged(zeros(2, 2), zeros(2))
    @test !DRModels._report_converged(true, -1e18, [0.0, 0.0])
end

@testset "#983 leaves the #1190 clamped-log-SD fit unconverged" begin
    # NB2 crossed fit with a random-effect log SD at the -8 clamp and a singular
    # Hessian there (#1190). Its route is not touched by #983; the fit must still
    # not count as converged.
    lines = readlines(joinpath(@__DIR__, "fixtures", "j3_se", "nb2_runoff_s7_n800", "data.csv"))
    rows = [split(l, ',') for l in lines[2:end]]
    d = (; y = [parse(Float64, r[1]) for r in rows], x = [parse(Float64, r[2]) for r in rows],
           g = String[r[3] for r in rows], h = String[r[4] for r in rows])
    fit = with_logger(NullLogger()) do
        drm(bf(@formula(y ~ 1 + x + (1 | g) + (1 | h)), @formula(sigma ~ 1)),
            NegBinomial2(); data = d)
    end
    @test any(k -> fit.theta[k] <= -8.0 + 1e-3, 4:5)
    @test !is_converged(fit)
    @test convergence_status(fit) === :boundary
end
