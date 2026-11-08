# Regression: Gaussian ordinary correlated q2 `(1 + x | g)` SD regression lost fits to
# `AssertionError: isfinite(phi_c) && isfinite(dphi_c)` (LineSearches HagerZhang).
#
# Cause: the first L-BFGS step is the raw gradient (length ~1e2-1e3, the objective is a sum
# over rows), so the line search probes points where `exp(-2 eta_sigma)` and the group
# information matrix overflow; the Cholesky of an Inf/NaN matrix "succeeds" and the
# objective (or its AD gradient) is NaN. Fix: probes with a non-finite value or gradient
# map to the 1e18 sentinel inside the optimiser only (`_optimize_with_fallback`), with a
# BackTracking retry.
#
#  1. a platform-independent reproduction of the failure on a toy objective;
#  2. two campaign datasets that threw before the fix, fitted against native drmTMB 0.8.1.
using DRModels, Test, Random, LinearAlgebra
const Optim = DRModels.Optim   # Optim is a dependency of DRModels, not of the test project

@testset "line-search guard: toy objective, sentinel then NaN probes" begin
    # Quadratic with its minimum at (1, 1), started where the raw gradient (-100, -100)
    # throws the first probe far away. Far probes hit a finite 1e18 sentinel (the non-PD
    # branch of the real objective), nearer ones NaN (the overflow branch), as in the
    # failing fits: the bisection after the sentinel lands in the NaN zone and HagerZhang
    # asserts.
    function f(x)
        r = maximum(abs, x)
        r > 10 && return oftype(sum(x), 1e18)
        r > 2 && return oftype(sum(x), NaN)
        return 50 * sum(abs2, x .- 1)
    end
    θ0 = [0.0, 0.0]
    # The unguarded call is the pre-fix behaviour.
    @test_throws AssertionError Optim.optimize(f, θ0, Optim.LBFGS(),
        Optim.Options(g_tol = 1e-8); autodiff = :forward)
    res = DRModels._optimize_with_fallback(f, θ0, 1e-8)
    @test Optim.minimizer(res) ≈ [1.0, 1.0] atol = 1e-6
    # Same options as the Gaussian q2/q3/q4 route.
    res2 = DRModels._optimize_with_fallback(f, θ0, 1e-8;
        opts = Optim.Options(g_tol = 1e-8, x_abstol = NaN, x_reltol = NaN,
                             f_abstol = NaN, f_reltol = NaN))
    @test Optim.minimizer(res2) ≈ [1.0, 1.0] atol = 1e-6
    # A finite, well-behaved objective is untouched: guarded and plain runs coincide.
    g(x) = sum(abs2, x .- [1.0, -2.0]) + 0.1 * x[1] * x[2]
    plain = Optim.optimize(g, θ0, Optim.LBFGS(), Optim.Options(g_tol = 1e-8); autodiff = :forward)
    guarded = DRModels._optimize_with_fallback(g, θ0, 1e-8)
    @test Optim.minimizer(plain) == Optim.minimizer(guarded)
    @test Optim.minimum(plain) == Optim.minimum(guarded)
end

# Fixture layout: line 1 = z0 (150 group values), line 2 = z1, then one `y,x` row per
# observation (150 groups x 10 rows, group label g<ceil(row / 10)>).
function _lsg_read(seed)
    lines = readlines(joinpath(@__DIR__, "fixtures", "gaussian_ordinary_q2_linesearch",
                               "seed$(seed).csv"))
    z0g = parse.(Float64, split(lines[1], ","))
    z1g = parse.(Float64, split(lines[2], ","))
    cols = [parse.(Float64, split(l, ",")) for l in lines[3:end]]
    y = [c[1] for c in cols]; x = [c[2] for c in cols]
    gi = [cld(i, 10) for i in eachindex(y)]
    return (; y, x, z0 = z0g[gi], z1 = z1g[gi], g = ["g$(k)" for k in gi])
end

# Native drmTMB 0.8.1: logLik, coefficients (mu: 2, sigma: 1, sd(intercept): 2, sd(x): 2,
# atanh rho) and standard errors.
const _LSG_REF = Dict(
    179 => (ll = -1147.892103505830,
            est = [0.260372018309142, 0.501495834053336, -0.943024469678685,
                   -0.404024143893304, 0.441054071692155, -0.808707533090891,
                   -0.431948503258894, 0.260873206486163],
            se = [0.0492620183338752, 0.0349182699081055, 0.0204040103770504,
                  0.0608927012412733, 0.0684345766690861, 0.0663972137840717,
                  0.0829039662295016, 0.0893035719767287]),
    244 => (ll = -1148.506977218764,
            est = [0.275175884648277, 0.506752138853266, -0.933487156894081,
                   -0.455927772216004, 0.437911451707439, -0.831412966217391,
                   -0.318127807958732, 0.334439784583212],
            se = [0.04666808075247, 0.0355215821546863, 0.0204133011260036,
                  0.0611403804727113, 0.0675463183900702, 0.0651209842446531,
                  0.0720740243740236, 0.0899822073419369]))

@testset "Gaussian ordinary q2 SD regression survives the first-step overflow (seed $seed)" for seed in (179, 244)
    d = _lsg_read(seed)
    ref = _LSG_REF[seed]
    f = bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1),
           @formula(sd(g, coef(intercept)) ~ z0), @formula(sd(g, coef(x)) ~ z1))
    fit = drm(f, Gaussian(); data = d)
    @test fit.converged
    @test isapprox(loglikelihood(fit), ref.ll; atol = 1e-8)
    th = fit.theta
    rho_julia = th[end] / sqrt(1 + th[end]^2)       # Julia stores the Cholesky angle
    @test isapprox(rho_julia, tanh(ref.est[end]); atol = 1e-7)
    @test maximum(abs.(th[1:end-1] .- ref.est[1:end-1]) ./ ref.se[1:end-1]) < 1e-6
end
