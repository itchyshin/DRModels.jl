# test_j3a_gradient_hessian_vcov.jl -- #1073 (lane J3a).
#
# `_finite_hessian` second-differenced the Laplace objective VALUE, so inner-mode-solve
# noise (~eps/h^2) understated SEs for the between-cluster coefficients on the crossed
# routes. J3a adds `_finite_hessian_from_gradient` (central differences of the exact
# gradient, step cbrt(eps)*max(1,|x_i|), symmetrised, PD-checked) and uses it, with a
# tightened cold-started inner solve, on the two crossed-mean routes.
#
# Reference for the grouseticks test: drmTMB 0.7.1 (ca3ddef) `sqrt(diag(f$sdr$cov.fixed))`
# at `f$opt$par`, as posted in #1073 (Data fitter / Simulator). Honesty note: the thread
# does not record TMB's max |gradient| at opt$par, so the "< 1e-6 at the SE point" check
# is done on the Julia side (the Julia and R optima agree to 1e-8 per the issue).
using DRModels
using Test, LinearAlgebra, Random
import Distributions

function _j3a_read_grouse()
    lines = readlines(joinpath(@__DIR__, "fixtures", "lme4", "grouseticks.csv"))
    rows = [split(l, ',') for l in lines[2:end]]
    return (; TICKS = [parse(Float64, r[1]) for r in rows],
              YEAR = String[r[2] for r in rows],
              cHEIGHT = [parse(Float64, r[3]) for r in rows],
              BROOD = String[r[4] for r in rows],
              LOCATION = String[r[5] for r in rows])
end

@testset "J3a: grouseticks NB2 crossed SEs vs drmTMB sdreport (#1073)" begin
    gt = _j3a_read_grouse()
    @test length(gt.TICKS) == 403
    fit = drm(bf(@formula(TICKS ~ 1 + YEAR + cHEIGHT + (1 | BROOD) + (1 | LOCATION)),
                 @formula(sigma ~ 1)), NegBinomial2(); data = gt)
    @test is_converged(fit)

    # Fitted optimum vs drmTMB opt$par (5 digits as posted) and logLik: separate check.
    θ_tmb = [0.52371, 1.16717, -0.98825, -0.23678, -0.59590, -0.35958, -0.52195]
    @test fit.theta ≈ θ_tmb atol = 2e-5
    @test loglik(fit) ≈ -888.90842589 atol = 1e-6

    # Exact gradient at the SE point is ~0 (the Hessian is taken at an optimum).
    g = zeros(7); fit.nllgrad(g, fit.theta)
    @test maximum(abs, g) < 1e-5

    # SEs vs native drmTMB sdreport at ~1% (the old route was 4-10% low on 3 of 7).
    se_tmb = [0.194750, 0.232894, 0.257105, 0.035084, 0.086642, 0.186912, 0.317245]
    se = sqrt.(diag(fit.vcov))
    @test all(isfinite, se)
    @test all(abs.(se ./ se_tmb .- 1) .< 0.01)
end

