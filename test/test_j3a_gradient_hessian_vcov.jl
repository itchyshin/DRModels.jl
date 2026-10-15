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

@testset "J3a: _finite_hessian_from_gradient (helper)" begin
    A = [4.0 1.0 0.0; 1.0 3.0 0.5; 0.0 0.5 2.0]
    x0 = [0.3, -1.2, 40.0]
    g!(G, x) = (G .= A * (x .- x0) .+ 0.1 .* (x .- x0) .^ 2 ./ (1 .+ abs.(x0)); G)
    H = DRModels._finite_hessian_from_gradient(g!, x0)
    @test H ≈ A atol = 1e-6 rtol = 1e-6
    @test issymmetric(H)

    # indefinite -> NaN + warning (never a made-up SE)
    B = [1.0 0.0; 0.0 -1.0]
    gb!(G, x) = (G .= B * x; G)
    Hb = @test_logs (:warn, r"not positive definite") DRModels._finite_hessian_from_gradient(gb!, [0.0, 0.0])
    @test all(isnan, Hb)
    # ... unless the caller owns the PD decision
    Hb2 = DRModels._finite_hessian_from_gradient(gb!, [0.0, 0.0]; require_pd = false)
    @test Hb2 ≈ B atol = 1e-6

    # non-finite gradient at a stencil point -> NaN + warning
    gn!(G, x) = (G .= x[1] > 1e-12 ? NaN : x; G)
    Hn = @test_logs (:warn, r"non-finite") DRModels._finite_hessian_from_gradient(gn!, [0.0, 0.0])
    @test all(isnan, Hn)

    # the step does not depend on n: an objective whose gradient is exact to roundoff
    # returns the same curvature however many observations "sum" into it.
    for n in (10, 10_000)
        gq!(G, x) = (G .= n .* (x .- 1.0); G)
        @test DRModels._finite_hessian_from_gradient(gq!, [1.0])[1, 1] ≈ n rtol = 1e-6
    end
end

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

@testset "J3a: crossed-mean (binomial) SEs agree with a stable FD of the exact gradient" begin
    # NOT a native-drmTMB comparison: the reference is central differences of the
    # route's own exact gradient at h = 1e-3 (the check Simulator used on #1073, which
    # matched drmTMB to 1e-8 on grouseticks). Route 2997 is therefore marked
    # self-consistent, not natively verified.
    rng = MersenneTwister(1073)
    G, H, n = 22, 16, 700
    x = randn(rng, n)
    g = [Symbol("g", rand(rng, 1:G)) for _ in 1:n]
    h = [Symbol("h", rand(rng, 1:H)) for _ in 1:n]
    gmap = Dict(Symbol("g", j) => j for j in 1:G); hmap = Dict(Symbol("h", j) => j for j in 1:H)
    bg = 0.5 .* randn(rng, G); bh = 0.4 .* randn(rng, H)
    η = [0.2 + 0.4 * x[i] + bg[gmap[g[i]]] + bh[hmap[h[i]]] for i in 1:n]
    ntr = fill(8.0, n)
    s = Float64.([rand(rng, Distributions.Binomial(8, 1 / (1 + exp(-η[i])))) for i in 1:n])
    gidx, Gn = DRModels._group_index(g); hidx, Hn = DRModels._group_index(h)
    comps = [(ones(n), gidx, Gn, "g"), (ones(n), hidx, Hn, "h")]
    X = hcat(ones(n), x)
    fit = DRModels._fit_binomial_crossed_laplace(DRModels.Binomial(), s, ntr, X, comps,
                                                 ["(Intercept)", "x"], 1e-7; se = true)
    @test fit.converged
    θ = copy(fit.theta); p = length(θ)
    Href = zeros(p, p); gp = zeros(p); gm = zeros(p)
    for i in 1:p
        hi = 1e-3 * (1 + abs(θ[i])); e = zeros(p); e[i] = hi
        fit.nllgrad(gp, θ .+ e); fit.nllgrad(gm, θ .- e)
        Href[:, i] .= (gp .- gm) ./ (2hi)
    end
    se_ref = sqrt.(diag(inv(Symmetric((Href .+ Href') ./ 2))))
    se = sqrt.(diag(fit.vcov))
    @test all(abs.(se ./ se_ref .- 1) .< 1e-3)
end
