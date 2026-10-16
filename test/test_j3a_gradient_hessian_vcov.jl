# test_j3a_gradient_hessian_vcov.jl -- #1073 (lane J3a).
#
# `_finite_hessian` second-differenced the Laplace objective VALUE, so inner-mode-solve
# noise (~eps/h^2) understated SEs for the between-cluster coefficients on the crossed
# routes. J3a adds `_finite_hessian_from_gradient` (central differences of the exact
# gradient, step cbrt(eps)*max(1,|x_i|), symmetrised, PD-checked) and uses it, with a
# tightened cold-started inner solve, on the two crossed-mean routes.
#
# Reference for the grouseticks test: drmTMB 0.7.1 (ca3ddef) `sqrt(diag(f$sdr$cov.fixed))`
# at `f$opt$par`, as posted in #1073 (Data fitter / Simulator) and re-run locally on the same
# data: for that reference fit `fit$opt$convergence == 0`, `fit$sdr$pdHess == TRUE` (drmTMB
# exposes TMB's `pdHess`, the `pd.hess` flag) and max |gradient| at `opt$par` = 1.5e-10.
# The issue thread does not record TMB's max |gradient| itself, so the "< 1e-6 at the SE
# point" check is also done on the Julia side (the Julia and R optima agree to 1e-8).
using DRModels
using Test, LinearAlgebra, Random, Logging
import Distributions, TOML

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

@testset "J3a: crossed-mean (binomial) SEs vs an independent ForwardDiff Hessian" begin
    # Reference: test/j3_dense_reference.jl -- dense Laplace marginal, Float64 mode carried in
    # Dual arithmetic, Hessian by ForwardDiff. It shares no code with the route and differences
    # neither a gradient nor a value. (Native drmTMB agreement for this route is in J3b.)
    include("j3_dense_reference.jl")
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
    θ = copy(fit.theta)
    spec = DenseSpec(2, X, hcat(indicator(gidx, Gn), indicator(hidx, Hn)), obs_binomial(s, ntr),
                     prior_iid([Gn, Hn], [3, 4]))
    # the dense objective is the route's objective up to a constant
    θ2 = θ .+ [0.01, -0.02, 0.03, -0.02]
    @test dense_laplace_nll(θ2, spec) - dense_laplace_nll(θ, spec) ≈ fit.nll(θ2) - fit.nll(θ) atol = 1e-6
    se_ref = dense_laplace_se(θ, spec)
    se = sqrt.(diag(fit.vcov))
    @test all(abs.(se ./ se_ref .- 1) .< 1e-4)
end

@testset "J3a: which Hessian is used when a coordinate is at its clamp (review fixes)" begin
    pμ = 4
    # The log-dispersion's REAL bound is the family's: NB2 / Gamma / Beta clamp `-2 log sigma`
    # to [-8, 8] inside `aux_from`, i.e. log sigma to [-4, 4]; Tweedie uses it raw ([-8, 8]).
    # A value in (3, 4) is INTERIOR for NB2 and must take the gradient path (it used to fall
    # back silently because it was checked against the log-SD box [-8, 3]).
    bnb = DRModels._dispersion_clamp(Val(:nb2_fixed))
    @test bnb == 4.0 && DRModels._dispersion_clamp(Val(:gamma_fixed)) == 4.0 &&
          DRModels._dispersion_clamp(Val(:beta_fixed)) == 4.0
    @test DRModels._dispersion_clamp(Val(:tweedie_fixed)) == 8.0
    @test DRModels._dispersion_clamp(Val(:betabinomial_fixed)) == 4.0
    @test_throws ArgumentError DRModels._dispersion_clamp(Val(:poisson_unregistered))
    path(θ; b = bnb) = DRModels._se_hessian_path(θ; disp = (pμ + 1,), sd = (pμ + 2):(pμ + 3), disp_bound = b)
    θ = [0.1, 0.2, 0.3, 0.4, 3.5, -0.3, -0.5]
    @test path(θ) === :gradient                    # theta_sigma_hat = 3.5 > 3, inside NB2's [-4, 4]
    θ[pμ + 1] = 3.9995                             # within tol of NB2's upper clamp
    @test path(θ) === :value
    θ[pμ + 1] = 5.0                                # outside NB2's box (flat), inside Tweedie's
    @test path(θ) === :value
    @test path(θ; b = DRModels._dispersion_clamp(Val(:tweedie_fixed))) === :gradient
    θ[pμ + 1] = 3.5; θ[pμ + 2] = 3.0               # a log-SD at ITS upper clamp
    @test path(θ) === :value
    θ[pμ + 2] = -0.3; θ[pμ + 3] = -8.0             # a log-SD at its lower clamp
    @test path(θ) === :value

    # The fallback is not silent.
    @test_logs (:warn, r"falling back to the value-based finite-difference Hessian") DRModels._warn_value_hessian("test")

    # End to end: a real NB2 crossed fit whose RE SD collapses to its clamp must warn and
    # still return a finite or NaN vcov of the right shape (never throw).
    rng = MersenneTwister(1073)
    n = 300
    a = ["a$(rand(rng, 1:30))" for _ in 1:n]; b = ["b$(rand(rng, 1:20))" for _ in 1:n]
    x = randn(rng, n)
    y = Float64.([rand(rng, Distributions.NegativeBinomial(3.0, 3.0 / (3.0 + exp(0.3 + 0.4 * x[i])))) for i in 1:n])
    fit = @test_logs (:warn, r"falling back to the value-based finite-difference Hessian") match_mode = :any drm(
        bf(@formula(y ~ x + (1 | a) + (1 | b)), @formula(sigma ~ 1)), NegBinomial2();
        data = (; y, x, a, b))
    @test size(vcov(fit)) == (length(fit.theta), length(fit.theta))
    @test minimum(fit.theta[end-1:end]) < -7.9    # the collapse that triggered the branch

    # The branch decision above is only right if the exact gradient really is valid for
    # log-dispersion in (3, 4): at theta_sigma = 3.5 (CV ~ 33) the stored gradient must agree with
    # central differences of the stored objective. When this check was written, a real NB2 crossed
    # fit that far out ran off with converged = false (#1088), so this point is evaluated rather
    # than fitted; the real fit follows below.
    gt = _j3a_read_grouse()
    gfit = drm(bf(@formula(TICKS ~ 1 + YEAR + cHEIGHT + (1 | BROOD) + (1 | LOCATION)),
                  @formula(sigma ~ 1)), NegBinomial2(); data = gt)
    θ5 = copy(gfit.theta); θ5[5] = 3.5
    g5 = zeros(7); gfit.nllgrad(g5, θ5)
    fd5 = [(gfit.nll(θ5 .+ 1e-4 .* (1:7 .== k)) - gfit.nll(θ5 .- 1e-4 .* (1:7 .== k))) / 2e-4 for k in 1:7]
    @test g5[5] != 0                                 # not flat: the objective is not clamped at 3.5
    @test maximum(abs.(g5 .- fd5)) < 1e-3 * max(1.0, maximum(abs.(g5)))

    # ... and a REAL fit that lands there (#1088). Fixture nb2_disp3to4_s1_n5000 (size 0.0015,
    # RE SD 1.2, 6 x 6 levels; test/parity/gen_1088_nb2_runoff_*): native drmTMB ca3ddef converges
    # (opt$convergence 0, sdr$pdHess, log sigma 3.203, both RE SDs interior), so the gradient path
    # is the one taken and its SEs have a reference. It needs both #1088 changes: the first pass
    # leaves the clamp box (log sigma 2574) and is recovered by the box re-fit, and the cold
    # 1e-12 SE inner solve stalls on its rounding floor at a stencil point (NaN vcov) unless the
    # tolerance ladder retries it at 1e-10 / 1e-8.
    dir34 = joinpath(@__DIR__, "fixtures", "j3_se", "nb2_disp3to4_s1_n5000")
    ref34 = TOML.parsefile(joinpath(dir34, "ref.toml"))
    rows34 = [split(l, ',') for l in readlines(joinpath(dir34, "data.csv"))[2:end]]
    d34 = (; y = [parse(Float64, r[1]) for r in rows34], x = [parse(Float64, r[2]) for r in rows34],
             g = String[r[3] for r in rows34], h = String[r[4] for r in rows34])
    @test ref34["opt_convergence"] == 0 && ref34["pd_hess"] && ref34["comparable"]
    # no warning at all: no clamp, no value-Hessian fallback, no NaN vcov, no run-off
    fit34 = @test_logs min_level = Logging.Warn drm(
        bf(@formula(y ~ 1 + x + (1 | g) + (1 | h)), @formula(sigma ~ 1)), NegBinomial2(); data = d34)
    θ34 = fit34.theta
    @test is_converged(fit34)
    @test 3 < θ34[3] < 4                             # theta_sigma_hat in (3, 4), fitted
    @test DRModels._se_hessian_path(θ34; disp = (3,), sd = 4:5, disp_bound = bnb) === :gradient
    @test maximum(abs.(θ34 .- ref34["theta"])) < 2e-4
    @test loglik(fit34) ≈ ref34["loglik"] atol = 1e-5
    g34 = zeros(5); fit34.nllgrad(g34, θ34)
    @test maximum(abs, g34) < 1e-4
    se34 = sqrt.(diag(vcov(fit34)))
    @test all(isfinite, se34)
    @test all(abs.(se34 ./ ref34["se"] .- 1) .< 0.01)   # vs drmTMB sdreport
end
