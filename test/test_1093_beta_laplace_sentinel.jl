# #1093: Beta `(1 | g)` `marginal = :Laplace` returned the -1e18 failed-fit
# placeholder (converged = false) on an interior optimum. Cause: the shared
# `_phylo_mean_mode` backtracking line search (used when a data-term weight is
# negative, as for Beta) rejected every step once the true decrease fell below the
# floating-point resolution of the joint objective, and failed closed at the mode.
# The same failure inside a finite-difference Hessian stencil gave NaN SEs.
#
# Data are simulated in Julia (the original drmTMB CSV is not in the repo); the
# reference for the optimum is the GHQ default route, which fits the same model.
using DRModels
using Test, Random, LinearAlgebra, Logging
import Distributions

function _sim_beta_1093(seed; G = 20, n = 400, nsq = 0, phi = 3.0, b0 = 0.3, b1 = 0.8, sdu = 0.5)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = n ÷ G)
    x = randn(rng, n)
    u = sdu .* randn(rng, G)
    mu = 1 ./ (1 .+ exp.(-(b0 .+ b1 .* x .+ u[g])))
    y = [rand(rng, Distributions.Beta(m * phi, (1 - m) * phi)) for m in mu]
    if nsq > 0
        idx = randperm(rng, n)[1:nsq]
        y[idx[1:nsq÷2]] .= 1e-6
        y[idx[nsq÷2+1:end]] .= 1 - 1e-6
    end
    return (; y = clamp.(y, 1e-6, 1 - 1e-6), x, g = string.(g))
end

_fit_beta_1093(d; kw...) = drm(bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1)),
                               DRModels.Beta(); data = d, kw...)

@testset "#1093 Beta :Laplace near-boundary responses" begin
    # Seeds 1, 2, 5, 6 returned loglik = -1e18, converged = false on the base commit.
    for seed in (1, 2, 5, 6)
        d = _sim_beta_1093(seed; nsq = 40)
        f = _fit_beta_1093(d; marginal = :Laplace)
        @test loglik(f) > -1e15
        @test is_converged(f)
        @test isfinite(aic(f)) && isfinite(deviance(f))
        # same model on the GHQ default route: the optima agree to Laplace-vs-GHQ error
        gh = _fit_beta_1093(d)
        @test abs(loglik(f) - loglik(gh)) < 0.5
        # Seed 5 lands on the RE-variance boundary (log sd = -15.7): its Wald SEs are the
        # vcov guard's concern (J3), so the SE check covers the interior fits only.
        seed == 5 || @test all(isfinite, sqrt.(diag(vcov(f))))
    end
end

@testset "#1093 Beta :Laplace SEs on interior data (Wave K route)" begin
    # Seeds whose finite-difference Hessian stencil hit the placeholder on the base
    # commit (NaN vcov although the point estimate was fine).
    for seed in (56, 64, 70, 87)
        d = _sim_beta_1093(seed; G = 30, n = 600, phi = 20.0, b0 = 0.2, b1 = 0.5)
        f = with_logger(NullLogger()) do
            _fit_beta_1093(d; marginal = :Laplace)
        end
        @test is_converged(f)
        se = sqrt.(diag(vcov(f)))
        @test all(isfinite, se) && all(>(0), se)
        # independent check of the SEs: Hessian by central differences of the analytic
        # gradient (a different estimator from the objective-difference stencil)
        θ̂ = f.theta
        Hg = DRModels._ordinary_laplace_grad_hessian(f.nllgrad, θ̂)
        @test se ≈ sqrt.(diag(inv(Symmetric(Hg)))) rtol = 0.02
    end
end

@testset "#1093 sentinel fits: no raw -1e18 from deviance/aic/bic" begin
    d = _sim_beta_1093(3; nsq = 40)
    f = _fit_beta_1093(d; marginal = :Laplace)
    vals = Any[getfield(f, k) for k in fieldnames(typeof(f))]
    vals[findfirst(==(:loglik), fieldnames(typeof(f)))] = -1e18
    bad = typeof(f).name.wrapper(vals...)
    for fn in (aic, bic, deviance)
        v = @test_logs (:warn, r"sentinel") match_mode = :any fn(bad)
        @test isnan(v)
    end
end

@testset "#1093 genuinely non-PD Hessian: NaN SEs with a warning, never silent" begin
    H = [1.0 0.0; 0.0 -1.0]           # indefinite: a saddle, not a minimum
    V = @test_logs (:warn,) DRModels._vcov_from_fd_hessian(H; context = "1093")
    @test all(isnan, V)
end
