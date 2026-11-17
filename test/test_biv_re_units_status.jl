# Two properties of the bivariate Gaussian `(1 | p | g)` route that must not depend on how the data are
# written down.
#   1. Units: multiplying both responses by a constant leaves the correlations (residual and group)
#      unchanged and moves every scaled estimate by exactly that constant.
#   2. A fit of the mean pair at a bound is not called converged: a group SD at zero, or a group
#      correlation at plus or minus one, gets `:boundary` or `:degenerate` from `convergence_status`,
#      never `:converged`. (The group SDs count for the SD rule; a group correlation at one is caught here
#      through the covariance and standard-error rules, as there is no separate rule for it.)
using DRModels
using Test, LinearAlgebra
using StableRNGs

include("biv_ordinary_re_helpers.jl")       # _bre_data, _BRE_F0, _bre_summ (shared, guarded)

@testset "bivariate (1 | p | g): rescaling both responses leaves correlations and scaled estimates unchanged" begin
    d = _bre_data("interior_bal.csv")
    n = length(d.y1)
    f0 = drm(_BRE_F0, Gaussian(); data = d)
    s0 = _bre_summ(f0.theta[8:10])               # (log SD1, log SD2, group correlation) from the log-Cholesky entries
    for k in (250.0, 0.004)
        dk = merge(d, (; y1 = k .* d.y1, y2 = k .* d.y2))
        fk = drm(_BRE_F0, Gaussian(); data = dk)
        sk = _bre_summ(fk.theta[8:10])
        @test fk.converged && convergence_status(fk) === :converged
        # logLik of the rescaled data is lower by one log k per observed cell (2n cells)
        @test fk.loglik ≈ f0.loglik - 2n * log(k) atol = 1e-6
        # fixed effects of the two means scale with k
        @test fk.theta[1:4] ≈ k .* f0.theta[1:4] rtol = 1e-6
        # log residual SDs shift by log k, the residual correlation coefficient does not move
        @test fk.theta[5:6] ≈ f0.theta[5:6] .+ log(k) atol = 1e-6
        @test fk.theta[7] ≈ f0.theta[7] atol = 1e-6
        # group SDs shift by log k, the group correlation is unchanged
        @test sk[1:2] ≈ s0[1:2] .+ log(k) atol = 1e-6
        @test sk[3] ≈ s0[3] atol = 1e-6
        # the group covariance scales by k^2
        @test fk.ranef.Sigma_g ≈ k^2 .* f0.ranef.Sigma_g rtol = 1e-6
    end
end

@testset "bivariate (1 | p | g): a fit at a bound is not called converged" begin
    function sim(seed; group_sd, rho_z)
        rng = StableRNG(seed)
        G = 8; per = 5; n = G * per
        g = ["g$(i)" for i in repeat(1:G, inner = per)]
        x1 = randn(rng, n); x2 = randn(rng, n)
        u = randn(rng, G)[repeat(1:G, inner = per)]
        e1 = randn(rng, n); e2 = randn(rng, n)
        y1 = 0.5 .+ 0.3 .* x1 .+ group_sd .* u .+ e1
        y2 = -0.2 .+ 0.2 .* x2 .+ rho_z .* group_sd .* u .+ 0.8 .* e2
        return (; g, x1, x2, y1, y2)
    end
    # no group variation in the data: the fitted group SDs collapse to zero
    fit0 = drm(_BRE_F0, Gaussian(); data = sim(12; group_sd = 0.0, rho_z = 1.0))
    sd0 = sqrt.(diag(fit0.ranef.Sigma_g))
    @test all(<(1e-4), sd0)
    @test convergence_status(fit0) in (:boundary, :degenerate)
    @test !is_converged(fit0)
    # one group effect shared (up to scale) by the two responses: the group correlation sits at +1
    fit1 = drm(_BRE_F0, Gaussian(); data = sim(7; group_sd = 0.5, rho_z = 1.0))
    S = fit1.ranef.Sigma_g
    @test abs(S[1, 2] / sqrt(S[1, 1] * S[2, 2])) > 0.9999
    @test convergence_status(fit1) in (:boundary, :degenerate)
    @test !is_converged(fit1)
end
