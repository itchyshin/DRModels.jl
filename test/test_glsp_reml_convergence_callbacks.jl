# The REML fitter profiles the fixed effects and latent mode, then optimizes only
# the phylogenetic covariance coordinates. Its convergence diagnostics must use
# that restricted objective and active-coordinate score, while profiles retain
# the ordinary ML objective. A non-unit tree height also exercises the raw/unit
# branch-length transformation in `_glsp_unscale_fit`.
using DRModels
using Test, Random, LinearAlgebra

@testset "Gaussian sigma-phylo REML keeps its profiled objective and score" begin
    Random.seed!(404)
    p = 24; m = 6; n = p * m
    phy = random_balanced_tree(p; branch_length=0.3)
    C = sigma_phy_dense(phy; σ²_phy=1.0)
    LC = cholesky(Symmetric(C)).L
    u_mu = 0.6 .* (LC * randn(p))
    u_sig = 0.5 .* (LC * randn(p))
    species = repeat(1:p, inner=m)
    y = [1.0 + u_mu[species[i]] + exp(log(0.5) + u_sig[species[i]]) * randn() for i in 1:n]
    data = (; y, species)
    form = bf(@formula(y ~ phylo(1 | species)),
              @formula(sigma ~ phylo(1 | species)))

    @test !isapprox(phylo_tree_height(phy), 1.0; atol=1e-12)
    fits = [
        drm(form, Gaussian(); data, tree=phy, method=:REML),
        drm(form, Gaussian(); data, tree=phy, method=:REML, phylo_coupled=true),
        drm(bf(@formula(y ~ 1), @formula(sigma ~ phylo(1 | species))),
            Gaussian(); data, tree=phy, method=:REML),
    ]

    for fit in fits
        # `drm` rescales internally then reports on the input covariance scale;
        # `phylo_scale` is a symbolic label, not the numeric tree height.
        @test fit.phylo_scale === :covariance
        @test fit.nllgrad !== nothing
        @test fit.nll isa DRModels._FitObjectiveGradient
        if !(fit.nll isa DRModels._FitObjectiveGradient)
            continue
        end
        objective = fit.nll.objective
        score! = fit.nll.gradient!
        @test isfinite(objective(fit.theta))
        @test objective(fit.theta) ≈ -reml_loglik(fit) atol=1e-8

        score = zeros(length(fit.theta))
        score!(score, fit.theta)
        @test all(isfinite, score)
        # Beta is profiled, not an optimizer coordinate on this REML route.
        @test all(iszero, score[1:2])
        h = 1e-5
        fdscore = zeros(length(fit.theta))
        for j in 3:length(fit.theta)
            tp = copy(fit.theta); tp[j] += h
            tm = copy(fit.theta); tm[j] -= h
            fdscore[j] = (objective(tp) - objective(tm)) / (2h)
        end
        @test score[3:end] ≈ fdscore[3:end] atol=1e-6 rtol=1e-4
        if convergence_status(fit) === :converged
            @test maximum(abs, score[3:end]) <=
                  DRModels._TWIN_RELATIVE_GRADIENT_TOLERANCE * max(1.0, abs(objective(fit.theta)))
        end
        @test isfinite(fit.nll.profile_objective(fit.theta))
    end
    @test is_converged(fits[1])
    @test convergence_status(fits[1]) === :converged
end
