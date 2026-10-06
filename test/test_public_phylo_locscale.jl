# test_public_phylo_locscale.jl — #202 closeout: PUBLIC `drm()` phylogenetic
# location–scale via grammar B `(1 | p | phylo(species))` on both axes.
#
# Complements `test_phylo_locscale.jl` (private `_fit_locscale` + Gamma) and
# `test_locscale_frontend.jl` (public iid `(1 | p | species)`). This file gates
# the missing tip gap: family `drm()` must forward `tree=` into the locscale
# frontend so structured phylo Q is used.
#
# Anchors:
#   1. NB2 recovery of mean slope + mean-axis SD + SCALE-axis SD (headline).
#   2. Gamma public route smoke (private Gamma recovery already in #253).
#   3. Dual issue-text `phylo(1|sp)` on both axes still rejected (grammar A out).
#
# NB2 ψ convention (locscale_kernels.jl): ψ = log σ, size r = exp(−2ψ).
# Draws MUST use that mapping — `r = exp(ψ)` is the pre-unification convention
# and collapses scale-axis recovery on public+private paths alike.
using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions

_public_phylo_nb2_draw(ημ, ηψ) = (r = exp(-2 * ηψ); μ = exp(ημ);
                                  Float64(rand(Distributions.NegativeBinomial(r, r / (r + μ)))))
_public_phylo_gamma_draw(ημ, ηψ) = (α = exp(ηψ); μ = exp(ημ);
                                        rand(Distributions.Gamma(α, μ / α)))

@testset "public phylo location–scale (#202 closeout)" begin

    @testset "NB2 grammar B: recovery of μ- and σ-axis structure" begin
        Random.seed!(7)                           # locked: recovers under ψ = log σ
        p = 40; m = 20; n = p * m                 # m ≥ 2 (scale-RE identifiability)
        phy = random_balanced_tree(p; branch_length = 0.30)
        C = sigma_phy_dense(phy; σ²_phy = 1.0)
        LC = cholesky(Symmetric(C)).L

        sd_mu_true = 0.50
        sd_psi_true = 0.50
        cor_true = 0.30
        Λtrue = [sd_mu_true^2                    cor_true * sd_mu_true * sd_psi_true;
                 cor_true * sd_mu_true * sd_psi_true   sd_psi_true^2]
        LΛ = cholesky(Symmetric(Λtrue)).L
        A = LC * randn(p, 2) * LΛ'

        species = repeat(1:p, inner = m)
        x = randn(n)
        βμ = [0.30, 0.45]
        βψ = [-0.35]                              # log σ intercept (r = exp(0.70))
        y = [_public_phylo_nb2_draw(βμ[1] + βμ[2] * x[i] + A[species[i], 1],
                                    βψ[1] + A[species[i], 2]) for i in 1:n]
        data = (; y, x, species)

        fit = drm(bf(@formula(y ~ x + (1 | p | phylo(species))),
                     @formula(sigma ~ 1 + (1 | p | phylo(species)))),
                  NegBinomial2(); data = data, tree = phy, se = false, g_tol = 1e-6)

        Λ = vc(fit)[:species]
        sd_mu = sqrt(Λ[1, 1])
        sd_psi = sqrt(Λ[2, 2])
        cor_mu_psi = Λ[1, 2] / (sd_mu * sd_psi)

        @test isfinite(loglik(fit))
        @test isposdef(Symmetric(Λ))
        @test coef(fit, :mu)[2] ≈ βμ[2] atol = 0.15
        @test sd_mu ≈ sd_mu_true atol = 0.22
        @test sd_psi ≈ sd_psi_true atol = 0.25
        @test sd_psi > 0.18
        @test isfinite(cor_mu_psi) && -1.0 ≤ cor_mu_psi ≤ 1.0
        # Named group-level summary path (never residual rho12).
        @test fit.nll isa DRModels.LocScaleObjective
    end

    @testset "Gamma grammar B: public route reaches phylo locscale" begin
        Random.seed!(7)
        p = 16; m = 6; n = p * m
        phy = random_balanced_tree(p; branch_length = 0.25)
        C = sigma_phy_dense(phy; σ²_phy = 1.0)
        LC = cholesky(Symmetric(C)).L
        A = LC * randn(p, 2) * Diagonal([0.45, 0.40])
        species = repeat(1:p, inner = m)
        x = randn(n)
        y = [_public_phylo_gamma_draw(0.2 + 0.35 * x[i] + A[species[i], 1],
                                      0.6 + A[species[i], 2]) for i in 1:n]
        data = (; y, x, species)

        fit = drm(bf(@formula(y ~ x + (1 | p | phylo(species))),
                     @formula(sigma ~ 1 + (1 | p | phylo(species)))),
                  DRModels.Gamma(); data = data, tree = phy, se = false, g_tol = 1e-5)

        Λ = vc(fit)[:species]
        @test size(Λ) == (2, 2)
        @test isfinite(loglik(fit))
        @test sqrt(Λ[2, 2]) > 0.05
        @test fit.nll isa DRModels.LocScaleObjective
    end

    @testset "grammar A dual phylo(1|sp) still rejected on NB2" begin
        Random.seed!(3)
        p = 6; m = 3; n = p * m
        phy = random_balanced_tree(p)
        species = repeat(1:p, inner = m)
        x = randn(n); y = Float64.(rand(0:5, n))
        data = (; y, x, species)
        @test_throws Exception drm(
            bf(@formula(y ~ x + phylo(1 | species)),
               @formula(sigma ~ 1 + phylo(1 | species))),
            NegBinomial2(); data = data, tree = phy, se = false)
    end
end
