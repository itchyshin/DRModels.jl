using DRModels
using Test, LinearAlgebra, Random

# Reuse the existing coevolution-test DGP (test_gaussian_bivariate_phylo.jl): a
# q=4 phylogenetic bivariate location–scale model with a known 4×4 among-axis
# Σ_a. Defined locally so this file runs standalone.

const _COEVO_BETA = (
    mu1 = [1.0, 0.5],
    mu2 = [-0.3, 0.4],
    s1 = [-0.4],
    s2 = [-0.5],
    rho = [0.3],
)

const _COEVO_SIGMA_A = Matrix(Symmetric([
    0.25 0.10 0.05 0.00
    0.10 0.25 0.00 0.04
    0.05 0.00 0.09 0.02
    0.00 0.04 0.02 0.09
]))

function _coevo_data(; p::Int = 10, nrep::Int = 3, seed::Int = 188)
    rng = MersenneTwister(seed)
    phy = random_balanced_tree(p; branch_length = 0.2)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    Q_cond = phy.Q_topology[keep, keep]
    P = prior_precision(Q_cond, inv(_COEVO_SIGMA_A))
    F = cholesky(Symmetric(P))
    u_aug = F.UP \ randn(rng, size(P, 1))
    pos = Dict(node => i for (i, node) in enumerate(keep))
    leaf_pos = [pos[phy.leaf_indices[k]] for k in 1:p]

    species_idx = repeat(1:p, inner = nrep)
    species = [phy.leaf_names[k] for k in species_idx]
    n = length(species_idx)
    x = randn(rng, n)
    y1 = Vector{Float64}(undef, n)
    y2 = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        k = species_idx[i]
        u = @view u_aug[(4 * (leaf_pos[k] - 1) + 1):(4 * leaf_pos[k])]
        m1 = _COEVO_BETA.mu1[1] + _COEVO_BETA.mu1[2] * x[i] + u[1]
        m2 = _COEVO_BETA.mu2[1] + _COEVO_BETA.mu2[2] * x[i] + u[2]
        s1 = exp(_COEVO_BETA.s1[1] + u[3])
        s2 = exp(_COEVO_BETA.s2[1] + u[4])
        ρ = DRModels.RHO_GUARD * tanh(_COEVO_BETA.rho[1])
        e = cholesky(Symmetric([s1^2 ρ*s1*s2; ρ*s1*s2 s2^2])).L * randn(rng, 2)
        y1[i] = m1 + e[1]
        y2[i] = m2 + e[2]
    end
    return (; data = (; y1, y2, x, species), phy)
end

_coevo_formula() = bf(
    mu1 = @formula(y1 ~ x + phylo(1 | species)),
    mu2 = @formula(y2 ~ x + phylo(1 | species)),
    sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
    sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
    rho12 = @formula(rho12 ~ 1),
)

@testset "q=4 coevolution among-axis accessors (#188)" begin
    fixture = _coevo_data(p = 12, nrep = 3)
    fit = drm(_coevo_formula(), Gaussian(); data = fixture.data, tree = fixture.phy,
              q4_iterations = 120, q4_n_newton = 30, q4_vcov = false)

    axes = (:mu1, :mu2, :sigma1, :sigma2)

    @testset "coevolution_cor: 4×4 symmetric PD, unit diagonal" begin
        rc = coevolution_cor(fit)
        @test rc.axes == axes
        R = rc.cor
        @test size(R) == (4, 4)
        @test R ≈ R'                              # symmetric
        @test all(isfinite, R)
        @test diag(R) ≈ ones(4)                   # exact unit diagonal
        @test all(abs.(R) .<= 1 + 1e-8)           # valid correlations
        @test isposdef(Symmetric(R))              # positive-definite
        @test minimum(eigvals(Symmetric(R))) > 0  # PD (spectral check)
        # agrees with the stored Σ_a back-transform
        Σ = fit.ranef.Sigma_a
        sd = sqrt.(diag(Σ))
        @test R ≈ Diagonal(inv.(sd)) * Σ * Diagonal(inv.(sd)) rtol = 1e-8
    end

    @testset "coevolution_vc: per-axis variances/SDs positive + finite" begin
        vc4 = coevolution_vc(fit)
        @test vc4.axes == axes
        @test Set(keys(vc4.variance)) == Set(axes)
        @test Set(keys(vc4.sd)) == Set(axes)
        for a in axes
            @test isfinite(vc4.variance[a]) && vc4.variance[a] > 0
            @test isfinite(vc4.sd[a]) && vc4.sd[a] > 0
            @test vc4.sd[a] ≈ sqrt(vc4.variance[a])
        end
        @test vc4.cov ≈ fit.ranef.Sigma_a
        @test [vc4.variance[a] for a in axes] ≈ diag(fit.ranef.Sigma_a)
    end

    @testset "coevolution_summary: tidy long form is consistent" begin
        s = coevolution_summary(fit)
        @test s.axes == axes
        @test length(s.variance) == 4 && all(>(0), s.variance) && all(isfinite, s.variance)
        @test length(s.sd) == 4 && all(>(0), s.sd) && all(isfinite, s.sd)
        @test s.sd ≈ sqrt.(s.variance)
        @test length(s.pair) == 6
        @test length(s.correlation) == 6
        @test length(s.covariance) == 6
        @test all(isfinite, s.correlation)
        @test all(abs.(s.correlation) .<= 1 + 1e-8)
        # long-form entries match the matrices they were unpacked from
        @inbounds for (k, (a, b)) in enumerate(s.pair)
            ia = findfirst(==(a), collect(axes))
            ib = findfirst(==(b), collect(axes))
            @test s.correlation[k] ≈ s.cor[ia, ib]
            @test s.covariance[k] ≈ s.cov[ia, ib]
        end
        @test s.cov ≈ fit.ranef.Sigma_a
    end

    @testset "guard: errors on a non-coevolution fit" begin
        # A plain bivariate residual-correlation fit (no phylo marker) has no Σ_a.
        rfit = drm(
            bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
               sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
               rho12 = @formula(rho12 ~ 1)),
            Gaussian(); data = fixture.data,
        )
        @test_throws ErrorException coevolution_cor(rfit)
        @test_throws ErrorException coevolution_vc(rfit)
        @test_throws ErrorException coevolution_summary(rfit)
    end
end
