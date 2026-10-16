# #995 / #996: the Gaussian phylo(1 | g) quantile residual must be the
# POPULATION-level PIT, i.e. the marginal  y_i ~ N(x_i'β̂, σ̂_e² + σ̂_b² C_ii).
#
# The sparse L-BFGS and conjugate-EM routes store BLUP-CONDITIONAL fitted means
# (`fit.means[:mu] = Xβ̂ + S û`). The σ_b Gauss–Hermite integration used to be
# centred on that mean, so the random effect was counted twice (#995), and it
# used σ_b² for every tip although a tip's prior variance is σ_b²·C_ii on the
# raw branch-length scale (#996). For a Gaussian response the exact reference
# is closed-form, so the residual is pinned against it directly:
#
#     r_i = (y_i − x_i'β̂) / sqrt(σ̂_e² + σ̂_b² C_ii).
#
# Two trees: height 1 (C_ii = 1, isolates #995) and height 3 (C_ii = 3, #996).
using DRModels
using Test, Random, LinearAlgebra, Statistics

function _sim_phylo_995(phy; σb, σe, seed, reps = 5)
    Random.seed!(seed)
    p = phy.n_leaves
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    u = σb .* (cholesky(Symmetric(C + 1e-10I)).L * randn(p))
    species = repeat(1:p, inner = reps)
    n = length(species)
    x = randn(n)
    y = 0.4 .+ 0.7 .* x .+ u[species] .+ σe .* randn(n)
    return (data = (y = y, x = x, species = species), Cii = diag(C)[species])
end

function _marginal_reference_995(fit, sim)
    bl = Dict(fit.blocks)
    β = fit.theta[bl[:mu]]
    σe = exp(fit.theta[first(bl[:sigma])])
    σb = exp(fit.theta[first(bl[:resd])])
    d = sim.data
    return (d.y .- β[1] .- β[2] .* d.x) ./ sqrt.(σe^2 .+ σb^2 .* sim.Cii)
end

@testset "#995/#996 Gaussian phylo PIT is centred on the fixed-effect mean" begin
    f = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1))
    cases = (
        (label = "height 1 (#995)", bl = 1 / 6, σb = 1.0, seed = 20261001),
        (label = "height 3 (#996)", bl = 0.5, σb = 0.5, seed = 20261002),
    )
    for c in cases
        phy = random_balanced_tree(64; branch_length = c.bl)
        sim = _sim_phylo_995(phy; σb = c.σb, σe = 0.4, seed = c.seed)
        @testset "$(c.label)" begin
            # 64 tips = depth 6, so every tip variance is 6 · branch_length.
            @test all(isapprox.(sim.Cii, 6 * c.bl; rtol = 1e-8))
            fits = (
                sparse = drm(f, Gaussian(); data = sim.data, tree = phy),
                reml = drm(f, Gaussian(); data = sim.data, tree = phy, method = :REML),
                em = drm(f, Gaussian(); data = sim.data, tree = phy, algorithm = :em),
            )
            for (nm, fit) in pairs(fits)
                r = residuals(fit; type = :quantile)
                ref = _marginal_reference_995(fit, sim)
                @test r ≈ ref atol = 1e-4
            end
        end
    end
end
