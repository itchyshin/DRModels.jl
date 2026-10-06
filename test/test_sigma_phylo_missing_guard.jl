# Over-parameterisation guard for the Gaussian missing-response routes.
#
# With too few observed responses, the phylo prior regularises the latent field
# so the σ-phylo fit does NOT crash — but the FIXED-effect part is
# under-determined and the engine would silently return a meaningless
# over-parameterised result (positive logLik, negative residual dof). The guard
# must count the total estimated parameters (pμ + pσ + phylo-variance
# components), not just the mean coefficient count pμ.
using DRModels
using Test

@testset "Gaussian missing-response over-parameterisation guard" begin
    p = 20; m = 2
    phy = random_balanced_tree(p; branch_length = 0.2)
    species = repeat(1:p, inner = m); n = length(species)
    x = Float64[sin(i) for i in 1:n]
    yfull = 0.3 .+ 0.5 .* x .+ 0.4 .* Float64[cos(2i) for i in 1:n]

    @testset "both-phylo: n_obs < pμ+pσ+nvar errors (was a silent positive-logLik overfit)" begin
        # pμ=2 (1+x), pσ=2 (1+x), nvar=2 (both-phylo separate) → total dof = 6.
        ymiss = fill(NaN, n); ymiss[1:3] .= yfull[1:3]    # n_obs = 3 < 6
        dat = (; y = ymiss, x, species)
        form = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x + phylo(1 | species)))
        @test_throws ErrorException drm(form, Gaussian(); data = dat, tree = phy)
    end

    @testset "asymmetric σ-phylo: n_obs < pμ+pσ+1 errors" begin
        # Asymmetric = phylo on SIGMA, fixed-effect mean: pμ=2 (1+x), pσ=1 (sigma
        # fixed intercept), nvar=1 (σ-phylo SD) → total dof = 4.
        ymiss = fill(NaN, n); ymiss[1:3] .= yfull[1:3]    # n_obs = 3 < 4
        dat = (; y = ymiss, x, species)
        form = bf(@formula(y ~ x), @formula(sigma ~ phylo(1 | species)))
        @test_throws ErrorException drm(form, Gaussian(); data = dat, tree = phy)
    end

    @testset "enough observed responses still fits (real, non-saturated)" begin
        ymiss = copy(yfull); ymiss[1:4] .= NaN            # n_obs = n-4 ≫ dof
        dat = (; y = ymiss, x, species)
        form = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ phylo(1 | species)))
        fit = drm(form, Gaussian(); data = dat, tree = phy)
        @test isfinite(loglik(fit))
        @test loglik(fit) < 0          # a determined fit has negative logLik (not the overfit +value)
    end

    @testset "fixed-effect route: n_obs < pμ+pσ errors" begin
        nf = 30; xf = Float64[sin(i) for i in 1:nf]
        yf = fill(NaN, nf); yf[1:2] .= 0.3 .+ 0.5 .* xf[1:2]   # n_obs = 2 < pμ(2)+pσ(2) = 4
        datf = (; y = yf, x = xf)
        form = bf(@formula(y ~ x), @formula(sigma ~ x))
        @test_throws ArgumentError drm(form, Gaussian(); data = datf)
    end
end
