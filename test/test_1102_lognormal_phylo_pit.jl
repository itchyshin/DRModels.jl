# #1102 / #1104 (remainders of #995 / PR #1078).
#
# #1102: a LogNormal `phylo(1 | g)` fit delegates to the Gaussian fit on log(y).
# Its stored `fitted` was the LOG-scale conditional mean Xb + u, so the
# quantile residual took log of an already-logged mean and double-counted
# sigma_b. `fitted` is now the response-scale median exp(Xb + u), as on the
# plain LogNormal route, and the PIT is the closed form
#
#     u_i = Phi((log y_i - x_i'b) / sqrt(sigma_e^2 + sigma_b^2 C_ii)).
#
# #1104: the dense `algorithm = :gls` Gaussian phylo route stores fitted = X b
# with C_ii = 1, so its PIT is already right and must not warn (#995 warning).
using DRModels
using Test, Random, LinearAlgebra, Statistics
import Distributions

function _sim_1102(; seed = 11, ntip = 32, reps = 4, σb = 0.6, σe = 0.4)
    Random.seed!(seed)
    phy = random_balanced_tree(ntip; branch_length = 1 / log2(ntip))   # height 1
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    u = σb .* (cholesky(Symmetric(C + 1e-10I)).L * randn(ntip))
    species = repeat(1:ntip, inner = reps)
    n = length(species)
    x = randn(n)
    ly = 0.4 .+ 0.7 .* x .+ u[species] .+ σe .* randn(n)
    return (phy = phy, data = (y = exp.(ly), ly = ly, x = x, species = species),
            Cii = diag(C)[species])
end

@testset "#1102 / #1104 structured PIT remainders" begin
@testset "#1102 LogNormal phylo quantile residual is the closed-form marginal" begin
    s = _sim_1102()
    @test all(isapprox.(s.Cii, 1.0; atol = 1e-8))
    f = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1))
    fit = drm(f, LogNormal(); data = s.data, tree = s.phy)
    bl = Dict(fit.blocks)
    β = fit.theta[bl[:mu]]
    σe = exp(fit.theta[first(bl[:sigma])])
    σb = exp(fit.theta[first(bl[:resd])])
    z = (log.(s.data.y) .- β[1] .- β[2] .* s.data.x) ./ sqrt.(σe^2 .+ σb^2 .* s.Cii)
    ref = Distributions.cdf.(Distributions.Normal(), z)
    u = Distributions.cdf.(Distributions.Normal(), residuals(fit; type = :quantile))
    @test maximum(abs.(u .- ref)) < 1e-8
    # `fitted` is the response-scale conditional median, not log-scale.
    @test abs(mean(fitted(fit)) - mean(s.data.y)) < abs(mean(fitted(fit)) - mean(log.(s.data.y)))
    @test all(fitted(fit) .> 0)
    @test fit.family isa LogNormal
end

@testset "#1104 dense :gls phylo route does not warn; other structured routes still do" begin
    s = _sim_1102(ntip = 32, reps = 6)
    gd = (y = s.data.ly, x = s.data.x, species = s.data.species)
    # `maxlog = 1` on the warning: the still-warns case must run first.
    K = sigma_phy_dense(s.phy; σ²_phy = 1.0)
    rfit = drm(bf(@formula(y ~ x + relmat(1 | species))), Gaussian(); data = gd, K = K)
    @test DRModels._structured_mean_re(rfit)
    @test_logs (:warn, r"#995") match_mode = :any residuals(rfit; type = :quantile)

    f = bf(@formula(y ~ x + phylo(1 | species)))
    dfit = drm(f, Gaussian(); data = gd, tree = s.phy, algorithm = :gls)
    @test DRModels._structured_mean_re(dfit)
    @test dfit.phylo_scale === :correlation
    @test_logs residuals(dfit; type = :quantile)
    # the silent PIT is the closed-form marginal (C_ii = 1)
    bl = Dict(dfit.blocks)
    β = dfit.theta[bl[:mu]]
    σe = exp(dfit.theta[first(bl[:sigma])]); σb = exp(dfit.theta[first(bl[:resd])])
    ref = (gd.y .- β[1] .- β[2] .* gd.x) ./ sqrt(σe^2 + σb^2)
    @test residuals(dfit; type = :quantile) ≈ ref atol = 1e-8
    # the sparse route is exact and silent too
    sfit = drm(f, Gaussian(); data = gd, tree = s.phy)
    @test sfit.ranef isa DRModels.LocOnlyRanef
    @test_logs residuals(sfit; type = :quantile)
end
end
