# #996: non-Gaussian phylo(1 | g) Laplace routes. `fit.means[:mu]` is the
# population-level mean on these routes, so the quantile residual's only error
# was the random-effect variance: it used sigma_b^2 for every tip, but tip i has
# prior variance sigma_b^2 * C_ii on the raw branch-length scale (as on the
# Gaussian sparse / EM routes, #1078). The fit now stores the tree diagonal and
# the PIT marginalises each tip over N(0, sigma_b^2 C_ii), with no structured-PIT
# warning (the fixed-effect centre and per-tip variance are both exact).
using DRModels
using Test, Random, LinearAlgebra, Statistics
import Distributions

# Non-ultrametric caterpillar: C_ii runs from about 0.3 to 3.6, so a scalar
# sigma_b^2 cannot be right for every tip.
function _sim_996(; seed = 996, ntip = 24, reps = 5, σb = 0.4)
    Random.seed!(seed)
    phy = random_caterpillar_tree(ntip; branch_length = 0.15)
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    u = σb .* (cholesky(Symmetric(C + 1e-10I)).L * randn(ntip))
    species = repeat(1:ntip, inner = reps)
    n = length(species)
    x = randn(n)
    η = 0.3 .+ 0.3 .* x .+ u[species]
    return (phy = phy, species = species, x = x, η = η, n = n, Cii = diag(C)[species])
end

# Independent reference: marginal CDF of row i at y, integrating b ~ N(0, v_i)
# out of `cond(η0 + b)` with a 64-node Gauss-Hermite rule built here by
# Golub-Welsch (not the package's own).
function _gh(n)
    J = SymTridiagonal(zeros(n), sqrt.(1:n-1) ./ sqrt(2))
    e = eigen(J)
    return e.values, e.vectors[1, :] .^ 2     # nodes and weights for exp(-z^2), weights sum to 1
end
function _marg(cond, η0, v)
    z, w = _gh(64)                            # E f(b), b ~ N(0, v): b = sqrt(2v) z
    return sum(w[k] * cond(η0 + sqrt(2v) * z[k]) for k in eachindex(z))
end

@testset "#996 non-Gaussian phylo PIT uses sigma_b^2 C_ii" begin
    s = _sim_996()
    @test maximum(s.Cii) / minimum(s.Cii) > 5          # the tip variances really differ
    z, w = _gh(64)
    @test sum(w) ≈ 1 atol = 1e-10
    @test sum(w .* z .^ 2) ≈ 0.5 atol = 1e-8             # E[z^2] = 1/2 for exp(-z^2)

    # --- Poisson (discrete: same seeded rng, one draw per row, so u is reproducible) ---
    yp = [rand(Distributions.Poisson(exp(η))) for η in s.η]
    pdat = (y = yp, x = s.x, species = s.species)
    pfit = drm(bf(@formula(y ~ x + phylo(1 | species))), Poisson(); data = pdat, tree = s.phy)
    bl = Dict(pfit.blocks)
    β = pfit.theta[bl[:mu]]; σb = exp(pfit.theta[first(bl[:resd])])
    η0 = β[1] .+ β[2] .* s.x
    # no structured-PIT warning, and the fit still has no stored BLUPs
    @test_logs residuals(pfit; type = :quantile, rng = MersenneTwister(1))
    @test_throws ArgumentError ranef(pfit)
    rr = residuals(pfit; type = :quantile, rng = MersenneTwister(1))
    rg = MersenneTwister(1)
    ref = map(1:s.n) do i
        F(k) = k < 0 ? 0.0 : _marg(b -> Distributions.cdf(Distributions.Poisson(exp(b)), k), η0[i], σb^2 * s.Cii[i])
        a = F(yp[i] - 1); b = F(yp[i])
        Distributions.quantile(Distributions.Normal(), clamp(a + (b - a) * rand(rg), eps(), 1 - eps()))
    end
    @test maximum(abs.(rr .- ref)) < 1e-6

    # --- Binomial (Bernoulli) and NB2 with sigma ~ x: the other Laplace spines ---
    _discrete_ref(fit, y, η0, cdf_k) = begin
        bl = Dict(fit.blocks); σb = exp(fit.theta[first(bl[:resd])])
        rg = MersenneTwister(2)
        map(eachindex(y)) do i
            F(k) = k < 0 ? 0.0 : _marg(b -> cdf_k(i, b, k), η0[i], σb^2 * s.Cii[i])
            a = F(y[i] - 1); b = F(y[i])
            Distributions.quantile(Distributions.Normal(), clamp(a + (b - a) * rand(rg), eps(), 1 - eps()))
        end
    end
    yb = [rand() < 1 / (1 + exp(-η)) ? 1 : 0 for η in s.η]
    bdat = (y = yb, x = s.x, species = s.species)
    bfit = drm(bf(@formula(y ~ x + phylo(1 | species))), Binomial(); data = bdat, tree = s.phy)
    bβ = bfit.theta[Dict(bfit.blocks)[:mu]]
    @test_logs residuals(bfit; type = :quantile, rng = MersenneTwister(2))
    bref = _discrete_ref(bfit, yb, bβ[1] .+ bβ[2] .* s.x,
        (i, b, k) -> k < 1 ? 1 - 1 / (1 + exp(-b)) : 1.0)
    @test maximum(abs.(residuals(bfit; type = :quantile, rng = MersenneTwister(2)) .- bref)) < 1e-6

    yn = [rand(Distributions.NegativeBinomial(3.0, 3.0 / (3.0 + exp(η)))) for η in s.η]
    ndat = (y = yn, x = s.x, species = s.species)
    nfit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x)), NegBinomial2();
               data = ndat, tree = s.phy)
    nβ = nfit.theta[Dict(nfit.blocks)[:mu]]
    @test_logs residuals(nfit; type = :quantile, rng = MersenneTwister(2))
    nref = _discrete_ref(nfit, yn, nβ[1] .+ nβ[2] .* s.x,
        (i, b, k) -> (φ = 1 / nfit.scales[:sigma][i]^2;
                      Distributions.cdf(Distributions.NegativeBinomial(φ, φ / (φ + exp(b))), k)))
    @test maximum(abs.(residuals(nfit; type = :quantile, rng = MersenneTwister(2)) .- nref)) < 1e-6

    # --- Gamma (continuous: no randomisation) ---
    α = 6.0
    yg = [rand(Distributions.Gamma(α, exp(η) / α)) for η in s.η]
    gdat = (y = yg, x = s.x, species = s.species)
    gfit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)), Gamma(); data = gdat, tree = s.phy)
    gb = Dict(gfit.blocks)
    gβ = gfit.theta[gb[:mu]]; gσb = exp(gfit.theta[first(gb[:resd])])
    gσ = gfit.scales[:sigma][1]
    gη0 = gβ[1] .+ gβ[2] .* s.x
    @test_logs residuals(gfit; type = :quantile)
    gref = map(1:s.n) do i
        sh = 1 / gσ^2                                   # shape = 1/sigma^2 (mean mu)
        F = _marg(b -> Distributions.cdf(Distributions.Gamma(sh, exp(b) / sh), yg[i]), gη0[i], gσb^2 * s.Cii[i])
        Distributions.quantile(Distributions.Normal(), clamp(F, eps(), 1 - eps()))
    end
    # 1e-4: the package integrates on 32 nodes, the reference on 64 (the bug is O(0.1-1))
    @test maximum(abs.(residuals(gfit; type = :quantile) .- gref)) < 1e-4
end
