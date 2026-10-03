# residuals(fit; type = :quantile) routes a fit with random effects by the
# AXIS its random intercept lives on (#923). The σ_b-marginal route (#760)
# integrates a lone random intercept out of the MEAN. A Gaussian
# `sigma ~ 1 + (1 | g)` fit also stores exactly one `:resd` block (named
# `g_logsigma`), and used to take that mean route: it shifted the mean by
# N(0, τ²) with τ the log-σ random-effect SD, which is a different model.
#
# Each reference below is computed independently of the engine: by adaptive
# quadrature (QuadGK) or in closed form, never by the engine's 32-node
# Gauss–Hermite rule.
using DRModels
using Test, Random, Statistics, LinearAlgebra
import Distributions, QuadGK

const _QRA_N01 = Distributions.Normal()
_qra_q(F) = Distributions.quantile(_QRA_N01, clamp(F, eps(), 1 - eps()))

# Lone mean random intercept, Gaussian: integrating b ~ N(0, σ_b²) out of
# N(μ + b, σ²) gives N(μ, σ² + σ_b²) exactly.
_qra_mean_marginal(fit, sb) =
    _qra_q.(Distributions.cdf.(_QRA_N01, (fit.obs[:mu] .- fit.means[:mu]) ./
                                         sqrt.(fit.scales[:sigma] .^ 2 .+ sb^2)))

@testset "quantile residuals: random-intercept axis routing (#923)" begin
    Random.seed!(923)
    G = 30; m = 20; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)

    @testset "Gaussian sigma ~ 1 + (1 | g): marginal over the log-σ intercept" begin
        τ = 0.6
        bg = τ .* randn(G)
        y = 0.5 .- 0.3 .* x .+ exp.(log(0.5) .+ bg[g]) .* randn(n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + (1 | g))), Gaussian();
                  data = (; y, x, g))
        @test fit.converged
        @test only(last(only(filter(p -> first(p) === :resd, fit.coefnames)))) == "g_logsigma"
        τ̂ = re_sd(fit)[:g_logsigma]
        μ = fit.means[:mu]; σ0 = fit.scales[:sigma]

        q = residuals(fit; type = :quantile)

        # Reference: F_i(y) = ∫ Φ((y − μ_i) / (σ0_i e^b)) φ(b; 0, τ̂²) db.
        prior = Distributions.Normal(0.0, τ̂)
        q_ref = map(1:n) do i
            F, _ = QuadGK.quadgk(b -> Distributions.cdf(_QRA_N01, (y[i] - μ[i]) / (σ0[i] * exp(b))) *
                                      Distributions.pdf(prior, b), -Inf, Inf; rtol = 1e-12)
            _qra_q(F)
        end
        # The engine's 32-node rule against adaptive quadrature: ~1e-6 at the
        # worst row (the log-σ mixture is less smooth in b than the mean one).
        @test maximum(abs.(q .- q_ref)) < 1e-5

        # Not the mean-axis marginal the old routing used (it moved residuals
        # by more than 0.4 on these data), and not the b = 0 reference either.
        @test maximum(abs.(q .- _qra_mean_marginal(fit, τ̂))) > 0.1
        @test maximum(abs.(q .- (y .- μ) ./ σ0)) > 0.1

        # Correct model: the σ-marginal residuals are close to N(0, 1). The
        # mean-axis route gave sd ≈ 0.75 here, the b = 0 reference sd ≈ 1.25.
        @test abs(std(q) - 1) < 0.1
    end

    @testset "mean (1 | g) + sigma (1 | g): two blocks, random effects at 0" begin
        bμ = 0.4 .* randn(G); bσ = 0.4 .* randn(G)
        y = 0.5 .- 0.3 .* x .+ bμ[g] .+ exp.(log(0.5) .+ bσ[g]) .* randn(n)
        fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1 + (1 | g))), Gaussian();
                  data = (; y, x, g))
        @test DRModels._ranef_marginal_mix(fit, fit.family, fit.means[:mu]) === nothing
        q = residuals(fit; type = :quantile)
        @test q ≈ _qra_q.(Distributions.cdf.(_QRA_N01, (y .- fit.means[:mu]) ./ fit.scales[:sigma])) atol = 1e-10
    end

    @testset "lone mean (1 | g): σ_b-marginal" begin
        y = 0.5 .- 0.3 .* x .+ (0.5 .* randn(G))[g] .+ 0.5 .* randn(n)
        fit = drm(bf(@formula(y ~ x + (1 | g))), Gaussian(); data = (; y, x, g))
        mix = DRModels._ranef_marginal_mix(fit, fit.family, fit.means[:mu])
        @test mix !== nothing
        q = residuals(fit; type = :quantile)
        @test maximum(abs.(q .- _qra_mean_marginal(fit, re_sd(fit)[:g]))) < 1e-6
    end

    @testset "lone phylo(1 | species) on the mean: σ_b-marginal" begin
        # Balanced 32-tip tree, five levels of 0.2: ultrametric of height 1, so
        # each tip's phylogenetic variance is σ_b².
        phy = random_balanced_tree(32; branch_length = 0.2)
        @test phylo_tree_height(phy) ≈ 1.0
        species = repeat(1:32, inner = 8); xs = randn(length(species))
        C = sigma_phy_dense(phy)
        a = cholesky(Symmetric(0.25 .* C + 1e-10I)).L * randn(32)
        y = 0.2 .+ 0.4 .* xs .+ a[species] .+ 0.5 .* randn(length(species))
        fit = drm(bf(@formula(y ~ x + phylo(1 | species))), Gaussian();
                  data = (; y, x = xs, species), tree = phy)
        @test only(last(only(filter(p -> first(p) === :resd, fit.coefnames)))) == "species"
        mix = DRModels._ranef_marginal_mix(fit, fit.family, fit.means[:mu])
        @test mix !== nothing
        q = residuals(fit; type = :quantile)
        @test maximum(abs.(q .- _qra_mean_marginal(fit, re_sd(fit)[:species]))) < 1e-6
    end

    @testset "lone relmat(1 | id) on the mean: σ_b-marginal" begin
        Gk = 40
        K = [0.5^abs(i - j) for i in 1:Gk, j in 1:Gk]      # unit diagonal
        id = repeat(1:Gk, inner = 10); xr = randn(length(id))
        a = cholesky(Symmetric(0.25 .* K)).L * randn(Gk)
        y = 0.2 .+ 0.4 .* xr .+ a[id] .+ 0.5 .* randn(length(id))
        fit = drm(bf(@formula(y ~ x + relmat(1 | id))), Gaussian();
                  data = (; y, x = xr, id), K = K)
        mix = DRModels._ranef_marginal_mix(fit, fit.family, fit.means[:mu])
        @test mix !== nothing
        q = residuals(fit; type = :quantile)
        @test maximum(abs.(q .- _qra_mean_marginal(fit, re_sd(fit)[:id]))) < 1e-6
    end

    @testset "a non-Gaussian log-σ intercept is not marginalised" begin
        # `_fit_sigma_axis_re` (the non-Gaussian σ-axis route) tags its block
        # `<g>_logsigma` too. No σ-marginal mapping is verified for those
        # families, so the guard keeps the random effect at 0 rather than
        # routing it to the mean.
        k = 5
        fit = DRModels.DrmFit(DRModels.Gamma(), [:mu => 1:1, :sigma => 2:2, :resd => 3:3],
                              [:mu => ["(Intercept)"], :sigma => ["(Intercept)"], :resd => ["g_logsigma"]],
                              [log(2.0), log(1.5), log(0.5)], zeros(3, 3), 0.0, k, true,
                              Dict(:mu => fill(2.0, k)), Dict(:mu => fill(2.0, k)),
                              Dict(:sigma => fill(1.5, k)))
        @test DRModels._ranef_marginal_mix(fit, fit.family, fit.means[:mu]) === nothing
    end
end
