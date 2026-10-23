# #1103: a Gaussian `(1 | g)` mean fit judges every row against the closed-form
# marginal N(μ, σ² + σ_b²). The 32-node Gauss–Hermite mixture it used to take
# loses accuracy once σ_b / σ is large (6e-3 at 4, 0.12 at 10).
using DRModels
using Test, Random
import Distributions

@testset "Gaussian (1 | g) quantile residual is the closed-form marginal (#1103)" begin
    Random.seed!(1103)
    G = 40; m = 4; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    σb, σe = 3.0, 0.3                     # σ_b / σ = 10
    y = 0.5 .+ 0.2 .* x .+ σb .* randn(G)[g] .+ σe .* randn(n)
    fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian();
              data = (; y, x, g = string.(g)))
    @test fit.converged
    σ̂b = re_sd(fit)[first(keys(re_sd(fit)))]
    exact = Distributions.quantile.(Distributions.Normal(),
        Distributions.cdf.(Distributions.Normal(),
            (fit.obs[:mu] .- fit.means[:mu]) ./ sqrt.(fit.scales[:sigma] .^ 2 .+ σ̂b^2)))
    q = residuals(fit; type = :quantile)
    @test σ̂b / fit.scales[:sigma][1] > 5
    @test maximum(abs.(q .- exact)) < 1e-10
end
