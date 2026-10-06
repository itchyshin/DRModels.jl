# #759 — `ranef(fit)` silently returned an empty Dict for a non-Gaussian GLMM
# random-intercept fit (the Gauss-Hermite/Laplace marginal routes never
# compute conditional modes/BLUPs), and its docstring pointed at the closed
# #73. This guards that a non-Gaussian GLMM with a random effect throws an
# informative error instead of silently returning an empty Dict, while a
# fit with no random effects at all (the legitimate empty case) still
# returns one.
using DRModels
using Test, Random
import Distributions

@testset "#759 — ranef() on a non-Gaussian GLMM errors instead of silently empty" begin
    Random.seed!(20270927)
    G = 30; m = 20; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [-0.2, 0.3]; σb = 0.4
    bg = σb .* randn(G)
    μ = 1 ./ (1 .+ exp.(-(β[1] .+ β[2] .* x .+ bg[g])))
    ntr = fill(1, n)
    s = [rand(Distributions.Binomial(ntr[i], μ[i])) for i in 1:n]
    fail = ntr .- s
    data = (; s = Float64.(s), fail = Float64.(fail), x, g)

    fit = drm(bf(@formula(cbind(s, fail) ~ x + (1 | g))), Binomial(); data = data)

    @test_throws ArgumentError ranef(fit)

    # A fit with no random effect at all is still a legitimate empty Dict.
    fit_fixed = drm(bf(@formula(cbind(s, fail) ~ x)), Binomial(); data = data)
    @test isempty(ranef(fit_fixed))
end
