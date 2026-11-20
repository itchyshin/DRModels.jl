# #1050 (Beta-binomial routes): `converged` is the unit-free Newton step in SE units
# plus the iteration-cap check (the #1044 flag), not Optim's absolute `g_tol = 1e-8`
# gradient test. A predictor scaled up moves the gradient with it, so the same
# stationary fit stalls above the absolute bar and reads unconverged.
# Covers `_fit_betabinomial` (g_tol truncation only: at large predictor scale the fixed
# route stops in a line-search assertion, a separate defect), `_fit_betabinomial_ranef`
# (x * 5e5) and `_fit_betabinomial_corr_ranef` (x * 3e4). Seeds below are the ones that
# read unconverged on main and converged with the unit-free flag (Totoro, Julia 1.12.6,
# G = 40, m = 15). Beyond about 1e6 the Hessian is near-singular in correlation form and
# `_unitfree_converged` falls back to its absolute rule, so those scales stay unconverged.
using DRModels
using Test, Random
import Distributions

function _gen_1050_bb(rep, scale; G = 40, m = 15)
    Random.seed!(rep); n = G * m
    g = repeat(1:G, inner = m); x0 = randn(n)
    bg = 0.5 .* randn(G); b1 = 0.3 .* randn(G)
    ntr = fill(20, n); φ = 12.0
    μ = 1 ./ (1 .+ exp.(-(0.3 .+ 0.6 .* x0 .+ bg[g] .+ b1[g] .* x0)))
    s = [rand(Distributions.BetaBinomial(ntr[i], μ[i] * φ, (1 - μ[i]) * φ)) for i in 1:n]
    return (; s = Float64.(s), fail = Float64.(ntr .- s), x = x0 .* scale, g)
end

@testset "#1050 Beta-binomial unit-free converged flag" begin
    sig = @formula(sigma ~ 1)

    # Random intercept, x * 5e5: a stationary fit that main reads as unconverged.
    for rep in (1, 3, 4, 6, 8)
        d = _gen_1050_bb(rep, 5e5)
        fit = drm(bf(@formula(cbind(s, fail) ~ x + (1 | g)), sig), BetaBinomial(); data = d)
        @test fit.converged
    end

    # Correlated intercept + slope, x * 3e4.
    for rep in (4,)
        d = _gen_1050_bb(rep, 3e4)
        fit = drm(bf(@formula(cbind(s, fail) ~ x + (1 + x | g)), sig), BetaBinomial(); data = d)
        @test fit.converged
    end

    # A truncated fit still reads unconverged: `g_tol = 1e6` stops at the start,
    # which main reports as converged (Optim's g criterion fires trivially).
    d = _gen_1050_bb(1, 1.0)
    for f in (@formula(cbind(s, fail) ~ x),
              @formula(cbind(s, fail) ~ x + (1 | g)),
              @formula(cbind(s, fail) ~ x + (1 + x | g)))
        @test !drm(bf(f, sig), BetaBinomial(); data = d, g_tol = 1e6).converged
    end

    # Ordinary fits are unchanged: still converged at unit scale.
    @test drm(bf(@formula(cbind(s, fail) ~ x), sig), BetaBinomial(); data = d).converged
end
