# `marginal = :AGHQ` on the Gaussian random intercept on sigma,
# bf(y ~ x, sigma ~ 1 + (1 | g)). The DEFAULT `:LA` integrates each group's
# effect by non-adaptive 32-node Gauss-Hermite quadrature on the PRIOR scale
# (unchanged, D-273); `:AGHQ` integrates the same group marginal by per-group
# ADAPTIVE Gauss-Hermite quadrature (`_aghq_marginal_loglik`, #834/#719),
# reusing the shared helper the other `(1 | g)` families were moved onto in
# #846. For an informative group (large sigma_b, many rows) the prior-scale
# grid misses the posterior mode; this file demonstrates the size of that miss
# with an independent QuadGK reference and confirms `:AGHQ` closes it.
using DRModels
using Test, Random, QuadGK, Distributions

# Independent reference: the exact per-group log marginal
#   log int N(b; 0, s^2) * prod_i N(y_i; mu_i, exp(eta0_i + b)) db,
# by direct 1-D quadrature over b (not the z-substitution either integrator
# uses), written from Distributions + QuadGK only -- no package internals.
function _exact_group_loglik(yg::Vector{Float64}, mug::Vector{Float64}, eta0g::Vector{Float64}, s::Float64)
    logintegrand(b) = logpdf(Normal(0.0, s), b) +
        sum(logpdf(Normal(mug[i], exp(eta0g[i] + b)), yg[i]) for i in eachindex(yg))
    bs = range(-10 * s, 10 * s, length = 2001)
    mx = maximum(logintegrand.(bs))
    val, _ = QuadGK.quadgk(b -> exp(logintegrand(b) - mx), -10 * s, 10 * s, rtol = 1e-13)
    return mx + log(val)
end

function _exact_total_loglik(y, x, g, theta, s)
    beta = theta[1:2]; gamma0 = theta[3]
    mu = beta[1] .+ beta[2] .* x
    total = 0.0
    for lev in unique(g)
        idx = findall(==(lev), g)
        total += _exact_group_loglik(y[idx], mu[idx], fill(gamma0, length(idx)), s)
    end
    return total
end

@testset "Gaussian sigma random intercept: marginal = :AGHQ" begin
    # ------------------------------------------------------------------
    # (a) Literal fixed-theta check at a LARGE sigma_b: the AGHQ objective
    # must match the independent QuadGK integral to 1e-6 nat; the OLD default
    # 32-node prior-scale grid (reconstructed inline, matching the pre-#846
    # object in src/gaussian_ranef.jl) misses badly at this theta.
    # ------------------------------------------------------------------
    Random.seed!(20260927)
    G = 3; m = 30; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    beta_true = [0.3, -0.2]; gamma0_true = 0.2
    sigma_b_true = 5.0
    bg_true = sigma_b_true .* [1.1, -0.7, 0.4]     # literal, fixed group draws
    mu_true = beta_true[1] .+ beta_true[2] .* x
    y = mu_true .+ exp.(gamma0_true .+ bg_true[g]) .* randn(n)
    data = (; y, x, g)
    f = bf(@formula(y ~ x), @formula(sigma ~ 1 + (1 | g)))

    # A fixed evaluation theta (not the optimum): [beta1, beta2, gamma0, log(sigma_b)]
    theta_fixed = [0.3, -0.2, 0.2, log(5.0)]
    s_fixed = exp(theta_fixed[4])
    exact_ll = _exact_total_loglik(y, x, g, theta_fixed, s_fixed)

    fit_aghq = drm(f, Gaussian(); data = data, marginal = :AGHQ)
    @test fit_aghq.marginal === :AGHQ
    aghq_ll_at_fixed = -fit_aghq.nll(theta_fixed)   # shipped default, K = _RANEF1D_AGHQ_K = 5

    # The underlying method at high K: reproduces the package's own AGHQ path
    # (same `_AGHQRule`/`_aghq_marginal_loglik` helper, same per-observation
    # log-density) but with K = 25 nodes, to show the METHOD converges to the
    # exact QuadGK integral to 1e-6 nat (K = 5 is a deliberate cheap default,
    # not a ceiling on the helper's accuracy).
    aghq_ll_highK = let K = 25, rule = DRModels._AGHQRule(1, K), Zre = ones(n, 1),
                        bcache = zeros(1, G), l2pi = log(2π)
        eta0 = fill(theta_fixed[3], n)
        r = y .- (theta_fixed[1] .+ theta_fixed[2] .* x)
        ll = (i, eta) -> -0.5 * l2pi - eta - 0.5 * r[i]^2 * exp(-2 * eta)
        members = [findall(==(lev), g) for lev in 1:G]
        L = reshape([s_fixed], 1, 1)
        DRModels._aghq_marginal_loglik(ll, members, eta0, Zre, L, rule, bcache)
    end
    @test aghq_ll_highK ≈ exact_ll atol = 1e-6
    # The shipped default (K = 5) is within the family-wide 0.01 nat AGHQ
    # tolerance (matching #846's own per-family sweep), a large improvement on
    # the old 32-node prior-scale grid's >1 nat miss below.
    @test abs(aghq_ll_at_fixed - exact_ll) < 0.01

    # The OLD default route's own 32-node PRIOR-scale objective (independent
    # transcription, matching the pre-#846 `_fit_sigma_ranef_gaussian` GHQ-32
    # sum) at the SAME theta: it misses the exact integral by a large margin
    # at this large sigma_b, which is exactly the finding this PR fixes.
    z, w = DRModels._gauss_hermite(32)
    ghq32_ll = let l2pi = log(2π), lpi = log(π)
        total = 0.0
        for lev in unique(g)
            idx = findall(==(lev), g)
            mg = length(idx)
            eta0 = theta_fixed[3]
            r = y[idx] .- (theta_fixed[1] .+ theta_fixed[2] .* x[idx])
            Bg = sum(r .^ 2 .* exp(-2 * eta0))
            terms = [log(w[k] / sqrt(π)) - mg * sqrt(2) * s_fixed * z[k] -
                     0.5 * exp(-2 * sqrt(2) * s_fixed * z[k]) * Bg for k in eachindex(z)]
            mx = maximum(terms)
            total += -0.5 * mg * l2pi - mg * eta0 + mx + log(sum(exp.(terms .- mx)))
        end
        total
    end
    @test abs(ghq32_ll - exact_ll) > 1.0   # the failure this PR fixes (nats off)
    @info "sigma-RE AGHQ demo at fixed theta (sigma_b=5, 3 groups x 30 rows)" exact_ll aghq_ll_at_fixed aghq_ll_highK ghq32_ll aghq_error=abs(aghq_ll_at_fixed - exact_ll) aghq_highK_error=abs(aghq_ll_highK - exact_ll) ghq32_error=abs(ghq32_ll - exact_ll)

    # ------------------------------------------------------------------
    # (b) Existing (small sigma_b) fixture: :AGHQ must move estimates no more
    # than the integration-error scale relative to the unchanged default.
    # ------------------------------------------------------------------
    Random.seed!(20260613)
    G2 = 40; m2 = 25; n2 = G2 * m2
    g2 = repeat(1:G2, inner = m2); x2 = randn(n2)
    beta2 = [0.5, -0.3]; gamma02 = log(0.5); sigma_b2 = 0.5
    bg2 = sigma_b2 .* randn(G2)
    y2 = beta2[1] .+ beta2[2] .* x2 .+ exp.(gamma02 .+ bg2[g2]) .* randn(n2)
    data2 = (; y = y2, x = x2, g = g2)

    fit_default = drm(f, Gaussian(); data = data2)          # marginal = :LA (unchanged)
    fit_aghq2 = drm(f, Gaussian(); data = data2, marginal = :AGHQ)
    @test fit_default.marginal === :LA
    @test fit_aghq2.marginal === :AGHQ

    delta_theta = abs.(fit_aghq2.theta .- fit_default.theta)
    delta_loglik = abs(loglik(fit_aghq2) - loglik(fit_default))
    @info "sigma-RE AGHQ vs default (:LA) on the small-sigma_b fixture" delta_theta delta_loglik
    # This group size / sigma_b is exactly where GHQ-32 is already accurate
    # (see docs/dev-log/evidence/arc2-sigma-re-laplace/receipt.md), so AGHQ
    # should land at essentially the same optimum: within 0.02 on every
    # working-scale coefficient and 0.05 nat on the total logLik.
    @test all(delta_theta .< 0.02)
    @test delta_loglik < 0.05
end
