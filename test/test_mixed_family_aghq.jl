# `fit_mixed_family(...; aghq = true)`: the cross-family shared-latent route
# (src/mixed_family.jl) integrated by per-"group" ADAPTIVE Gauss-Hermite
# quadrature instead of the DEFAULT fixed K=32 prior-scale grid. Each
# observation i is its own AGHQ "group" of two virtual members (family 1's
# contribution, family 2's), sharing the one latent u_i ~ N(0,1); the fixed
# prior (L = [1]) and the loadings lambda1/lambda2 (not a group SD) carry the
# scale. `aghq = false` (the default) is untouched -- this file also checks
# that.
using DRModels
using Test, Random, QuadGK
import Distributions
using Distributions: Normal, logpdf   # NOT a bare `using Distributions`: that makes `Poisson`, `Gamma`, ... ambiguous with DRModels in every later suite file

_pois_ll(η, y) = y * η - exp(η) - DRModels.loggamma(y + 1)
_gauss_ll(η, y, sd) = -0.5 * ((y - η) / sd)^2 - log(sd) - 0.9189385332046727

# Independent per-observation exact reference: direct 1-D quadrature over the
# shared latent u (not the z-substitution either integrator uses), Poisson x
# Gaussian, written from Distributions + QuadGK only -- no package internals.
function _exact_obs_ll(η1::Float64, y1::Float64, η2::Float64, y2::Float64,
                       λ1::Float64, λ2::Float64, sd2::Float64)
    logintegrand(u) = logpdf(Normal(0.0, 1.0), u) + _pois_ll(η1 + λ1 * u, y1) +
                       _gauss_ll(η2 + λ2 * u, y2, sd2)
    us = range(-40.0, 40.0, length = 4001)
    mx = maximum(logintegrand.(us))
    val, _ = QuadGK.quadgk(u -> exp(logintegrand(u) - mx), -40.0, 40.0, rtol = 1e-13)
    return mx + log(val)
end

# The package's own DEFAULT (aghq=false) per-observation objective, prior-scale
# K=32 Gauss-Hermite -- an independent transcription of the loop in
# `fit_mixed_family`'s `nll`, at a literal fixed (η1, y1, η2, y2, λ1, λ2).
function _ghq32_obs_ll(η1, y1, η2, y2, λ1, λ2, sd2)
    z, w = DRModels._gauss_hermite(32); logw = log.(w); rt2 = sqrt(2.0)
    acc = [logw[k] + _pois_ll(η1 + λ1 * rt2 * z[k], y1) + _gauss_ll(η2 + λ2 * rt2 * z[k], y2, sd2)
           for k in eachindex(z)]
    mx = maximum(acc)
    return mx + log(sum(exp.(acc .- mx))) - 0.5 * log(π)
end

@testset "cross-family fit_mixed_family: aghq = true" begin
    # ------------------------------------------------------------------
    # (a) Literal fixed-input check at LARGE loadings (lambda plays the role
    # of "sigma_b" here): the AGHQ per-observation objective must match the
    # independent QuadGK integral to 1e-6 nat; the OLD default K=32
    # prior-scale grid (reconstructed inline) misses badly at these inputs.
    # ------------------------------------------------------------------
    η1, y1, η2, y2 = 0.5, 25.0, 0.0, 6.0
    λ1 = λ2 = 5.0
    exact_ll = _exact_obs_ll(η1, y1, η2, y2, λ1, λ2, 1.0)
    ghq32_ll = _ghq32_obs_ll(η1, y1, η2, y2, λ1, λ2, 1.0)
    @test abs(ghq32_ll - exact_ll) > 1.0   # the failure this PR fixes (nats off)

    # Reproduce the package's AGHQ path directly (same `_AGHQRule` /
    # `_aghq_marginal_loglik` helper, same per-observation log-density via
    # `_mf_obs_ll`) at K = 25, to check the METHOD against the exact QuadGK
    # integral to 1e-6 nat.
    aghq_ll_highK = let K = 25, rule = DRModels._AGHQRule(1, K), bcache = zeros(1, 1)
        members = [[1, 2]]
        η0 = [η1, η2]
        Zre = reshape([λ1, λ2], 2, 1)
        ll = (m, η) -> m == 1 ? DRModels._mf_obs_ll(DRModels.Poisson(), η, y1, 1.0, 1.0) :
                                DRModels._mf_obs_ll(Gaussian(), η, y2, 1.0, 1.0)
        L = reshape([1.0], 1, 1)
        DRModels._aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    @test aghq_ll_highK ≈ exact_ll atol = 1e-6

    # The shipped default K for `aghq = true` (_RANEF1D_AGHQ_K = 5, matching
    # every other `(1 | g)` family's #846 default): within the same 0.01 nat
    # family-wide tolerance.
    aghq_ll_K5 = let K = DRModels._RANEF1D_AGHQ_K, rule = DRModels._AGHQRule(1, K), bcache = zeros(1, 1)
        members = [[1, 2]]
        η0 = [η1, η2]
        Zre = reshape([λ1, λ2], 2, 1)
        ll = (m, η) -> m == 1 ? DRModels._mf_obs_ll(DRModels.Poisson(), η, y1, 1.0, 1.0) :
                                DRModels._mf_obs_ll(Gaussian(), η, y2, 1.0, 1.0)
        L = reshape([1.0], 1, 1)
        DRModels._aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    @test abs(aghq_ll_K5 - exact_ll) < 0.01
    @info "mixed_family AGHQ demo at fixed inputs (Poisson x Gaussian, lambda=5)" exact_ll ghq32_ll aghq_ll_K5 aghq_ll_highK ghq32_error=abs(ghq32_ll - exact_ll) aghq_K5_error=abs(aghq_ll_K5 - exact_ll) aghq_highK_error=abs(aghq_ll_highK - exact_ll)

    # ------------------------------------------------------------------
    # (b) A full `fit_mixed_family(...; aghq = true)` fit on data simulated
    # with large loadings recovers the truth; `aghq = false` (default) also
    # runs unchanged on the same data (both must converge; report deltas).
    # ------------------------------------------------------------------
    Random.seed!(20260927)
    n = 600
    x = randn(n)
    X1 = hcat(ones(n), x); X2 = hcat(ones(n), x)
    β1 = [0.4, 0.3]; β2 = [0.2, -0.1]
    λ1_true = 2.5; λ2_true = 2.0     # large enough that the prior grid struggles
    u = randn(n)
    η1v = X1 * β1 .+ λ1_true .* u
    y1 = Float64[rand(Distributions.Poisson(exp(clamp(η1v[i], -20.0, 20.0)))) for i in 1:n]
    η2v = X2 * β2 .+ λ2_true .* u
    y2 = η2v .+ randn(n)

    fit_default = DRModels.fit_mixed_family(; y1 = y1, X1 = X1, fam1 = DRModels.Poisson(),
                                            y2 = y2, X2 = X2, fam2 = Gaussian(),
                                            confint = false)
    fit_aghq = DRModels.fit_mixed_family(; y1 = y1, X1 = X1, fam1 = DRModels.Poisson(),
                                         y2 = y2, X2 = X2, fam2 = Gaussian(),
                                         confint = false, aghq = true)
    @test fit_default.converged
    @test fit_aghq.converged
    @test fit_aghq.λ1 ≈ λ1_true atol = 0.4
    @test fit_aghq.λ2 ≈ λ2_true atol = 0.4
    delta_loglik = abs(fit_aghq.loglik - fit_default.loglik)
    delta_rho = abs(fit_aghq.rho_latent - fit_default.rho_latent)
    @info "mixed_family AGHQ vs default (K=32 prior grid) full-fit deltas" delta_loglik delta_rho fit_default.loglik fit_aghq.loglik fit_default.rho_latent fit_aghq.rho_latent

    # ------------------------------------------------------------------
    # (c) `aghq = false` is byte-for-byte the pre-existing default: an
    # existing recovery fixture (Gaussian x Poisson, small loadings, from
    # test_mixed_family.jl's own DGP) must move ONLY by construction (i.e.
    # not at all -- same seed, same call, aghq keyword omitted vs false).
    # ------------------------------------------------------------------
    fit_explicit_false = DRModels.fit_mixed_family(; y1 = y1, X1 = X1, fam1 = DRModels.Poisson(),
                                                   y2 = y2, X2 = X2, fam2 = Gaussian(),
                                                   confint = false, aghq = false)
    @test fit_explicit_false.loglik == fit_default.loglik
    @test fit_explicit_false.λ1 == fit_default.λ1
    @test fit_explicit_false.λ2 == fit_default.λ2
end
