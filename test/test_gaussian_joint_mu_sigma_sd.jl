# Gaussian joint mean + sigma ordinary random intercepts, each with an SD formula
# (twin of drmTMB test-gaussian-mu-sigma-joint-sd.R):
#   y ~ x + (1 | id),  sigma ~ 1 + (0 + z | site) | (1 | site)
#   sd(id) ~ w,        sd(site, sigma[, coef(z)]) ~ v
# The marginal is non-Gaussian, so the oracle is an independent dense Laplace
# in the RAW effect scale u = (a, b) (own mode finder, own derivatives; no
# DRModels internals), evaluated at the fitted parameters. Tolerances fixed
# before the first run: |logLik - oracle| < 1e-4; oracle mode |grad|_inf < 1e-8,
# Hessian PD; constant-SD limit logLik 1e-5, coefficients 1e-3; recovery
# |alpha_hat - alpha| < 0.5 (mean SD formula) / 0.6 (sigma SD formula).
using DRModels, Random, Test, LinearAlgebra

function _gjs_fixture(; seed = 20261101, n_id = 50, n_site = 80, n_each = 32,
                      alpha_mu = [log(0.6), 0.4], alpha_sigma = [log(0.35), 0.6],
                      slope = true)
    rng = MersenneTwister(seed)
    n = n_id * n_each
    row = 1:n
    id = [string("id", lpad(string((r - 1) ÷ n_each + 1), 2, '0')) for r in row]
    site = [string("s", lpad(string((r - 1) % n_site + 1), 2, '0')) for r in row]
    gi = [(r - 1) ÷ n_each + 1 for r in row]
    gs = [(r - 1) % n_site + 1 for r in row]
    w_id = collect(range(-1, 1; length = n_id))
    v_site = collect(range(-1, 1; length = n_site))
    sd_id = exp.(alpha_mu[1] .+ alpha_mu[2] .* w_id)
    sd_site = exp.(alpha_sigma[1] .+ alpha_sigma[2] .* v_site)
    a = sd_id .* randn(rng, n_id)
    b = sd_site .* randn(rng, n_site)
    x = randn(rng, n)
    z = slope ? randn(rng, n) : ones(n)
    eta = log(0.5) .+ b[gs] .* z
    y = 0.2 .+ 0.5 .* x .+ a[gi] .+ exp.(eta) .* randn(rng, n)
    data = (; y, x, z, id, site, w = w_id[gi], v = v_site[gs])
    return (; data, w_id, v_site, gi, gs, alpha_mu, alpha_sigma)
end

# Dense Laplace over u = (a, b), a_g ~ N(0, sd_a_g^2), b_j ~ N(0, sd_b_j^2),
# eta_i = s0 + b[gs_i] z_i; written from the model equations only.
function _gjs_oracle(y, mu0, z, gi, gs, s0, sd_a, sd_b)
    na, nb = length(sd_a), length(sd_b)
    m = na + nb
    pr(u) = begin
        a = u[1:na]; b = u[na+1:m]
        r = y .- mu0 .- a[gi]
        eta = s0 .+ b[gs] .* z
        return r, eta, exp.(-2 .* eta), a, b
    end
    negjoint(u) = begin
        r, eta, ie, a, b = pr(u)
        sum(eta .+ 0.5 .* r .^ 2 .* ie .+ 0.5 * log(2π)) +
            sum(0.5 .* a .^ 2 ./ sd_a .^ 2 .+ log.(sd_a) .+ 0.5 * log(2π)) +
            sum(0.5 .* b .^ 2 ./ sd_b .^ 2 .+ log.(sd_b) .+ 0.5 * log(2π))
    end
    grad(u) = begin
        r, eta, ie, a, b = pr(u)
        ge = 1 .- r .^ 2 .* ie
        g = zeros(m)
        for i in eachindex(y)
            g[gi[i]] += -r[i] * ie[i]
            g[na+gs[i]] += ge[i] * z[i]
        end
        g[1:na] .+= a ./ sd_a .^ 2
        g[na+1:m] .+= b ./ sd_b .^ 2
        return g
    end
    hess(u) = begin
        r, eta, ie, a, b = pr(u)
        H = zeros(m, m)
        for i in eachindex(y)
            H[gi[i], gi[i]] += ie[i]
            H[na+gs[i], na+gs[i]] += 2 * r[i]^2 * ie[i] * z[i]^2
            x = 2 * r[i] * ie[i] * z[i]
            H[gi[i], na+gs[i]] += x
            H[na+gs[i], gi[i]] += x
        end
        for k in 1:na; H[k, k] += 1 / sd_a[k]^2; end
        for j in 1:nb; H[na+j, na+j] += 1 / sd_b[j]^2; end
        return H
    end
    u = zeros(m)
    # coarse gradient descent with backtracking, then pure Newton to 1e-11
    for _ in 1:2000
        g = grad(u)
        maximum(abs, g) < 1e-3 && break
        H = hess(u)
        step = try cholesky(Symmetric(H)) \ g catch; g ./ (diag(H) .+ 1) end
        t = 1.0
        f0 = negjoint(u)
        while t > 1e-10 && negjoint(u .- t .* step) > f0
            t /= 2
        end
        u .-= t .* step
    end
    for _ in 1:50
        g = grad(u)
        maximum(abs, g) < 1e-11 && break
        u .-= hess(u) \ g
    end
    H = hess(u)
    ll = -negjoint(u) + 0.5 * m * log(2π) - 0.5 * logdet(cholesky(Symmetric(H)))
    return (; loglik = ll, max_grad = maximum(abs, grad(u)),
            min_eig = minimum(eigvals(Symmetric(H))))
end

function _gjs_oracle_from_fit(fit, fx, key_mu, key_sig)
    d = fx.data
    am = coef(fit, key_mu); asg = coef(fit, key_sig)
    sd_a = exp.(am[1] .+ am[2] .* fx.w_id)
    sd_b = exp.(asg[1] .+ asg[2] .* fx.v_site)
    X = hcat(ones(length(d.y)), d.x)
    return _gjs_oracle(d.y, X * coef(fit, :mu), d.z, fx.gi, fx.gs,
                       coef(fit, :sigma)[1], sd_a, sd_b)
end

@testset "Gaussian joint mu + sigma ordinary RE, each with SD formula" begin
    fx = _gjs_fixture()
    fit = drm(bf(@formula(y ~ x + (1 | id)),
                 @formula(sigma ~ 1 + (0 + z | site)),
                 @formula(sd(id) ~ w),
                 @formula(sd(site, sigma, coef(z)) ~ v)),
              Gaussian(); data = fx.data)
    @test fit.converged
    @test length(fit.blocks) == 4
    kmu = fit.blocks[3].first; ksg = fit.blocks[4].first
    @test length(coef(fit, kmu)) == 2 && length(coef(fit, ksg)) == 2
    @test maximum(abs.(coef(fit, kmu) .- fx.alpha_mu)) < 0.5
    @test coef(fit, ksg)[2] > 0
    @test maximum(abs.(coef(fit, ksg) .- fx.alpha_sigma)) < 0.6
    orc = _gjs_oracle_from_fit(fit, fx, kmu, ksg)
    @test orc.max_grad < 1e-8
    @test orc.min_eig > 0
    @test abs(loglik(fit) - orc.loglik) < 1e-4
    pp = predict_parameters(fit, fx.data)
    am = coef(fit, kmu)
    @test pp[kmu] ≈ exp.(am[1] .+ am[2] .* fx.data.w) atol = 1e-8
    @test all(isfinite, fit.vcov)
end

@testset "Gaussian joint mu + sigma intercepts, SD formulas, intercept shape" begin
    fx = _gjs_fixture(; seed = 20261102, n_id = 30, n_site = 40, n_each = 24, slope = false)
    fit = drm(bf(@formula(y ~ x + (1 | id)),
                 @formula(sigma ~ 1 + (1 | site)),
                 @formula(sd(id) ~ w),
                 @formula(sd(site, sigma) ~ v)),
              Gaussian(); data = fx.data)
    @test fit.converged
    kmu = fit.blocks[3].first; ksg = fit.blocks[4].first
    orc = _gjs_oracle_from_fit(fit, fx, kmu, ksg)
    @test orc.max_grad < 1e-8
    @test orc.min_eig > 0
    @test abs(loglik(fit) - orc.loglik) < 1e-4
    # The oracle is sensitive: both SD predictors attached to reversed group levels
    # shift it by far more than the 1e-4 tolerance.
    d = fx.data
    am = coef(fit, kmu); asg = coef(fit, ksg)
    bad = _gjs_oracle(d.y, hcat(ones(length(d.y)), d.x) * coef(fit, :mu), d.z, fx.gi, fx.gs,
                      coef(fit, :sigma)[1],
                      exp.(am[1] .+ am[2] .* reverse(fx.w_id)),
                      exp.(asg[1] .+ asg[2] .* reverse(fx.v_site)))
    @test abs(bad.loglik - orc.loglik) > 1e-2
    @test sort(collect(keys(ranef(fit)))) == [:id, :site_logsigma]
    @test all(>(0), sigma(fit))
end

@testset "Gaussian joint SD formulas: constant-SD limits match the existing Laplace routes" begin
    fx = _gjs_fixture(; seed = 20261103, n_id = 30, n_site = 30, n_each = 30)
    d = (; fx.data..., g = fx.data.id)
    # Same grouping on mean and sigma: the existing per-group 2x2 Laplace route.
    cfit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1 + (1 | g)),
                  @formula(sd(g) ~ 1), @formula(sd(g, sigma) ~ 1)), Gaussian(); data = d)
    base = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1 + (1 | g))),
               Gaussian(); data = d)
    @test cfit.converged && base.converged
    @test abs(loglik(cfit) - loglik(base)) < 1e-5
    @test coef(cfit, :mu) ≈ coef(base, :mu) atol = 1e-3
    @test coef(cfit, :sigma) ≈ coef(base, :sigma) atol = 1e-3
    # Sigma-only: the Laplace marginal of the plain sigma random-intercept route.
    sfit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + (1 | g)),
                  @formula(sd(g, sigma) ~ 1)), Gaussian(); data = d)
    sbase = drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + (1 | g))),
                Gaussian(); data = d, marginal = :Laplace)
    @test sfit.converged
    @test abs(loglik(sfit) - loglik(sbase)) < 1e-5
    @test coef(sfit, :mu) ≈ coef(sbase, :mu) atol = 1e-3
    # A genuine SD regression nests the constant model: logLik cannot be lower.
    wfit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + (1 | g)),
                  @formula(sd(g, sigma) ~ w)), Gaussian(); data = d)
    @test loglik(wfit) >= loglik(sfit) - 1e-6
end

@testset "Gaussian joint SD formulas: refusals" begin
    fx = _gjs_fixture(; seed = 20261104, n_id = 10, n_site = 20, n_each = 8)
    d = fx.data
    full(; kw...) = drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (0 + z | site)),
                           @formula(sd(id) ~ w), @formula(sd(site, sigma, coef(z)) ~ v)),
                        Gaussian(); data = d, kw...)
    # Predictors must be constant within group (both sides).
    w = copy(d.w); w[2] += 0.1
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (0 + z | site)),
        @formula(sd(id) ~ w), @formula(sd(site, sigma, coef(z)) ~ v)), Gaussian();
        data = (; d..., w))
    v = copy(d.v); v[3] += 0.1
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (0 + z | site)),
        @formula(sd(id) ~ w), @formula(sd(site, sigma, coef(z)) ~ v)), Gaussian();
        data = (; d..., v))
    # REML, missing responses, wrong group, missing coef selector.
    @test_throws ArgumentError full(; method = :REML)
    y = copy(d.y); y[3] = NaN
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (0 + z | site)),
        @formula(sd(id) ~ w), @formula(sd(site, sigma, coef(z)) ~ v)), Gaussian();
        data = (; d..., y))
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (0 + z | site)),
        @formula(sd(id) ~ w), @formula(sd(id, sigma, coef(z)) ~ v)), Gaussian(); data = d)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (0 + z | site)),
        @formula(sd(id) ~ w), @formula(sd(site, sigma) ~ v)), Gaussian(); data = d)
    # A mean random effect needs its own SD formula in this route.
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (0 + z | site)),
        @formula(sd(site, sigma, coef(z)) ~ v)), Gaussian(); data = d)
    # Mean-side sd() with an unsupported sigma shape keeps its refusal.
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (1 | site)),
        @formula(sd(id) ~ w)), Gaussian(); data = d)
    # Dense engine size cap.
    big = _gjs_fixture(; seed = 20261105, n_id = 5, n_site = 700, n_each = 140)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (1 | site)),
        @formula(sd(id) ~ 1), @formula(sd(site, sigma) ~ 1)), Gaussian(); data = big.data)
end
