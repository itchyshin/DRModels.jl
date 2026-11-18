# Gaussian sigma-side structured intercept with an SD formula (twin of drmTMB
# test-gaussian-sigma-structured-sd.R, animal / relmat providers):
#   y ~ x,  sigma ~ 1 + animal(1 | id, A) | relmat(1 | id, K)
#   sd(id, <level>, sigma) ~ z
#   eta_i = s0 + exp(alpha0 + alpha1 z_g) u_g,  u ~ N(0, K)  (K raw, not normalised)
# The marginal is non-Gaussian; the oracle is an independent dense Laplace over u
# (own mode finder and derivatives; no DRModels internals) at the fitted
# parameters. Tolerances fixed before the first run: |logLik - oracle| < 1e-4,
# oracle mode |grad|_inf < 1e-8 with PD Hessian, constant-SD limit logLik 1e-5
# against the same fit with `sd ~ 1` evaluated at the oracle, recovery
# |alpha_hat - alpha| < 0.8 (n_g = 48 groups; loose, scope is likelihood).
using DRModels, Random, Test, LinearAlgebra

function _gss_fixture(; seed = 2026110103, n_g = 48, n_each = 20)
    rng = MersenneTwister(seed)
    diagonal = collect(range(0.7, 1.4; length = n_g))
    K = [0.3^abs(i - j) * sqrt(diagonal[i] * diagonal[j]) for i in 1:n_g, j in 1:n_g]
    z = collect(range(-1, 1; length = n_g))
    u = cholesky(Symmetric(K)).L * randn(rng, n_g)
    alpha = [log(0.45), 0.5]
    group = repeat(1:n_g; inner = n_each)
    x = randn(rng, length(group))
    scale = exp.(alpha[1] .+ alpha[2] .* z)
    y = 0.2 .+ 0.5 .* x .+ exp.(log(0.5) .+ scale[group] .* u[group]) .* randn(rng, length(group))
    id = [string("s", lpad(string(g), 2, '0')) for g in group]
    return (; data = (; y, x, id, z = z[group]), K, z, group, alpha)
end

function _gss_oracle(y, mu0, K, group, s0, scale)
    m = size(K, 1)
    P = inv(K); P = (P + P') / 2
    ldK = logdet(cholesky(Symmetric(K)))
    eta(u) = s0 .+ scale[group] .* u[group]
    f(u) = begin
        e = eta(u); r = y .- mu0
        sum(0.5 * log(2π) .+ e .+ 0.5 .* r .^ 2 .* exp.(-2 .* e)) +
            0.5 * dot(u, P * u) + 0.5 * ldK + 0.5 * m * log(2π)
    end
    grad(u) = begin
        e = eta(u); r = y .- mu0
        w = 1 .- r .^ 2 .* exp.(-2 .* e)
        g = zeros(m)
        for i in eachindex(y); g[group[i]] += scale[group[i]] * w[i]; end
        g .+ P * u
    end
    hess(u) = begin
        e = eta(u); r = y .- mu0
        H = copy(P)
        for i in eachindex(y)
            H[group[i], group[i]] += scale[group[i]]^2 * 2 * r[i]^2 * exp(-2 * e[i])
        end
        H
    end
    u = zeros(m)
    for _ in 1:5000                        # damped Newton to the coarse basin
        g = grad(u); maximum(abs, g) < 1e-3 && break
        H = hess(u)
        step = H \ g
        t = 1.0; f0 = f(u)
        while t > 1e-10 && f(u .- t .* step) > f0; t /= 2; end
        u .-= t .* step
    end
    for _ in 1:50                          # pure Newton polish
        g = grad(u); maximum(abs, g) < 1e-11 && break
        u .-= hess(u) \ g
    end
    H = hess(u)
    return (; loglik = -f(u) + 0.5 * m * log(2π) - 0.5 * logdet(cholesky(Symmetric(H))),
            max_grad = maximum(abs, grad(u)),
            min_eig = minimum(eigvals(Symmetric(H))))
end

function _gss_fit(fx, kind; constant = false)
    d = fx.data; K = fx.K
    if kind === :animal
        return drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + animal(1 | id)),
                      constant ? @formula(sd(id, animal, sigma) ~ 1) :
                                 @formula(sd(id, animal, sigma) ~ z)),
                   Gaussian(); data = d, A = K)
    end
    return drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + relmat(1 | id)),
                  constant ? @formula(sd(id, relmat, sigma) ~ 1) :
                             @formula(sd(id, relmat, sigma) ~ z)),
               Gaussian(); data = d, K = K)
end

@testset "Gaussian sigma $kind structured SD regression" for (kind, seed) in
        ((:animal, 2026110102), (:relmat, 2026110103))
    fx = _gss_fixture(; seed)
    fit = _gss_fit(fx, kind)
    @test fit.converged
    @test fit.blocks[3].first === :sd
    a = coef(fit, :sd)
    @test length(a) == 2
    @test abs(a[2] - fx.alpha[2]) < 0.8
    X = hcat(ones(length(fx.data.y)), fx.data.x)
    orc = _gss_oracle(fx.data.y, X * coef(fit, :mu), fx.K, fx.group,
                      coef(fit, :sigma)[1], exp.(a[1] .+ a[2] .* fx.z))
    @test orc.max_grad < 1e-8
    @test orc.min_eig > 0
    @test abs(loglik(fit) - orc.loglik) < 1e-4
    pp = predict_parameters(fit, fx.data)
    @test pp[:sd] ≈ exp.(a[1] .+ a[2] .* fx.data.z) atol = 1e-8
    @test all(isfinite, fit.vcov)
end

@testset "Gaussian sigma structured SD: oracle sensitivity and constant-SD nesting" begin
    fx = _gss_fixture(; seed = 2026110105)
    fit = _gss_fit(fx, :relmat)
    const_fit = _gss_fit(fx, :relmat; constant = true)
    @test fit.converged && const_fit.converged
    X = hcat(ones(length(fx.data.y)), fx.data.x)
    a = coef(fit, :sd)
    scale = exp.(a[1] .+ a[2] .* fx.z)
    good = _gss_oracle(fx.data.y, X * coef(fit, :mu), fx.K, fx.group, coef(fit, :sigma)[1], scale)
    # Misaligned levels (SD predictor reversed) and a non-raw (normalised-diagonal) K
    # both shift the oracle by far more than the 1e-4 tolerance.
    swapped = _gss_oracle(fx.data.y, X * coef(fit, :mu), fx.K, fx.group, coef(fit, :sigma)[1],
                          reverse(scale))
    @test abs(swapped.loglik - good.loglik) > 1e-2
    Kn = fx.K ./ sqrt.(diag(fx.K) * diag(fx.K)')
    normed = _gss_oracle(fx.data.y, X * coef(fit, :mu), Kn, fx.group, coef(fit, :sigma)[1], scale)
    @test abs(normed.loglik - good.loglik) > 1e-2
    # Constant SD fit also matches the oracle; the SD regression nests it.
    ac = coef(const_fit, :sd)
    @test length(ac) == 1
    oc = _gss_oracle(fx.data.y, X * coef(const_fit, :mu), fx.K, fx.group,
                     coef(const_fit, :sigma)[1], fill(exp(ac[1]), length(fx.z)))
    @test oc.max_grad < 1e-8
    @test abs(loglik(const_fit) - oc.loglik) < 1e-4
    @test loglik(fit) >= loglik(const_fit) - 1e-6
end

@testset "Gaussian sigma structured SD: refusals" begin
    fx = _gss_fixture(; seed = 2026110106, n_g = 12, n_each = 8)
    d = fx.data; K = fx.K
    # Phylo and spatial sigma effects are fitted (test_gaussian_sigma_phylo_spatial_sd.jl)
    # but need their own inputs: a tree, coordinates.
    @test_throws ArgumentError drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + phylo(1 | id)),
        @formula(sd(id, phylogenetic, sigma) ~ z)), Gaussian(); data = d)
    @test_throws ArgumentError drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + spatial(1 | id)),
        @formula(sd(id, spatial, sigma) ~ z)), Gaussian(); data = d)
    # Level / provider mismatch, missing matrix, missing response (REML now fits: see
    # test_gaussian_structured_sigma_reml.jl).
    @test_throws ArgumentError drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + relmat(1 | id)),
        @formula(sd(id, animal, sigma) ~ z)), Gaussian(); data = d, K = K)
    @test_throws ArgumentError drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + relmat(1 | id)),
        @formula(sd(id, relmat, sigma) ~ z)), Gaussian(); data = d)
    y = copy(d.y); y[2] = NaN
    @test_throws ArgumentError drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + relmat(1 | id)),
        @formula(sd(id, relmat, sigma) ~ z)), Gaussian(); data = (; d..., y), K = K)
    # Wrong-size and non-positive-definite K.
    @test_throws ArgumentError drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + relmat(1 | id)),
        @formula(sd(id, relmat, sigma) ~ z)), Gaussian(); data = d, K = K[1:5, 1:5])
    Kbad = copy(K); Kbad[1, 1] = -1.0
    @test_throws ArgumentError drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + relmat(1 | id)),
        @formula(sd(id, relmat, sigma) ~ z)), Gaussian(); data = d, K = Kbad)
    # Predictor varying within a level is refused.
    z = copy(d.z); z[2] += 0.2
    @test_throws ArgumentError drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + relmat(1 | id)),
        @formula(sd(id, relmat, sigma) ~ z)), Gaussian(); data = (; d..., z), K = K)
end

@testset "Gaussian sigma animal SD: A matched to levels by label" begin
    fx = _gss_fixture(; seed = 2026110107, n_g = 24, n_each = 12)
    ids = unique(fx.data.id)
    perm = reverse(1:24)
    Ashuf = fx.K[perm, perm]
    form = bf(@formula(y ~ x), @formula(sigma ~ 1 + animal(1 | id)),
              @formula(sd(id, animal, sigma) ~ z))
    base = drm(form, Gaussian(); data = fx.data, A = fx.K, ids = ids)
    shuf = drm(form, Gaussian(); data = fx.data, A = Ashuf, ids = ids[perm])
    @test abs(loglik(base) - loglik(shuf)) < 1e-6
    @test coef(base, :sd) ≈ coef(shuf, :sd) atol = 1e-4
end
