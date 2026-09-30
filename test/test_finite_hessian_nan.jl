# test_finite_hessian_nan.jl -- `_finite_hessian` fails to a NaN vcov, not a ridge.
#
# Old behaviour: on a non-finite Hessian it warned and added 1e12 to the diagonal, so
# the callers reported near-zero, fabricated SEs; and it never treated the 1e18
# failed-fit objective sentinel as a failure (differencing it gives a huge finite
# garbage Hessian). New behaviour (owner decision 14, 2026-09-30): an all-NaN Hessian
# of the same shape plus a warning, which `_vcov_from_hessian` passes through as an
# all-NaN vcov (the NaN-vcov convention of the other vcov_guard helpers).
# Healthy fits are unchanged: the stencil is identical.

using DRModels
using Test, LinearAlgebra, Random, SparseArrays
import Distributions

# Verbatim copy of the pre-change stencil WITHOUT any failure handling (healthy fits
# never reach the ridge), so healthy H can be compared to 1e-12 against the old value.
function _fhn_old_healthy_hessian(f, x; h::Real = 1e-4)
    n = length(x); H = zeros(n, n); fx = f(x)
    hs = [max(h, h * (1 + abs(x[i]))) for i in 1:n]
    for i in 1:n
        ei = zeros(n); ei[i] = hs[i]
        H[i, i] = (f(x .+ ei) - 2fx + f(x .- ei)) / hs[i]^2
        for j in (i+1):n
            ej = zeros(n); ej[j] = hs[j]
            H[i, j] = (f(x .+ ei .+ ej) - f(x .+ ei .- ej) -
                       f(x .- ei .+ ej) + f(x .- ei .- ej)) / (4 * hs[i] * hs[j])
            H[j, i] = H[i, j]
        end
    end
    return H
end

# Route 1: Poisson crossed-intercept sparse Laplace (`_fit_poisson_crossed_laplace`).
function _fhn_poisson_crossed()
    Random.seed!(20260930)
    G = 14; H = 12; n = 500
    g = rand(1:G, n); h = rand(1:H, n); x = randn(n)
    bg = 0.45 .* randn(G); bh = 0.35 .* randn(H)
    λ = exp.(0.25 .+ 0.45 .* x .+ bg[g] .+ bh[h])
    y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])
    X = hcat(ones(n), x)
    gidx, Gf = DRModels._group_index(g); hidx, Hf = DRModels._group_index(h)
    comps = [(ones(n), gidx, Gf, "g"), (ones(n), hidx, Hf, "h")]
    fit = DRModels._fit_poisson_crossed_laplace(DRModels.Poisson(), y, X, comps,
                                                ["(Intercept)", "x"], 1e-7)
    return fit, n
end

# Route 2: bivariate q=2 structured relmat (`gaussian_bivariate.jl`).
function _fhn_q2_relmat()
    rng = MersenneTwister(20260930)
    G = 10; nrep = 4
    K = [0.55 ^ abs(i - j) for i in 1:G, j in 1:G] + 1e-6I
    Q = sparse(Matrix(inv(cholesky(Symmetric(K)))))
    Λ = [0.20 0.05; 0.05 0.17]
    P = DRModels.prior_precision(Q, inv(Λ))
    u = cholesky(Symmetric(P)).UP \ randn(rng, size(P, 1))
    group = repeat(1:G, inner = nrep); n = length(group); x = randn(rng, n)
    L = cholesky(Symmetric([0.10 0.025; 0.025 0.14])).L
    Y = zeros(n, 2)
    for i in 1:n
        b = 2 * (group[i] - 1)
        Y[i, 1] = 0.2 + 0.25x[i] + u[b + 1]
        Y[i, 2] = -0.15 + 0.10x[i] + u[b + 2]
        Y[i, :] .+= L * randn(rng, 2)
    end
    dat = (; y1 = Y[:, 1], y2 = Y[:, 2], x = x, grp = ["g$(i)" for i in group])
    form = bf(mu1 = @formula(y1 ~ x + relmat(1 | grp)),
              mu2 = @formula(y2 ~ x + relmat(1 | grp)),
              sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
              rho12 = @formula(rho12 ~ 1))
    fit = drm(form, Gaussian(); data = dat, K = K, g_tol = 1e-6, method = :ML)
    return fit, n
end

@testset "_finite_hessian: failure -> all-NaN vcov (no 1e12 ridge)" begin
    quad = x -> 0.5 * (4.0 * x[1]^2 + 9.0 * x[2]^2 + x[1] * x[2])
    x0 = [0.3, -0.2]

    @testset "healthy quadratic: finite H, no warning" begin
        H = @test_logs DRModels._finite_hessian(quad, x0)
        @test all(isfinite, H)
        @test H ≈ [4.0 0.5; 0.5 9.0] atol = 1e-5
    end

    @testset "$name -> NaN H + warning, of the same shape" for (name, bad) in (
            "objective at the evaluation point is the 1e18 sentinel" => (x -> 1e18),
            "objective at the evaluation point is NaN" => (x -> NaN),
            "a stencil probe is NaN (old path: ridge)" =>
                (x -> x[1] > x0[1] + 1e-8 ? NaN : quad(x)),
            "a stencil probe is the 1e18 sentinel (old path: garbage finite H)" =>
                (x -> x[2] > x0[2] + 1e-8 ? 1e18 : quad(x)))
        H = @test_logs (:warn, r"reporting a NaN vcov") DRModels._finite_hessian(bad, x0)
        @test size(H) == (2, 2)
        @test all(isnan, H)
        V = DRModels._vcov_from_hessian(H; context = "test")
        @test size(V) == (2, 2)
        @test all(isnan, V)
    end
end

@testset "_finite_hessian NaN: ordinary-Laplace caller does not throw" begin
    nll = θ -> 1e18
    V = @test_logs (:warn, r"reporting a NaN vcov") DRModels._ordinary_laplace_vcov(
        nll, [0.1, 0.2, 0.3], 100; se = true, context = "test")
    @test V isa Matrix{Float64}
    @test size(V) == (3, 3) && all(isnan, V)
end

@testset "route: Poisson crossed sparse Laplace" begin
    fit, n = _fhn_poisson_crossed()
    θ̂ = fit.theta; h = DRModels._fd_hessian_step(n)
    # healthy: same stencil as before. This objective is STATEFUL (the inner Newton
    # solve warm-starts from the previous evaluation), so two evaluations of the same
    # stencil differ at ~1e-9 relative; the bit-for-bit (1e-12) comparison against
    # main's vcov(fit) was done across two checkouts on Totoro (see the PR).
    Hold = _fhn_old_healthy_hessian(fit.nll, θ̂; h = h)
    Hnew = DRModels._finite_hessian(fit.nll, θ̂; h = h)
    @test maximum(abs, Hnew .- Hold) <= 1e-7 * maximum(abs, Hold)
    @test all(isfinite, vcov(fit))
    Vold = DRModels._vcov_from_hessian(Hold)
    @test maximum(abs, vcov(fit) .- Vold) <= 1e-5 * maximum(abs, Vold)
    # degenerate: the same objective, failing (sentinel) past a clamp-like boundary
    bad = θ -> θ[end] > θ̂[end] + 1e-9 ? 1e18 : fit.nll(θ)
    H = @test_logs (:warn, r"reporting a NaN vcov") DRModels._finite_hessian(bad, θ̂; h = h)
    @test all(isnan, H)
    V = DRModels._vcov_from_hessian(H; context = "sparse-Laplace Poisson (crossed)")
    @test size(V) == (length(θ̂), length(θ̂)) && all(isnan, V)
end

@testset "route: bivariate q=2 structured relmat" begin
    fit, n = _fhn_q2_relmat()
    θ̂ = fit.theta; h = DRModels._fd_hessian_step(2n)
    Hold = _fhn_old_healthy_hessian(fit.nll, θ̂; h = h)
    Hnew = DRModels._finite_hessian(fit.nll, θ̂; h = h)
    @test maximum(abs, Hnew .- Hold) <= 1e-12 * max(1, maximum(abs, Hold))
    Vold = DRModels._vcov_from_hessian(Hold)
    @test maximum(abs, vcov(fit) .- Vold) <= 1e-12 * max(1, maximum(abs, Vold))
    bad = θ -> θ[1] < θ̂[1] - 1e-9 ? NaN : fit.nll(θ)   # old path: ridge -> SE ~ 0
    H = @test_logs (:warn, r"reporting a NaN vcov") DRModels._finite_hessian(bad, θ̂; h = h)
    @test all(isnan, H)
    V = DRModels._vcov_from_hessian(H; context = "bivariate Gaussian q=2 structured")
    @test all(isnan, V)
end
