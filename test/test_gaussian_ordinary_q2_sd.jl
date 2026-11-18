# Gaussian ordinary correlated q2 `(1 + x | id)` with one SD formula per coefficient
# (twin of drmTMB test-gaussian-correlated-q2-sd.R). Same code as the q3/q4 route, q = 2.
# Oracle: dense per-group MvNormal at the fitted parameters (atol 1e-6); nesting vs the
# existing plain `(1 + x | id)` fit (1e-6); loose recovery; refusals.
using DRModels, Test, Random, LinearAlgebra
import Distributions

function _q2sd_fixture(; G = 60, m = 8, seed = 20261108)
    rng = MersenneTwister(seed)
    rho = 0.4; a0 = [-0.5, -0.8]; a1 = [0.5, -0.4]
    w = collect(range(-1.5, 1.5; length = G))[randperm(rng, G)]
    n = G * m; gi = repeat(1:G, inner = m)
    x = randn(rng, n)
    y = 0.2 .+ 0.5 .* x .+ 0.4 .* randn(rng, n)
    Rm = [1 rho; rho 1.0]
    for g in 1:G
        D = Diagonal(exp.(a0 .+ a1 .* w[g]))
        b = cholesky(Symmetric(D * Rm * D)).L * randn(rng, 2)
        rows = ((g-1)*m+1):(g*m)
        y[rows] .+= b[1] .+ x[rows] .* b[2]
    end
    data = (; y, x, w = w[gi], id = ["g$(g)" for g in gi])
    return (; data, gi, w, G, rho, a0, a1)
end

function _q2sd_oracle(fx, beta, sigma, A, rho)
    ll = 0.0
    for g in 1:fx.G
        rows = findall(==(g), fx.gi)
        U = hcat(ones(length(rows)), fx.data.x[rows])
        D = Diagonal(exp.(A[:, 1] .+ A[:, 2] .* fx.w[g]))
        V = U * (D * [1 rho; rho 1] * D) * U' + sigma^2 * I
        ll += Distributions.logpdf(Distributions.MvNormal(U * beta, Symmetric(V)), fx.data.y[rows])
    end
    return ll
end

# one theta -> correlation: L = [1 0; t 1], R12 = t / sqrt(1 + t^2)
_q2sd_rho(theta) = theta / sqrt(1 + theta^2)

@testset "Gaussian ordinary q2 correlated block with per-coefficient SD formulas" begin
    fx = _q2sd_fixture(); d = fx.data
    sdI = @formula(sd(id, coef(intercept)) ~ w); sdx = @formula(sd(id, coef(x)) ~ w)
    f2(extra...) = bf(@formula(y ~ x + (1 + x | id)), @formula(sigma ~ 1), extra...)

    fit = drm(f2(sdI, sdx), Gaussian(); data = d)
    @test fit.converged
    keys_ = [p.first for p in fit.blocks if p.first ∉ (:mu, :sigma, :recov)]
    A = reduce(vcat, [coef(fit, k)' for k in keys_])
    @test size(A) == (2, 2)
    rho = _q2sd_rho(coef(fit, :recov)[1])
    beta = coef(fit, :mu); sigma = exp(coef(fit, :sigma)[1])
    orc = _q2sd_oracle(fx, beta, sigma, A, rho)
    @test abs(loglik(fit) - orc) < 1e-6
    @test abs(orc - _q2sd_oracle(fx, beta, sigma, A[[2, 1], :], rho)) > 0.5   # swapped SD columns
    @test abs(orc - _q2sd_oracle(fx, beta, sigma, A, 0.0)) > 0.5               # R = I
    @test all(abs.(A[:, 2] .- fx.a1) .< 1.0) && all(abs.(A[:, 1] .- fx.a0) .< 1.0)
    @test abs(rho - fx.rho) < 0.4
    @test all(isfinite, fit.vcov)

    # nesting: constant SD on both coefficients equals the existing plain q2 fit
    fc = drm(f2(@formula(sd(id, coef(intercept)) ~ 1), @formula(sd(id, coef(x)) ~ 1)),
             Gaussian(); data = d)
    fplain = drm(bf(@formula(y ~ x + (1 + x | id)), @formula(sigma ~ 1)), Gaussian(); data = d)
    @test fc.converged
    @test abs(loglik(fc) - loglik(fplain)) < 1e-6
    @test loglik(fit) ≥ loglik(fc) - 1e-6

    # one SD formula only (intercept or slope): the other SD stays constant, as in native R
    for s in (sdI, sdx)
        f1 = drm(f2(s), Gaussian(); data = d)
        @test f1.converged
        @test loglik(fc) - 1e-6 ≤ loglik(f1) ≤ loglik(fit) + 1e-6
    end

    @testset "refusals and unchanged routes" begin
        @test_throws ArgumentError drm(f2(sdI, @formula(sd(id, coef(intercept)) ~ 1)), Gaussian(); data = d)
        @test_throws ArgumentError drm(f2(sdI, @formula(sd(id, coef(v)) ~ w)), Gaussian(); data = d)
        @test_throws ArgumentError drm(f2(sdI, sdx), Gaussian(); data = d, method = :REML)
        dv = merge(d, (w = randn(length(d.y)),))
        @test_throws ArgumentError drm(f2(sdI, sdx), Gaussian(); data = dv)
        d5 = merge(d, (z = randn(length(d.y)), u = randn(length(d.y)), v = randn(length(d.y))))
        @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 + x + z + u + v | id)),
            @formula(sigma ~ 1), sdI), Gaussian(); data = d5)
        # unlabelled sd(id) on a slope block is still the old intercept-only refusal
        @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 + x | id)), @formula(sigma ~ 1),
            @formula(sd(id) ~ w)), Gaussian(); data = d)
        # intercept-only route unchanged
        fi = drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1), @formula(sd(id) ~ w)),
                 Gaussian(); data = d)
        @test fi.converged
    end
end
