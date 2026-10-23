# Sparse-Laplace proving slice for crossed non-Gaussian random effects:
# Poisson y ~ x + (1|g) + (1|h). This is an engine-lane test and deliberately
# calls the internal fitter directly; the public formula routing is Claude-owned.
using DRModels
using Test, Random
import Distributions

@testset "Poisson crossed random intercepts — sparse Laplace recovery" begin
    Random.seed!(20260706)
    G = 20; H = 18; n = 900
    g = rand(1:G, n); h = rand(1:H, n); x = randn(n)
    β = [0.25, 0.45]; σg = 0.45; σh = 0.35
    bg = σg .* randn(G); bh = σh .* randn(H)
    λ = exp.(β[1] .+ β[2] .* x .+ bg[g] .+ bh[h])
    y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])
    X = hcat(ones(n), x)
    gidx, Gfit = DRModels._group_index(g)
    hidx, Hfit = DRModels._group_index(h)
    comps = [(ones(n), gidx, Gfit, "g"), (ones(n), hidx, Hfit, "h")]

    fit = DRModels._fit_poisson_crossed_laplace(DRModels.Poisson(), y, X, comps, ["(Intercept)", "x"], 1e-7)

    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.12
    rs = re_sd(fit)
    @test rs[:g] ≈ σg atol = 0.20
    @test rs[:h] ≈ σh atol = 0.20
    @test isfinite(loglik(fit))
    @test all(fitted(fit) .> 0)
end

@testset "Poisson single-factor Laplace gate uses GHQ path" begin
    Random.seed!(20260707)
    G = 16; m = 25; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.2, 0.35]; σb = 0.5
    bg = σb .* randn(G)
    λ = exp.(β[1] .+ β[2] .* x .+ bg[g])
    y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])
    X = hcat(ones(n), x)
    gidx, Gfit = DRModels._group_index(g)
    comps = [(ones(n), gidx, Gfit, "g")]

    fit_laplace_gate = DRModels._fit_poisson_crossed_laplace(DRModels.Poisson(), y, X, comps, ["(Intercept)", "x"], 1e-8)
    fit_ghq = DRModels._fit_poisson_ranef(DRModels.Poisson(), y, X, gidx, Gfit, ["(Intercept)", "x"], :g, 1e-8)

    @test coef(fit_laplace_gate) ≈ coef(fit_ghq) atol = 1e-10
    @test loglik(fit_laplace_gate) ≈ loglik(fit_ghq) atol = 1e-10
end

@testset "Poisson K=3 crossed — exact Laplace gradient (#165)" begin
    # K ≥ 3 components exercise the generic `_fit_poisson_crossed_laplace`
    # path (K = 1 → GHQ, K = 2 → the dense two-block path). This gate pins the
    # exact implicit-function gradient that replaced the frozen-mode + FD polish.
    Random.seed!(20260709)
    G = 12; H = 10; L = 8; n = 700
    g = rand(1:G, n); h = rand(1:H, n); l = rand(1:L, n); x = randn(n)
    β = [0.2, 0.4]; σg = 0.45; σh = 0.35; σl = 0.30
    bg = σg .* randn(G); bh = σh .* randn(H); bl = σl .* randn(L)
    λ = exp.(β[1] .+ β[2] .* x .+ bg[g] .+ bh[h] .+ bl[l])
    y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])
    X = hcat(ones(n), x)
    gidx, Gfit = DRModels._group_index(g)
    hidx, Hfit = DRModels._group_index(h)
    lidx, Lfit = DRModels._group_index(l)
    comps = [(ones(n), gidx, Gfit, "g"),
             (ones(n), hidx, Hfit, "h"),
             (ones(n), lidx, Lfit, "l")]

    fit = DRModels._fit_poisson_crossed_laplace(DRModels.Poisson(), y, X, comps,
                                           ["(Intercept)", "x"], 1e-7)

    # Recovery (loose — three small grouping factors).
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.15
    rs = re_sd(fit)
    @test rs[:g] ≈ σg atol = 0.22
    @test rs[:h] ≈ σh atol = 0.22
    @test rs[:l] ≈ σl atol = 0.22
    @test isfinite(loglik(fit))

    # Exact-gradient gate: evaluate OFF the optimum so the implicit (db̂/dθ)
    # terms are nonzero and a frozen-mode gradient would fail.
    θ̂ = coef(fit)
    θtest = θ̂ .+ [0.10, -0.08, 0.12, -0.10, 0.09]
    g_an = zeros(length(θtest))
    fit.nllgrad(g_an, θtest)
    h = 1e-5
    g_fd = similar(g_an)
    for k in eachindex(θtest)
        e = zeros(length(θtest)); e[k] = h
        g_fd[k] = (fit.nll(θtest .+ e) - fit.nll(θtest .- e)) / (2h)
    end
    @test g_an ≈ g_fd rtol = 1e-4 atol = 1e-4

    # Converged is not for sale on this route either (2026-08-27 audit catch):
    # the generic crossed path was the ONE fit function still reporting raw
    # `Optim.converged(res)` — invisible to the #491 sweep because it never
    # called `_laplace_outer_converged` at all. The honest default fit reports
    # converged; a deliberately sloppy g_tol must not.
    @test fit.converged
    fit_sloppy = DRModels._fit_poisson_crossed_laplace(DRModels.Poisson(), y, X, comps,
                                                  ["(Intercept)", "x"], 10.0)
    @test !fit_sloppy.converged
end

@testset "Poisson crossed intercepts via drm() routing" begin
    Random.seed!(20260708)
    G = 20; H = 16; n = 800
    g = rand(1:G, n); h = rand(1:H, n); x = randn(n)
    β = [0.3, 0.4]; σg = 0.5; σh = 0.4
    bg = σg .* randn(G); bh = σh .* randn(H)
    λ = exp.(β[1] .+ β[2] .* x .+ bg[g] .+ bh[h])
    y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])
    dat = (; y, x, g, h)
    fit = drm(bf(@formula(y ~ x + (1 | g) + (1 | h))), Poisson(); data = dat)
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.12
    rs = re_sd(fit)
    @test rs[:g] ≈ σg atol = 0.20
    @test rs[:h] ≈ σh atol = 0.20
    @test isfinite(loglik(fit))
end
