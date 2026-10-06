# test_crossed_lognormal.jl — #736 twin gap: LogNormal() refuses crossed random
# effects `(1 | g) + (1 | h)`; drmTMB's `lognormal()` fits them. `log(y)` is
# exactly Gaussian, so LogNormal() delegates crossed random intercepts on the
# mean WHOLESALE to `drm(f, Gaussian(); data = data-with-logged-response)` —
# the same identity already used for `phylo`/`relmat` structured markers on the
# mean (src/lognormal.jl `_fit_lognormal_structured`) — with the reported
# log-likelihood shifted by the parameter-free Jacobian `-sum(log y)`. The
# crossed Gaussian route (`_fit_multi_ranef_gaussian`) is the closed-form exact
# marginal likelihood, not a Laplace approximation, so no new integrator is
# introduced.
using DRModels
using Test, Random

@testset "#736 LogNormal crossed random effects (1|g)+(1|h)" begin

    # Pre-fix red gate (manually confirmed against origin/main before this
    # branch's implementation commit): `drm(bf(@formula(y ~ x + (1 | g) + (1 | h))),
    # LogNormal(); data = dat)` raised `"LogNormal() supports a single
    # random-effect term on the mean"`. Not asserted here as a live @test
    # because the fix below makes it pass by design; the recovery and identity
    # tests below are the fix's acceptance gate.

    @testset "known-DGP recovery: crossed random intercepts on the log-mean" begin
        Random.seed!(20324905)   # the ADEMP cell 64 seed (issue #736)
        G = 12; H = 10; n = 220
        g = rand(1:G, n); h = rand(1:H, n); x = randn(n)
        β = [0.1, 0.35]; σ = 0.4; σg = 0.276; σh = 0.235
        bg = σg .* randn(G); bh = σh .* randn(H)
        logy = β[1] .+ β[2] .* x .+ bg[g] .+ bh[h] .+ σ .* randn(n)
        y = exp.(logy)
        dat = (; y, x, g, h)

        fit = drm(bf(@formula(y ~ x + (1 | g) + (1 | h))), LogNormal(); data = dat)

        @test fit.converged
        @test coef(fit, :mu)[1] ≈ β[1] atol = 0.15
        @test coef(fit, :mu)[2] ≈ β[2] atol = 0.15
        @test exp(coef(fit, :sigma)[1]) ≈ σ atol = 0.12
        rs = re_sd(fit)
        @test rs[:g] ≈ σg atol = 0.20
        @test rs[:h] ≈ σh atol = 0.20
        @test isfinite(loglik(fit))
    end

    @testset "identity with Gaussian-on-log(y) (same delegation as phylo/relmat)" begin
        Random.seed!(20260928)
        G = 14; H = 9; n = 260
        g = rand(1:G, n); h = rand(1:H, n); x = randn(n)
        β = [0.2, -0.3]; σ = 0.3; σg = 0.4; σh = 0.25
        bg = σg .* randn(G); bh = σh .* randn(H)
        logy = β[1] .+ β[2] .* x .+ bg[g] .+ bh[h] .+ σ .* randn(n)
        y = exp.(logy)

        fit_g = drm(bf(@formula(y ~ x + (1 | g) + (1 | h))),
                    Gaussian(); data = (; y = log.(y), x, g, h))
        fit_ln = drm(bf(@formula(y ~ x + (1 | g) + (1 | h))),
                     LogNormal(); data = (; y, x, g, h))

        sumlogy = sum(log, y)
        @test fit_ln.converged
        @test fit_ln.theta == fit_g.theta
        @test coef(fit_ln, :mu) == coef(fit_g, :mu)
        @test loglik(fit_ln) ≈ loglik(fit_g) - sumlogy atol = 1e-8
        rs_ln = re_sd(fit_ln); rs_g = re_sd(fit_g)
        @test rs_ln[:g] == rs_g[:g]
        @test rs_ln[:h] == rs_g[:h]
    end
end
