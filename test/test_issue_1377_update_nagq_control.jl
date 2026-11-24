# #1377: `update` and a bootstrap refit dropped `nAGQ` on Poisson `(1 | g)`
# `marginal = :AGHQ`, and `update` dropped a non-default `control`.
using DRModels
using Test

@testset "#1377 update and bootstrap keep nAGQ; update restates control" begin
    G, m = 12, 4
    g = repeat(1:G, inner = m)
    y = [Float64((i % 5) + (g[i] % 4)) for i in eachindex(g)]
    d = (; y, g)
    f = bf(@formula(y ~ 1 + (1 | g)))
    fit3 = drm(f, Poisson(); data = d, marginal = :AGHQ, nAGQ = 3, se = false)
    fit5 = drm(f, Poisson(); data = d, marginal = :AGHQ, nAGQ = 5, se = false)
    @test fit3.scales[:aghq_nodes] == [3.0]
    @test abs(loglik(fit3) - loglik(fit5)) > 1e-4
    @test DRModels._update_stored_options(fit3).nAGQ == 3
    @test DRModels._bootstrap_refit_kwargs(fit3).nAGQ == 3
    upd = update(fit3, f; data = d, se = false)
    @test loglik(upd) ≈ loglik(fit3)
    @test abs(loglik(upd) - loglik(fit5)) > 1e-4

    n = 30
    x = [sin(i) for i in 1:n]
    yg = [0.3 + 0.4 * x[i] + 0.15 * cos(2i) for i in 1:n]
    dg = (; y = yg, x)
    fg = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
    ctl = DRMControl(se = false, maxit = 50)
    fitc = drm(fg, Gaussian(); data = dg, control = ctl)
    ex = try
        update(fitc, fg; data = dg)
        nothing
    catch e
        e
    end
    @test ex isa ArgumentError
    @test occursin("control", ex.msg)
    kept = update(fitc, fg; data = dg, control = ctl)
    @test all(isnan, vcov(kept))
    fresh = update(fitc, fg; data = dg, control = nothing)
    @test all(isfinite, vcov(fresh))
    bare = drm(fg, Gaussian(); data = dg)
    @test coef(update(bare, fg; data = dg)) ≈ coef(bare)
end
