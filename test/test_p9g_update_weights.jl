# P9g: `update` on a fit made with prior weights must not refit unweighted by accident.
# drmTMB's `update()` re-runs the original call, so it keeps the weights. DRModels.update
# takes new `data`, so the caller says which weights go with it: `weights = weights(fit)`,
# new weights, or `weights = nothing` to drop them on purpose.
using DRModels
using Test, Random

@testset "update() with prior weights needs weights= restated (P9g)" begin
    Random.seed!(20261112)
    n = 200
    x = randn(n)
    y = 0.4 .+ 0.7 .* x .+ 0.8 .* randn(n)
    w = 0.5 .+ rand(n)
    data = (; y, x)
    f_full = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
    f_red = bf(@formula(y ~ 1), @formula(sigma ~ 1))

    wfit = drm(f_full, Gaussian(); data = data, weights = w)
    ufit = drm(f_full, Gaussian(); data = data)

    # Fails on main: the refit succeeded and silently dropped the weights.
    err = try
        update(wfit, f_red; data = data)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("weights = weights(fit)", sprint(showerror, err))
    @test occursin("weights = nothing", sprint(showerror, err))

    # Restating the weights reproduces the weighted refit.
    re_w = update(wfit, f_red; data = data, weights = weights(wfit))
    direct_w = drm(f_red, Gaussian(); data = data, weights = w)
    @test weights(re_w) == w
    @test loglik(re_w) ≈ loglik(direct_w) atol = 1e-10
    @test coef(re_w) ≈ coef(direct_w) atol = 1e-8

    # `weights = nothing` drops them on purpose and gives the unweighted fit.
    re_u = update(wfit, f_red; data = data, weights = nothing)
    direct_u = drm(f_red, Gaussian(); data = data)
    @test weights(re_u) === nothing
    @test loglik(re_u) ≈ loglik(direct_u) atol = 1e-10

    # New weights for new rows are accepted.
    w2 = reverse(w)
    re_new = update(wfit, f_red; data = data, weights = w2)
    @test weights(re_new) == w2

    # An unweighted seed fit is unaffected: no weights keyword needed.
    re_plain = update(ufit, f_red; data = data)
    @test weights(re_plain) === nothing
    @test loglik(re_plain) ≈ loglik(direct_u) atol = 1e-10
end
