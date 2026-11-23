# #936 (twin of drmTMB #1751 / PR #1786, owner ruling SHINICHI-0900-1119): the simple
# Gaussian / Student-t fit (fixed effects plus an ordinary `(1 | g)`, ML) is fitted on
# y / sd(y) when sd(y) is outside [0.01, 100], then mapped back. The reported fit must
# be the original-scale fit: coefficients and the log SD of (1 | g) move by the map
# below, SEs by the same Jacobian, the log-likelihood by -n log(k), and the flags do
# not depend on the units of y.
using DRModels
using Test, Random, StableRNGs, LinearAlgebra, Statistics

function _data_936(rep; n = 240, G = 24, nu = nothing)
    rng = StableRNG(936_000 + rep)
    g = repeat(1:G, inner = n ÷ G)
    x = randn(rng, n)
    u = 0.7 .* randn(rng, G)
    e = nu === nothing ? randn(rng, n) :
        randn(rng, n) ./ sqrt.([sum(abs2, randn(rng, nu)) / nu for _ in 1:n])
    y = 2.0 .+ 0.5 .* x .+ u[g] .+ exp.(0.2 .+ 0.1 .* x) .* e
    return (; y, x, g = string.(g))
end

_scale_data_936(d, k) = (; y = d.y .* k, x = d.x, g = d.g)

# Original-scale fit of y * k from the fit of y: parameter map and Jacobian.
function _map_936(fit, k)
    b = Dict(fit.blocks); nm = Dict(fit.coefnames)
    p = length(fit.theta)
    M = ones(p); M[b[:mu]] .= k
    sh = Int[first(b[:sigma]) + findfirst(==("(Intercept)"), nm[:sigma]) - 1]
    haskey(b, :resd) && push!(sh, first(b[:resd]))
    θ = fit.theta .* M; θ[sh] .+= log(k)
    return θ, fit.vcov .* (M * M')
end

function _check_936(f, fam, d; ks = (1e3, 1e-6, 1e6), rtol = 1e-5)
    base = drm(f, fam; data = d)
    @test std(d.y) > 1e-2 && std(d.y) < 1e2          # the reference fit is the ordinary path
    n = length(d.y)
    for k in ks
        fk = drm(f, fam; data = _scale_data_936(d, k))
        θ, V = _map_936(base, k)
        scl = max.(abs.(θ), 1.0)
        @test maximum(abs.(fk.theta .- θ) ./ scl) < rtol
        @test isapprox(fk.loglik, base.loglik - n * log(k); rtol = 1e-8)
        @test all(isfinite, fk.vcov)
        @test isapprox(sqrt.(diag(fk.vcov)), sqrt.(diag(V)); rtol = 1e-3)
        @test fk.converged == base.converged
        @test isapprox(fk.means[:mu], base.means[:mu] .* k; rtol = 1e-4)
        @test isapprox(fk.obs[:mu], d.y .* k; rtol = 1e-12)
        @test isapprox(fk.scales[:sigma], base.scales[:sigma] .* k; rtol = 1e-4)
        @test isapprox(fk.nll(fk.theta), -fk.loglik; rtol = 1e-8)
    end
    return base
end

@testset "#936 rescale the response inside the fitter" begin
    d = _data_936(1)
    @testset "Gaussian fixed effects, sigma ~ x" begin
        _check_936(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(), d)
    end
    @testset "Gaussian (1 | g)" begin
        base = _check_936(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(), d)
        for k in (1e3, 1e-6)
            fk = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian();
                     data = _scale_data_936(d, k))
            @test isapprox(ranef(fk)[:g], ranef(base)[:g] .* k; rtol = 1e-3, atol = 1e-6 * k)
        end
    end
    @testset "Student-t fixed effects" begin
        dt = _data_936(2; nu = 4)
        _check_936(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Student(), dt;
                   rtol = 1e-4)
    end
    @testset "Student-t (1 | g)" begin
        dt = _data_936(3; nu = 4)
        _check_936(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(nu ~ 1)), Student(), dt;
                   rtol = 1e-3)
    end
    @testset "the rescaled path is the one taken" begin
        f = bf(@formula(y ~ x), @formula(sigma ~ 1))
        @test DRModels._y_rescale_plan(f, Gaussian(), (; data = d), (:data, :g_tol)) === nothing
        for k in (1e3, 1e-6)
            @test DRModels._y_rescale_plan(f, Gaussian(), (; data = _scale_data_936(d, k)),
                                           (:data, :g_tol)) !== nothing
        end
        # Same standardised problem at every scale, so the optimiser takes the same path.
        its = [drm(f, Gaussian(); data = _scale_data_936(d, k)).iterations for k in (1e3, 1e-6, 1e6)]
        @test allequal(its)
        # Outside the slice: a random slope or an `offset` is not planned.
        dk = _scale_data_936(d, 1e6)
        @test DRModels._y_rescale_plan(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1)), Gaussian(),
                                       (; data = dk), (:data, :g_tol)) === nothing
        @test DRModels._y_rescale_plan(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1 + (1 | g))), Gaussian(),
                                       (; data = dk), (:data, :g_tol)) === nothing
    end
    @testset "predict follows the original units" begin
        f = bf(@formula(y ~ x), @formula(sigma ~ 1))
        base = drm(f, Gaussian(); data = d)
        fk = drm(f, Gaussian(); data = _scale_data_936(d, 1e6))
        nd = (; x = [-1.0, 0.0, 1.5])
        @test isapprox(predict(fk, nd), predict(base, nd) .* 1e6; rtol = 1e-5)
    end
    @testset "inside the window and other routes are unchanged" begin
        f = bf(@formula(y ~ x), @formula(sigma ~ 1))
        # sd(y) inside the window: not rescaled, so bit-for-bit the ordinary fit.
        a = drm(f, Gaussian(); data = d); b = drm(f, Gaussian(); data = d)
        @test a.theta == b.theta
        # REML and weights stay on the original scale and still work.
        dk = _scale_data_936(d, 1e6)
        @test drm(f, Gaussian(); data = dk, method = :REML) isa DRModels.DrmFit
        @test drm(f, Gaussian(); data = dk, weights = fill(1.0, length(dk.y))) isa DRModels.DrmFit
    end
end

@testset "rescaled fit: Jacobian counts the fitted rows, not the input rows (#936)" begin
    # A finite response with a missing predictor is dropped by the fitter. The original-scale
    # log-likelihood must then equal the fit on the complete rows (one log(s) term per fitted row).
    y = 1e-3 .* [1.0, 3, 2, 5, 4, 8, 6, 9, 7, 11, 10, 12]
    x = Union{Missing,Float64}[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, missing]
    f = bf(@formula(y ~ x), @formula(sigma ~ 1))
    fit = drm(f, Gaussian(); data = (; y, x))
    keep = .!ismissing.(x)
    ref = drm(f, Gaussian(); data = (; y = y[keep], x = Float64.(x[keep])))
    @test fit.nobs == 11
    @test isapprox(coef(fit), coef(ref); rtol = 1e-6)
    @test isapprox(loglikelihood(fit), loglikelihood(ref); atol = 1e-6)
    @test isapprox(fit.nll(fit.theta), -loglikelihood(ref); atol = 1e-6)
end

@testset "rescale is skipped when a right-hand side reads the response (#936)" begin
    # `sigma ~ 1 + y` builds its design from the response column. Fitting y / s would build
    # it on y / s, and the restore only shifts the sigma intercept, so the slice must not apply.
    rng = StableRNG(936_777)
    n = 80
    x = randn(rng, n)
    y = 1e-3 .* (2.0 .+ 0.5 .* x .+ 0.3 .* randn(rng, n))
    @test std(y) < 1e-2                                   # outside the window: the hook would fire
    f = bf(@formula(y ~ x), @formula(sigma ~ 1 + y))
    @test DRModels._y_rescale_plan(f, Gaussian(), (; data = (; y, x)), (:data, :g_tol)) === nothing
    f0 = bf(@formula(y ~ x), @formula(sigma ~ 1))
    @test DRModels._y_rescale_plan(f0, Gaussian(), (; data = (; y, x)), (:data, :g_tol)) !== nothing
    fit = drm(f, Gaussian(); data = (; y, x))
    bσ = fit.theta[Dict(fit.blocks)[:sigma]]
    @test isapprox(fit.scales[:sigma], exp.(hcat(ones(n), y) * bσ); rtol = 1e-8)
end
