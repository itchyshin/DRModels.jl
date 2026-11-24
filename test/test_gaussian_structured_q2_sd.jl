# Gaussian independent two-field structured intercept-plus-slope with one SD
# formula per coefficient:
#   y ~ x + animal|relmat(1 + x | g),  sd(g, <level>, coef(intercept)) ~ z0,
#   sd(g, <level>, coef(x)) ~ z1.
# Exact marginal likelihood y ~ N(X b, sigma^2 I + L0 C L0' + L1 C L1'), C RAW.
# Oracle is an independent dense MvNormal log density (no fitter code shared).
# Tolerances (fixed before the first run): nll vs oracle atol 1e-6; SD
# coefficient recovery |est - truth| < 1.0 (G = 30 levels, loose, no calibration
# claim); fit sits at an optimum of the oracle surface (+/- 0.05 steps).
using DRModels, Test, Random, LinearAlgebra
import Distributions

function _gq2_fixture(; G = 30, m = 12, seed = 20261107)
    rng = MersenneTwister(seed)
    R = 0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    s = collect(range(0.7, 1.6; length = G))
    C = Matrix(Diagonal(s) * R * Diagonal(s))
    labels = ["id_$(i)" for i in randperm(rng, G)]
    z0 = repeat([-1.0, 1.0], inner = G ÷ 2)
    z1 = collect(range(-1.0, 1.0; length = G))[randperm(rng, G)]
    a0, a1 = [-0.7, 0.4], [-0.9, 0.5]
    L = cholesky(Symmetric(C)).L
    u0 = exp.(a0[1] .+ a0[2] .* z0) .* (L * randn(rng, G))
    u1 = exp.(a1[1] .+ a1[2] .* z1) .* (L * randn(rng, G))
    gi = repeat(1:G, inner = m)
    x = randn(rng, G * m)
    y = 0.3 .+ 0.2 .* x .+ u0[gi] .+ x .* u1[gi] .+ 0.4 .* randn(rng, G * m)
    ord = randperm(rng, length(y))
    group = labels[gi][ord]
    fitted = unique(group)
    orig = [findfirst(==(l), labels) for l in fitted]
    gidx = [findfirst(==(l), fitted) for l in group]
    data = (; y = y[ord], x = x[ord], z0 = z0[orig][gidx], z1 = z1[orig][gidx], g = group)
    return (; data, C = C[orig, orig], gidx, z0 = z0[orig], z1 = z1[orig], a0, a1)
end

function _gq2_oracle(theta, fx)
    y, x, gi = fx.data.y, fx.data.x, fx.gidx
    n, G = length(y), length(fx.z0)
    beta, sigma, a0, a1 = theta[1:2], exp(theta[3]), theta[4:5], theta[6:7]
    d0 = exp.(hcat(ones(G), fx.z0) * a0); d1 = exp.(hcat(ones(G), fx.z1) * a1)
    L0 = zeros(n, G); L1 = zeros(n, G)
    for i in 1:n
        L0[i, gi[i]] = d0[gi[i]]; L1[i, gi[i]] = x[i] * d1[gi[i]]
    end
    V = sigma^2 * Matrix(I, n, n) + L0 * fx.C * L0' + L1 * fx.C * L1'
    return -Distributions.logpdf(
        Distributions.MvNormal(hcat(ones(n), x) * beta, Symmetric(V)), y)
end

_gq2_formula(kind) = bf(
    kind === :animal ? @formula(y ~ x + animal(1 + x | g)) : @formula(y ~ x + relmat(1 + x | g)),
    @formula(sigma ~ 1),
    kind === :animal ? @formula(sd(g, animal, coef(intercept)) ~ z0) :
                       @formula(sd(g, relmat, coef(intercept)) ~ z0),
    kind === :animal ? @formula(sd(g, animal, coef(x)) ~ z1) :
                       @formula(sd(g, relmat, coef(x)) ~ z1))
_gq2_kw(kind, fx) = kind === :animal ? (; A = fx.C) : (; K = fx.C)

@testset "Gaussian structured intercept-and-slope SD regression (q2)" begin
    for kind in (:animal, :relmat)
        @testset "$kind" begin
            fx = _gq2_fixture()
            data, formula, kw = fx.data, _gq2_formula(kind), _gq2_kw(kind, fx)
            fit = drm(formula, Gaussian(); data, kw...)
            @test fit.converged
            @test length(coef(fit, :mu)) == 2
            sdnames = [first(p) for p in fit.blocks if first(p) ∉ (:mu, :sigma)]
            @test length(sdnames) == 2
            a0 = coef(fit, sdnames[1]); a1 = coef(fit, sdnames[2])
            @test length(a0) == 2 && length(a1) == 2

            theta_hat = vcat(coef(fit, :mu), coef(fit, :sigma), a0, a1)
            theta_true = [0.3, 0.2, log(0.4), fx.a0..., fx.a1...]
            for th in (theta_true, theta_hat, theta_hat .+ 0.07)
                @test fit.nll(th) ≈ _gq2_oracle(th, fx) atol = 1e-6
            end
            @test fit.loglik ≈ -_gq2_oracle(theta_hat, fx) atol = 1e-6
            base = _gq2_oracle(theta_hat, fx)
            for j in eachindex(theta_hat), h in (-0.05, 0.05)
                th = copy(theta_hat); th[j] += h
                @test _gq2_oracle(th, fx) > base - 1e-6
            end
            # Raw (non-normalised) kernel is on the likelihood surface.
            d = sqrt.(diag(fx.C)); Cn = fx.C ./ (d * d')
            fx2 = merge(fx, (; C = Cn))
            @test abs(base - _gq2_oracle(theta_hat, fx2)) > 0.05

            # Loose recovery of the SD coefficients (stated tolerance 1.0).
            @test abs(a0[1] - fx.a0[1]) < 1.0
            @test abs(a1[1] - fx.a1[1]) < 1.0
            @test abs(coef(fit, :mu)[1] - 0.3) < 0.5
            @test abs(coef(fit, :mu)[2] - 0.2) < 0.5

            # Constant SDs nest inside the z model.
            cf = bf(kind === :animal ?
                    @formula(y ~ x + animal(1 + x | g)) : @formula(y ~ x + relmat(1 + x | g)),
                    @formula(sigma ~ 1),
                    kind === :animal ? @formula(sd(g, animal, coef(intercept)) ~ 1) :
                                       @formula(sd(g, relmat, coef(intercept)) ~ 1),
                    kind === :animal ? @formula(sd(g, animal, coef(x)) ~ 1) :
                                       @formula(sd(g, relmat, coef(x)) ~ 1))
            cfit = drm(cf, Gaussian(); data, kw...)
            @test cfit.converged
            @test fit.loglik >= cfit.loglik - 1e-6

            # Refusals.
            zbad = copy(data.z0); zbad[1] += 1.0
            @test_throws ArgumentError drm(formula, Gaussian();
                data = merge(data, (; z0 = zbad)), kw...)
            @test_throws ArgumentError drm(formula, Gaussian();
                data = merge(data, (; x = zeros(length(data.x)))), kw...)
            @test_throws Exception drm(formula, Gaussian(); data)          # no kernel
            # REML is admitted for this unlabelled, `sigma ~ 1` cell (oracle-checked in
            # test_gaussian_structured_q2_reml.jl); it no longer refuses.
            @test drm(formula, Gaussian(); data, kw..., method = :REML).estim_method === :REML
        end
    end
    fx = _gq2_fixture()
    # Only one SD formula: refused.
    one = bf(@formula(y ~ x + animal(1 + x | g)), @formula(sigma ~ 1),
             @formula(sd(g, animal, coef(intercept)) ~ z0))
    @test_throws ArgumentError drm(one, Gaussian(); data = fx.data, A = fx.C)
    # No SD formulas (wave 9): the bare term is the native independent two-SD model, completed
    # with `sd(g, animal, coef(·)) ~ 1` for both coefficients -- same fit as the explicit form.
    # REML is completed too (wave 11): the same fit as the explicit form under method = :REML.
    bare = drm(bf(@formula(y ~ x + animal(1 + x | g))), Gaussian(); data = fx.data, A = fx.C)
    expl = drm(bf(@formula(y ~ x + animal(1 + x | g)), @formula(sigma ~ 1),
                  @formula(sd(g, animal, coef(intercept)) ~ 1), @formula(sd(g, animal, coef(x)) ~ 1)),
               Gaussian(); data = fx.data, A = fx.C)
    @test loglik(bare) == loglik(expl)
    bare_reml = drm(bf(@formula(y ~ x + animal(1 + x | g))), Gaussian(); data = fx.data, A = fx.C, method = :REML)
    expl_reml = drm(bf(@formula(y ~ x + animal(1 + x | g)), @formula(sigma ~ 1),
                       @formula(sd(g, animal, coef(intercept)) ~ 1), @formula(sd(g, animal, coef(x)) ~ 1)),
                    Gaussian(); data = fx.data, A = fx.C, method = :REML)
    @test bare_reml.estim_method === :REML
    @test loglik(bare_reml) == loglik(expl_reml)
    # Wrong provider input.
    @test_throws ArgumentError drm(_gq2_formula(:animal), Gaussian(); data = fx.data,
                                   K = fx.C)
    @test_throws ArgumentError drm(_gq2_formula(:relmat), Gaussian(); data = fx.data,
                                   A = fx.C)
end

# Prediction on new data is population-level (X beta, random effects at zero),
# for both providers; agrees with fitted() and with drm_bridge(newdata = ...).
@testset "Gaussian structured q2: predict / predict_parameters / bridge newdata" begin
    for kind in (:animal, :relmat)
        fx = _gq2_fixture()
        data, kw = fx.data, _gq2_kw(kind, fx)
        fit = drm(_gq2_formula(kind), Gaussian(); data, kw...)
        n = length(data.x)
        beta = coef(fit, :mu)
        @test predict(fit, data) ≈ fitted(fit; re = :zero) atol = 1e-10
        @test predict(fit, data; type = :link) ≈ hcat(ones(n), data.x) * beta atol = 1e-10
        nd = (; x = [-0.7, 0.0, 0.5, 1.1], g = fill("never_seen", 4),
              z0 = zeros(4), z1 = zeros(4))
        Xnd = hcat(ones(4), nd.x)
        @test predict(fit, nd) ≈ Xnd * beta atol = 1e-10
        @test predict(fit, merge(nd, (; g = fill(first(data.g), 4)))) ≈ Xnd * beta atol = 1e-10
        sel = predict(fit, nd; type = :link, se = true)
        @test sel.se ≈ sqrt.(diag(Xnd * vcov(fit)[1:2, 1:2] * Xnd')) atol = 1e-8
        @test predict_parameters(fit, nd)[:mu] ≈ Xnd * beta atol = 1e-10
        @test predict_parameters(fit, nd)[:sigma] ≈ fill(exp(coef(fit, :sigma)[1]), 4) atol = 1e-10
        sd = kind === :animal ? "animal" : "relmat"
        br = drm_bridge(; formula = "y ~ x + $sd(1 + x | g); sigma ~ 1; " *
                        "sd(g, $sd, coef(intercept)) ~ z0; sd(g, $sd, coef(x)) ~ z1",
                        family = "gaussian", data, kw..., newdata = nd)
        @test br["converged"]
        @test br["coefficients"] ≈ coef(fit) atol = 1e-6
        @test length(unique(br["coef_names"])) == length(br["coef_names"])
        @test br["dpars_newdata"]["mu"] ≈ Xnd * beta atol = 1e-6
    end
end
