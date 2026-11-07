# Marginal simulator (`simulate(fit; data, ...)` default re = :marginal, and
# `bootstrap_result`) for the independent two-field structured intercept-plus-
# slope SD routes: {Poisson, NegBinomial2, Gaussian} x {animal, relmat}.
#
# Construction under test: b0, b1 ~ N(0, C) independent; eta = Xb + sd0[g] b0[g]
# + x sd1[g] b1[g]. With s_i^2 = C_gg (sd0_g^2 + x_i^2 sd1_g^2) the marginal
# moments are analytic:
#   Gaussian: E y = fitted mean,  Var y = sigma^2 + s_i^2
#   Poisson : m = exp(e + s^2/2), Var y = m + m^2 (exp(s^2) - 1)
#   NB2     : m as above,         Var y = m + m^2 (exp(s^2) (1 + sigma^2) - 1)
# Tolerances (fixed before the first run), nsim = 1500 replicates:
#   mean: per-row |z| < 5 for >= 99% of rows (z = (ybar - E y) / sqrt(Var y / nsim))
#         and |mean(ybar) - mean(E y)| < 0.05 * sqrt(mean(Var y));
#   variance: |mean(sample var) / mean(analytic Var) - 1| < 0.10
#         (counts are heavy tailed; Gaussian is held to 0.05).
using DRModels, Test, Random, LinearAlgebra, Statistics
import Distributions

function _q2sim_fixture(family; G = 30, m = 12, seed = 202611073)
    rng = MersenneTwister(seed)
    R = 0.35 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    s = collect(range(0.7, 1.4; length = G))
    C = Matrix(Diagonal(s) * R * Diagonal(s))
    gidx = repeat(1:G, inner = m)
    z0 = repeat([-1.0, 1.0], inner = G ÷ 2)
    z1 = collect(range(-1.0, 1.0; length = G))[randperm(rng, G)]
    x = repeat(collect(range(-1.2, 1.2; length = m)), outer = G)
    a0, a1 = [-0.9, 0.35], [-1.0, 0.45]
    L = cholesky(Symmetric(C)).L
    u0 = exp.(a0[1] .+ a0[2] .* z0) .* (L * randn(rng, G))
    u1 = exp.(a1[1] .+ a1[2] .* z1) .* (L * randn(rng, G))
    eta = 0.2 .+ 0.3 .* x .+ u0[gidx] .+ x .* u1[gidx]
    y = if family isa Gaussian
        eta .+ 0.4 .* randn(rng, length(eta))
    elseif family isa Poisson
        Float64.([rand(rng, Distributions.Poisson(exp(v))) for v in eta])
    else
        r = 1 / 0.45^2
        Float64.([rand(rng, Distributions.NegativeBinomial(r, r / (r + exp(v))))
                  for v in eta])
    end
    return (; data = (; y, x, z0 = z0[gidx], z1 = z1[gidx],
                      g = ["u$(i)" for i in gidx]), C, gidx)
end

_q2sim_sigma2(fit) = fit.scales[:sigma][1]^2

function _q2sim_formula(kind, family)
    mu = kind === :animal ? @formula(y ~ x + animal(1 + x | g)) :
                            @formula(y ~ x + relmat(1 + x | g))
    sd0 = kind === :animal ? @formula(sd(g, animal, coef(intercept)) ~ z0) :
                             @formula(sd(g, relmat, coef(intercept)) ~ z0)
    sd1 = kind === :animal ? @formula(sd(g, animal, coef(x)) ~ z1) :
                             @formula(sd(g, relmat, coef(x)) ~ z1)
    family isa Poisson ? bf(mu, sd0, sd1) : bf(mu, @formula(sigma ~ 1), sd0, sd1)
end

@testset "q2 SD marginal simulator: $(nameof(typeof(family))) $kind" for
        family in (Poisson(), NegBinomial2(), Gaussian()), kind in (:animal, :relmat)
    fx = _q2sim_fixture(family)
    data = fx.data
    kw = kind === :animal ? (; A = fx.C) : (; K = fx.C)
    fit = drm(_q2sim_formula(kind, family), family; data, kw...)
    @test fit.converged

    # Analytic marginal moments from the fitted parameters.
    targets = DRModels._sd_target_parts(fit.formula)
    sdv = map((:intercept, :x)) do member
        t = only(filter(item -> item.target.member === member, targets))
        blk = DRModels._sd_target_block(t.target; disambiguate = true)
        Z = hcat(ones(length(data.y)), member === :intercept ? data.z0 : data.z1)
        exp.(Z * coef(fit, blk))
    end
    cgg = diag(fx.C)[fx.gidx]
    s2 = cgg .* (sdv[1] .^ 2 .+ data.x .^ 2 .* sdv[2] .^ 2)
    eta = Vector{Float64}(predict(fit, data; type = :link))
    if family isa Gaussian
        sig2 = fit.scales[:sigma] .^ 2
        Ey, Vy = eta, sig2 .+ s2
    else
        mm = exp.(eta .+ s2 ./ 2)
        k = family isa Poisson ? 1.0 : 1 + _q2sim_sigma2(fit)
        Ey, Vy = mm, mm .+ mm .^ 2 .* (exp.(s2) .* k .- 1)
    end

    nsim = 1500
    Y = simulate(fit; nsim, rng = MersenneTwister(77), data, kw...)
    @test size(Y) == (length(data.y), nsim)
    @test all(isfinite, Y)
    ybar = vec(mean(Y; dims = 2))
    z = (ybar .- Ey) ./ sqrt.(Vy ./ nsim)
    @test mean(abs.(z) .< 5) >= 0.99
    @test abs(mean(ybar) - mean(Ey)) < 0.05 * sqrt(mean(Vy))
    vratio = mean(vec(var(Y; dims = 2))) / mean(Vy)
    @test abs(vratio - 1) < (family isa Gaussian ? 0.05 : 0.10)

    # The default marginal draw must add between-group variation beyond the
    # conditional (re = :zero) draw.
    Y0 = simulate(fit; nsim = 200, rng = MersenneTwister(78), data, re = :zero)
    @test mean(var(Y; dims = 2)) > mean(var(Y0; dims = 2))

    boot = bootstrap_result(fit; data, B = 3, rng = MersenneTwister(79),
        failures = :skip, kw...)
    # A tiny-B refit can fail to converge and is then dropped (a refit property,
    # not a simulator one), so failures are skipped; require every replicate to
    # be accounted for, none to have thrown inside the simulator, >= 1 usable,
    # and the result shape (one seed per replicate, a non-empty summary).
    @test boot.attempted == 3
    @test boot.used + boot.failed == 3
    @test boot.used >= 1
    @test length(boot.seeds) == 3
    @test !isempty(boot.summary)
    @test all(!occursin("marginal", string(r.message)) for r in boot.failures)
end
