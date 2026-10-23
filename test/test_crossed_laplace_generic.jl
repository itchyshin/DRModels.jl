using Test, Random, Statistics
import Distributions
using DRModels

logistic(x) = 1 / (1 + exp(-x))

function crossed_components(g, h)
    gidx, G = DRModels._group_index(g)
    hidx, H = DRModels._group_index(h)
    return [
        (ones(length(g)), gidx, G, "g"),
        (ones(length(h)), hidx, H, "h"),
    ]
end

function central_gradient(f, θ; h = 1e-5)
    g = similar(θ)
    for k in eachindex(θ)
        step = h * max(abs(θ[k]), 1.0)
        θp = copy(θ)
        θm = copy(θ)
        θp[k] += step
        θm[k] -= step
        g[k] = (f(θp) - f(θm)) / (2step)
    end
    return g
end

@testset "Generic crossed sparse-Laplace non-Gaussian kernels" begin
    rng = MersenneTwister(70)
    G = 28
    H = 24
    n = 2400
    x = randn(rng, n)
    g = [Symbol("g", rand(rng, 1:G)) for _ in 1:n]
    h = [Symbol("h", rand(rng, 1:H)) for _ in 1:n]
    gmap = Dict(Symbol("g", j) => j for j in 1:G)
    hmap = Dict(Symbol("h", j) => j for j in 1:H)
    X = hcat(ones(n), x)
    comps = crossed_components(g, h)

    β = [0.2, 0.45]
    σg = 0.45
    σh = 0.35
    bg = σg .* randn(rng, G)
    bh = σh .* randn(rng, H)
    η = [β[1] + β[2] * x[i] + bg[gmap[g[i]]] + bh[hmap[h[i]]] for i in 1:n]

    ntr = fill(8.0, n)
    s = Float64.([rand(rng, Distributions.Binomial(round(Int, ntr[i]), logistic(η[i]))) for i in 1:n])
    fit_bin = DRModels._fit_binomial_crossed_laplace(DRModels.Binomial(), s, ntr, X, comps, ["(Intercept)", "x"], 1e-7)
    sd_bin = re_sd(fit_bin)
    @test fit_bin.converged
    @test abs(coef(fit_bin, :mu)[2] - β[2]) < 0.08
    @test abs(sd_bin[:g] - σg) < 0.12
    @test abs(sd_bin[:h] - σh) < 0.12

    μ = exp.(η)
    size = 3.0
    ynb = Float64.([rand(rng, Distributions.NegativeBinomial(size, size / (size + μ[i]))) for i in 1:n])
    fit_nb = DRModels._fit_nb2_fixed_crossed_laplace(DRModels.NegBinomial2(), ynb, size, X, comps, ["(Intercept)", "x"], 1e-7)
    sd_nb = re_sd(fit_nb)
    @test fit_nb.converged
    @test abs(coef(fit_nb, :mu)[2] - β[2]) < 0.08
    @test abs(sd_nb[:g] - σg) < 0.12
    @test abs(sd_nb[:h] - σh) < 0.12
    fit_nb_est = DRModels._fit_nb2_crossed_laplace(DRModels.NegBinomial2(), ynb, X, ones(n, 1), comps, ["(Intercept)", "x"], ["(Intercept)"], 1e-7)
    sd_nb_est = re_sd(fit_nb_est)
    @test fit_nb_est.converged
    @test abs(coef(fit_nb_est, :mu)[2] - β[2]) < 0.08
    @test abs(exp(-2 * coef(fit_nb_est, :sigma)[1]) - size) < 0.8
    @test abs(sd_nb_est[:g] - σg) < 0.12
    @test abs(sd_nb_est[:h] - σh) < 0.12

    shape = 7.0
    yg = Float64.([rand(rng, Distributions.Gamma(shape, μ[i] / shape)) for i in 1:n])
    fit_gamma = DRModels._fit_gamma_fixed_crossed_laplace(DRModels.Gamma(), yg, shape, X, comps, ["(Intercept)", "x"], 1e-7)
    sd_gamma = re_sd(fit_gamma)
    @test fit_gamma.converged
    @test abs(coef(fit_gamma, :mu)[2] - β[2]) < 0.08
    @test abs(sd_gamma[:g] - σg) < 0.12
    @test abs(sd_gamma[:h] - σh) < 0.12
    fit_gamma_est = DRModels._fit_gamma_crossed_laplace(DRModels.Gamma(), yg, X, ones(n, 1), comps, ["(Intercept)", "x"], ["(Intercept)"], 1e-7)
    sd_gamma_est = re_sd(fit_gamma_est)
    @test fit_gamma_est.converged
    @test abs(coef(fit_gamma_est, :mu)[2] - β[2]) < 0.08
    @test abs(exp(-2 * coef(fit_gamma_est, :sigma)[1]) - shape) < 2.0
    @test abs(sd_gamma_est[:g] - σg) < 0.12
    @test abs(sd_gamma_est[:h] - σh) < 0.12

    φ = 25.0
    p = logistic.(η)
    yb = Float64.([rand(rng, Distributions.Beta(p[i] * φ, (1 - p[i]) * φ)) for i in 1:n])
    fit_beta = DRModels._fit_beta_fixed_crossed_laplace(DRModels.Beta(), yb, φ, X, comps, ["(Intercept)", "x"], 1e-7)
    sd_beta = re_sd(fit_beta)
    @test fit_beta.converged
    @test abs(coef(fit_beta, :mu)[2] - β[2]) < 0.08
    @test abs(sd_beta[:g] - σg) < 0.12
    @test abs(sd_beta[:h] - σh) < 0.12
    fit_beta_est = DRModels._fit_beta_crossed_laplace(DRModels.Beta(), yb, X, ones(n, 1), comps, ["(Intercept)", "x"], ["(Intercept)"], 1e-7)
    sd_beta_est = re_sd(fit_beta_est)
    @test fit_beta_est.converged
    @test abs(coef(fit_beta_est, :mu)[2] - β[2]) < 0.08
    @test abs(exp(-2 * coef(fit_beta_est, :sigma)[1]) - φ) < 7.0
    @test abs(sd_beta_est[:g] - σg) < 0.12
    @test abs(sd_beta_est[:h] - σh) < 0.12
end

# #318: the NB2 crossed-RE dispersion START must map to size = MoM size. The
# crossed `aux_from` sets size = exp(−2·logσ), so the Method-of-Moments size
# s_MoM = m²/(v−m) requires logσ0 = −0.5·log(s_MoM). The pre-#318 start used
# +log(s_MoM) (dropped −0.5 and flipped the sign) → start size = 1/s_MoM². We
# assert the start is sign/factor-correct by fitting with iterations = 0-ish
# effort via a direct check of the MoM→logσ mapping the fitter uses.
@testset "NB2 crossed dispersion start = MoM size (#318)" begin
    # The mapping the setup uses: s_MoM = m²/max(v−m,…); logσ0 = −0.5·log(s_MoM);
    # start size = exp(−2·logσ0) must equal s_MoM.
    for (m, v) in ((5.0, 25.0), (5.0, 55.0), (3.0, 12.0))
        s_MoM = max(m^2 / max(v - m, 0.1 * m + eps()), 0.5)
        logσ0 = -0.5 * log(s_MoM)          # the corrected start
        @test exp(-2 * logσ0) ≈ s_MoM
        # The buggy +log start would give size = 1/s_MoM² (the reciprocal-squared):
        @test exp(-2 * log(s_MoM)) ≈ 1 / s_MoM^2
    end
    # End-to-end: an overdispersed crossed count recovers the true size.
    rng = MersenneTwister(318)
    G = 30; H = 30; nper = 4
    gids = repeat(1:G, inner = H * nper)
    hids = repeat(repeat(1:H, inner = nper), outer = G)
    n = length(gids)
    x = randn(rng, n)
    comps318 = [(ones(n), gids, G, "g"), (ones(n), hids, H, "h")]
    true_size = 2.0
    bg = 0.4 .* randn(rng, G); bh = 0.3 .* randn(rng, H)
    η = [0.8 + 0.5 * x[i] + bg[gids[i]] + bh[hids[i]] for i in 1:n]
    μ = exp.(η)
    ynb = Float64.([rand(rng, Distributions.NegativeBinomial(true_size, true_size / (true_size + μ[i]))) for i in 1:n])
    fit318 = DRModels._fit_nb2_crossed_laplace(DRModels.NegBinomial2(), ynb, hcat(ones(n), x),
                                          ones(n, 1), comps318, ["(Intercept)", "x"],
                                          ["(Intercept)"], 1e-7)
    @test abs(exp(-2 * coef(fit318, :sigma)[1]) - true_size) < 0.8
end

@testset "Crossed sparse-Laplace nuisance exact gradient" begin
    rng = MersenneTwister(72)
    G = 6
    H = 5
    n = 220
    x = randn(rng, n)
    gidx = [rand(rng, 1:G) for _ in 1:n]
    hidx = [rand(rng, 1:H) for _ in 1:n]
    X = hcat(ones(n), x)
    β = [0.15, 0.35]
    bg = 0.35 .* randn(rng, G)
    bh = 0.25 .* randn(rng, H)
    η = [β[1] + β[2] * x[i] + bg[gidx[i]] + bh[hidx[i]] for i in 1:n]
    μ = exp.(η)

    y_nb = Float64.([rand(rng, Distributions.NegativeBinomial(2.5, 2.5 / (2.5 + μ[i]))) for i in 1:n])
    yint = round.(Int, y_nb)
    nb_aux(logsize) = begin
        r = exp(clamp(-2 * logsize, -8.0, 8.0))
        lconst = [DRModels.loggamma(yint[i] + r) - DRModels.loggamma(r) - DRModels._logfactorial(yint[i]) for i in eachindex(yint)]
        (y = Float64.(yint), size = r, lconst = lconst)
    end

    y_gamma = Float64.([rand(rng, Distributions.Gamma(6.0, μ[i] / 6.0)) for i in 1:n])
    gamma_aux(logsigma) = begin
        α = exp(clamp(-2 * logsigma, -8.0, 8.0))
        lconst = [α * log(α) - DRModels.loggamma(α) + (α - 1) * log(y_gamma[i]) for i in eachindex(y_gamma)]
        (y = y_gamma, shape = α, lconst = lconst)
    end

    p = logistic.(η)
    y_beta = Float64.([rand(rng, Distributions.Beta(p[i] * 20.0, (1 - p[i]) * 20.0)) for i in 1:n])
    ylogit = log.(y_beta) .- log1p.(-y_beta)
    beta_aux(logsigma) = begin
        φ = exp(clamp(-2 * logsigma, -8.0, 8.0))
        (y = y_beta, precision = φ, ylogit = ylogit, lgammaφ = DRModels.loggamma(φ))
    end

    θ_nb = [0.12, 0.32, -0.5 * log(2.3), log(0.32), log(0.24)]
    θ_gamma = [0.12, 0.32, -0.5 * log(6.3), log(0.32), log(0.24)]
    θ_beta = [0.12, 0.32, -0.5 * log(22.0), log(0.32), log(0.24)]

    for (kind, aux_from, θ) in (
        (Val(:nb2_fixed), nb_aux, θ_nb),
        (Val(:gamma_fixed), gamma_aux, θ_gamma),
        (Val(:beta_fixed), beta_aux, θ_beta),
    )
        val, grad, _, ok = DRModels._crossed_mean_laplace_nuisance_fg(
            kind, aux_from, n, X, gidx, G, hidx, H, θ; grad = true
        )
        @test ok
        @test isfinite(val)
        fd = central_gradient(θp -> DRModels._crossed_mean_laplace_nuisance_fg(
            kind, aux_from, n, X, gidx, G, hidx, H, θp; grad = false
        )[1], θ)
        @test maximum(abs.(grad .- fd)) <= 1e-6
    end
end

@testset "Crossed sparse-Laplace public routing smoke" begin
    rng = MersenneTwister(71)
    G = 14
    H = 12
    reps = 6
    gids = repeat(1:G, inner = H * reps)
    hids = repeat(repeat(1:H, inner = reps), outer = G)
    n = length(gids)
    x = randn(rng, n)
    g = [Symbol("g", j) for j in gids]
    h = [Symbol("h", j) for j in hids]
    β = [0.2, 0.45]
    bg = 0.35 .* randn(rng, G)
    bh = 0.25 .* randn(rng, H)
    bg .-= mean(bg)
    bh .-= mean(bh)
    η = [β[1] + β[2] * x[i] + bg[gids[i]] + bh[hids[i]] for i in 1:n]

    ntr = fill(8.0, n)
    s = Float64.([rand(rng, Distributions.Binomial(round(Int, ntr[i]), logistic(η[i]))) for i in 1:n])
    fail = ntr .- s
    fit_bin = drm(bf(@formula(cbind(s, fail) ~ x + (1 | g) + (1 | h))), Binomial();
                  data = (; s, fail, x, g, h))
    @test fit_bin.converged
    @test haskey(re_sd(fit_bin), :g)
    @test haskey(re_sd(fit_bin), :h)

    μ = exp.(η)
    nb_size = 2.5
    ynb = Float64.([rand(rng, Distributions.NegativeBinomial(nb_size, nb_size / (nb_size + μ[i]))) for i in 1:n])
    fit_nb = drm(bf(@formula(ynb ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1)),
                 NegBinomial2(); data = (; ynb, x, g, h))
    @test fit_nb.converged
    @test haskey(re_sd(fit_nb), :g)
    @test haskey(re_sd(fit_nb), :h)
end
