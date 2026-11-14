# b34: the conditional structured-field export used by the R bridge.
#
# For a structured relmat() / animal() / spatial() mean intercept or slope with
# a predictor-dependent SD, the Gaussian, Poisson, NB2, Gamma and Binomial
# fitters record, at the converged theta, the per-row link-scale contribution of
# the field (`eta_re`) and the inner Laplace mode of the field (`modes`), in
# first-seen group order. This file checks them against an ORACLE that does not
# call any fitter code: a Newton solve of
#     min_b  0.5 b' Q b + sum_i nll_i(eta0_i + load_i b_{g(i)}),   Q = K^-1,
# with hand-derived per-family derivatives, at the fitted theta.
#
# Tolerances fixed before the runs: modes and eta_re within 1e-6 of the oracle
# on the scale (1 + max |oracle|); the stored objective equal to nll(theta-hat) to 1e-4
# (the NB2 fitter once overwrote a shared boxed `value`, a shift of 28). Emulated-bug checks
# (population-level value, reversed group order) must sit far above 1e-6.
# One fixture per family; no calibration or broad parity claim.
using DRModels, Test, Random, LinearAlgebra
import Distributions

const _B34_TOL = 1e-6

# Per-observation (d1, d2) of the negative log-likelihood in eta.
function _b34_derivs(family::Symbol, y, eta, sigma)
    if family === :poisson
        mu = exp(eta)
        return mu - y, mu
    elseif family === :nbinom2
        r = 1 / sigma^2
        mu = exp(eta)
        return r * (mu - y) / (r + mu), r * mu * (r + y) / (r + mu)^2
    elseif family === :gamma
        k = 1 / sigma^2
        ratio = y / exp(eta)
        return k * (1 - ratio), k * ratio
    elseif family === :binomial
        p = 1 / (1 + exp(-eta))
        return p - y, p * (1 - p)
    else  # :gaussian
        return (eta - y) / sigma^2, 1 / sigma^2
    end
end

function _b34_oracle(family, y, Xmu, beta, sdobs, gidx, K, sigma)
    G = size(K, 1)
    Q = inv(Symmetric(K))
    eta0 = Xmu * beta
    b = zeros(G)
    for _ in 1:100
        grad = Q * b
        H = copy(Matrix(Q))
        for i in eachindex(y)
            g = gidx[i]
            d1, d2 = _b34_derivs(family, y[i], eta0[i] + sdobs[i] * b[g], sigma)
            grad[g] += sdobs[i] * d1
            H[g, g] += sdobs[i]^2 * d2
        end
        step = H \ grad
        b .-= step
        norm(step) < 1e-13 * (1 + norm(b)) && break
    end
    return b
end

function _b34_fixture(family::Symbol; G = 14, m = 8, seed = 20261103, slope = false)
    rng = MersenneTwister(seed)
    labels_pool = ["id_$(i)" for i in randperm(rng, G)]
    rows = shuffle(rng, repeat(1:G, inner = m))            # shuffled data rows
    first_seen = unique(rows)                              # matrix order
    R = 0.45 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    scale = collect(range(0.7, 1.6; length = G))
    Kfull = Diagonal(scale) * R * Diagonal(scale)          # raw K, non-unit diagonal
    K = Matrix(Kfull[first_seen, first_seen])
    pos = Dict(g => k for (k, g) in enumerate(first_seen))
    gidx = [pos[g] for g in rows]
    zgroup = repeat([-1.0, 1.0], G ÷ 2)[randperm(rng, G)]  # in first-seen order
    z = zgroup[gidx]
    x = randn(rng, length(rows))
    alpha = [-0.6, 0.5]
    latent = cholesky(Symmetric(K)).L * randn(rng, G)
    w = slope ? x : ones(length(x))
    eta = 0.3 .+ 0.2 .* x .+ w .* (exp.(alpha[1] .+ alpha[2] .* zgroup) .* latent)[gidx]
    y = if family === :gaussian
        eta .+ 0.5 .* randn(rng, length(eta))
    elseif family === :poisson
        Float64[rand(rng, Distributions.Poisson(exp(v))) for v in eta]
    elseif family === :nbinom2
        Float64[rand(rng, Distributions.NegativeBinomial(1 / 0.6^2,
            (1 / 0.6^2) / ((1 / 0.6^2) + exp(v)))) for v in eta]
    elseif family === :gamma
        Float64[rand(rng, Distributions.Gamma(1 / 0.35^2, exp(v) * 0.35^2)) for v in eta]
    else
        Float64[rand(rng, Distributions.Bernoulli(1 / (1 + exp(-(v - 0.2))))) for v in eta]
    end
    g = labels_pool[rows]
    return (; data = (; y, x, z, g), K, gidx, zgroup, w)
end

_b34_family(f::Symbol) = f === :gaussian ? Gaussian() : f === :poisson ? Poisson() :
    f === :nbinom2 ? NegBinomial2() : f === :gamma ? Gamma() : Binomial()

@testset "structured conditional export vs dense Newton oracle" begin
    for family in (:gaussian, :poisson, :nbinom2, :gamma, :binomial),
        slope in (false, true)
        @testset "$(family) $(slope ? "relmat slope" : "relmat intercept")" begin
            fx = _b34_fixture(family; slope, seed = 20261103 + 7 * slope)
            data, K = fx.data, fx.K
            term = slope ? @formula(y ~ x + relmat(0 + x | g)) :
                           @formula(y ~ x + relmat(1 | g))
            extra = family === :gaussian ? (;) : (; se = false)
            fit = drm(bf(term, @formula(sd(g, relmat) ~ z)), _b34_family(family);
                      data, K, extra...)
            @test fit.converged

            cond = DRModels.structured_conditional(fit)
            @test cond !== nothing
            @test length(cond.modes) == 1
            mode, eta_re = only(cond.modes), cond.eta_re
            @test length(mode) == size(K, 1)
            @test length(eta_re) == length(data.y)

            # The stored objective is the objective at theta-hat, not a stencil point
            # or a value overwritten by the mode solve. Re-evaluating nll warm-starts
            # the inner mode differently, which moves it by up to ~1e-5; the NB2
            # shared-box bug this guards moved it by 28. Tolerance fixed at 1e-4.
            @test loglik(fit) ≈ -fit.nll(fit.theta) atol = 1e-4

            pmu = 2
            theta = fit.theta
            beta = theta[1:pmu]
            alpha = theta[(end - 1):end]
            Xmu = hcat(ones(length(data.y)), data.x)
            sdobs = exp.(alpha[1] .+ alpha[2] .* data.z) .* fx.w
            sigma = family in (:gaussian, :nbinom2, :gamma) ? fit.scales[:sigma][1] : 1.0
            # The oracle uses 1 / sigma^2 as the NB2 / Gamma size; the fitter clamps
            # the NB2 size at exp(8) (log sigma < -4), so require an interior fit.
            family in (:nbinom2, :gamma) && @test log(sigma) > -3
            oracle = _b34_oracle(family, data.y, Xmu, beta, sdobs, fx.gidx, K, sigma)
            scale = 1 + maximum(abs, oracle)

            @test maximum(abs, mode .- oracle) <= _B34_TOL * scale
            @test maximum(abs, eta_re .- sdobs .* oracle[fx.gidx]) <= _B34_TOL * scale

            # emulated bugs sit far above the tolerance (>= 1000 x)
            @test maximum(abs, eta_re) > 1000 * _B34_TOL                 # population-level value
            @test maximum(abs, mode .- reverse(oracle)) > 1000 * _B34_TOL # wrong group order
            @test maximum(abs, eta_re .- sdobs .* oracle[reverse(fx.gidx)]) > 1000 * _B34_TOL
        end
    end
end

@testset "bridge payload carries the conditional field" begin
    fx = _b34_fixture(:poisson)
    data, K = fx.data, fx.K
    out = drm_bridge(; formula = "y ~ x + relmat(1 | g); sd(g, relmat) ~ z",
                     family = "poisson", data, K = K)
    direct = drm(bf(@formula(y ~ x + relmat(1 | g)), @formula(sd(g, relmat) ~ z)),
                 Poisson(); data, K, se = false)
    payload = out["conditional_structured"]
    @test payload["kind"] == "structured_eta_re_v1"
    cond = DRModels.structured_conditional(direct)
    @test payload["eta_re"] ≈ cond.eta_re atol = 1e-6
    @test only(payload["modes"]) ≈ only(cond.modes) atol = 1e-6
    # `cond_*` scales are not response-distribution parameters
    @test !any(startswith(k, "cond_") for k in keys(out["dpars"]))
    # population-level `fitted` slot is unchanged
    @test out["fitted"] ≈ fitted(direct; re = :zero) atol = 1e-8

    # a fit without a structured field exports nothing
    plain = drm_bridge(; formula = "y ~ x", family = "poisson", data)
    @test !haskey(plain, "conditional_structured")
end

@testset "a non-finite conditional field is marked as a failed export" begin
    scales = Dict{Symbol,Any}()
    DRModels._attach_conditional_structured!(scales, [0.1, NaN], [0.2])
    @test get(scales, :cond_failed, [0.0]) == [1.0]
    @test !haskey(scales, :cond_eta_re)
    ok = Dict{Symbol,Any}()
    DRModels._attach_conditional_structured!(ok, [0.1, 0.3], [0.2])
    @test !haskey(ok, :cond_failed)
    @test ok[:cond_eta_re] == [0.1, 0.3]
    # DrmFit.scales is Dict{Symbol,Vector{Float64}}: a scalar flag used to throw a
    # MethodError and discard a converged fit. The typed dict must accept the flag.
    typed = Dict{Symbol,Vector{Float64}}(:sigma => [1.0, 1.0])
    DRModels._attach_conditional_structured!(typed, [0.1, Inf], [0.2])
    @test typed[:cond_failed] == [1.0]
    typed2 = Dict{Symbol,Vector{Float64}}()
    @test DRModels._mark_cond_failed!(typed2) === typed2
    @test typed2[:cond_failed] == [1.0]
end

# DrmFit.scales is a Dict{Symbol,Vector{Float64}}: the failed-export marker must
# be storable there. (Previously `scales[:cond_failed] = 1.0` threw a MethodError
# the first time a conditional-mode inner solve failed on a real fit.)
@testset "failed-export marker is storable in a typed DrmFit scales dict" begin
    scales = Dict{Symbol,Vector{Float64}}(:sigma => [1.0])
    DRModels._attach_conditional_structured!(scales, [0.1, NaN], [0.2])
    @test haskey(scales, :cond_failed)
    @test scales[:cond_failed] == [1.0]
    @test !haskey(scales, :cond_eta_re)
    marked = Dict{Symbol,Vector{Float64}}()
    DRModels._mark_cond_failed!(marked)
    @test haskey(marked, :cond_failed)
end
