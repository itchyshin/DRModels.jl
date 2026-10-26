# test_offset_fixed_coef_938.jl: #938, owner decision of 25 Oct 2026.
#
# `offset(z)` in the MEAN formula enters the linear predictor with its coefficient
# fixed at 1 for NegBinomial2 and the other non-Poisson families, as drmTMB does.
# It is never a design column and never a parameter. An offset in any other
# formula (`sigma`, `nu`, `zi`, ...) stays refused, as drmTMB refuses it. A route
# that cannot hold the offset fixed refuses it by name instead of estimating it.
#
# Checks, per family: no `offset(...)` coefficient; a constant offset c moves only
# the intercept (by -c) and leaves the log-likelihood unchanged; a varying offset
# is recovered from simulated data; the fitted means carry the offset. NB2 also
# gets an independent likelihood check, the `(1 | g)` routes and `predict(newdata)`.

using DRModels
using Test
using StableRNGs
using Distributions: NegativeBinomial, logpdf
import Distributions

const _OFF_C = log(2.0)

_logistic938(x) = 1 / (1 + exp(-x))
_allnames938(fit) = reduce(vcat, [collect(String, v) for (_, v) in fit.coefnames])

# One simulated data set per family: covariate x, a varying offset z, a constant
# offset column c, and a response generated with η = 0.3 + 0.5 x + z.
function _offset_data_938(fam::Symbol; n = 1500, seed = 938)
    rng = StableRNG(seed)
    x = randn(rng, n)
    z = 0.6 .* (rand(rng, n) .- 0.5)
    η = 0.3 .+ 0.5 .* x .+ z
    c = fill(_OFF_C, n)
    if fam === :nb2 || fam === :tnb2
        μ = exp.(η .+ (fam === :tnb2 ? 0.8 : 0.0)); r = 4.0
        y = Float64[]
        for m in μ
            v = rand(rng, NegativeBinomial(r, r / (r + m)))
            while fam === :tnb2 && v == 0
                v = rand(rng, NegativeBinomial(r, r / (r + m)))
            end
            push!(y, v)
        end
    elseif fam === :tpois
        y = Float64[]
        for m in exp.(η .+ 0.8)
            v = rand(rng, Distributions.Poisson(m))
            while v == 0
                v = rand(rng, Distributions.Poisson(m))
            end
            push!(y, v)
        end
    elseif fam === :gamma
        α = 6.0
        y = [rand(rng, Distributions.Gamma(α, exp(e) / α)) for e in η]
    elseif fam === :lognormal
        y = exp.(η .+ 0.4 .* randn(rng, n))
    elseif fam === :gaussian
        y = η .+ 0.5 .* randn(rng, n)
    elseif fam === :student
        y = η .+ 0.5 .* rand(rng, Distributions.TDist(5), n)
    elseif fam === :skewnormal
        # mu is the MEAN: shift the location by ω δ sqrt(2/π)
        δ = 2.0 / sqrt(1 + 2.0^2)
        y = [rand(rng, Distributions.SkewNormal(e - 0.6 * δ * sqrt(2 / π), 0.6, 2.0)) for e in η]
    elseif fam === :beta
        φ = 20.0
        y = [rand(rng, Distributions.Beta(φ * _logistic938(e), φ * (1 - _logistic938(e)))) for e in η]
    elseif fam === :zoib
        φ = 20.0
        y = [rand(rng, Distributions.Beta(φ * _logistic938(e), φ * (1 - _logistic938(e)))) for e in η]
        k = rand(rng, n) .< 0.1
        y[k] .= Float64.(rand(rng, sum(k)) .< 0.5)
    elseif fam === :binomial || fam === :betabinomial
        ntr = 12
        p = _logistic938.(η)
        s = fam === :binomial ?
            [Float64(rand(rng, Distributions.Binomial(ntr, pi))) for pi in p] :
            [Float64(rand(rng, Distributions.BetaBinomial(ntr, 15pi, 15(1 - pi)))) for pi in p]
        return (; s = s, f = ntr .- s, x = x, z = z, c = c)
    elseif fam === :tweedie
        μ = exp.(η .+ 0.5); ϕ = 1.0; p = 1.5
        λ = μ .^ (2 - p) ./ (ϕ * (2 - p)); γ = ϕ * (p - 1) .* μ .^ (p - 1)
        y = map(1:n) do i
            N = rand(rng, Distributions.Poisson(λ[i]))
            N == 0 ? 0.0 : rand(rng, Distributions.Gamma(N * (2 - p) / (p - 1), γ[i]))
        end
    elseif fam === :cumulative
        cuts = [-0.8, 0.4, 1.5]
        y = map(η) do e
            u = rand(rng)
            k = 1
            while k <= 3 && u > _logistic938(cuts[k] - e)
                k += 1
            end
            Float64(k)
        end
    else
        error("unknown family $fam")
    end
    return (; y = y, x = x, z = z, c = c)
end

# family object, response spelling, extra formula parts (a sigma ~ 1 etc.)
const _OFFSET_FAMILIES_938 = [
    (:nb2,          NegBinomial2()),
    (:tnb2,         TruncatedNegBinomial2()),
    (:tpois,        TruncatedPoisson()),
    (:gamma,        DRModels.Gamma()),
    (:lognormal,    DRModels.LogNormal()),
    (:gaussian,     Gaussian()),
    (:student,      Student()),
    (:skewnormal,   SkewNormal()),
    (:beta,         DRModels.Beta()),
    (:zoib,         ZeroOneBeta()),
    (:binomial,     DRModels.Binomial()),
    (:betabinomial, DRModels.BetaBinomial()),
    (:tweedie,      Tweedie()),
    (:cumulative,   CumulativeLogit()),
]

_lhs938(tag) = tag in (:binomial, :betabinomial) ? "cbind(s, f)" : "y"
_f938(tag, rhs) = bf(eval(Meta.parse("@formula($(_lhs938(tag)) ~ $rhs)")))

@testset "#938 offset() in the mean, coefficient fixed at 1, every non-Poisson family" begin
    for (tag, fam) in _OFFSET_FAMILIES_938
        @testset "$tag" begin
            d = _offset_data_938(tag)
            fz = drm(_f938(tag, "1 + x + offset(z)"), fam; data = d)
            # never a coefficient
            @test !any(n -> occursin("offset", n), _allnames938(fz))
            βz = coef(fz, :mu)
            if tag === :cumulative
                @test length(βz) == 1                      # slope only; cutpoints carry the location
                @test βz[1] ≈ 0.5 atol = 0.15
            else
                @test length(βz) == 2
                @test βz ≈ [0.3 + (tag in (:tnb2, :tpois) ? 0.8 : tag === :tweedie ? 0.5 : 0.0), 0.5] atol = 0.15
            end
            # a constant offset c only moves the location by -c
            f0 = drm(_f938(tag, "1 + x"), fam; data = d)
            fc = drm(_f938(tag, "1 + x + offset(c)"), fam; data = d)
            @test loglik(fc) ≈ loglik(f0) atol = 1e-5
            if tag === :cumulative
                @test coef(fc, :mu) ≈ coef(f0, :mu) atol = 1e-4
                # the cutpoints absorb the location: every one moves by +c
                @test fc.scales[:ordinal_cuts] ≈ f0.scales[:ordinal_cuts] .+ _OFF_C atol = 1e-4
                @test fitted(fc) ≈ fitted(f0) rtol = 1e-4
            else
                @test coef(fc, :mu) ≈ coef(f0, :mu) .- [_OFF_C, 0.0] atol = 1e-4
                # the fitted means include the offset, so they equal the offset-free fit's
                @test fitted(fc) ≈ fitted(f0) rtol = 1e-4
            end
            # the varying offset is not a no-op
            @test abs(loglik(fz) - loglik(f0)) > 1e-3
        end
    end
end

@testset "#938 Gaussian: offset(z) is the fit of y - z, with z added back" begin
    d = _offset_data_938(:gaussian)
    fz = drm(bf(@formula(y ~ 1 + x + offset(z)), @formula(sigma ~ 1 + x)), Gaussian(); data = d)
    fs = drm(bf(@formula(ys ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian();
             data = merge(d, (; ys = d.y .- d.z)))
    @test loglik(fz) ≈ loglik(fs) rtol = 1e-10
    @test coef(fz) ≈ coef(fs) rtol = 1e-8
    @test fitted(fz) ≈ fitted(fs) .+ d.z rtol = 1e-10
    @test residuals(fz) ≈ d.y .- fitted(fz) atol = 1e-10
    nd = (; x = [0.0, 1.0], z = [1.0, -1.0])
    @test predict(fz, nd) ≈ coef(fz, :mu)[1] .+ coef(fz, :mu)[2] .* nd.x .+ nd.z rtol = 1e-10
    # REML is the same shift
    fr = drm(bf(@formula(y ~ 1 + x + offset(z))), Gaussian(); data = d, method = :REML)
    frs = drm(bf(@formula(ys ~ 1 + x)), Gaussian(); data = merge(d, (; ys = d.y .- d.z)), method = :REML)
    @test loglik(fr) ≈ loglik(frs) rtol = 1e-10
    # routes that do not take it refuse it by name
    dg = merge(d, (; g = repeat(string.("g", 1:30), inner = 50)))
    @test_throws ArgumentError drm(bf(@formula(y ~ 1 + x + offset(z) + (1 | g))), Gaussian(); data = dg)
    @test_throws ArgumentError drm(bf(@formula(y ~ 1 + x * offset(z))), Gaussian(); data = d)
end

@testset "#938 NegBinomial2: independent likelihood, predict(newdata), (1 | g) routes" begin
    d = _offset_data_938(:nb2)
    fit = drm(bf(@formula(y ~ 1 + x + offset(z)), @formula(sigma ~ 1)), NegBinomial2(); data = d)
    β = coef(fit, :mu); r = exp(-2 * coef(fit, :sigma)[1])
    μ = exp.(β[1] .+ β[2] .* d.x .+ d.z)
    @test loglik(fit) ≈ sum(logpdf.(NegativeBinomial.(r, r ./ (r .+ μ)), round.(Int, d.y))) rtol = 1e-8
    @test fitted(fit) ≈ μ rtol = 1e-8
    nd = (; x = [-1.0, 0.0, 2.0], z = [0.0, log(3.0), -0.2])
    @test predict(fit, nd) ≈ exp.(β[1] .+ β[2] .* nd.x .+ nd.z) rtol = 1e-10
    @test predict(fit, nd; type = :link) ≈ β[1] .+ β[2] .* nd.x .+ nd.z rtol = 1e-10
    @test_throws ArgumentError predict(fit, (; x = [0.0]))      # the offset column is required
    # simulate draws around the offset-carrying means, with and without newdata
    S = simulate(fit; nsim = 400, rng = StableRNG(5))
    @test sum(S) / length(S) ≈ sum(μ) / length(μ) rtol = 0.03
    Sn = simulate(fit; nsim = 4000, rng = StableRNG(6), newdata = nd)
    @test vec(sum(Sn, dims = 2)) ./ 4000 ≈ exp.(β[1] .+ β[2] .* nd.x .+ nd.z) rtol = 0.08

    # one ordinary (1 | g): default :LA (GHQ) and TMB-convention :Laplace
    rng = StableRNG(9381)
    G = 40; n = 400
    g = repeat(1:G, inner = n ÷ G); u = 0.4 .* randn(rng, G)
    x = randn(rng, n); z = 0.6 .* (rand(rng, n) .- 0.5)
    μg = exp.(0.3 .+ 0.5 .* x .+ z .+ u[g])
    y = [Float64(rand(rng, NegativeBinomial(4.0, 4.0 / (4.0 + m)))) for m in μg]
    dg = (; y = y, x = x, z = z, c = fill(_OFF_C, n), g = string.("g", g))
    for marg in (:LA, :Laplace)
        f0 = drm(bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = dg, marginal = marg)
        fc = drm(bf(@formula(y ~ 1 + x + offset(c) + (1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = dg, marginal = marg)
        fz = drm(bf(@formula(y ~ 1 + x + offset(z) + (1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = dg, marginal = marg)
        @test length(coef(fz, :mu)) == 2
        @test loglik(fc) ≈ loglik(f0) atol = 1e-5
        @test coef(fc, :mu) ≈ coef(f0, :mu) .- [_OFF_C, 0.0] atol = 1e-4
        @test coef(fz, :mu) ≈ [0.3, 0.5] atol = 0.2
        @test abs(loglik(fz) - loglik(f0)) > 1e-3
    end

    # zero-inflated and hurdle NB2 apply the offset to the count mean
    for part in (:zi, :hu)
        pf = part === :zi ? @formula(zi ~ 1) : @formula(hu ~ 1)
        f0 = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1), pf), NegBinomial2(); data = d)
        fc = drm(bf(@formula(y ~ 1 + x + offset(c)), @formula(sigma ~ 1), pf), NegBinomial2(); data = d)
        @test loglik(fc) ≈ loglik(f0) atol = 1e-5
        @test coef(fc, :mu) ≈ coef(f0, :mu) .- [_OFF_C, 0.0] atol = 1e-4
    end
end

@testset "#938 refusals: sigma offsets and routes that cannot hold it fixed" begin
    d = _offset_data_938(:nb2)
    # an offset in the sigma formula stays refused, as in drmTMB
    @test_throws ArgumentError drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + offset(z))),
                                   NegBinomial2(); data = d)
    dg = _offset_data_938(:gamma)
    @test_throws ArgumentError drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + offset(z))),
                                   DRModels.Gamma(); data = dg)
    dn = _offset_data_938(:gaussian)
    @test_throws ArgumentError drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + offset(z))),
                                   Gaussian(); data = dn)
    # routes without offset support refuse rather than estimate it
    rng = StableRNG(9382)
    n = 300; g = repeat(1:30, inner = 10); h = repeat(1:10, 30)
    dd = merge(d, (; g = string.("g", g[1:n]), h = string.("h", h[1:n])))
    dd = (; y = d.y[1:n], x = d.x[1:n], z = d.z[1:n], g = dd.g, h = dd.h)
    @test_throws ArgumentError drm(bf(@formula(y ~ 1 + x + offset(z) + (1 | g)), @formula(sigma ~ 1)),
                                   NegBinomial2(); data = dd, marginal = :VA)
    @test_throws ArgumentError drm(bf(@formula(y ~ 1 + x + offset(z) + (1 | g) + (1 | h)), @formula(sigma ~ 1)),
                                   NegBinomial2(); data = dd)
    @test_throws ArgumentError drm(bf(@formula(y ~ 1 + x + offset(z) + (1 | g)), @formula(sigma ~ 1)),
                                   DRModels.Gamma(); data = merge(dg, (; g = repeat(string.("g", 1:30), inner = 50))))
end

@testset "#938 bridge: R `offset()` reaches the fixed-coefficient marker" begin
    d = _offset_data_938(:nb2)
    dat = (; y = d.y, x = d.x, z = d.z, expo = exp.(d.z))
    nat = drm(bf(@formula(y ~ 1 + x + offset(z)), @formula(sigma ~ 1)), NegBinomial2(); data = dat)
    out = DRModels.drm_bridge(; formula = "y ~ x + offset(z); sigma ~ 1", family = "nbinom2", data = dat)
    @test out["loglik"] ≈ loglik(nat) rtol = 1e-8
    @test !any(n -> occursin("offset", n), out["coef_names"])
    # a transform inside offset(), as R users write it
    out2 = DRModels.drm_bridge(; formula = "y ~ x + offset(log(expo)); sigma ~ 1", family = "nbinom2", data = dat)
    @test out2["loglik"] ≈ loglik(nat) rtol = 1e-6
    # an offset in sigma is still refused, through the bridge too
    @test_throws ArgumentError DRModels.drm_bridge(; formula = "y ~ x; sigma ~ 1 + offset(z)",
                                                   family = "nbinom2", data = dat)
end
