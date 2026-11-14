# fitted() keeps the conditional modes on the zero-inflated Poisson, zero-inflated NB2 and
# zero-truncated NB2 random-intercept routes and on the CumulativeLogit independent
# intercept + slope route (all fitted by the dense Laplace spine of `indep_slopes_laplace.jl`).
#
# These fits kept no modes (main: `fitted(fit)` is the population value; the conditional default refuses): logLik agreed with
# drmTMB 0.8.1 and `fitted()` differed by 0.7 to 2.1. Now `fitted(fit)` is drmTMB's `fitted()`:
# the expected response with the modes in the count mean (ZIP / ZINB: (1 - zi) * mu, truncated NB2:
# mu / (1 - P(0)), ordinal: the expected category score).
#
# References: test/fixtures/fitted_modes_zi_ordinal/fitted_modes_reference.csv, written by
# fitted_modes_r.R in the same folder (drmTMB 0.8.1, R 4.6.1) from the data of
# test/fixtures/p17_fitted/data.csv. Bounds are those of test_fitted_conditional.jl and are not
# loosened: 1e-6 on fitted values (2e-5 for the ordinal route), 1e-4 on logLik.
# The modes are also checked against a joint density maximised here, independently of the package.
using DRModels
using Test
using DelimitedFiles
using ForwardDiff
using LinearAlgebra
using SpecialFunctions: loggamma

const _FM_DIR = joinpath(@__DIR__, "fixtures", "fitted_modes_zi_ordinal")
const _FM_FACS = Set(["g", "g2", "h", "f", "f2", "gna", "gu", "y_cat", "y_ordf"])

function _fm_data()
    m, h = readdlm(joinpath(@__DIR__, "fixtures", "p17_fitted", "data.csv"), ',', Any; header = true)
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(String.(vec(h)))
        c = m[:, j]
        v = if n in _FM_FACS
            any(x -> x == "NA", c) ? Union{Missing,String}[x == "NA" ? missing : String(x) for x in c] : String.(c)
        elseif any(x -> x == "NA", c)
            Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c]
        elseif all(x -> x isa Integer, c)
            Int.(c)
        else
            Float64.(c)
        end
        push!(cols, Symbol(n) => v)
    end
    return (; cols...)
end

function _fm_reference()
    m = readdlm(joinpath(_FM_DIR, "fitted_modes_reference.csv"), ',', Any; skipstart = 1)
    ids = String.(m[:, 1])
    return Dict(id => (cond = Float64.(m[ids .== id, 3]), ll = Float64(m[findfirst(==(id), ids), 4]))
                for id in unique(ids))
end

_lgs(x) = 1 / (1 + exp(-x))
function _nb2_lp(y, μ, σ)
    r = 1 / σ^2
    return loggamma(y + r) - loggamma(r) - loggamma(y + 1) +
           r * log(r / (r + μ)) + y * log(μ / (r + μ))
end

# Row log densities as functions of the linear predictor η (own formulas, written from the models).
_dens_zip(y, η, π) = y == 0 ? log(π + (1 - π) * exp(-exp(η))) : log(1 - π) + y * η - exp(η) - loggamma(y + 1)
_dens_zinb(y, η, π, σ) = y == 0 ? log(π + (1 - π) * exp(_nb2_lp(0, exp(η), σ))) : log(1 - π) + _nb2_lp(y, exp(η), σ)
_dens_tnb(y, η, σ) = _nb2_lp(y, exp(η), σ) - log(1 - exp(_nb2_lp(0, exp(η), σ)))
function _dens_ord(k, K, η, cuts)
    pu = k == K ? one(η) : _lgs(cuts[k] - η)             # P(y <= k), 1 for the top category
    pl = k == 1 ? zero(η) : _lgs(cuts[k-1] - η)          # P(y <= k - 1), 0 for the bottom one
    return log(pu - pl)
end

# Conditional modes: maximise sum_i log f(y_i | η_i + z_i' b_g) - b_g'b_g / (2 sd^2) per group by Newton
# with ForwardDiff derivatives; returns the per-row contribution z_i' b̂.
function _fm_modes(rowdens, grp, sds, W)
    n = length(grp); eta_re = zeros(n)
    for g in unique(grp)
        rows = findall(==(g), grp)
        negj(b) = -sum(rowdens(i, sum(W[i][k] * b[k] for k in eachindex(b))) for i in rows) +
                  sum(b .^ 2 ./ (2 .* sds .^ 2))
        b = zeros(length(sds))
        for _ in 1:100
            gr = ForwardDiff.gradient(negj, b); H = ForwardDiff.hessian(negj, b)
            st = H \ gr; b = b .- st
            norm(st) < 1e-13 && break
        end
        for i in rows
            eta_re[i] = sum(W[i][k] * b[k] for k in eachindex(b))
        end
    end
    return eta_re
end

const _FM_BOUND = 1e-6

@testset "fitted(): modes on the zero-inflated, truncated and ordinal Laplace routes" begin
    d = _fm_data(); ref = _fm_reference()
    grp = d.g; x1 = d.x1; n = length(grp)

    @testset "ZIP (1 | g)" begin
        id = "E|zip|mu=(1|g)|zi=1"
        fit = drm(bf(@formula(y_zip ~ x1 + (1 | g)), @formula(zi ~ 1)), DRModels.Poisson(); data = d)
        r = ref[id]
        @test abs(loglikelihood(fit) - r.ll) < 1e-4
        cond = fitted(fit)
        @test length(cond) == nobs(fit) == length(r.cond)
        @test maximum(abs.(cond .- r.cond)) < _FM_BOUND
        pop = fitted(fit; re = :zero)
        @test maximum(abs.(pop .- cond)) > 0.1                    # the modes matter
        @test pop == fit.means[:mu] .* (1 .- fit.scales[:zi])
        β = coef(fit, :mu); π = _lgs(only(coef(fit, :zi))); sd = exp(only(coef(fit, :resd)))
        ηp = β[1] .+ β[2] .* x1
        er = _fm_modes((i, e) -> _dens_zip(d.y_zip[i], ηp[i] + e, π), grp, [sd], [[1.0] for _ in 1:n])
        @test maximum(abs.(cond .- (1 - π) .* exp.(ηp .+ er))) < 1e-6
        @test residuals(fit) ≈ d.y_zip .- cond
    end

    @testset "ZINB (1 | g)" begin
        id = "E|zinb|mu=(1|g)|zi=1"
        fit = drm(bf(@formula(y_zinb ~ x1 + (1 | g)), @formula(sigma ~ 1), @formula(zi ~ 1)), DRModels.NegBinomial2(); data = d)
        r = ref[id]
        @test abs(loglikelihood(fit) - r.ll) < 1e-4
        cond = fitted(fit)
        @test maximum(abs.(cond .- r.cond)) < _FM_BOUND
        @test maximum(abs.(fitted(fit; re = :zero) .- cond)) > 0.1
        β = coef(fit, :mu); σ = exp(only(coef(fit, :sigma))); π = _lgs(only(coef(fit, :zi))); sd = exp(only(coef(fit, :resd)))
        ηp = β[1] .+ β[2] .* x1
        er = _fm_modes((i, e) -> _dens_zinb(d.y_zinb[i], ηp[i] + e, π, σ), grp, [sd], [[1.0] for _ in 1:n])
        @test maximum(abs.(cond .- (1 - π) .* exp.(ηp .+ er))) < 1e-6
    end

    @testset "zero-truncated NB2 (1 | g)" begin
        id = "E|tnb|mu=(1|g)"
        fit = drm(bf(@formula(y_tnb ~ x1 + (1 | g)), @formula(sigma ~ 1)), DRModels.TruncatedNegBinomial2(); data = d)
        r = ref[id]
        @test abs(loglikelihood(fit) - r.ll) < 1e-4
        cond = fitted(fit)
        @test maximum(abs.(cond .- r.cond)) < _FM_BOUND
        @test maximum(abs.(fitted(fit; re = :zero) .- cond)) > 0.1
        β = coef(fit, :mu); σ = exp(only(coef(fit, :sigma))); sd = exp(only(coef(fit, :resd)))
        ηp = β[1] .+ β[2] .* x1
        er = _fm_modes((i, e) -> _dens_tnb(d.y_tnb[i], ηp[i] + e, σ), grp, [sd], [[1.0] for _ in 1:n])
        μc = exp.(ηp .+ er)
        p0 = [exp(_nb2_lp(0, m, σ)) for m in μc]
        @test maximum(abs.(cond .- μc ./ (1 .- p0))) < 1e-6        # truncated mean at the conditional mu
    end

    @testset "CumulativeLogit (1 | g) + (0 + x1 | g)" begin
        id = "I|cumul|mu=x1+(1|g)+(0+x1|g)"
        fit = drm(bf(@formula(y_ord ~ x1 + (1 | g) + (0 + x1 | g))), DRModels.CumulativeLogit(); data = d)
        r = ref[id]
        @test abs(loglikelihood(fit) - r.ll) < 1e-4
        cond = fitted(fit)
        @test maximum(abs.(cond .- r.cond)) < 2e-5
        @test maximum(abs.(fitted(fit; re = :zero) .- cond)) > 0.05
        β = only(coef(fit, :mu)); cuts = DRModels._cumulative_cuts(coef(fit, :cutpoints))
        sds = exp.(coef(fit, :resd)); K = length(cuts) + 1
        ηp = β .* x1
        er = _fm_modes((i, e) -> _dens_ord(d.y_ord[i], K, ηp[i] + e, cuts), grp, sds, [[1.0, x1[i]] for i in 1:n])
        sc = [sum(k * (_lgs(k == K ? Inf : cuts[k] - (ηp[i] + er[i])) -
                       _lgs(k == 1 ? -Inf : cuts[k-1] - (ηp[i] + er[i]))) for k in 1:K) for i in 1:n]
        @test maximum(abs.(cond .- sc)) < 1e-6
    end

    @testset "no modes lost elsewhere: a sigma random effect still refuses" begin
        fit = drm(bf(@formula(y_tnb ~ x1 + (1 | g)), @formula(sigma ~ 1)), DRModels.TruncatedNegBinomial2(); data = d)
        @test haskey(fit.scales, :cond_fit_eta_re)
        @test !haskey(fit.scales, :cond_eta_re)            # quantile residuals and the bridge are untouched
    end
end
