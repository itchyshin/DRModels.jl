# fitted(): conditional modes on four routes that kept none before: a LogNormal `sigma` random
# intercept, a mean `(1 | g)` with a `sigma` `(1 | g2)` on another group, and a Gaussian `(1 | g)` fit
# with prior `weights` (ML and REML).
#
# Two kinds of check. (1) The drmTMB 0.8.1 reference values for these cells are in
# test_fitted_conditional.jl (same fixture, 1e-6). (2) Here the modes are checked without the package's
# own mode solver: the joint density over the random effects is maximised with a generic optimiser at the
# fitted parameters, and the resulting fitted values are compared to `fitted(fit)` to 1e-6.
using DRModels
using Test
using DelimitedFiles
using LinearAlgebra

const _FM_O = DRModels.Optim
const _FM_DIR = joinpath(@__DIR__, "fixtures", "p17_fitted")

function _fm_data()
    m, h = readdlm(joinpath(_FM_DIR, "data.csv"), ',', Any; header = true)
    facs = Set(["g", "g2", "h", "f", "f2", "gna", "gu", "y_cat", "y_ordf"])
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(String.(vec(h)))
        c = m[:, j]
        any(x -> x == "NA", c) && continue
        push!(cols, Symbol(n) => (n in facs ? String.(c) : all(x -> x isa Integer, c) ? Int.(c) : Float64.(c)))
    end
    return (; cols...)
end

_fm_idx(v) = (u = unique(v); (indexin(v, u), length(u)))
_fm_theta(fit, b) = fit.theta[Dict(fit.blocks)[b]]
_fm_maximise(f, n) = _FM_O.minimizer(_FM_O.optimize(f, zeros(n), _FM_O.LBFGS(),
    _FM_O.Options(g_tol = 1e-11, iterations = 5000); autodiff = :forward))
_fm_lnorm(x, m, s) = -log(s) - 0.5 * log(2π) - 0.5 * ((x - m) / s)^2

@testset "fitted(): modes on sigma and weights routes" begin
    d = _fm_data()
    X = hcat(ones(length(d.x1)), d.x1)

    @testset "mean (1 | g) with sigma (1 | g2): modes by direct maximisation" begin
        fit = drm(bf(@formula(y_g ~ x1 + (1 | g)), @formula(sigma ~ x1 + (1 | g2))), Gaussian(); data = d)
        βμ = _fm_theta(fit, :mu); βσ = _fm_theta(fit, :sigma)
        sd = exp.(_fm_theta(fit, :resd))
        gi, G = _fm_idx(d.g); hi, H = _fm_idx(d.g2)
        # latent v ~ N(0, I): effect = sd * v
        nld(v) = begin
            a = sd[1] .* v[1:G]; b = sd[2] .* v[(G+1):(G+H)]
            s = 0.5 * sum(abs2, v)
            for i in eachindex(d.y_g)
                s -= _fm_lnorm(d.y_g[i], X[i, :]' * βμ + a[gi[i]], exp(X[i, :]' * βσ + b[hi[i]]))
            end
            s
        end
        v = _fm_maximise(nld, G + H)
        a = sd[1] .* v[1:G]
        @test maximum(abs.((fitted(fit) .- fitted(fit; re = :zero)) .- a[gi])) < 1e-6
        @test fitted(fit; re = :zero) == fit.means[:mu]
        @test all(isfinite, residuals(fit))
    end

    @testset "LogNormal sigma (1 | g): conditional sigma by direct maximisation" begin
        fit = drm(bf(@formula(y_ln ~ x1), @formula(sigma ~ x1 + (1 | g))), LogNormal(); data = d)
        βμ = _fm_theta(fit, :mu); βσ = _fm_theta(fit, :sigma); sd = exp(only(_fm_theta(fit, :resd)))
        gi, G = _fm_idx(d.g)
        ly = log.(d.y_ln)
        nld(v) = begin
            b = sd .* v
            s = 0.5 * sum(abs2, v)
            for i in eachindex(ly)
                s -= _fm_lnorm(ly[i], X[i, :]' * βμ, exp(X[i, :]' * βσ + b[gi[i]]))
            end
            s
        end
        b = sd .* _fm_maximise(nld, G)
        want = [exp(X[i, :]' * βμ + exp(X[i, :]' * βσ + b[gi[i]])^2 / 2) for i in eachindex(ly)]
        @test maximum(abs.(fitted(fit) .- want)) < 1e-6
        # population value: sigma at its fixed-effect level, as before
        pop = [exp(X[i, :]' * βμ + exp(X[i, :]' * βσ)^2 / 2) for i in eachindex(ly)]
        @test maximum(abs.(fitted(fit; re = :zero) .- pop)) < 1e-9
        @test maximum(abs.(fitted(fit) .- fitted(fit; re = :zero))) > 1e-3
    end

    for (label, kw) in (("ML", (;)), ("REML", (; method = :REML)))
        @testset "Gaussian (1 | g) with weights, $label: modes by direct maximisation" begin
            fit = drm(bf(@formula(y_g ~ x1 + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, weights = d.w, kw...)
            βμ = _fm_theta(fit, :mu); σ = exp(only(_fm_theta(fit, :sigma))); sd = exp(only(_fm_theta(fit, :resd)))
            gi, G = _fm_idx(d.g)
            nld(v) = 0.5 * sum(abs2, v) - sum(d.w[i] * _fm_lnorm(d.y_g[i], X[i, :]' * βμ + sd * v[gi[i]], σ)
                                              for i in eachindex(d.y_g))
            a = sd .* _fm_maximise(nld, G)
            @test maximum(abs.((fitted(fit) .- fitted(fit; re = :zero)) .- a[gi])) < 1e-6
        end
    end
end

# A missing response row and a dropped predictor row. The conditional sigma and the other modes are solved
# on the used rows, so `fitted(fit)` and `residuals(fit)` must give the used rows only, as drmTMB does.
# Reference: drmTMB 0.8.1, test/fixtures/p17_fitted/p17_fitted_sigma_missing_r.R, same data, 1e-6.
function _fm_data_na()
    m, h = readdlm(joinpath(_FM_DIR, "data.csv"), ',', Any; header = true)
    facs = Set(["g", "g2", "h", "f", "f2", "gna", "gu", "y_cat", "y_ordf"])
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(String.(vec(h)))
        c = m[:, j]
        v = if n in facs
            any(x -> x == "NA", c) ? Union{Missing,String}[x == "NA" ? missing : String(x) for x in c] : String.(c)
        elseif any(x -> x == "NA", c)
            Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c]
        else
            all(x -> x isa Integer, c) ? Int.(c) : Float64.(c)
        end
        push!(cols, Symbol(n) => v)
    end
    d = (; cols...)
    na = ismissing.(d.y_gm)
    miss(v) = Union{Missing,Float64}[na[i] ? missing : Float64(v[i]) for i in eachindex(v)]
    return merge(d, (; y_g_m = miss(d.y_g), y_ln_m = miss(d.y_ln), yi_m = miss(d.yi)))
end

const _FM_NA_CELLS = [
    ("MR|lognormal|sigma=x1+(1|g)", d -> drm(bf(@formula(y_ln_m ~ x1), @formula(sigma ~ x1 + (1 | g))), LogNormal(); data = d), 144),
    ("MR|gaussian|mu=(1|g)|sigma=(1|g2)", d -> drm(bf(@formula(y_g_m ~ x1 + (1 | g)), @formula(sigma ~ x1 + (1 | g2))), Gaussian(); data = d), 144),
    ("MR|gaussian|(1|g)+weights", d -> drm(bf(@formula(y_g_m ~ x1 + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, weights = d.w), 144),
    ("MR|gaussian|(1|g)+weights+REML", d -> drm(bf(@formula(y_g_m ~ x1 + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, weights = d.w, method = :REML), 144),
    ("MR|meta|(1|g)|REML", d -> drm(bf(@formula(yi_m ~ x1 + (1 | g) + meta_V(vi)), @formula(sigma ~ 1)), Gaussian(); data = d, method = :REML), 144),
    ("DP|lognormal|sigma=x1m+(1|g)", d -> drm(bf(@formula(y_ln ~ x1m), @formula(sigma ~ x1m + (1 | g))), LogNormal(); data = d), 142),
    ("DP|gaussian|mu=(1|g)|sigma=(1|g2)", d -> drm(bf(@formula(y_g ~ x1m + (1 | g)), @formula(sigma ~ x1m + (1 | g2))), Gaussian(); data = d), 142),
    ("DP|gaussian|(1|g)+weights", d -> drm(bf(@formula(y_g ~ x1m + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, weights = d.w), 142),
    ("DP|gaussian|(1|g)+weights+REML", d -> drm(bf(@formula(y_g ~ x1m + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, weights = d.w, method = :REML), 142),
    ("DP|meta|(1|g)|REML", d -> drm(bf(@formula(yi ~ x1m + (1 | g) + meta_V(vi)), @formula(sigma ~ 1)), Gaussian(); data = d, method = :REML), 142),
    ("MRDP|lognormal|sigma=x1m+(1|g)", d -> drm(bf(@formula(y_ln_m ~ x1m), @formula(sigma ~ x1m + (1 | g))), LogNormal(); data = d), 128),
]

@testset "fitted(): missing response and dropped predictor rows on the sigma and weights routes" begin
    d = _fm_data_na()
    m = readdlm(joinpath(_FM_DIR, "p17_fitted_sigma_missing_reference.csv"), ',', Any; skipstart = 1)
    for (id, fit_it, nused) in _FM_NA_CELLS
        @testset "$id" begin
            ref = Float64.(m[String.(m[:, 1]) .== id, 3])
            fit = fit_it(d)
            cond = fitted(fit)
            @test nobs(fit) == nused == length(cond) == length(ref)
            @test maximum(abs.(cond .- ref)) < 1e-6                       # drmTMB fitted(fit)
            zero = fitted(fit; re = :zero)
            @test length(zero) == nused
            res = residuals(fit)
            @test length(res) == nused && !any(isnan, res)
            y = first(values(fit.obs))
            used = .!isnan.(y)
            @test count(used) == nused || all(used)                       # rows dropped for a predictor are not in `obs`
            @test length(residuals(fit; re = :zero)) == nused
        end
    end
end
