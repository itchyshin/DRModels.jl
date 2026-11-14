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
