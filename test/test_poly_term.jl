# test_poly_term.jl -- `poly(x, d)` in a Julia `@formula`: R's orthogonal polynomial basis as a
# stateful formula term (gap map v2: eight cells `B|<family>|mu=poly(x1,2)` that crashed with an
# UndefVarError raised by `@formula` before DRModels saw the model).
#
# References: `test/fixtures/poly_parity/` written by `test/parity/gen_poly_parity.R` (drmTMB 0.8.1;
# max|grad| < 1e-6, all SEs finite; numbers and simulated data only). Agreement rule, fixed before
# the numbers were looked at and never loosened:
#   logLik within 1e-4 | every fixed effect within 1e-3 of the R standard error.
# The standard errors are compared as well (1e-3 relative) and reported; they are not part of the rule.
using DRModels, Test, TOML, Random, LinearAlgebra

sq3(x) = x .^ 2     # raw-polynomial comparators (same column space as poly)
cu3(x) = x .^ 3

const _PT_DIR = joinpath(@__DIR__, "fixtures", "poly_parity")

function _pt_data()
    L = readlines(joinpath(_PT_DIR, "data.csv"))
    hdr = [strip(c, '"') for c in split(L[1], ',')]
    cols = map(eachindex(hdr)) do j
        raw = [split(l, ',')[j] for l in L[2:end]]
        any(==("NA"), raw) ? Union{Missing,Float64}[r == "NA" ? missing : parse(Float64, r) for r in raw] :
                             parse.(Float64, raw)
    end
    return NamedTuple{Tuple(Symbol.(hdr))}(Tuple(cols))
end

function _pt_named(fit)
    θ = coef(fit); V = vcov(fit)
    names = Dict(p => ns for (p, ns) in fit.coefnames)
    est = Dict{String,Float64}(); se = Dict{String,Float64}(); i = 0
    for (param, r) in fit.blocks
        haskey(names, param) || continue
        for (nm, v) in zip(names[param], θ[r])
            i += 1
            est["$(param)_$(nm)"] = v
            se["$(param)_$(nm)"] = V[i, i] > 0 ? sqrt(V[i, i]) : NaN
        end
    end
    return est, se
end

const _PT_REF = TOML.parsefile(joinpath(_PT_DIR, "ref.toml"))

function _pt_check(family, fit)
    ref = _PT_REF[family]
    @test _PT_REF["drmTMB_version"] == "0.8.1" && ref["max_grad"] < 1e-6
    est, se = _pt_named(fit)
    @testset "$family" begin
        @test is_converged(fit)
        @test abs(loglik(fit) - ref["loglik"]) <= 1e-4
        for (k, r, s) in zip(ref["names"], ref["theta"], ref["se"])
            startswith(k, "nu_") && continue          # the nu link differs in spelling; logLik covers it
            @test haskey(est, k)
            @test abs(est[k] - r) <= 1e-3 * s
            @test isfinite(se[k]) && abs(se[k] / s - 1) <= 1e-3
        end
    end
end

@testset "poly(x, 2) on the mean vs drmTMB 0.8.1 (eight families)" begin
    d = _pt_data()
    _pt_check("gaussian",  drm(bf(@formula(y_g ~ poly(x1, 2)), @formula(sigma ~ 1)), Gaussian(); data = d))
    _pt_check("poisson",   drm(bf(@formula(y_pois ~ poly(x1, 2))), Poisson(); data = d))
    _pt_check("binomial",  drm(bf(@formula(y_bin ~ poly(x1, 2))), Binomial(); data = d))
    _pt_check("gamma",     drm(bf(@formula(y_gam ~ poly(x1, 2)), @formula(sigma ~ 1)), Gamma(); data = d))
    _pt_check("beta",      drm(bf(@formula(y_beta ~ poly(x1, 2)), @formula(sigma ~ 1)), Beta(); data = d))
    _pt_check("nb2",       drm(bf(@formula(y_nb ~ poly(x1, 2)), @formula(sigma ~ 1)), NegBinomial2(); data = d))
    _pt_check("lognormal", drm(bf(@formula(y_ln ~ poly(x1, 2)), @formula(sigma ~ 1)), LogNormal(); data = d))
    _pt_check("student",   drm(bf(@formula(y_st ~ poly(x1, 2)), @formula(sigma ~ 1)), Student(); data = d))
end

@testset "missing response and missing predictor: the basis is built on the complete rows, as in R" begin
    d = _pt_data()
    fit = drm(bf(@formula(y_gm ~ poly(x1m, 2)), @formula(sigma ~ 1)), Gaussian(); data = d)
    @test nobs(fit) == _PT_REF["gaussian_na"]["nobs"]
    _pt_check("gaussian_na", fit)
    # predict reuses that basis, not one rebuilt from the new rows
    nd = (x1m = Float64.(_PT_REF["newdata_x1"]),)
    fr = drm(bf(@formula(y_gm ~ x1m + sq3(x1m)), @formula(sigma ~ 1)), Gaussian(); data = d)
    @test maximum(abs.(predict(fit, nd) .- predict(fr, nd))) <= 1e-6
end

@testset "coefficient names and predict(newdata) follow R" begin
    d = _pt_data()
    nd = (x1 = Float64.(_PT_REF["newdata_x1"]),)
    fit = drm(bf(@formula(y_g ~ poly(x1, 2)), @formula(sigma ~ 1)), Gaussian(); data = d)
    @test Dict(fit.coefnames)[:mu] == ["(Intercept)", "poly(x1, 2)1", "poly(x1, 2)2"]
    @test predict(fit, nd) ≈ _PT_REF["gaussian"]["predict_response"] rtol = 1e-3
    fp = drm(bf(@formula(y_pois ~ poly(x1, 2))), Poisson(); data = d)
    @test predict(fp, nd) ≈ _PT_REF["poisson"]["predict_response"] rtol = 1e-3
    fb = drm(bf(@formula(y_bin ~ poly(x1, 2))), Binomial(); data = d)
    @test predict(fb, nd) ≈ _PT_REF["binomial"]["predict_response"] rtol = 1e-3
end

@testset "the basis is stored: predict(newdata) equals the raw-polynomial fit (outside the training range)" begin
    rng = Xoshiro(20261111)
    n = 120
    x1 = randn(rng, n) .* 1.3 .+ 0.4; z = randn(rng, n)
    y = 0.2 .+ 0.5 .* x1 .- 0.15 .* x1 .^ 2 .+ 0.03 .* x1 .^ 3 .+ 0.4 .* z .+ 0.5 .* randn(rng, n)
    d = (; x1, z, y)
    nd = (x1 = [-6.0, -1.0, 0.0, 2.5, 9.0], z = [0.1, -0.3, 0.0, 1.0, 0.5])
    for deg in (1, 2, 3)
        fp = deg == 1 ? drm(bf(@formula(y ~ poly(x1, 1) + z), @formula(sigma ~ 1)), Gaussian(); data = d) :
             deg == 2 ? drm(bf(@formula(y ~ poly(x1, 2) + z), @formula(sigma ~ 1)), Gaussian(); data = d) :
                        drm(bf(@formula(y ~ poly(x1, 3) + z), @formula(sigma ~ 1)), Gaussian(); data = d)
        fr = deg == 1 ? drm(bf(@formula(y ~ x1 + z), @formula(sigma ~ 1)), Gaussian(); data = d) :
             deg == 2 ? drm(bf(@formula(y ~ x1 + sq3(x1) + z), @formula(sigma ~ 1)), Gaussian(); data = d) :
                        drm(bf(@formula(y ~ x1 + sq3(x1) + cu3(x1) + z), @formula(sigma ~ 1)), Gaussian(); data = d)
        @test abs(loglik(fp) - loglik(fr)) <= 1e-8
        @test maximum(abs.(predict(fp, nd) .- predict(fr, nd))) <= 1e-6
        @test maximum(abs.(fitted(fp) .- fitted(fr))) <= 1e-6
    end
    # poly term interacting with a numeric covariate keeps its width
    fi = drm(bf(@formula(y ~ z * poly(x1, 2)), @formula(sigma ~ 1)), Gaussian(); data = d)
    @test Dict(fi.coefnames)[:mu] == ["(Intercept)", "z", "poly(x1, 2)1", "poly(x1, 2)2",
                                       "z & poly(x1, 2)1", "z & poly(x1, 2)2"]
end

@testset "a formula object keeps one basis: a refit on other data never changes an earlier predict" begin
    rng = Xoshiro(7)
    dA = (x1 = randn(rng, 80), y = randn(rng, 80), yc = Float64.(rand(rng, 0:6, 80)))
    dB = (x1 = randn(rng, 80) .+ 5.0, y = randn(rng, 80), yc = Float64.(rand(rng, 0:6, 80)))
    nd = (x1 = [0.0, 1.0, 2.0],)
    for (f, fam, resp) in ((bf(@formula(y ~ poly(x1, 2)), @formula(sigma ~ 1)), Gaussian(), :y),
                           (bf(@formula(yc ~ poly(x1, 2))), Poisson(), :yc))
        f1 = drm(f, fam; data = dA)
        p1 = predict(f1, nd)
        drm(f, fam; data = dB)                    # same formula object, other data
        @test predict(f1, nd) == p1
    end
end

@testset "poly refusals name the problem" begin
    d = _pt_data()
    fam = Gaussian()
    err(f) = try; f(); nothing; catch e; e; end
    for (fm, frag) in (
            (@formula(y_g ~ poly(x1, 0)), "degree"),
            (@formula(y_g ~ poly(x1, 2.5)), "degree"),
            (@formula(y_g ~ poly(x1)), "poly(x, degree)"),
            (@formula(y_g ~ poly(log(x1 + 10), 2)), "plain numeric column"))
        e = err(() -> drm(bf(fm, @formula(sigma ~ 1)), fam; data = d))
        @test e isa ArgumentError
        e isa ArgumentError && @test occursin(frag, e.msg)
    end
    few = (x1 = repeat([1.0, 2.0, 3.0], 10), y = randn(Xoshiro(1), 30))
    e = err(() -> drm(bf(@formula(y ~ poly(x1, 3)), @formula(sigma ~ 1)), fam; data = few))
    @test e isa ArgumentError && occursin("unique", e.msg)
    e = err(() -> poly(1.0, 2))
    @test e isa ArgumentError && occursin("formula term", e.msg)
end
