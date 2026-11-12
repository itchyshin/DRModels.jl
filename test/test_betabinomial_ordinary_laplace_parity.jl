# test_betabinomial_ordinary_laplace_parity.jl -- BetaBinomial with one ordinary `(1 | g)` and
# `marginal = :Laplace` vs drmTMB 0.8.1 (P22c). References: `test/fixtures/bb_laplace_parity/` written by
# `test/parity/gen_bb_laplace_parity.R` (drmTMB 0.8.1; max|grad| < 1e-6, all SEs finite; numbers and
# simulated data only). Bounds are the lane's and are never loosened:
#   logLik 1e-4 (absolute) | fixed effects 1e-3 R SE | log SD 5e-3 | SEs 1e-3 relative.
using DRModels, Test, TOML, Random, LinearAlgebra

const _BBL_DIR = joinpath(@__DIR__, "fixtures", "bb_laplace_parity")

function _bbl_data(route)
    L = readlines(joinpath(_BBL_DIR, route, "data.csv"))
    hdr = [strip(c, '"') for c in split(L[1], ',')]
    cols = Dict{Symbol,Any}()
    for (j, h) in enumerate(hdr)
        raw = [strip(split(l, ',')[j], '"') for l in L[2:end]]
        cols[Symbol(h)] = all(s -> tryparse(Float64, s) !== nothing, raw) ? parse.(Float64, raw) : String.(raw)
    end
    return NamedTuple{Tuple(keys(cols))}(Tuple(values(cols)))
end

function _bbl_named(fit)
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

const _BBL_FORM = bf(@formula(cbind(successes, failures) ~ x + (1 | g)))

@testset "BetaBinomial (1 | g) marginal = :Laplace vs drmTMB 0.8.1, seven data sets" begin
    for k in 1:7
        route = "bb_$k"
        d = _bbl_data(route)
        ref = TOML.parsefile(joinpath(_BBL_DIR, route, "ref.toml"))
        @test ref["max_grad"] < 1e-6 && ref["drmTMB_version"] == "0.8.1"
        fit = drm(_BBL_FORM, BetaBinomial(); data = d, marginal = :Laplace)
        est, se = _bbl_named(fit)
        @testset "$route" begin
            @test is_converged(fit)
            @test fit.marginal === :Laplace
            @test abs(loglik(fit) - ref["loglik"]) <= 1e-4
            for (nm, r, s) in zip(ref["names"], ref["theta"], ref["se"])
                @test haskey(est, nm)
                tol = startswith(nm, "resd_") ? 5e-3 : 1e-3 * s
                @test abs(est[nm] - r) <= tol
                @test isfinite(se[nm]) && abs(se[nm] / s - 1) <= 1e-3
            end
            # the default integrator is unchanged and is a different number (the premise of P22c)
            fla = drm(_BBL_FORM, BetaBinomial(); data = d)
            @test fla.marginal === :LA
            @test abs(loglik(fla) - loglik(fit)) > 1e-3
        end
    end
end

@testset "BetaBinomial :Laplace post-fit functions and refusals" begin
    d = _bbl_data("bb_1")
    fit = drm(_BBL_FORM, BetaBinomial(); data = d, marginal = :Laplace)
    n = length(d.x)
    @test nobs(fit) == n
    @test loglikelihood(fit) == loglik(fit)
    V = vcov(fit)
    @test size(V) == (4, 4) && issymmetric(V) && minimum(eigvals(Symmetric(V))) > 0
    @test predict(fit, d) ≈ fitted(fit)
    @test predict(fit, merge(d, (g = fill("NEW", n),))) ≈ fitted(fit)
    @test length(simulate(fit; data = d, re = :zero, rng = Random.Xoshiro(1))) == n
    @test length(simulate(fit; data = d, rng = Random.Xoshiro(1))) == n
    @test size(bootstrap_summary(fit; data = d, B = 2, rng = Random.Xoshiro(1)), 1) == length(coef(fit))
    @test length(residuals(fit; type = :quantile)) == n
    for bad in (bf(@formula(cbind(successes, failures) ~ x + offset(x) + (1 | g))),
                bf(@formula(cbind(successes, failures) ~ x + (1 | g)), @formula(sigma ~ x)),
                bf(@formula(cbind(successes, failures) ~ x)))
        @test_throws ArgumentError drm(bad, BetaBinomial(); data = d, marginal = :Laplace)
    end
    dh = merge(d, (h = repeat(["a", "b", "c", "d"], n ÷ 4),))
    @test_throws ArgumentError drm(bf(@formula(cbind(successes, failures) ~ x + (1 | g) + (1 | h))),
                                   BetaBinomial(); data = dh, marginal = :Laplace)
    @test_throws ArgumentError drm(_BBL_FORM, BetaBinomial(); data = d, marginal = :Laplace, method = :REML)
end
