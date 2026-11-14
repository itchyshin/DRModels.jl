# test_zi_re_laplace_parity.jl -- one ordinary random intercept `(1 | g)` with zero inflation,
# a hurdle, zero truncation, or a SkewNormal response: DRModels.jl vs drmTMB 0.8.1.
#
# These cells were refusals ("random effects cannot be combined with zi/hu yet",
# "supports fixed effects only"). They now fit by the Laplace approximation, the integrator
# drmTMB uses, through the shared spine of `src/indep_slopes_laplace.jl` with the new observation
# terms of `src/zi_re_laplace.jl`. References: `test/fixtures/zi_re_parity/<route>/` written by
# `test/parity/gen_zi_re_parity.R` (drmTMB 0.8.1; max|grad| < 1e-6, all SEs finite; numbers and
# simulated data only). The bounds are fixed and are never loosened:
#   logLik 1e-4 (absolute) | fixed effects 1e-3 R SE | log SD 5e-3 | SEs 1e-3 relative.
using DRModels, Test, TOML

const _ZR_DIR = joinpath(@__DIR__, "fixtures", "zi_re_parity")
const _ZR_FITTED_TOL = 1e-6   # the fitted() bound of test_fitted_conditional.jl for Laplace routes

function _zr_data(route)
    L = readlines(joinpath(_ZR_DIR, route, "data.csv"))
    hdr = [strip(c, '"') for c in split(L[1], ',')]
    cols = Dict{Symbol,Any}()
    for (j, h) in enumerate(hdr)
        raw = [strip(split(l, ',')[j], '"') for l in L[2:end]]
        cols[Symbol(h)] = all(s -> tryparse(Float64, s) !== nothing, raw) ? parse.(Float64, raw) : String.(raw)
    end
    return NamedTuple{Tuple(keys(cols))}(Tuple(values(cols)))
end

function _zr_named(fit)
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

function _zr_check(route, fit)
    ref = TOML.parsefile(joinpath(_ZR_DIR, route, "ref.toml"))
    @test ref["max_grad"] < 1e-6 && ref["drmTMB_version"] == "0.8.1"
    est, se = _zr_named(fit)
    @testset "$route" begin
        @test is_converged(fit)
        @test fit.marginal === :Laplace
        @test abs(loglik(fit) - ref["loglik"]) <= 1e-4
        for (k, r, s) in zip(ref["names"], ref["theta"], ref["se"])
            @test haskey(est, k)
            tol = startswith(k, "resd_") ? 5e-3 : 1e-3 * s
            @test abs(est[k] - r) <= tol
            @test isfinite(se[k]) && abs(se[k] / s - 1) <= 1e-3
        end
    end
end

@testset "(1 | g) with zi / hu / truncation / skew-normal vs drmTMB 0.8.1 (Laplace)" begin
    d = _zr_data("zip_g")
    _zr_check("zip_g", drm(bf(@formula(y ~ x + (1 | g)), @formula(zi ~ 1)), Poisson(); data = d))
    d = _zr_data("zinb_g")
    _zr_check("zinb_g", drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(zi ~ 1)),
                            NegBinomial2(); data = d))
    d = _zr_data("tnb_g")
    _zr_check("tnb_g", drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)),
                           TruncatedNegBinomial2(); data = d))
    d = _zr_data("hnb_g")
    _zr_check("hnb_g", drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(hu ~ 1)),
                                              TruncatedNegBinomial2(); data = d))
    # the same model spelled with the NegBinomial2 family (what drmTMB's truncated_nbinom2() + hu is)
    _zr_check("hnb_g", drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(hu ~ 1)),
                           NegBinomial2(); data = d))
    d = _zr_data("skew_g")
    _zr_check("skew_g", drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(nu ~ 1)),
                            SkewNormal(); data = d))
end

# Zero-inflated NB2 data on which the fixed-effects start of the Laplace fit sits on the zero-inflation
# boundary (logit about -30, flat likelihood). Before the interior restarts of `zi_re_laplace.jl` the fit
# stalled there with `converged = false`, 8.3 (`zinb_r3`) and 10.3 (`zinb_r6`) log-likelihood units below
# drmTMB. References come from `test/parity/gen_zi_re_parity.R` (routes `zinb_r3`, `zinb_r6`).
@testset "ZINB (1 | g) with the fixed-effects start on the zero-inflation boundary vs drmTMB 0.8.1" begin
    for route in ("zinb_r3", "zinb_r6")
        d = _zr_data(route)
        fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(zi ~ 1)), NegBinomial2(); data = d)
        _zr_check(route, fit)
        # the zero-inflation coefficient is interior, not parked at the clamp
        @test abs(coef(fit, :zi)[1]) < 5
    end
end

@testset "Binomial with three crossed random intercepts vs drmTMB 0.8.1 (Laplace)" begin
    d = _zr_data("binom3")
    _zr_check("binom3", drm(bf(@formula(y ~ x + (1 | g) + (1 | g2) + (1 | h))), Binomial(); data = d))
    # two components keep their own (sparse) route; AGHQ on three is refused by name
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g) + (1 | g2) + (1 | h))), Binomial();
                                   data = d, marginal = :AGHQ)
end

@testset "CumulativeLogit (1 | g) + (0 + x | g) vs drmTMB 0.8.1 (Laplace)" begin
    d = _zr_data("cumul_gx")
    fit = drm(bf(@formula(y ~ x + (1 | g) + (0 + x | g))), CumulativeLogit(); data = d)
    _zr_check("cumul_gx", fit)
end

@testset "post-fit paths and refusals of the new random-effect routes" begin
    using Random, Statistics
    d = _zr_data("zip_g")
    zf = bf(@formula(y ~ x + (1 | g)), @formula(zi ~ 1))
    fit = drm(zf, Poisson(); data = d)
    # same fit through the explicit Laplace request; update repeats it
    @test loglik(drm(zf, Poisson(); data = d, marginal = :Laplace)) ≈ loglik(fit)
    @test loglik(update(fit, fit.formula; data = d)) ≈ loglik(fit)
    # fitted() is conditional by default and this route keeps no random-effect modes, so the default
    # refuses by name and points to `re = :zero`. With `re = :zero` fitted() is the expected response,
    # (1 - pi) * exp(mu) with the random effect at zero, pinned row by row to drmTMB 0.8.1; predict()
    # is the count-component mean (as for fixed-effects ZIP).
    ferr = try; fitted(fit); nothing; catch e; e; end
    @test ferr isa ArgumentError
    @test occursin("re = :zero", sprint(showerror, ferr))
    fz = fitted(fit; re = :zero)
    fref = readlines(joinpath(_ZR_DIR, "zip_g", "fitted_r.csv"))[2:end]
    fpop = [parse(Float64, split(l, ',')[3]) for l in fref]
    @test length(fz) == length(fpop) == length(d.y)
    @test maximum(abs.(fz .- fpop)) <= _ZR_FITTED_TOL
    @test mean(fz) < mean(predict(fit, d))
    # fresh-random-effect simulation reproduces the data's zero share and mean (marginal draw)
    ys = [simulate(fit; data = d, rng = Random.Xoshiro(s)) for s in 1:30]
    @test abs(mean(count(==(0), y) / length(y) for y in ys) - count(==(0), d.y) / length(d.y)) < 0.04
    @test abs(mean(mean.(ys)) - mean(d.y)) < 0.15
    @test size(bootstrap_summary(fit; data = d, B = 2, rng = Random.Xoshiro(1)), 1) == 4
    r = residuals(fit; type = :quantile, rng = Random.Xoshiro(1))
    @test 0.85 < std(r) < 1.15
    # unsupported shapes are refused by name, never fitted as a different model
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g)), @formula(zi ~ 1)), Poisson(); data = d, marginal = :AGHQ)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g)), @formula(zi ~ 1)), Poisson(); data = d, marginal = :VA)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g)), @formula(zi ~ 1)), Poisson(); data = d, method = :REML)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 + x | g)), @formula(zi ~ 1)), Poisson(); data = d)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g)), @formula(zi ~ 1 + (1 | g))), Poisson(); data = d)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g)), @formula(hu ~ 1)), Poisson(); data = d)
    # an offset is NOT combined with these routes (not verified against drmTMB): refused by name
    for (fm, fam) in ((bf(@formula(y ~ x + offset(x) + (1 | g)), @formula(zi ~ 1)), Poisson()),
                      (bf(@formula(y ~ x + offset(x) + (1 | g)), @formula(sigma ~ 1), @formula(zi ~ 1)), NegBinomial2()))
        err = try drm(fm, fam; data = d); nothing catch e; e end
        @test err !== nothing && occursin("offset", sprint(showerror, err))
    end
    @test_throws ArgumentError drm(bf(@formula(y ~ x + offset(x) + (1 | g)), @formula(sigma ~ 1)),
                                   TruncatedNegBinomial2(); data = _zr_data("tnb_g"))
    @test_throws ArgumentError drm(bf(@formula(y ~ x + offset(x) + (1 | g)), @formula(sigma ~ 1), @formula(nu ~ 1)),
                                   SkewNormal(); data = _zr_data("skew_g"))
    # `marginal` is accepted only where the route exists
    ds = _zr_data("skew_g")
    @test_throws r"SkewNormal route does not accept the keyword `marginal`" drm(
        bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), SkewNormal(); data = ds, marginal = :Laplace)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(nu ~ 1)), SkewNormal();
                                   data = ds, marginal = :AGHQ)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1 + (1 | g)), @formula(nu ~ 1)),
                                   SkewNormal(); data = ds)
    sk = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(nu ~ 1)), SkewNormal(); data = ds)
    @test loglik(update(sk, sk.formula; data = ds)) ≈ loglik(sk)
    # the multi-term fits: refit works, fresh-draw paths are refused by name
    dc = _zr_data("cumul_gx")
    cf = drm(bf(@formula(y ~ x + (1 | g) + (0 + x | g))), CumulativeLogit(); data = dc)
    @test loglik(update(cf, cf.formula; data = dc)) ≈ loglik(cf)
    @test_throws ArgumentError bootstrap_summary(cf; data = dc, B = 2, rng = Random.Xoshiro(1))
    @test_throws ArgumentError residuals(cf; type = :quantile)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + offset(x) + (1 | g) + (0 + x | g))), CumulativeLogit(); data = dc)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g) + (0 + x | g))), CumulativeLogit(); data = dc, marginal = :AGHQ)
    db = _zr_data("binom3")
    bf3 = drm(bf(@formula(y ~ x + (1 | g) + (1 | g2) + (1 | h))), Binomial(); data = db)
    # The refusal is gone: a fresh-random-effect simulator for several ordinary random intercepts is
    # on this tree. Its draws carry the random effects: the variance of the group means is the data's,
    # not the binomial noise alone (about a third of it).
    @test size(bootstrap_summary(bf3; data = db, B = 2, rng = Random.Xoshiro(1)), 1) == 5
    gl = unique(db.g)
    gmv(y) = var([mean(y[db.g .== k]) for k in gl])
    vdata = gmv(db.y)
    vdraw = mean(gmv(simulate(bf3; data = db, rng = Random.Xoshiro(k))) for k in 1:100)
    pbar = mean(db.y)
    vnoise = mean(pbar * (1 - pbar) / count(==(k), db.g) for k in gl)
    @test vdraw > 2 * vnoise
    @test 0.75 < vdraw / vdata < 1.33
    @test_throws ArgumentError residuals(bf3; type = :quantile)
end

@testset "non-constant zi / hu / sigma / nu formulas vs drmTMB 0.8.1 (Laplace)" begin
    d = _zr_data("zip_zix")
    _zr_check("zip_zix", drm(bf(@formula(y ~ x + (1 | g)), @formula(zi ~ x)), Poisson(); data = d))
    d = _zr_data("zinb_sx_zx")
    _zr_check("zinb_sx_zx", drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x), @formula(zi ~ x)),
                                NegBinomial2(); data = d))
    d = _zr_data("tnb_sx")
    _zr_check("tnb_sx", drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x)), TruncatedNegBinomial2(); data = d))
    d = _zr_data("hnb_sx_hx")
    _zr_check("hnb_sx_hx", drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x), @formula(hu ~ x)),
                               TruncatedNegBinomial2(); data = d))
    d = _zr_data("skew_sx")
    _zr_check("skew_sx", drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x), @formula(nu ~ 1)), SkewNormal(); data = d))
    d = _zr_data("skew_nx")
    _zr_check("skew_nx", drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(nu ~ x)), SkewNormal(); data = d))
end

@testset "fresh-draw simulate for zinb, tnb, hnb, skew; nAGQ and size guards" begin
    using Random, Statistics
    cases = (("zinb_g", bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(zi ~ 1)), NegBinomial2()),
             ("tnb_g", bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), TruncatedNegBinomial2()),
             ("hnb_g", bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(hu ~ 1)), NegBinomial2()),
             ("skew_g", bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(nu ~ 1)), SkewNormal()))
    for (route, f, fam) in cases
        d = _zr_data(route); fit = drm(f, fam; data = d)
        ys = [simulate(fit; data = d, rng = Random.Xoshiro(s)) for s in 1:30]
        zshare(y) = count(==(0), y) / length(y)
        @test abs(mean(zshare.(ys)) - zshare(d.y)) < 0.05
        @test abs(mean(mean.(ys)) - mean(d.y)) < 0.2 * max(1.0, abs(mean(d.y)))
        @test abs(mean(var.(ys)) / var(d.y) - 1) < 0.3
    end
    d = _zr_data("zip_g")
    zf = bf(@formula(y ~ x + (1 | g)), @formula(zi ~ 1))
    @test_throws ArgumentError drm(zf, Poisson(); data = d, nAGQ = 5)
    @test_throws ArgumentError drm(zf, Poisson(); data = d, nAGQ = 5, marginal = :Laplace)
    @test loglik(drm(zf, Poisson(); data = d)) ≈ -422.95270749241683 atol = 1e-6
    # dense engine size limit (same limit as the other dense Laplace engines)
    n = 1500
    big = (y = Float64.(rand(Random.Xoshiro(1), 0:1, n)), x = randn(Random.Xoshiro(2), n),
           a = string.("a", rand(Random.Xoshiro(3), 1:300, n)), b = string.("b", rand(Random.Xoshiro(4), 1:300, n)),
           c = string.("c", rand(Random.Xoshiro(5), 1:300, n)))
    err = try drm(bf(@formula(y ~ x + (1 | a) + (1 | b) + (1 | c))), Binomial(); data = big); nothing catch e; e end
    @test err isa ArgumentError && occursin("limited to 600 latent coordinates", sprint(showerror, err))
end
