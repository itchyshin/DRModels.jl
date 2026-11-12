# test_poly_term.jl -- `poly(x, d)` in a Julia `@formula`: R's orthogonal polynomial basis as a
# stateful formula term (gap map v2: eight cells `B|<family>|mu=poly(x1,2)` that crashed with an
# UndefVarError raised by `@formula` before DRModels saw the model).
#
# References: `test/fixtures/poly_parity/` written by `test/parity/gen_poly_parity.R` (drmTMB 0.8.1;
# max|grad| < 1e-6, all SEs finite; numbers and simulated data only). Agreement rule, fixed before
# the numbers were looked at and never loosened:
#   logLik within 1e-4 | every fixed effect within 1e-3 of the R standard error.
# The standard errors are compared as well (1e-3 relative) and reported; they are not part of the rule.
using DRModels, Test, TOML, Random, LinearAlgebra, Serialization

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

@testset "Gaussian REML with poly(x, 2): the restricted log-likelihood depends on the basis, so it needs R's" begin
    d = _pt_data()
    fit = drm(bf(@formula(y_g ~ poly(x1, 2)), @formula(sigma ~ 1)), Gaussian(); data = d, method = :REML)
    ref = _PT_REF["gaussian_reml"]
    @test is_converged(fit)
    @test abs(loglik(fit) - ref["loglik"]) <= 1e-4
    est, se = _pt_named(fit)
    for (k, r, s) in zip(ref["names"], ref["theta"], ref["se"])
        @test abs(est[k] - r) <= 1e-3 * s
    end
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

@testset "a refit with the same formula object on other data never changes an earlier fit's predict" begin
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

# ---------------------------------------------------------------------------------------------
# Every fit builds its own basis (R recomputes `poly` on each call). A `bf(...)` object reused for a
# second fit on other data must behave exactly like a fresh one, the basis belongs to the FIT, and
# `predict(fit, newdata)` always uses the basis of that fit.
# ---------------------------------------------------------------------------------------------
_pr_close(a, b) = maximum(abs.(a .- b)) <= 1e-12 * (1 + maximum(abs.(b)))

function _pr_same(f1, f2, nd)
    @test _pr_close(coef(f1), coef(f2))
    @test abs(loglik(f1) - loglik(f2)) <= 1e-12 * (1 + abs(loglik(f2)))
    @test _pr_close(vcov(f1), vcov(f2))
    @test _pr_close(predict(f1, nd), predict(f2, nd))
end

function _pr_data(seed; shift = 0.0, scale = 1.0, n = 90)
    rng = Xoshiro(seed)
    x = randn(rng, n) .* scale .+ shift
    z = (x .- shift) ./ scale
    (x = x, y = 0.3 .+ 0.4 .* z .- 0.3 .* z .^ 2 .+ 0.5 .* randn(rng, n),
     yc = Float64.(rand(rng, 0:6, n)), yb = Float64.(rand(rng, n) .< 0.3 .+ 0.2 .* (z .> 0)),
     yg = exp.(0.2 .* z .+ 0.3 .* randn(rng, n)) .+ 0.5)
end

const _PR_SPECS = (
    ("Gaussian ML",   () -> bf(@formula(y ~ poly(x, 2)), @formula(sigma ~ 1)),  Gaussian(), (;)),
    ("Gaussian REML", () -> bf(@formula(y ~ poly(x, 2)), @formula(sigma ~ 1)),  Gaussian(), (; method = :REML)),
    ("Poisson",       () -> bf(@formula(yc ~ poly(x, 2))),                      Poisson(),  (;)),
    ("Binomial",      () -> bf(@formula(yb ~ poly(x, 2))),                      Binomial(), (;)),
    ("Gamma",         () -> bf(@formula(yg ~ poly(x, 2)), @formula(sigma ~ 1)), Gamma(),    (;)),
)

@testset "a reused formula object gives exactly the fit of a fresh one, and predict follows its own fit" begin
    dA = _pr_data(11); dB = _pr_data(12; shift = 5.0, scale = 2.0)
    ndA = (x = [-1.0, 0.0, 1.5],); ndB = (x = [4.0, 5.0, 6.5],)
    for (label, mk, fam, kw) in _PR_SPECS
        @testset "$label" begin
            f = mk()
            fA = drm(f, fam; data = dA, kw...)
            pA = predict(fA, ndA)
            fB = drm(f, fam; data = dB, kw...)                 # same object, other data
            fresh = drm(mk(), fam; data = dB, kw...)
            _pr_same(fB, fresh, ndB)
            _pr_same(fA, drm(mk(), fam; data = dA, kw...), ndA)
            @test predict(fA, ndA) == pA                        # the first fit keeps its own basis
            @test isempty(f.schema_cache[])                      # fitting never writes to the user's object
            # third fit on the first data again: still the fresh result
            _pr_same(drm(f, fam; data = dA, kw...), fA, ndA)
        end
    end
end

@testset "cross-validation loop with one formula object equals the loop with fresh formulas" begin
    d = _pr_data(21; shift = 3.0, scale = 1.5, n = 100)
    folds = (1:2:100, 2:2:100)
    sub(d, idx) = map(c -> c[idx], d)
    for (label, mk, fam, kw) in _PR_SPECS[[1, 3]]
        f = mk(); pr_reused = Float64[]; pr_fresh = Float64[]
        for k in 1:2
            test = collect(folds[k]); train = collect(folds[3 - k])
            append!(pr_reused, predict(drm(f, fam; data = sub(d, train), kw...), sub(d, test)))
            append!(pr_fresh, predict(drm(mk(), fam; data = sub(d, train), kw...), sub(d, test)))
        end
        @test _pr_close(pr_reused, pr_fresh)
    end
end

@testset "shifted data (x + 1000) after another fit: converges and matches drmTMB 0.8.1" begin
    d = _pt_data()
    ref = TOML.parsefile(joinpath(_PT_DIR, "ref_shift.toml"))
    @test ref["drmTMB_version"] == "0.8.1"
    d0 = merge(d, (x1s = d.x1,)); d1 = merge(d, (x1s = d.x1 .+ 1000,))
    nd = (x1s = Float64.(ref["newdata_x1s"]),)
    for (key, mk, fam, kw, resp) in (
            ("gaussian_shift",      () -> bf(@formula(y_g ~ poly(x1s, 2)), @formula(sigma ~ 1)), Gaussian(), (;), :y_g),
            ("gaussian_reml_shift", () -> bf(@formula(y_g ~ poly(x1s, 2)), @formula(sigma ~ 1)), Gaussian(), (; method = :REML), :y_g),
            ("poisson_shift",       () -> bf(@formula(y_pois ~ poly(x1s, 2))),                   Poisson(),  (;), :y_pois))
        @testset "$key" begin
            r = ref[key]; f = mk()
            drm(f, fam; data = d0, kw...)                         # a first fit on unshifted data
            fit = drm(f, fam; data = d1, kw...)                   # same object, shifted data
            @test is_converged(fit)
            @test abs(loglik(fit) - r["loglik"]) <= 1e-4
            est, se = _pt_named(fit)
            for (k, v, s) in zip(r["names"], r["theta"], r["se"])
                @test abs(est[k] - v) <= 1e-3 * s
                @test isfinite(se[k]) && abs(se[k] / s - 1) <= 1e-3
            end
            @test predict(fit, nd) ≈ r["predict_response"] rtol = 1e-6
        end
    end
end

@testset "refits made inside the package use the basis of their own data" begin
    dA = _pr_data(31); dB = _pr_data(32; shift = 5.0, scale = 2.0)
    mk = () -> bf(@formula(y ~ poly(x, 2)), @formula(sigma ~ 1))
    f = mk()
    drm(f, Gaussian(); data = dA)
    fB = drm(f, Gaussian(); data = dB)
    # update with the same formula object on other data = fresh fit
    fu = update(fB, f; data = dA)
    _pr_same(fu, drm(mk(), Gaussian(); data = dA), (x = [-1.0, 0.0, 1.0],))
    # simulate from a fit, then refit on the simulated response
    sim = simulate(fB; rng = Xoshiro(5))
    dS = merge(dB, (y = sim,))
    _pr_same(drm(f, Gaussian(); data = dS), drm(mk(), Gaussian(); data = dS), (x = [4.0, 5.0, 6.5],))
    # parametric bootstrap from the fit and from the (reused) formula object, serial and threaded:
    # every refit builds its own basis, so the intervals equal those of a fresh formula object
    fresh = drm(mk(), Gaussian(); data = dB)
    ci_row(r) = (r.estimate, r.lower, r.upper)
    bt_fit   = bootstrap_ci(fB; data = dB, B = 6, rng = Xoshiro(9))
    bt_fresh = bootstrap_ci(fresh; data = dB, B = 6, rng = Xoshiro(9))
    @test _pr_close(collect(Iterators.flatten(ci_row.(bt_fit))), collect(Iterators.flatten(ci_row.(bt_fresh))))
    bt_f  = bootstrap_ci(f, Gaussian(); data = dB, B = 6, rng = Xoshiro(9))
    bt_f2 = bootstrap_ci(mk(), Gaussian(); data = dB, B = 6, rng = Xoshiro(9))
    @test _pr_close(collect(Iterators.flatten(ci_row.(bt_f))), collect(Iterators.flatten(ci_row.(bt_f2))))
    bt_thr = bootstrap_ci(f, Gaussian(); data = dB, B = 6, rng = Xoshiro(9), threads = true)
    @test _pr_close(collect(Iterators.flatten(ci_row.(bt_thr))), collect(Iterators.flatten(ci_row.(bt_f2))))
    # profile likelihood on the reused-formula fit equals that of the fresh fit
    pr_a = confint(fB; method = :profile); pr_b = confint(fresh; method = :profile)
    @test _pr_close([r.lower for r in pr_a], [r.lower for r in pr_b])
    @test _pr_close([r.upper for r in pr_a], [r.upper for r in pr_b])
    @test all(r -> isfinite(r.lower) && isfinite(r.upper), confint(fB))
end

@testset "two fits from one formula object running at the same time" begin
    dA = _pr_data(41); dB = _pr_data(42; shift = 5.0, scale = 2.0)
    for (label, mk, fam, kw) in _PR_SPECS[[1, 3]]
        f = mk()
        ts = [Threads.@spawn(drm(f, fam; data = d, kw...)) for d in (dA, dB, dA, dB)]
        fits = fetch.(ts)
        for (ft, d, nd) in zip(fits, (dA, dB, dA, dB), ((x = [-1.0, 0.5],), (x = [4.0, 6.0],), (x = [-1.0, 0.5],), (x = [4.0, 6.0],)))
            _pr_same(ft, drm(mk(), fam; data = d, kw...), nd)
        end
    end
end

@testset "bivariate Gaussian: a reused formula object follows each fit's own basis" begin
    mkb = () -> bf(mu1 = @formula(y1 ~ poly(x, 2)), mu2 = @formula(y2 ~ x),
                   sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
    bd(seed; shift, scale) = begin
        d = _pr_data(seed; shift, scale); rng = Xoshiro(seed + 1000)
        merge(d, (y1 = d.y, y2 = 0.2 .* d.x .+ 0.5 .* randn(rng, length(d.x))))
    end
    dA = bd(81; shift = 0.0, scale = 1.0); dB = bd(82; shift = 5.0, scale = 2.0)
    nd = (x = [4.0, 5.0, 6.5],)
    f = mkb()
    fA = drm(f, Gaussian(); data = dA); pA = predict(fA, (x = [-1.0, 0.5],))
    fB = drm(f, Gaussian(); data = dB)
    fresh = drm(mkb(), Gaussian(); data = dB)
    @test _pr_close(coef(fB), coef(fresh))
    @test abs(loglik(fB) - loglik(fresh)) <= 1e-12 * (1 + abs(loglik(fresh)))
    @test _pr_close(predict(fB, nd)[:mu1], predict(fresh, nd)[:mu1])
    @test predict(fA, (x = [-1.0, 0.5],))[:mu1] == pA[:mu1]
    @test isempty(f.schema_cache[])
end

@testset "R bridge: poly columns are built per call from that call's data" begin
    dA = _pr_data(71); dB = _pr_data(72; shift = 5.0, scale = 2.0)
    for d in (dA, dB, dA)
        o = drm_bridge(; formula = "y ~ poly(x, 2); sigma ~ 1", family = "gaussian", data = d)
        fresh = drm(bf(@formula(y ~ poly(x, 2)), @formula(sigma ~ 1)), Gaussian(); data = d)
        @test abs(o["loglik"] - loglik(fresh)) <= 1e-8
        @test any(occursin("poly(x, 2)1", n) for n in o["coef_names"])
    end
end

@testset "same mechanism for factor levels: a reused formula object follows each fit's own levels" begin
    rng = Xoshiro(61); n = 60
    mk = () -> bf(@formula(y ~ g + x), @formula(sigma ~ 1))
    dA = (x = randn(rng, n), g = repeat(["a", "b", "c"], 20)); dA = merge(dA, (y = randn(rng, n),))
    dB = (x = randn(rng, n), g = repeat(["u", "v", "w"], 20)); dB = merge(dB, (y = randn(rng, n),))
    f = mk()
    drm(f, Gaussian(); data = dA)
    fB = drm(f, Gaussian(); data = dB)
    fresh = drm(mk(), Gaussian(); data = dB)
    @test fB.coefnames == fresh.coefnames
    @test _pr_close(coef(fB), coef(fresh))
    nd = (x = [0.0, 1.0], g = ["v", "w"])
    @test _pr_close(predict(fB, nd), predict(fresh, nd))
end

@testset "the basis lives in the fit: deepcopy and serialization keep predict" begin
    dA = _pr_data(51); dB = _pr_data(52; shift = 5.0, scale = 2.0)
    nd = (x = [4.0, 5.0, 6.5],)
    f = bf(@formula(yc ~ poly(x, 2)))
    drm(f, Poisson(); data = dA)
    fB = drm(f, Poisson(); data = dB)
    p = predict(fB, nd)
    @test predict(deepcopy(fB), nd) == p
    buf = IOBuffer(); serialize(buf, fB); seekstart(buf)
    @test predict(deserialize(buf), nd) == p
    drm(f, Poisson(); data = dA)                                   # a later fit does not touch fB
    @test predict(fB, nd) == p
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
