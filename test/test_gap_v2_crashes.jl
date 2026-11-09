# Gap map v2 (docs/dev-log/2026-11-09-gap-map-v2.md): Julia exceptions that were not refusals.
# Dispositions and thresholds: docs/dev-log/2026-11-09-julia-crashes.md.
# Data and R references (drmTMB 0.8.1): test/fixtures/gap_v2_crashes/.
using DRModels
using Test
using DelimitedFiles
using StatsModels: @formula

const _GV2_DIR = joinpath(@__DIR__, "fixtures", "gap_v2_crashes")
const _GV2_FACTORS = Set(["g", "g2", "h", "f", "f2", "gna", "gu", "y_cat", "y_ordf"])
const _GV2 = let (m, h) = readdlm(joinpath(_GV2_DIR, "data_v2.csv"), ',', Any; header = true)
    h = String.(vec(h))
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(h)
        c = m[:, j]
        v = if n in _GV2_FACTORS
            any(==("NA"), c) ? Union{Missing,String}[x == "NA" ? missing : String(x) for x in c] : String.(c)
        elseif any(==("NA"), c)
            Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c]
        elseif all(x -> x isa Integer, c)
            Int.(c)
        else
            Float64.(c)
        end
        push!(cols, Symbol(n) => v)
    end
    (; cols...)
end

# id => (key => value) from the R reference table (long format, quoted by write.csv).
const _GV2_REF = let m = readdlm(joinpath(_GV2_DIR, "r_reference.csv"), ',', Any; header = true)[1]
    ref = Dict{String,Dict{String,Float64}}()
    for r in 1:size(m, 1)
        get!(ref, String(m[r, 1]), Dict{String,Float64}())[String(m[r, 2])] = Float64(m[r, 3])
    end
    ref
end
# R's coefficient order, as listed in the reference file (not sorted).
function _gv2_coefs_ordered(id)
    m = readdlm(joinpath(_GV2_DIR, "r_reference.csv"), ',', Any; header = true)[1]
    return [Float64(m[r, 3]) for r in 1:size(m, 1) if String(m[r, 1]) == id && startswith(String(m[r, 2]), "coef:")]
end

# Pre-registered thresholds (docs/dev-log/2026-11-09-julia-crashes.md).
const _GV2_LL_TOL = 1e-4
const _GV2_COEF_TOL = 1e-3

function _gv2_agrees(fit, id; check_coef = true)
    @test abs(loglikelihood(fit) - _GV2_REF[id]["loglik"]) <= _GV2_LL_TOL
    @test nobs(fit) == _GV2_REF[id]["nobs"]
    if check_coef
        rc = _gv2_coefs_ordered(id)
        jc = coef(fit)
        @test length(jc) == length(rc)
        @test maximum(abs.(jc .- rc)) <= _GV2_COEF_TOL
    end
end

_gv2_msg(f) = try f(); nothing catch e; e end

const _GV2_G = Gaussian()

@testset "gap map v2 crashes" begin
    d = _GV2

    @testset "factor as a random slope is refused (7 cells)" begin
        want(v) = "drm: random-slope variable `$v` must be numeric, but it is a factor " *
                  "(element type String). drmTMB refuses this too. Use a numeric 0/1 indicator " *
                  "column for one level, e.g. `(0 + $(v)_b | g)`, or put the factor in the fixed " *
                  "effects and keep `(1 | g)`."
        for (fm, fam, resp) in (
                (@formula(y_g ~ x1 + (f | g)), Gaussian(), :y_g),
                (@formula(y_g ~ x1 + (1 + f | g)), Gaussian(), :y_g),
                (@formula(y_g ~ x1 + (0 + f | g)), Gaussian(), :y_g),
                (@formula(y_pois ~ x1 + (f | g)), Poisson(), :y_pois),
                (@formula(y_pois ~ x1 + (1 + f | g)), Poisson(), :y_pois),
                (@formula(y_bin ~ x1 + (f | g)), Binomial(), :y_bin),
                (@formula(y_bin ~ x1 + (1 + f | g)), Binomial(), :y_bin))
            err = _gv2_msg(() -> drm(bf(fm), fam; data = d))
            @test err isa ArgumentError
            @test err isa ArgumentError && err.msg == want("f")
        end
        # a numeric slope on the same data still fits
        @test isfinite(loglikelihood(drm(bf(@formula(y_g ~ x1 + (1 + x1 | g)), @formula(sigma ~ 1)), _GV2_G; data = d)))
    end

    @testset "random effect in the zi formula is refused" begin
        err = _gv2_msg(() -> drm(bf(@formula(y_zip ~ x1), @formula(zi ~ 1 + (1 | g))), Poisson(); data = d))
        @test err isa ArgumentError
        @test err isa ArgumentError && err.msg ==
            "drm: random effects in the `zi` formula are not implemented (drmTMB refuses them too). " *
            "Put the random effect on `mu`, e.g. `y ~ x + (1 | g)`, or remove it from the `zi` formula."
    end

    @testset "cross-family pair with an unsupported family is refused" begin
        fm = bf(mu1 = @formula(y_ln ~ x1), mu2 = @formula(y_gam ~ x1),
                sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1))
        err = _gv2_msg(() -> drm(fm, (LogNormal(), Gamma()); data = d))
        @test err isa ArgumentError
        @test err isa ArgumentError && err.msg ==
            "drm: the cross-family route supports Gaussian, Poisson, Binomial, NegBinomial2, Beta " *
            "and Gamma; LogNormal() is not implemented here (drmTMB refuses it too). Fit the two " *
            "responses separately, or choose one of those families."
        # the supported pairs still fit
        mf = drm(bf(mu1 = @formula(y_g ~ x1), mu2 = @formula(y_pois ~ x1), sigma1 = @formula(sigma1 ~ 1)),
                 (Gaussian(), Poisson()); data = d)
        @test isfinite(mf.loglik)
    end

    @testset "Gaussian REML, y ~ f * x1, sigma ~ x1 + f: fits and agrees with drmTMB" begin
        fit = drm(bf(@formula(y_g ~ f * x1), @formula(sigma ~ x1 + f)), _GV2_G; data = d, method = :REML)
        _gv2_agrees(fit, "reml_fx1_sigma_x1f")
    end

    @testset "ordinal response given as labels: fits and agrees with drmTMB" begin
        fit = @test_logs (:info, r"coded 1\.\.4 in sorted label order \(o1 < o2 < o3 < o4\)") drm(
            bf(@formula(y_ordf ~ x1)), CumulativeLogit(); data = d)
        _gv2_agrees(fit, "cumul_label"; check_coef = false)
        @test abs(coef(fit)[1] - _GV2_REF["cumul_label"]["coef:mu.x1"]) <= _GV2_COEF_TOL
        # identical to the fit on the integer codes
        d2 = merge(d, (; y_ordf = parse.(Int, replace.(d.y_ordf, "o" => ""))))
        fit2 = drm(bf(@formula(y_ordf ~ x1)), CumulativeLogit(); data = d2)
        @test loglikelihood(fit2) == loglikelihood(fit)
        # an element type that is neither numeric nor a label is refused
        d3 = merge(d, (; y_ordf = Any[(1, 2) for _ in 1:length(d.x1)]))
        err = _gv2_msg(() -> drm(bf(@formula(y_ordf ~ x1)), CumulativeLogit(); data = d3))
        @test err isa ArgumentError
        @test err isa ArgumentError && startswith(err.msg,
            "CumulativeLogit(): the response `y_ordf` must be integer categories coded 1, 2, …, K, or string/symbol labels")
    end

    @testset "weights with a missing response: weights follow their rows" begin
        fit = @test_logs (:warn, r"16 of 160 rows have a missing/NaN response") drm(
            bf(@formula(y_gm ~ x1), @formula(sigma ~ 1)), _GV2_G; data = d, weights = d.w)
        _gv2_agrees(fit, "weights_na_response"; check_coef = false)
        # same fit as dropping the rows and their weights by hand
        keep = .!ismissing.(d.y_gm)
        dk = (; y_gm = Float64.(d.y_gm[keep]), x1 = d.x1[keep])
        fit2 = drm(bf(@formula(y_gm ~ x1), @formula(sigma ~ 1)), _GV2_G; data = dk, weights = d.w[keep])
        @test coef(fit2) ≈ coef(fit) rtol = 1e-8
        @test coef(fit)[1:2] ≈ _gv2_coefs_ordered("weights_na_response")[1:2] atol = _GV2_COEF_TOL
        # a vector of the wrong length is still refused
        @test_throws DimensionMismatch drm(bf(@formula(y_gm ~ x1), @formula(sigma ~ 1)), _GV2_G;
                                           data = d, weights = d.w[1:100])
    end

    @testset "workaround spellings agree with the R forms Julia cannot parse" begin
        # R `(1 + x1 || g)`: `@formula` itself cannot parse `||` (the error is raised at macro
        # expansion, before `drm` runs). The expansion `(1 | g) + (0 + x1 | g)` is what R fits.
        fit = drm(bf(@formula(y_g ~ x1 + (1 | g) + (0 + x1 | g)), @formula(sigma ~ 1)), _GV2_G; data = d)
        _gv2_agrees(fit, "dbar_gaussian"; check_coef = false)
        @test abs(loglikelihood(fit) - _GV2_REF["indep_gaussian"]["loglik"]) <= _GV2_LL_TOL
        # R `poly(x1, 2)` (orthogonal basis): `@formula` has no `poly`. The raw quadratic spans the
        # same column space, so the ML log-likelihood is the same.
        sqx(x) = x .^ 2
        fit = drm(bf(@formula(y_g ~ x1 + sqx(x1)), @formula(sigma ~ 1)), _GV2_G; data = d)
        _gv2_agrees(fit, "poly2_gaussian"; check_coef = false)
        # rows with a missing grouping label: refused (#1112); the listwise-cleaned data match drmTMB,
        # which drops those rows.
        f = bf(@formula(y_g ~ x1 + (1 | gna)), @formula(sigma ~ 1))
        @test_throws ArgumentError drm(f, _GV2_G; data = d)
        fit = drm(f, _GV2_G; data = drm_listwise(f, d; verbose = false))
        _gv2_agrees(fit, "na_group_gaussian"; check_coef = false)
    end
end
