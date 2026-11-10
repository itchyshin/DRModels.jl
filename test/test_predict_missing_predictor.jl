# predict() on fits whose data have a missing predictor, and on an ordinal label response.
#
# Gap map v2 (g6e): `predict(fit, d)` with a missing predictor threw in Julia (5 K cells), and
# `predict` threw `Float64(::String)` on the ordinal label-response cell.
#
# What drmTMB 0.8.1 does (fixtures/predict_miss/reference.csv, refusals.csv; regenerate with
# `Rscript test/fixtures/predict_miss/gen_fixtures.R`):
#   * `predict(fit, newdata = <rows with a missing value in a predictor the mean formula uses>)`
#     is an error ("New prediction data contains missing value for required predictor"). It does
#     NOT return NA in those rows, so the Julia refusal IS the parity behaviour and is pinned below.
#   * The gap-map harness called R's `predict(fit)` (no newdata: the fitted complete rows) against
#     Julia's `predict(fit, d_full)`; those are different calls. The like-for-like call is
#     `predict(fit, newdata = complete rows)`, pinned here to R at 1e-8 on every row.
#   * A predictor missing only in a parameter the prediction does not use (here `sigma ~ x1m`) is
#     not an error in R; the mu prediction has no NA. Same in Julia.
#   * Ordinal label response: the label column of `newdata` is ignored by `predict`, as in R.
# Fits are checked first (|logLik diff| <= 1e-4, mu coefficients within 1e-3 R SE); a prediction is
# pinned only where the fits agree. The Poisson `(1 | g)` cell does not meet the logLik bound (gap
# 4.6e-3, Laplace vs R), so its predictions are not pinned; its refusal is.
using DRModels, Test, DelimitedFiles

const _PM_DIR = joinpath(@__DIR__, "fixtures", "predict_miss")

function _pm_read(file)
    m, h = readdlm(joinpath(_PM_DIR, file), ',', Any; header = true)
    return m, String.(vec(h))
end

const _PM_D = let (m, h) = _pm_read("data.csv")
    col(n) = m[:, findfirst(==(n), h)]
    num(n) = (c = col(n); any(==("NA"), c) ? Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c] :
                                             Float64.(c))
    (; id = Int.(col("id")), g = String.(col("g")), x1 = num("x1"), x1m = num("x1m"), y_g = num("y_g"),
       y_gm = num("y_gm"), y_pois = num("y_pois"), y_pm = num("y_pm"), y_nb = num("y_nb"),
       y_ordf = String.(col("y_ordf")))
end

const _PM_REF = let (m, h) = _pm_read("reference.csv")
    c(n) = m[:, findfirst(==(n), h)]
    out = Dict{Tuple{String,String},Tuple{Vector{Int},Vector{Float64}}}()
    for (cell, call) in unique(zip(String.(c("cell")), String.(c("call"))))
        keep = [i for i in eachindex(m[:, 1]) if c("cell")[i] == cell && c("call")[i] == call]
        out[(cell, call)] = (Int.(c("id")[keep]), Float64.(c("value")[keep]))
    end
    out
end

const _PM_FITS = let (m, h) = _pm_read("fits.csv")
    c(n) = m[:, findfirst(==(n), h)]
    out = Dict{String,NamedTuple}()
    for cell in unique(String.(c("cell")))
        k = findall(==(cell), String.(c("cell")))
        out[cell] = (; ll = Float64(c("logLik")[k[1]]), est = Float64.(c("estimate")[k]), se = Float64.(c("se")[k]))
    end
    out
end

const _PM_REFUSAL = let (m, h) = _pm_read("refusals.csv")
    Dict(String(m[i, 1]) => String(m[i, 2]) for i in axes(m, 1))
end

_pm_rows(idx) = NamedTuple{keys(_PM_D)}(map(v -> v[idx], values(_PM_D)))
const _PM_OK = findall(!ismissing, _PM_D.x1m)                       # rows with a complete predictor
_pm_newdata(; cols = (:id, :g, :x1, :x1m)) =
    NamedTuple{cols}(map(k -> collect(skipmissing(getproperty(_PM_D, k)[_PM_OK])), cols))

function _pm_check_fit(fit, cell)
    r = _PM_FITS[cell]
    @test abs(loglikelihood(fit) - r.ll) <= 1e-4
    β = coef(fit, :mu)
    @test length(β) == length(r.est)
    @test all(abs.(β .- r.est) .<= 1e-3 .* r.se)
end

function _pm_check_pred(p, cell, call)
    ids, ref = _PM_REF[(cell, call)]
    @test length(p) == length(ref)
    @test all(isfinite, p)
    @test maximum(abs.(p .- ref)) <= 1e-8
end

const _PM_CELLS = [
    ("K_gauss_mu", () -> drm(bf(@formula(y_g ~ x1m), @formula(sigma ~ 1)), DRModels.Gaussian(); data = _PM_D), :identity),
    ("K_pois", () -> drm(bf(@formula(y_pois ~ x1m)), DRModels.Poisson(); data = _PM_D), :log),
    ("K_nb2", () -> drm(bf(@formula(y_nb ~ x1m), @formula(sigma ~ 1)), DRModels.NegBinomial2(); data = _PM_D), :log),
]

@testset "predict with a missing predictor: like-for-like with drmTMB 0.8.1" begin
    for (cell, mk, _) in _PM_CELLS
        @testset "$cell" begin
            fit = mk()
            _pm_check_fit(fit, cell)
            nd = _pm_newdata()
            _pm_check_pred(predict(fit, nd), cell, "newdata_response")
            _pm_check_pred(predict(fit, nd; type = :link), cell, "newdata_link")
            # A missing predictor in newdata is refused, as in R (R: no NA row is returned).
            @test occursin("missing", _PM_REFUSAL[cell])
            err = try predict(fit, _PM_D); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("x1m", sprint(showerror, err))
            @test occursin("missing", sprint(showerror, err))
        end
    end
    # Gaussian random-intercept cell: same logLik and fixed effects as R; population-level newdata
    # predictions pinned. (R's `predict(fit)` with no newdata adds the fitted random effects; Julia
    # has no no-newdata `predict`, so that call is not compared.)
    @testset "K_gauss_both_re" begin
        fit = drm(bf(@formula(y_gm ~ x1m + (1 | g)), @formula(sigma ~ 1)), DRModels.Gaussian(); data = _PM_D)
        _pm_check_fit(fit, "K_gauss_both_re")
        nd = _pm_newdata()
        _pm_check_pred(predict(fit, nd), "K_gauss_both_re", "newdata_response")
        @test_throws ArgumentError predict(fit, _PM_D)
    end
    # Poisson random-intercept cell: R and Julia logLik differ by about 4.6e-3 (Laplace vs R), so
    # predictions are not pinned; only the shared refusal is.
    @testset "K_pois_both_re (refusal only)" begin
        fit = drm(bf(@formula(y_pm ~ x1m + (1 | g))), DRModels.Poisson(); data = _PM_D)
        @test_throws ArgumentError predict(fit, _PM_D)
        @test occursin("missing", _PM_REFUSAL["K_pois_both_re"])
    end
    # A predictor that only the sigma formula uses: R does not refuse and returns no NA for mu.
    @testset "K_gauss_sigma (NA only in the sigma formula)" begin
        fit = drm(bf(@formula(y_g ~ x1), @formula(sigma ~ x1m)), DRModels.Gaussian(); data = _PM_D)
        _pm_check_fit(fit, "K_gauss_sigma")
        @test _PM_REFUSAL["K_gauss_sigma"] == "NO-ERROR"
        p = predict(fit, _PM_D)
        ids, ref = _PM_REF[("K_gauss_sigma", "newdata_full_with_NA")]
        @test length(p) == length(ref) == length(_PM_D.id)
        @test !any(ismissing, p)
        @test maximum(abs.(p .- ref)) <= 1e-8
    end
end

@testset "predict on an ordinal label response (gap-map cell I|cumul|response=label)" begin
    fit = drm(bf(@formula(y_ordf ~ x1)), DRModels.CumulativeLogit();
              data = (; y_ordf = _PM_D.y_ordf, x1 = _PM_D.x1))
    _pm_check_fit(fit, "I_ord_label")
    d_all = (; y_ordf = _PM_D.y_ordf, x1 = _PM_D.x1)
    p = predict(fit, d_all)                                  # label column present in newdata
    _pm_check_pred(p, "I_ord_label", "newdata_all_with_label")
    _pm_check_pred(p, "I_ord_label", "fit_rows")             # R's predict(fit) is the same 160 values
    idx = [5, 1, 9]
    _pm_check_pred(predict(fit, (; y_ordf = _PM_D.y_ordf[idx], x1 = _PM_D.x1[idx])), "I_ord_label", "newdata_with_label")
    # predict_parameters ignores a label response the same way, and agrees with a numeric response column
    pp = predict_parameters(fit, d_all)
    codes_all = [parse(Int, s[2:end]) for s in _PM_D.y_ordf]
    pp_num = predict_parameters(fit, (; y_ordf = codes_all, x1 = _PM_D.x1))
    @test keys(pp) == keys(pp_num)
    for k in keys(pp)
        @test all(isfinite, pp[k])
        @test maximum(abs.(pp[k] .- pp_num[k])) <= 1e-12
    end
    @test maximum(abs.(pp[:mu] .- p)) <= 1e-12
    # newdata without the response column keeps working (unchanged behaviour)
    p0 = predict(fit, (; x1 = _PM_D.x1))
    @test maximum(abs.(p0 .- p)) <= 1e-12
    # an integer-coded response in newdata gives the same numbers
    codes = [parse(Int, s[2:end]) for s in _PM_D.y_ordf]
    @test maximum(abs.(predict(fit, (; y_ordf = codes, x1 = _PM_D.x1)) .- p)) <= 1e-12
end
