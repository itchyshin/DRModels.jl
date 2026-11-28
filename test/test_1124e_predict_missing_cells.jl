# Gap map v2 — "predict with a missing predictor" (5 K cells, drmTMB 0.8.1).
# refs #1498
#
# Cell IDs from gap_map_merged_tip.tsv:
#   K|gaussian|NA in mu predictor       -> fixture id K_gauss_mu
#   K|gaussian|NA in both + (1|g)       -> K_gauss_both_re
#   K|poisson|NA in predictor           -> K_pois
#   K|poisson|NA in both + (1|g)        -> K_pois_both_re
#   K|nb2|NA in predictor               -> K_nb2
#
# Like-for-like behaviour (refuse missing mean predictors; pin complete-case newdata)
# is covered in test_predict_missing_predictor.jl; this file names the five cells for
# the lane ledger and re-checks refusal + one pinned prediction per family.
using DRModels
using Test
using DelimitedFiles
using StatsModels: @formula

const _K1124_DIR = joinpath(@__DIR__, "fixtures", "predict_miss")

function _k1124_read(file)
    m, h = readdlm(joinpath(_K1124_DIR, file), ',', Any; header = true)
    return m, String.(vec(h))
end

const _K1124_D = let (m, h) = _k1124_read("data.csv")
    col(n) = m[:, findfirst(==(n), h)]
    function num(n)
        c = col(n)
        return any(==("NA"), c) ? Union{Missing, Float64}[x == "NA" ? missing : Float64(x) for x in c] :
                                  Float64.(c)
    end
    (; id = Int.(col("id")), g = String.(col("g")), x1 = num("x1"), x1m = num("x1m"), y_g = num("y_g"),
       y_gm = num("y_gm"), y_pois = num("y_pois"), y_nb = num("y_nb"), y_pm = num("y_pm"))
end

const _K1124_REF = let (m, h) = _k1124_read("reference.csv")
    c(n) = m[:, findfirst(==(n), h)]
    out = Dict{String, Vector{Float64}}()
    for i in eachindex(m[:, 1])
        cell, call = String(c("cell")[i]), String(c("call")[i])
        call == "newdata_response" &&
            cell in ("K_gauss_mu", "K_pois", "K_nb2") || continue
        push!(get!(out, cell, Float64[]), Float64(c("value")[i]))
    end
    out
end

const _K1124_OK = findall(!ismissing, _K1124_D.x1m)
_k1124_nd(; extra = ()) = merge((
        id = _K1124_D.id[_K1124_OK], g = _K1124_D.g[_K1124_OK], x1 = _K1124_D.x1[_K1124_OK],
        x1m = _K1124_D.x1m[_K1124_OK],
    ), extra)

const _K1124_CELLS = [
    ("K|gaussian|NA in mu predictor", "K_gauss_mu",
     () -> drm(bf(@formula(y_g ~ x1m), @formula(sigma ~ 1)), Gaussian(); data = _K1124_D)),
    ("K|poisson|NA in predictor", "K_pois",
     () -> drm(bf(@formula(y_pois ~ x1m)), Poisson(); data = _K1124_D)),
    ("K|nb2|NA in predictor", "K_nb2",
     () -> drm(bf(@formula(y_nb ~ x1m), @formula(sigma ~ 1)), NegBinomial2(); data = _K1124_D)),
]

@testset "gap map v2: predict with a missing predictor (5 cells, refs #1498)" begin
    nd = _k1124_nd()
    for (gap_id, cell, mk) in _K1124_CELLS
        @testset "$gap_id" begin
            fit = mk()
            p = predict(fit, nd)
            ref = _K1124_REF[cell]
            @test length(p) == length(ref)
            @test maximum(abs.(p .- ref)) <= 1e-8
            err = try
                predict(fit, _K1124_D)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("x1m", sprint(showerror, err))
        end
    end

    @testset "K|gaussian|NA in both + (1|g)" begin
        fit_g = drm(bf(@formula(y_gm ~ x1m + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = _K1124_D)
        @test_throws ArgumentError predict(fit_g, _K1124_D)
        @test all(isfinite, predict(fit_g, nd))
    end

    @testset "K|poisson|NA in both + (1|g) (refusal only)" begin
        fit = drm(bf(@formula(y_pm ~ x1m + (1 | g))), Poisson(); data = _K1124_D)
        @test_throws ArgumentError predict(fit, _K1124_D)
    end
end