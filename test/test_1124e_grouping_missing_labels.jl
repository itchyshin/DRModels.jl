# Gap map v2 — "grouping labels with missing values" (2 cells, drmTMB 0.8.1).
# refs #1497
#
# Cells:
#   J|gaussian|group has NA labels
#   J|poisson|group has NA labels
#
# drmTMB fits after dropping rows whose grouping factor is NA (nobs = 150 on data_v2).
# Julia refuses on raw data (#1112) and matches drmTMB via `drm_listwise`.
using DRModels
using Test
using DelimitedFiles
using StatsModels: @formula

const _E1124_DIR = joinpath(@__DIR__, "fixtures", "gap_v2_crashes")
const _E1124_FACS = Set(["g", "g2", "h", "f", "f2", "gna", "gu", "y_cat", "y_ordf"])

const _E1124_D = let (m, h) = readdlm(joinpath(_E1124_DIR, "data_v2.csv"), ',', Any; header = true)
    h = String.(vec(h))
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(h)
        c = m[:, j]
        v = if n in _E1124_FACS
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

const _E1124_R_G = let m = readdlm(joinpath(_E1124_DIR, "r_reference.csv"), ',', Any; header = true)[1]
    Dict(String(m[r, 2]) => Float64(m[r, 3]) for r in 1:size(m, 1) if String(m[r, 1]) == "na_group_gaussian")
end

# drmTMB 0.8.1 on data_v2 (gap_map_merged_tip.tsv); Poisson reference not in r_reference.csv yet.
const _E1124_POIS_LL = -252.7332178655
const _E1124_NA_DROP = 150

const _E1124_LL_TOL = 1e-4

@testset "gap map v2: grouping labels with missing values (2 cells, refs #1497)" begin
  d = _E1124_D

  @testset "J|gaussian|group has NA labels" begin
    f = bf(@formula(y_g ~ x1 + (1 | gna)), @formula(sigma ~ 1))
    err = try
      drm(f, Gaussian(); data = d)
      nothing
    catch e
      e
    end
    @test err isa ArgumentError
    @test occursin("gna", sprint(showerror, err))
    @test occursin("missing", sprint(showerror, err))
    fit = drm(f, Gaussian(); data = drm_listwise(f, d; verbose = false))
    @test nobs(fit) == _E1124_NA_DROP
    @test abs(loglikelihood(fit) - _E1124_R_G["loglik"]) <= _E1124_LL_TOL
    @test nobs(fit) == Int(_E1124_R_G["nobs"])
  end

  @testset "J|poisson|group has NA labels" begin
    f = bf(@formula(y_pois ~ x1 + (1 | gna)))
    err = try
      drm(f, Poisson(); data = d, marginal = :Laplace)
      nothing
    catch e
      e
    end
    @test err isa ArgumentError
    @test occursin("gna", sprint(showerror, err))
    fit = drm(f, Poisson(); data = drm_listwise(f, d; verbose = false), marginal = :Laplace)
    @test nobs(fit) == _E1124_NA_DROP
    @test abs(loglikelihood(fit) - _E1124_POIS_LL) <= _E1124_LL_TOL
  end
end