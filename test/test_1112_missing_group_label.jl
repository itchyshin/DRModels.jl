# A `missing` / NaN in a random-effect grouping column must not be silently
# pooled into one extra level (#1112, gap-map #1497). drmTMB drops those rows;
# DRModels.jl refuses with an ArgumentError naming the grouping column (the
# guided complete-case route is `drm_listwise`).

using DRModels
using Test
using Random
using DelimitedFiles
using StatsModels: @formula

const _GNA_MSG_SUBSTR = (
    "grouping column `gna`",
    "missing",
    "drm_listwise",
)

function _data_1112(; n = 120, G = 12, seed = 1112)
    rng = MersenneTwister(seed)
    g = repeat(["g" * lpad(k, 2, '0') for k in 1:G]; inner = n ÷ G)
    x = randn(rng, n)
    b = 0.7 .* randn(rng, G)
    y = 0.5 .* x .+ b[repeat(1:G; inner = n ÷ G)] .+ 0.5 .* randn(rng, n)
    return (; y, x, g)
end

@testset "missing grouping label is refused, not pooled (#1112)" begin
    d = _data_1112()
    gm = Vector{Union{Missing,String}}(d.g); gm[1:8:80] .= missing
    f = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))

    # Complete data still fits.
    @test nobs(drm(f, Gaussian(); data = d)) == 120

    # missing label -> loud ArgumentError naming the column
    err = try drm(f, Gaussian(); data = (; d.y, d.x, g = gm)); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("g", sprint(showerror, err)) && occursin("missing", sprint(showerror, err))

    # NaN in a numeric code column
    gn = Float64.(repeat(1:12; inner = 10)); gn[3] = NaN
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian();
                                   data = (; d.y, d.x, g = gn))

    # Other routes share the choke point
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g))), Gaussian(); data = (; d.y, d.x, g = gm))
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g))), Gamma();
                                   data = (; y = exp.(d.y), d.x, g = gm), marginal = :Laplace)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g))), Poisson();
                                   data = (; y = round.(Int, exp.(d.y)), d.x, g = gm), marginal = :Laplace)

    # Documented workaround works and matches the dropped-rows fit
    dm = (; d.y, d.x, g = gm)
    clean = drm_listwise(f, dm; verbose = false)
    fit = drm(f, Gaussian(); data = clean)
    @test nobs(fit) == 110
    @test length(ranef(fit)[:g]) == 12
end

@testset "gap-map v2: gna with missing grouping labels (#1497)" begin
    dir = joinpath(@__DIR__, "fixtures", "gap_v2_crashes")
    factors = Set(["g", "g2", "h", "f", "f2", "gna", "gu", "y_cat", "y_ordf"])
    m, h = readdlm(joinpath(dir, "data_v2.csv"), ',', Any; header = true)
    h = String.(vec(h))
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(h)
        c = m[:, j]
        v = if n in factors
            any(==("NA"), c) ? Union{Missing,String}[x == "NA" ? missing : String(x) for x in c] :
                String.(c)
        elseif any(==("NA"), c)
            Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c]
        else
            Float64.(c)
        end
        push!(cols, Symbol(n) => v)
    end
    d = (; cols...)
    for (label, f, fam) in (
            ("J|gaussian|group has NA labels",
             bf(@formula(y_g ~ x1 + (1 | gna)), @formula(sigma ~ 1)), Gaussian()),
            ("J|poisson|group has NA labels",
             bf(@formula(y_pois ~ x1 + (1 | gna))), Poisson()))
        @testset "$label" begin
            err = try drm(f, fam; data = d); nothing catch e; e end
            @test err isa ArgumentError
            msg = sprint(showerror, err)
            @test all(s -> occursin(s, msg), _GNA_MSG_SUBSTR)
            fit = drm(f, fam; data = drm_listwise(f, d; verbose = false))
            @test nobs(fit) == 150
        end
    end
end
