# Observed-row mask of a response column (Julia 1.12 miscompile, j112rows).
#
# On Julia 1.12.6 and 1.12.7, seen on an AMD EPYC 9655 (Zen 5; other CPUs
# untested), the earlier `_coerce_response_column` loop returned about half of
# the rows as "not observed" for a `Vector{Union{Missing,Int}}` response of
# about 31 or more rows (also Int32 and UInt64, with or without a missing
# value), so a fit silently used too few rows. These tests pin the mask and
# the coerced values against a plain independent computation, for several
# element types and lengths, and pin the fit-level symptom (`nobs`).
# Nothing here is skipped on any Julia version.
using DRModels
using Test, Random, Logging

with_logger_quiet(f) = with_logger(f, NullLogger())

# Independent expectation: no loop of ours, no `@inbounds`.
_expect_observed(raw) = [!(ismissing(v) || (v isa AbstractFloat && isnan(v))) for v in raw]

function _mixed_column(T, n; miss = Int[], seed = 1)
    rng = MersenneTwister(seed)
    col = Vector{Union{Missing,T}}(undef, n)
    for i in 1:n
        col[i] = T <: Bool ? (rand(rng) < 0.5) : T(rand(rng, 0:3))
    end
    col[filter(<=(n), miss)] .= missing
    return col
end

@testset "response mask: Union{Missing,T} columns (j112rows)" begin
    for T in (Int, Int32, Int64, UInt64, UInt8, Bool, Float64, Float32)
        for n in (1, 7, 8, 9, 16, 29, 30, 31, 32, 33, 34, 35, 64, 160, 1000, 10_001)
            for miss in (Int[], [3], [3, 20, 77], collect(1:5:n), collect(1:n))
                raw = _mixed_column(T, n; miss = miss, seed = n)
                y, observed = DRModels._coerce_response_column(raw)
                want = _expect_observed(raw)
                @test observed == want
                @test length(y) == n
                @test all(isnan, y[.!want])
                @test y[want] == Float64.(raw[want])
            end
        end
    end
end

@testset "response mask: Union{Missing,T} columns that hold no missing value (j112rows)" begin
    # The fault also hit a column whose element type allows `missing` but which
    # holds none, so complete data in such a column is not safe by itself.
    for T in (Int, Int32, UInt64, UInt8, Float64)
        for n in (29, 30, 31, 32, 33, 34, 35, 64, 120, 200)
            col = Vector{Union{Missing,T}}(T.(rand(MersenneTwister(n), 0:3, n)))
            @test !any(ismissing, col)
            y, observed = DRModels._coerce_response_column(col)
            @test all(observed)
            @test y == Float64.(col)
        end
    end
    # Fit level: nobs must be the full row count (main gave 16 on the fault).
    for T in (Int, UInt64), n in (31, 120, 200)
        rng = MersenneTwister(n)
        x = randn(rng, n)
        counts = [rand(rng, 0:1) == 0 ? 0 : rand(rng, 1:4) for _ in 1:n]
        col = Vector{Union{Missing,T}}(T.(counts))
        f = with_logger_quiet() do
            drm(bf(@formula(y ~ x)), Poisson(); data = (; y = col, x = x))
        end
        @test nobs(f) == n
    end
end

@testset "response mask: NaN as missing code, plain columns (j112rows)" begin
    for n in (8, 32, 160, 1000)
        yf = Float64.(0:(n - 1))
        yf[[min(3, n), min(20, n)]] .= NaN
        y, observed = DRModels._coerce_response_column(yf)
        @test observed == .!isnan.(yf)
        yi = collect(0:(n - 1))
        y, observed = DRModels._coerce_response_column(yi)
        @test all(observed) && y == Float64.(yi)
        yu = Vector{Union{Missing,Float64}}(yf)
        yu[[min(5, n), min(7, n)]] .= missing
        y, observed = DRModels._coerce_response_column(yu)
        @test observed == _expect_observed(yu)
    end
end

@testset "response mask: guard refuses a wrong mask (j112rows)" begin
    raw = _mixed_column(Int, 160; miss = [3, 20, 77])
    y, observed = DRModels._coerce_response_column(raw)
    @test DRModels._check_response_coercion(raw, y, observed, "y") === nothing
    bad = copy(observed); bad[100] = !bad[100]
    @test_throws ErrorException DRModels._check_response_coercion(raw, y, bad, "y")
    bad = copy(observed); bad[3] = true
    @test_throws ErrorException DRModels._check_response_coercion(raw, y, bad, "y")
    ybad = copy(y); ybad[50] += 1.0
    @test_throws ErrorException DRModels._check_response_coercion(raw, ybad, observed, "y")
    ybad = copy(y); ybad[3] = 0.0
    @test_throws ErrorException DRModels._check_response_coercion(raw, ybad, observed, "y")
    @test_throws ErrorException DRModels._check_response_coercion(raw, y[1:end-1], observed, "y")

    # The message says what happened and what the user can do.
    bad = copy(observed); bad[100] = !bad[100]
    err = try
        DRModels._check_response_coercion(raw, y, bad, "y"); nothing
    catch e
        e
    end
    @test err isa ErrorException
    msg = err.msg
    @test occursin("row 100", msg)
    @test occursin("Your data are fine", msg)
    @test occursin("Julia 1.10", msg) && occursin("1.13", msg)
    @test occursin("-C generic", msg)
    @test occursin("Float64", msg)
end

@testset "response mask: fit-level symptom, Poisson with missing counts (j112rows)" begin
    rng = MersenneTwister(20261112)
    n = 160
    x = randn(rng, n)
    counts = [rand(rng, 0:1) == 0 ? 0 : rand(rng, 1:4) for _ in 1:n]   # about half zeros
    miss = [3, 20, 77]
    keep = trues(n); keep[miss] .= false
    ycol = Vector{Union{Missing,Int}}(counts); ycol[miss] .= missing
    f = with_logger_quiet() do
        drm(bf(@formula(y ~ x)), Poisson(); data = (; y = ycol, x = x))
    end
    @test nobs(f) == n - length(miss)
    ref = drm(bf(@formula(y ~ x)), Poisson(); data = (; y = counts[keep], x = x[keep]))
    @test nobs(ref) == count(keep)
    @test coef(f) ≈ coef(ref) atol = 1e-9 rtol = 1e-9
    @test loglik(f) ≈ loglik(ref) atol = 1e-9 rtol = 1e-9
    @test length(fitted(f)) == n

    # complete-case helper keeps the right rows
    d = with_logger_quiet() do
        drm_listwise(bf(@formula(y ~ x)), (; y = ycol, x = x))
    end
    @test length(d.y) == n - length(miss)
    @test d.y == counts[keep]
end
