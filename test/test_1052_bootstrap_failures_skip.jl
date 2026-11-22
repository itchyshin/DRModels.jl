# #1052: a thrown bootstrap refit is recorded. The default is `failures = :skip`
# (record, keep successful draws, `bootstrap_incomplete` plus a warning), as in
# drmTMB (SHINICHI-0900-1119). Pass `failures = :error` to abort. `used == 0` still errors and names the first failure.
module Test1052BootstrapFailuresSkip

using DRModels
using Test
using Random

@testset "#1052 thrown refit recorded; default is :skip" begin
    rng = MersenneTwister(20261030)
    n = 40
    x = randn(rng, n)
    y = 0.3 .+ 0.5 .* x .+ randn(rng, n)
    form = bf(@formula(y ~ x), @formula(sigma ~ 1))
    data = (; y, x)
    fit0 = drm(form, Gaussian(); data)

    calls = Ref(0)
    function flaky(datab)
        calls[] += 1
        calls[] == 2 && error("forced refit failure")
        return drm(form, Gaussian(); data = datab)
    end

    res = @test_logs (:warn, r"bootstrap_incomplete") match_mode = :any DRModels._bootstrap_result(
        fit0, form, data, 4, 0.95, MersenneTwister(3), false, flaky; failures = :skip)
    @test res.attempted == 4
    @test res.used == 3
    @test res.failed == 1
    @test res.status == "bootstrap_incomplete"
    @test occursin("forced refit failure", res.failures[1].message)

    calls[] = 0
    default_res = @test_logs (:warn, r"bootstrap_incomplete") match_mode = :any DRModels._bootstrap_result(
        fit0, form, data, 4, 0.95, MersenneTwister(3), false, flaky)
    @test default_res.used == 3
    @test default_res.status == "bootstrap_incomplete"
    @test occursin("forced refit failure", default_res.failures[1].message)

    calls[] = 0
    default_err = try
        DRModels._bootstrap_result(
            fit0, form, data, 4, 0.95, MersenneTwister(3), false, flaky; failures = :error)
        nothing
    catch err
        err
    end
    @test default_err isa ErrorException
    @test occursin("bootstrap failed in 1 of 4 replicates", default_err.msg)
    @test occursin("forced refit failure", default_err.msg)

    calls[] = 0
    @test_throws ErrorException DRModels._bootstrap_result(
        fit0, form, data, 4, 0.95, MersenneTwister(3), false, flaky; failures = :error)

    allfail = try
        DRModels._bootstrap_result(fit0, form, data, 2, 0.95, MersenneTwister(3), false,
                                   _ -> error("always fails here"); failures = :skip)
        nothing
    catch err
        err
    end
    @test allfail isa ErrorException
    @test occursin("all 2 bootstrap replicates failed", allfail.msg)
    @test occursin("always fails here", allfail.msg)
end

# Public-API fixture. The covariate is a vector that counts element reads and
# throws on read number `trip` (0 = never). A serial bootstrap reads the data in a
# fixed order, so a call with B = k reads `T_k` times; tripping read `T_k + 1`
# makes the first read of replicate k + 1 throw, i.e. exactly one refit fails.
struct TripVec <: AbstractVector{Float64}
    parent::Vector{Float64}
    count::Base.RefValue{Int}
    trip::Int
end
Base.size(v::TripVec) = size(v.parent)
function Base.getindex(v::TripVec, i::Int)
    v.count[] += 1
    v.count[] == v.trip && error("forced refit failure")
    return v.parent[i]
end

function trip_fixture(trip::Int)
    rng = MersenneTwister(20261030)
    n = 40
    x = randn(rng, n)
    y = 0.3 .+ 0.5 .* x .+ randn(rng, n)
    cnt = Ref(0)
    return (; y, x = TripVec(x, cnt, trip)), cnt
end

const TRIP_FORM = bf(@formula(y ~ x), @formula(sigma ~ 1))

# Reads used by a clean public `bootstrap_result` call with `B` replicates.
function reads_for(B)
    data, cnt = trip_fixture(0)
    bootstrap_result(TRIP_FORM, Gaussian(); data, B, rng = MersenneTwister(3))
    return cnt[]
end

const PUBLIC_CALLS = (
    result = (; kw...) -> bootstrap_result(TRIP_FORM, Gaussian(); kw...),
    summary = (; kw...) -> bootstrap_summary(TRIP_FORM, Gaussian(); kw...),
    ci = (; kw...) -> bootstrap_ci(TRIP_FORM, Gaussian(); kw...),
)

@testset "#1052 public API: omitted keyword defaults to :skip, :error aborts" begin
    t3 = reads_for(3)
    @test reads_for(4) > t3
    for (name, f) in pairs(PUBLIC_CALLS)
        # omitted `failures`
        data, _ = trip_fixture(t3 + 1)
        out = @test_logs (:warn, r"bootstrap_incomplete") match_mode = :any f(;
            data, B = 4, rng = MersenneTwister(3))
        if name === :result
            @test out.used == 3 && out.failed == 1
            @test out.status == "bootstrap_incomplete"
            @test occursin("forced refit failure", out.failures[1].message)
        else
            @test !isempty(out)
            @test all(isfinite(r.lower) && isfinite(r.upper) for r in out)
        end
        # explicit :skip matches the default
        data, _ = trip_fixture(t3 + 1)
        out2 = @test_logs (:warn, r"bootstrap_incomplete") match_mode = :any f(;
            data, B = 4, rng = MersenneTwister(3), failures = :skip)
        name === :result && @test out2.used == 3
        # explicit :error aborts
        data, _ = trip_fixture(t3 + 1)
        err = try
            f(; data, B = 4, rng = MersenneTwister(3), failures = :error)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("bootstrap failed in 1 of 4 replicates", err.msg)
    end
end

@testset "#1052 one surviving draw gives NaN limits, not a zero-width interval" begin
    t1 = reads_for(1)
    for (name, f) in pairs(PUBLIC_CALLS)
        data, _ = trip_fixture(t1 + 1)
        out = f(; data, B = 2, rng = MersenneTwister(3))
        if name === :result
            @test out.used == 1 && out.failed == 1
            @test out.status == "bootstrap_unavailable"
            @test length(out.failures) == 1
            rows = out.summary
        else
            rows = out
        end
        @test all(isnan(r.lower) && isnan(r.upper) for r in rows)
        @test all(isfinite(r.estimate) for r in rows)
    end
end

@testset "#1052 InterruptException is not recorded as a replicate failure" begin
    rng = MersenneTwister(20261030)
    x = randn(rng, 40)
    y = 0.3 .+ 0.5 .* x .+ randn(rng, 40)
    fit0 = drm(TRIP_FORM, Gaussian(); data = (; y, x))
    @test_throws InterruptException DRModels._bootstrap_result(
        fit0, TRIP_FORM, (; y, x), 3, 0.95, MersenneTwister(3), false,
        _ -> throw(InterruptException()))
end

end
