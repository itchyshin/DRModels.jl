using Test

# Registration guard for the test shards. It checks that, in every shard layout
# this repository has, each test file runs in exactly one shard:
#   * the CI layout (runtests.jl, DRM_TEST_SHARD="k/4"), by running the real
#     runner in list-only mode;
#   * the time-balanced layouts (shard_by_time.jl with shard_assign_N8.txt and
#     shard_assign_N16.txt), through its own shard_files().
# A new test_*.jl file needs no edit to any list: the CI layout discovers it,
# and shard_by_time.jl assigns an unlisted file by a checksum of its name.

module _ShardLayoutDriver
include(joinpath(@__DIR__, "shard_by_time.jl"))
end

# Names that appear zero times or more than once across the shard lists.
# `shards` is a vector of file-name vectors; `expected` the files that must run.
function _sl_problems(shards::AbstractVector, expected)
    counts = Dict{String,Int}()
    for s in shards, f in s
        counts[f] = get(counts, f, 0) + 1
    end
    exp = Set(expected)
    never = sort([f for f in exp if get(counts, f, 0) == 0])
    twice = sort([f for (f, n) in counts if n > 1])
    extra = sort([f for f in keys(counts) if !(f in exp)])
    return (never = never, twice = twice, extra = extra)
end
_sl_ok(p) = isempty(p.never) && isempty(p.twice) && isempty(p.extra)

function _sl_excludes(dir)
    txt = read(joinpath(dir, "runtests.jl"), String)
    m = match(r"const _TEST_EXCLUDE = \[(.*?)\n\]"s, txt)
    return String[x.captures[1] for x in eachmatch(r"\"([^\"]+\.jl)\"", m.captures[1])]
end

@testset "shard layouts: every test file runs exactly once" begin
    dir = @__DIR__
    excl = _sl_excludes(dir)
    @test !isempty(excl)

    @testset "the checker itself flags zero and double runs" begin
        @test _sl_ok(_sl_problems([["a.jl"], ["b.jl"]], ["a.jl", "b.jl"]))
        @test _sl_problems([["a.jl"], String[]], ["a.jl", "b.jl"]).never == ["b.jl"]
        @test _sl_problems([["a.jl", "b.jl"], ["b.jl"]], ["a.jl", "b.jl"]).twice == ["b.jl"]
        @test _sl_problems([["a.jl", "c.jl"]], ["a.jl"]).extra == ["c.jl"]
    end

    @testset "unlisted-file hash is pinned (same on every Julia version)" begin
        @test _ShardLayoutDriver._fnv1a32("test_a.jl") == 0x38b1fa8f
        @test _ShardLayoutDriver._fnv1a32("test_new_feature.jl") == 0x1015326f
        @test _ShardLayoutDriver._unlisted_shard("test_a.jl", 16) == 16
    end

    @testset "CI layout: runtests.jl with DRM_TEST_SHARD=k/4" begin
        expected = filter(f -> occursin(r"^test[_-].*\.jl$", f) && !(f in excl),
                          readdir(dir))
        N = 4
        shards = Vector{Vector{String}}()
        for k in 1:N
            out = IOBuffer()
            cmd = setenv(
                `$(Base.julia_cmd()) --project=$(Base.active_project()) -e
                 "include(\"runtests.jl\")"`,
                merge(ENV, Dict("DRM_TEST_LIST_ONLY" => "1",
                                "DRM_TEST_SHARD" => "$k/$N"));
                dir = dir)
            @test success(pipeline(cmd; stdout = out, stderr = devnull))
            lines = split(String(take!(out)), '\n')
            push!(shards, String[m.captures[1] for l in lines
                                 for m in (match(r"^\s*\[shard \d+/\d+\]\s+(\S+\.jl)\s*$", l),)
                                 if m !== nothing])
        end
        p = _sl_problems(shards, expected)
        @test p.never == String[]
        @test p.twice == String[]
        @test p.extra == String[]
    end

    @testset "time-balanced layouts: shard_by_time.jl, N = $n" for n in (8, 16)
        assign = joinpath(dir, "shard_assign_N$(n).txt")
        expected = filter(f -> occursin(r"^test_.*\.jl$", f) && !(f in excl),
                          readdir(dir))
        shards = [_ShardLayoutDriver.shard_files(assign, k) for k in 1:n]
        p = _sl_problems(shards, expected)
        @test p.never == String[]
        @test p.twice == String[]
        @test p.extra == String[]
    end

    @testset "scratch copy: unlisted files, stable names, stale lists" begin
        scratch = mktempdir()
        try
            cp(joinpath(dir, "shard_by_time.jl"), joinpath(scratch, "shard_by_time.jl"))
            cp(joinpath(dir, "runtests.jl"), joinpath(scratch, "runtests.jl"))
            for f in ("test_listed_a.jl", "test_listed_b.jl", "test_new_x.jl",
                      "test_new_y.jl", "helper_not_a_test.jl")
                write(joinpath(scratch, f), "# scratch\n")
            end
            assign = joinpath(scratch, "assign.txt")
            write(assign, "1 test_listed_a.jl\n2 test_listed_b.jl\n3\n")
            mod = Module(:_ShardScratchDriver)
            Base.include(mod, joinpath(scratch, "shard_by_time.jl"))
            sf(k) = Base.invokelatest(mod.shard_files, assign, k)

            # Every real file once; the listed files stay where they are listed;
            # the helper file is never picked up.
            shards = [sf(k) for k in 1:3]
            expected = ["test_listed_a.jl", "test_listed_b.jl", "test_new_x.jl", "test_new_y.jl"]
            @test _sl_ok(_sl_problems(shards, expected))
            @test "test_listed_a.jl" in shards[1]
            @test "test_listed_b.jl" in shards[2]
            @test !any(s -> "helper_not_a_test.jl" in s, shards)

            # An unlisted file's shard depends on its name only: adding more
            # unlisted files must not move the existing ones.
            where_(f) = only(k for k in 1:3 if f in sf(k))
            before = Dict(f => where_(f) for f in ("test_new_x.jl", "test_new_y.jl"))
            for f in ("test_aaa_early.jl", "test_zzz_late.jl", "test_mmm_mid.jl")
                write(joinpath(scratch, f), "# scratch\n")
            end
            @test Dict(f => where_(f) for f in keys(before)) == before

            # A listed file that does not exist fails loudly.
            write(assign, "1 test_listed_a.jl test_gone.jl\n2 test_listed_b.jl\n3\n")
            @test_throws ErrorException sf(1)
            # Listing a file twice is caught by the checker.
            write(assign, "1 test_listed_a.jl\n2 test_listed_a.jl test_listed_b.jl\n3\n")
            twice = _sl_problems([sf(k) for k in 1:3],
                                 vcat(expected, ["test_aaa_early.jl", "test_zzz_late.jl", "test_mmm_mid.jl"]))
            @test twice.twice == ["test_listed_a.jl"]
        finally
            rm(scratch; recursive = true, force = true)
        end
    end
end
