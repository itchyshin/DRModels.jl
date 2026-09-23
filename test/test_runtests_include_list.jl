using Test

# Guards the shape of test/runtests.jl itself. Its MAINTENANCE NOTE explains why:
# every top-level include line changes shape on a merge, git keeps BOTH sides of
# the conflict, and whole blocks of includes are silently duplicated (52 of them
# on 2026-09-03; nine more, plus a plain `include` block that ran in every CI
# shard, on 2026-09-19). A duplicated file costs its full runtime twice; a plain
# `include` of a test file bypasses the shard split and the per-file BLAS guard.
# This runs in every shard and reads the file as text, so it costs nothing.

const _RUNTESTS_PATH = joinpath(@__DIR__, "runtests.jl")

# Test-file includes at column 0 only; anything indented is inside a helper or
# a conditional and is that block's business.
function _runtests_include_lines(src::AbstractString)
    out = Tuple{Int,String,String}[]  # (line number, form, path)
    for (i, line) in enumerate(split(src, '\n'))
        m = match(r"^(_shard_include|include)\(\"([^\"]+)\"\)", line)
        m === nothing && continue
        push!(out, (i, String(m.captures[1]), String(m.captures[2])))
    end
    return out
end

@testset "runtests.jl include list" begin
    src = read(_RUNTESTS_PATH, String)
    lines = _runtests_include_lines(src)
    @test !isempty(lines)

    @testset "no test file is included twice" begin
        seen = Dict{String,Int}()
        duplicates = String[]
        for (i, _, path) in lines
            haskey(seen, path) && push!(duplicates, "$path (lines $(seen[path]) and $i)")
            seen[path] = i
        end
        @test isempty(duplicates)
    end

    @testset "test files go through _shard_include" begin
        # shard_util.jl defines _shard_include, so it is the one plain include.
        plain = ["$path (line $i)" for (i, form, path) in lines
                 if form == "include" && path != "shard_util.jl"]
        @test isempty(plain)
    end

    @testset "every included file exists" begin
        for (_, _, path) in lines
            @test isfile(joinpath(@__DIR__, path))
        end
    end
end
