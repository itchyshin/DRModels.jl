using Test

# Guards the shape of test/runtests.jl itself. Its CONFLICT-FREE REGISTRATION
# note explains why: test_*.jl files are now auto-discovered via readdir(),
# not hand-listed, specifically so a new test file needs no edit here and two
# PRs adding a file at the same time no longer conflict. What remains
# hand-maintained is two short arrays, `_TEST_ORDER` and `_TEST_EXCLUDE` --
# still small enough to conflict on a merge, so this guards THEIR shape: no
# duplicate entries, no overlap between the two, and every named file exists.
# It also keeps the older guard against a plain `include(...)` of a test file
# (bypasses the shard split and the per-file BLAS guard) and against a
# duplicated top-level `_shard_include(...)` literal (the two remaining ones,
# for shard_util.jl and the guard-file duplication history described below).
#
# History: before auto-discovery, every top-level include line changed shape
# on a merge and git resolved a conflict by keeping BOTH sides -- 52 duplicated
# includes on 2026-09-03, nine more plus a plain `include` block that ran in
# every CI shard on 2026-09-19. This runs in every shard and reads the file as
# text, so it costs nothing.

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

# Pulls the quoted string literals out of `const _NAME = [ "a", "b", ... ]`,
# tolerant of comments and multi-line layout. "s" = dotall, so `.` crosses
# newlines inside the array literal.
function _parse_string_array_const(src::AbstractString, name::AbstractString)
    m = match(Regex("const\\s+" * name * "\\s*=\\s*\\[(.*?)\\]", "s"), src)
    m === nothing && error("could not find `const $name = [...]` in runtests.jl")
    body = m.captures[1]
    return String[String(x.captures[1]) for x in eachmatch(r"\"([^\"]+)\"", body)]
end

@testset "runtests.jl include list" begin
    src = read(_RUNTESTS_PATH, String)
    lines = _runtests_include_lines(src)
    @test !isempty(lines)

    @testset "no literal include is duplicated" begin
        seen = Dict{String,Int}()
        duplicates = String[]
        for (i, _, path) in lines
            haskey(seen, path) && push!(duplicates, "$path (lines $(seen[path]) and $i)")
            seen[path] = i
        end
        @test isempty(duplicates)
    end

    @testset "test files go through _shard_include, not a plain include" begin
        # shard_util.jl defines _shard_include, so it is the one plain include.
        plain = ["$path (line $i)" for (i, form, path) in lines
                 if form == "include" && path != "shard_util.jl"]
        @test isempty(plain)
    end

    @testset "every literal-included file exists" begin
        for (_, _, path) in lines
            @test isfile(joinpath(@__DIR__, path))
        end
    end

    order = _parse_string_array_const(src, "_TEST_ORDER")
    exclude = _parse_string_array_const(src, "_TEST_EXCLUDE")

    @testset "_TEST_ORDER has no duplicates and every file exists" begin
        @test length(order) == length(unique(order))
        for f in order
            @test isfile(joinpath(@__DIR__, f))
        end
    end

    @testset "_TEST_EXCLUDE has no duplicates and every file exists" begin
        @test length(exclude) == length(unique(exclude))
        for f in exclude
            @test isfile(joinpath(@__DIR__, f))
        end
    end

    @testset "_TEST_ORDER and _TEST_EXCLUDE do not overlap" begin
        @test isempty(intersect(order, exclude))
    end

    @testset "auto-discovery still finds files beyond ORDER/EXCLUDE" begin
        discovered = filter(f -> occursin(r"^test_.*\.jl$", f) &&
                                  !(f in order) && !(f in exclude),
                             readdir(@__DIR__))
        @test !isempty(discovered)
    end
end
