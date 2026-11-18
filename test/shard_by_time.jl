# shard_by_time.jl -- time-balanced shard driver for the DRModels test suite.
#
# This is a STANDALONE driver; it does not replace or alter runtests.jl (whose
# `DRM_TEST_SHARD="k/N"` modulo split stays the CI default). Use it to run one
# shard of a precomputed, time-balanced assignment, for example on a cluster:
#
#   julia --project=test test/shard_by_time.jl ASSIGN_FILE K [options]
#
#   ASSIGN_FILE  one line per shard: `k file1.jl file2.jl ...` (whitespace
#                separated; blank lines and `#` comments ignored). The number of
#                lines is the shard count N.
#   K            this shard, 1..N.
#
# Options
#   --slow-list FILE   file of slow-tier test names (one per line, `#` comments).
#                      Default: test/slow_tier.txt when it exists.
#   --skip-slow        drop slow-tier files from this shard (fast tier).
#   --only-slow        keep only slow-tier files from this shard (slow tier).
#   --list             print the shard's file list and exit (no loading/running).
#
# Files in test/ that match test_*.jl but are absent from ASSIGN_FILE (new files
# added after the assignment was computed) are NOT dropped: each goes to shard
# `fnv1a32(file name) % N + 1`, a rule that depends on the name alone (identical
# on every Julia version), so adding a file never moves another file and a new
# test file needs NO edit to the assignment file. Every file runs in exactly one
# shard. A file named in the assignment but missing on disk (or excluded) is an
# error: a stale list must fail loudly, not run one file fewer.
# Files runtests.jl itself excludes (`_TEST_EXCLUDE`) are excluded here too.

const _TESTDIR = @__DIR__

function _read_names(path)
    out = String[]
    for line in eachline(path)
        s = strip(first(split(line, '#'; limit = 2)))
        isempty(s) || append!(out, split(s))
    end
    return out
end

# `_TEST_EXCLUDE` as defined in runtests.jl, parsed so the two cannot drift.
function _runtests_excludes()
    txt = read(joinpath(_TESTDIR, "runtests.jl"), String)
    m = match(r"const _TEST_EXCLUDE = \[(.*?)\n\]"s, txt)
    m === nothing && return String[]
    return [x.captures[1] for x in eachmatch(r"\"(test_[^\"]+\.jl)\"", m.captures[1])]
end

# Shard (1..N) for a file that the assignment does not list: FNV-1a (32 bit) of
# the file name's bytes, modulo N. Plain UInt32 arithmetic, so the result is the
# same on every Julia version and platform (unlike `hash`).
function _fnv1a32(name::AbstractString)
    h = 0x811c9dc5
    for b in codeunits(name)
        h = (h ⊻ b) * 0x01000193
    end
    return h
end
_unlisted_shard(name::AbstractString, N::Integer) = Int(_fnv1a32(name) % UInt32(N)) + 1

"""
    shard_files(assign_path, k; slow = String[], mode = :all) -> Vector{String}

Files for shard `k` of the assignment in `assign_path`, with modulo fallback for
unassigned files. `mode` is `:all`, `:skip_slow` or `:only_slow`.
"""
function shard_files(assign_path::AbstractString, k::Integer; slow = String[], mode::Symbol = :all)
    assign = Dict{Int,Vector{String}}()
    for line in eachline(assign_path)
        s = strip(first(split(line, '#'; limit = 2)))
        isempty(s) && continue
        toks = split(s)
        kk = tryparse(Int, toks[1])
        kk === nothing && error("assignment line must start with a shard number: $(repr(line))")
        append!(get!(assign, kk, String[]), toks[2:end])
    end
    N = length(assign)
    sort(collect(keys(assign))) == collect(1:N) ||
        error("assignment shard numbers must be 1..N without gaps; got $(sort(collect(keys(assign))))")
    1 <= k <= N || error("shard k=$k outside 1:$N")
    excl = Set(_runtests_excludes())
    ondisk = sort(filter(f -> occursin(r"^test_.*\.jl$", f) && !(f in excl), readdir(_TESTDIR)))
    ondisk_set = Set(ondisk)
    assigned = Set(vcat(values(assign)...))
    unknown = filter(f -> !(f in assigned), ondisk)
    mine = [f for f in assign[k] if f in ondisk_set]
    stale = sort(unique(filter(f -> !(f in ondisk_set), vcat(values(assign)...))))
    isempty(stale) ||
        error("assignment names files that are missing on disk or excluded: ", join(stale, " "))
    append!(mine, [f for f in unknown if _unlisted_shard(f, N) == k])
    slowset = Set(slow)
    mode === :skip_slow && filter!(f -> !(f in slowset), mine)
    mode === :only_slow && filter!(f -> f in slowset, mine)
    return mine
end

function _main(args)
    pos = String[]; slow_path = nothing; mode = :all; list_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--slow-list"; slow_path = args[i += 1]
        elseif a == "--skip-slow"; mode = :skip_slow
        elseif a == "--only-slow"; mode = :only_slow
        elseif a == "--list"; list_only = true
        elseif startswith(a, "--"); error("unknown option $a")
        else push!(pos, a)
        end
        i += 1
    end
    length(pos) == 2 || error("usage: shard_by_time.jl ASSIGN_FILE K [--slow-list F] [--skip-slow|--only-slow] [--list]")
    slow_path === nothing && isfile(joinpath(_TESTDIR, "slow_tier.txt")) &&
        (slow_path = joinpath(_TESTDIR, "slow_tier.txt"))
    slow = slow_path === nothing ? String[] : _read_names(slow_path)
    k = parse(Int, pos[2])
    files = shard_files(pos[1], k; slow = slow, mode = mode)
    println("shard_by_time: shard $k, $(length(files)) files, mode=$mode, slow-list=$(length(slow)) names")
    if list_only
        foreach(f -> println("  ", f), files)
        return
    end
    # Some test files read fixtures relative to the working directory after an
    # include_string (where @__DIR__ is lost); runtests.jl runs with cwd = test/.
    cd(_TESTDIR)
    @eval Main begin
        using DRModels
        using Test, LinearAlgebra, SparseArrays, Random
    end
    # Same BLAS pin as runtests.jl: the suite's numerical pins assume 1 thread.
    Base.invokelatest(LinearAlgebraBLASPin)
    t0 = time()
    times = Pair{String,Float64}[]
    Base.invokelatest(_run_files, files, times)
    println("shard_by_time: DONE shard $k, files=$(length(files)), total=$(round(time() - t0; digits = 0)) s")
    for (f, t) in sort(times; by = last, rev = true)
        println("  TIME ", rpad(f, 60), round(t; digits = 1), " s")
    end
end

LinearAlgebraBLASPin() = Main.LinearAlgebra.BLAS.set_num_threads(1)

function _run_files(files, times)
    Test = Main.Test
    @eval Main begin
        Test.@testset "DRModels time-sharded run" begin
            # A failing file must not abort the rest of the shard: record it, keep
            # going, and exit non-zero at the end.
            failed = String[]
            for f in $files
                t = @elapsed try
                    include(joinpath($_TESTDIR, f))
                catch err
                    push!(failed, f)
                    println("SHARDFAIL ", f, ": ", first(split(sprint(showerror, err), '\n'))); flush(stdout)
                end
                push!($times, f => t)
                println("SHARD ", f, " ", round(t; digits = 1), " s"); flush(stdout)
            end
            isempty(failed) || (println("shard_by_time: FAILED FILES: ", join(failed, " ")); flush(stdout))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    _main(ARGS)
end
