#!/usr/bin/env julia
# build_check_log.jl — render the combined DRModels.jl gate log to stdout.
#
# The log is split to avoid merge collisions (see docs/dev-log/check-log.d/README.md):
#   • docs/dev-log/check-log.md  — frozen historical table (one big table).
#   • docs/dev-log/check-log.d/  — one file per slice, each a single table row;
#                                  new files never conflict with each other.
#
# This script concatenates the two for a combined view. It does NOT rewrite
# check-log.md (a generated, committed file would re-introduce the very conflicts
# we are removing). Rows from check-log.d/ are emitted in filename (≈ date) order.
#
# Usage:
#   julia tools/build_check_log.jl            # print combined log to stdout
#   julia tools/build_check_log.jl --check    # exit 1 if any check-log.d/ entry is malformed

const ROOT     = normpath(joinpath(@__DIR__, ".."))
const FROZEN   = joinpath(ROOT, "docs", "dev-log", "check-log.md")
const ENTRYDIR = joinpath(ROOT, "docs", "dev-log", "check-log.d")

"Collect the single-row entry files from check-log.d/ (excluding README.md), sorted."
function entry_files()
    isdir(ENTRYDIR) || return String[]
    files = filter(readdir(ENTRYDIR)) do f
        endswith(f, ".md") && f != "README.md"
    end
    return sort(files)
end

"A valid entry row is a Markdown table row: starts and ends with `|`, ≥ 5 columns."
function valid_row(line)
    s = strip(line)
    startswith(s, "|") || return false
    endswith(s, "|") || return false
    return count(==('|'), s) >= 6   # 5 columns ⇒ 6 pipes
end

function main()
    check = "--check" in ARGS

    bad = String[]
    rows = String[]
    for f in entry_files()
        content = read(joinpath(ENTRYDIR, f), String)
        got_row = false
        for line in eachline(IOBuffer(content))
            isempty(strip(line)) && continue
            if valid_row(line)
                push!(rows, strip(line)); got_row = true
            else
                push!(bad, "$f: not a table row → $(strip(line))")
            end
        end
        got_row || push!(bad, "$f: no table row found")
    end

    if check
        if isempty(bad)
            println("check-log.d/: all $(length(rows)) entries well-formed ✓")
            return 0
        else
            println(stderr, "check-log.d/ malformed entries:")
            foreach(b -> println(stderr, "  ✗ ", b), bad)
            return 1
        end
    end

    # Combined view: frozen history, then the per-slice entries under a fresh header.
    print(read(FROZEN, String))
    if !isempty(rows)
        println()
        println("## Entries since 2026-06-02 (from `check-log.d/`)\n")
        println("| Date | Slice / Issue | Gate run | Result | By |")
        println("|---|---|---|---|---|")
        foreach(println, rows)
    end
    if !isempty(bad)
        println(stderr, "\n⚠ malformed check-log.d/ entries (skipped):")
        foreach(b -> println(stderr, "  ✗ ", b), bad)
    end
    return 0
end

exit(main())
