#!/usr/bin/env julia
# Folds per-PR news fragments (news/<slug>.md) into NEWS.md's `## Development`
# section, then deletes the fragments it consumed. See news/README.md for the
# fragment convention this replaces (hand-editing NEWS.md directly, which
# conflicted on GitHub whenever two PRs landed close together).
#
# Usage (from the repo root):
#   julia tools/assemble_news.jl            # writes NEWS.md, deletes fragments
#   julia tools/assemble_news.jl --dry-run  # preview only, no files touched

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const NEWS_PATH = joinpath(REPO_ROOT, "NEWS.md")
const NEWS_DIR = joinpath(REPO_ROOT, "news")
const DEV_HEADING = "## Development"

function fragment_files()
    isdir(NEWS_DIR) || return String[]
    names = filter(f -> endswith(f, ".md") && lowercase(f) != "readme.md",
                    readdir(NEWS_DIR))
    return sort(joinpath.(NEWS_DIR, names))
end

function assemble(; dry_run::Bool = false)
    frags = fragment_files()
    if isempty(frags)
        println("No news fragments found in $NEWS_DIR — nothing to do.")
        return
    end

    news = read(NEWS_PATH, String)
    idx = findfirst(DEV_HEADING, news)
    idx === nothing && error("could not find \"$DEV_HEADING\" heading in $NEWS_PATH")
    # Insert after the heading AND the blank line that conventionally follows
    # it, so the fragment lands where a hand-written first bullet would.
    anchor = DEV_HEADING * "\n\n"
    a = findfirst(anchor, news)
    insert_at = a === nothing ? (idx[end] + 1) : (a[end] + 1)

    bodies = String[]
    for f in frags
        body = strip(read(f, String))
        isempty(body) && continue
        push!(bodies, body)
    end
    isempty(bodies) && (println("All fragments were empty — nothing to do."); return)

    block = join(bodies, "\n\n") * "\n\n"
    new_news = news[1:insert_at-1] * block * news[insert_at:end]

    if dry_run
        println("--- would write $NEWS_PATH ---")
        println(block)
        println("--- would delete ---")
        foreach(println, frags)
        return
    end

    write(NEWS_PATH, new_news)
    foreach(rm, frags)
    println("Folded $(length(frags)) fragment(s) into $NEWS_PATH and deleted them.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    dry_run = "--dry-run" in ARGS
    assemble(; dry_run)
end
