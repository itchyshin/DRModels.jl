#!/usr/bin/env julia

"""Smoke the cached Vitepress writer on the compatible visible reader pages.

This runs Documenter's `pagesonly = true` mode on the canonical first-fit page
and the bridge page. The legacy transition URL remains unlisted in docs/make.jl
because this installed Vitepress writer cannot render Documenter's `hide(...)`
navigation tuple. The writer emits Markdown only: it does not run npm, build the
Vitepress theme, deploy, or make a live-site claim.
"""

using Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DOCS_ROOT = joinpath(REPO_ROOT, "docs")
const DEFAULT_BUILD_DIR = joinpath(DOCS_ROOT, "build", "parity-pilot-vitepress-pagesonly-markdown")
const CANONICAL_PAGE = joinpath(DOCS_ROOT, "src", "getting-started.md")
const EXPECTED_EXAMPLES = 9
const STARTED_AT = time()

Pkg.activate(DOCS_ROOT)
Pkg.offline(true)
Pkg.develop(path = REPO_ROOT)

using DRModels
using Documenter
using DocumenterVitepress

function assert_isolated_drm()
    loaded = realpath(pathof(DRModels))
    source = realpath(joinpath(REPO_ROOT, "src", "DRModels.jl"))
    loaded == source || error("loaded DRModels is not the isolated worktree: $loaded")
    return loaded
end

function pilot_build_dir(args::Vector{String})
    candidate = if isempty(args)
        DEFAULT_BUILD_DIR
    elseif length(args) == 2 && args[1] == "--build-dir"
        abspath(args[2])
    else
        error("usage: parity_docs_pilot.jl [--build-dir docs/build/<fresh-name>]")
    end
    build_root = abspath(joinpath(DOCS_ROOT, "build"))
    relative = relpath(candidate, build_root)
    (!isabspath(relative) && relative != "." && !startswith(relative, "..")) ||
        error("pilot build directory must be a fresh child of $build_root: $candidate")
    return candidate
end

function main(args::Vector{String} = ARGS)
    build_dir = pilot_build_dir(args)
    ispath(build_dir) && error("pilot output already exists; a fresh build directory is required: $build_dir")
    loaded = assert_isolated_drm()
    source_examples = sum(contains(line, "```@example") for line in eachline(CANONICAL_PAGE))
    source_examples == EXPECTED_EXAMPLES || error("expected $EXPECTED_EXAMPLES canonical @example blocks, found $source_examples")
    pages = [
        "Getting started" => "getting-started.md",
        "R ↔ Julia bridge" => "r-julia-bridge.md",
    ]
    makedocs(
        root = DOCS_ROOT,
        source = "src",
        build = relpath(build_dir, DOCS_ROOT),
        clean = true,
        sitename = "DRModels.jl two-page documentation pilot",
        pagesonly = true,
        # Out-of-scope local cross-references are recorded as warnings; failed
        # @example blocks stay fatal and cannot yield a pass token.
        warnonly = [:cross_references, :linkcheck, :footnote],
        pages = pages,
        format = DocumenterVitepress.MarkdownVitepress(
            repo = "github.com/itchyshin/DRModels.jl",
            devbranch = "main",
            devurl = "dev",
            build_vitepress = false,
            install_npm = false,
        ),
    )
    expected = [
        joinpath(build_dir, ".documenter", "getting-started.md"),
        joinpath(build_dir, ".documenter", "r-julia-bridge.md"),
    ]
    missing = filter(!isfile, expected)
    isempty(missing) || error("pilot did not emit expected Markdown: $(join(missing, ", "))")
    emitted = sort(filter(path -> endswith(path, ".md"),
        [joinpath(root, file) for (root, _dirs, files) in walkdir(build_dir) for file in files]))
    sort(expected) == emitted || error("pagesonly pilot emitted unexpected Markdown: $(join(emitted, ", "))")
    println("DOCS_PILOT_PASS backend=DocumenterVitepress markdown_only=true pages=2 examples=$source_examples loaded_drm=isolated")
    println("DOCS_PILOT_BUILD_DIR=$(build_dir)")
    println("DOCS_PILOT_LOADED_DRM=$(loaded)")
    println("DOCS_PILOT_SECONDS=$(round(time() - STARTED_AT; digits = 3))")
end

main()
