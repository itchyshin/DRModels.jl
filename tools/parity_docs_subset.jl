#!/usr/bin/env julia
# Execute selected real documentation pages with fatal example errors.
# Writes Vitepress Markdown only, keeps a fresh output directory, and reports
# the exact selected pages/example count. Never claims whole-site coverage.
using DRModels, Documenter, DocumenterVitepress, LinearAlgebra

include(joinpath(@__DIR__, "parity_docs_navigation.jl"))

const REPO = realpath(joinpath(@__DIR__, ".."))
const DOCS = joinpath(REPO, "docs")

function main(args)
    length(args) >= 4 && iseven(length(args)) || error("use --build-dir PATH with --page PATH or --pages-file PATH")
    build = nothing
    selected = String[]
    navigation = "subset"
    for i in 1:2:length(args)
        flag, value = args[i:i+1]
        if flag == "--build-dir"
            isnothing(build) || error("duplicate build directory")
            build = abspath(value)
        elseif flag == "--navigation"
            value in ("subset", "production") || error("unknown navigation mode $value")
            navigation = value
        elseif flag == "--page"
            push!(selected, value)
        elseif flag == "--pages-file"
            append!(selected, filter(!isempty, strip.(readlines(value))))
        else
            error("unknown option $flag")
        end
    end
    isnothing(build) && error("build directory required")
    isempty(selected) && error("at least one page required")
    length(unique(selected)) == length(selected) || error("duplicate pages")
    relative = relpath(build, joinpath(DOCS, "build"))
    (relative != "." && !isabspath(relative) && !startswith(relative, "..")) || error("build must be a child of docs/build")
    ispath(build) && error("fresh output required: $build")
    for page in selected
        (!isabspath(page) && !startswith(normpath(page), "..") && endswith(page,".md")) || error("invalid page $page")
        isfile(joinpath(DOCS,"src",page)) || error("missing page $page")
    end
    realpath(pathof(DRModels)) == realpath(joinpath(REPO,"src","DRModels.jl")) || error("wrong DRModels checkout loaded")
    BLAS.set_num_threads(1)
    println("DOCS_RUNTIME julia=$(VERSION) threads=$(Threads.nthreads()) blas=$(BLAS.get_num_threads()) loaded_drm=$(pathof(DRModels))")
    println("DOCS_SELECTED=$(join(selected, ','))")
    count_examples = sum(sum(startswith(strip(line), "```@example") for line in eachline(joinpath(DOCS,"src",page))) for page in selected)
    pages = [splitext(basename(page))[1] => page for page in selected]
    if navigation == "production"
        all_source = sort([relpath(joinpath(dir,file), joinpath(DOCS,"src"))
            for (dir,_,files) in walkdir(joinpath(DOCS,"src")) for file in files if endswith(file,".md")])
        sort(selected) == all_source || error("production run must include every source page")
        pages = production_navigation(read(joinpath(DOCS,"make.jl"), String))
        paths = navigation_paths(pages)
        length(paths) == length(unique(paths)) || error("duplicate production navigation route")
        all(p -> p in selected, paths) || error("production navigation references a missing page")
        println("DOCS_PRODUCTION_NAVIGATION visible=$(length(paths)) emitted=$(length(selected))")
    end
    started = time()
    makedocs(; root=DOCS, source="src", build=relpath(build,DOCS), clean=true,
        sitename="DRModels.jl", authors="Shinichi Nakagawa", pagesonly=(navigation == "subset"),
        pages=pages, modules=(navigation == "production" ? [DRModels] : Module[]),
        warnonly=(navigation == "subset" ? [:cross_references, :linkcheck, :footnote] : false),
        format=DocumenterVitepress.MarkdownVitepress(
            repo="github.com/itchyshin/DRModels.jl", devbranch="main", devurl="dev",
            build_vitepress=false, install_npm=false))
    expected = sort([joinpath(build,".documenter",p) for p in selected])
    emitted = sort([joinpath(dir,file) for (dir,_,files) in walkdir(build) for file in files if endswith(file,".md")])
    expected == emitted || error("emitted page inventory differs from request")
    println("DOCS_SUBSET_PASS pages=$(length(selected)) examples=$count_examples seconds=$(round(time()-started;digits=3))")
    println("DOCS_SUBSET_BUILD=$build")
end

main(ARGS)
