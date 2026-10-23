# #985: user-facing exports that the manual's reference pages must list in an
# `@docs` block (docs/make.jl keeps `checkdocs = :none`, so Documenter itself
# would never report a missing entry). Each name must also carry a docstring,
# otherwise the `@docs` block breaks the docs build ("no docs found").
using DRModels
using Test

const _REF_DIR = joinpath(@__DIR__, "..", "docs", "src", "reference")

# Names that appear as a line inside any ```@docs fence of the reference pages.
function _reference_docs_names()
    names = Set{String}()
    for f in readdir(_REF_DIR; join = true)
        endswith(f, ".md") || continue
        inblock = false
        for line in eachline(f)
            s = strip(line)
            if startswith(s, "```@docs")
                inblock = true
            elseif startswith(s, "```")
                inblock = false
            elseif inblock && !isempty(s)
                push!(names, s)
            end
        end
    end
    return names
end

@testset "#985 reference pages list the user-facing exports" begin
    if !isdir(_REF_DIR)
        @info "docs/src/reference not found; skipping #985 check"
    else
        listed = _reference_docs_names()
        for nm in ("bridge_diagnostics", "drm_formula", "offset")
            @testset "$nm" begin
                @test nm in listed
                @test Base.Docs.hasdoc(DRModels, Symbol(nm))
            end
        end
    end
end
