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

# `Base.Docs.hasdoc` only exists from Julia 1.11; the module's doc table works on 1.10 too.
# An alias (`const drm_formula = bf`) carries no docstring of its own: Documenter finds the
# one on the function it points to, so resolve the name to that function first.
function _has_docstring(nm::Symbol)
    v = getfield(DRModels, nm)
    b = v isa Function ? Base.Docs.Binding(parentmodule(v), nameof(v)) :
        Base.Docs.Binding(DRModels, nm)
    return haskey(Base.Docs.meta(b.mod), b)
end

@testset "#985 reference pages list the user-facing exports" begin
    if !isdir(_REF_DIR)
        @info "docs/src/reference not found; skipping #985 check"
    else
        listed = _reference_docs_names()
        for nm in ("bridge_diagnostics", "drm_formula", "offset")
            @testset "$nm" begin
                @test nm in listed
                @test _has_docstring(Symbol(nm))
            end
        end
    end
end
