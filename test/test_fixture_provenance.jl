# #473: "drmTMB 0.7.0" spans 16 builds; fixtures can't tell which. Every
# test/parity/**/expected.meta.toml comparing against drmTMB must record a
# `drmtmb_code_hash` line -- either a real hash (tools/drmtmb_provenance.R) or
# the explicit literal "unknown (pre-#473)" for fixtures whose generating
# build cannot be recovered. A fixture with neither is provenance-blind: a
# later disagreement can't tell a DRM.jl regression from the comparator
# having moved underneath it.
#
# xfam-external-gllvm/expected.meta.toml is NOT a drmTMB fixture (comparator
# is gllvm, recorded as gllvm_version) and is excluded from this check.
using Test, TOML

@testset "fixture provenance (#473)" begin
    fixtures_root = joinpath(@__DIR__, "parity")
    meta_paths = String[]
    for (root, _, files) in walkdir(fixtures_root)
        for fn in files
            if fn == "expected.meta.toml"
                push!(meta_paths, joinpath(root, fn))
            end
        end
    end

    @test !isempty(meta_paths)

    for path in meta_paths
        meta = TOML.parsefile(path)
        rel = relpath(path, fixtures_root)
        if haskey(meta, "gllvm_version")
            # Not a drmTMB comparator fixture; #473 does not apply.
            continue
        end
        @testset "$rel" begin
            @test haskey(meta, "drmtmb_code_hash")
            if haskey(meta, "drmtmb_code_hash")
                hash = meta["drmtmb_code_hash"]
                @test hash isa AbstractString && !isempty(hash)
            end
        end
    end
end
