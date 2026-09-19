using Test
include(joinpath(@__DIR__, "..", "parity_docs_navigation.jl"))

@testset "Production navigation is data, never executed" begin
    source = "makedocs(pages = [\"Home\" => \"index.md\", \"Guides\" => [\"a.md\", \"B\" => \"b.md\"]]); error(\"must not execute\")"
    nav = production_navigation(source)
    @test nav == ["Home" => "index.md", "Guides" => ["a.md", "B" => "b.md"]]
    @test navigation_paths(nav) == ["index.md", "a.md", "b.md"]
    for bad in ["makedocs(pages = read(\"secret\"))", "makedocs(pages = [run(`echo bad`)])",
                "makedocs(pages = [\"Home\" => (error(\"bad\"); \"index.md\")])",
                "makedocs(pages = [joinpath(\"x\", \"y\")])", "makedocs()",
                "makedocs(pages = [\"a.md\"]); makedocs(pages = [\"b.md\"])"]
        @test_throws ArgumentError production_navigation(bad)
    end
    real = production_navigation(read(joinpath(@__DIR__, "..", "..", "docs", "make.jl"), String))
    paths = navigation_paths(real)
    # Keep an explicit reviewed page count: adding a public page must update
    # this contract deliberately, rather than silently bypassing navigation.
    @test length(paths) == 52
    @test length(unique(paths)) == 52
    @test !("get-started.md" in paths)
    @test "getting-started.md" in paths
    @test length(real) == 6
    @test "reference/engine-internals.md" in paths
    @test count(p -> p isa Pair && p.first == "Tutorials" && p.second isa Vector, real) == 1
    @test count(p -> p isa Pair && p.first == "Development" && p.second isa Vector, real) == 1
    @test all(isfile(joinpath(@__DIR__, "..", "..", "docs", "src", p)) for p in paths)
end
println("PRODUCTION_NAVIGATION_CONTRACT_PASS")
