using Documenter, DocumenterVitepress, Test, Logging

# Reproduce the directory contract in an isolated, fresh documentation project.
# This intentionally runs no DRModels fits. The negative case must remain fatal.
function nested_probe(broken)
    # macOS /var and /private/var alias the same directory. Use one spelling
    # throughout; the installed writer otherwise constructs invalid relpaths.
    # Vitepress reads Git tags even with build_vitepress=false. Keep the fresh
    # project inside this checkout's ignored build area so that lookup is valid.
    scratch = joinpath(realpath(joinpath(@__DIR__, "..", "..")), "docs", "build")
    mkpath(scratch)
    root = realpath(mktempdir(scratch; cleanup=false))
    mkpath(joinpath(root, "src", "nested"))
    code = broken ? "error(\"DELIBERATE_NESTED_FAILURE\")" :
        "@assert basename(pwd()) == \"nested\"\nprintln(\"NESTED_EXECUTED\")"
    write(joinpath(root, "src", "nested", "probe.md"),
        "# Nested example\n\n```@example nested\n$code\n```\n")
    println("NESTED_PROBE_ROOT=$root broken=$broken")
    makedocs(; root, source="src", build="build", clean=true,
        remotes=nothing, sitename="Nested example contract", pagesonly=true,
        pages=["Nested" => "nested/probe.md"], warnonly=false,
        format=DocumenterVitepress.MarkdownVitepress(
            repo="github.com/itchyshin/DRModels.jl", devbranch="main", devurl="dev",
            inventory_version="0.0.0", build_vitepress=false, install_npm=false))
    output = joinpath(root, "build", ".documenter", "nested", "probe.md")
    @test isfile(output)
    @test occursin(r"(?m)^NESTED_EXECUTED$", read(output, String))
end

@testset "Nested Documenter execution contract" begin
    nested_probe(false)
    messages = IOBuffer()
    failure = with_logger(SimpleLogger(messages)) do
        try
            nested_probe(true)
            nothing
        catch err
            err
        end
    end
    output = String(take!(messages))
    print(output)
    @test failure isa Exception
    @test occursin("DELIBERATE_NESTED_FAILURE", output)
end
println("NESTED_DOCUMENTATION_CONTRACT_PASS")
