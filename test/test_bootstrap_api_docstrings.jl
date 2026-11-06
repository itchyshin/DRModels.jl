@testset "bootstrap API docstrings list forwarded controls" begin
    inference_source = read(joinpath(@__DIR__, "..", "src", "inference.jl"), String)
    function docstring_for(name)
        lines = split(inference_source, '\n')
        first_signature = findfirst(line -> occursin("    $(name)(", line), lines)
        @test first_signature !== nothing
        opening = findlast(==("\"\"\""), lines[1:first_signature-1])
        closing = findnext(==("\"\"\""), lines, first_signature + 1)
        return join(lines[opening+1:closing-1], '\n')
    end

    ci_doc = docstring_for("bootstrap_ci")
    summary_doc = docstring_for("bootstrap_summary")

    for (name, doc) in (("bootstrap_ci", ci_doc), ("bootstrap_summary", summary_doc))
        @test occursin("failures = :error", doc)
        @test occursin("check_converged = true", doc)
        signatures = [strip(line) for line in split(doc, '\n') if startswith(strip(line), "$(name)(")]
        @test length(signatures) == 4
        @test occursin("family::Gaussian", signatures[1])
        @test occursin("family;", signatures[2])
        @test occursin("fit::DrmFit{<:Gaussian}", signatures[3])
        @test occursin("fit::DrmFit;", signatures[4])
        @test occursin("algorithm = :auto", signatures[1])
        @test !occursin("algorithm", signatures[2])
        @test occursin("algorithm = :auto", signatures[3])
        @test !occursin("algorithm", signatures[4])
        @test occursin("g_tol = 1e-8", signatures[1])
        @test occursin("g_tol = 1e-8", signatures[3])
    end
end
