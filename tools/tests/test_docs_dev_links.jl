using Test

# #670 / #681: the Part 2 location-scale-scale and bivariate-nongaussian tutorials
# are not on /stable (owner ruling 2026-10-11: no release, link to /dev). A relative
# link from a /stable page to either one is therefore a 404. Every link to them
# from README.md or docs/src must be the absolute /dev URL.
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DEV = "https://itchyshin.github.io/DRModels.jl/dev/tutorials/"
const SLUGS = ["location-scale-scale", "bivariate-nongaussian"]

function markdown_files()
    files = [joinpath(ROOT, "README.md")]
    for (dir, _, names) in walkdir(joinpath(ROOT, "docs", "src"))
        append!(files, joinpath.(dir, filter(endswith(".md"), names)))
    end
    files
end

function tutorial_links()
    found = Tuple{String,String}[]
    for f in markdown_files(), m in eachmatch(r"\]\(([^)\s]+)\)", read(f, String))
        target = m.captures[1]
        startswith(target, "https://itchyshin.github.io/drmTMB") && continue  # R twin articles
        any(s -> occursin(s, target), SLUGS) && push!(found, (relpath(f, ROOT), target))
    end
    found
end

@testset "Links to /dev-only tutorials are absolute /dev URLs (#670, #681)" begin
    links = tutorial_links()
    @test length(links) >= 5  # guards against the scan silently matching nothing
    for (file, target) in links
        @test (file, any(s -> target == DEV * s, SLUGS)) == (file, true)
    end
end
println("DOCS_DEV_LINKS_PASS")
