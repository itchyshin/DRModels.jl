using Documenter
using DocumenterVitepress
using DRModels

# Five compact menus keep all reference and tutorial routes reachable.
# The site uses the DocumenterVitepress backend (a VitePress/Vue build, the
# docs.makie.org look). Node is supplied by NodeJS_20_jll — no system install.
# Fail on broken examples, links and omitted module docstrings.
makedocs(
    sitename = "DRModels.jl",
    authors = "Shinichi Nakagawa",
    modules = [DRModels],
    # Only pages named below are public. The source tree also retains
    # development notes and evidence records that are intentionally not a
    # reader-facing documentation surface.
    pagesonly = true,
    # Keep the public API documented without requiring private implementation
    # helpers to appear on a reader-facing page.
    checkdocs = :exports,
    warnonly = false,
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/itchyshin/DRModels.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    pages = [
        "Start" => [
            "Home" => "index.md",
            "Getting started" => "getting-started.md",
            "Detailed capabilities & limits" => "capabilities.md",
            "R ↔ Julia bridge" => "r-julia-bridge.md",
            "Rosetta (R ↔ Julia)" => "rosetta.md",
            hide("Former get started route" => "get-started.md"),
        ],
        "Model guides" => [
            "model-guides/model-map.md",
            "model-guides/which-scale.md",
            "model-guides/distribution-families.md",
            "families.md",
            "model-guides/model-workflow.md",
            "model-guides/model-selection.md",
            "model-guides/convergence.md",
            "model-guides/marginal-la-vs-va.md",
            "model-guides/cross-family-methods.md",
            "model-guides/meta-analysis.md",
            "model-guides/large-data.md",
            "Cross-family bivariate" => "cross-family.md",
        ],
        "Tutorials" => [
            "tutorials/location-scale.md",
            "tutorials/location-scale-scale.md",
            "tutorials/robust-student.md",
            "tutorials/count-nbinom2.md",
            "tutorials/proportion-beta-binomial.md",
            "tutorials/bivariate-coscale.md",
            "tutorials/bivariate-nongaussian.md",
            "tutorials/meta-analysis.md",
            "tutorials/structural-dependence.md",
            "tutorials/animal-models.md",
            "tutorials/phylogenetic-models.md",
            "tutorials/spatial-models.md",
            "tutorials/relmat-known-matrices.md",
            "tutorials/phylogenetic-spatial.md",
        ],
        "Diagnostics" => [
            "diagnostics-and-validation/figure-gallery.md",
            "diagnostics-and-validation/prediction-and-postfit.md",
            "diagnostics-and-validation/profile-likelihood.md",
            "diagnostics-and-validation/implementation-map.md",
            "diagnostics-and-validation/exact-gaussian-diagnostics.md",
            "diagnostics-and-validation/testing-likelihoods.md",
            "diagnostics-and-validation/simulation-plot-grammar.md",
            "diagnostics-and-validation/small-sample-behaviour.md",
        ],
        "Reference" => [
            "reference/package.md",
            "reference/model-specification.md",
            "reference/structured-effect-markers.md",
            "reference/model-fitting-and-postfit.md",
            "reference/visualization.md",
            "API stability" => "api-stability.md",
        ],
    ],
)

# Use DocumenterVitepress.deploydocs (NOT Documenter's): it flattens the VitePress
# build output (build/1/*) into the version root on gh-pages and rewrites the site
# `base`. Plain Documenter.deploydocs deploys build/ verbatim → the site lands as
# build/1/ and every asset/nav link 404s (the bug this site hit). Mirrors GLLVM.jl.
# push_preview = true (mirrors GLLVM.jl): PR docs land at
# https://itchyshin.github.io/DRModels.jl/previews/PR<N>/ so phone/GitHub review
# links work. With false, Documenter still posts documenter/deploy SUCCESS
# pointing at that URL while Deploying: ✘ → 404.
# DRM_DOCS_DEPLOY=false makes this a BUILD-ONLY run (no gh-pages contact).
# Documenter.yml sets it on pull_request runs. Rationale, in full, there: a run
# that pushes to gh-pages has to join the repo-wide serialisation group, and a
# job in that group can be evicted while pending -- which silently blocks the
# REQUIRED `docs` check. A build-only run stays out of the group, so the gate
# measures the build and nothing else. The build itself is unchanged: makedocs
# above still runs with `warnonly = false`, so a broken docstring, link or
# example fails the check exactly as before.
if get(ENV, "DRM_DOCS_DEPLOY", "true") == "true"
    DocumenterVitepress.deploydocs(;
        repo = "github.com/itchyshin/DRModels.jl.git",
        target = joinpath(@__DIR__, "build"),
        devbranch = "main",
        branch = "gh-pages",
        push_preview = true,
    )
else
    @info "DRM_DOCS_DEPLOY=false — built only, skipping deployment to gh-pages."
end
