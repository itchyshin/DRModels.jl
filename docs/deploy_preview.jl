# deploy_preview.jl — publish an ALREADY-BUILT docs/build to previews/PR<N>.
#
# Since #757 the main/tag `docs-deploy` job uses it too: Documenter's deploy_folder
# reads the run's event and ref, so the same call publishes dev/ on a push to main,
# vX.Y/ (+ stable/) on a version tag (one folder per minor version for 0.x), and previews/PR<N>/ on a pull request.
#
# WHY A SECOND ENTRY POINT rather than re-running make.jl. #750 took the required
# `docs` check out of the repo-wide gh-pages serialisation group by making a
# pull_request run build-only. That fixed the eviction that blocked PR #748 thirty
# times over, but it also removed PR previews, because `push_preview` never fires.
#
# This script restores the preview from a SEPARATE, NON-REQUIRED job that carries
# the serialisation group on its own. If that job is evicted while pending -- which
# is exactly what the group does under load -- the result is a missing preview
# rather than a blocked merge. That trade is the whole point of the split.
#
# It deliberately does NOT rebuild. DocumenterVitepress.deploydocs publishes what is
# already in `target`, so the preview is byte-for-byte the artefact the required
# gate measured. Rebuilding would cost a second npm/VitePress build AND allow the
# preview to differ from what was checked.
#
# THE THING MOST LIKELY TO BREAK THIS: deploydocs reads `joinpath(root, target,
# "bases.txt")` and then deploys `target/1`, `target/2`, ... one Documenter
# deploydocs call per base. So the downloaded artefact MUST restore docs/build to
# the same relative path, INCLUDING bases.txt and the numbered subdirectories. The
# workflow asserts that below before calling this, so a missing file fails loudly
# with a readable message instead of deep inside Documenter.
using DRModels
using DocumenterVitepress

target = joinpath(@__DIR__, "build")
bases = joinpath(target, "bases.txt")
isfile(bases) || error(
    "deploy_preview: $(bases) is missing. The docs artefact did not restore " *
    "docs/build to its original relative path, so deploydocs has no bases to " *
    "publish. Check the download-artifact step's `path:` in Documenter.yml.")

@info "deploy_preview: publishing the prebuilt docs" target bases = read(bases, String)

# NOTE: no `versions=` kwarg. DocumenterVitepress.deploydocs HARD-ERRORS on one
# (src/DocumenterVitepress.jl:145); versions are steered by bases.txt instead.
DocumenterVitepress.deploydocs(;
    repo = "github.com/itchyshin/DRModels.jl.git",
    target = target,
    devbranch = "main",
    branch = "gh-pages",
    push_preview = true,
)
