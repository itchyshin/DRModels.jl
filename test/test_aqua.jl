# Aqua is a TEST-ONLY dependency (declared in test/Project.toml, never in the
# package's runtime [deps] — adding it there would itself be an Aqua stale-deps
# violation). Under `Pkg.test()` / CI the test environment is stacked over the
# package, so `using Aqua` and `using DRModels` both resolve and the block below is a
# no-op. Run the battery via `Pkg.test()` / CI (not against the bare package
# project), so `using Aqua` resolves from the stacked test environment.
using Aqua
using DRModels
using Test

# Julia General-registry hygiene battery (Aqua.jl).
#
# `ambiguities = false`: method-ambiguity detection is disabled. DRModels dispatches
# heavily through Distributions / StatsModels / StatsAPI generics, and Aqua's
# ambiguity pass reports *cross-package* ambiguities (e.g. between Base and our
# distribution-family methods) that are not introduced or fixable by DRModels. This
# is the standard Aqua exclusion for packages that extend external generics; it
# is out of scope for registry hygiene.
#
# Everything else (stale deps, undefined exports, project-extras consistency,
# unbound type parameters, method piracy, and `deps_compat`) runs at the
# default strictness. `deps_compat = true` enforces a `[compat]` entry for
# every `[deps]` package and for `julia` (see Project.toml). The
# `persistent_tasks` probe is intentionally deferred: Aqua creates an
# unrelated temporary project and resolves the package by its UUID through
# General, where this UUID is still registered as `DRM` until the approved
# repository/registry rename gate. It cannot therefore test the pre-registration
# `DRModels` identity; the other Aqua checks remain active.
@testset "Aqua.jl quality assurance" begin
    Aqua.test_all(DRModels; ambiguities = false, deps_compat = true, persistent_tasks = false)
end
