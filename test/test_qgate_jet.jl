# Included by runtests.jl ONLY when JET is available (test/Project.toml env,
# e.g. under Pkg.test / CI). Kept separate because `JET.@test_opt` is a macro:
# it expands at lowering, before any runtime `_HAS_JET` guard, so an inline call
# would UndefVarError in a JET-less environment.
#
# Workflow Q — JET gate (Karpinski): type-stability of the hot reparameterization
# kernels used on every q=4 fit (`lc_to_Λ` / `Λ_to_lc`). Full CHOLMOD-facing
# Takahashi paths retain known stdlib-boundary Unions (same class as GLLVM's
# intentional exclusion) and are not gated here.

let
    lc = Float64[0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    JET.@test_opt target_modules = (DRModels,) DRModels.lc_to_Λ(lc)
    Λ = DRModels.lc_to_Λ(lc)
    JET.@test_opt target_modules = (DRModels,) DRModels.Λ_to_lc(Λ)
end
