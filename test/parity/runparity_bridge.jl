# runparity_bridge.jl — gated drm_bridge fixture runner (DRM_PARITY_TESTS=1).
#
# Same committed drmTMB-generated numbers as runparity.jl (0.6.0 meta), but fits via
# `drm_bridge` (string formula + family string + data) — the marshalling path R
# calls. xfam-external-gllvm is OUT of the #370 cohort (unsupported family /
# cross-package estimand) and is skipped here.

using DRModels
using Test
using TOML

isdefined(@__MODULE__, :ParityExpected) || include("compare.jl")
isdefined(@__MODULE__, :load_expected) || include("loadfixture.jl")

# Cohort admitted by #370 + #383 (+4 FE) + #385 (nbinom2-dispersion).
# Anything else under fixtures/ is skipped here (native runparity.jl still
# walks the full directory).
const _BRIDGE_PARITY_COHORT = Set([
    "gaussian-locscale",
    "gaussian-bivariate-rho12",
    "robust-student",
    "count-nbinom2",
    "proportion-beta",
    "meta-analysis-V",
    "count-poisson",
    "positive-gamma",
    "binomial-trials",
    "positive-lognormal",
    "nbinom2-dispersion",
])

let fixtures_root = joinpath(@__DIR__, "fixtures")
    cases = String[]
    if isdir(fixtures_root)
        for name in sort(readdir(fixtures_root))
            startswith(name, "_") && continue
            name in _BRIDGE_PARITY_COHORT || continue
            dir = joinpath(fixtures_root, name)
            isdir(dir) || continue
            isfile(joinpath(dir, "expected.toml")) || continue
            push!(cases, dir)
        end
    end

    if isempty(cases)
        @info "No drm_bridge parity cohort fixtures found under test/parity/fixtures/"
        @test true
    else
        @testset "drm_bridge fixtures" begin
            for dir in cases
                casename = basename(dir)
                @testset "$casename" begin
                    expected = load_expected(dir)
                    fit_meta = TOML.parsefile(joinpath(dir, "expected.toml"))["fit"]
                    formula_text = String(get(fit_meta, "formula", ""))
                    family = String(get(fit_meta, "family", expected.family))
                    data = load_data(dir)
                    out = drm_bridge(; formula = formula_text, family = family, data = data)
                    result = compare_bridge(out, expected)
                    if !result.passed
                        @error "drm_bridge parity FAILED for case `$casename`" failures = result.failures
                    end
                    @test result.passed
                end
            end
        end
    end
end
