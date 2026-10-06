# runparity_bridge_formula.jl — gated drm_bridge R-formula-construct fixture
# runner (DRM_PARITY_TESTS=1), issue #467.
#
# Fixtures under fixtures/bridge-*/ prove that `drm_bridge`'s translation of
# R's `scale()`, `I()`, `factor()`, general `- term` removal, and `(...)^k`
# crossing reproduces drmTMB's fitted numbers on byte-identical data (see
# gen_bridge_formula_fixtures.R). Each fixture's `[coef]`/`[vcov]` keys are
# ALREADY written in `drm_bridge`'s own flat naming (including the synthetic
# `__bridge_<kind>_<n>` columns), so `compare_bridge` needs no renaming here.

using DRModels
using Test
using TOML

isdefined(@__MODULE__, :ParityExpected) || include("compare.jl")
isdefined(@__MODULE__, :load_expected) || include("loadfixture.jl")

const _BRIDGE_FORMULA_COHORT = Set([
    "bridge-scale",
    "bridge-I",
    "bridge-factor",
    "bridge-minus-term",
    "bridge-power",
    "bridge-poly",
    "bridge-poly-cross",
])

let fixtures_root = joinpath(@__DIR__, "fixtures")
    cases = String[]
    if isdir(fixtures_root)
        for name in sort(readdir(fixtures_root))
            name in _BRIDGE_FORMULA_COHORT || continue
            dir = joinpath(fixtures_root, name)
            isdir(dir) || continue
            isfile(joinpath(dir, "expected.toml")) || continue
            push!(cases, dir)
        end
    end

    if isempty(cases)
        @info "No drm_bridge formula-construct fixtures found under test/parity/fixtures/bridge-*"
        @test true
    else
        @testset "drm_bridge R-formula-construct fixtures (#467)" begin
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
                        @error "drm_bridge formula-construct parity FAILED for `$casename`" failures = result.failures
                    end
                    @test result.passed
                end
            end
        end
    end
end
