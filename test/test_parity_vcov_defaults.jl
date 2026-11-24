# test_parity_vcov_defaults.jl — SE and vcov against the compare.jl defaults.
#
# `compare_fit` / `compare_bridge` already compare a stored covariance matrix
# and, when the fixture has an `[se]` block, per-coefficient standard errors.
# A fixture `[tol]` block can override those bars. This file is the always-on
# assertion that does not: `rtol_vcov` / `atol_vcov` / `rtol_se` / `atol_se`
# are stripped before the comparison, so the keyword defaults in
# `test/parity/compare.jl` apply (`rtol_vcov = 1e-3`, `atol_vcov = 1e-8`,
# `rtol_se = 1e-3`, `atol_se = 1e-8`). Other `[tol]` keys (loglik, aic, coef)
# are left as the fixture wrote them. This file does not edit any fixture.
#
# Two cells miss `atol_vcov = 1e-8` and are not given a new or looser bar
# (refs #1503, waiting on Shinichi). Measured on Julia 1.13.4 at a029c2710
# with the vcov keys removed:
#   meta-analysis-V — vcov[mu_(Intercept), sigma_(Intercept)] |Δ| = 1.4619471914728503e-8
#   robust-student  — vcov[mu_x, nu_(Intercept)]              |Δ| = 2.0435463875702732e-8
# (each entry is symmetric). Their other axes, including SE, are still checked.
# A later fit that brings either cell inside 1e-8 still passes; this file
# does not pin the disagreement, and it does not assert a wider tolerance.

module ParityVcovDefaults

using DRModels
using Random
using Test
using TOML

include(joinpath(@__DIR__, "parity", "compare.jl"))
include(joinpath(@__DIR__, "parity", "loadfixture.jl"))

# Helpers only. The `let fixtures_root` block at the bottom of runparity.jl
# is the gated suite; including it would run that suite (and its fixture
# `[tol]` overrides) a second time.
const _RUNPARITY_SRC = read(joinpath(@__DIR__, "parity", "runparity.jl"), String)
const _RUNPARITY_CUT = findfirst("let fixtures_root", _RUNPARITY_SRC)
_RUNPARITY_CUT === nothing && error("runparity.jl no longer has the fixture loop this test slices off")
include_string(@__MODULE__,
    _RUNPARITY_SRC[1:prevind(_RUNPARITY_SRC, first(_RUNPARITY_CUT))],
    "parity/runparity_helpers.jl")

# Cells whose stored atol_vcov is looser than compare.jl's default. Not asserted
# on the vcov axis. Do not add them by widening the default.
const _VCOV_HELD = Set(("meta-analysis-V", "robust-student"))

const _VCOV_SE_TOL_KEYS = ("rtol_vcov", "atol_vcov", "rtol_se", "atol_se")

"""
    _at_compare_defaults(exp) -> ParityExpected

Copy `exp` with the vcov and SE tolerance overrides removed, so `compare_fit`
/ `compare_bridge` use the keyword defaults in `compare.jl`.
"""
function _at_compare_defaults(exp::ParityExpected)
    tol = Dict{String,Float64}(k => v for (k, v) in exp.tol if !(k in _VCOV_SE_TOL_KEYS))
    return ParityExpected(;
        family = exp.family,
        coef = exp.coef,
        loglik = exp.loglik,
        aic = exp.aic,
        df = exp.df,
        n = exp.n,
        vcov_order = exp.vcov_order,
        vcov = exp.vcov,
        tol = tol,
        ranef_group = exp.ranef_group,
        ranef = exp.ranef,
        se = exp.se,
        se_not_comparable = exp.se_not_comparable)
end

function _fit_and_compare(dir::AbstractString, casename::AbstractString, expected::ParityExpected)
    fit_meta = TOML.parsefile(joinpath(dir, "expected.toml"))["fit"]
    formula_text = String(get(fit_meta, "formula", ""))
    family = String(get(fit_meta, "family", expected.family))
    data = load_data(dir)
    if startswith(casename, "bridge-")
        out = drm_bridge(; formula = formula_text, family = family, data = data)
        return compare_bridge(out, expected)
    end
    fam = _parity_family(expected.family)
    fam === nothing && error("family `$(expected.family)` is not supported by the parity runner")
    bundle = _parity_formula(formula_text, expected.family)
    bundle === nothing &&
        error("formula `$(formula_text)` for `$(expected.family)` is not supported by the parity runner")
    fit = drm(bundle, fam; data = data)
    return compare_fit(fit, expected)
end

@testset "vcov/SE defaults ignore a looser fixture override" begin
    Random.seed!(20261124)
    n = 80
    x = randn(n)
    x .-= sum(x) / length(x)   # keep Cov(intercept, slope) small so atol, not rtol, binds
    y = 0.2 .+ 0.5 .* x .+ exp(-0.4) .* randn(n)
    fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
    named = drm_coef_named(fit)
    order = sort!(collect(keys(named)))
    # vcov(fit) follows θ order, which is the block order, not the sorted names.
    fitnames = String[]
    namemap = Dict(p => ns for (p, ns) in fit.coefnames)
    for (param, r) in fit.blocks
        haskey(namemap, param) || continue
        for nm in namemap[param]
            push!(fitnames, "$(param)_$(nm)")
        end
    end
    idx = Dict(nm => i for (i, nm) in enumerate(fitnames))
    perm = [idx[nm] for nm in order]
    V = Matrix{Float64}(vcov(fit)[perm, perm])
    se = drm_se_named(fit)
    # A fixture-style override wider than the default must not be consulted.
    loose = Dict{String,Float64}("atol_vcov" => 1e-6, "atol_se" => 1e-6)
    function expected_from(Vmat, semap)
        ParityExpected(;
            family = "gaussian",
            coef = named,
            loglik = loglik(fit),
            aic = -2 * loglik(fit) + 2 * dof(fit),
            df = dof(fit),
            n = nobs(fit),
            vcov_order = order,
            vcov = Vmat,
            tol = loose,
            se = semap)
    end
    strict_ok = _at_compare_defaults(expected_from(V, se))
    for key in _VCOV_SE_TOL_KEYS
        @test !haskey(strict_ok.tol, key)
    end
    res_ok = compare_fit(fit, strict_ok)
    @test res_ok.passed
    # |Δ| = 2e-8 is the size of the two held fixture cross-terms. It is outside
    # atol_vcov = 1e-8 and, on a near-zero entry, outside rtol_vcov as well.
    abs(V[1, 2]) < 1e-3 ||
        error("cross-term $(V[1, 2]) is too large for the atol probe")
    Vbump = copy(V)
    Vbump[1, 2] += 2e-8
    Vbump[2, 1] += 2e-8
    res_bump = compare_fit(fit, _at_compare_defaults(expected_from(Vbump, se)))
    @test res_bump.passed == false
    @test any(f -> startswith(f, "vcov[") && occursin("atol=1.0e-8", f), res_bump.failures)
    Vtiny = copy(V)
    Vtiny[1, 2] += 1e-9
    Vtiny[2, 1] += 1e-9
    @test compare_fit(fit, _at_compare_defaults(expected_from(Vtiny, se))).passed
    se_bump = copy(se)
    # A raw +2e-8 sits inside rtol when the SE is O(0.1). One percent is past
    # rtol_se = 1e-3, and the failure line still has to quote atol = 1e-8,
    # which is what shows the looser fixture override was not used.
    se_bump[order[1]] = se[order[1]] * 1.01 + 1e-6
    res_se = compare_fit(fit, _at_compare_defaults(expected_from(V, se_bump)))
    @test res_se.passed == false
    @test any(f -> startswith(f, "se[") && occursin("atol=1.0e-8", f), res_se.failures)
end

@testset "parity vcov/SE at compare.jl defaults" begin
    fixtures_root = joinpath(@__DIR__, "parity", "fixtures")
    cases = String[]
    for name in sort(readdir(fixtures_root))
        startswith(name, "_") && continue
        # Cross-package estimand, same exclusion as runparity.jl. Not a
        # stored vcov cell, and load_expected errors on the missing aic.
        name == "xfam-external-gllvm" && continue
        dir = joinpath(fixtures_root, name)
        isdir(dir) || continue
        isfile(joinpath(dir, "expected.toml")) || continue
        push!(cases, dir)
    end
    @test length(cases) == 18
    for dir in cases
        casename = basename(dir)
        @testset "$casename" begin
            expected = load_expected(dir)
            @test expected.vcov !== nothing
            strict = _at_compare_defaults(expected)
            for key in _VCOV_SE_TOL_KEYS
                @test !haskey(strict.tol, key)
            end
            result = _fit_and_compare(dir, casename, strict)
            if casename in _VCOV_HELD
                # Vcov-element misses stay unasserted. Anything else (coef,
                # loglik, aic, SE, a missing vcov name) still has to pass
                # at the bars this comparison already uses.
                other = filter(f -> !startswith(f, "vcov["), result.failures)
                if !isempty(other)
                    @error "Non-vcov parity failure for held cell `$casename`" failures = other
                end
                @test isempty(other)
            else
                if !result.passed
                    @error "Parity vcov/SE FAILED for `$casename`" failures = result.failures
                end
                @test result.passed
            end
        end
    end
end

end # module ParityVcovDefaults
