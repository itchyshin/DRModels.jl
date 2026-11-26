# Always-on Workflow G guard: every in-cohort parity fixture that stores [vcov]
# (and [se] when present) must agree with drmTMB on those axes at compare.jl's
# keyword defaults — rtol_vcov = 1e-3, atol_vcov = 1e-8, rtol_se = 1e-3,
# atol_se = 1e-8 — without the fixture's rtol_vcov / atol_vcov / rtol_se /
# atol_se overrides. Other [tol] keys (e.g. atol_coef, atol_loglik) stay in
# force so this lane does not duplicate the gated suite's coef/loglik contract.
# refs #1503

using DRModels
using Test
using TOML

include(joinpath(@__DIR__, "parity", "compare.jl"))
include(joinpath(@__DIR__, "parity", "loadfixture.jl"))

const _VCOV_SE_TOL_KEYS = ("rtol_vcov", "atol_vcov", "rtol_se", "atol_se")
# Measured outside default atol_vcov = 1e-8; fixture [tol] atol_vcov = 3e-8 kept for gated suite.
const _VCOV_DEFAULT_BROKEN = Set(["meta-analysis-V", "robust-student"])

function _strip_vcov_se_tol(exp::ParityExpected)
    tol = copy(exp.tol)
    for k in _VCOV_SE_TOL_KEYS
        delete!(tol, String(k))
    end
    return ParityExpected(;
        family = exp.family, coef = exp.coef, loglik = exp.loglik, aic = exp.aic,
        df = exp.df, n = exp.n, vcov_order = exp.vcov_order, vcov = exp.vcov,
        tol = tol, ranef_group = exp.ranef_group, ranef = exp.ranef,
        se = exp.se, se_not_comparable = exp.se_not_comparable)
end

function _vcov_se_failures(failures::AbstractVector{<:AbstractString})
    filter(f -> startswith(f, "vcov") || startswith(f, "se["), failures)
end

# --- fitting helpers (mirrors test/parity/runparity.jl, no gated side effects) ---
function _defaults_parity_family(family::AbstractString)
    fam = lowercase(strip(family))
    fam == "gaussian" && return Gaussian()
    fam == "biv_gaussian" && return Gaussian()
    fam == "gaussian_bivariate" && return Gaussian()
    fam == "student" && return Student()
    fam == "nbinom2" && return NegBinomial2()
    fam == "beta" && return DRModels.Beta()
    fam == "poisson" && return DRModels.Poisson()
    fam == "gamma" && return DRModels.Gamma()
    fam == "binomial" && return DRModels.Binomial()
    fam == "lognormal" && return DRModels.LogNormal()
    return nothing
end

function _defaults_formula_from_expr(expr)
    (expr isa Expr && expr.head === :call && expr.args[1] === :~) || return nothing
    return eval(Expr(:macrocall, Symbol("@formula"), LineNumberNode(0), expr))
end

function _defaults_parse_formula_part(part::AbstractString)
    expr = Meta.parse(part)
    if expr isa Expr && expr.head === :(=)
        length(expr.args) == 2 || return nothing
        key = expr.args[1]
        key isa Symbol || return nothing
        form = _defaults_formula_from_expr(expr.args[2])
        form === nothing && return nothing
        return key => form
    end
    form = _defaults_formula_from_expr(expr)
    form === nothing && return nothing
    return nothing => form
end

const _DEFAULTS_BIVARIATE_KEYS = Set((:mu1, :mu2, :sigma1, :sigma2, :rho12))

function _defaults_parity_formula(formula_text::AbstractString, family::AbstractString)
    parts = filter(!isempty, strip.(split(formula_text, ';')))
    isempty(parts) && return nothing
    parsed = map(_defaults_parse_formula_part, parts)
    any(isnothing, parsed) && return nothing
    keyed = Dict{Symbol,Any}()
    positional = Any[]
    for item in parsed
        key, form = item
        if key === nothing
            push!(positional, form)
        else
            keyed[key] = form
        end
    end
    if any(k -> k in _DEFAULTS_BIVARIATE_KEYS, keys(keyed))
        (isempty(positional) && haskey(keyed, :mu1) && haskey(keyed, :mu2)) || return nothing
        return bf(; mu1 = keyed[:mu1], mu2 = keyed[:mu2],
            sigma1 = get(keyed, :sigma1, nothing), sigma2 = get(keyed, :sigma2, nothing),
            rho12 = get(keyed, :rho12, nothing))
    end
    isempty(keyed) || return nothing
    return bf(positional...)
end

function _defaults_compare_native(dir::AbstractString)
    expected_raw = load_expected(dir)
    expected_raw.vcov === nothing && return nothing, nothing
    casename = basename(dir)
    casename == "xfam-external-gllvm" && return nothing, nothing

    fam = _defaults_parity_family(expected_raw.family)
    fam === nothing && return :skip_family, casename
    fit_meta = TOML.parsefile(joinpath(dir, "expected.toml"))["fit"]
    formula_text = String(get(fit_meta, "formula", ""))
    bundle = _defaults_parity_formula(formula_text, expected_raw.family)
    bundle === nothing && return :skip_formula, casename
    data = load_data(dir)
    fit = drm(bundle, fam; data = data)
    expected = _strip_vcov_se_tol(expected_raw)
    result = compare_fit(fit, expected)
    return result, casename
end

function _defaults_compare_bridge(dir::AbstractString)
    expected_raw = load_expected(dir)
    expected_raw.vcov === nothing && return nothing, nothing
    casename = basename(dir)
    fit_meta = TOML.parsefile(joinpath(dir, "expected.toml"))["fit"]
    formula_text = String(get(fit_meta, "formula", ""))
    family = String(get(fit_meta, "family", expected_raw.family))
    data = load_data(dir)
    out = drm_bridge(; formula = formula_text, family = family, data = data)
    expected = _strip_vcov_se_tol(expected_raw)
    result = compare_bridge(out, expected)
    return result, casename
end

@testset "parity vcov / SE at compare.jl defaults (refs #1503)" begin
    fixtures_root = joinpath(@__DIR__, "parity", "fixtures")
    cases = String[]
    for name in sort(readdir(fixtures_root))
        startswith(name, "_") && continue
        dir = joinpath(fixtures_root, name)
        isdir(dir) || continue
        isfile(joinpath(dir, "expected.toml")) || continue
        push!(cases, dir)
    end

    for dir in cases
        casename = basename(dir)
        @testset "$casename" begin
            if casename == "xfam-external-gllvm"
                @test_skip "out of Workflow G / #370 cohort (cross-package estimand)"
                continue
            end
            tpath = joinpath(dir, "expected.toml")
            if !occursin("[vcov]", read(tpath, String))
                @test_skip "fixture has no [vcov]"
                continue
            end
            if startswith(casename, "bridge-")
                outcome, _ = _defaults_compare_bridge(dir)
            else
                outcome, _ = _defaults_compare_native(dir)
            end
            if outcome === nothing
                @test_skip "parity runner cannot fit this cell"
                continue
            end
            if outcome in (:skip_family, :skip_formula)
                @test_skip outcome
                continue
            end
            result = outcome
            axis_failures = _vcov_se_failures(result.failures)
            has_se = !isempty(load_expected(dir).se)

            @testset "vcov (rtol_vcov=1e-3, atol_vcov=1e-8, no fixture override)" begin
                if casename in _VCOV_DEFAULT_BROKEN
                    # meta-analysis-V: cross-term |Δ|≈1.46e-8; robust-student: mu_x×nu |Δ|≈2.04e-8
                    @test_broken isempty(axis_failures)
                    # Still record which vcov entries fail so CI output stays actionable.
                    vcov_only = filter(f -> startswith(f, "vcov"), axis_failures)
                    @test !isempty(vcov_only)
                else
                    vcov_only = filter(f -> startswith(f, "vcov"), axis_failures)
                    if !isempty(vcov_only)
                        @error "vcov default-bar failure" case = casename failures = vcov_only
                    end
                    @test isempty(vcov_only)
                end
            end

            if has_se
                @testset "SE (rtol_se=1e-3, atol_se=1e-8, no fixture override)" begin
                    se_only = filter(f -> startswith(f, "se["), axis_failures)
                    if !isempty(se_only)
                        @error "SE default-bar failure" case = casename failures = se_only
                    end
                    @test isempty(se_only)
                end
            end
        end
    end
end