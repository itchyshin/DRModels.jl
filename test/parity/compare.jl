# compare.jl — pure numerical-parity comparison contract (no file I/O).
#
# This file is the heart of Workflow G (issue #17, deferred .jl code): it turns a
# DRModels.jl fit and a drmTMB reference (`ParityExpected`) into a pass/fail verdict
# with human-readable failure strings. It is deliberately I/O-free so it can be
# unit-tested in full without any fixtures on disk — `loadfixture.jl` owns the
# TOML/CSV parsing.
#
# IMPORTANT (honesty / license): this code does NOT itself contain or imply any
# drmTMB parity claim. It only *compares* a DRModels.jl fit to whatever expected
# numbers it is handed. Real drmTMB `expected.toml` fixtures are generated
# out-of-band by a maintainer with local R + drmTMB (see GENERATING.md); the
# always-on smoke test feeds it a DRModels.jl self-consistency expected instead.

"""
    ParityExpected

A drmTMB (or self-consistency) reference for one parity case. All fields are
plain numbers/strings parsed from `expected.toml` (see `loadfixture.jl`) — no
DRModels.jl objects — so this struct is the stable comparison contract.

Fields:
- `family::String` — family tag, e.g. `"gaussian"`.
- `coef::Dict{String,Float64}` — point estimates keyed by flat name
  `"<param>_<coefname>"`, e.g. `"mu_(Intercept)"`, `"sigma_x"`.
- `loglik::Float64` — reference maximised log-likelihood.
- `aic::Float64` — reference AIC (`-2·loglik + 2·df`).
- `df::Int` — number of estimated parameters.
- `n::Int` — number of observations.
- `vcov_order::Union{Nothing,Vector{String}}` — flat names giving the row/column
  order of `vcov` (nothing when no vcov is supplied).
- `vcov::Union{Nothing,Matrix{Float64}}` — reference covariance matrix in
  `vcov_order` order (nothing when not supplied).
- `tol::Dict{String,Float64}` — per-case tolerance overrides (`[tol]` block); any
  of `rtol_coef`, `atol_coef`, `rtol_vcov`, `atol_vcov`, `atol_loglik`,
  `atol_aic`, `rtol_se`, `atol_se`.
- `se::Dict{String,Float64}` — OPTIONAL per-coefficient Wald standard errors
  (`[se]` block), keyed by the same flat `"<param>_<coefname>"` names as `coef`.
  Empty when the fixture supplies none (the default; SE checks are then skipped).
  SE agreement between engines is NOT an interval-coverage claim.
- `se_not_comparable::Vector{String}` — names (same flat convention) whose SE
  must NOT be numerically compared, from the reserved `not_comparable` array key
  inside the `[se]` block. Use for parameters pinned at a parameter-space
  boundary in the reference fit (e.g. a variance component on DRModels.jl's
  `_LAPLACE_LOG_SD_FLOOR = log(1e-6)`, which drmTMB does not share): an SE
  taken at a constrained boundary optimum is not comparable to one taken at an
  interior optimum, and no tolerance absorbs that. Such names are reported in
  [`compare_se`](@ref)'s `skipped` output — declined, never silently passed.
"""
struct ParityExpected
    family::String
    coef::Dict{String,Float64}
    loglik::Float64
    aic::Float64
    df::Int
    n::Int
    vcov_order::Union{Nothing,Vector{String}}
    vcov::Union{Nothing,Matrix{Float64}}
    tol::Dict{String,Float64}
    # Optional group-level (location–scale) covariance reference: the grouping
    # factor name + any of "sd_mu", "sd_sigma", "cor" (in DRModels.jl's convention —
    # the generator applies the drmTMB→DRModels.jl reparameterisation, see GENERATING.md).
    ranef_group::Union{Nothing,String}
    ranef::Dict{String,Float64}
    # Optional per-coefficient standard errors ([se] block); empty ⇒ not checked.
    se::Dict{String,Float64}
    # Names whose SE is declared boundary-pinned / not comparable ([se] block's
    # reserved `not_comparable` array key); always skipped-and-reported.
    se_not_comparable::Vector{String}
end

# Convenience constructor: vcov, ranef, se + tol default to nothing/empty.
function ParityExpected(; family::AbstractString, coef::AbstractDict, loglik::Real,
        aic::Real, df::Integer, n::Integer,
        vcov_order = nothing, vcov = nothing, tol = Dict{String,Float64}(),
        ranef_group = nothing, ranef = Dict{String,Float64}(),
        se = Dict{String,Float64}(), se_not_comparable = String[])
    ParityExpected(String(family),
        Dict{String,Float64}(String(k) => Float64(v) for (k, v) in coef),
        Float64(loglik), Float64(aic), Int(df), Int(n),
        vcov_order === nothing ? nothing : Vector{String}(String.(vcov_order)),
        vcov === nothing ? nothing : Matrix{Float64}(vcov),
        Dict{String,Float64}(String(k) => Float64(v) for (k, v) in tol),
        ranef_group === nothing ? nothing : String(ranef_group),
        Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ranef),
        Dict{String,Float64}(String(k) => Float64(v) for (k, v) in se),
        Vector{String}(String.(se_not_comparable)))
end

"""
    drm_coef_named(fit) -> Dict{String,Float64}

Flatten a DRModels.jl fit's coefficients into the fixture's flat naming
`"<param>_<coefname>"`. This is the bridge between DRModels.jl's block layout
(`fit.blocks :: Vector{Pair{Symbol,UnitRange}}` + `fit.coefnames ::
Vector{Pair{Symbol,Vector{String}}}`) and the flat, order-independent name keys
in `expected.toml`.

For each block `param => range`, we look up the matching coefficient-name vector
in `fit.coefnames`, then zip those names with the `coef(fit)` slice for that
range, emitting `"\$(param)_\$(name)" => estimate`. So a Gaussian location–scale
fit with `mu ~ 1 + x`, `sigma ~ 1 + x` yields keys
`"mu_(Intercept)"`, `"mu_x"`, `"sigma_(Intercept)"`, `"sigma_x"`.
"""
function drm_coef_named(fit)::Dict{String,Float64}
    θ = coef(fit)
    namemap = Dict(p => ns for (p, ns) in fit.coefnames)
    out = Dict{String,Float64}()
    for (param, r) in fit.blocks
        haskey(namemap, param) || continue   # blocks without named coefs (rare) are skipped
        names = namemap[param]
        slice = θ[r]
        length(names) == length(slice) || error(
            "drm_coef_named: name/coef length mismatch for `$param` " *
            "($(length(names)) names vs $(length(slice)) coefs)")
        for (nm, est) in zip(names, slice)
            out["$(param)_$(nm)"] = est
        end
    end
    return out
end

"""
    drm_se_named(fit) -> Dict{String,Float64}

Flatten a DRModels.jl fit's per-coefficient Wald standard errors — `sqrt` of the
diagonal of `vcov(fit)` — into the fixture's flat naming
`"<param>_<coefname>"`, using the same block walk as [`drm_coef_named`](@ref)
(vcov rows/cols follow θ's ordering). A non-positive diagonal entry yields
`NaN` for that name. Throws if `vcov(fit)` is unavailable; callers that must
not throw (e.g. [`compare_se`](@ref)) wrap this in a `try`.
"""
function drm_se_named(fit)::Dict{String,Float64}
    V = vcov(fit)
    V === nothing && error("drm_se_named: vcov(fit) returned nothing")
    namemap = Dict(p => ns for (p, ns) in fit.coefnames)
    out = Dict{String,Float64}()
    i = 0
    for (param, r) in fit.blocks
        haskey(namemap, param) || continue
        for nm in namemap[param]
            i += 1
            d = V[i, i]
            out["$(param)_$(nm)"] = d > 0 ? sqrt(d) : NaN
        end
    end
    return out
end

# Resolve an effective tolerance: per-case override (`[tol]`) wins over default.
_tol(exp::ParityExpected, key::AbstractString, default::Float64) =
    get(exp.tol, String(key), default)

# Scalar within-tolerance check (relative OR absolute), mirroring `isapprox`.
_within(a, b, rtol, atol) = abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))

"""
    compare_fit(fit, expected::ParityExpected;
                rtol_coef=1e-4, atol_coef=1e-6,
                rtol_vcov=1e-3, atol_vcov=1e-8, atol_loglik=1e-4) -> (passed, failures)

Compare a DRModels.jl `fit` against a `ParityExpected` reference under the README's
tolerance table. Returns a NamedTuple `(passed::Bool, failures::Vector{String})`
— it NEVER throws on a numerical mismatch; instead it accumulates a
human-readable line per failure of the form
`"<case quantity>: drmTMB=… DRModels.jl=… |Δ|=… > tol"`.

Checks, in order:
1. metadata — `n` and `df` must match the fit.
2. coef — every expected coef name must exist in `drm_coef_named(fit)` and be
   within (`rtol_coef`, `atol_coef`). A missing name is itself a failure.
3. loglik — scalar, within `atol_loglik`.
4. aic — scalar (derived `-2·loglik + 2·df`), within `atol_loglik` by default
   (override with `[tol] atol_aic`).
5. vcov — if `expected.vcov` is supplied, reorder the fit's `vcov(fit)` to
   `expected.vcov_order` (using `drm_coef_named`'s ordering of the fit) and
   compare element-wise within (`rtol_vcov`, `atol_vcov`).
6. ranef — if `expected.ranef_group` is supplied, compare the named group-level
   covariance summaries within (`rtol_ranef`, `atol_ranef`).
7. se — if `expected.se` is non-empty, compare per-coefficient standard errors
   (see [`compare_se`](@ref)) within (`rtol_se`, `atol_se`; defaults 1e-3,
   1e-8). Skipped entirely for the (default) empty `[se]` block; names in
   `expected.se_not_comparable` (boundary-pinned in the reference fit) are
   never numerically compared.

Per-case overrides from the fixture's `[tol]` block take precedence over the
keyword defaults (keys: `rtol_coef`, `atol_coef`, `rtol_vcov`, `atol_loglik`,
`atol_vcov`, `atol_aic`, `rtol_se`, `atol_se`).
"""
function compare_fit(fit, expected::ParityExpected;
        rtol_coef::Real = 1e-4, atol_coef::Real = 1e-6,
        rtol_vcov::Real = 1e-3, atol_vcov::Real = 1e-8,
        atol_loglik::Real = 1e-4)

    failures = String[]

    rc = _tol(expected, "rtol_coef", Float64(rtol_coef))
    ac = _tol(expected, "atol_coef", Float64(atol_coef))
    rv = _tol(expected, "rtol_vcov", Float64(rtol_vcov))
    av = _tol(expected, "atol_vcov", Float64(atol_vcov))
    al = _tol(expected, "atol_loglik", Float64(atol_loglik))
    aa = _tol(expected, "atol_aic", al)

    got = drm_coef_named(fit)

    # 1. metadata.
    nfit = nobs(fit)
    nfit == expected.n ||
        push!(failures, "nobs: drmTMB=$(expected.n) DRModels.jl=$(nfit)")
    dfit = dof(fit)
    dfit == expected.df ||
        push!(failures, "df: drmTMB=$(expected.df) DRModels.jl=$(dfit)")

    # 2. coefficients — name-matched.
    for name in sort!(collect(keys(expected.coef)))
        want = expected.coef[name]
        if !haskey(got, name)
            push!(failures, "coef[$name]: expected name absent from DRModels.jl fit " *
                "(have: $(join(sort!(collect(keys(got))), ", ")))")
            continue
        end
        have = got[name]
        if !_within(want, have, rc, ac)
            push!(failures, "coef[$name]: drmTMB=$(want) DRModels.jl=$(have) " *
                "|Δ|=$(abs(want - have)) > (rtol=$(rc), atol=$(ac))")
        end
    end

    # 3. loglik.
    llh = loglik(fit)
    if !_within(expected.loglik, llh, 0.0, al)
        push!(failures, "loglik: drmTMB=$(expected.loglik) DRModels.jl=$(llh) " *
            "|Δ|=$(abs(expected.loglik - llh)) > atol=$(al)")
    end

    # 4. aic (derived from loglik + df).
    aic_fit = -2 * llh + 2 * dof(fit)
    if !_within(expected.aic, aic_fit, 0.0, aa)
        push!(failures, "aic: drmTMB=$(expected.aic) DRModels.jl=$(aic_fit) " *
            "|Δ|=$(abs(expected.aic - aic_fit)) > atol=$(aa)")
    end

    # 5. vcov (optional) — reorder fit's vcov to the expected name order.
    if expected.vcov !== nothing && expected.vcov_order !== nothing
        order = expected.vcov_order
        # Build the fit's flat-name → vcov-index map by walking blocks in order
        # (vcov rows/cols follow θ's ordering, same as drm_coef_named).
        fitnames = String[]
        namemap = Dict(p => ns for (p, ns) in fit.coefnames)
        for (param, r) in fit.blocks
            haskey(namemap, param) || continue
            for nm in namemap[param]
                push!(fitnames, "$(param)_$(nm)")
            end
        end
        idx = Dict(nm => i for (i, nm) in enumerate(fitnames))
        Vfit = vcov(fit)
        missing_names = [nm for nm in order if !haskey(idx, nm)]
        if !isempty(missing_names)
            push!(failures, "vcov: expected order names absent from DRModels.jl fit: " *
                join(missing_names, ", "))
        else
            perm = [idx[nm] for nm in order]
            Vp = Vfit[perm, perm]
            k = length(order)
            if size(expected.vcov) != (k, k)
                push!(failures, "vcov: expected matrix is $(size(expected.vcov)) " *
                    "but order has $k names")
            else
                for i in 1:k, j in 1:k
                    want = expected.vcov[i, j]
                    have = Vp[i, j]
                    if !_within(want, have, rv, av)
                        push!(failures, "vcov[$(order[i]),$(order[j])]: " *
                            "drmTMB=$(want) DRModels.jl=$(have) " *
                            "|Δ|=$(abs(want - have)) > (rtol=$(rv), atol=$(av))")
                    end
                end
            end
        end
    end

    # 6. group-level covariance (optional) — drmTMB VarCorr (reparam'd to DRModels.jl
    # convention by the generator) vs DRModels.jl's `vc(fit)` for the location–scale Λ.
    if expected.ranef_group !== nothing
        rr = _tol(expected, "rtol_ranef", 1e-3)
        ar = _tol(expected, "atol_ranef", 1e-6)
        V = vc(fit)
        gkey = Symbol(expected.ranef_group)
        if !haskey(V, gkey)
            push!(failures, "ranef[$(expected.ranef_group)]: group absent from DRModels.jl fit " *
                "(have: $(join(string.(keys(V)), ", ")))")
        else
            Σ = V[gkey]
            sd_mu = sqrt(Σ[1, 1]); sd_sigma = sqrt(Σ[2, 2])
            cor = Σ[1, 2] / (sd_mu * sd_sigma)
            got_re = Dict("sd_mu" => sd_mu, "sd_sigma" => sd_sigma, "cor" => cor)
            for key in sort!(collect(keys(expected.ranef)))
                want = expected.ranef[key]
                if !haskey(got_re, key)
                    push!(failures, "ranef[$key]: unknown key (expected one of sd_mu, sd_sigma, cor)")
                    continue
                end
                have = got_re[key]
                if !_within(want, have, rr, ar)
                    push!(failures, "ranef[$key]: drmTMB=$(want) DRModels.jl=$(have) " *
                        "|Δ|=$(abs(want - have)) > (rtol=$(rr), atol=$(ar))")
                end
            end
        end
    end

    # 7. per-coefficient standard errors (optional) — see compare_se. Default
    # (rtol_se, atol_se) live in _compare_se!; per-case [tol] overrides apply.
    # Fixture-declared `not_comparable` names are skipped (the fixture itself is
    # the visible record of that decision).
    # `se_skipped` is a REAL vector, not a throwaway. Passing `String[]` here
    # discarded every declined name, so a fixture that declared all of its SE
    # names `not_comparable` would report `passed = true` having compared
    # nothing, with no trace of it in the result — the exact "silent pass" this
    # struct's docstring promises does not happen. Route 1 is the path
    # `runparity.jl` uses, so that promise has to hold here, not only in
    # `compare_se`.
    se_skipped = String[]
    if !isempty(expected.se)
        got_se = try
            drm_se_named(fit)
        catch err
            push!(failures, "se: standard errors unavailable from DRModels.jl fit " *
                "($(sprint(showerror, err)))")
            nothing
        end
        got_se === nothing ||
            _compare_se!(failures, se_skipped, got_se, expected)
    end

    # `skipped` is additive: existing callers destructure `.passed`/`.failures`
    # and are unaffected.
    return (passed = isempty(failures), failures = failures, skipped = se_skipped)
end

# Shared SE-comparison core: name-matched per-coefficient standard errors, with
# the same missing-name and tolerance-reporting conventions as the coef check.
# Mutates `failures` and `skipped`; returns nothing. Effective tolerances
# resolve `[tol]` overrides (`rtol_se`, `atol_se`) over the keyword defaults.
#
# Boundary discipline: a name in `expected.se_not_comparable` (fixture-declared,
# reference fit on a boundary) or in `boundary` (caller-detected, DRModels.jl fit on
# a boundary — e.g. a log-SD within eps of `DRModels._LAPLACE_LOG_SD_FLOOR`) is
# NEVER numerically compared: its SE comes from a constrained boundary optimum
# and is not the same quantity as an interior-optimum SE. It is recorded in
# `skipped` as "name (reason)" so declining is visible, not silent.
function _compare_se!(failures::Vector{String}, skipped::Vector{String},
        got::AbstractDict, expected::ParityExpected;
        rtol_se::Real = 1e-3, atol_se::Real = 1e-8,
        boundary = String[])
    rs = _tol(expected, "rtol_se", Float64(rtol_se))
    as = _tol(expected, "atol_se", Float64(atol_se))
    declared = Set(String.(expected.se_not_comparable))
    detected = Set(String.(boundary))
    # Names declared not-comparable but given no [se] value are still recorded.
    for name in sort!(collect(setdiff(declared, keys(expected.se))))
        push!(skipped, "$name (fixture: not_comparable)")
    end
    for name in sort!(collect(keys(expected.se)))
        if name in declared
            push!(skipped, "$name (fixture: not_comparable)")
            continue
        end
        if name in detected
            push!(skipped, "$name (DRModels.jl fit: boundary-pinned)")
            continue
        end
        want = expected.se[name]
        if !haskey(got, name)
            push!(failures, "se[$name]: expected name absent " *
                "(have: $(join(sort!(collect(keys(got))), ", ")))")
            continue
        end
        have = Float64(got[name])
        if !isfinite(have)
            push!(failures, "se[$name]: drmTMB=$(want) DRModels.jl=$(have) " *
                "(non-finite — possible boundary-pinned parameter; if so, " *
                "declare it in [se] not_comparable)")
            continue
        end
        if !_within(want, have, rs, as)
            push!(failures, "se[$name]: drmTMB=$(want) DRModels.jl=$(have) " *
                "|Δ|=$(abs(want - have)) > (rtol=$(rs), atol=$(as))")
        end
    end
    return nothing
end

"""
    compare_se(fit, expected::ParityExpected;
               rtol_se=1e-3, atol_se=1e-8, boundary=String[])
        -> (passed, failures, skipped)
    compare_se(se::AbstractDict, expected::ParityExpected; ...)
        -> (passed, failures, skipped)

Compare per-coefficient Wald STANDARD ERRORS against the fixture's optional
`[se]` block (flat `"<param>_<coefname>"` keys, exactly like `[coef]`). The
first method derives the fit's SEs via [`drm_se_named`](@ref); the second takes
an already-flattened name → SE dictionary (e.g. from a `drm_bridge` payload).
Returns `(passed::Bool, failures::Vector{String}, skipped::Vector{String})` and
never throws on a numerical mismatch. An `expected` with an empty `se`
dictionary passes trivially — the axis is simply unmeasured for that fixture.

Boundary discipline: an SE taken at a constrained boundary optimum (a variance
component pinned at DRModels.jl's `_LAPLACE_LOG_SD_FLOOR = log(1e-6)`, a floor
drmTMB does not share) is not the same quantity as an interior-optimum SE, so
such names are never numerically compared — no tolerance absorbs a
different-kind-of-point. Two routes mark them: the fixture's `[se]` reserved
`not_comparable` array key (reference-fit side, → `expected.se_not_comparable`)
and the `boundary` keyword (DRModels.jl-fit side; boundary detection is
family-specific, so the caller that can probe θ against the floor supplies the
flat names). Both land in `skipped` with their reason — "we compared and they
agreed" and "we declined to compare" stay distinguishable downstream.

Tolerance rationale (defaults `rtol_se = 1e-3`, `atol_se = 1e-8`): SEs are
second-order quantities — square roots of the diagonal of an inverse observed
information — so they inherit half the relative error of the `vcov` entries
they come from; `rtol_se = 1e-3` is the same order as `rtol_vcov` and ~500×
looser than the cross-engine agreement measured on 2026-08-24
(`docs/dev-log/evidence/parity-se.tsv`, max relative diff 2.2e-6), while still
rejecting any genuine information-matrix discrepancy (observed vs expected
information differs at the percent level). Override per-case via `[tol]`
`rtol_se` / `atol_se`.

NOTE (honesty): SE agreement between two engines is NOT interval coverage.
This check must never be cited as a coverage claim; `interval_status`
assertions elsewhere in the test suite stay authoritative.
"""
function compare_se(fit, expected::ParityExpected;
        rtol_se::Real = 1e-3, atol_se::Real = 1e-8, boundary = String[])
    failures = String[]
    skipped = String[]
    if !isempty(expected.se)
        got = try
            drm_se_named(fit)
        catch err
            push!(failures, "se: standard errors unavailable from DRModels.jl fit " *
                "($(sprint(showerror, err)))")
            nothing
        end
        got === nothing || _compare_se!(failures, skipped, got, expected;
            rtol_se = rtol_se, atol_se = atol_se, boundary = boundary)
    end
    return (passed = isempty(failures), failures = failures, skipped = skipped)
end

function compare_se(se::AbstractDict, expected::ParityExpected;
        rtol_se::Real = 1e-3, atol_se::Real = 1e-8, boundary = String[])
    failures = String[]
    skipped = String[]
    got = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in pairs(se))
    _compare_se!(failures, skipped, got, expected;
        rtol_se = rtol_se, atol_se = atol_se, boundary = boundary)
    return (passed = isempty(failures), failures = failures, skipped = skipped)
end

"""
    compare_bridge(out, expected::ParityExpected;
                   rtol_coef=1e-4, atol_coef=1e-6,
                   rtol_vcov=1e-3, atol_vcov=1e-8, atol_loglik=1e-4)
        -> (passed, failures)

Compare a flattened `drm_bridge` dictionary (`out`) against a
[`ParityExpected`](@ref) reference under the same tolerance contract as
[`compare_fit`](@ref).

`out` must supply at least `"coef"` (name → Float64), `"loglik"`, `"nobs"`,
and `"df"`. Optional `"aic"` / `"vcov"` (+ `"vcov_names"` or `"coef_names"` for
row order) are checked when present on both sides. When the fixture supplies an
`[se]` block, per-coefficient standard errors are checked too — from an `"se"`
dictionary in `out` if present, otherwise derived from the diagonal of
`out["vcov"]` (see [`compare_se`](@ref) for the tolerance contract). This is
the marshalling-path parity gate for R callers (`engine = "julia"`); it never
throws on a numerical mismatch.
"""
function compare_bridge(out::AbstractDict, expected::ParityExpected;
        rtol_coef::Real = 1e-4, atol_coef::Real = 1e-6,
        rtol_vcov::Real = 1e-3, atol_vcov::Real = 1e-8,
        atol_loglik::Real = 1e-4)

    failures = String[]

    rc = _tol(expected, "rtol_coef", Float64(rtol_coef))
    ac = _tol(expected, "atol_coef", Float64(atol_coef))
    rv = _tol(expected, "rtol_vcov", Float64(rtol_vcov))
    av = _tol(expected, "atol_vcov", Float64(atol_vcov))
    al = _tol(expected, "atol_loglik", Float64(atol_loglik))
    aa = _tol(expected, "atol_aic", al)

    coef_raw = get(out, "coef", nothing)
    coef_raw === nothing &&
        return (passed = false, failures = ["bridge out missing \"coef\""])
    got = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in pairs(coef_raw))

    nfit = Int(out["nobs"])
    nfit == expected.n ||
        push!(failures, "nobs: drmTMB=$(expected.n) DRModels.jl=$(nfit)")
    dfit = Int(out["df"])
    dfit == expected.df ||
        push!(failures, "df: drmTMB=$(expected.df) DRModels.jl=$(dfit)")

    for name in sort!(collect(keys(expected.coef)))
        want = expected.coef[name]
        if !haskey(got, name)
            push!(failures, "coef[$name]: expected name absent from drm_bridge out " *
                "(have: $(join(sort!(collect(keys(got))), ", ")))")
            continue
        end
        have = got[name]
        if !_within(want, have, rc, ac)
            push!(failures, "coef[$name]: drmTMB=$(want) DRModels.jl=$(have) " *
                "|Δ|=$(abs(want - have)) > (rtol=$(rc), atol=$(ac))")
        end
    end

    llh = Float64(out["loglik"])
    if !_within(expected.loglik, llh, 0.0, al)
        push!(failures, "loglik: drmTMB=$(expected.loglik) DRModels.jl=$(llh) " *
            "|Δ|=$(abs(expected.loglik - llh)) > atol=$(al)")
    end

    if haskey(out, "aic")
        aic_fit = Float64(out["aic"])
        if !_within(expected.aic, aic_fit, 0.0, aa)
            push!(failures, "aic: drmTMB=$(expected.aic) DRModels.jl=$(aic_fit) " *
                "|Δ|=$(abs(expected.aic - aic_fit)) > atol=$(aa)")
        end
    end

    if expected.vcov !== nothing && expected.vcov_order !== nothing && haskey(out, "vcov")
        order = expected.vcov_order
        fitnames = if haskey(out, "vcov_names")
            String[String(nm) for nm in out["vcov_names"]]
        elseif haskey(out, "coef_names")
            String[String(nm) for nm in out["coef_names"]]
        else
            # Fall back to Dict iteration order — unstable; prefer explicit names.
            sort!(collect(keys(got)))
        end
        idx = Dict(nm => i for (i, nm) in enumerate(fitnames))
        Vfit = Matrix{Float64}(out["vcov"])
        missing_names = [nm for nm in order if !haskey(idx, nm)]
        if !isempty(missing_names)
            push!(failures, "vcov: expected order names absent from drm_bridge out: " *
                join(missing_names, ", "))
        else
            perm = [idx[nm] for nm in order]
            Vp = Vfit[perm, perm]
            k = length(order)
            if size(expected.vcov) != (k, k)
                push!(failures, "vcov: expected matrix is $(size(expected.vcov)) " *
                    "but order has $k names")
            else
                for i in 1:k, j in 1:k
                    want = expected.vcov[i, j]
                    have = Vp[i, j]
                    if !_within(want, have, rv, av)
                        push!(failures, "vcov[$(order[i]),$(order[j])]: " *
                            "drmTMB=$(want) DRModels.jl=$(have) " *
                            "|Δ|=$(abs(want - have)) > (rtol=$(rv), atol=$(av))")
                    end
                end
            end
        end
    end

    # per-coefficient standard errors (optional [se] block) — prefer an explicit
    # "se" dictionary in the payload; else derive from the diagonal of "vcov".
    if !isempty(expected.se)
        if haskey(out, "se")
            got_se = Dict{String,Float64}(
                String(k) => Float64(v) for (k, v) in pairs(out["se"]))
            _compare_se!(failures, String[], got_se, expected)
        elseif haskey(out, "vcov")
            senames = if haskey(out, "vcov_names")
                String[String(nm) for nm in out["vcov_names"]]
            elseif haskey(out, "coef_names")
                String[String(nm) for nm in out["coef_names"]]
            else
                String[]
            end
            if isempty(senames)
                push!(failures, "se: bridge out has \"vcov\" but no " *
                    "\"vcov_names\"/\"coef_names\" to name its diagonal")
            else
                Vfit = Matrix{Float64}(out["vcov"])
                got_se = Dict{String,Float64}(
                    nm => (Vfit[i, i] > 0 ? sqrt(Vfit[i, i]) : NaN)
                    for (i, nm) in enumerate(senames))
                _compare_se!(failures, String[], got_se, expected)
            end
        else
            push!(failures, "se: fixture supplies [se] but bridge out has " *
                "neither \"se\" nor \"vcov\"")
        end
    end

    return (passed = isempty(failures), failures = failures)
end
