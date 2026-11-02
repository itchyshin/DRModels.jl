# input_validation.jl — argument and data checks run at the `drm` entry points.
#
# Each check here refuses an input that a fitter used to accept and then drop or
# misread without a word:
#
# * a formula for a parameter the family does not have (`nu ~ z` on Poisson) —
#   fitted without it (#1027, #939);
# * `offset(x)` outside the Poisson mean — estimated as a free slope (#938);
# * a `tree`, `K`, `A` or `coords` keyword, or `profile_ci = true`, with no
#   matching marker in the formula (fitted as the plain model) (#1010);
# * a formula column shorter or longer than the response — read past its end
#   under `@inbounds` (segfault, or a silently wrong `meta_V` fit; #1022);
# * phylo group labels that are not tip names — mapped to tips by first-seen
#   order when the level count happened to equal the tip count (#1026);
# * an asymmetric or non-finite bivariate `K` / `A` — one triangle read (#1020);
# * a negative or non-finite `meta_V` sampling variance on the no-random-effect route (#1014).
#
# Every message names the offending argument.

using StatsModels: StatsModels

# ── #1027: secondary parameters each family models ──────────────────────────
#
# The secondary formulas a family's `drm` method consumes (besides the mean).
# `nothing` means the family method already refuses unknown parts itself
# (Binomial, the truncated families) and is left alone. `bf` always adds
# `sigma ~ 1`, so a CONSTANT sigma formula is accepted for a family without
# sigma; only a sigma formula with covariates is refused there.
_family_dpars(::Any) = nothing
_family_dpars(::Gaussian) = (:sigma,)
_family_dpars(::LogNormal) = (:sigma,)
_family_dpars(::Gamma) = (:sigma,)
_family_dpars(::Beta) = (:sigma,)
_family_dpars(::BetaBinomial) = (:sigma,)
_family_dpars(::Student) = (:sigma, :nu)
_family_dpars(::SkewNormal) = (:sigma, :nu)
_family_dpars(::Tweedie) = (:sigma, :nu)
_family_dpars(::NegBinomial2) = (:sigma, :zi, :hu)
_family_dpars(::Poisson) = (:zi, :hu)
_family_dpars(::ZeroOneBeta) = (:sigma, :zoi, :coi)
_family_dpars(::CumulativeLogit) = ()

_family_label(fam) = string(nameof(typeof(fam)), "()")

function _require_family_dpars(f::DrmFormula, fam)
    allowed = _family_dpars(fam)
    allowed === nothing && return nothing
    for (p, rhs) in f.forms
        p === :mu && continue
        # `sd(g) ~ …` parts are stored as `sd_g` / `sdphy_g`; the #544 guard
        # (`_lss_only_gaussian_guard`) and the Gaussian LSS router own them.
        sp = String(p)
        (startswith(sp, "sd_") || startswith(sp, "sdphy_")) && continue
        p in allowed && continue
        p === :sigma && rhs isa ConstantTerm && continue
        has = isempty(allowed) ? "only the mean `mu`" :
            "`mu` and " * join(("`$a`" for a in allowed), ", ")
        what = p === :sigma ? "`sigma ~ …` formula with covariates" : "`$p ~ …` formula"
        throw(ArgumentError("drm: $(_family_label(fam)) has no `$p` parameter, so the " *
            "$what cannot be fitted and would be dropped. This family models $has. " *
            "Remove the `$p` formula or choose a family that has that parameter."))
    end
    return nothing
end

# ── #938: `offset(...)` outside the mean formula ─────────────────────────────
#
# `offset(x)` is an identity marker. Every univariate family extracts it from the
# MEAN formula and adds it to the linear predictor with its coefficient fixed at
# 1 (Poisson since #727/#941, the other families since #938), on the routes that
# can hold it fixed; every other route refuses it by name (`_refuse_offset_route`,
# and the `_design` backstop for anything left over). In any other formula
# (`sigma`, `nu`, `zi`, `hu`, ...) it is refused here, as drmTMB refuses it.
_has_offset_term(t) =
    (t isa Tuple || t isa AbstractVector) ? any(_has_offset_term, t) :
    t isa StatsModels.InteractionTerm ? any(_has_offset_term, t.terms) :   # `x * offset(z)`
    t isa FunctionTerm ? (t.f === offset || any(_has_offset_term, t.args)) : false

function _require_offset_supported(f::DrmFormula, fam)
    for (p, rhs) in f.forms
        _has_offset_term(rhs) || continue
        # The mean: each family's routes apply it (coefficient 1) or refuse it by name.
        p === :mu && continue
        throw(ArgumentError("drm: `offset(...)` in the `$p` formula is not supported for " *
            "$(_family_label(fam)); an offset enters only the mean formula, with its " *
            "coefficient fixed at 1 (drmTMB refuses an offset in `$p` too). Add the term " *
            "as an ordinary covariate if a free slope is intended."))
    end
    return nothing
end

# ── #1022 / #1013: columns the formula reads from `data` ─────────────────────
#
# Variables named by the formula, walking marker calls such as `(1 | g)`,
# `phylo(1 | g)`, `meta_V(v)` and `offset(x)`. The third argument of
# `temporal(1 | id, time, ar1)` is a structure keyword (`ar1`, `ou`,
# `homtoep`), not a data column, so it is skipped.
function _formula_data_symbols!(out::Vector{Symbol}, t)
    if t isa Tuple || t isa AbstractVector
        for s in t
            _formula_data_symbols!(out, s)
        end
    elseif t isa FunctionTerm
        args = t.args
        if t.f === temporal && length(args) == 3
            args = args[1:2]
        end
        for a in args
            _formula_data_symbols!(out, a)
        end
    elseif t isa StatsModels.AbstractTerm
        append!(out, StatsModels.termvars(t))
    end
    return out
end

function _formula_column_symbols(f::DrmFormula)
    cols = Symbol[f.response]
    f.response2 === nothing || push!(cols, f.response2)
    for (_, rhs) in f.forms
        _formula_data_symbols!(cols, rhs)
    end
    return unique(cols)
end

function _formula_column_symbols(f::BivariateDrmFormula)
    cols = Symbol[f.response1, f.response2]
    for (_, rhs) in f.forms
        _formula_data_symbols!(cols, rhs)
    end
    return unique(cols)
end

_formula_response_symbol(f::DrmFormula) = f.response
_formula_response_symbol(f::BivariateDrmFormula) = f.response1

# The column `name` of `data`, or `nothing` when `data` has no such column (a
# missing column is reported later by the design builder, with its own message).
function _maybe_table_column(data, name::Symbol)
    if data isa NamedTuple
        return haskey(data, name) ? getproperty(data, name) : nothing
    elseif data isa AbstractDict
        haskey(data, name) && return data[name]
        s = String(name)
        return haskey(data, s) ? data[s] : nothing
    end
    Tables.istable(data) || return nothing
    cols = Tables.columns(data)
    name in Tables.columnnames(cols) || return nothing
    return Tables.getcolumn(cols, name)
end

"""
    _require_formula_column_lengths(f, data; context = "drm")

Every column the formula reads from `data` (responses, predictors, grouping
factors, random-slope variables, `meta_V` variances, temporal time columns)
must have one value per row of the response. A hand-built `NamedTuple` can
hold vectors of different lengths, and several fitters index these columns
under `@inbounds` (#1022), so the mismatch is refused here, naming the column.
"""
function _require_formula_column_lengths(f::Union{DrmFormula,BivariateDrmFormula}, data;
                                         context::AbstractString = "drm")
    resp = _formula_response_symbol(f)
    ycol = _maybe_table_column(data, resp)
    ycol isa AbstractVector || return nothing
    n = length(ycol)
    for s in _formula_column_symbols(f)
        col = _maybe_table_column(data, s)
        col isa AbstractVector || continue
        length(col) == n ||
            throw(ArgumentError("$context: column `$s` has $(length(col)) value(s) but the " *
                "response `$resp` has $n; every column the formula uses must have one " *
                "value per row."))
    end
    return nothing
end

# ── #1010: structured-data keywords need a matching marker ───────────────────

function _structured_marker_kinds!(kinds::Set{Symbol}, t)
    if t isa Tuple || t isa AbstractVector
        for s in t
            _structured_marker_kinds!(kinds, s)
        end
    elseif t isa FunctionTerm
        t.f === phylo && push!(kinds, :phylo)
        t.f === relmat && push!(kinds, :relmat)
        t.f === animal && push!(kinds, :animal)
        t.f === spatial && push!(kinds, :spatial)
        for a in t.args
            _structured_marker_kinds!(kinds, a)
        end
    end
    return kinds
end

function _structured_marker_kinds(f::DrmFormula)
    kinds = Set{Symbol}()
    for (p, rhs) in f.forms
        startswith(String(p), "sdphy_") && push!(kinds, :phylo)   # sd(g, phylogenetic) ~ …
        _structured_marker_kinds!(kinds, rhs)
    end
    return kinds
end

"""
    _require_structured_keywords_used(f, fam; tree, K, A, coords)

A supplied `tree`, `K`, `A` or `coords` must have a formula marker that reads
it; otherwise the fitter drops it and returns the unstructured model (#1010).
`K` is read by `relmat(1 | g)` and, outside the Gaussian family, also by
`spatial(1 | g)` as a precomputed spatial covariance.
"""
function _require_structured_keywords_used(f::DrmFormula, fam; tree = nothing, K = nothing,
                                           A = nothing, coords = nothing)
    (tree === nothing && K === nothing && A === nothing && coords === nothing) && return nothing
    kinds = _structured_marker_kinds(f)
    unused(kw, marker) = throw(ArgumentError("drm: `$kw = …` was supplied but the formula " *
        "has no $marker term, so `$kw` would be ignored and an unstructured model fitted. " *
        "Add the marker to the formula or drop the `$kw` argument."))
    tree === nothing || :phylo in kinds ||
        unused("tree", "`phylo(1 | g)` (or `sd(g, phylogenetic) ~ …`)")
    if K !== nothing
        k_ok = :relmat in kinds || (!(fam isa Gaussian) && :spatial in kinds)
        k_ok || unused("K", fam isa Gaussian ? "`relmat(1 | g)`" :
                            "`relmat(1 | g)` or `spatial(1 | g)`")
    end
    A === nothing || :animal in kinds || unused("A", "`animal(1 | g)`")
    coords === nothing || :spatial in kinds || unused("coords", "`spatial(1 | g)`")
    return nothing
end

"""
    _require_profile_ci_used(profile_ci, structured_sigma)

`profile_ci = true` only precomputes the profile intervals of the Gaussian
`phylo(1 | g)`-on-`sigma` location-scale routes (`structured_sigma` is the
`(kind, group)` pair `_split_ranef` returns for the `sigma` formula). On every
other route the flag was dropped and a fit with no profile intervals came back
(#1010), so it is refused here like `tree`, `K`, `A` and `coords`.
"""
function _require_profile_ci_used(profile_ci::Bool, structured_sigma)
    profile_ci || return nothing
    (structured_sigma !== nothing && structured_sigma[1] === :phylo) && return nothing
    throw(ArgumentError("drm: `profile_ci = true` was supplied but the formula has no " *
        "`phylo(1 | g)` term on `sigma`, the only Gaussian route that precomputes profile " *
        "intervals, so `profile_ci` would be ignored and a fit with no profile intervals " *
        "returned. Add the marker to the `sigma` formula or drop the `profile_ci` argument; " *
        "`profile_result(fit)` and `confint(fit, method = :profile)` profile any other " *
        "fit after it is fitted."))
end

"""
    _drm_entry_guard(f, fam, data; tree, K, A, coords)

The input checks every univariate `drm(::DrmFormula, family)` method runs
first: family parameters (#1027), `offset(...)` placement (#938), formula
column lengths (#1022) and unused structured keywords (#1010).
"""
function _drm_entry_guard(f::DrmFormula, fam, data; tree = nothing, K = nothing,
                          A = nothing, coords = nothing)
    _require_family_dpars(f, fam)
    _require_offset_supported(f, fam)
    _require_formula_column_lengths(f, data)
    _require_structured_keywords_used(f, fam; tree = tree, K = K, A = A, coords = coords)
    return nothing
end

# ── #1026: phylo labels are matched by name, never by position ───────────────
#
# Called by the tip matchers once neither every label is a tip name nor every
# label an integer tip index. Labels that carry names (strings, symbols,
# categorical levels) are refused: mapping them to tips by first-seen order put
# species on the wrong tips without a warning whenever the level count equalled
# the tip count. Purely numeric labels (codes outside `1:n_tips`) carry no name
# to match, so the caller keeps its positional fallback for them.
function _refuse_unmatched_tip_labels(labels, leaf_names; context::AbstractString = "drm")
    all(l -> l isa Real, labels) && return nothing
    tips = Set(String.(leaf_names))
    keys_ = unique(string.(labels))
    unmatched = filter(k -> !(k in tips), keys_)
    show_some(v) = join(("\"$x\"" for x in first(v, 3)), ", ") * (length(v) > 3 ? ", …" : "")
    throw(ArgumentError("$context: phylo group labels must be the tree's tip names or " *
        "integer tip indices 1:$(length(leaf_names)). $(length(unmatched)) of " *
        "$(length(keys_)) label(s) are not tip names (e.g. $(show_some(unmatched))); " *
        "tree tips include $(show_some(collect(leaf_names))). Labels are matched by name, " *
        "not by order of appearance — rename the labels or the tree tips so they agree."))
end

# ── #1020: user covariance on the bivariate routes ───────────────────────────
#
# `isposdef(Symmetric(C))` reads one triangle: an asymmetric matrix, or a NaN in
# the ignored triangle, passed. Same rule as the univariate
# `_checked_relmat_symmetric`, plus an explicit finite check that names the cell.
function _require_symmetric_finite_covariance(C::AbstractMatrix, label::AbstractString)
    for I in CartesianIndices(C)
        isfinite(C[I]) ||
            throw(ArgumentError("drm: $label contains a non-finite value ($(C[I])) at row " *
                "$(I[1]), column $(I[2])."))
    end
    Cm = Matrix{Float64}(C)
    isapprox(Cm, Cm'; rtol = sqrt(eps(Float64))) ||
        throw(ArgumentError("drm: $label is not symmetric (outside a sqrt(eps) relative " *
            "tolerance); only one triangle would be used. Check how the matrix was built."))
    return Cm
end

# ── #1014: known sampling variances ─────────────────────────────────────────
function _require_meta_V_variances(vv, name)
    # Non-negative, like drmTMB (`validate_known_v_diag` refuses only negative and
    # non-finite values; a zero-variance study is legitimate). The fitted variance
    # is v + σ², so a zero v is numerically safe.
    for (i, v) in pairs(vv)
        (isfinite(v) && v >= 0) ||
            throw(ArgumentError("drm: `meta_V($name)` sampling variances must be finite and " *
                "non-negative; row $i has $v."))
    end
    return nothing
end

# ── #957: `A` matched to the data by label ──────────────────────────────────
#
# Rows and columns of the relatedness matrix `A` used to be taken as the levels
# of the `animal(1 | g)` group in order of first appearance, with no check: a
# matrix in sorted-id order (the natural export from a pedigree tool) gave a
# different, silently wrong fit. `A` is now matched by label when labels are
# given, either as a table whose column names are the labels or as a plain
# matrix plus `ids = …` (the label of each row). An unlabelled matrix keeps the
# old first-appearance convention and warns once per session.

_animal_group(f::DrmFormula) = _marker_group(f, animal)

function _marker_group(f::DrmFormula, marker)
    for (_, rhs) in f.forms
        g = _animal_group_term(rhs, marker)
        g === nothing || return g
    end
    return nothing
end

function _animal_group_term(t, marker = animal)
    if t isa Tuple || t isa AbstractVector
        for s in t
            g = _animal_group_term(s, marker)
            g === nothing || return g
        end
    elseif t isa FunctionTerm
        if t.f === marker
            bar = t.args[1]
            (bar isa FunctionTerm && length(bar.args) == 2 && bar.args[2] isa Term) &&
                return bar.args[2].sym
        end
        for a in t.args
            g = _animal_group_term(a, marker)
            g === nothing || return g
        end
    end
    return nothing
end

"""
    _resolve_relatedness_labels(f, data, A, ids) -> A′

Return `A` with its rows and columns in the order the fitters expect (the
levels of the `animal(1 | g)` group in order of first appearance), matching by
label when `A` is a table (column names are the labels) or `ids` is given.
Levels without a row in `A` are an `ArgumentError`; rows of `A` for labels not
in the data are dropped (the marginal of `A` over the observed labels).
"""
function _resolve_relatedness_labels(f::DrmFormula, data, A, ids; coords = nothing)
    if A === nothing
        (ids === nothing || coords !== nothing) ||
            throw(ArgumentError("drm: `ids = …` labels the rows of the relatedness matrix " *
                "`A` or of the site coordinates `coords`, but neither was supplied."))
        return nothing
    end
    labels_A, M = if ids !== nothing
        (collect(ids), A isa AbstractMatrix ? A : Tables.matrix(A))
    elseif !(A isa AbstractMatrix) && Tables.istable(A)
        (String.(collect(Tables.columnnames(Tables.columns(A)))), Tables.matrix(A))
    else
        (nothing, A)
    end
    grp = _animal_group(f)
    grp === nothing && return M      # no animal() term: the #1010 guard has already refused `A`
    if labels_A === nothing
        @warn "drm: `A` has no labels, so its rows and columns are taken to be the levels " *
              "of `$grp` in order of first appearance in the data. Pass `ids = …` (the " *
              "label of each row of `A`), or `A` as a table whose column names are the " *
              "labels, to match `A` by label instead." maxlog = 1
        return M
    end
    n = length(labels_A)
    size(M) == (n, n) ||
        throw(ArgumentError("drm: `A` is $(size(M, 1))×$(size(M, 2)) but has $n label(s); " *
            "give one label per row."))
    pos = Dict{String,Int}()
    for (i, l) in enumerate(labels_A)
        k = string(l)
        haskey(pos, k) &&
            throw(ArgumentError("drm: the label \"$k\" appears more than once in the labels of `A`."))
        pos[k] = i
    end
    levels = unique(collect(_table_column(data, grp)))   # `_group_index` order
    absent = [string(l) for l in levels if !haskey(pos, string(l))]
    isempty(absent) ||
        throw(ArgumentError("drm: $(length(absent)) level(s) of `$grp` have no row in `A` " *
            "(e.g. " * join(("\"$x\"" for x in first(absent, 3)), ", ") *
            (length(absent) > 3 ? ", …" : "") * "). Every level must be one of the labels of `A`."))
    perm = [pos[string(l)] for l in levels]
    return Matrix{Float64}(M)[perm, perm]
end

# ── #957 (spatial half): `coords` matched to the sites by label ────────────
#
# Row k of `coords` used to be the k-th level of the `spatial(1 | site)` group in
# first-appearance order, checked only by row count; a matrix sorted by site id
# gave a different likelihood and a collapsed range with `converged = true`.
# Same contract as `A`: labels come from `ids = …` (one per row of `coords`) or
# from a Dict keyed by site label; an unlabelled matrix keeps the old convention
# and warns once per session. `ids` labels `A` when the formula has an
# `animal()` term, otherwise `coords`.

const _COORDS_WARNED = Ref(false)

function _warn_unlabelled_coords(grp)
    _COORDS_WARNED[] && return nothing
    _COORDS_WARNED[] = true
    @warn "drm: `coords` has no labels, so its rows are taken to be the levels of `$grp` " *
          "in order of first appearance in the data. Pass `ids = …` (the site label of each " *
          "row of `coords`), or `coords` as a Dict keyed by site label, to match `coords` " *
          "by label instead."
    return nothing
end

"""
    _resolve_coords_labels(f, data, coords, ids) -> coords′

Return `coords` with its rows in the order the spatial fitters expect (the levels
of the `spatial(1 | g)` group in order of first appearance), matching by site
label when `coords` is a `Dict` (site label => coordinate) or `ids` labels its
rows. A level with no coordinates is an `ArgumentError`; coordinates for sites not
in the data are dropped.
"""
function _resolve_coords_labels(f::DrmFormula, data, coords, ids)
    coords === nothing && return nothing
    grp = _marker_group(f, spatial)
    grp === nothing && return coords          # no spatial() term: the entry guard handles `coords`
    labels_c, M = if coords isa AbstractDict
        ks = collect(keys(coords))
        rows = [v isa Real ? [Float64(v)] : Float64.(collect(v)) for v in values(coords)]
        length(unique(length.(rows))) <= 1 ||
            throw(ArgumentError("drm: the coordinates in `coords` have different lengths."))
        (ks, isempty(rows) ? zeros(0, 0) : permutedims(reduce(hcat, rows)))
    elseif ids !== nothing && _animal_group(f) === nothing
        (collect(ids), coords)
    else
        (nothing, coords)
    end
    if labels_c === nothing
        _warn_unlabelled_coords(grp)
        return coords
    end
    isvec = M isa AbstractVector
    n = length(labels_c)
    size(M, 1) == n ||
        throw(ArgumentError("drm: `coords` has $(size(M, 1)) row(s) but $n label(s); " *
            "give one label per row."))
    pos = Dict{String,Int}()
    for (i, l) in enumerate(labels_c)
        k = string(l)
        haskey(pos, k) &&
            throw(ArgumentError("drm: the label \"$k\" appears more than once in the labels of `coords`."))
        pos[k] = i
    end
    levels = unique(collect(_table_column(data, grp)))   # `_group_index` order
    absent = [string(l) for l in levels if !haskey(pos, string(l))]
    isempty(absent) ||
        throw(ArgumentError("drm: $(length(absent)) level(s) of `$grp` have no row in `coords` " *
            "(e.g. " * join(("\"$x\"" for x in first(absent, 3)), ", ") *
            (length(absent) > 3 ? ", …" : "") * "). Every level must be one of the labels of `coords`."))
    perm = [pos[string(l)] for l in levels]
    return isvec ? Vector{Float64}(M)[perm] : Matrix{Float64}(M)[perm, :]
end
