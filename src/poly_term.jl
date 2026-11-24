# poly_term.jl -- `poly(x, d)` in a Julia `@formula`: R's orthogonal polynomial basis.
#
# `@formula(y ~ poly(x1, 2))` used to fail with `UndefVarError: poly` raised by the macro (the
# function name is looked up in the caller's scope) before DRModels saw the model. `poly` is now
# exported, so the term parses, and it is a STATEFUL formula term like R's: every fit computes the
# basis from its own data and stores it in the fit (as `alpha` and `norm2`, R's own names), so
# `predict(fit, newdata)` rebuilds that fit's TRAINING basis at the new rows instead of a basis of
# the new rows. A `bf(...)` object holds no basis: reusing it for a second fit gives the same result
# as a fresh one, as R recomputes `poly` on every call.
#
# Basis (R's `stats::poly(x, degree)`, `raw = FALSE`): three-term recurrence on the centred data,
#   Z0 = 1,  Z1 = x - alpha1,  Z_{i+1} = (x - alpha_i) Z_i - (norm2_{i+1} / norm2_i) Z_{i-1},
#   alpha_i = sum(x Z_i^2) / sum(Z_i^2),  norm2_{i+1} = sum(Z_i^2),
# columns Z_1..Z_d divided by sqrt(norm2_2..norm2_{d+1}) (so each has sum of squares 1 on the
# fitting data). Coefficients are therefore R's, named `poly(x1, 2)1`, `poly(x1, 2)2`.
#
# Supported: `poly(column, integer degree >= 1)` on a fixed-effect formula (`mu` or any other
# distributional parameter), alone or combined with other terms and interactions. Refused with a
# message: a degree that is not a positive integer below the number of distinct values, an argument
# other than a plain column, and a direct call outside a formula. Only the positional form
# `poly(x, d)` with a literal integer `d` is accepted: `@formula` itself rejects keyword arguments
# (`raw = true`, `degree = 2`, `coefs = ...`) with its own parser error before DRModels runs, and
# several variables are refused here (write powers as function terms, or use the R bridge).

using StatsModels: AbstractTerm, FunctionTerm, ConstantTerm, Term, Schema
import StatsModels: apply_schema, modelcols, width, terms, termvars
import StatsAPI: coefnames

"""
    poly(x, degree)

R's orthogonal polynomial basis, for use inside a `@formula`, e.g.
`@formula(y ~ poly(x1, 2))`. The columns are those of `stats::poly(x1, 2)` and the coefficients
are named `poly(x1, 2)1`, `poly(x1, 2)2`, so the main-term names match drmTMB's (interaction
names use DRModels' `&`, as for every term). Every fit computes the basis from its own rows and
stores it in the fit; `predict(fit, newdata)` uses the basis of that fit. Only `poly(column, d)`
with a plain column and a literal positive integer degree is supported: `raw = true`,
`degree = 2` and `coefs = ...` are rejected by `@formula` itself. Needs `using DRModels` (or
write `DRModels.poly` in the formula after a plain `import DRModels`). Calling `poly` outside a
formula throws.
"""
poly(args...; kwargs...) = throw(ArgumentError(
    "`poly` is a formula term, not a function: write it inside a formula, " *
    "e.g. `@formula(y ~ poly(x1, 2))`."))

struct PolyTerm <: AbstractTerm
    sym::Symbol
    degree::Int
    alpha::Vector{Float64}
    norm2::Vector{Float64}          # length degree + 2; norm2[1] = 1 and norm2[2] = n as in R
end

width(t::PolyTerm) = t.degree
termvars(t::PolyTerm) = [t.sym]
coefnames(t::PolyTerm) = ["poly($(t.sym), $(t.degree))$j" for j in 1:t.degree]
Base.show(io::IO, t::PolyTerm) = print(io, "poly($(t.sym), $(t.degree))")

# Key under which a poly term's stored basis sits in the (cached) schema.
_poly_key(sym::Symbol, degree::Integer) = Term(Symbol("poly($sym, $degree)"))

function _poly_args(t::FunctionTerm{typeof(poly)})
    ok = length(t.args) == 2 && t.args[1] isa Term && t.args[2] isa ConstantTerm
    ok || throw(ArgumentError(
        "poly(x, degree): `$(t.exorig)` is not supported. Use `poly(column, degree)` with a plain " *
        "numeric column and an integer degree, e.g. `poly(x1, 2)`. A transformed argument and several " *
        "variables are not supported; write the powers as function terms instead."))
    deg = t.args[2].n
    (deg isa Integer || (deg isa AbstractFloat && isinteger(deg))) && deg >= 1 || throw(ArgumentError(
        "poly(x, degree): the degree in `$(t.exorig)` must be a positive integer."))
    return t.args[1].sym, Int(deg)
end

# R's recurrence on the fitting data -> stored basis.
function _poly_fit(x::AbstractVector{<:Real}, sym::Symbol, degree::Int)
    xs = Float64.(x)
    all(isfinite, xs) || throw(ArgumentError("poly($sym, $degree): `$sym` must be finite."))
    length(unique(xs)) > degree || throw(ArgumentError(
        "poly($sym, $degree): `degree` must be less than the number of unique values of `$sym` " *
        "($(length(unique(xs))))."))
    n = length(xs); xbar = sum(xs) / n; xc = xs .- xbar
    alpha = zeros(degree); norm2 = ones(degree + 2); norm2[2] = n
    Zprev = zeros(n); Z = ones(n)
    for i in 1:degree
        alpha[i] = sum(xc .* Z .^ 2) / norm2[i + 1]
        Znew = (xc .- alpha[i]) .* Z .- (i > 1 ? norm2[i + 1] / norm2[i] : 0.0) .* Zprev
        Zprev, Z = Z, Znew
        norm2[i + 2] = sum(abs2, Z)
    end
    return PolyTerm(sym, degree, alpha .+ xbar, norm2)
end

function modelcols(t::PolyTerm, d::NamedTuple)
    x = Float64.(collect(getproperty(d, t.sym))); n = length(x)
    Z = ones(n, t.degree + 1)
    Z[:, 2] .= x .- t.alpha[1]
    for i in 2:t.degree
        Z[:, i + 1] .= (x .- t.alpha[i]) .* Z[:, i] .- (t.norm2[i + 1] / t.norm2[i]) .* Z[:, i - 1]
    end
    return Z[:, 2:end] ./ transpose(sqrt.(t.norm2[3:end]))
end

# `@formula` leaves `poly(x, d)` as a FunctionTerm; the stored basis is looked up in the schema.
function apply_schema(t::FunctionTerm{typeof(poly)}, sch::Schema, Mod::Type)
    sym, degree = _poly_args(t)
    b = get(sch, _poly_key(sym, degree), nothing)
    b === nothing && throw(ArgumentError(
        "poly($sym, $degree): no stored basis for this term. The basis is computed when the model " *
        "is fitted and reused by `predict`; it cannot be rebuilt from new data alone."))
    return b
end

# Reserved `schema_cache` key (as `_FACTOR_LEVELS_KEY`): the poly bases of THIS fit, by term key.
# The cache lives in the fit's own copy of the formula (`_fit_formula`), never in the user's `bf(...)`
# object. Most non-Gaussian fitters build their designs without the `schema_cache`, so, like the factor
# levels, the basis is recorded once at the `drm` entry (`_record_factor_levels!`) from the rows that
# fit uses, and every later design of that fit, fit-time and `predict`, reads this one stored basis.
const _POLY_BASES_KEY = :__poly_bases__

# Complete rows as R's model frame would keep them: every column the formulas and the response
# name is present (and finite when numeric).
function _poly_complete_rows(f, tbl)
    cols = Set{Symbol}()
    for (_, rhs) in f.forms
        c = _rhs_columns(rhs)
        c === nothing || union!(cols, c)
    end
    push!(cols, f.response)
    ok = trues(length(first(tbl)))
    for c in cols
        haskey(tbl, c) || continue
        col = tbl[c]
        col isa AbstractVector || continue
        for (i, v) in enumerate(col)
            (ismissing(v) || (v isa Real && !isfinite(v))) && (ok[i] = false)
        end
    end
    return ok
end

# Never throws: a term whose basis cannot be built is recorded as absent, and `_design` then
# raises the specific refusal at fit time.
function _record_poly_bases!(f, data)
    hasproperty(f, :schema_cache) && hasproperty(f, :forms) || return nothing
    try
        found = Any[]
        for (_, rhs) in f.forms
            _collect_poly_terms!(found, rhs)
        end
        isempty(found) && return nothing
        tbl = Tables.columntable(data)
        ok = _poly_complete_rows(f, tbl)
        # The formula passed in is the fit's own copy (`_fit_formula`), so this dictionary is empty
        # unless this very fit entered `drm` before (a wrapper re-entering on the observed rows,
        # which rebuilds the same basis from the same complete rows).
        bases = Dict{Term,PolyTerm}()
        for t in found
            try
                sym, degree = _poly_args(t)
                haskey(tbl, sym) || continue
                col = tbl[sym]
                col isa AbstractVector && nonmissingtype(eltype(col)) <: Real || continue
                bases[_poly_key(sym, degree)] = _poly_fit(Float64.(col[ok]), sym, degree)
            catch
            end
        end
        f.schema_cache[][_POLY_BASES_KEY] = bases
    catch
    end
    return nothing
end

# Add the stored basis of every poly term of `rhs` to `sch` (a schema `_design` has just built, or
# the cached training schema). At fit time with nothing recorded, the basis comes from `data`
# (the fitting rows); in `predict` a missing basis is an error, never a basis of the new rows.
function _add_poly_schema!(sch::Schema, rhs, data, schema_cache, fitting::Bool)
    found = Any[]
    _collect_poly_terms!(found, rhs)
    isempty(found) && return sch
    stored = schema_cache === nothing ? nothing : get(schema_cache[], _POLY_BASES_KEY, nothing)
    for t in found
        sym, degree = _poly_args(t)
        key = _poly_key(sym, degree)
        haskey(sch, key) && continue
        b = stored isa Dict ? get(stored, key, nothing) : nothing
        if b === nothing
            fitting || throw(ArgumentError(
                "predict: the stored basis of poly($sym, $degree) is missing, and it cannot be " *
                "rebuilt from `newdata`. Refit the model with `drm` and predict again."))
            col = try
                _table_column(data, sym)
            catch e
                _missing_column_lookup(e) || rethrow()
                throw(ArgumentError("poly($sym, $degree): column `$sym` is not in the data."))
            end
            col isa AbstractVector && nonmissingtype(eltype(col)) <: Real || throw(ArgumentError(
                "poly($sym, $degree): `$sym` must be a numeric column."))
            any(ismissing, col) && throw(ArgumentError(
                "poly($sym, $degree): `$sym` contains `missing`; drop those rows first."))
            b = _poly_fit(Float64.(col), sym, degree)
        end
        merge!(sch, Schema(key => b))
    end
    return sch
end

_collect_poly_terms!(out, t::FunctionTerm{typeof(poly)}) = (push!(out, t); out)
_collect_poly_terms!(out, t::Union{Tuple,AbstractVector}) = (foreach(x -> _collect_poly_terms!(out, x), t); out)
function _collect_poly_terms!(out, t)
    for p in (:terms, :args)
        hasproperty(t, p) || continue
        v = getproperty(t, p)
        v isa Union{Tuple,AbstractVector} && foreach(x -> _collect_poly_terms!(out, x), v)
    end
    return out
end
