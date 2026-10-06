# loadfixture.jl — file I/O for the R-parity suite, kept separate so compare.jl
# stays pure (no disk access). Uses only stdlib: `TOML` (expected.toml) and
# `DelimitedFiles.readdlm` (data.csv). NO CSV.jl / no added dependency.
#
# Pairs with compare.jl: `load_expected` produces a `ParityExpected`, `load_data`
# produces a NamedTuple that drops straight into `drm(...; data = …)`.

using TOML
using DelimitedFiles: readdlm

"""
    load_expected(dir) -> ParityExpected

Parse `joinpath(dir, "expected.toml")` into a [`ParityExpected`]. Expected TOML
layout (see `test/parity/README.md`):

```toml
[fit]
family = "gaussian"
formula = "y ~ x; sigma ~ x"   # traceability only; not parsed here
loglik = -256.51
aic    = 521.02
df     = 4
n      = 200

[coef]
"mu_(Intercept)" = 1.2031
"sigma_x"        = 0.0902

[vcov]                          # optional
order = ["mu_(Intercept)", "mu_x", "sigma_(Intercept)", "sigma_x"]
data  = [ [ … ], [ … ], … ]     # row-major numeric matrix

[tol]                           # optional per-case overrides
atol_loglik = 1e-3
```
"""
function load_expected(dir)::ParityExpected
    path = joinpath(dir, "expected.toml")
    t = TOML.parsefile(path)

    haskey(t, "fit") || error("load_expected: $path missing required [fit] block")
    fit = t["fit"]
    coef = get(t, "coef", Dict{String,Any}())

    vcov_order = nothing
    vcov = nothing
    if haskey(t, "vcov")
        v = t["vcov"]
        haskey(v, "order") || error("load_expected: [vcov] block needs an `order` array")
        haskey(v, "data") || error("load_expected: [vcov] block needs a `data` matrix")
        vcov_order = Vector{String}(String.(v["order"]))
        rows = v["data"]                       # vector of row-vectors (row-major)
        k = length(vcov_order)
        M = Matrix{Float64}(undef, k, k)
        length(rows) == k || error("load_expected: [vcov].data has $(length(rows)) rows, " *
            "expected $k to match `order`")
        for (i, row) in enumerate(rows)
            length(row) == k || error("load_expected: [vcov].data row $i has $(length(row)) " *
                "entries, expected $k")
            for j in 1:k
                M[i, j] = Float64(row[j])
            end
        end
        vcov = M
    end

    tol = Dict{String,Float64}()
    if haskey(t, "tol")
        for (k, v) in t["tol"]
            tol[String(k)] = Float64(v)
        end
    end

    # Optional [ranef] group-level covariance block (location–scale models).
    ranef_group = nothing
    ranef = Dict{String,Float64}()
    if haskey(t, "ranef")
        r = t["ranef"]
        haskey(r, "group") || error("load_expected: [ranef] block needs a `group` name")
        ranef_group = String(r["group"])
        for key in ("sd_mu", "sd_sigma", "cor")
            haskey(r, key) && (ranef[key] = Float64(r[key]))
        end
    end

    # Optional [se] block: per-coefficient Wald standard errors, keyed by the same
    # flat "<param>_<coefname>" convention as [coef]. The reserved `not_comparable`
    # array key is NOT an SE value — it names parameters whose SE must be declined
    # rather than compared (e.g. a variance component pinned at DRModels.jl's
    # _LAPLACE_LOG_SD_FLOOR, a boundary drmTMB does not share). Absent ⇒ SE checks
    # are skipped entirely, so every pre-existing fixture keeps passing unchanged.
    se = Dict{String,Float64}()
    se_not_comparable = String[]
    if haskey(t, "se")
        s = t["se"]
        for (k, v) in s
            if String(k) == "not_comparable"
                se_not_comparable = Vector{String}(String.(v))
            else
                se[String(k)] = Float64(v)
            end
        end
    end

    return ParityExpected(;
        family = String(fit["family"]),
        coef = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in coef),
        loglik = Float64(fit["loglik"]),
        aic = Float64(fit["aic"]),
        df = Int(fit["df"]),
        n = Int(fit["n"]),
        vcov_order = vcov_order,
        vcov = vcov,
        tol = tol,
        ranef_group = ranef_group,
        ranef = ranef,
        se = se,
        se_not_comparable = se_not_comparable)
end

"""
    load_data(dir) -> NamedTuple

Read `joinpath(dir, "data.csv")` with `readdlm(...; header = true)` and return a
NamedTuple of column vectors keyed by the (Symbol) header, so it drops straight
into `drm(...; data = nt)`. Numeric columns become `Vector{Float64}`; any
non-numeric column is kept as-is (e.g. string grouping factors).

#474: `readdlm` returns a `Matrix{Any}` the moment ANY column in the file is
non-numeric, so every column's *container* eltype reads `Any` even where a
column's own values are all numbers — a per-column `eltype(col) <: Number`
check is blind to that and silently drops the column to categorical. Decide
numeric-ness from the column's actual values instead, so one string column
cannot contaminate its numeric neighbours.
"""
function load_data(dir)::NamedTuple
    path = joinpath(dir, "data.csv")
    raw, header = readdlm(path, ','; header = true)
    cols = Symbol.(strip.(string.(vec(header))))
    pairs = map(enumerate(cols)) do (j, name)
        col = raw[:, j]
        coltyped = all(v -> v isa Number, col) ? Float64.(col) : col
        name => coltyped
    end
    return NamedTuple(pairs)
end
