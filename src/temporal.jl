# temporal.jl — temporal AR1 / OU random effects on the Gaussian mean (wave 1).
#
# Julia twin of drmTMB's `temporal(1 | id, time = occ, structure = "ar1" | "ou")`
# (drmTMB branches `claude/lane-temporal-ar1-v2`, `codex/temporal-ou-v1-20260908`;
# semantics only — no drmTMB source is copied, drmTMB is GPL and this file is MIT).
#
# MODEL. Rows are grouped into series by `id` (series in first-seen order) and
# sorted within a series by `time` (ties are refused, see below). Each row i of
# series s carries one latent state x_i; the states of a series form a
# stationary unit-variance Markov chain
#     x_1 ~ N(0, 1),   x_i = a_i x_{i-1} + sqrt(1 - a_i²) e_i,   e_i ~ N(0, 1),
# with the transition a_i set by the gap to the previous occasion:
#     AR1: a_i = φ^{g_i},       g_i = t_i − t_{i−1} ∈ {1, 2, …},  φ = tanh(θ)
#     OU : a_i = exp(−λ Δt_i),  Δt_i = t_i − t_{i−1} > 0,         λ = exp(θ)
# so Corr(x_i, x_j) = φ^{|t_i − t_j|} (AR1) or exp(−λ|t_i − t_j|) (OU). The
# response is
#     y = Xβ + σ_t x + Z_id b + ε,  b ~ N(0, σ_b² I) (optional `(1 | id)`),
#     ε ~ N(0, σ² I)   (`sigma ~ 1` only),
# and series are independent. Every ingredient is Gaussian, so the marginal
# likelihood is exact.
#
# ENGINE (per series, O(n_s)). The chain precision Q = R⁻¹ is TRIDIAGONAL:
#     w_i = 1 / (1 − a_i²)   (i ≥ 2)
#     Q_11 = 1 + a_2² w_2,  Q_ii = w_i + a_{i+1}² w_{i+1},  Q_nn = w_n,
#     Q_{i,i−1} = −a_i w_i,  logdet Q = Σ_{i≥2} log w_i.
# The temporal-plus-residual covariance V_t = σ² I + σ_t² Q⁻¹ = Q⁻¹ B with the
# tridiagonal, SPD B = σ² Q + σ_t² I. Hence (Q and B commute)
#     logdet V_t = logdet B − logdet Q,     V_t⁻¹ v = B⁻¹ (Q v),
# both from ONE tridiagonal Cholesky of B. Unlike the Woodbury form
# rᵀr/σ² − bᵀH⁻¹b this involves no difference of O(1/σ²) terms, so it stays
# accurate as σ → 0 or σ_t → 0 (the #764 cancellation class cannot arise).
# The optional same-id intercept is a rank-one update V = V_t + σ_b² 11ᵀ:
#     u = V_t⁻¹ 1,  c = 1ᵀu,  d = uᵀr,
#     logdet V = logdet V_t + log(1 + σ_b² c),
#     rᵀV⁻¹r   = rᵀV_t⁻¹r − d² / (1/σ_b² + c).
# Conditional modes at θ̂ (k = d / (1/σ_b² + c), k = 0 without the intercept):
#     temporal effect σ_t E[x | y] = σ_t² Q⁻¹ V⁻¹ r = σ_t² B⁻¹ (r − k 1),
#     intercept       E[b | y]     = σ_b² (d − c k).
# A dense oracle (−logpdf(MvNormal(Xβ, σ_t² R + σ_b² J + σ² I))) checks this to
# 1e-10 in test/test_temporal_ar1.jl and test/test_temporal_ou.jl.
#
# CLAIM BOUNDARY (wave 1, owner decision D-310): Gaussian family, the mean
# formula only, one unlabelled intercept `temporal(1 | id, …)`, `sigma ~ 1`,
# ML, at most one ordinary `(1 | id)` on the SAME id (as drmTMB). Everything
# else is refused by name: other families, a temporal term on `sigma`, slopes,
# labelled bars, REML, a second temporal or any structured term, `meta_V`,
# `sd()` submodels, random effects on `sigma`, missing responses, non-default
# `algorithm`/`marginal`/`penalty`/`sparse`, and the bootstrap.

"""
    temporal(1 | id, time, structure)

Temporal random-intercept marker on the Gaussian **mean**: a stationary AR1
(`ar1`) or Ornstein–Uhlenbeck (`ou`) process within each series `id`, indexed
by the column `time`.

```julia
drm(bf(@formula(y ~ x + temporal(1 | id, occ, ar1)), @formula(sigma ~ 1)),
    Gaussian(); data)
drm(bf(@formula(y ~ x + temporal(1 | id, elapsed, ou)), @formula(sigma ~ 1)),
    Gaussian(); data)
```

This is the twin of drmTMB's
`temporal(1 | id, time = occ, structure = "ar1")`. StatsModels' `@formula`
cannot parse keyword arguments or string literals, so the Julia spelling is
positional with a bare structure name (the same convention as
`sd(g, phylogenetic)`): `temporal(1 | id, occ, ar1)`. Through the R bridge
(`drm_bridge`) the exact drmTMB spelling is accepted and translated.

Semantics (as drmTMB): the first state of each series is stationary N(0, 1);
AR1 needs integer `time` and keeps real gaps as integer powers, `φ^gap`
(`φ = tanh θ`, may be negative); OU takes numeric elapsed time with
correlation `exp(−λ Δt)` over a gap `Δt` (`λ = exp θ > 0`, in the units of
`time`). Only gaps matter, so the time origin is irrelevant. Each `(id, time)`
pair must be unique; rows may be in any order. The fitted quantities are the
process SD (`re_sd(fit)[:id]`), the persistence `φ` or decay `λ`
([`temporal_parameters`](@ref)), the residual SD `σ`, and, when the formula
also has `(1 | id)`, the stable intercept SD (`re_sd(fit)[:id_iid]`).

Wave-1 scope: Gaussian family, `sigma ~ 1`, ML only, intercept-only and
unlabelled, at most one ordinary `(1 | id)` on the same `id`, no other
structured term. Every other use is refused with an error.
"""
temporal(x...) = x   # marker; intercepted during formula parsing

const _TEMPORAL_SPELLING = "Use `temporal(1 | id, occ, ar1)` or `temporal(1 | id, elapsed, ou)` " *
    "(drmTMB: `temporal(1 | id, time = occ, structure = \"ar1\")`; StatsModels' `@formula` " *
    "cannot carry keyword arguments, so the Julia spelling is positional)."

const _TEMPORAL_SCOPE = "temporal() is implemented only for the univariate Gaussian MEAN formula " *
    "(`y ~ … + temporal(1 | id, time, ar1|ou)` with `sigma ~ 1`, ML) in this release"

# Refusal used by `_split_ranef` for every caller that has not opted in to the
# temporal marker (every non-Gaussian family, the `sigma` formula, the bivariate
# and mixed-family routes, the Laplace/AGHQ Gaussian route, the bootstrap …).
_temporal_refuse_here() = throw(ArgumentError("drm: " * _TEMPORAL_SCOPE *
    "; this model / route does not support it. Other families, a temporal term on `sigma`, " *
    "bivariate and mixed-family models, REML, and the bootstrap are not implemented for temporal()."))

# Parse one `temporal(...)` FunctionTerm into (group, time, structure).
function _parse_temporal_term(t)
    args = t.args
    length(args) == 3 ||
        throw(ArgumentError("drm: `temporal()` requires a bar term, a time column and a " *
            "structure. " * _TEMPORAL_SPELLING))
    bar, tm, st = args
    (bar isa FunctionTerm && bar.f === (|) && length(bar.args) == 2) ||
        throw(ArgumentError("drm: the first argument of `temporal()` must be a random-effect " *
            "bar such as `1 | id`. " * _TEMPORAL_SPELLING))
    lhs, grp = bar.args
    (lhs isa ConstantTerm && lhs.n == 1 && grp isa Term) ||
        throw(ArgumentError("drm: `temporal()` currently supports one intercept-only unlabelled " *
            "random effect, `temporal(1 | id, …)`; temporal slopes and covariance-block labels " *
            "are not implemented."))
    tm isa Term ||
        throw(ArgumentError("drm: the time argument of `temporal()` must name an occasion " *
            "column. " * _TEMPORAL_SPELLING))
    (st isa Term && st.sym in (:ar1, :ou)) ||
        throw(ArgumentError("drm: the structure of `temporal()` must be `ar1` or `ou`. " *
            _TEMPORAL_SPELLING))
    return (group = grp.sym, time = tm.sym, structure = st.sym)
end

# drmTMB's public label for the term.
_temporal_label(tt) = "temporal(1 | $(tt.group), time = $(tt.time), structure = \"$(tt.structure)\")"

# Every temporal term on a right-hand side (used by `structured_effects`).
function _collect_temporal(rhs)
    terms = rhs isa Tuple ? collect(rhs) : Any[rhs]
    return [_parse_temporal_term(t) for t in terms if t isa FunctionTerm && t.f === temporal]
end

# Series layout and drmTMB's data checks. Returns the row indices of each series
# (sorted by time, ties broken by row), the per-row gap to the previous occasion
# of the same series (0 for a series' first row), and the series levels.
function _temporal_layout(tt, data; has_ordinary::Bool)
    sname = tt.structure === :ar1 ? "AR1" : "OU"
    ids = _table_column(data, tt.group)
    times = _table_column(data, tt.time)
    n = length(ids)
    length(times) == n || error("drm: internal — temporal id/time length mismatch")
    (any(ismissing, ids) || any(ismissing, times)) &&
        throw(ArgumentError("drm: Temporal $sname identifiers and times must be complete; " *
            "columns `$(tt.group)` / `$(tt.time)` contain missing values."))
    all(v -> v isa Real && !(v isa Bool), times) ||
        throw(ArgumentError("drm: Temporal inputs must be finite numeric values; `$(tt.time)` " *
            "cannot be a categorical, date-time, Bool or other non-numeric value."))
    tv = Float64.(times)
    all(isfinite, tv) ||
        throw(ArgumentError("drm: Temporal inputs must be finite numeric values; `$(tt.time)` " *
            "contains a non-finite value."))
    tt.structure === :ar1 && any(v -> v != round(v), tv) &&
        throw(ArgumentError("drm: Temporal AR1 occasions must be finite integers; `$(tt.time)` " *
            "cannot be fractional for `ar1`. Use the original integer sampling occasion (its " *
            "gaps are part of the AR1 model), or `ou` for elapsed time."))
    gidx, S = _group_index(ids)
    rows = [Int[] for _ in 1:S]
    for i in 1:n
        push!(rows[gidx[i]], i)
    end
    gap = zeros(Float64, n)
    ndup = 0
    for s in 1:S
        r = rows[s]
        sort!(r; by = i -> (tv[i], i))
        for k in 2:length(r)
            g = tv[r[k]] - tv[r[k-1]]
            g == 0 && (ndup += 1)
            gap[r[k]] = g
        end
    end
    ndup == 0 ||
        throw(ArgumentError("drm: Temporal $sname series-time keys must be unique: $ndup " *
            "duplicated `($(tt.group), $(tt.time))` key(s) found. Use one response per series " *
            "and occasion, or aggregate the data before fitting."))
    # drmTMB's identifiability check on the distinct positive within-series
    # lags (all pairs, not only consecutive gaps): at least 2 (3 with an
    # ordinary intercept), and for AR1 at least one odd lag (the sign of φ).
    # The pair scan stops as soon as the requirement is met, so a long series
    # does not cost O(n²).
    required = has_ordinary ? 3 : 2
    needodd = tt.structure === :ar1
    distinct = Set{Float64}()
    hasodd = false
    done = false
    for s in 1:S
        ts = tv[rows[s]]
        for a in 1:length(ts), b in (a+1):length(ts)
            lag = ts[b] - ts[a]
            lag > 0 || continue
            push!(distinct, lag)
            needodd && isodd(Int(lag)) && (hasodd = true)
            if length(distinct) >= required && (!needodd || hasodd)
                done = true
                break
            end
        end
        done && break
    end
    ok = done
    distinct = sort(collect(distinct))
    ok || throw(ArgumentError("drm: Temporal $sname occasions do not provide the required lag " *
        "variation: found distinct positive lags $(distinct); this model needs at least " *
        "$required" * (tt.structure === :ar1 ? ", including an odd lag" : "") * ". Keep genuine " *
        "sampling gaps and collect more distinct within-series occasions."))
    has_ordinary && S < 2 &&
        throw(ArgumentError("drm: a temporal $sname model with an ordinary random intercept " *
            "requires multiple series; fit the temporal process without `(1 | $(tt.group))` " *
            "for one series, or provide observations from at least two ids."))
    return (rows = rows, gap = gap, nseries = S, gidx = gidx)
end

using Statistics: var, median

# --- tridiagonal kernels (AD-friendly; d = diagonal, e = sub-diagonal) -------
# Cholesky B = L Lᵀ, L lower bidiagonal with diagonal `l` and sub-diagonal `m`.
# Returns `nothing` when a pivot is not positive.
function _tridiag_chol(d::AbstractVector{T}, e::AbstractVector{T}) where {T}
    n = length(d)
    l = Vector{T}(undef, n); m = Vector{T}(undef, max(n - 1, 0))
    p = d[1]
    p > 0 || return nothing
    l[1] = sqrt(p)
    @inbounds for i in 2:n
        m[i-1] = e[i-1] / l[i-1]
        p = d[i] - m[i-1]^2
        p > 0 || return nothing
        l[i] = sqrt(p)
    end
    return l, m
end

# Solve (L Lᵀ) x = v.
function _tridiag_solve(l, m, v::AbstractVector)
    n = length(l)
    T = promote_type(eltype(l), eltype(v))
    z = Vector{T}(undef, n)
    z[1] = v[1] / l[1]
    @inbounds for i in 2:n
        z[i] = (v[i] - m[i-1] * z[i-1]) / l[i]
    end
    z[n] /= l[n]                              # back substitution Lᵀ x = z
    @inbounds for i in (n-1):-1:1
        z[i] = (z[i] - m[i] * z[i+1]) / l[i]
    end
    return z
end

# y = Q v for tridiagonal Q.
function _tridiag_mul(d, e, v)
    n = length(d)
    T = promote_type(eltype(d), eltype(v))
    out = Vector{T}(undef, n)
    @inbounds for i in 1:n
        s = d[i] * v[i]
        i > 1 && (s += e[i-1] * v[i-1])
        i < n && (s += e[i] * v[i+1])
        out[i] = s
    end
    return out
end

# Transition a and innovation variance 1 − a² for one gap.
@inline function _temporal_transition(structure::Symbol, ψ, gap)
    if structure === :ar1
        a = ψ^Int(gap)                       # ψ = φ
        return a, one(a) - a * a
    else
        x = ψ * gap                          # ψ = λ
        return exp(-x), -expm1(-2 * x)
    end
end

# Chain precision (d, e, logdet Q) for one series with gaps `g` (g[1] unused).
function _temporal_chain_precision(structure::Symbol, ψ, g::AbstractVector)
    n = length(g)
    T = typeof(ψ)
    d = zeros(T, n); e = zeros(T, max(n - 1, 0))
    d[1] = one(T)
    ldQ = zero(T)
    for i in 2:n
        a, v = _temporal_transition(structure, ψ, g[i])
        v > 0 || return nothing
        w = inv(v)
        d[i] += w
        d[i-1] += a * a * w
        e[i-1] = -a * w
        ldQ += log(w)
    end
    return d, e, ldQ
end

# Per-series quantities at (σ², σ_t², σ_b² or nothing, ψ): logdet V_s, rᵀV_s⁻¹r
# and (when `modes`) the conditional temporal effects and intercept mode.
function _temporal_series(structure, ψ, σ2, st2, sb2, g, r; modes::Bool = false)
    pr = _temporal_chain_precision(structure, ψ, g)
    pr === nothing && return nothing
    d, e, ldQ = pr
    ch = _tridiag_chol(σ2 .* d .+ st2, σ2 .* e)
    ch === nothing && return nothing
    l, m = ch
    ldV = 2 * sum(log, l) - ldQ
    Vr = _tridiag_solve(l, m, _tridiag_mul(d, e, r))        # V_t⁻¹ r
    quad = dot(r, Vr)
    k = zero(quad); c = zero(quad); dd = zero(quad)
    if sb2 !== nothing
        u = _tridiag_solve(l, m, _tridiag_mul(d, e, ones(eltype(d), length(r))))  # V_t⁻¹ 1
        c = sum(u); dd = dot(u, r)
        den = inv(sb2) + c
        ldV += log1p(sb2 * c)
        quad -= dd^2 / den
        k = dd / den
    end
    modes || return ldV, quad
    teff = st2 .* _tridiag_solve(l, m, r .- k)
    bmode = sb2 === nothing ? zero(quad) : sb2 * (dd - c * k)
    return ldV, quad, teff, bmode
end

# θ = [βμ; log σ; (log σ_b); log σ_t; θ_temporal]
function _fit_temporal_gaussian(fam::Gaussian, y, Xμ, Xσ, nmμ, nmσ, tt, lay, has_ordinary, g_tol)
    n = length(y)
    pμ = size(Xμ, 2)
    size(Xσ, 2) == 1 || error("drm: internal — temporal route needs `sigma ~ 1`")
    structure = tt.structure
    S = lay.nseries
    grows = lay.rows
    ggap = [lay.gap[r] for r in grows]
    iσ = pμ + 1
    ib = has_ordinary ? pμ + 2 : 0
    it = pμ + (has_ordinary ? 3 : 2)
    iψ = it + 1
    np = iψ
    ψof(θ) = structure === :ar1 ? tanh(θ[iψ]) : exp(θ[iψ])

    function nll(θ)
        T = eltype(θ)
        r = y .- Xμ * θ[1:pμ]
        σ2 = exp(2 * θ[iσ]); st2 = exp(2 * θ[it])
        sb2 = has_ordinary ? exp(2 * θ[ib]) : nothing
        ψ = ψof(θ)
        total = zero(T)
        for s in 1:S
            out = _temporal_series(structure, ψ, σ2, st2, sb2, ggap[s], r[grows[s]])
            out === nothing && return convert(T, 1e18)
            total += out[1] + out[2]
        end
        isfinite(total) || return convert(T, 1e18)
        return 0.5 * total + 0.5 * n * log(2π)
    end

    # Starts (as drmTMB): OLS β, the residual variance split evenly over the
    # residual + temporal (+ intercept) components, and two persistence starts
    # (AR1: φ = ±0.3; OU: correlation 0.3 / 0.7 over the median positive gap),
    # keeping the lower objective.
    βμ0 = Xμ \ y
    v0 = var(y .- Xμ * βμ0)
    (isfinite(v0) && v0 > 0) || (v0 = 1.0)
    ncomp = has_ordinary ? 3 : 2
    base = zeros(np)
    base[1:pμ] .= βμ0
    base[iσ] = log(sqrt(v0 / ncomp))
    has_ordinary && (base[ib] = log(sqrt(v0 / ncomp)))
    base[it] = log(sqrt(v0 / ncomp))
    ψstarts = if structure === :ar1
        [atanh(0.3), -atanh(0.3)]
    else
        pg = filter(>(0), lay.gap)
        ref = isempty(pg) ? 1.0 : median(pg)
        [log(-log(ρ) / ref) for ρ in (0.3, 0.7)]
    end
    best = nothing
    for ψ0 in ψstarts
        θ0 = copy(base); θ0[iψ] = ψ0
        res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
        (best === nothing || Optim.minimum(res) < Optim.minimum(best)) && (best = res)
    end
    θ̂ = Optim.minimizer(best)
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))

    # Conditional modes at θ̂, temporal effects back in DATA ROW order.
    r̂ = y .- Xμ * θ̂[1:pμ]
    σ2 = exp(2 * θ̂[iσ]); st2 = exp(2 * θ̂[it])
    sb2 = has_ordinary ? exp(2 * θ̂[ib]) : nothing
    ψ̂ = ψof(θ̂)
    teff = zeros(n); bmodes = zeros(S)
    for s in 1:S
        _, _, te, bm = _temporal_series(structure, ψ̂, σ2, st2, sb2, ggap[s], r̂[grows[s]]; modes = true)
        teff[grows[s]] .= te
        bmodes[s] = bm
    end

    grp = String(tt.group)
    resd_names = has_ordinary ? ["$(grp)_iid", grp] : [grp]
    pblock = structure === :ar1 ? :temporal_phi : :temporal_decay
    blocks = [:mu => 1:pμ, :sigma => iσ:iσ,
              :resd => (has_ordinary ? (ib:it) : (it:it)), pblock => iψ:iψ]
    names = [:mu => nmμ, :sigma => nmσ, :resd => resd_names, pblock => [_temporal_label(tt)]]
    means = Dict(:mu => Xμ * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => fill(exp(θ̂[iσ]), n))
    effects = Dict{Symbol,Vector{Float64}}(tt.group => teff)
    has_ordinary && (effects[Symbol("$(grp)_iid")] = bmodes)
    info = (label = _temporal_label(tt), structure = structure, group = tt.group,
            time = tt.time, nseries = S)
    fit = DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(best), means, obs, scales)
    return _withranef(_withnll(fit, nll), (effects = effects, temporal = info))
end

# Router: validate the wave-1 scope, then fit. Called from `_drm_gaussian_fit`
# before any other route can claim the formula.
function _drm_gaussian_temporal(f, fam::Gaussian, tt, re, metav, structured, sigma_re,
        structured_sigma, y, Xμ, Xσ, nmμ, nmσ, data; method, algorithm, penalty,
        phylo_coupled, sparse, has_missing_response, g_tol)
    lbl = _temporal_label(tt)
    nope(what) = throw(ArgumentError("drm: `$lbl` cannot be combined with $what; " *
        _TEMPORAL_SCOPE * "."))
    length(_collect_temporal(Dict(f.forms)[:mu])) == 1 ||
        throw(ArgumentError("drm: only one temporal effect is implemented in `mu`. " * _TEMPORAL_SPELLING))
    method === :ML || nope("`method = :$method` (temporal fits are ML only; REML is not implemented)")
    (structured === nothing && isempty(_collect_structured(Dict(f.forms)[:mu]))) ||
        nope("another structured effect (phylo/relmat/animal/spatial) in this first slice")
    metav === nothing || nope("`meta_V(...)`")
    (isempty(sigma_re) && structured_sigma === nothing) ||
        nope("a random or structured effect on `sigma` (temporal models require `sigma ~ 1`)")
    (isempty(_sdphylo_parts(f)) && isempty(_sd_parts(f))) || nope("an `sd(…) ~ …` submodel")
    keys_extra = setdiff(Set(first.(f.forms)), Set((:mu, :sigma)))
    isempty(keys_extra) || nope("additional distributional parameters ($(join(sort(String.(collect(keys_extra))), ", ")))")
    (size(Xσ, 2) == 1 && all(==(1.0), view(Xσ, :, 1))) ||
        nope("predictors on `sigma` (temporal Gaussian models currently require `sigma ~ 1`)")
    algorithm === :auto || nope("`algorithm = :$algorithm`")
    penalty === nothing || nope("`penalty`")
    phylo_coupled && nope("`phylo_coupled = true`")
    (sparse === nothing || sparse === false) || nope("`sparse = $sparse`")
    has_missing_response && nope("missing responses (drop the missing-response rows before " *
        "calling `drm`, e.g. with `drm_listwise`)")
    has_ordinary = false
    if !isempty(re)
        length(re) == 1 || nope("more than one ordinary random effect")
        rl, g = re[1]
        kind, _ = _re_kind(rl)
        (kind === :intercept && g === tt.group) ||
            throw(ArgumentError("drm: the ordinary random effect paired with `$lbl` must be " *
                "`(1 | $(tt.group))` using the same id; use either no ordinary random effect " *
                "or `(1 | $(tt.group))`."))
        has_ordinary = true
    end
    lay = _temporal_layout(tt, data; has_ordinary = has_ordinary)
    return _fit_temporal_gaussian(fam, y, Xμ, Xσ, nmμ, nmσ, tt, lay, has_ordinary, g_tol)
end

"""
    temporal_parameters(fit) -> NamedTuple

The fitted temporal process of a `temporal(...)` Gaussian fit, on natural
scales: `label` (drmTMB's term label), `structure` (`:ar1` / `:ou`), `sd`
(process SD σ_t; drmTMB `sdpars\$mu["temporal_sd: <label>"]`), `phi` (AR1
persistence, one occasion apart; drmTMB `corpars\$temporal`) or `decay` (OU
rate λ in the units of `time`; drmTMB `decaypars\$temporal`) — the other is
`nothing` — `sd_iid` (the ordinary `(1 | id)` SD, or `nothing`) and `sigma`
(residual SD). Errors on a fit without a temporal term.
"""
function temporal_parameters(fit::DrmFit)
    info = fit.ranef isa NamedTuple && haskey(fit.ranef, :temporal) ? fit.ranef.temporal : nothing
    info === nothing &&
        throw(ArgumentError("temporal_parameters: this fit has no `temporal(...)` term."))
    sds = re_sd(fit)
    g = info.group
    iid = Symbol("$(g)_iid")
    ψ = only(coef(fit, info.structure === :ar1 ? :temporal_phi : :temporal_decay))
    return (label = info.label, structure = info.structure, sd = sds[g],
            phi = info.structure === :ar1 ? tanh(ψ) : nothing,
            decay = info.structure === :ou ? exp(ψ) : nothing,
            sd_iid = get(sds, iid, nothing),
            sigma = exp(only(coef(fit, :sigma))))
end
