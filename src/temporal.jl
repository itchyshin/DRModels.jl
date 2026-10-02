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
# ENGINE (per series, O(n_s)): the exact Gaussian marginal by the
# prediction-error decomposition of the scalar state-space model
#     s_i = a_i s_{i−1} + η_i,  Var η_i = σ_t² (1 − a_i²),  s_1 ~ N(0, σ_t²),
#     r_i = s_i + ε_i,          Var ε_i = σ²,              r = y − Xβ,
# i.e. a Kalman filter over the series' rows in time order. With predicted
# variance P⁻_i, innovation v_i = r_i − m⁻_i and F_i = P⁻_i + σ², the
# temporal-plus-residual covariance factors as V_t = L diag(F) Lᵀ (L unit
# lower triangular), so
#     logdet V_t = Σ log F_i,     rᵀ V_t⁻¹ r = Σ v_i² / F_i.
# Every update is a sum of positive terms (P⁻ = a² P + σ_t²(1 − a²),
# P = P⁻ σ² / F) and 1 − a² is formed without cancellation (`-expm1` for OU,
# the factorisation (1 − φ²) Σ φ^{2k} with 1 − φ² = sech² θ for AR1), so the
# evaluation stays accurate as σ → 0, σ_t → 0 and as the persistence → 1 /
# decay → 0. (The equivalent tridiagonal-precision form, Q = R⁻¹, was tried
# first and rejected: Q has entries 1/(1 − a²), and an optimiser line search
# into λ → 0 lost every digit of its pivots — a negative "1ᵀ V⁻¹ 1" was
# measured on the OU recovery fixture.)
# The optional same-id intercept is a rank-one update V = V_t + σ_b² 11ᵀ. The
# filter is linear in the observation vector, so running it on 1 alongside r
# with the same gains gives u-weighted sums with no extra factorisation:
#     c = 1ᵀ V_t⁻¹ 1 = Σ v_i(1)²/F_i,   d = 1ᵀ V_t⁻¹ r = Σ v_i(1) v_i(r)/F_i,
#     logdet V = logdet V_t + log(1 + σ_b² c),
#     rᵀV⁻¹r   = rᵀV_t⁻¹r − d² / (1/σ_b² + c).
# Conditional modes at θ̂ (k = d / (1/σ_b² + c), k = 0 without the intercept):
# the temporal effect σ_t E[x | y] = σ_t² R V⁻¹ r = σ_t² R V_t⁻¹ (r − k 1) is
# the Rauch–Tung–Striebel smoothed state for the data r − k 1, and the
# intercept mode is E[b | y] = σ_b² (d − c k).
# A dense oracle (−logpdf(MvNormal(Xβ, σ_t² R + σ_b² J + σ² I))) checks the
# objective and the modes to 1e-10 / 1e-8 in test/test_temporal_ar1.jl and
# test/test_temporal_ou.jl.
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

# Transition a and innovation fraction 1 − a² for one gap, both without
# cancellation. AR1: ψ = θ (φ = tanh θ), gap a positive integer g,
# 1 − φ^{2g} = sech²θ · Σ_{k<g} φ^{2k}. OU: ψ = λ.
@inline function _temporal_transition(structure::Symbol, ψ, gap)
    if structure === :ar1
        φ = tanh(ψ)
        g = Int(gap)
        a = φ^g
        if g <= 64
            φ2 = φ * φ
            acc = one(φ); term = one(φ)
            for _ in 2:g
                term *= φ2
                acc += term
            end
            return a, acc / cosh(ψ)^2
        end
        return a, one(a) - a * a
    else
        x = ψ * gap
        return exp(-x), -expm1(-2 * x)
    end
end

# Kalman filter over one series. `z` is the residual vector r in time order;
# `with_one` also filters the vector of ones (for the ordinary intercept).
# Returns (Σ log F, Σ v²/F, c, d) and, with `keep`, the filter history for
# the smoother.
function _temporal_filter(structure, ψ, σ2, st2, g, z; with_one::Bool = false, keep::Bool = false)
    n = length(z)
    T = promote_type(typeof(ψ), typeof(σ2), typeof(st2), eltype(z))
    ldF = zero(T); q = zero(T); c = zero(T); d = zero(T)
    m = zero(T); m1 = zero(T); P = zero(T)
    hist = keep ? (mf = zeros(T, n), Pf = zeros(T, n), mp = zeros(T, n), Pp = zeros(T, n),
                   a = zeros(T, n)) : nothing
    for i in 1:n
        if i == 1
            mp = zero(T); mp1 = zero(T); Pp = st2
        else
            a, v = _temporal_transition(structure, ψ, g[i])
            v > 0 || return nothing
            mp = a * m; mp1 = a * m1; Pp = a * a * P + st2 * v
            keep && (hist.a[i] = a)
        end
        F = Pp + σ2
        F > 0 || return nothing
        e = z[i] - mp
        ldF += log(F); q += e * e / F
        K = Pp / F
        m = mp + K * e
        if with_one
            e1 = one(T) - mp1
            c += e1 * e1 / F; d += e1 * e / F
            m1 = mp1 + K * e1
        end
        P = Pp * σ2 / F
        if keep
            hist.mf[i] = m; hist.Pf[i] = P; hist.mp[i] = mp; hist.Pp[i] = Pp
        end
    end
    return ldF, q, c, d, hist
end

# Rauch–Tung–Striebel smoother of the filter history: E[s | z] in time order.
function _temporal_smooth(hist)
    n = length(hist.mf)
    ms = copy(hist.mf)
    for i in (n-1):-1:1
        Pp = hist.Pp[i+1]
        J = Pp > 0 ? hist.Pf[i] * hist.a[i+1] / Pp : zero(Pp)
        ms[i] = hist.mf[i] + J * (ms[i+1] - hist.mp[i+1])
    end
    return ms
end

# Per-series logdet V_s and rᵀV_s⁻¹r at (σ², σ_t², σ_b² or nothing, ψ), and
# with `modes` the conditional temporal effects and intercept mode.
function _temporal_series(structure, ψ, σ2, st2, sb2, g, r; modes::Bool = false)
    out = _temporal_filter(structure, ψ, σ2, st2, g, r; with_one = sb2 !== nothing)
    out === nothing && return nothing
    ldV, quad, c, d, _ = out
    k = zero(quad)
    if sb2 !== nothing
        den = inv(sb2) + c
        ldV += log1p(sb2 * c)
        quad -= d^2 / den
        k = d / den
    end
    modes || return ldV, quad
    sm = _temporal_filter(structure, ψ, σ2, st2, g, r .- k; keep = true)
    teff = _temporal_smooth(sm[5])
    bmode = sb2 === nothing ? zero(quad) : sb2 * (d - c * k)
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
    # ψ handed to the kernels: θ itself for AR1 (φ = tanh θ is formed there so
    # that 1 − φ² = sech²θ is exact), λ = exp θ for OU.
    ψof(θ) = structure === :ar1 ? θ[iψ] : exp(θ[iψ])

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
