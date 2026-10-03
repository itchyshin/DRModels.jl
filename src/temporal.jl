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
# WAVE 2 (owner decision D-311): the paired phylogenetic-temporal provider of
# drmTMB (branch `codex/phylo-temporal-ou-exec-v1-20260909`, semantics only),
#     y = Xβ + a_species + b_species(t) + ε,  a ~ N(0, σ_a² C),
# C the tip CORRELATION matrix of the tree, b an independent OU path per
# species (the series ARE the species: `phylo(1 | species)` and
# `temporal(1 | species, elapsed, ou)` must share the grouping). The two
# components are additive: related species share a stable baseline, not their
# temporal departures. Engine: the per-series filter above reduces each species
# to a Gaussian factor in a_s, and the tree is integrated by an exact pruning
# pass (see "paired phylo() stable field" below). drmTMB's refusals are
# mirrored: OU only, an unlabelled intercept-only phylo term on the same id, no
# ordinary `(1 | species)`, ≥ 3 species, each with ≥ 2 times, ≥ 3 distinct
# positive lags, tree tips exactly the observed species.
#
# WAVE 2 also adds `homtoep` (drmTMB branch `codex/temporal-homtoep-v1-20260910`,
# semantics only): a free Toeplitz covariance on a complete equally spaced
# panel; see "homogeneous Toeplitz" below.
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
by the column `time`, or a homogeneous Toeplitz (`homtoep`) within-series
covariance on a complete, equally spaced panel.

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

Post-fit conventions (some differ from drmTMB):

- `fitted(fit)` and `predict(fit, newdata)` are POPULATION-level, `Xβ̂` (the
  DRModels convention for every structured term); drmTMB's `fitted()` adds the
  conditional temporal effects. Those are in `ranef(fit)[:id]`, data-row order
  (and `ranef(fit)[:id_iid]` per series for the ordinary intercept).
- `simulate(fit)` draws from the fitted marginal model: a fresh stationary
  chain per series (and a fresh `(1 | id)` intercept) plus residual noise, as
  drmTMB's default `simulate()`. `bootstrap_ci` uses the same draws.
- `vcov`/`stderror`/Wald `confint` cover every coordinate (observed Hessian),
  as on DRModels' other routes; drmTMB exposes only AR1 mean-coefficient Wald
  intervals because the calibration of the others is not established. No
  calibration is claimed here either.

Wave-1 scope: Gaussian family, `sigma ~ 1`, ML only, intercept-only and
unlabelled, at most one ordinary `(1 | id)` on the same `id`. Every other use
is refused with an error.

**Phylogenetic stable intercept + OU (wave 2).** drmTMB's paired provider
`phylo(1 | species, tree = tree) + temporal(1 | species, time = elapsed,
structure = "ou")` is spelled

```julia
drm(bf(@formula(y ~ x + phylo(1 | species) + temporal(1 | species, elapsed, ou)),
       @formula(sigma ~ 1)), Gaussian(); data, tree = newick)
```

and fits `y = Xβ + a_species + b_species(t) + ε` with `a ~ N(0, σ_a² C)` (`C`
the tree's tip **correlation** matrix, the scale drmTMB uses) and an
independent OU path `b` per species. Related species share a stable
baseline; their temporal departures are independent. The stable SD is
`re_sd(fit)[:species_phylo]` (`temporal_parameters(fit).sd_phylo`; drmTMB
`sd_phylo_stable`), the process SD `re_sd(fit)[:species]` (drmTMB
`sd_temporal`) and the rate `temporal_parameters(fit).decay` (drmTMB
`decay_temporal`). `ranef(fit)[:species_phylo]` holds the per-species stable
modes (first-seen order) and `ranef(fit)[:species]` the temporal modes (data
rows). As in drmTMB the pairing needs `ou`, an unlabelled intercept-only
`phylo()` on the same grouping, no ordinary `(1 | species)`, at least three
species with at least two times each, at least three distinct positive lags,
and tree tips that are exactly the observed species. `simulate` draws a fresh
phylogenetic vector, fresh OU paths and fresh noise.

**Homogeneous Toeplitz (wave 2).** `temporal(1 | id, occ, homtoep)` (drmTMB
`structure = "homtoep"`) needs every `id` observed at the same 3–12 equally
spaced integer occasions. Each series' responses then have covariance `σ² R`,
`R` a free positive-definite Toeplitz correlation matrix (one correlation per
lag, `temporal_parameters(fit).cor`, drmTMB `cor_lag1…`), estimated through its
partial autocorrelations (`coef(fit, :temporal_pac)`, atanh scale). `sigma` is
the TOTAL within-series SD: as in drmTMB there is no separate process SD,
residual SD or `(1 | id)` (not identified when every lag is free), and no
latent states (`ranef(fit)` is empty). Incomplete or unequally spaced panels,
fewer than 3 or more than 12 occasions, fractional occasions and an ordinary
`(1 | id)` are refused with drmTMB's messages.
"""
temporal(x...) = x   # marker; intercepted during formula parsing

const _TEMPORAL_SPELLING = "Use `temporal(1 | id, occ, ar1)`, `temporal(1 | id, elapsed, ou)` or " *
    "`temporal(1 | id, occ, homtoep)` (drmTMB: `temporal(1 | id, time = occ, structure = \"ar1\")`; " *
    "StatsModels' `@formula` cannot carry keyword arguments, so the Julia spelling is positional)."

const _TEMPORAL_SCOPE = "temporal() is implemented only for the univariate Gaussian MEAN formula " *
    "(`y ~ … + temporal(1 | id, time, ar1|ou|homtoep)` with `sigma ~ 1`, ML) in this release"

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
    (st isa Term && st.sym in (:ar1, :ou, :homtoep)) ||
        throw(ArgumentError("drm: the structure of `temporal()` must be `ar1`, `ou` or `homtoep`. " *
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
function _temporal_layout(tt, data; has_ordinary::Bool, paired::Bool = false,
                          validate_only::Bool = false)
    sname = uppercase(String(tt.structure))
    discrete = tt.structure in (:ar1, :homtoep)       # integer occasions
    col(nm) = try
        _table_column(data, nm)
    catch
        throw(ArgumentError("drm: Temporal $sname inputs must be columns in `data`; missing " *
            "temporal column `$(nm)`."))
    end
    ids = col(tt.group)
    times = col(tt.time)
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
    discrete && any(v -> abs(v) >= 2.0^53, tv) &&
        throw(ArgumentError("drm: Temporal $sname occasions must be integers of magnitude below " *
            "2^53 (exactly representable); recode `$(tt.time)` relative to an origin."))
    tt.structure === :ar1 && any(v -> v != round(v), tv) &&
        throw(ArgumentError("drm: Temporal AR1 occasions must be finite integers; `$(tt.time)` " *
            "cannot be fractional for `ar1`. Use the original integer sampling occasion (its " *
            "gaps are part of the AR1 model), or `ou` for elapsed time."))
    tt.structure === :homtoep && any(v -> v != round(v), tv) &&
        throw(ArgumentError("drm: Temporal HOMTOEP occasions must be finite integers; " *
            "`$(tt.time)` cannot be fractional for `homtoep`. Use an integer occasion; " *
            "homogeneous Toeplitz is defined by discrete lag."))
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
    # drmTMB validates id / time completeness, type and key uniqueness on ALL
    # rows "before response omission"; the structural rules below then see only
    # the retained rows (the homtoep missing-response path calls this twice).
    validate_only && return nothing
    # Homogeneous Toeplitz (wave 2): drmTMB's panel rules — 3 to 12 common,
    # equally spaced occasions, every series observing all of them.
    occasions = nothing
    if tt.structure === :homtoep
        occasions = sort(unique(tv))
        K = length(occasions)
        K <= 12 ||
            throw(ArgumentError("drm: Temporal HOMTOEP supports at most 12 common occasions; found " *
                "$K distinct retained occasions. Use AR1 or OU for a longer time series, or " *
                "predeclare a coarser common schedule."))
        K >= 3 ||
            throw(ArgumentError("drm: Temporal HOMTOEP needs at least three common occasions; found " *
                "$K retained occasions. Use AR1 or OU when the design has fewer than three " *
                "repeated occasions."))
        length(unique(diff(occasions))) == 1 ||
            throw(ArgumentError("drm: Temporal HOMTOEP occasions must be equally spaced; retained " *
                "occasions are $(occasions). Use OU for irregular elapsed time."))
        incomplete = [s for s in 1:S if tv[rows[s]] != occasions]
        isempty(incomplete) ||
            throw(ArgumentError("drm: Temporal HOMTOEP requires every ID to retain the complete " *
                "retained schedule; incomplete series: " *
                "$(join(string.(unique(ids)[incomplete[1:min(end, 10)]]), ", "))" *
                (length(incomplete) > 10 ? ", … ($(length(incomplete)) in all)" : "") *
                ". Use OU or AR1 for incomplete repeated records, or retain a common complete panel."))
        # drmTMB #1449: the K − 1 free lag correlations need at least K series
        # (the lag-(K−1) correlation rests on one pair per series). A floor,
        # not a design recommendation.
        S >= K ||
            throw(ArgumentError("drm: Temporal HOMTOEP needs at least as many series as occasions; " *
                "found $S series for $K occasions. Collect more series, use fewer common occasions, " *
                "or fit AR1 or OU, which estimate one persistence or decay parameter."))
    end
    # Paired phylo() + OU (wave 2): drmTMB's support checks for the stable
    # between-species field — at least three species, each with at least two
    # distinct times (keys are unique, so two rows) — before the lag check.
    if paired
        S >= 3 ||
            throw(ArgumentError("drm: The paired phylogenetic-temporal OU model requires at least " *
                "three observed species; supply repeated observations from at least three tree tips."))
        short = [s for s in 1:S if length(rows[s]) < 2]
        isempty(short) ||
            throw(ArgumentError("drm: Each species in the paired phylogenetic-temporal OU model " *
                "needs at least two distinct times; insufficient time variation for " *
                "$(join(string.(unique(ids)[short]), ", "))."))
    end
    # drmTMB's identifiability check on the distinct positive within-series
    # lags (all pairs, not only consecutive gaps): at least 2 (3 with an
    # ordinary intercept or a paired phylo() field), and for AR1 at least one
    # odd lag (the sign of φ).
    # The pair scan stops as soon as the requirement is met, so a long series
    # does not cost O(n²).
    required = (has_ordinary || paired) ? 3 : 2
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
    ok || !paired || throw(ArgumentError("drm: The paired phylogenetic-temporal OU model requires " *
        "at least three distinct positive lags; found $(distinct). Keep genuine elapsed-time gaps " *
        "and collect more distinct within-species intervals."))
    ok || throw(ArgumentError("drm: Temporal $sname occasions do not provide the required lag " *
        "variation: found distinct positive lags $(distinct); this model needs at least " *
        "$required" * (tt.structure === :ar1 ? ", including an odd lag" : "") * ". Keep genuine " *
        "sampling gaps and collect more distinct within-series occasions."))
    has_ordinary && S < 2 &&
        throw(ArgumentError("drm: a temporal $sname model with an ordinary random intercept " *
            "requires multiple series; fit the temporal process without `(1 | $(tt.group))` " *
            "for one series, or provide observations from at least two ids."))
    return (rows = rows, gap = gap, nseries = S, gidx = gidx, levels = unique(ids),
            occasions = occasions)
end

using Statistics: var, median, std
using Printf: @sprintf

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
        # Large gap: φ^g and 1 − φ^{2g} from log tanh|θ| = log1p(−e^{−2|θ|}) −
        # log1p(e^{−2|θ|}), which stays accurate where tanh θ itself rounds to ±1
        # (|θ| ≳ 19); the direct form there gave 1 − a² = 0 and a flat 1e18
        # objective. Near θ = 0 the direct form is exact and keeps the AD
        # derivative finite.
        abs(ψ) < 1 && return a, one(a) - a * a
        x = abs(ψ)
        lt = log1p(-exp(-2x)) - log1p(exp(-2x))          # log tanh|θ| < 0
        a = (ψ < 0 && isodd(g) ? -one(x) : one(x)) * exp(g * lt)
        return a, -expm1(2g * lt)
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

# --- paired phylo() stable field (wave 2) ----------------------------------
#
# The stable between-species field is a_s = σ_a ũ_s / √h_s, ũ a unit-rate
# Brownian motion on the tree (root fixed at 0, so ũ ~ N(0, Q⁻¹) with Q the
# AugmentedPhy precision without the root) and h_s the root-to-tip depth of
# species s. Cov(a) = σ_a² D^{-1/2} Σ D^{-1/2}: the tip CORRELATION matrix, the
# scale drmTMB (`ape::vcv(tree, corr = TRUE)`) and DRModels' closed-form
# `phylo(1 | g)` mean route (`_phylo_correlation`) both use.
#
# Given the per-series temporal-plus-residual blocks W_s (wave 1's filter), a
# species' rows depend on a only through a_s 1, so the series reduces to a
# Gaussian factor in a_s with precision c_s = 1ᵀW_s⁻¹1 and linear term
# d_s = 1ᵀW_s⁻¹r_s — exactly the `with_one` sums the filter already returns
# for the ordinary intercept. Integrating ũ out over the tree is then an
# upward (pruning) pass: a node carries a message exp(−½αx² + βx); a leaf
# starts at α = σ_a² c_s / h_s, β = σ_a d_s / √h_s, crossing a branch of
# length ℓ maps (α, β) → (α, β)/(1 + ℓα), siblings add, and the root sits at
# 0. Then, exactly,
#     logdet V = Σ_s logdet W_s + Σ_branches log1p(ℓα),
#     rᵀV⁻¹r  = Σ_s rᵀW_s⁻¹r − Σ_branches ℓβ²/(1 + ℓα).
# This is the sparse Cholesky of the joint (tree + series) precision in its
# zero-fill elimination order (each series' OU states, then leaves → root),
# written as scalar recursions: O(n) per evaluation, generic in the number
# type (ForwardDiff runs through it) and made of positive terms only, so it
# stays accurate as σ_a → 0 (log1p, α, β → 0) and as σ_a → ∞.

# The tree as parent / branch-length arrays. A Newick string is read with the
# shared Newick reader (`_parse_node!`) but WITHOUT the sparse-precision
# assembly, so zero-length branches are admitted here, as in drmTMB (which
# requires finite, non-negative lengths): a zero branch has no precision 1/ℓ,
# but in the pruning pass it is just ℓ = 0 (log1p(0) = 0, the message passes
# through unchanged). An `AugmentedPhy` (lengths > 0 by construction) is read
# back from its precision. A branch length absent from the Newick reads as 0.
function _temporal_phylo_tree(tree)
    if tree isa AugmentedPhy
        Q = tree.Q_topology; N = tree.n_total
        parent = zeros(Int, N); blen = zeros(N); seen = falses(N)
        order = [tree.root_index]; seen[tree.root_index] = true; cursor = 1
        while cursor <= length(order)
            i = order[cursor]; cursor += 1
            for k in nzrange(Q, i)
                j = rowvals(Q)[k]
                (j == i || seen[j]) && continue
                qij = nonzeros(Q)[k]
                qij < 0 || continue
                seen[j] = true; parent[j] = i
                blen[j] = -1.0 / qij              # off-diagonal of Q is −1/branch length
                push!(order, j)
            end
        end
        length(order) == N || error("drm: internal — the phylogeny is not connected")
        return (parent = parent, blen = blen, root = tree.root_index, n = N,
                leaf_names = tree.leaf_names, leaf_nodes = tree.leaf_indices)
    end
    tree isa AbstractString ||
        throw(ArgumentError("drm: `tree` must be a Newick string or an `AugmentedPhy`."))
    str = String(tree)
    occursin('\0', str) && throw(ArgumentError("Newick strings cannot contain literal NUL bytes"))
    c = _NewickCursor(str, firstindex(str))
    _skip_newick_whitespace!(c)
    parent = Int[]; is_leaf = Bool[]; name = String[]; len = Float64[]
    leaf_nodes = Int[]; leaf_names = String[]
    root = _parse_node!(c, parent, is_leaf, name, len, leaf_nodes, leaf_names)
    _skip_newick_whitespace!(c)
    _peek(c) == ';' || throw(ArgumentError("Newick string must end with ';' at position $(c.i)"))
    _advance(c); _skip_newick_whitespace!(c)
    _peek(c) == '\0' || throw(ArgumentError("extra characters after end of tree at position $(c.i)"))
    leaf_names = _phy_validate_leaf_names(leaf_names, length(leaf_nodes))
    _phy_validate_topology(parent, is_leaf)
    for v in eachindex(parent)
        v == root && continue
        (isfinite(len[v]) && len[v] >= 0) ||
            throw(ArgumentError("drm: `tree` branch lengths must be finite and non-negative " *
                "(got $(len[v]) above node $(v == 0 ? "?" : name[v] == "" ? "#$v" : name[v]))."))
    end
    blen = copy(len); blen[root] = 0.0                   # the root's own length is ignored
    return (parent = parent, blen = blen, root = root, n = length(parent),
            leaf_names = leaf_names, leaf_nodes = leaf_nodes)
end

# Upward/downward plan for the tree: `post` lists every non-root node with its
# children before it, `parent`/`blen` its parent and branch length, `leaf` the
# tree node of series s and `h` that leaf's root-to-tip depth. The user route
# requires an ultrametric tree, as drmTMB does (its tolerance: tip depths within
# sqrt(eps) × max(1, |height|, |depths|)); the likelihood itself is exact on
# any tree, and the tests exercise it there with `require_ultrametric = false`.
function _temporal_phylo_plan(tr, levels; require_ultrametric::Bool = true)
    N = tr.n
    children = [Int[] for _ in 1:N]
    for v in 1:N
        v == tr.root && continue
        push!(children[tr.parent[v]], v)
    end
    order = [tr.root]; cursor = 1
    depth = zeros(N)
    while cursor <= length(order)
        i = order[cursor]; cursor += 1
        for j in children[i]
            depth[j] = depth[i] + tr.blen[j]
            push!(order, j)
        end
    end
    length(order) == N || error("drm: internal — the phylogeny is not connected")
    by_name = Dict(tr.leaf_names[t] => tr.leaf_nodes[t] for t in eachindex(tr.leaf_names))
    leaf = [by_name[string(l)] for l in levels]
    tipdepth = depth[tr.leaf_nodes]
    if require_ultrametric
        # Message wording mirrors drmTMB's user-facing text for twin parity
        # (drmTMB is copyright Shinichi Nakagawa; used with the owner's permission).
        hgt = tipdepth[1]
        tol = sqrt(eps(Float64)) * max(1.0, abs(hgt), maximum(abs, tipdepth))
        spread = maximum(abs.(tipdepth .- hgt))
        spread <= tol ||
            throw(ArgumentError("drm: `tree` must be ultrametric: root-to-tip distances differ " *
                "by $(spread), above the tolerance of $(tol) for this tree (sqrt(eps) scaled by " *
                "the tree height, so relative, not absolute). If the difference is negligible for " *
                "your tree, equalise the tip depths (for example `phytools::force.ultrametric()` " *
                "in R) and refit; a large difference is a real problem with the tree."))
    end
    h = depth[leaf]
    all(>(0), h) || throw(ArgumentError("drm: `tree` must have positive root-to-tip height."))
    return (post = reverse(order[2:end]), parent = tr.parent, blen = tr.blen, leaf = leaf, h = h,
            root = tr.root, n = N)
end

# Upward pass. `α0`, `β0` are the per-series leaf messages. Returns
# (Σ log1p(ℓα), Σ ℓβ²/(1+ℓα)) and, with `keep`, the subtree messages α, β.
function _temporal_phylo_prune(plan, α0, β0; keep::Bool = false)
    T = promote_type(eltype(α0), eltype(β0))
    α = zeros(T, plan.n); β = zeros(T, plan.n)
    for s in eachindex(plan.leaf)
        α[plan.leaf[s]] += α0[s]; β[plan.leaf[s]] += β0[s]
    end
    ld = zero(T); qc = zero(T)
    for v in plan.post
        ℓ = plan.blen[v]
        den = 1 + ℓ * α[v]
        ld += log1p(ℓ * α[v])
        qc += ℓ * β[v]^2 / den
        p = plan.parent[v]
        α[p] += α[v] / den
        β[p] += β[v] / den
    end
    return keep ? (ld, qc, α, β) : (ld, qc)
end

# Downward pass: E[ũ | y] at every node from the upward messages,
# E[ũ_v | ũ_parent, y] = (ũ_parent + ℓβ_v) / (1 + ℓα_v), linear in ũ_parent.
function _temporal_phylo_down(plan, α, β)
    x = zeros(eltype(α), plan.n)
    for v in Iterators.reverse(plan.post)
        ℓ = plan.blen[v]
        x[v] = (x[plan.parent[v]] + ℓ * β[v]) / (1 + ℓ * α[v])
    end
    return x
end

# One fresh draw of the stable field a (per series): Brownian motion down the
# tree, then the tip-wise correlation scaling.
function _temporal_phylo_draw(plan, σa, rng)
    x = zeros(plan.n)
    for v in Iterators.reverse(plan.post)
        x[v] = x[plan.parent[v]] + sqrt(plan.blen[v]) * randn(rng)
    end
    return σa .* x[plan.leaf] ./ sqrt.(plan.h)
end

# --- homogeneous Toeplitz (wave 2) -----------------------------------------
#
# `temporal(1 | id, occ, homtoep)`: every series observes the same K (3–12)
# equally spaced occasions, and its K responses have covariance σ² R with R a
# free positive-definite Toeplitz CORRELATION matrix, R_{ij} = ρ_{|i−j|}. σ is
# the TOTAL within-series SD (`sigma`): with one response per series–occasion,
# a separate process SD, residual SD or `(1 | id)` would not be identified
# when every lag correlation is free (drmTMB's decision, mirrored), so there
# are none. R is parameterised by its partial autocorrelations
# π_m = tanh κ_m ∈ (−1, 1), m = 1…K−1, which map one-to-one onto the
# positive-definite Toeplitz correlation matrices (Durbin–Levinson).
#
# Engine: the Durbin–Levinson recursion gives the order-m one-step predictor
# coefficients φ_{m,j} (φ_{m,m} = π_m, φ_{m,j} = φ_{m−1,j} − π_m φ_{m−1,m−j})
# and innovation variances v_m = Π_{j≤m} (1 − π_j²), so with
# e_t = r_t − Σ_{j<t} φ_{t−1,j} r_{t−j}
#     logdet(σ²R) = Σ_t [log σ² + log v_{t−1}],  rᵀ(σ²R)⁻¹r = Σ_t e_t² / (σ² v_{t−1}).
# That is the exact Cholesky (innovations) factorisation of R⁻¹, O(K²) per
# series; log v is a sum of −2 log cosh κ_j, formed without the 1 − tanh²
# cancellation, so it stays accurate as |π| → 1. Lag correlations follow from
# ρ_m = Σ_{j<m} φ_{m−1,j} ρ_{m−j} + π_m v_{m−1}.

# −2 log cosh κ = log(1 − tanh² κ), cancellation-free for any κ.
@inline _log1m_tanh2(κ) = (a = abs(κ); -2 * (a + log1p(exp(-2a)) - log(one(a) + one(a))))

# Predictor coefficients (K−1 rows, row m holds φ_{m,1..m}) and log v_0…v_{K−1}.
function _homtoep_levinson(κ)
    T = eltype(κ)
    K = length(κ) + 1
    Φ = zeros(T, K - 1, K - 1)
    logv = zeros(T, K)
    for m in 1:(K-1)
        pac = tanh(κ[m])
        Φ[m, m] = pac
        for j in 1:(m-1)
            Φ[m, j] = Φ[m-1, j] - pac * Φ[m-1, m-j]
        end
        logv[m+1] = logv[m] + _log1m_tanh2(κ[m])
    end
    return Φ, logv
end

# Lag correlations ρ_1…ρ_{K−1} of the Toeplitz matrix with PACs tanh κ.
function _homtoep_correlations(κ)
    Φ, logv = _homtoep_levinson(κ)
    K = length(κ) + 1
    ρ = zeros(eltype(Φ), K)
    ρ[1] = 1
    for m in 1:(K-1)
        acc = Φ[m, m] * exp(logv[m])
        for j in 1:(m-1)
            acc += Φ[m-1, j] * ρ[m-j+1]
        end
        ρ[m+1] = acc
    end
    return ρ[2:end]
end

# −2 log-likelihood contribution of one complete series r (occasion order),
# without the 2π constant.
function _homtoep_series(Φ, logv, logσ2, r)
    K = length(r)
    T = promote_type(eltype(Φ), typeof(logσ2), eltype(r))
    tot = zero(T)
    for t in 1:K
        e = r[t]
        for j in 1:(t-1)
            e -= Φ[t-1, j] * r[t-j]
        end
        lv = logσ2 + logv[t]
        tot += lv + e^2 * exp(-lv)
    end
    return tot
end

# θ = [βμ; log σ; κ_1 … κ_{K−1}]
function _fit_temporal_homtoep(fam::Gaussian, y, Xμ, Xσ, nmμ, nmσ, tt, lay, g_tol)
    n = length(y)
    pμ = size(Xμ, 2)
    size(Xσ, 2) == 1 || error("drm: internal — temporal route needs `sigma ~ 1`")
    K = length(lay.occasions)
    S = lay.nseries
    grows = lay.rows                                   # each in occasion order
    iσ = pμ + 1
    iκ = (pμ + 2):(pμ + K)
    np = pμ + K

    function nll(θ)
        T = eltype(θ)
        r = y .- Xμ * θ[1:pμ]
        Φ, logv = _homtoep_levinson(θ[iκ])
        logσ2 = 2 * θ[iσ]
        total = zero(T)
        for s in 1:S
            total += _homtoep_series(Φ, logv, logσ2, view(r, grows[s]))
        end
        isfinite(total) || return convert(T, 1e18)
        return 0.5 * total + 0.5 * n * log(2π)
    end

    # Starts: OLS β, the residual SD as the total SD, and PACs atanh(0.3) (as
    # drmTMB) or 0 (independence), keeping the lower objective.
    βμ0 = Xμ \ y
    v0 = var(y .- Xμ * βμ0)
    (isfinite(v0) && v0 > 0) || (v0 = 1.0)
    best = nothing
    for κ0 in (atanh(0.3), 0.0)
        θ0 = zeros(np)
        θ0[1:pμ] .= βμ0
        θ0[iσ] = log(sqrt(v0))
        θ0[iκ] .= κ0
        res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
        (best === nothing || Optim.minimum(res) < Optim.minimum(best)) && (best = res)
    end
    θ̂ = Optim.minimizer(best)
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))

    blocks = [:mu => 1:pμ, :sigma => iσ:iσ, :temporal_pac => iκ]
    names = [:mu => nmμ, :sigma => nmσ, :temporal_pac => ["pac_lag$(m)" for m in 1:(K-1)]]
    means = Dict(:mu => Xμ * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => fill(exp(θ̂[iσ]), n))
    info = (label = _temporal_label(tt), structure = :homtoep, group = tt.group,
            time = tt.time, nseries = S, rows = grows, gaps = [lay.gap[r] for r in grows],
            has_ordinary = false, phylo = nothing, levels = lay.levels,
            occasions = lay.occasions)
    fit = DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(best), means, obs, scales)
    # No latent temporal states: the Toeplitz block IS the marginal covariance.
    return _withranef(_withnll(fit, nll), (effects = Dict{Symbol,Vector{Float64}}(), temporal = info))
end

# θ = [βμ; log σ; (log σ_b | log σ_a); log σ_t; θ_temporal]. `phylo` is the
# tree plan of a paired phylo() + OU fit (wave 2), else `nothing`; its stable
# SD σ_a takes the slot of the ordinary intercept SD σ_b (the two are never
# combined — drmTMB refuses `(1 | species)` alongside the paired field).
function _fit_temporal_gaussian(fam::Gaussian, y, Xμ, Xσ, nmμ, nmσ, tt, lay, has_ordinary, g_tol;
                                phylo = nothing)
    n = length(y)
    pμ = size(Xμ, 2)
    size(Xσ, 2) == 1 || error("drm: internal — temporal route needs `sigma ~ 1`")
    paired = phylo !== nothing
    (paired && has_ordinary) && error("drm: internal — paired phylo() fit with an ordinary intercept")
    has_stable = has_ordinary || paired
    structure = tt.structure
    S = lay.nseries
    grows = lay.rows
    ggap = [lay.gap[r] for r in grows]
    iσ = pμ + 1
    ib = has_stable ? pμ + 2 : 0
    it = pμ + (has_stable ? 3 : 2)
    iψ = it + 1
    np = iψ
    # ψ handed to the kernels: θ itself for AR1 (φ = tanh θ is formed there so
    # that 1 − φ² = sech²θ is exact), λ = exp θ for OU.
    ψof(θ) = structure === :ar1 ? θ[iψ] : exp(θ[iψ])
    invsqrth = paired ? 1 ./ sqrt.(phylo.h) : nothing

    function nll(θ)
        T = eltype(θ)
        r = y .- Xμ * θ[1:pμ]
        σ2 = exp(2 * θ[iσ]); st2 = exp(2 * θ[it])
        sb2 = has_ordinary ? exp(2 * θ[ib]) : nothing
        ψ = ψof(θ)
        total = zero(T)
        if paired
            σa = exp(θ[ib])
            α0 = zeros(T, S); β0 = zeros(T, S)
            for s in 1:S
                out = _temporal_filter(structure, ψ, σ2, st2, ggap[s], r[grows[s]]; with_one = true)
                out === nothing && return convert(T, 1e18)
                ldF, q, c, d, _ = out
                total += ldF + q
                α0[s] = σa^2 * c * invsqrth[s]^2
                β0[s] = σa * d * invsqrth[s]
            end
            ld, qc = _temporal_phylo_prune(phylo, α0, β0)
            total += ld - qc
        else
            for s in 1:S
                out = _temporal_series(structure, ψ, σ2, st2, sb2, ggap[s], r[grows[s]])
                out === nothing && return convert(T, 1e18)
                total += out[1] + out[2]
            end
        end
        isfinite(total) || return convert(T, 1e18)
        return 0.5 * total + 0.5 * n * log(2π)
    end

    # Starts (as drmTMB): OLS β, the residual variance split evenly over the
    # residual + temporal (+ intercept / phylo) components, and two persistence
    # starts (AR1: φ = ±0.3; OU: correlation 0.3 / 0.7 over the median positive
    # gap), keeping the lower objective.
    βμ0 = Xμ \ y
    v0 = var(y .- Xμ * βμ0)
    (isfinite(v0) && v0 > 0) || (v0 = 1.0)
    ncomp = has_stable ? 3 : 2
    base = zeros(np)
    base[1:pμ] .= βμ0
    base[iσ] = log(sqrt(v0 / ncomp))
    has_stable && (base[ib] = log(sqrt(v0 / ncomp)))
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
    if paired
        # Stable field: upward messages, then the downward pass for E[ũ | y];
        # a_s = σ_a E[ũ_leaf] / √h_s. Given a, the series are independent, so
        # the temporal mode is the smoothed state of r_s − a_s 1.
        σa = exp(θ̂[ib])
        α0 = zeros(S); β0 = zeros(S)
        for s in 1:S
            _, _, c, d, _ = _temporal_filter(structure, ψ̂, σ2, st2, ggap[s], r̂[grows[s]]; with_one = true)
            α0[s] = σa^2 * c * invsqrth[s]^2
            β0[s] = σa * d * invsqrth[s]
        end
        _, _, αm, βm = _temporal_phylo_prune(phylo, α0, β0; keep = true)
        x = _temporal_phylo_down(phylo, αm, βm)
        bmodes .= σa .* x[phylo.leaf] .* invsqrth
        for s in 1:S
            sm = _temporal_filter(structure, ψ̂, σ2, st2, ggap[s], r̂[grows[s]] .- bmodes[s]; keep = true)
            teff[grows[s]] .= _temporal_smooth(sm[5])
        end
    else
        for s in 1:S
            _, _, te, bm = _temporal_series(structure, ψ̂, σ2, st2, sb2, ggap[s], r̂[grows[s]]; modes = true)
            teff[grows[s]] .= te
            bmodes[s] = bm
        end
    end

    grp = String(tt.group)
    stable_name = paired ? "$(grp)_phylo" : "$(grp)_iid"
    resd_names = has_stable ? [stable_name, grp] : [grp]
    pblock = structure === :ar1 ? :temporal_phi : :temporal_decay
    blocks = [:mu => 1:pμ, :sigma => iσ:iσ,
              :resd => (has_stable ? (ib:it) : (it:it)), pblock => iψ:iψ]
    # drmTMB names the paired fit's rate `decay_temporal` (its scientific
    # meaning) and keeps the formula label for the independent-series fits.
    names = [:mu => nmμ, :sigma => nmσ, :resd => resd_names,
             pblock => [paired ? "decay_temporal" : _temporal_label(tt)]]
    means = Dict(:mu => Xμ * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => fill(exp(θ̂[iσ]), n))
    effects = Dict{Symbol,Vector{Float64}}(tt.group => teff)
    has_stable && (effects[Symbol(stable_name)] = bmodes)
    info = (label = _temporal_label(tt), structure = structure, group = tt.group,
            time = tt.time, nseries = S, rows = grows, gaps = ggap,
            has_ordinary = has_ordinary, phylo = phylo, levels = lay.levels, occasions = nothing)
    fit = DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(best), means, obs, scales)
    return _withranef(_withnll(fit, nll), (effects = effects, temporal = info))
end

# The refusal messages of the temporal routes intentionally mirror drmTMB's
# user-facing wording for twin parity (drmTMB is copyright Shinichi Nakagawa;
# used with the owner's permission). No drmTMB R logic is copied.
const _PHYLO_TEMPORAL_SPELLING = "Use `phylo(1 | species) + temporal(1 | species, elapsed, ou)` " *
    "with `drm(...; tree = tree)` (drmTMB: `phylo(1 | species, tree = tree) + " *
    "temporal(1 | species, time = elapsed, structure = \"ou\")`)."

# Router: validate the wave-1 / wave-2 scope, then fit. Called from
# `_drm_gaussian_fit` before any other route can claim the formula. The one
# structured partner admitted is drmTMB's paired provider: an unlabelled
# `phylo(1 | species)` stable intercept on the SAME grouping as an OU
# `temporal()` term (wave 2, src/temporal.jl header).
function _drm_gaussian_temporal(f, fam::Gaussian, tt, re, metav, structured, sigma_re,
        structured_sigma, y, Xμ, Xσ, nmμ, nmσ, data; method, algorithm, penalty,
        phylo_coupled, sparse, has_missing_response, g_tol, structured_slope = nothing,
        tree = nothing)
    lbl = _temporal_label(tt)
    nope(what) = throw(ArgumentError("drm: `$lbl` cannot be combined with $what; " *
        _TEMPORAL_SCOPE * "."))
    length(_collect_temporal(Dict(f.forms)[:mu])) == 1 ||
        throw(ArgumentError("drm: only one temporal effect is implemented in `mu`. " * _TEMPORAL_SPELLING))
    method === :ML || nope("`method = :$method` (temporal fits are ML only; REML is not implemented)")
    all_structured = _collect_structured(Dict(f.forms)[:mu])
    paired = !isempty(all_structured)
    if paired
        (length(all_structured) == 1 && all_structured[1][1] === :phylo) ||
            nope("$(length(all_structured) == 1 ? "a `$(all_structured[1][1])(…)` structured effect" :
                 "more than one structured effect"); the only structured partner implemented is " *
                 "the paired `phylo(1 | species)` stable intercept with an OU temporal term")
        tt.structure === :ou ||
            throw(ArgumentError("drm: The paired `phylo()` plus `temporal()` provider requires " *
                "structure `ou`. Use independent AR1 without `phylo()`, or " *
                "`phylo(1 | species) + temporal(1 | species, elapsed, ou)` for this combined slice. " *
                _PHYLO_TEMPORAL_SPELLING))
        structured_slope === nothing ||
            throw(ArgumentError("drm: The paired phylogenetic-temporal provider requires an " *
                "unlabelled `phylo(1 | species)` intercept; phylogenetic slopes and covariance-block " *
                "labels are deferred for this combined slice."))
        pgrp = all_structured[1][2]
        pgrp === tt.group ||
            throw(ArgumentError("drm: Paired `phylo()` and `temporal()` terms must use the same " *
                "grouping ID; the phylogenetic group is `$(pgrp)` but the temporal group is " *
                "`$(tt.group)`. " * _PHYLO_TEMPORAL_SPELLING))
        isempty(re) ||
            throw(ArgumentError("drm: The paired phylogenetic-temporal OU model does not allow an " *
                "ordinary random intercept; the stable between-species component is already " *
                "`phylo(1 | $(tt.group))`."))
        tree === nothing &&
            throw(ArgumentError("drm: `phylo(1 | $(tt.group))` with `temporal()` needs the tree: " *
                "pass `drm(...; tree = tree)` (a Newick string or an `AugmentedPhy`)."))
    end
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
    # Missing responses: the homtoep route mirrors drmTMB (validate id / time on
    # all rows, drop the missing-response rows, then apply the panel rules to the
    # retained rows). The other temporal routes still refuse them.
    if has_missing_response && tt.structure === :homtoep
        _temporal_layout(tt, data; has_ordinary = false, validate_only = true)
        keep = _observed_response_mask(y)
        @warn "drm: $(count(!, keep)) of $(length(keep)) rows have a missing response and were " *
              "dropped before the homogeneous Toeplitz panel rules were applied (drmTMB's response " *
              "omission); a series that loses an occasion is then incomplete and is refused."
        y = Float64.(y[keep]); Xμ = Xμ[keep, :]; Xσ = Xσ[keep, :]
        data = NamedTuple{(tt.group, tt.time)}((_table_column(data, tt.group)[keep],
                                                _table_column(data, tt.time)[keep]))
        has_missing_response = false
    end
    has_missing_response && nope("missing responses (drop the missing-response rows before " *
        "calling `drm`, e.g. with `drm_listwise`)")
    has_ordinary = false
    tt.structure === :homtoep && !isempty(re) &&
        throw(ArgumentError("drm: Temporal HOMTOEP currently does not allow an ordinary random " *
            "intercept. Fit the direct temporal process alone while the first Toeplitz provider " *
            "is validated."))
    if !isempty(re)
        length(re) == 1 || nope("more than one ordinary random effect")
        rl, g = re[1]
        kind = try
            first(_re_kind(rl))
        catch
            throw(ArgumentError("drm: the ordinary random effect paired with `$lbl` must be " *
                "`(1 | $(tt.group))`; `($(rl) | $(g))` (a labelled, correlated or otherwise " *
                "non-intercept bar) is not implemented with temporal()."))
        end
        (kind === :intercept && g === tt.group) ||
            throw(ArgumentError("drm: the ordinary random effect paired with `$lbl` must be " *
                "`(1 | $(tt.group))` using the same id; use either no ordinary random effect " *
                "or `(1 | $(tt.group))`."))
        has_ordinary = true
    end
    lay = _temporal_layout(tt, data; has_ordinary = has_ordinary, paired = paired)
    tt.structure === :homtoep &&
        return _fit_temporal_homtoep(fam, y, Xμ, Xσ, nmμ, nmσ, tt, lay, g_tol)
    paired || return _fit_temporal_gaussian(fam, y, Xμ, Xσ, nmμ, nmσ, tt, lay, has_ordinary, g_tol)
    # drmTMB: the tree tips must be exactly the observed species (by name).
    tr = _temporal_phylo_tree(tree)
    obs = string.(lay.levels)
    extra = setdiff(tr.leaf_names, obs)           # tips with no observations
    absent = setdiff(obs, tr.leaf_names)          # observed species not in the tree
    (isempty(extra) && isempty(absent)) ||
        throw(ArgumentError("drm: The paired phylogenetic-temporal OU model requires tree tips to " *
            "match the observed species. " *
            (isempty(extra) ? "" : "The tree has $(length(extra)) tip(s) with no observations, for " *
                "example $(join(first(extra, 3), ", ")); prune it to the observed species first " *
                "(R: `ape::keep.tip(tree, unique(species))`). ") *
            (isempty(absent) ? "" : "$(length(absent)) observed species are not tree tips, for " *
                "example $(join(first(absent, 3), ", ")). ") *
            "Supply data and a tree for the same set of species before fitting."))
    plan = _temporal_phylo_plan(tr, obs)
    return _withphyloscale(_fit_temporal_gaussian(fam, y, Xμ, Xσ, nmμ, nmσ, tt, lay, false, g_tol;
                                                  phylo = plan), :correlation)
end

"""
    temporal_parameters(fit) -> NamedTuple

The fitted temporal process of a `temporal(...)` Gaussian fit, on natural
scales: `label` (drmTMB's term label), `structure` (`:ar1` / `:ou`), `sd`
(process SD σ_t; drmTMB `sdpars\$mu["temporal_sd: <label>"]`), `phi` (AR1
persistence, one occasion apart; drmTMB `corpars\$temporal`) or `decay` (OU
rate λ in the units of `time`; drmTMB `decaypars\$temporal`) — the other is
`nothing` — `sd_iid` (the ordinary `(1 | id)` SD, or `nothing`), `sd_phylo`
(the stable phylogenetic SD of a paired `phylo(1 | species) + temporal(…, ou)`
fit, on the tip-correlation scale; drmTMB `sdpars\$mu["sd_phylo_stable"]`, or
`nothing`) and `sigma` (residual SD). For the paired fit drmTMB names the
process SD `sd_temporal` and the rate `decay_temporal`. For a homogeneous
Toeplitz fit (`homtoep`) `sigma` is the TOTAL within-series SD, `sd`, `phi`,
`decay`, `sd_iid` and `sd_phylo` are `nothing`, `cor` holds the lag
correlations ρ_1…ρ_{K−1} (drmTMB `corpars\$temporal`, `cor_lag1…`) and `pac`
the partial autocorrelations; both are `nothing` for the other structures.
Errors on a fit without a temporal term.
"""
function temporal_parameters(fit::DrmFit)
    info = fit.ranef isa NamedTuple && haskey(fit.ranef, :temporal) ? fit.ranef.temporal : nothing
    info === nothing &&
        throw(ArgumentError("temporal_parameters: this fit has no `temporal(...)` term."))
    σ = exp(only(coef(fit, :sigma)))
    if info.structure === :homtoep
        κ = coef(fit, :temporal_pac)
        return (label = info.label, structure = :homtoep, sd = nothing, phi = nothing,
                decay = nothing, sd_iid = nothing, sd_phylo = nothing, sigma = σ,
                cor = _homtoep_correlations(κ), pac = tanh.(κ))
    end
    sds = re_sd(fit)
    g = info.group
    iid = Symbol("$(g)_iid")
    ψ = only(coef(fit, info.structure === :ar1 ? :temporal_phi : :temporal_decay))
    return (label = info.label, structure = info.structure, sd = sds[g],
            phi = info.structure === :ar1 ? tanh(ψ) : nothing,
            decay = info.structure === :ou ? exp(ψ) : nothing,
            sd_iid = get(sds, iid, nothing),
            sd_phylo = get(sds, Symbol("$(g)_phylo"), nothing),
            sigma = σ, cor = nothing, pac = nothing)
end

# --- post-fit helpers ------------------------------------------------------

_is_temporal_fit(fit) = fit.ranef isa NamedTuple && haskey(fit.ranef, :temporal)

# `bf` guard: a temporal() term is only implemented on the univariate `mu`
# formula; refuse it by name anywhere else (bivariate `mu1`/`mu2`, `sigma`,
# `nu`, `rho12`, `sd(...)` …) before any design is built.
function _temporal_check_forms(forms, allowed)
    for (k, rhs) in forms
        terms = rhs isa Tuple ? collect(rhs) : Any[rhs]
        any(t -> t isa FunctionTerm && t.f === temporal, terms) || continue
        k in allowed && continue
        throw(ArgumentError("bf: `temporal()` in the `$(k)` formula is not implemented; " *
            _TEMPORAL_SCOPE * " (bivariate models and non-mean parameters are refused)."))
    end
    return nothing
end

# One draw from the fitted MARGINAL model, in data-row order: Xβ̂ + (for a
# paired fit) a fresh phylogenetic stable vector + a fresh
# stationary chain per series (s_1 ~ N(0, σ_t²), s_k = a_k s_{k−1} +
# σ_t √(1 − a_k²) z_k, with a_k = φ^gap or e^{−λΔt}) + a fresh `(1 | id)`
# intercept when present + N(0, σ²) noise — drmTMB's default `simulate()`
# (`drm_fresh_temporal_mu_values`).
function _temporal_simulate(fit::DrmFit, rng)
    info = fit.ranef.temporal
    info.structure === :homtoep && return _homtoep_simulate(fit, info, rng)
    ψ = only(coef(fit, info.structure === :ar1 ? :temporal_phi : :temporal_decay))
    ψk = info.structure === :ar1 ? ψ : exp(ψ)
    sds = re_sd(fit)
    st = sds[info.group]
    sb = info.has_ordinary ? sds[Symbol("$(info.group)_iid")] : 0.0
    σ = exp(only(coef(fit, :sigma)))
    y = copy(fit.means[:mu])
    # Paired phylo() fit: one fresh stable phylogenetic vector per draw (drmTMB
    # `drm_structured_mu_random_effect_draws`), drawn before the chains.
    stable = info.phylo === nothing ? nothing :
        _temporal_phylo_draw(info.phylo, sds[Symbol("$(info.group)_phylo")], rng)
    for (si, (rows, g)) in enumerate(zip(info.rows, info.gaps))
        b = stable !== nothing ? stable[si] : info.has_ordinary ? sb * randn(rng) : 0.0
        s = st * randn(rng)
        y[rows[1]] += s + b
        for k in 2:length(rows)
            a, v = _temporal_transition(info.structure, ψk, g[k])
            s = a * s + st * sqrt(max(v, 0.0)) * randn(rng)
            y[rows[k]] += s + b
        end
    end
    y .+= σ .* randn(rng, length(y))
    return y
end

# --- temporal boundary diagnostic (drmTMB `check_temporal_boundary`) --------
#
# A temporal parameter can reach an interpretability boundary while the
# optimiser, Hessian and standard errors all look regular: the residual SD can
# collapse into the temporal process, an OU decay can run to 0 (a per-series
# constant) or to ∞ relative to the sampling gaps (white noise), and AR1
# persistence can run to ±1. drmTMB's thresholds (relative to the data):
# σ̂ / sd(y) < 1e-3; OU decay × longest within-series span < 1e-4 (correlation
# > 0.9999 at every observed lag) or decay × shortest positive gap > 30
# (correlation < 1e-13 at every observed lag); |φ̂| > 0.999. drmTMB then reports
# `convergence_status() == "boundary"` (still `is_converged`); DRModels.jl
# warns at fit time and reports the findings in `check_drm(fit).temporal_boundary`.
# drmTMB's `drm_inference_at_boundary` also counts its random-effect SD rule
# (`check_random_effect_sd_boundary`, sd_boundary = 1e-4: the smallest fitted
# SD — here the process SD, the phylogenetic stable SD or the `(1 | id)` SD —
# below 1e-4), so it is part of the same finding list.
const _TEMPORAL_BOUNDARY = (sigma_ratio = 1e-3, decay_span = 1e-4, decay_gap = 30.0,
                            ar1_phi = 0.999, sd = 1e-4)

_is_paired_phylo_temporal(fit) = _is_temporal_fit(fit) && fit.ranef.temporal.phylo !== nothing

"""
    _temporal_boundary(fit) -> Union{Nothing,Vector{String}}

`nothing` for a fit without a `temporal()` term; otherwise the list of
boundary findings (empty when every temporal parameter is interior), each
`"name=value"` as drmTMB's `temporal_boundary` row prints them.
"""
function _temporal_boundary(fit::DrmFit)
    _is_temporal_fit(fit) || return nothing
    info = fit.ranef.temporal
    y = fit.obs[:mu]
    ψ = info.structure === :ou ? exp(only(coef(fit, :temporal_decay))) :
        info.structure === :ar1 ? tanh(only(coef(fit, :temporal_phi))) : nothing
    sds = any(p -> first(p) === :resd, fit.blocks) ? re_sd(fit) : Dict{Symbol,Float64}()
    return _temporal_boundary_rules(info.structure, exp(only(coef(fit, :sigma))),
                                    length(y) > 1 ? std(y) : NaN, ψ, info.gaps; sds = sds)
end

# The rules themselves, on plain numbers: `ψ` is the OU decay λ or the AR1 φ;
# `gaps` the per-series gap vectors (first entry 0).
function _temporal_boundary_rules(structure, σ, sdy, ψ, gaps; sds = Dict{Symbol,Float64}())
    lim = _TEMPORAL_BOUNDARY
    found = String[]
    if !isempty(sds)
        nm, v = argmin(last, collect(sds))
        (isfinite(v) && v < lim.sd) &&
            push!(found, "sd_min=" * @sprintf("%.3g", v) * " (term=$(nm))")
    end
    (isfinite(σ) && isfinite(sdy) && sdy > 0 && σ / sdy < lim.sigma_ratio) &&
        push!(found, "sigma_ratio=" * @sprintf("%.3g", σ / sdy))
    if structure === :ou
        spans = [sum(g) for g in gaps]                      # last − first time per series
        pos = [x for g in gaps for x in g if x > 0]
        (isfinite(ψ) && !isempty(spans) && ψ * maximum(spans) < lim.decay_span) &&
            push!(found, "decay_x_max_span=" * @sprintf("%.3g", ψ * maximum(spans)))
        (isfinite(ψ) && !isempty(pos) && ψ * minimum(pos) > lim.decay_gap) &&
            push!(found, "decay_x_min_gap=" * @sprintf("%.3g", ψ * minimum(pos)))
    elseif structure === :ar1
        (isfinite(ψ) && abs(ψ) > lim.ar1_phi) && push!(found, "phi=" * @sprintf("%.6g", ψ))
    end
    return found
end

# Wording mirrors drmTMB's user-facing text for twin parity (drmTMB is copyright
# Shinichi Nakagawa; used with the owner's permission).
function _temporal_boundary_message(found)
    return "A temporal parameter is at an interpretability boundary (" * join(found, "; ") *
        "): sd_min means a random-effect SD (temporal, phylogenetic or `(1 | id)`) is near " *
        "zero (below 1e-4); sigma_ratio means the residual SD collapsed into the temporal process; " *
        "decay_x_max_span means OU correlation is ~1 at every observed lag (a per-series " *
        "constant); decay_x_min_gap means OU correlation is ~0 at every observed lag (white " *
        "noise); phi means AR1 persistence is at ±1. The optimiser converged, but do not read " *
        "the variance components separately; collect more series or occasions, or fit a " *
        "simpler temporal model. (drmTMB reports `convergence_status() == \"boundary\"` here.)"
end

# Fit-time advisory, silent inside bootstrap / profile refits.
function _warn_temporal_boundary(fit)
    get(task_local_storage(), :drm_quiet_boundary, false) === true && return fit
    found = try
        _temporal_boundary(fit)
    catch
        nothing                      # a diagnostic must never break a fit
    end
    (found === nothing || isempty(found)) || @warn "drm: " * _temporal_boundary_message(found)
    return fit
end

# One draw from a homogeneous Toeplitz fit: per series, the innovations form of
# σ²R (Durbin–Levinson), x_t = Σ_j φ_{t−1,j} x_{t−j} + σ √v_{t−1} z_t — the
# same distribution as drmTMB's `t(chol(σ²R)) %*% z`.
function _homtoep_simulate(fit::DrmFit, info, rng)
    Φ, logv = _homtoep_levinson(coef(fit, :temporal_pac))
    σ = exp(only(coef(fit, :sigma)))
    y = copy(fit.means[:mu])
    K = length(info.occasions)
    x = zeros(K)
    for rows in info.rows
        for t in 1:K
            m = 0.0
            for j in 1:(t-1)
                m += Φ[t-1, j] * x[t-j]
            end
            x[t] = m + σ * exp(logv[t] / 2) * randn(rng)
        end
        y[rows] .+= x
    end
    return y
end

# --- homtoep inference scope (drmTMB #1449: validate_temporal_wald_parm,
# vcov.drmTMB, validate_temporal_profile_parm) --------------------------------
# drmTMB qualifies mean-coefficient likelihood PROFILES for homtoep in its
# retained primary panel cells and withholds everything else: Wald covariance
# and intervals, and profiles of σ or the lag correlations. DRModels.jl keeps
# the Hessian internally (profile step sizes, `check_drm`) but refuses the same
# public surfaces and prints `NaN` standard errors in `coeftable` / `show`.
_wald_withheld(fit) = _is_temporal_fit(fit) && fit.ranef.temporal.structure === :homtoep

const _HOMTOEP_WALD_MSG = "Homogeneous Toeplitz Wald coefficient covariance is unavailable: " *
    "mean-coefficient likelihood profiles are qualified in drmTMB's retained primary panel " *
    "cells; Wald covariance and intervals remain deferred. Use " *
    "`confint(fit; method = :profile, parm = :mu => \"x\")`."

_homtoep_refuse_wald(what) = throw(ArgumentError("$what: " * _HOMTOEP_WALD_MSG))

# Profile targets on a homtoep fit: the mean coefficients only. `parm = nothing`
# selects them all (drmTMB's default); any other requested block is refused.
function _homtoep_profile_parm(fit, parm)
    parm === nothing && return :mu
    bad = unique([j.param for j in _profile_jobs(fit, parm) if j.param !== :mu])
    isempty(bad) ||
        throw(ArgumentError("confint/profile: Temporal HOMTOEP profile intervals currently support " *
            "mean regression coefficients only; unsupported target block(s) " *
            "$(join(string.(':', bad), ", ")). Use `parm = :mu` or `parm = :mu => \"x\"`. Scale " *
            "and lag-correlation intervals remain deferred (as in drmTMB)."))
    return parm
end

# Conditional residuals of an AR1 / OU fit (and of the paired phylo() + OU
# fit): (y − Xβ̂ − ŝ − b̂)/σ̂ with ŝ the temporal mode (data-row order) and b̂
# the per-series `(1 | id)` or phylogenetic stable mode. drmTMB's Pearson and
# quantile residuals for these routes use its `predict(dpar = "mu")`, which
# adds every fitted mode, divided by the residual SD.
function _temporal_conditional_residuals(fit::DrmFit)
    info = fit.ranef.temporal
    eff = fit.ranef.effects
    r = fit.obs[:mu] .- fit.means[:mu] .- eff[info.group]
    for k in (Symbol("$(info.group)_iid"), Symbol("$(info.group)_phylo"))
        haskey(eff, k) || continue
        for (s, rows) in enumerate(info.rows)
            r[rows] .-= eff[k][s]
        end
    end
    return r ./ exp(only(coef(fit, :sigma)))
end

# Whitened residuals of a homtoep fit: per series, e_t / (σ √v_{t−1}) from the
# Durbin–Levinson predictor — L⁻¹ r with L the Cholesky factor of σ²R, which
# drmTMB returns as its Pearson residuals (`temporal_homtoep_marginal_whiten`).
function _homtoep_whiten(fit::DrmFit)
    info = fit.ranef.temporal
    Φ, logv = _homtoep_levinson(coef(fit, :temporal_pac))
    σ = exp(only(coef(fit, :sigma)))
    r = fit.obs[:mu] .- fit.means[:mu]
    out = similar(r)
    for rows in info.rows
        for (t, i) in enumerate(rows)
            e = r[i]
            for j in 1:(t-1)
                e -= Φ[t-1, j] * r[rows[t-j]]
            end
            out[i] = e / (σ * exp(logv[t] / 2))
        end
    end
    return out
end
