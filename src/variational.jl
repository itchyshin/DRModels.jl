# variational.jl — opt-in Gaussian-variational (VA/ELBO) marginal as an
# alternative to the Laplace (LA) marginal for latent-integral models. See #136.
# Public Experimental path: Poisson / Binomial / NB2 / Gamma / Beta `(1 | g)`
# (`sigma ~ 1` where the family has a scale). `_fit_va` remains the unwired-family
# stub. Phylo / crossed / ZI / 136e stay open; #136 stays open.

"""
    MarginalMethod

How a model's random-effect integral is approximated. Subtypes: `Laplace`
(mode + curvature; the default, what drmTMB/TMB use), `Variational`
(maximize an ELBO over a Gaussian q; opt-in, steadier on dispersion/shape — #136),
and `AGHQ` (1-D Liu–Pierce adaptive Gauss–Hermite; opt-in Poisson `(1 | g)` only — #448).
"""
abstract type MarginalMethod end

"""    Laplace <: MarginalMethod

Laplace marginal: Gaussian approximation at the posterior mode. The default.
On an ordinary `(1 | g)` (Poisson, Binomial, NegBinomial2, Gamma, Beta) the
public `:LA` path is **non-adaptive GHQ-32**, not 1-point Laplace and not AGHQ.

Naming convention: `marginal = :LA` selects the *default integrator of the
route*, which is Laplace on most random-effect routes but non-adaptive GHQ-32
on the single random-intercept routes (an ordinary `(1 | g)`, Gaussian
`sigma ~ 1 + (1 | g)`) and exact wherever the Gaussian marginal is closed-form.
`marginal = :Laplace` *forces* the Laplace approximation that drmTMB (TMB)
computes; it is implemented for an ordinary `(1 | g)` on Poisson, Binomial,
NegBinomial2, Gamma and Beta (`ordinary_laplace.jl`) and for a Gaussian random
intercept on `sigma`. The fit is tagged `marginal = :Laplace`."""
struct Laplace <: MarginalMethod end

"""    Variational <: MarginalMethod

Gaussian-variational (VA/ELBO) marginal — opt-in alternative to `Laplace`
for bias-sensitive random-effect models (#136). Public `drm` path (Experimental):
Poisson / Binomial / NegBinomial2 / Gamma / Beta random intercept `(1 | g)` via
`marginal = :VA` (scale families need `sigma ~ 1`). Phylo, crossed, correlated
slopes, ZI/hu, and 136e stay unwired; #136 stays open."""
struct Variational <: MarginalMethod end

"""    AGHQ <: MarginalMethod

1-D Liu–Pierce adaptive Gauss–Hermite quadrature (#448). Opt-in via
`marginal = :AGHQ` on Poisson `(1 | g)` only (`nAGQ=5` default). This is
plumbing — k=1 ≡ 1-point Laplace; k≈5 nll agrees with GHQ-32. Not a
recovery headline, not a capability-chip flip, not tensor AGHQ on phylo
Laplace. `:REML` is not wired to `:AGHQ` this slice."""
struct AGHQ <: MarginalMethod end

# Resolve a user-facing `method` symbol (:LA/:VA/:AGHQ, case-insensitive) to a type.
_marginal_method(m::MarginalMethod) = m
function _marginal_method(s::Symbol)
    t = Symbol(uppercase(String(s)))
    t === :LA && return Laplace()
    t === :VA && return Variational()
    t === :AGHQ && return AGHQ()
    t === :LAPLACE && throw(ArgumentError(
        "marginal = :$s is not available on this route. `:Laplace` forces the Laplace " *
        "approximation that drmTMB uses and is implemented for an ordinary random intercept " *
        "`(1 | g)` on Poisson/Binomial/NegBinomial2/Gamma/Beta (routed by `drm` to the " *
        "ordinary-Laplace front end, `_drm_ordinary_laplace`) and for a Gaussian random " *
        "intercept on `sigma`, `sigma ~ 1 + (1 | g)`. Here use :LA (the default integrator " *
        "for the route; on a single random intercept `(1 | g)` that is 32-node Gauss–Hermite " *
        "quadrature, not Laplace), :VA (variational, #136), or :AGHQ (1-D Liu–Pierce, " *
        "Poisson (1|g) only, #448)."))
    throw(ArgumentError("unknown marginal method `:$s`; use :LA (the default integrator for the route: GHQ-32 on an ordinary `(1 | g)` and on Gaussian `sigma ~ 1 + (1 | g)`, Laplace on most other random-effect structures), :Laplace (TMB-convention Laplace, ordinary `(1 | g)` on Poisson/Binomial/NegBinomial2/Gamma/Beta, or a Gaussian random intercept on `sigma`), :VA (variational, #136), or :AGHQ (1-D Liu–Pierce, Poisson (1|g) only, #448)"))
end

# Route-or-reject for the public `marginal = :AGHQ` front end (#448, #761). The
# certified cells are Poisson `(1 | g)` (1-D Liu–Pierce, #448) and crossed random
# intercepts `(1 | g) + (1 | h)` on Binomial/Poisson/NegBinomial2/Gamma/Beta/
# BetaBinomial when one grouping has at most `_CROSSED_AGHQ_HMAX` levels (nested
# AGHQ, #761). Every other family or structure must error rather than silently
# falling back to GHQ-32 / Laplace (that would mislabel `loglik` as AGHQ).
function _aghq_reject(fam, what)
    throw(ArgumentError(
        "marginal = :AGHQ is not available for $(nameof(typeof(fam)))() with $what. " *
        "The public AGHQ path covers two cells: Poisson with a single random intercept " *
        "`(1 | g)` (1-D Liu–Pierce, #448), and crossed random intercepts `(1 | g) + (1 | h)` " *
        "on Binomial/Poisson/NegBinomial2/Gamma/Beta/BetaBinomial when one grouping has at " *
        "most $(_CROSSED_AGHQ_HMAX) levels (nested adaptive Gauss–Hermite, #761). Phylo, " *
        "relmat, `(1 + x | g)`, associate_pairs QuadGK, and other structures stay on " *
        "marginal = :LA (the default; on `(1 | g)` that is GHQ-32, not AGHQ)."))
end

# Generic stub entry point. Public VA kernels live on the per-family
# `_fit_*_ranef_va` entry points (Poisson / Binomial / NB2 / Gamma / Beta
# random intercept). Every other family still errors clearly rather than
# silently falling back, so opt-in callers know the state.
function _fit_va(args...; kwargs...)
    error("The variational (VA/ELBO) marginal is implemented only for the Poisson " *
          "(`_fit_poisson_ranef_va`), Binomial/Bernoulli (`_fit_binomial_ranef_va`), " *
          "NegBinomial2 (`_fit_nb2_ranef_va`), Gamma (`_fit_gamma_ranef_va`) and Beta " *
          "(`_fit_beta_ranef_va`) random-intercept cases so far; other families are not " *
          "yet wired — see https://github.com/itchyshin/DRModels.jl/issues/136. Use " *
          "marginal = :LA (the default integrator: GHQ-32 on an ordinary `(1 | g)`, Laplace on most other random-effect structures).")
end

# Route-or-reject for the public `marginal = :VA` front end (#136). Public VA
# kernels cover a single Gaussian random intercept `(1 | g)` on the mean for
# Poisson, Binomial, NegBinomial2, Gamma, and Beta (`sigma ~ 1` where the family
# has a scale). Any other model under `:VA` — fixed-effects-only, correlated
# `(1 + x | g)`, crossed `(1 | g) + (1 | h)`, phylo/spatial/relmat/animal,
# locscale, non-constant `sigma`, or `zi`/`hu` — has no public VA path, so we
# reject here with a uniform message rather than silently falling back to
# Laplace (that would mislabel `loglik` as an ELBO). `fam` names the family;
# `what` describes the offending structure.
function _va_reject(fam, what)
    throw(ArgumentError(
        "marginal = :VA (variational ELBO, #136) is not available for $(nameof(typeof(fam)))() with $what. " *
        "The public VA path is Experimental and covers Poisson, Binomial, NegBinomial2, Gamma, and Beta " *
        "with a single random intercept `(1 | g)` (`sigma ~ 1` where the family has a scale). " *
        "Use marginal = :LA (the default integrator: GHQ-32 on an ordinary `(1 | g)`, Laplace on most other random-effect structures) for this model."))
end

# `method` is the ML/REML selector. LA/VA is `marginal` (Q1 / #136).
# Non-Gaussian `drm()` accepts `method = :ML` as a no-op and rejects `:VA`/`:LA`
# with a pointer to `marginal`, matching the Poisson Arc 0 message.
#
# `allow_reml` opens the opt-in Cox–Reid restricted route (#443). A family only sets it
# when at least one of its routes can honour `:REML`; the family's route dispatch is then
# responsible for calling `_reject_reml_route` on the routes that cannot. Returns the
# normalised selector (`nothing` / `:ML` / `:REML`) so the caller can branch on it.
function _reject_method_as_marginal(fam, method; allow_reml::Bool = false)
    method === nothing && return nothing
    ms = Symbol(uppercase(String(method)))
    famname = nameof(typeof(fam))
    if ms === :VA || ms === :LA || ms === :LAPLACE || ms === :AGHQ
        canon = ms === :LAPLACE ? :Laplace : ms          # the documented spelling, not the uppercased key
        throw(ArgumentError(
            "drm ($famname): `method = :$method` is not the Laplace/VA/AGHQ selector. " *
            "Use `marginal = :$canon` (`:LA` default, GHQ-32 on an ordinary `(1 | g)`; " *
            "`:Laplace` TMB-convention Laplace on an ordinary `(1 | g)`; " *
            "`:VA` opt-in ELBO, #136; " *
            "`:AGHQ` 1-D Liu–Pierce on Poisson (1|g) only, #448). " *
            "`method` is reserved for `:ML`/`:REML`."))
    end
    ms === :ML && return :ML
    if ms === :REML
        allow_reml && return :REML
        throw(ArgumentError(
            "drm ($famname): unknown `method = :$method`. $famname is ML-only; " *
            "for the integrator use `marginal = :LA` (default; GHQ-32 on an ordinary `(1 | g)`), `:Laplace` " *
            "(TMB-convention Laplace, ordinary `(1 | g)`), `:VA` (#136), or `:AGHQ` (#448)."))
    end
    throw(ArgumentError(
        "drm ($famname): unknown `method = :$method`. $famname is ML-only; " *
        "for the integrator use `marginal = :LA` (default; GHQ-32 on an ordinary `(1 | g)`), `:Laplace` " *
        "(TMB-convention Laplace, ordinary `(1 | g)`), `:VA` (#136), or `:AGHQ` (#448)."))
end

# `method = :REML` reached a route the restricted (Cox–Reid) objective is not certified
# on. Erroring is the honest answer: silently falling back to ML would report
# `estimation_method(fit) === :ML` for a caller who asked for REML, and silently applying
# the penalty would ship an uncharacterised estimator. `what` names the offending
# structure. Scope and evidence: #443, probe #441.
function _reject_reml_route(fam, what)
    famname = nameof(typeof(fam))
    throw(ArgumentError(
        "drm ($famname): `method = :REML` is not available with $what. " *
        "Restricted (Cox–Reid) estimation on non-Gaussian families is currently wired " *
        "for Poisson scalar `(1 | g)` (#443) and Poisson phylo/relmat/animal Laplace " *
        "(#450). ML is the default and is available here: omit `method` or pass " *
        "`method = :ML`."))
end

# ──────────────────────────────────────────────────────────────────────────────
# Phase-2 proof kernel: closed-form mean-field Gaussian VA for the Poisson
# random-intercept model (#136 §4.1–4.4 of the design note). Self-contained and
# fully ForwardDiff-differentiable; opt-in, never the default. The verified
# Laplace/GHQ baseline (`_fit_poisson_ranef`) is untouched.
#
# Model: log λ_i = η0_i + b_{g(i)},  b_g ~ N(0, σ²),  σ² = exp(2·logσ).
# Mean-field q(b_g) = N(m_g, s_g). With the Poisson MGF E_q[e^{b}] = e^{m+s/2}:
#
#   E_q[log p(y_i|η_i)] = y_i·(η0_i + m_g) − exp(η0_i + m_g + s_g/2) − log y_i!
#   KL(N(m_g,s_g)‖N(0,σ²)) = ½[ s_g/σ² + m_g²/σ² − 1 − log(s_g/σ²) ]
#   ELBO = Σ_g [ Σ_{i∈g} E_q[log p] − KL_g ]      (a lower bound on log p(y|θ))
#
# We minimise nll = −ELBO. Per outer θ, the inner (m_g, s_g) are profiled to the
# ELBO maximiser. Because q is mean-field and groups are disjoint, the inner
# problem factorises per group into a 2-D strictly-concave solve (concave in
# (m, log s)), solved by a damped Newton unroll (`_va_inner_solve`) that keeps the
# objective a smooth function of θ for the outer AD. Its fixed 25 steps usually
# converge to machine precision, but nothing guaranteed it (#965): the unroll now
# continues while the inner point is not stationary, and the fit is marked not
# converged when a group still is not at θ̂ (`_va_inner_converged`).
# ──────────────────────────────────────────────────────────────────────────────

# The 2×2 matrix [a b; b d] with det = a·d − b² is usably negative definite: a < 0,
# d < 0 and det clear of its own rounding error, about eps·(|a·d| + b²). Before
# #965 the test was det > eps·(1 + |a| + |b| + |d|), which compares a product of
# entries with a sum of them. It let a positive-definite H through (only det was
# tested), and it damped a negative-definite H whose Hrr is merely tiny: with s
# far below its optimum, Hrr ≈ −(τ + E)·s/2 can be 1e-25, and the damped r step
# then crept up by ~1e-2 per iteration instead of taking the capped Newton step.
_va_negdef(a, d, b, det) = a < 0 && d < 0 && det > 4 * eps(typeof(det)) * (abs(a * d) + b * b)

# Levenberg-damped 2×2 Newton step for the inner ELBO problem. An ascent step
# needs H ≺ 0 (`_va_negdef`). Otherwise grow a damping λ, subtracted from the
# diagonal (H − λI is "more negative-definite"), until it is; this replaces the
# old sign-flipping guard (det = ±eps(T)), which could keep a spuriously negative
# det and turn the ascent step into a descent, and it caps the eps(T)-division
# blow-up. Returns the ascent step (Δm, Δr) that solves (H − λI)·Δ = −g, or (0, 0)
# when no damping makes H − λI negative definite (a non-finite entry), instead of
# dividing by a bad det; the caller's stationarity check then reports the group.
# Smooth in the entries for outer AD when no damping is needed (λ = 0, the
# generic case).
function _va_damped_step(Hmm::T, Hmr::T, Hrr::T, gm::T, gr::T) where {T}
    tol = eps(T) * (one(T) + abs(Hmm) + abs(Hrr) + abs(Hmr))
    a = Hmm; d = Hrr
    det = a * d - Hmr * Hmr
    λ = zero(T)
    if !_va_negdef(a, d, Hmr, det)
        λ = max(tol, abs(Hmm) + abs(Hrr) + abs(Hmr))
        for _ in 1:60
            a = Hmm - λ; d = Hrr - λ
            det = a * d - Hmr * Hmr
            _va_negdef(a, d, Hmr, det) && break
            λ *= T(2)
        end
        _va_negdef(a, d, Hmr, det) || return zero(T), zero(T)
    end
    Δm = -(d * gm - Hmr * gr) / det
    Δr = -(-Hmr * gm + a * gr) / det
    return Δm, Δr
end

# Cap the inner step at |Δm| ≤ 4, |Δr| ≤ 2 by scaling BOTH components by one
# factor, so the capped step keeps the direction of the damped Newton step.
# Clamping each coordinate separately (before #965) can turn that ascent
# direction into one along which the ELBO falls; `_va_backtrack` then rejects
# every trial, the iterate never moves, and no number of iterations helps.
function _va_cap_step(Δm::T, Δr::T) where {T}
    c = min(one(T), T(4) / abs(Δm), T(2) / abs(Δr))
    return c * Δm, c * Δr
end

# Inner stationarity (#965). The measure is the squared Newton decrement
# λ² = gᵀ(−H)⁻¹g of the group ELBO F in (m, r = log s): λ²/2 estimates how far F
# is below its maximum, in log-likelihood units, and λ bounds the error that a
# non-stationary inner point puts into the outer gradient (ForwardDiff
# differentiates through the inner iterations). A plain gradient test would have
# to tolerate the Float64 rounding of a large-count group (|∇F| ≈ eps·Σyᵢ, about
# 2e-6 at Σyᵢ = 1e10), whereas rounding leaves λ² near eps²·Σyᵢ (about 5e-22
# there). Returns Inf when H is not negative definite or an entry is not finite:
# such a point is not a maximum.
function _va_newton_decrement2(gm::T, gr::T, Hmm::T, Hmr::T, Hrr::T) where {T}
    det = Hmm * Hrr - Hmr * Hmr
    (isfinite(gm) && isfinite(gr) && isfinite(det) && Hmm < 0 && det > 0) || return T(Inf)
    # (−H)⁻¹ = [−Hrr  Hmr; Hmr  −Hmm] / det
    return (-Hrr * gm * gm + 2 * Hmr * gm * gr - Hmm * gr * gr) / det
end

# A group's inner profile counts as solved when λ²/2, the ELBO gain a Newton step
# still promises, is too small to matter or too small to see:
#     λ² ≤ max(_VA_INNER_TOL, 32·eps·(1 + |F|)).
# The first bound puts the group ELBO within about 5e-15 of its maximum (λ ≤ 1e-7).
# The second is twice the tie tolerance of `_va_backtrack`: a gain below it is lost
# in the rounding of F, so the line search cannot take it and the solve is not
# asked to. It is the larger one when F is formed from large terms, e.g. NB2 at a
# large size, where loggamma(y + r) − loggamma(r) cancels terms of size r·log r.
# The driver reports ρ = λ² / bound, so a group is solved iff ρ ≤ 1.
const _VA_INNER_TOL = 1e-14
_va_inner_bound(F0) = max(_VA_INNER_TOL, 32 * eps(Float64) * (1 + abs(F0)))
# The inner unroll always takes its fixed `iters` Newton steps (25 Poisson, 30 for
# the quadrature families), as before #965. A group that is still not stationary
# continues, up to `_VA_INNER_MAXITERS` steps in all; whatever is left unsolved is
# reported. A group that ends the unroll with ρ ≤ `_VA_INNER_DONE` stops there; one
# that is solved only narrowly, or only after the unroll, takes two more Newton
# steps, so the returned point (and the derivative the outer ForwardDiff carries
# through it) is not left at the edge of the tolerance.
const _VA_INNER_MAXITERS = 200
const _VA_INNER_DONE = 1e-6

# Shared inner driver: damped Newton ascent on the group ELBO `F(m, r)` from the
# warm start (m, r). `gH(m, r)` returns (gm, gr, Hmm, Hmr, Hrr), the gradient and
# Hessian of F in (m, r). Takes exactly `iters` steps, then keeps stepping while
# the group is not solved (ρ > 1, see `_va_inner_bound`; plus the polishing steps
# described at `_VA_INNER_DONE`), stopping at `maxiters` steps or when the line
# search can no longer move the iterate. Returns (m, r, ρ) with ρ measured at the
# returned point. Iterates and ρ carry the working eltype; the extra steps and the
# stop test read only primal values, so the outer ForwardDiff still differentiates
# the unrolled map.
function _va_inner_solve(gH, F, m::T, r::T, iters::Int, maxiters::Int) where {T}
    maxiters = max(maxiters, iters)
    ρ = T(Inf)
    npolish = 0
    for it in 1:(maxiters + 1)
        gm, gr, Hmm, Hmr, Hrr = gH(m, r)
        F0 = F(m, r)
        ρ = _va_newton_decrement2(gm, gr, Hmm, Hmr, Hrr) / _va_inner_bound(F0)
        if it > iters
            it > maxiters && break
            if ρ <= 1
                (npolish == 2 || (it == iters + 1 && ρ <= _VA_INNER_DONE)) && break
                npolish += 1
            end
        end
        # Levenberg-damped ascent step, capped along its own direction.
        Δm, Δr = _va_cap_step(_va_damped_step(Hmm, Hmr, Hrr, gm, gr)...)
        # Backtracking with a sufficient-decrease tolerance. Near the optimum the
        # full Newton step changes the ELBO by ~roundoff, so a strict `≥ F0` test
        # would spuriously backtrack to α≈0 and stall; the tolerance accepts such
        # steps (α=1) while still rejecting the genuine overshoot the guard targets.
        m1, r1 = _va_backtrack(F, m, r, Δm, Δr, F0)
        # Past the fixed unroll, an iterate the line search cannot move stays where
        # it is on every later step: stop, keeping the ρ measured at that point.
        it >= iters && m1 == m && r1 == r && break
        m, r = m1, r1
    end
    return m, r, ρ
end

# Inner per-group solve. Given the group totals A = Σ y_i and S = Σ e^{η0_i}, and
# the prior precision τ = 1/σ², return (m, s, ρ): the ELBO-stationary (m, s) and
# its squared Newton decrement relative to the solved bound (`_va_inner_bound`;
# ρ ≤ 1 when solved). Differentiable in (A, S, τ): the iterates are carried in the
# working eltype, so ForwardDiff propagates through the unrolled Newton steps.
# Stationarity equations (§4.4):
#   m: A − e^{m+s/2}·S − τ·m = 0
#   s: 1/s − τ − e^{m+s/2}·S = 0          (parameterised in r = log s)
function _poisson_va_inner(A::T, S::T, τ::T; iters::Int = 25,
                           maxiters::Int = _VA_INNER_MAXITERS) where {T}
    # Warm start: prior-shrunk MAP-ish mean, modest variance.
    m = log((A + one(T)) / (S + one(T)))            # ≈ log of the per-member rate
    r = log(one(T) / (τ + S + one(T)))              # r = log s; s ≈ 1/(τ + curvature)
    # (m,r)-dependent part of the group ELBO, for the ascent (Armijo) check. The
    # KL log is floored (issue #324.7) so a collapsing s never returns -Inf here.
    F = (mm, rr) -> begin
        s = exp(rr)
        A * mm - exp(mm + s / 2) * S -
            T(0.5) * (s * τ + mm^2 * τ - one(T) - log(max(s * τ, floatmin(T))))
    end
    gH = (mm, rr) -> begin
        s = exp(rr)
        E = exp(mm + s / 2) * S                     # = e^{m+s/2}·S ≥ 0
        # Gradient of the per-group ELBO in (m, r=log s):
        gm = A - E - τ * mm
        gr = (one(T) / s - τ - E) * (s / 2)         # ∂ELBO/∂r = ∂ELBO/∂s · s
        # Hessian in (m, r). With ∂E/∂m = E, ∂E/∂s = E/2 ⇒ ∂E/∂r = (E/2)·s:
        Hmm = -E - τ
        Hmr = -(E / 2) * s
        # ∂gr/∂r with gr = ½ − (τ + E)·s/2 and ∂E/∂r = (E/2)·s. Written as a sum of
        # non-positive terms: the old form (−1/s − (E/2)·s)·(s/2) + gr cancels its
        # two ±½ terms, which rounds Hrr to 0 when (τ + E)·s is tiny (an all-zero
        # group with a huge Σe^{η0}), forcing damping and creeping r steps (#965).
        Hrr = -(τ + E) * (s / 2) - E * s * s / 4
        (gm, gr, Hmm, Hmr, Hrr)
    end
    m, r, ρ = _va_inner_solve(gH, F, m, r, iters, maxiters)
    return m, exp(r), ρ
end

# Shared backtracking line search for the inner ELBO ascent. Accepts the full
# step when it does not decrease the group ELBO beyond a tolerance scaled to |F0|
# (so roundoff near the optimum does not trigger a stall), halving α otherwise.
# If even the smallest trial still decreases F, take α=0 (do not move) rather than
# accept a tiny wrong step — keeps the unrolled map monotone for the outer AD.
function _va_backtrack(F, m::T, r::T, Δm::T, Δr::T, F0 = F(m, r)) where {T}
    tol = eps(T) * (one(T) + abs(F0)) * T(16)
    α = one(T)
    for _ in 1:20
        (F(m + α * Δm, r + α * Δr) >= F0 - tol) && return (m + α * Δm, r + α * Δr)
        α *= T(0.5)
    end
    return (m, r)                                   # no acceptable step: stay put
end

# Gradient and Hessian of a group ELBO `obj(u)`, u = [m, r], by ForwardDiff, in the
# (gm, gr, Hmm, Hmr, Hrr) layout `_va_inner_solve` expects.
function _va_ad_gH(obj, m, r)
    u = [m, r]
    g = ForwardDiff.gradient(obj, u)
    H = ForwardDiff.hessian(obj, u)                   # ∇²F
    return g[1], g[2], H[1, 1], H[1, 2], H[2, 2]
end

# Fit-level inner check (#965). `ρ[g]` is group g's inner ρ at θ̂ (squared Newton
# decrement over its bound, `_va_inner_bound`), recorded by one more Float64
# evaluation of the objective. Returns whether every group is solved (ρ ≤ 1); the
# fitters AND it into `converged`, since the outer optimiser's convergence test
# cannot see the inner solves. When a group is not, `loglik` is below the profiled
# ELBO and the gradient the optimiser used was not the profile's gradient, so warn
# too (quiet inside bootstrap and profile refits, like the clamp warnings).
function _va_inner_converged(ρ::AbstractVector{<:Real}, route::AbstractString)
    bad = count(d -> !(d <= 1), ρ)
    bad == 0 && return true
    get(task_local_storage(), :drm_quiet_boundary, false) === true && return false
    worst = maximum(d -> isnan(d) ? Inf : Float64(d), ρ)
    @warn "$route: the variational inner solve did not reach stationarity for $bad of " *
          "$(length(ρ)) groups at the fitted parameters (the largest squared Newton " *
          "decrement is $(round(worst; sigdigits = 3)) times its tolerance). The reported " *
          "ELBO is then below the profiled ELBO, and the optimiser did not follow the " *
          "profiled ELBO's gradient, so the fit is marked not converged (DRModels.jl #965). " *
          "It can also mean the group ELBO carries too much rounding error to resolve its " *
          "maximum (e.g. NegBinomial2 near its Poisson limit). `marginal = :LA` (the " *
          "default) has no inner solve of this kind."
    return false
end

# Closed-form Poisson random-intercept VA fit. Same call shape as
# `_fit_poisson_ranef`; returns a `DrmFit` whose `loglik` field carries the
# **ELBO** (a lower bound), not the exact marginal log-likelihood.
function _fit_poisson_ranef_va(fam::Poisson, y, Xμ, gidx, G, nmμ, grp, g_tol)
    n = length(y); pμ = size(Xμ, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    lf = [_logfactorial(round(Int, yi)) for yi in y]
    A = [sum(y[idx]; init = 0.0) for idx in members]    # Σ y_i per group (θ-free)
    lfsum = sum(lf)                                       # Σ log y_i!  (θ-free constant)

    # nll(θ) = −ELBO(θ, m*(θ), s*(θ)), with the inner (m,s) profiled per group.
    # Passing `dec` (a length-G vector) also records each group's inner ρ (#965).
    function nll(θ, dec = nothing)
        βμ = θ[1:pμ]; logσ = θ[pμ+1]
        σ2 = exp(2 * logσ); τ = one(eltype(θ)) / σ2
        η0 = Xμ * βμ
        eη0 = exp.(η0)
        elbo = zero(eltype(θ))
        for (g, idx) in enumerate(members)
            isempty(idx) && continue
            S = sum(@view eη0[idx])                       # Σ e^{η0_i} in group g
            Ag = convert(eltype(θ), A[g])
            m, s, ρ = _poisson_va_inner(Ag, S, τ)
            dec === nothing || (dec[g] = ρ)
            # E_q[Σ log p] = Σ y_i·η0_i + A_g·m − e^{m+s/2}·S − (Σ log y_i! handled globally)
            yη0 = sum(y[i] * η0[i] for i in idx)
            Eqlp = yη0 + Ag * m - exp(m + s / 2) * S
            # KL(N(m,s)‖N(0,σ²)) = ½[ s·τ + m²·τ − 1 − log(s·τ) ]. Floor s·τ so a
            # collapsing inner variance (s → 0) never sends log(s·τ) → −Inf and
            # KL → +Inf, which would poison the ELBO and the outer AD (#324.7).
            kl = 0.5 * (s * τ + m^2 * τ - one(eltype(θ)) - log(max(s * τ, floatmin(eltype(θ)))))
            elbo += Eqlp - kl
        end
        return -(elbo - lfsum)
    end

    θ0 = zeros(pμ + 1)
    θ0[1] = log(sum(y) / n + eps())
    θ0[pμ+1] = log(0.5)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :resd => (pμ+1):(pμ+1)]
    names = [:mu => nmμ, :resd => [String(grp)]]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict{Symbol,Vector{Float64}}()
    # One more Float64 evaluation at θ̂ gives the ELBO and each group's inner ρ; a
    # group whose inner solve is not stationary makes the fit not converged (#965).
    dec = zeros(G)
    ll = -nll(θ̂, dec)
    conv = _va_inner_converged(dec, "_fit_poisson_ranef_va") && drm_optim_converged(res)
    # loglik field carries the ELBO (a lower bound); label downstream as :VA.
    return _withnll(DrmFit(fam, blocks, names, θ̂, V, ll, n, conv, means, obs, scales), nll)
end

# ──────────────────────────────────────────────────────────────────────────────
# Phase-2 proof kernel: mean-field Gaussian VA for the Binomial/Bernoulli
# random-intercept (logistic GLMM) model (#136). Unlike Poisson, the logit link
# gives a logistic-normal latent integral with **no** closed-form E_q[log p]; we
# evaluate the per-group E_q[log-lik] by a small Gauss–Hermite quadrature over the
# Gaussian q. The objective is otherwise identical in shape to the Poisson VA:
# a mean-field Gaussian q per group, profiled inner (m_g, s_g), and the Gaussian–
# Gaussian KL. Fully ForwardDiff-differentiable; opt-in, never the default. The
# verified Laplace/GHQ baseline (`_fit_binomial_ranef`) is untouched.
#
# Model: logit μ_i = η0_i + b_{g(i)},  b_g ~ N(0, σ²),  σ² = exp(2·logσ),
#        s_i ~ Binomial(n_i, μ_i).
# Mean-field q(b_g) = N(m_g, s_g). Writing the b-dependent log-likelihood kernel
#   ℓ_i(b) = s_i·(η0_i + b) − n_i·log(1 + e^{η0_i + b})           (logit Binomial)
# (the log C(n_i,s_i) coefficient is b- and θ-free → a global constant), then with
# the substitution b = m + √(2s)·z_k (z_k, w_k the K-node Gauss–Hermite rule):
#   E_q[Σ_{i∈g} ℓ_i] ≈ Σ_k (w_k/√π) · Σ_{i∈g} ℓ_i(m + √(2s)·z_k)
#   KL(N(m,s)‖N(0,σ²)) = ½[ s/σ² + m²/σ² − 1 − log(s/σ²) ]
#   group ELBO  F_g(m,r) = E_q[Σℓ] − KL,     r = log s
#   ELBO = Σ_g F_g + Σ_i log C(n_i, s_i)     (a lower bound on log p(y|θ))
#
# We minimise nll = −ELBO. Per outer θ, the inner (m_g, r_g) are profiled to the
# ELBO maximiser by a short Newton unroll; because q is mean-field and groups are
# disjoint the inner problem factorises per group into a smooth 2-D concave solve.
# The inner gradient/Hessian of F_g in (m, r) are taken by ForwardDiff (nodes are
# constants, so the GHQ sum is smooth); carrying the iterates in the working
# eltype lets the outer ForwardDiff propagate through the unrolled steps.
# ──────────────────────────────────────────────────────────────────────────────

# b-dependent logit-Binomial log-likelihood kernel summed over a group, as a
# function of the latent value b (the binomial coefficient is dropped — it is a
# b-free, θ-free constant handled globally). η0idx/sidx/nidx are the group's
# linear predictors, successes and trials. Differentiable in b and η0.
function _binom_group_kernel(b, η0idx, sidx, nidx)
    v = zero(promote_type(typeof(b), eltype(η0idx)))
    @inbounds for j in eachindex(η0idx)
        η = η0idx[j] + b
        # s·η − n·log(1 + e^η) = −s·log(1 + e^{−η}) − (n − s)·log(1 + e^η), evaluated
        # with one exp and one log1p as −n·log(1 + e^{−|η|}) minus the s·|η| or
        # (n − s)·η term. Every piece is ≤ 0, so nothing cancels: the old
        # s·η − n·softplus(η) lost ~1e-11 when η ≫ 0 and s ≈ n (many trials, nearly
        # all successes), which sat above the inner line search's tolerance and froze
        # the solve short of stationarity (#965). One branch for value and
        # derivative keeps ForwardDiff consistent at η = 0.
        if η > zero(η)
            v -= nidx[j] * log1p(exp(-η)) + (nidx[j] - sidx[j]) * η
        else
            v -= nidx[j] * log1p(exp(η)) - sidx[j] * η
        end
    end
    return v
end

# Per-group GHQ ELBO F(m, r) = E_q[Σ ℓ] − KL, with q = N(m, s), s = e^r and the
# prior precision τ = 1/σ². `z` are the Gauss–Hermite nodes, `lwπ = log(w) −
# ½·log π` the log weights. Smooth in (m, r); used both for the inner solve and
# (at the optimum) the reported value.
function _binom_va_group_elbo(m, r, η0idx, sidx, nidx, τ, z, lwπ)
    s = exp(r)
    sd = sqrt(2 * s)                                   # √(2s): b = m + √(2s)·z_k
    Eqℓ = zero(promote_type(typeof(m), typeof(r)))
    @inbounds for k in eachindex(z)
        Eqℓ += exp(lwπ[k]) * _binom_group_kernel(m + sd * z[k], η0idx, sidx, nidx)
    end
    # Floor s·τ so a collapsing inner variance (s → 0) cannot send the KL log to
    # −Inf and the group ELBO to −Inf, corrupting the outer AD (#324.7).
    kl = 0.5 * (s * τ + m * m * τ - one(m) - log(max(s * τ, floatmin(typeof(s * τ)))))
    return Eqℓ - kl
end

# Inner per-group solve: maximise F(m, r) over (m, r) by the Newton unroll of
# `_va_inner_solve`. Gradient and Hessian in the 2-vector (m, r) come from
# ForwardDiff of the group ELBO closure (nodes are constants ⇒ smooth). Iterates
# carry the working eltype, so the outer ForwardDiff differentiates through the
# converged (m*, s*). Returns (m, s, ρ), ρ the inner stationarity measure at the
# returned point (`_va_inner_bound`, #965). `z`, `lwπ` are the Gauss–Hermite nodes and log w − ½ log π
# arrays.
function _binom_va_inner(η0idx, sidx, nidx, τ::T, z, lwπ; iters::Int = 30,
                         maxiters::Int = _VA_INNER_MAXITERS) where {T}
    p̄ = clamp(sum(sidx) / max(sum(nidx), one(T)), T(1e-3), one(T) - T(1e-3))
    # Warm start: a prior-shrunk logit of the group success rate, modest variance.
    m = log(p̄ / (one(T) - p̄))
    r = log(one(T) / (τ + one(T)))                    # s ≈ 1/(τ + 1): small spread
    obj = u -> _binom_va_group_elbo(u[1], u[2], η0idx, sidx, nidx, τ, z, lwπ)
    m, r, dec = _va_inner_solve((mm, rr) -> _va_ad_gH(obj, mm, rr),
                                (mm, rr) -> obj([mm, rr]), m, r, iters, maxiters)
    return m, exp(r), dec
end

# Mean-field Gaussian VA fit for the Binomial/Bernoulli random-intercept model.
# Same call shape as `_fit_binomial_ranef`; `loglik` carries the **ELBO** (a lower
# bound), not the exact marginal log-likelihood. θ = [β_μ; log σ_b].
function _fit_binomial_ranef_va(fam::Binomial, s, ntr, Xμ, gidx, G, nmμ, grp, g_tol)
    n = length(s); pμ = size(Xμ, 2)
    sint = round.(Int, s); nint = round.(Int, ntr)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    # GHQ node arrays (12 nodes: q is Gaussian so the quadrature is exact for the
    # latent integral up to high order; cheap and ample for a 1-D effect).
    z, w = _gauss_hermite(12)
    lwπ = log.(w) .- 0.5 * log(π)                      # log w − ½ log π
    # log C(n_i, s_i): b- and θ-free constant, summed once.
    lcoef = sum(_logfactorial(nint[i]) - _logfactorial(sint[i]) - _logfactorial(nint[i] - sint[i]) for i in 1:n)

    function nll(θ, dec = nothing)               # `dec`: record each group's inner ρ (#965)
        βμ = θ[1:pμ]; logσ = θ[pμ+1]
        σ2 = exp(2 * logσ); τ = one(eltype(θ)) / σ2
        η0 = Xμ * βμ
        elbo = zero(eltype(θ))
        for (g, idx) in enumerate(members)
            isempty(idx) && continue
            η0idx = @view η0[idx]
            sidx = @view sint[idx]
            nidx = @view nint[idx]
            m, sg, ρ = _binom_va_inner(η0idx, sidx, nidx, τ, z, lwπ)
            dec === nothing || (dec[g] = ρ)
            elbo += _binom_va_group_elbo(m, log(sg), η0idx, sidx, nidx, τ, z, lwπ)
        end
        return -(elbo + lcoef)
    end

    p̄ = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)
    θ0 = zeros(pμ + 1)
    θ0[1] = log(p̄ / (1 - p̄)); θ0[pμ+1] = log(0.5)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :resd => (pμ+1):(pμ+1)]
    names = [:mu => nmμ, :resd => [String(grp)]]
    means = Dict(:mu => _logistic.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => s ./ ntr)   # population μ (b=0)
    scales = Dict(:trials => Float64.(nint))
    # One more Float64 evaluation at θ̂ gives the ELBO and each group's inner ρ; a
    # group whose inner solve is not stationary makes the fit not converged (#965).
    dec = zeros(G)
    ll = -nll(θ̂, dec)
    conv = _va_inner_converged(dec, "_fit_binomial_ranef_va") && drm_optim_converged(res)
    # loglik field carries the ELBO (a lower bound); label downstream as :VA.
    return _withnll(DrmFit(fam, blocks, names, θ̂, V, ll, n, conv, means, obs, scales), nll)
end

# ──────────────────────────────────────────────────────────────────────────────
# Phase-2 proof kernel: mean-field Gaussian VA for the NegBinomial2 (NB2)
# random-intercept model (#136). Like the Binomial case, the log link gives a
# log-normal-of-the-mean latent integral with **no** closed-form E_q[log p]; we
# evaluate the per-group E_q[NB2 log-pmf] by a small Gauss–Hermite quadrature over
# the Gaussian q. The only structural differences from the Binomial VA are the
# log-pmf kernel and one extra free outer parameter — the NB2 size/dispersion
# θ = exp(logθ) (the `sigma` slot), held fixed during the per-group inner solve.
# Fully ForwardDiff-differentiable; opt-in, never the default. The verified
# GHQ/Laplace baseline (`_fit_negbin2_ranef`) is untouched.
#
# Model: log μ_i = η0_i + b_{g(i)},  b_g ~ N(0, σ²),  σ² = exp(2·logσ),
#        y_i ~ NB2(μ_i, θ),  Var = μ_i + μ_i²/θ.
# Mean-field q(b_g) = N(m_g, s_g). Writing the b-dependent log-pmf kernel
#   ℓ_i(b) = loggamma(y_i+θ) − loggamma(θ) + θ·log(θ/(θ+μ_i)) + y_i·log(μ_i/(θ+μ_i))
# with μ_i = exp(η0_i + b)  (the −log y_i! coefficient is b- and θ-free → a global
# constant; the b-free loggamma part is added outside the quadrature, see
# `_nb2_group_kernel`), then with b = m + √(2s)·z_k (z_k, w_k the K-node
# Gauss–Hermite rule):
#   E_q[Σ_{i∈g} ℓ_i] ≈ Σ_k (w_k/√π) · Σ_{i∈g} ℓ_i(m + √(2s)·z_k)
#   KL(N(m,s)‖N(0,σ²)) = ½[ s/σ² + m²/σ² − 1 − log(s/σ²) ]
#   group ELBO  F_g(m,r) = E_q[Σℓ] − KL,     r = log s
#   ELBO = Σ_g F_g − Σ_i log y_i!            (a lower bound on log p(y|θ))
#
# We minimise nll = −ELBO. Per outer (β, logσ, logθ) the inner (m_g, r_g) are
# profiled to the ELBO maximiser by a short Newton unroll; because q is mean-field
# and groups are disjoint the inner problem factorises per group into a smooth 2-D
# concave solve. The inner gradient/Hessian of F_g in (m, r) are taken by
# ForwardDiff (nodes and θ are constants in the inner solve, so the GHQ sum is
# smooth); carrying the iterates in the working eltype lets the outer ForwardDiff
# propagate through both the unrolled steps and the dependence on θ.
# ──────────────────────────────────────────────────────────────────────────────

# b-dependent part of the NB2 log-pmf, summed over a group, as a function of the
# latent value b (the −log y_i! term is dropped — it is b- and θ-free, handled
# globally). η0idx/yidx are the group's linear predictors and integer counts; `r` is
# the size θ. With μ = e^{η0+b} the log-pmf (as in sparse_laplace_glmm.jl) is
#   loggamma(y+r) − loggamma(r) + r·log(r/(r+μ)) + y·log(μ/(r+μ))
#     = [loggamma(y+r) − loggamma(r) − y·log r] + y·η − (y + r)·log(1 + μ/r).
# The bracket does not depend on b: `_nb2_group_const` adds it once per group,
# outside the quadrature, and the inner solve never sees it. What is left has no
# cancellation. The old form built every node's value from terms of size r·log r
# (about 2600 at r = 430) that cancel to a value of order 1, and that rounding noise
# in the group ELBO froze the inner solve short of stationarity (#965).
# log(1 + μ/r) is the stable softplus of η − log r, which does not overflow at a far
# quadrature node either. Differentiable in b, η0 and r.
function _nb2_group_kernel(b, η0idx, yidx, r)
    v = zero(promote_type(typeof(b), eltype(η0idx), typeof(r)))
    lr = log(r)
    @inbounds for j in eachindex(η0idx)
        η = η0idx[j] + b
        v += yidx[j] * η - (yidx[j] + r) * _log1pexp(η - lr)
    end
    return v
end

# The b-free part of a group's NB2 log-pmf (see `_nb2_group_kernel`):
# Σ_j loggamma(y_j + r) − loggamma(r) − y_j·log r. For a count up to
# `_NB2_CONST_SUMMAX` it is summed as Σ_{k<y} log(1 + k/r), which is exact and does
# not cancel: the loggamma difference loses about eps·r·log r, i.e. 1e-6 at
# r = 4e8 and hundreds of nats in the Poisson limit (r → e⁴⁰ at the clamp), where it
# made the reported ELBO positive. Larger counts keep the loggamma difference.
# Differentiable in r (loggamma is ForwardDiff-safe via SpecialFunctions).
const _NB2_CONST_SUMMAX = 1000
function _nb2_group_const(yidx, r)
    lr = log(r); lgr = loggamma(r)
    v = zero(typeof(r))
    @inbounds for y in yidx
        if y <= _NB2_CONST_SUMMAX
            for k in 1:(y - 1)
                v += log1p(k / r)
            end
        else
            v += loggamma(y + r) - lgr - y * lr
        end
    end
    return v
end

# Per-group GHQ ELBO F(m, r) = E_q[Σ ℓ] − KL, with q = N(m, s), s = e^r, the prior
# precision τ = 1/σ², and the NB2 size `rsize` = θ, WITHOUT the b-free
# `_nb2_group_const` (the fitter adds it to the reported value). `z` are the
# Gauss–Hermite nodes, `lwπ = log(w) − ½·log π` the log weights. Smooth in (m, r);
# used both for the inner solve and (at the optimum) the reported value. `rsize` is
# held fixed inside the inner solve, but enters E_q[Σℓ] so the outer AD sees
# ∂ELBO/∂logθ.
function _nb2_va_group_elbo(m, r, η0idx, yidx, rsize, τ, z, lwπ)
    s = exp(r)
    sd = sqrt(2 * s)                                   # √(2s): b = m + √(2s)·z_k
    Eqℓ = zero(promote_type(typeof(m), typeof(r), typeof(rsize)))
    @inbounds for k in eachindex(z)
        Eqℓ += exp(lwπ[k]) * _nb2_group_kernel(m + sd * z[k], η0idx, yidx, rsize)
    end
    # Floor s·τ so a collapsing inner variance (s → 0) cannot send the KL log to
    # −Inf and the group ELBO to −Inf, corrupting the outer AD (#324.7).
    kl = 0.5 * (s * τ + m * m * τ - one(m) - log(max(s * τ, floatmin(typeof(s * τ)))))
    return Eqℓ - kl
end

# Inner per-group solve: maximise F(m, r) over (m, r) by the Newton unroll of
# `_va_inner_solve`, at fixed size `rsize`. Gradient and Hessian in (m, r) come from
# ForwardDiff of the group-ELBO closure (nodes and θ are constants ⇒ smooth).
# Iterates carry the working eltype, so the outer ForwardDiff differentiates
# through the converged (m*, s*). Returns (m, s, ρ), ρ the inner stationarity
# measure at the returned point (`_va_inner_bound`, #965). `z`, `lwπ` are the GH nodes and
# log w − ½ log π arrays.
function _nb2_va_inner(η0idx, yidx, rsize::T, τ::T, z, lwπ; iters::Int = 30,
                       maxiters::Int = _VA_INNER_MAXITERS) where {T}
    ȳ = sum(yidx) / length(yidx)
    # Warm start: a prior-shrunk log of the group mean count, modest variance.
    m = log(ȳ + one(T)) - sum(η0idx) / length(η0idx)  # centre on the group's mean rate
    r = log(one(T) / (τ + one(T)))                    # s ≈ 1/(τ + 1): small spread
    obj = u -> _nb2_va_group_elbo(u[1], u[2], η0idx, yidx, rsize, τ, z, lwπ)
    m, r, dec = _va_inner_solve((mm, rr) -> _va_ad_gH(obj, mm, rr),
                                (mm, rr) -> obj([mm, rr]), m, r, iters, maxiters)
    return m, exp(r), dec
end

# #999: the NB2 / Gamma / Beta VA kernels score each group under its FIRST
# member's dispersion predictor, which is right only when the single `sigma`
# column is constant (an intercept). `pσ == 1` alone admits `sigma ~ 0 + x`.
_va_constant_column(X::AbstractMatrix) = size(X, 2) == 1 && all(==(X[1, 1]), view(X, :, 1))

# #999 / #959: those kernels clamp η_σ to ±`b` INSIDE the objective, so beyond the
# band the ELBO is flat in that coordinate and the optimiser can stop past it.
# Report the scale the objective actually scored (clamped), and say when the
# clamp was active, since the stored `coef(fit, :sigma)` is then not the value
# the likelihood used.
function _va_scored_sigma(Xσ, βσ, b::Float64, route::AbstractString)
    η = Xσ * βσ
    if any(e -> abs(e) > b, η)
        @warn "$route: the `sigma` linear predictor ended at $(round(maximum(abs, η); sigdigits = 4)) " *
              "in absolute value, past the ±$b clamp inside the variational objective. The " *
              "objective is flat there, so `coef(fit, :sigma)` is not identified by the data; " *
              "the reported scale is the clamped value that was scored (DRModels.jl #959)."
    end
    return exp.(clamp.(η, -b, b))
end

# Mean-field Gaussian VA fit for the NB2 random-intercept model. Same call shape as
# `_fit_negbin2_ranef`; `loglik` carries the **ELBO** (a lower bound), not the exact
# marginal log-likelihood. θ = [β_μ; β_σ (= log θ_size); log σ_b], matching the
# block layout of the GHQ baseline so accessors line up.
function _fit_nb2_ranef_va(fam::NegBinomial2, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    # Enforce the intercept-only-dispersion assumption the group scoring relies on:
    # the per-group dispersion is read from the FIRST member's predictor, which is
    # only correct when ησ is constant across a group (σ ~ 1). A non-intercept Xσ
    # would silently score every other member under the wrong dispersion (#325).
    (pσ == 1 && _va_constant_column(Xσ)) || error("_fit_nb2_ranef_va supports only sigma ~ 1 " *
                     "(intercept-only dispersion); got pσ=$(pσ)")
    yint = round.(Int, y)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    # GHQ node arrays (12 nodes: q is Gaussian so the quadrature is exact for the
    # latent integral up to high order; cheap and ample for a 1-D effect).
    z, w = _gauss_hermite(12)
    lwπ = log.(w) .- 0.5 * log(π)                      # log w − ½ log π
    lfsum = sum(_logfactorial(yint[i]) for i in 1:n)   # Σ log y_i!: b-/θ-free constant

    function nll(θ, dec = nothing)               # `dec`: record each group's inner ρ (#965)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; logσ = θ[pμ+pσ+1]
        σ2 = exp(2 * logσ); τ = one(eltype(θ)) / σ2
        η0 = Xμ * βμ
        ησ = clamp.(Xσ * βσ, -20.0, 20.0)              # log θ_size per obs (here intercept-only)
        elbo = zero(eltype(θ))
        for (g, idx) in enumerate(members)
            isempty(idx) && continue
            η0idx = @view η0[idx]
            yidx = @view yint[idx]
            # NB2 size for this group's observations. With `sigma ~ 1` (the only
            # case routed here) ησ is constant across i, so take the group's value.
            rsize = exp(-2 * ησ[idx[1]])      # ψ = log σ; NB2 size = 1/σ² = exp(−2ψ) (drmTMB)
            m, sg, ρ = _nb2_va_inner(η0idx, yidx, rsize, τ, z, lwπ)
            dec === nothing || (dec[g] = ρ)
            elbo += _nb2_va_group_elbo(m, log(sg), η0idx, yidx, rsize, τ, z, lwπ) +
                    _nb2_group_const(yidx, rsize)
        end
        return -(elbo - lfsum)
    end

    m̄ = sum(y) / n; v̄ = sum(abs2, y .- m̄) / max(n - 1, 1)
    θ0 = zeros(pμ + pσ + 1)
    θ0[1] = log(m̄ + eps())
    θ0[pμ+1] = -0.5 * log(max(m̄^2 / max(v̄ - m̄, 0.1 * m̄ + eps()), 0.5))   # MoM init (log σ)
    θ0[pμ+pσ+1] = log(0.5)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+1)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp)]]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))   # population μ (b=0)
    scales = Dict(:sigma => _va_scored_sigma(Xσ, θ̂[(pμ+1):(pμ+pσ)], 20.0, "_fit_nb2_ranef_va"))   # #999
    # One more Float64 evaluation at θ̂ gives the ELBO and each group's inner ρ; a
    # group whose inner solve is not stationary makes the fit not converged (#965).
    dec = zeros(G)
    ll = -nll(θ̂, dec)
    conv = _va_inner_converged(dec, "_fit_nb2_ranef_va") && drm_optim_converged(res)
    # loglik field carries the ELBO (a lower bound); label downstream as :VA.
    return _withnll(DrmFit(fam, blocks, names, θ̂, V, ll, n, conv, means, obs, scales), nll)
end

# ──────────────────────────────────────────────────────────────────────────────
# Phase-2 proof kernel: mean-field Gaussian VA for the Gamma random-intercept
# model (#136). Structurally identical to the NB2 VA: a log link on the mean gives
# a log-normal-of-the-mean latent integral with **no** closed-form E_q[log p], so
# the per-group E_q[Gamma log-density] is evaluated by Gauss–Hermite quadrature
# over the Gaussian q. The only changes from the NB2 VA are the per-observation
# log-density kernel and that the extra free outer parameter is the Gamma shape
# α = exp(−2·log σ) (the `sigma` slot carries σ = the coefficient of variation),
# held fixed during the per-group inner solve. Fully ForwardDiff-differentiable;
# opt-in, never the default. The verified GHQ baseline (`_fit_gamma_ranef`) is
# untouched.
#
# Model: log μ_i = η0_i + b_{g(i)},  b_g ~ N(0, σ_b²),  σ_b² = exp(2·logσ_b),
#        y_i ~ Gamma(shape α, scale μ_i/α),  α = exp(−2·log σ),  Var = μ_i²σ².
# Mean-field q(b_g) = N(m_g, s_g). Writing the b-dependent log-density kernel with
# μ_i = exp(η0_i + b) (every term that depends on α or b is kept; only `log y_i`
# is fixed data):
#   ℓ_i(b) = (α−1)·log y_i − α·y_i/μ_i − α·(η0_i + b) + α·log α − loggamma(α)
# then with b = m + √(2s)·z_k (z_k, w_k the K-node Gauss–Hermite rule):
#   E_q[Σ_{i∈g} ℓ_i] ≈ Σ_k (w_k/√π) · Σ_{i∈g} ℓ_i(m + √(2s)·z_k)
#   KL(N(m,s)‖N(0,σ_b²)) = ½[ s/σ_b² + m²/σ_b² − 1 − log(s/σ_b²) ]
#   group ELBO  F_g(m,r) = E_q[Σℓ] − KL,     r = log s
#   ELBO = Σ_g F_g                            (a lower bound on log p(y|θ))
#
# We minimise nll = −ELBO. Per outer (β, logσ, logσ_b) the inner (m_g, r_g) are
# profiled to the ELBO maximiser by a short Newton unroll; because q is mean-field
# and groups are disjoint the inner problem factorises per group into a smooth 2-D
# concave solve. The inner gradient/Hessian of F_g in (m, r) are taken by
# ForwardDiff (nodes and α are constants in the inner solve, so the GHQ sum is
# smooth); carrying the iterates in the working eltype lets the outer ForwardDiff
# propagate through both the unrolled steps and the dependence on α.
# ──────────────────────────────────────────────────────────────────────────────

# b-dependent Gamma log-density kernel summed over a group, as a function of the
# latent value b. η0idx/yidx/lyidx are the group's linear predictors, responses and
# precomputed log y_i; `α` is the shape exp(−2·log σ). Same algebraic form as the
# shape–scale Gamma log-density used in gamma.jl (`Gamma(α, μ/α)`), expanded:
#   (α−1)·log y − α·y/μ − α·(η0+b) + α·log α − loggamma(α),  μ = e^{η0+b}.
# Differentiable in b, η0 and α (loggamma is ForwardDiff-safe via SpecialFunctions).
function _gamma_group_kernel(b, η0idx, yidx, lyidx, α)
    v = zero(promote_type(typeof(b), eltype(η0idx), typeof(α)))
    lα = log(α)
    @inbounds for j in eachindex(η0idx)
        η = η0idx[j] + b
        μ = exp(η)
        # (α−1)·log y − α·y/μ − α·η + α·log α − loggamma(α)
        v += (α - one(α)) * lyidx[j] - α * yidx[j] / μ - α * η + α * lα - loggamma(α)
    end
    return v
end

# Per-group GHQ ELBO F(m, r) = E_q[Σ ℓ] − KL, with q = N(m, s), s = e^r, the prior
# precision τ = 1/σ_b², and the Gamma shape `α`. `z` are the Gauss–Hermite nodes,
# `lwπ = log(w) − ½·log π` the log weights. Smooth in (m, r); used both for the
# inner solve and (at the optimum) the reported value. `α` is held fixed inside the
# inner solve, but enters E_q[Σℓ] so the outer AD sees ∂ELBO/∂logσ.
function _gamma_va_group_elbo(m, r, η0idx, yidx, lyidx, α, τ, z, lwπ)
    s = exp(r)
    sd = sqrt(2 * s)                                   # √(2s): b = m + √(2s)·z_k
    Eqℓ = zero(promote_type(typeof(m), typeof(r), typeof(α)))
    @inbounds for k in eachindex(z)
        Eqℓ += exp(lwπ[k]) * _gamma_group_kernel(m + sd * z[k], η0idx, yidx, lyidx, α)
    end
    # Floor s·τ so a collapsing inner variance (s → 0) cannot send the KL log to
    # −Inf and the group ELBO to −Inf, corrupting the outer AD (#324.7).
    kl = 0.5 * (s * τ + m * m * τ - one(m) - log(max(s * τ, floatmin(typeof(s * τ)))))
    return Eqℓ - kl
end

# Inner per-group solve: maximise F(m, r) over (m, r) by the Newton unroll of
# `_va_inner_solve`, at fixed shape `α`. Gradient and Hessian in (m, r) come from
# ForwardDiff of the group-ELBO closure (nodes and α are constants ⇒ smooth).
# Iterates carry the working eltype, so the outer ForwardDiff differentiates
# through the converged (m*, s*). Returns (m, s, ρ), ρ the inner stationarity
# measure at the returned point (`_va_inner_bound`, #965). `z`, `lwπ` are the GH nodes and
# log w − ½ log π arrays.
function _gamma_va_inner(η0idx, yidx, lyidx, α::T, τ::T, z, lwπ; iters::Int = 30,
                         maxiters::Int = _VA_INNER_MAXITERS) where {T}
    ȳ = sum(yidx) / length(yidx)
    # Warm start: a prior-shrunk log of the group mean, modest variance.
    m = log(ȳ + eps(T)) - sum(η0idx) / length(η0idx)  # centre on the group's mean rate
    r = log(one(T) / (τ + one(T)))                    # s ≈ 1/(τ + 1): small spread
    obj = u -> _gamma_va_group_elbo(u[1], u[2], η0idx, yidx, lyidx, α, τ, z, lwπ)
    m, r, dec = _va_inner_solve((mm, rr) -> _va_ad_gH(obj, mm, rr),
                                (mm, rr) -> obj([mm, rr]), m, r, iters, maxiters)
    return m, exp(r), dec
end

# Mean-field Gaussian VA fit for the Gamma random-intercept model. Same call shape
# as `_fit_gamma_ranef`; `loglik` carries the **ELBO** (a lower bound), not the
# exact marginal log-likelihood. θ = [β_μ; β_σ (= log σ, so α = exp(−2β_σ)); log σ_b],
# matching the block layout of the GHQ baseline so accessors line up.
function _fit_gamma_ranef_va(fam::Gamma, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    # Intercept-only-dispersion guard: the per-group shape is read from the first
    # member's predictor, valid only when ησ is constant across a group (#325).
    (pσ == 1 && _va_constant_column(Xσ)) || error("_fit_gamma_ranef_va supports only sigma ~ 1 " *
                     "(intercept-only dispersion); got pσ=$(pσ)")
    ly = log.(y)                                       # b-/θ-free log responses
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    # GHQ node arrays (12 nodes: q is Gaussian so the quadrature is exact for the
    # latent integral up to high order; cheap and ample for a 1-D effect).
    z, w = _gauss_hermite(12)
    lwπ = log.(w) .- 0.5 * log(π)                      # log w − ½ log π

    function nll(θ, dec = nothing)               # `dec`: record each group's inner ρ (#965)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; logσb = θ[pμ+pσ+1]
        σ2 = exp(2 * logσb); τ = one(eltype(θ)) / σ2
        η0 = Xμ * βμ
        ησ = clamp.(Xσ * βσ, -15.0, 15.0)              # log σ per obs (here intercept-only)
        elbo = zero(eltype(θ))
        for (g, idx) in enumerate(members)
            isempty(idx) && continue
            η0idx = @view η0[idx]
            yidx = @view y[idx]
            lyidx = @view ly[idx]
            # Gamma shape for this group's observations. With `sigma ~ 1` (the only
            # case routed here) ησ is constant across i, so take the group's value.
            α = exp(-2 * ησ[idx[1]])
            m, sg, ρ = _gamma_va_inner(η0idx, yidx, lyidx, α, τ, z, lwπ)
            dec === nothing || (dec[g] = ρ)
            elbo += _gamma_va_group_elbo(m, log(sg), η0idx, yidx, lyidx, α, τ, z, lwπ)
        end
        return -elbo
    end

    ȳ = sum(y) / n; v = sum(abs2, y .- ȳ) / max(n - 1, 1)
    α0 = max(ȳ^2 / max(v, eps()), 0.5)                 # MoM shape
    θ0 = zeros(pμ + pσ + 1)
    θ0[1] = log(ȳ + eps()); θ0[pμ+1] = -0.5 * log(α0)  # σ = 1/√α ⇒ log σ = −½ log α
    θ0[pμ+pσ+1] = log(0.5)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+1)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp)]]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))   # population μ (b=0)
    scales = Dict(:sigma => _va_scored_sigma(Xσ, θ̂[(pμ+1):(pμ+pσ)], 15.0, "_fit_gamma_ranef_va"))   # #999
    # One more Float64 evaluation at θ̂ gives the ELBO and each group's inner ρ; a
    # group whose inner solve is not stationary makes the fit not converged (#965).
    dec = zeros(G)
    ll = -nll(θ̂, dec)
    conv = _va_inner_converged(dec, "_fit_gamma_ranef_va") && drm_optim_converged(res)
    # loglik field carries the ELBO (a lower bound); label downstream as :VA.
    return _withnll(DrmFit(fam, blocks, names, θ̂, V, ll, n, conv, means, obs, scales), nll)
end

# ──────────────────────────────────────────────────────────────────────────────
# Phase-2 proof kernel: mean-field Gaussian VA for the Beta random-intercept model
# (#136). Structurally identical to the Gamma/NB2 VA: a logit link on the mean
# gives a logistic-normal latent integral with **no** closed-form E_q[log p], so
# the per-group E_q[Beta log-density] is evaluated by Gauss–Hermite quadrature over
# the Gaussian q. The free outer parameter beyond β is the Beta precision
# φ = exp(−2·log σ) (the `sigma` slot carries σ; drmTMB's precision mapping), held
# fixed during the per-group inner solve. Fully ForwardDiff-differentiable; opt-in,
# never the default. The verified GHQ baseline (`_fit_beta_ranef`) is untouched.
#
# Model: logit μ_i = η0_i + b_{g(i)},  b_g ~ N(0, σ_b²),  σ_b² = exp(2·logσ_b),
#        y_i ~ Beta(μ_iφ, (1−μ_i)φ),  φ = exp(−2·log σ),  Var = μ_i(1−μ_i)/(1+φ).
# Mean-field q(b_g) = N(m_g, s_g). Writing the b-dependent log-density kernel with
# μ_i = logistic(η0_i + b)  (every term that depends on φ or b is kept; only
# `log y_i` and `log(1−y_i)` are fixed data):
#   ℓ_i(b) = loggamma(φ) − loggamma(μ_iφ) − loggamma((1−μ_i)φ)
#            + (μ_iφ − 1)·log y_i + ((1−μ_i)φ − 1)·log(1−y_i)
# then with b = m + √(2s)·z_k (z_k, w_k the K-node Gauss–Hermite rule):
#   E_q[Σ_{i∈g} ℓ_i] ≈ Σ_k (w_k/√π) · Σ_{i∈g} ℓ_i(m + √(2s)·z_k)
#   KL(N(m,s)‖N(0,σ_b²)) = ½[ s/σ_b² + m²/σ_b² − 1 − log(s/σ_b²) ]
#   group ELBO  F_g(m,r) = E_q[Σℓ] − KL,     r = log s
#   ELBO = Σ_g F_g                            (a lower bound on log p(y|θ))
#
# We minimise nll = −ELBO; the inner (m_g, r_g) profile and outer AD work exactly
# as in the Gamma case (nodes and φ constant in the inner solve ⇒ smooth GHQ sum;
# iterates carry the working eltype so the outer ForwardDiff sees ∂ELBO/∂logσ).
# ──────────────────────────────────────────────────────────────────────────────

# b-dependent Beta log-density kernel summed over a group, as a function of the
# latent value b. η0idx/lyidx/l1myidx are the group's linear predictors and the
# precomputed log y_i and log(1−y_i); `φ` is the precision exp(−2·log σ). Same
# algebraic form as the Beta log-density used in beta.jl (`Beta(μφ, (1−μ)φ)`):
#   loggamma(φ) − loggamma(μφ) − loggamma((1−μ)φ) + (μφ−1)·log y + ((1−μ)φ−1)·log(1−y),
#   μ = logistic(η0+b).
# Differentiable in b, η0 and φ (loggamma is ForwardDiff-safe via SpecialFunctions).
function _beta_group_kernel(b, η0idx, lyidx, l1myidx, φ)
    v = zero(promote_type(typeof(b), eltype(η0idx), typeof(φ)))
    lgφ = loggamma(φ)
    @inbounds for j in eachindex(η0idx)
        μ = _logistic(η0idx[j] + b)
        a1 = μ * φ; a2 = (one(μ) - μ) * φ
        v += lgφ - loggamma(a1) - loggamma(a2) +
             (a1 - one(a1)) * lyidx[j] + (a2 - one(a2)) * l1myidx[j]
    end
    return v
end

# Per-group GHQ ELBO F(m, r) = E_q[Σ ℓ] − KL, with q = N(m, s), s = e^r, the prior
# precision τ = 1/σ_b², and the Beta precision `φ`. `z` are the Gauss–Hermite nodes,
# `lwπ = log(w) − ½·log π` the log weights. Smooth in (m, r); used both for the
# inner solve and (at the optimum) the reported value. `φ` is held fixed inside the
# inner solve, but enters E_q[Σℓ] so the outer AD sees ∂ELBO/∂logσ.
function _beta_va_group_elbo(m, r, η0idx, lyidx, l1myidx, φ, τ, z, lwπ)
    s = exp(r)
    sd = sqrt(2 * s)                                   # √(2s): b = m + √(2s)·z_k
    Eqℓ = zero(promote_type(typeof(m), typeof(r), typeof(φ)))
    @inbounds for k in eachindex(z)
        Eqℓ += exp(lwπ[k]) * _beta_group_kernel(m + sd * z[k], η0idx, lyidx, l1myidx, φ)
    end
    # Floor s·τ so a collapsing inner variance (s → 0) cannot send the KL log to
    # −Inf and the group ELBO to −Inf, corrupting the outer AD (#324.7).
    kl = 0.5 * (s * τ + m * m * τ - one(m) - log(max(s * τ, floatmin(typeof(s * τ)))))
    return Eqℓ - kl
end

# Inner per-group solve: maximise F(m, r) over (m, r) by the Newton unroll of
# `_va_inner_solve`, at fixed precision `φ`. Gradient and Hessian in (m, r) come
# from ForwardDiff of the group-ELBO closure (nodes and φ are constants ⇒ smooth).
# Iterates carry the working eltype, so the outer ForwardDiff differentiates
# through the converged (m*, s*). Returns (m, s, ρ), ρ the inner stationarity
# measure at the returned point (`_va_inner_bound`, #965). `z`, `lwπ` are the GH nodes and
# log w − ½ log π arrays.
function _beta_va_inner(η0idx, lyidx, l1myidx, φ::T, τ::T, ȳ, z, lwπ; iters::Int = 30,
                        maxiters::Int = _VA_INNER_MAXITERS) where {T}
    p̄ = clamp(T(ȳ), T(1e-3), one(T) - T(1e-3))
    # Warm start: a prior-shrunk logit of the group mean, minus the fixed part.
    m = log(p̄ / (one(T) - p̄)) - sum(η0idx) / length(η0idx)
    r = log(one(T) / (τ + one(T)))                    # s ≈ 1/(τ + 1): small spread
    obj = u -> _beta_va_group_elbo(u[1], u[2], η0idx, lyidx, l1myidx, φ, τ, z, lwπ)
    m, r, dec = _va_inner_solve((mm, rr) -> _va_ad_gH(obj, mm, rr),
                                (mm, rr) -> obj([mm, rr]), m, r, iters, maxiters)
    return m, exp(r), dec
end

# Mean-field Gaussian VA fit for the Beta random-intercept model. Same call shape
# as `_fit_beta_ranef`; `loglik` carries the **ELBO** (a lower bound), not the
# exact marginal log-likelihood. θ = [β_μ; β_σ (= log σ, so φ = exp(−2β_σ)); log σ_b],
# matching the block layout of the GHQ baseline so accessors line up.
function _fit_beta_ranef_va(fam::Beta, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    # Intercept-only-precision guard: the per-group precision is read from the first
    # member's predictor, valid only when ησ is constant across a group (#325).
    (pσ == 1 && _va_constant_column(Xσ)) || error("_fit_beta_ranef_va supports only sigma ~ 1 " *
                     "(intercept-only dispersion); got pσ=$(pσ)")
    ly = log.(y); l1my = log.(1 .- y)                  # b-/θ-free log responses
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    z, w = _gauss_hermite(12)
    lwπ = log.(w) .- 0.5 * log(π)                      # log w − ½ log π

    function nll(θ, dec = nothing)               # `dec`: record each group's inner ρ (#965)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; logσb = θ[pμ+pσ+1]
        σ2 = exp(2 * logσb); τ = one(eltype(θ)) / σ2
        η0 = Xμ * βμ
        ησ = clamp.(Xσ * βσ, -15.0, 15.0)              # log σ per obs (here intercept-only)
        elbo = zero(eltype(θ))
        for (g, idx) in enumerate(members)
            isempty(idx) && continue
            η0idx = @view η0[idx]
            lyidx = @view ly[idx]
            l1myidx = @view l1my[idx]
            # Beta precision for this group's observations. With `sigma ~ 1` (the
            # only case routed here) ησ is constant across i, so take the group's value.
            φ = exp(-2 * ησ[idx[1]])
            ȳg = sum(@view y[idx]) / length(idx)
            m, sg, ρ = _beta_va_inner(η0idx, lyidx, l1myidx, φ, τ, ȳg, z, lwπ)
            dec === nothing || (dec[g] = ρ)
            elbo += _beta_va_group_elbo(m, log(sg), η0idx, lyidx, l1myidx, φ, τ, z, lwπ)
        end
        return -elbo
    end

    ȳ = sum(y) / n; v = sum(abs2, y .- ȳ) / max(n - 1, 1)
    φ0 = max(ȳ * (1 - ȳ) / max(v, eps()) - 1, 0.5)     # MoM precision
    θ0 = zeros(pμ + pσ + 1)
    θ0[1] = log(ȳ / (1 - ȳ)); θ0[pμ+1] = -0.5 * log(φ0)  # σ = 1/√φ ⇒ log σ = −½ log φ
    θ0[pμ+pσ+1] = log(0.5)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+1)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp)]]
    means = Dict(:mu => _logistic.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))   # population μ (b=0)
    scales = Dict(:sigma => _va_scored_sigma(Xσ, θ̂[(pμ+1):(pμ+pσ)], 15.0, "_fit_beta_ranef_va"))   # #999
    # One more Float64 evaluation at θ̂ gives the ELBO and each group's inner ρ; a
    # group whose inner solve is not stationary makes the fit not converged (#965).
    dec = zeros(G)
    ll = -nll(θ̂, dec)
    conv = _va_inner_converged(dec, "_fit_beta_ranef_va") && drm_optim_converged(res)
    # loglik field carries the ELBO (a lower bound); label downstream as :VA.
    return _withnll(DrmFit(fam, blocks, names, θ̂, V, ll, n, conv, means, obs, scales), nll)
end
