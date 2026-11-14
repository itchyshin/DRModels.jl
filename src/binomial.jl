# binomial.jl — plain Binomial / Bernoulli family. The default logit link on
# the mean success probability μ; probit and complementary-log-log are also
# available on fixed-effect and adaptive-quadrature routes. No
# dispersion parameter (mean-only, like Poisson — see BetaBinomial for the
# overdispersed version). Two response forms: `cbind(successes, failures) ~ x`
# (trials = successes + failures, exactly as drmTMB) or a plain 0/1 Bernoulli
# vector. Fixed effects and a random intercept `(1 | g)` on the mean (logit by
# default). `Distributions.Binomial` is used qualified — DRModels exports its own
# `Binomial` family.

import Distributions

# Numerically stable, UNCLAMPED Binomial(n, logistic(η)) log-likelihood. A clamp of
# η inside the objective makes it exactly flat (zero gradient under AD) wherever any
# |η_i| exceeds the clamp, so an L-BFGS step that overshoots there parks on the
# plateau and reports success with a garbage answer. `k·η − n·log1pexp(η)` is exact
# for any finite η (no `logistic` saturation to 0/1, no log(0)) and AD-safe.
@inline _log1pexp(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))
@inline _logchoose(n, k) = loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)
@inline _binomial_logit_ll(n, k, η, lc) = k * η - n * _log1pexp(η) + lc

# Without the clamp a wild line-search probe can reach a point where the AGHQ
# marginal (or its AD gradient) is NaN/Inf; Optim's line search asserts finiteness
# and throws. Map such probes to the repo's 1e18 failed-fit sentinel (zero
# gradient, hugely worse than any real fit) so the line search backtracks instead.
# Used only inside the optimiser call, never for the reported value / Hessian.
@inline _finite_or_sentinel(v::Real) = isfinite(v) ? v : oftype(v, 1e18)
@inline function _finite_or_sentinel(v::ForwardDiff.Dual)
    ok = isfinite(ForwardDiff.value(v)) && all(isfinite, ForwardDiff.partials(v))
    return ok ? v : oftype(v, 1e18)
end
_safe_objective(f) = θ -> _finite_or_sentinel(f(θ))

# L-BFGS with the default HagerZhang line search asserts (`B > A`, `isfinite(phi_c)`)
# on pathological brackets near a runaway optimum and throws, losing the fit. Retry
# from the start with BackTracking (no such assertions) and report whatever it
# reaches; `converged` stays honest (false if the gradient tolerance is not met).
function _optimize_with_fallback(nll, θ0, g_tol;
                                 opts = Optim.Options(g_tol = g_tol))
    f = _safe_objective(nll)
    try
        return Optim.optimize(f, θ0, Optim.LBFGS(), opts; autodiff = :forward)
    catch e
        e isa AssertionError || rethrow()
        return Optim.optimize(f, θ0, Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
                              opts; autodiff = :forward)
    end
end

# Covariance for the quadrature routes at a runaway optimum (e.g. separated data):
# the AD Hessian can be non-finite, and `_vcov_from_hessian` (deliberately) throws on
# that. Report the repo's NaN-covariance convention (standard errors Inf) with a
# warning so the fit itself is not lost.
function _vcov_or_nan(H::AbstractMatrix)
    if !all(isfinite, H)
        @warn "Hessian at the optimum is not finite (runaway or separated fit): covariance is NaN, standard errors Inf."
        return fill(NaN, size(H))
    end
    return _vcov_from_hessian(H)
end

"""
    Binomial(; link = :logit)

Binomial response family — successes out of known trials. The default `:logit`
link can be changed to `:probit` or `:cloglog`; no scale/dispersion parameter
(mean-only, like [`Poisson`](@ref)). Accepts either a two-column response via
[`cbind`](@ref) (`trials = successes + failures`) or a plain `0/1` Bernoulli
vector. With inverse link `h`, the likelihood is `Binomial(n, μ)` with
`μ = h(η)`. Non-logit links are available for fixed effects and ordinary
adaptive-quadrature random-effect and SD-regression routes. Structured,
crossed-Laplace, variational, and TMB-convention Laplace routes currently
require `:logit`, except for `:probit` with one phylogenetic SD-regression
target. A random intercept `(1 | g)` on the mean fits a GLMM by
per-group ADAPTIVE Gauss–Hermite quadrature (`nq` nodes on the mode-
centred scale, #712/#713); crossed intercepts such as `(1 | g) + (1 | h)` use
the sparse-Laplace engine. A phylogenetic random intercept on the mean,
`phylo(1 | species)`, also uses the sparse-Laplace engine. A correlated random
intercept + slope `(1 + x | g)` is fit by the same per-group adaptive
quadrature helper, `nq` nodes per axis (the same scheme as
[`BetaBinomial`](@ref)); it needs within-group variation in `x` — a slope
predictor that is constant within every level of `g` leaves the slope SD and
group-level correlation unidentified, and is refused with an informative
error (mirrors drmTMB's `drm_validate_q2_slope_variation`). An independent
random slope `(0 + x | g)` and `marginal = :VA` on `(1 + x | g)` remain out of
scope. Independent scalar ordinary effects can each have a group-level SD
regression, integrated by per-group adaptive Gauss–Hermite quadrature. For an
intercept, write `sd(g) ~ z`; with an independent slope on the same group, select
each coefficient in its own formula, for example
`sd(g, coef(intercept)) ~ z0` and `sd(g, coef(x)) ~ z1`. The predictors must be
constant within group. The ordinary independent-effect route requires
`marginal = :LA`; provider-specific structured SD support is documented in the
family guide. Missing responses are refused. Ordinary independent SD fits use
local quadrature state and can use threaded profiling; structured Laplace SD
fits use shared warm-start state and require `threads = false`.
`predict_parameters(fit, newdata)[:sd]` returns the fitted random-effect
SD on each row, repeating the value for rows in the same group; it accepts
predictor-only new data and checks it against the fitted SD design schema. With
`se = true` it returns the prediction and delta-method SE. In-sample this agrees
with `marginal_parameters(fit)[:sd]`.

For a sparse mesh field, a single spatial random intercept can use
`mesh = (; projection, precision, alignment_ids)`, where `projection` maps
observations to mesh vertices and `precision` is the sparse vertex precision
matrix. Optional `alignment_ids` must match `data.row_id` and are used to
reorder projection rows after data reordering. The mesh route fits
`spatial(1 | g)` with `sd(g, spatial) ~ z` by sparse Laplace approximation and
supports seeded marginal simulation. It currently requires complete responses,
one spatial intercept and one spatial SD formula; additional random effects and
other structured inputs are refused. This mesh implementation is Julia-only;
native TMB parity and interval-coverage calibration have not been established.

!!! note "Separation: detect and warn (fixed-effects-only fits)"
    A fixed-effects-only Bernoulli/Binomial fit is screened at fit time for
    (quasi-)complete separation: the maximum-likelihood estimate does not exist
    when some direction `d` makes the signed design `X̃ d ≥ 0` on every row and
    `> 0` on at least one (Albert & Anderson 1984; Konis 2007), decided here by a
    non-negative least-squares (Farkas) solve. The fit is still returned, a
    warning names the affected coefficients, their standard errors are `Inf`
    (their `vcov` rows/columns are NaN with an `Inf` diagonal) and their Wald
    `confint` / `coeftable` intervals are `(-Inf, Inf)`. A coefficient is flagged
    when ANY separating direction moves it (which can list more coefficients
    than R's `detectseparation`, which reports one direction); an aliased column
    that is a multiple of a flagged one is named too. The check is invariant to
    column units and robust to a single far outlier. When it proves there is no
    separation the fit is left alone, even if a Wald `|z|` is tiny. Only when it
    cannot decide (iteration budget exhausted) is a scale-free near-separation
    rule consulted: a fitted probability within 1e-8
    of 0/1, a coefficient whose own contribution `|βⱼ|·range(xⱼ)` spans that
    whole probability range (> 36.84 on the logit scale), and Wald `|z| < 0.05`;
    a failed (non-finite) SE is never read as near separation. If that rule
    does not fire, it warns that the check was inconclusive rather than
    reporting no separation. Bootstrap replicate refits do not repeat the
    warning. No refusal and no penalised estimator by default. drmTMB runs the
    same check with the same constants, so the two packages flag the same
    coefficients (the coefficient values themselves are arbitrary stopping
    points and are not expected to agree). Random-intercept routes are not
    screened.

!!! warning "Crossed intercepts: Laplace is biased low for Bernoulli data"
    The default crossed fit is the Laplace approximation drmTMB and lme4 use. For
    Bernoulli / small-`n` Binomial data with a large random-intercept SD and few
    observations per group it underestimates the log-likelihood (≈9.4 nat on a
    300 × 4 Bernoulli design with σ_g = 2.5) and shrinks σ. When one grouping has
    at most 8 levels, `marginal = :AGHQ` integrates the same model by nested
    adaptive Gauss–Hermite quadrature and corrects this (#761). Its `loglik` is
    not comparable with a Laplace fit's, so `lrtest` refuses mixed `marginal`s.

!!! note
    `DRModels.Binomial` shadows `Distributions.Binomial`; if you need the
    distribution too (e.g. to simulate), qualify it as `Distributions.Binomial`.

```julia
fit = drm(bf(@formula(cbind(successes, failures) ~ x)), Binomial(); data = dat)   # logistic regression
fit = drm(bf(@formula(y ~ x + (1 | g))), Binomial(); data = dat)                  # 0/1 logistic GLMM
fit = drm(bf(@formula(cbind(successes, failures) ~ x + (1 | g) + (1 | h))), Binomial(); data = dat)
fit_q = drm(bf(@formula(cbind(successes, failures) ~ x + (1 | g) + (1 | h))), Binomial();
            data = dat, marginal = :AGHQ)                   # accurate crossed integral, h ≤ 8 levels
fit_phy = drm(bf(@formula(cbind(successes, failures) ~ x + phylo(1 | species))),
              Binomial(); data = dat, tree = tr, se = false)
fit_phy_probit_sd = drm(
    bf(@formula(cbind(successes, failures) ~ x + phylo(1 | species)),
       @formula(sd(species, phylogenetic) ~ z)),
    Binomial(link = :probit); data = dat, tree = tr, se = false)
fit_sd = drm(bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z)),
             Binomial(); data = dat)
predict_parameters(fit_sd, dat)[:sd]  # one fitted random-effect SD per row
fit_relmat_sd = drm(bf(@formula(y ~ x + relmat(1 | id)),
                       @formula(sd(id, relmat) ~ z)),
                    Binomial(); data = dat, K = K)
fitted(fit)        # fitted success probabilities on the response scale

# Experimental: Binomial random-intercept variational (ELBO) marginal.
fit_va = drm(bf(@formula(y ~ x + (1 | g))), Binomial(); data = dat, marginal = :VA)

# TMB-convention Laplace on an ordinary `(1 | g)`, as native drmTMB fits it (ML only).
fit_lap = drm(bf(@formula(y ~ x + (1 | g))), Binomial(); data = dat, marginal = :Laplace)
```
"""
struct Binomial
    link::Symbol
    function Binomial(link::Symbol)
        link in (:logit, :probit, :cloglog) || throw(ArgumentError(
            "Binomial link must be one of :logit, :probit, or :cloglog; got :$(link)."))
        new(link)
    end
end

Binomial(; link::Symbol = :logit) = Binomial(link)

@inline _binomial_mean(fam::Binomial, eta) =
    fam.link === :logit ? _logistic(eta) :
    fam.link === :probit ? Distributions.cdf(Distributions.Normal(), eta) :
    -expm1(-exp(eta))

@inline function _binomial_start(fam::Binomial, p)
    p = clamp(p, 1e-6, 1 - 1e-6)
    fam.link === :logit && return log(p / (1 - p))
    fam.link === :probit && return Distributions.quantile(Distributions.Normal(), p)
    return log(-log1p(-p))
end

@inline function _binomial_link_ll(fam::Binomial, n, k, eta, lc)
    fam.link === :logit && return _binomial_logit_ll(n, k, eta, lc)
    if fam.link === :probit
        normal = Distributions.Normal()
        ll = lc
        k > 0 && (ll += k * Distributions.logcdf(normal, eta))
        failures = n - k
        failures > 0 && (ll += failures * Distributions.logccdf(normal, eta))
        return ll
    end
    ll = lc
    if k > 0
        # At the positive tail this probability is 1 to Float64 precision;
        # avoid differentiating through exp(eta) after it has overflowed.
        logp = eta < -20 ? eta : eta > 36 ? zero(eta) : log(-expm1(-exp(eta)))
        ll += k * logp
    end
    failures = n - k
    failures > 0 && (ll -= failures * exp(eta))
    return ll
end

# FX3b: `se = false` is recorded on the fit (`DrmFit.uncertainty`), so
# `convergence_status` does not read the skipped (NaN) covariance as a failure.
@drm_entry function drm(f::DrmFormula, fam::Binomial; se::Bool = true, kwargs...)
    fit = _withuncertainty(_drm_binomial_fit(f, fam; se = se, kwargs...), se)
    return _with_prior_table_input(fit, get(kwargs, :priors, nothing))
end

# `ids = …` labels the rows of `A` (animal term) or of `coords` (spatial term), as on
# the Gaussian, Poisson, NegBinomial2 and Gamma routes. The Binomial fitter used to have
# no `ids` keyword, so the advice in the unlabelled-`A` warning ended in a MethodError.
# `A` is reordered to the first-seen group order the routes read; `coords` becomes a
# label => coordinate Dict, which the spatial routes already resolve.
function _binomial_label_inputs(f::DrmFormula, data, A, Ainv, coords, ids)
    ids === nothing && return A, coords
    Ainv === nothing || throw(ArgumentError("drm: `ids = …` labels the rows of `A`; the rows of " *
        "`Ainv` follow the order of first appearance of the group in the data, so pass " *
        "`A` with `ids`, or `Ainv` without `ids`."))
    if _animal_group(f) !== nothing
        return _resolve_relatedness_labels(f, data, A, ids; coords = coords), coords
    end
    A === nothing && coords === nothing && throw(ArgumentError(
        "drm: `ids = …` labels the rows of the relatedness matrix `A` or of the site " *
        "coordinates `coords`, but neither was supplied."))
    if coords !== nothing && !(coords isa AbstractDict)
        n = coords isa AbstractVector ? length(coords) : size(coords, 1)
        n == length(ids) || throw(ArgumentError("drm: `coords` has $n row(s) but `ids` has " *
            "$(length(ids)) label(s); give one label per row."))
        coords = Dict(string(l) => (coords isa AbstractVector ? [Float64(coords[i])] :
                                    Float64.(coords[i, :])) for (i, l) in enumerate(ids))
    end
    return A, coords
end

# A bare correlated block of three or four coefficients, `(1 + x + z [+ w] | g)`, with no
# `sd()` formula. drmTMB fits it as the correlated model with one constant SD per
# coefficient, which is the SD-regression route with every coefficient SD left at its
# default `~ 1`; the q = 2 block `(1 + x | g)` keeps its own two-dimensional route.
function _binomial_bare_corr_block(f::DrmFormula, fam::Binomial)
    fam.link === :logit || return false
    all(p -> first(p) in (:mu, :sigma), f.forms) || return false
    mu = get(Dict(f.forms), :mu, nothing)
    mu === nothing && return false
    try
        _, re, mv, st = _split_ranef(mu)
        mv === nothing && st === nothing && length(re) == 1 || return false
        vars = _re_multi_corr_vars(re[1][1])
        return vars !== nothing && length(vars) >= 2
    catch
        return false
    end
end

# Outside SD-regression routes, Binomial admits `phylo` alone among the
# structured markers. `relmat` is currently supported for the structured SD
# regression route below, not for a general Binomial random-effect fit. Without
# these arguments in the signature a user
# writing `relmat(1 | g)` with `K = K` got a bare
#     MethodError: no method matching drm(::DrmFormula, ::Binomial; K=…)
# — a dispatch failure instead of the explanation this method already carries a
# few lines below. Taking the arguments lets that refusal actually be reached.
function _drm_binomial_fit(f::DrmFormula, fam::Binomial; data, tree = nothing,
             tree1 = nothing, tree2 = nothing, K = nothing,
             A = nothing, Ainv = nothing, coords = nothing, mesh = nothing,
             g_tol::Real = 1e-8,
             se::Bool = true, marginal::Symbol = :LA, method = nothing,
             nq::Int = _BINOMIAL_RANEF_AGHQ_K, priors = nothing, ids = nothing)
    f = _default_structured_q2_sd(f, fam)   # bare structured (1 + x | g): default `sd(…, coef(·)) ~ 1`
    f = _default_structured_q1_sd(f, fam; method, K, coords)   # bare structured (1 | g): default `sd(g, kind) ~ 1`
    Tables.istable(priors) && (priors = _table_priors_for_model(f, fam, data, priors))
    _drm_entry_guard(f, fam, data; tree = tree, K = K, A = A, Ainv = Ainv, coords = coords)   # #1027 #1022 #1010
    A, coords = _binomial_label_inputs(f, data, A, Ainv, coords, ids)   # red team: `ids = …` as on the other families
    if priors !== nothing
        fam.link === :logit && marginal === :LA && method === nothing || throw(ArgumentError("fixed-effect priors require ordinary binomial logit ML"))
        isempty(_sd_parts(f)) && isempty(_sdphylo_parts(f)) || throw(ArgumentError("fixed-effect priors do not support SD formulas"))
        for response in _formula_responses(f)
            response === nothing && continue
            all(_observed_response_mask(_table_column(data, response))) || throw(ArgumentError("fixed-effect priors require complete responses"))
        end
        _, re0, mv0, st0 = _split_ranef(Dict(f.forms)[:mu])
        isempty(re0) && mv0 === nothing && st0 === nothing || throw(ArgumentError("fixed-effect priors do not support random or structured effects"))
    end
    _reject_method_as_marginal(fam, method)
    nq >= 1 || throw(ArgumentError("Binomial `nq` must be ≥ 1; got $nq"))
    fam.link !== :logit && marginal === :VA && throw(ArgumentError(
        "Binomial links other than :logit are not available with `marginal = :VA`."))
    if !isempty(_sd_parts(f)) || !isempty(_sdphylo_parts(f)) ||
       (priors === nothing && marginal === :LA && _binomial_bare_corr_block(f, fam))
        return _drm_binomial_sd_regression(f, fam; data, tree, tree1, tree2,
            K, A, Ainv, coords, mesh, g_tol, se, marginal, method, nq)
    end
    mesh === nothing || throw(ArgumentError(
        "Binomial `mesh` is currently used only for spatial random-effect SD regression."))
    tree1 === nothing && tree2 === nothing || throw(ArgumentError(
        "Binomial `tree1` and `tree2` are used only by `phylo_interaction()` SD models."))
    Ainv === nothing || throw(ArgumentError(
        "Binomial `Ainv` is currently supported for animal random-effect SD regression only."))
    fam.link !== :logit && _scalar_laplace_requested(marginal) &&
        throw(ArgumentError("Binomial links other than :logit are not available with `marginal = :Laplace`."))
    _scalar_laplace_requested(marginal) &&     # Arc 2: TMB-convention Laplace, ordinary (1 | g)
        return _drm_ordinary_laplace(f, fam; data = data, tree = tree, K = K, A = A,
                                     coords = coords, g_tol = g_tol, se = se, method = method)
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, tree = tree, K = K, A = A, coords = coords,
            g_tol = g_tol, se = se, marginal = marginal, method = method, nq = nq)
    end
    missing_fit !== nothing && return missing_fit

    marg = _marginal_method(marginal)                     # :LA (default) or :VA (#136)
    isaghq = marg isa AGHQ                                # :AGHQ: crossed intercepts only (#761)
    isva = marg isa Variational
    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    fixed_mu, offset_term = _extract_offset(fixed_mu)   # #938: offset(z), coefficient fixed at 1
    offset_term === nothing || (isempty(re) && mv === nothing && st === nothing) ||
        _refuse_offset_route(fam, "a random, structured or `meta_V` term on the mean")
    isaghq && !(length(re) > 1 && st === nothing) &&
        _aghq_reject(fam, "this model (Binomial `marginal = :AGHQ` covers crossed random intercepts `(1 | g) + (1 | h)` only, #761)")
    mv === nothing ||
        error("Binomial() does not support meta_V markers")
    for (pname, r) in f.forms          # Binomial is mean-only — reject any other parameter formula
        pname === :mu && continue
        pname === :sigma || error("Binomial() is mean-only; no sigma/dispersion parameter")
        r === ConstantTerm(1) ||
            error("Binomial() is mean-only; no sigma/dispersion parameter")
    end
    s, ntr = _binomial_response(f, data)
    prior_cache = priors === nothing ? nothing : Ref(Dict{Symbol,Any}())
    y, Xμ, nmμ = _design(f.response, fixed_mu, data; schema_cache=prior_cache, schema_key=:mu)      # successes column is a dummy LHS
    if priors !== nothing
        all(==(1), _ctl_weights(length(s))) || throw(ArgumentError("fixed-effect priors currently require unit weights"))
        rank(Xμ) == size(Xμ, 2) || throw(ArgumentError("fixed-effect priors require full-rank design matrices"))
        rows = _resolve_fixed_priors(priors, [:mu => nmμ], Dict(:mu => _fixed_intercept_index(f, :mu, fixed_mu, nmμ; cache=prior_cache)))
        return _withformula(_fit_binomial(fam, s, ntr, Xμ, nmμ, g_tol; offset=_offset_for_fit(fam, offset_term, data, size(Xμ, 1)), prior_rows=rows), f)
    end
    if st !== nothing
        fam.link === :logit || throw(ArgumentError(
            "Binomial links other than :logit are not available for structured random effects."))
        isva && _va_reject(fam, "a phylogenetic/structured random effect")
        isempty(re) ||
            error("Binomial() phylo structured effects cannot be combined with ordinary random effects yet")
        kind, grp = st
        kind === :phylo ||
        error("Binomial() currently supports only phylo(1 | group) among structured markers")
        tree === nothing && error("phylo(1 | $grp) needs `tree = ...`")
        labels = getproperty(data, grp)
        return _withformula(_fit_binomial_phylo_laplace(fam, s, ntr, Xμ, labels, tree, nmμ, grp, g_tol; se = se), f)
    end
    if !isempty(re)                                       # random intercept (1|g) → GHQ/Laplace or VA (#136)
        if length(re) > 1
            isva && _va_reject(fam, "crossed/multiple random intercepts")
            fam.link === :logit || throw(ArgumentError(
                "Binomial links other than :logit are not yet supported for crossed random intercepts."))
            if _has_indep_slopes(re)
                isaghq && _aghq_reject(fam, "independent random slopes `(0 + x | g)` (Laplace only)")
                return _withformula(_fit_indep_re_laplace(fam, Val(:binomial),
                                                          collect(zip(s, ntr)), Xμ, nothing, nothing,
                                                          _indep_slope_comps(re, data, length(s)),
                                                          nmμ, String[], String[], g_tol), f)
            end
            all(_re_kind(r[1])[1] === :intercept for r in re) ||
                error("Binomial() supports multiple random effects only as crossed/nested intercepts " *
                      "`(1 | g) + (1 | h)` or independent intercepts and slopes `(1 | g) + (0 + x | g)`; " *
                      "a correlated `(1 + x | g)` must be the only random-effect term")
            comps = map(re) do r
                grp = r[2]; gidx, G = _group_index(getproperty(data, grp); col = grp)
                (ones(length(s)), gidx, G, String(grp))
            end
            isaghq && return _withformula(_fit_binomial_crossed_aghq(fam, s, ntr, Xμ, comps, nmμ, g_tol; se = se), f)   # #761
            return _withformula(_fit_binomial_crossed_laplace(fam, s, ntr, Xμ, comps, nmμ, g_tol), f)
        end
        (rk, var) = _re_kind(re[1][1]); grp = re[1][2]; gidx, G = _group_index(getproperty(data, grp); col = grp)
        if rk === :intercept                              # (1 | g) → 1-D GHQ (Laplace) or VA (#136)
            isva && return _withformula(_withmarginal(
                _fit_binomial_ranef_va(fam, s, ntr, Xμ, gidx, G, nmμ, grp, g_tol), :VA), f)
            return _withformula(_fit_binomial_ranef(
                fam, s, ntr, Xμ, gidx, G, nmμ, grp, g_tol; nq = nq), f)
        elseif rk === :corr                                # (1 + x | g) → 2-D GHQ tensor (#753)
            isva && _va_reject(fam, "a correlated random slope `(1 + x | g)`")
            _require_finite_data_column(data, var)
            xs = Float64.(getproperty(data, var))
            _binomial_check_slope_identifiable(xs, gidx, G, grp)
            return _withformula(_fit_binomial_corr_ranef(
                fam, s, ntr, Xμ, xs, gidx, G, nmμ, grp, g_tol; nq = nq), f)
        elseif rk === :slope                              # (0 + x | g) → TMB-convention Laplace (independent route)
            isva && _va_reject(fam, "an independent random slope `(0 + x | g)`")
            isaghq && _aghq_reject(fam, "an independent random slope `(0 + x | g)` (Laplace only)")
            fam.link === :logit || throw(ArgumentError(
                "Binomial links other than :logit are not yet supported for independent random slopes."))
            _require_finite_data_column(data, var)
            return _withformula(_fit_indep_re_laplace(fam, Val(:binomial),
                                                      collect(zip(s, ntr)), Xμ, nothing, nothing,
                                                      _indep_slope_comps(re, data, length(s)),
                                                      nmμ, String[], String[], g_tol), f)
        else
            error("Binomial() supports `(1 | g)`, `(0 + x | g)` or `(1 + x | g)` on the mean")
        end
    end
    isva && _va_reject(fam, "no random intercept (fixed-effects-only)")
    return _withformula(_fit_binomial(fam, s, ntr, Xμ, nmμ, g_tol; offset = _offset_for_fit(fam, offset_term, data, size(Xμ, 1))), f)
end

# Successes and trials from a 0/1 Bernoulli response or `cbind(successes, failures)`.
# Shared by the default route and the `marginal = :Laplace` route (ordinary_laplace.jl).
function _binomial_response(f::DrmFormula, data)
    if f.response2 === nothing                            # plain 0/1 Bernoulli vector
        s = Float64.(getproperty(data, f.response))
        all(yi -> yi == 0 || yi == 1, s) ||
            error("Binomial() with a single-column response requires a 0/1 (Bernoulli) vector; use cbind(successes, failures) for trial counts")
        ntr = ones(length(s))
    else                                                  # cbind(successes, failures): n = s + f
        _require_finite_inputs(; response=(_table_column(data, f.response), string(f.response)),
                               response_allow_missing=false)
        _require_finite_inputs(; response=(_table_column(data, f.response2), string(f.response2)),
                               response_allow_missing=false)
        s = Float64.(getproperty(data, f.response))       # successes
        fl = Float64.(getproperty(data, f.response2))     # failures
        (all(si -> si ≥ 0 && isinteger(si), s) && all(fi -> fi ≥ 0 && isinteger(fi), fl)) ||
            error("Binomial() requires non-negative integer successes and failures")
        ntr = s .+ fl                                     # trials
    end
    return s, ntr
end

function _fit_binomial(fam::Binomial, s, ntr, Xμ, nmμ, g_tol; offset = nothing, prior_rows = nothing)
    n = length(s); pμ = size(Xμ, 2)
    active_prior = prior_rows !== nothing && !all(r -> r.kind === :ridge && r.precision == 0, prior_rows)
    if active_prior
        penalized = Set(r.index for r in prior_rows if r.precision > 0)
        length(penalized) < pμ && @warn "Binomial prior leaves some coefficients unpenalized; a finite optimum requires every separating direction to be regularized. Check convergence and sensitivity."
    end
    sint = round.(Int, s); nint = round.(Int, ntr)
    lc = [_logchoose(nint[i], sint[i]) for i in 1:n]
    w = _ctl_weights(n)                  # prior weights: row log-lik multipliers (#931)
    function data_nll(θ)
        ημ = _eta_with_offset(Xμ, θ, offset)          # unclamped: see `_binomial_logit_ll`
        v = zero(eltype(θ))
        @inbounds for i in 1:n
            v -= w[i] * (_binomial_link_ll(fam, nint[i], sint[i], ημ[i], lc[i]))
        end
        return v
    end
    nll = prior_rows === nothing ? data_nll : θ -> data_nll(θ) + _fixed_prior_value(prior_rows, θ)
    p̄ = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)   # overall success rate
    θ0 = zeros(pμ); θ0[1] = _binomial_start(fam, p̄)
    θ0[1:pμ] .= _offset_start(θ0[1:pμ], Xμ, offset)   # #938
    res = _ctl_optimize((x0, opts, method) -> Optim.optimize(nll, x0, method, opts; autodiff = :forward),
                       θ0; g_tol = g_tol)   # control: maxit / start / multi_start (#948 #994)
    θ̂ = Optim.minimizer(res); V = _ctl_se(() -> _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂)), θ̂)
    blocks = [:mu => 1:pμ]; names = [:mu => nmμ]
    means = Dict(:mu => _binomial_mean.(Ref(fam), _eta_with_offset(Xμ, θ̂, offset)))                # fitted success probability
    separated = false
    if !active_prior
        V, separated = _separation_check(Xμ, s, ntr, θ̂, V, means[:mu], nmμ;
                                         near_logit = fam.link === :logit)
    end
    obs = Dict(:mu => s ./ ntr)                            # observed proportion (for residuals)
    scales = Dict(:trials => Float64.(nint))
    fit = _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -data_nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
    fit = separated ? _withgeometry(fit, :binomial_separation) : fit
    return prior_rows === nothing ? fit : _with_fixed_priors(fit, prior_rows, data_nll)
end

# Default nodes for the Binomial 1-D `(1 | g)` route (#712/#713 grouped-trial
# gap; raised 3 -> 5 by owner decision 17, #908). Was a fixed 32-node grid on the
# PRIOR scale; with grouped trials (n trials/row > 1) and an informative group SD
# the posterior is a narrow spike and the grid missed it by up to 9.73 nat. Swept
# vs an independent AGHQ-40 on the grouped-trial DGP (G=100, 30 obs/group, 20
# trials/row, RE SD 0.5-0.8): K=1 (Laplace) is the pre-existing multi-nat error,
# K=3 lands inside 0.01 nat. K=3 was too coarse on sparse binary data
# (`MASS::bacteria`: sd_mu 5% low, 0.22 nat off a 61-node reference, #908); K=5
# matches the other families' default. See test/test_binomial_aghq.jl and
# test/test_binomial_default_nodes.jl.
const _BINOMIAL_RANEF_AGHQ_K = 5

# Binomial logistic GLMM with a random intercept (1|g) on the logit mean.
# b_g ~ N(0,σ_b²) is integrated out per group by adaptive Gauss–Hermite
# quadrature (`_aghq_marginal_loglik`, #834/#712/#713: nodes b̂_g + √2 c z at
# each group's mode), q = 1, `nq` nodes — replaces the old fixed 32-node
# prior-scale grid, which under-covered grouped trials (n trials/row > 1) with
# an informative group SD. θ = [β_μ; log σ_b].
function _fit_binomial_ranef(fam::Binomial, s, ntr, Xμ, gidx, G, nmμ, grp, g_tol; nq::Int = _BINOMIAL_RANEF_AGHQ_K)
    n = length(s); pμ = size(Xμ, 2)
    sint = round.(Int, s); nint = round.(Int, ntr)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    lc = [_logchoose(nint[i], sint[i]) for i in 1:n]
    rule = _AGHQRule(1, nq); Zre = ones(n, 1); bcache = zeros(1, G)   # #834/#712/#713: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; σb = exp(θ[pμ+1])
        L = reshape([σb], 1, 1)
        η0 = Xμ * βμ
        ll = (i, η) -> _binomial_link_ll(fam, nint[i], sint[i], η, lc[i])
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    p̄ = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)
    θ0 = zeros(pμ + 1)
    θ0[1] = _binomial_start(fam, p̄); θ0[pμ+1] = log(0.5)
    res = _optimize_with_fallback(nll, θ0, g_tol)
    θ̂ = Optim.minimizer(res); V = _vcov_or_nan(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :resd => (pμ+1):(pμ+1)]
    names = [:mu => nmμ, :resd => [String(grp)]]
    means = Dict(:mu => _binomial_mean.(Ref(fam), Xμ * θ̂[1:pμ])); obs = Dict(:mu => s ./ ntr)
    scales = Dict(:trials => Float64.(nint))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Identifiability guard for the correlated random slope (1 + x | g) (#753): the
# slope RE variance and the group-level (intercept, slope) correlation are
# unidentified when `xs` is constant within EVERY level of `grp` — mirrors
# drmTMB's `drm_validate_q2_slope_variation` (R/mspl-estimator.R), reimplemented
# here (never vendored: drmTMB is GPL, DRModels.jl is MIT). One level with a single
# observation (or a constant slope predictor) among others with variation is fine: the
# other levels identify the slope SD and correlation. The check used to refuse the
# data when ANY level was constant, which drmTMB fits (red team, 2026-11-08).
function _binomial_check_slope_identifiable(xs, gidx, G, grp)
    seen = [Set{Float64}() for _ in 1:G]
    for i in eachindex(xs)
        push!(seen[gidx[i]], xs[i])
    end
    any(length(s) ≥ 2 for s in seen) ||
        error("Binomial() correlated random slope `(1 + x | g)` needs within-group variation in the " *
              "slope predictor — it is constant within every level of `$grp`, so the slope SD " *
              "and the group-level intercept–slope correlation are unidentified. " *
              "Use a predictor that varies within `$grp`.")
end

# Binomial logistic GLMM with a correlated random intercept + slope (1 + x | g)
# on the logit mean (#753). Per group (b0,b1) ~ N(0, Σ); logit μ_i =
# Xμ_iᵀβ + b0_g + b1_g·x_i. Because groups are disjoint the per-group 2-D
# integral factorises; it is done by per-group ADAPTIVE Gauss–Hermite
# quadrature (`_aghq_marginal_loglik`, #834: nodes b̂_g + √2 C z at each
# group's mode), `nq` nodes per axis — the same scheme as
# `_fit_betabinomial_corr_ranef` (betabinomial.jl) minus the precision φ
# (Binomial has no dispersion parameter). Σ is the log-Cholesky
# parameterisation L = [exp(a) 0; cc exp(b)] (the `vc` convention), so vc(fit)
# reconstructs Σ = L Lᵀ.
function _fit_binomial_corr_ranef(fam::Binomial, s, ntr, Xμ, xs, gidx, G, nmμ, grp, g_tol; nq::Int = _CORR_RANEF_AGHQ_K)
    n = length(s); pμ = size(Xμ, 2)
    sint = round.(Int, s); nint = round.(Int, ntr)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    lc = [_logchoose(nint[i], sint[i]) for i in 1:n]
    rule = _AGHQRule(2, nq); Zre = hcat(ones(n), Float64.(xs)); bcache = zeros(2, G)   # #834: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]
        L = _corr_ranef_L(θ[pμ+1], θ[pμ+2], θ[pμ+3])
        η0 = Xμ * βμ
        ll = (i, η) -> _binomial_link_ll(fam, nint[i], sint[i], η, lc[i])
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    p̄ = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)
    θ0 = zeros(pμ + 3)
    θ0[1] = _binomial_start(fam, p̄)
    θ0[pμ+1] = log(0.4); θ0[pμ+2] = log(0.4); θ0[pμ+3] = 0.0
    res = _optimize_with_fallback(nll, θ0, g_tol)
    θ̂ = Optim.minimizer(res); V = _vcov_or_nan(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :recov => (pμ+1):(pμ+3)]
    names = [:mu => nmμ, :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"]]
    means = Dict(:mu => _binomial_mean.(Ref(fam), Xμ * θ̂[1:pμ])); obs = Dict(:mu => s ./ ntr)
    scales = Dict(:trials => Float64.(nint))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end
