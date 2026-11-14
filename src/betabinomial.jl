# betabinomial.jl — Beta-binomial family: successes out of known trials, with
# extra-binomial overdispersion. Two-column response `cbind(successes, failures)`
# (trials = successes + failures), exactly as drmTMB. Logit link on the mean
# success probability μ; the `sigma` slot is σ with precision φ = 1/σ² (same
# mapping as Beta) — likelihood BetaBinomial(n, μφ, (1-μ)φ). `Distributions.
# BetaBinomial` is used qualified — DRModels exports its own `BetaBinomial` family.

import Distributions

"""
    cbind(successes, failures)

Formula marker for a two-column count response, e.g.
`bf(cbind(successes, failures) ~ x, sigma ~ z)` with [`BetaBinomial`](@ref).
Trials are `successes + failures`. Mirrors drmTMB's `cbind(...)` response.
Only inspected structurally on a formula left-hand side.
"""
cbind(a, b) = hcat(a, b)

# `_LOGIT_GUARD` is an overflow guard, not a model bound: at |η| = 700 a parameter
# is e^-700 ≈ 1e-304 and the likelihood of any observation on the wrong side is
# ≈ -700 nats, so no sane start is ever beaten by the (flat) region beyond it; it only
# keeps `loggamma(0)` / NaN gradients out of the line search.
const _LOGIT_GUARD = 700.0

# BetaBinomial(n, μφ, (1-μ)φ) log-pmf with μ = logistic(η), parameterised directly
# on the logit η and UNCLAMPED: α = φ·σ(η), β = φ·σ(−η) never saturate to 0 until
# |η| ≈ 745, so no clamp is needed to keep the pmf proper. A clamp of η inside the
# objective makes it flat beyond the clamp and lets L-BFGS park on the plateau
# (see `_binomial_logit_ll`). Equal to `Distributions.logpdf(BetaBinomial(...))`
# off the plateau up to rounding.
@inline function _betabinomial_logit_ll(n, k, η, φ)
    η = clamp(η, -_LOGIT_GUARD, _LOGIT_GUARD)
    α = φ * _logistic(η); β = φ * _logistic(-η)
    return _logchoose(n, k) + (loggamma(k + α) + loggamma(n - k + β) - loggamma(n + α + β)) -
           (loggamma(α) + loggamma(β) - loggamma(α + β))
end

"""
    BetaBinomial()

Beta-binomial response family — successes out of known trials with
extra-binomial overdispersion. Logit link on the mean success probability `μ`;
the `sigma` slot carries `σ` with precision `φ = 1/σ²` (so `coef(fit, :sigma)`
is `log σ`). Likelihood `BetaBinomial(n, μφ, (1-μ)φ)`. Requires a two-column
response via [`cbind`](@ref). Mirrors `drmTMB`'s `beta_binomial`.
Crossed random intercepts on the mean, such as `(1 | g) + (1 | h)`, use the
sparse-Laplace engine when `sigma ~ 1`. A phylogenetic random intercept on the
mean, `phylo(1 | species)`, also uses the sparse-Laplace engine; both
routes require constant σ (overdispersion). One ordinary mean random intercept
or one independent numeric mean slope also supports a predictor-dependent SD
with `sd(g) ~ predictors`, provided `sigma ~ 1`; the SD predictors must be
constant within groups. For a random slope, the SD is the coefficient SD and
the observation-level slope predictor multiplies that coefficient in the mean
linear predictor. These routes use adaptive Gauss–Hermite quadrature and do
not add a free Bernoulli-style residual scale.

One ordinary random intercept on `sigma` also supports a predictor-dependent
SD with `sd(group, sigma) ~ predictors`. Here the second selector names the
distributional parameter. The mean formula must be fixed-effect only, and the
SD predictors must be constant within groups. This models between-group
variation in log overdispersion separately from variation in the logit mean.

```julia
fit = drm(bf(@formula(cbind(successes, failures) ~ x), @formula(sigma ~ 1)), BetaBinomial(); data = dat)
fit = drm(bf(@formula(cbind(successes, failures) ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1)), BetaBinomial(); data = dat)
fit_phy = drm(bf(@formula(cbind(successes, failures) ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              BetaBinomial(); data = dat, tree = tr, se = false)
fit_sd = drm(bf(@formula(cbind(successes, failures) ~ x + (1 | g)),
                @formula(sigma ~ 1), @formula(sd(g) ~ treatment)),
              BetaBinomial(); data = dat)
fit_sigma_sd = drm(bf(@formula(cbind(successes, failures) ~ x),
                      @formula(sigma ~ 1 + (1 | g)),
                      @formula(sd(g, sigma) ~ treatment)),
                   BetaBinomial(); data = dat)
```
"""
struct BetaBinomial end

# FX3b: `se = false` is recorded on the fit (`DrmFit.uncertainty`), so
# `convergence_status` does not read the skipped (NaN) covariance as a failure.
@drm_entry function drm(f::DrmFormula, fam::BetaBinomial; se::Bool = true, kwargs...)
    return _withuncertainty(_drm_betabinomial_fit(f, fam; se = se, kwargs...), se)
end

function _drm_betabinomial_fit(f::DrmFormula, fam::BetaBinomial; data, tree = nothing, g_tol::Real = 1e-8,
             se::Bool = true, marginal::Symbol = :LA)
    _drm_entry_guard(f, fam, data; tree = tree)   # #1027 #1022 #1010
    sd_targets = _sd_target_parts(f)
    any(item -> item.target.dpar === :sigma, sd_targets) &&
        return _drm_betabinomial_sigma_sd_regression(f, fam; data, g_tol, se, marginal)
    !isempty(sd_targets) &&
        return _drm_betabinomial_sd_regression(f, fam; data, g_tol, se, marginal)
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, tree = tree, g_tol = g_tol, se = se, marginal = marginal)
    end
    missing_fit !== nothing && return missing_fit

    f.response2 === nothing &&
        error("BetaBinomial() needs a two-column response: bf(cbind(successes, failures) ~ …)")
    marg = _marginal_method(marginal)                     # :LA (default) or :AGHQ (#761)
    isaghq = marg isa AGHQ
    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    fixed_mu, offset_term = _extract_offset(fixed_mu)   # #938: offset(z), coefficient fixed at 1
    offset_term === nothing || (isempty(re) && mv === nothing && st === nothing) ||
        _refuse_offset_route(fam, "a random, structured or `meta_V` term on the mean")
    isaghq && !(length(re) > 1 && st === nothing) &&
        _aghq_reject(fam, "this model (BetaBinomial `marginal = :AGHQ` covers crossed random " *
                          "intercepts `(1 | g) + (1 | h)` only, #761)")
    mv === nothing ||
        error("BetaBinomial() does not support meta_V markers")
    let sigre = _drm_sigma_ranef_laplace_family(f, fam; data, g_tol, se, fixed_mu, re, mv, st,
                                                offset_term = offset_term, isva = false, isaghq = isaghq)
        sigre === nothing || return sigre
    end
    for (pname, r) in f.forms          # only the mean may carry a random effect
        pname === :mu && continue
        _, re2, mv2, st2 = _split_ranef(r)
        (isempty(re2) && mv2 === nothing && st2 === nothing) ||
            error("BetaBinomial(): only the mean formula may carry a random effect")
    end
    _require_finite_inputs(; response=(_table_column(data, f.response), string(f.response)),
                           response_allow_missing=false)
    _require_finite_inputs(; response=(_table_column(data, f.response2), string(f.response2)),
                           response_allow_missing=false)
    s = Float64.(getproperty(data, f.response))          # successes
    fl = Float64.(getproperty(data, f.response2))        # failures
    (all(si -> si ≥ 0 && isinteger(si), s) && all(fi -> fi ≥ 0 && isinteger(fi), fl)) ||
        error("BetaBinomial() requires non-negative integer successes and failures")
    ntr = s .+ fl                                        # trials
    _, Xμ, nmμ = _design(f.response, fixed_mu, data)     # successes column is a dummy LHS
    _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    if st !== nothing
        isempty(re) ||
            error("BetaBinomial() phylo structured effects cannot be combined with ordinary random effects yet")
        kind, grp = st
        kind === :phylo ||
            error("BetaBinomial() currently supports only phylo(1 | group) among structured markers")
        tree === nothing && error("phylo(1 | $grp) needs `tree = ...`")
        (size(Xσ, 2) == 1 && all(x -> x == 1.0, @view Xσ[:, 1])) ||
            error("BetaBinomial() phylo(1 | group) currently supports only a constant sigma formula")
        labels = getproperty(data, grp)
        return _withformula(_fit_betabinomial_phylo_laplace(fam, s, ntr, Xμ, labels, tree, nmμ, nmσ, grp, g_tol; se = se), f)
    end
    if !isempty(re)                    # random effect on the logit mean → GHQ/Laplace
        if length(re) > 1
            all(_re_kind(r[1])[1] === :intercept for r in re) ||
                error("BetaBinomial() supports multiple random effects only as crossed/nested intercepts, e.g. `(1 | g) + (1 | h)`")
            (size(Xσ, 2) == 1 && all(x -> x == 1.0, @view Xσ[:, 1])) ||
                error("BetaBinomial() crossed random intercepts currently support only a constant sigma formula")
            comps = map(re) do r
                grp = r[2]; gidx, G = _group_index(getproperty(data, grp); col = grp)
                (ones(length(s)), gidx, G, String(grp))
            end
            isaghq && return _withformula(_fit_betabinomial_crossed_aghq(fam, s, ntr, Xμ, comps, nmμ, nmσ, g_tol; se = se), f)   # #761
            return _withformula(_fit_betabinomial_crossed_laplace(fam, s, ntr, Xμ, comps, nmμ, nmσ, g_tol; se = se), f)
        end
        (rk, var) = _re_kind(re[1][1]); grp = re[1][2]; gidx, G = _group_index(getproperty(data, grp); col = grp)
        if rk === :intercept                              # (1 | g) → 1-D GHQ
            return _withformula(_fit_betabinomial_ranef(fam, s, ntr, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol), f)
        elseif rk === :corr                               # (1 + x | g) → 2-D GHQ
            _require_finite_data_column(data, var)
            xs = Float64.(getproperty(data, var))
            return _withformula(_fit_betabinomial_corr_ranef(fam, s, ntr, Xμ, Xσ, xs, gidx, G, nmμ, nmσ, grp, g_tol), f)
        else
            error("BetaBinomial() supports `(1 | g)` or `(1 + x | g)` random effects on the mean")
        end
    end
    return _withformula(_fit_betabinomial(fam, s, ntr, Xμ, Xσ, nmμ, nmσ, g_tol; offset = _offset_for_fit(fam, offset_term, data, size(Xμ, 1))), f)
end

# Predictor-dependent SD for one ordinary random coefficient on the log-sigma
# (beta-binomial overdispersion) predictor. The mean remains fixed-effect only.
function _drm_betabinomial_sigma_sd_regression(f::DrmFormula, fam::BetaBinomial; data,
                                               g_tol::Real = 1e-8, se::Bool = true,
                                               marginal::Symbol = :LA,
                                               nAGQ::Int = 11)
    marginal === :LA || throw(ArgumentError(
        "BetaBinomial sigma SD regression currently supports `marginal = :LA` only."))
    nAGQ >= 1 || throw(ArgumentError("nAGQ must be ≥ 1; got $nAGQ"))
    f.response2 === nothing && throw(ArgumentError(
        "BetaBinomial() needs a two-column response: bf(cbind(successes, failures) ~ …)"))
    targets = _sd_target_parts(f)
    length(targets) == 1 || throw(ArgumentError(
        "BetaBinomial sigma SD regression requires exactly one `sd(group, sigma) ~ …` formula."))
    target = only(targets)
    descriptor = target.target
    (descriptor.level === :iid && descriptor.dpar === :sigma &&
     descriptor.block === nothing) ||
        throw(ArgumentError(
            "BetaBinomial sigma SD regression targets one ordinary sigma random coefficient."))
    grp, sd_rhs = descriptor.group, target.rhs
    rhs = Dict(f.forms)
    fixed_mu, re_mu, mv, st = _split_ranef(rhs[:mu])
    mv === nothing && isempty(re_mu) && st === nothing || throw(ArgumentError(
        "BetaBinomial sigma SD regression requires a fixed-effect-only `mu` formula."))
    sigma_rhs = get(rhs, :sigma, nothing)
    sigma_rhs === nothing && throw(ArgumentError("BetaBinomial sigma SD regression requires `sigma`."))
    fixed_sigma, re_sigma, mv_sigma, st_sigma = _split_ranef(sigma_rhs)
    mv_sigma === nothing && st_sigma === nothing && length(re_sigma) == 1 ||
        throw(ArgumentError(
            "BetaBinomial sigma SD regression currently supports one ordinary sigma intercept or independent numeric slope."))
    kind, slope = _re_kind(re_sigma[1][1])
    kind in (:intercept, :slope) || throw(ArgumentError(
        "BetaBinomial sigma SD regression supports one independent sigma intercept or numeric slope."))
    member = kind === :intercept ? nothing : slope
    descriptor.member === member || throw(ArgumentError(
        "BetaBinomial sigma SD target coefficient `$(descriptor.member)` does not match the fitted sigma coefficient `$member`."))
    sigma_group = re_sigma[1][2]
    sigma_group === grp || throw(ArgumentError(
            "`sd($grp, sigma)` must target the sigma random-effect group `$sigma_group`."))
    allowed = Set((:mu, :sigma, target.key))
    all(p -> first(p) in allowed, f.forms) || throw(ArgumentError(
        "BetaBinomial sigma SD regression supports fixed `mu`, one sigma random coefficient, and one sigma-targeted SD formula."))

    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "BetaBinomial sigma SD regression requires observed responses; remove incomplete rows before fitting."))
    successes = Float64.(getproperty(data, f.response))
    failures = Float64.(getproperty(data, f.response2))
    (all(isfinite, successes) && all(isfinite, failures) &&
     all(x -> x >= 0 && isinteger(x), successes) &&
     all(x -> x >= 0 && isinteger(x), failures)) || throw(ArgumentError(
        "BetaBinomial() requires finite non-negative integer successes and failures"))
    trials = successes .+ failures
    all(>(0), trials) || throw(ArgumentError(
        "BetaBinomial sigma SD regression requires positive trial counts"))

    _, Xmu, nmmu = _design(f.response, fixed_mu, data)
    _, Xsigma, nmsigma = _design(f.response, fixed_sigma, data)
    re_design = if kind === :intercept
        ones(length(successes))
    else
        hasproperty(data, slope) || throw(ArgumentError(
            "BetaBinomial sigma random-slope predictor `$slope` is absent from data."))
        raw = getproperty(data, slope)
        raw isa AbstractVector{<:Real} && length(raw) == length(successes) &&
            all(isfinite, raw) || throw(ArgumentError(
                "BetaBinomial sigma random-slope predictor `$slope` must be a finite numeric vector with one value per response."))
        Float64.(raw)
    end
    gidx, G = _group_index(getproperty(data, grp))
    Wsd, nmsd = _sd_group_design(f.response, sd_rhs, data, gidx, G, grp;
                                 schema_cache = f.schema_cache, schema_key = target.key)
    size(Wsd, 2) <= G && rank(Wsd) == size(Wsd, 2) || throw(ArgumentError(
        "BetaBinomial `sd($grp, sigma)` group-level design is rank deficient; use predictors that vary independently across groups."))
    return _withformula(_fit_betabinomial_sigma_sd_regression(
        fam, successes, trials, Xmu, Xsigma, Wsd, re_design, gidx, G,
        nmmu, nmsigma, nmsd, grp, g_tol; nAGQ, se,
        ranef_name = kind === :intercept ? Symbol(grp, "_logsigma") :
            Symbol(grp, "_", slope, "_logsigma")), f)
end

function _fit_betabinomial_sigma_sd_regression(fam::BetaBinomial, successes, trials,
                                               Xmu, Xsigma, Wsd, re_design, gidx, G,
                                               nmmu, nmsigma, nmsd, grp, g_tol;
                                               nAGQ::Int = 11,
                                               se::Bool = true,
                                               ranef_name::Symbol = Symbol(grp, "_logsigma"))
    n = length(successes)
    length(re_design) == n || throw(DimensionMismatch(
        "BetaBinomial sigma random-effect loading must have one value per response."))
    pmu, psigma, psd = size(Xmu, 2), size(Xsigma, 2), size(Wsd, 2)
    sint, nint = round.(Int, successes), round.(Int, trials)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, nAGQ)
    Zre = reshape(Float64.(re_design), n, 1)

    function evaluate(theta; return_modes::Bool = false)
        beta_mu = @view theta[1:pmu]
        beta_sigma = @view theta[(pmu+1):(pmu+psigma)]
        alpha = @view theta[(pmu+psigma+1):(pmu+psigma+psd)]
        eta_mu = Xmu * beta_mu
        eta_sigma = clamp.(Xsigma * beta_sigma, -15.0, 15.0)
        sd_group = exp.(Wsd * alpha)
        bcache = zeros(1, G) # local to each evaluation; profile workers cannot race
        ll = (i, eta) -> _betabinomial_logit_ll(
            nint[i], sint[i], eta_mu[i], exp(-2 * clamp(eta, -15.0, 15.0)))
        total = zero(eltype(theta))
        for gi in 1:G
            Lg = reshape([sd_group[gi]], 1, 1)
            value, bhat = _aghq_group_logint(ll, members[gi], eta_sigma,
                Zre, Lg, rule, @view(bcache[:, gi]))
            bcache[1, gi] = isfinite(_aghq_primal(bhat[1])) ? bhat[1] : 0.0
            total += value
        end
        return return_modes ? (total, vec(bcache)) : total
    end
    nll(theta) = -first(evaluate(theta))
    pbar = clamp(sum(successes) / max(sum(trials), 1), 1e-3, 1 - 1e-3)
    theta0 = zeros(pmu + psigma + psd)
    theta0[1] = log(pbar / (1 - pbar))
    theta0[pmu+1] = log(0.4)
    theta0[(pmu+psigma+1):end] .= Wsd \ fill(log(0.3), G)
    result = Optim.optimize(nll, theta0, Optim.LBFGS(),
        Optim.Options(g_tol = g_tol); autodiff = :forward)
    theta_hat = Optim.minimizer(result)
    V = se ? _vcov_from_hessian(ForwardDiff.hessian(nll, theta_hat)) :
             fill(NaN, length(theta_hat), length(theta_hat))
    _, mode_group = evaluate(theta_hat; return_modes = true)
    blocks = [:mu => 1:pmu, :sigma => (pmu+1):(pmu+psigma),
              :sd => (pmu+psigma+1):(pmu+psigma+psd)]
    names = [:mu => nmmu, :sigma => nmsigma, :sd => nmsd]
    eta_mu = Xmu * theta_hat[1:pmu]
    eta_sigma_fixed = Xsigma * theta_hat[(pmu+1):(pmu+psigma)]
    eta_sigma = eta_sigma_fixed .+ Zre[:, 1] .* mode_group[gidx]
    sd_group = exp.(Wsd * theta_hat[(pmu+psigma+1):end])
    means = Dict(:mu => _logistic.(eta_mu))
    observed = Dict(:mu => successes ./ trials)
    scales = Dict(:sigma => exp.(eta_sigma),
                  :sigma_fixed => exp.(eta_sigma_fixed),
                  :sigma_random_effect => mode_group,
                  :sigma_group_index => Float64.(gidx), :sd_group => sd_group,
                  :sd => sd_group[gidx], :sd_group_index => Float64.(gidx),
                  :trials => Float64.(nint), :aghq_nodes => [Float64(nAGQ)])
    fit = _withiterations(_withnll(DrmFit(fam, blocks, names, theta_hat, V,
        -nll(theta_hat), n, Optim.converged(result), means, observed, scales), nll),
        Optim.iterations(result))
    # The quadrature evaluation already returns the conditional modes for the
    # sigma-axis random coefficient. Expose its modes through the usual ranef() API,
    # with the suffix matching the existing log-sigma random-effect naming.
    return _withranef(fit, Dict(ranef_name => mode_group))
end

# Predictor-dependent SD for one ordinary IID mean intercept or independent
# numeric slope on the logit scale. BetaBinomial overdispersion remains fixed.
function _drm_betabinomial_sd_regression(f::DrmFormula, fam::BetaBinomial; data,
                                         g_tol::Real = 1e-8, se::Bool = true,
                                         marginal::Symbol = :LA,
                                         nAGQ::Int = _SD_IID_AGHQ_K)
    marginal === :LA || throw(ArgumentError(
        "BetaBinomial `sd(group) ~ …` currently supports `marginal = :LA` only."))
    nAGQ >= 1 || throw(ArgumentError("nAGQ must be ≥ 1; got $nAGQ"))
    f.response2 === nothing && throw(ArgumentError(
        "BetaBinomial() needs a two-column response: bf(cbind(successes, failures) ~ …)"))
    targets = _sd_target_parts(f)
    length(targets) == 1 || throw(ArgumentError(
        "BetaBinomial random-effect SD regression requires exactly one `sd(group) ~ …` formula."))
    target = only(targets)
    descriptor = target.target
    (descriptor.level === :iid && descriptor.dpar === :mu &&
     descriptor.block === nothing) || throw(ArgumentError(
        "BetaBinomial SD regression targets only one ordinary mean intercept or independent numeric slope."))
    grp, sd_rhs = descriptor.group, target.rhs
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing || throw(ArgumentError("BetaBinomial `sd($grp)` cannot use meta_V()."))
    st === nothing || throw(ArgumentError(
        "BetaBinomial `sd($grp)` currently requires one ordinary `(1 | $grp)` effect."))
    length(re) == 1 || throw(ArgumentError(
        "BetaBinomial `sd($grp)` currently requires exactly one ordinary random-effect coefficient."))
    kind, slope = _re_kind(re[1][1])
    member = kind === :intercept ? :intercept : kind === :slope ? slope : nothing
    kind in (:intercept, :slope) || throw(ArgumentError(
        "BetaBinomial `sd($grp)` supports `(1 | $grp)` or one independent numeric `(0 + x | $grp)` term; correlated blocks are not implemented."))
    descriptor.member in (nothing, member) || throw(ArgumentError(
        "BetaBinomial SD target `coef($(descriptor.member))` does not match the fitted random-effect coefficient `$(member)`."))
    regrp = re[1][2]
    regrp === grp || throw(ArgumentError(
        "BetaBinomial `sd($grp)` must name the random-effect grouping variable `$regrp`."))
    sigma_rhs = get(rhs, :sigma, nothing)
    sigma_rhs isa ConstantTerm && sigma_rhs == ConstantTerm(1) || throw(ArgumentError(
        "BetaBinomial `sd($grp)` requires a fixed `sigma ~ 1` overdispersion submodel."))
    for (pname, _) in f.forms
        (pname in (:mu, :sigma, Symbol("sd_", grp)) ||
         haskey(f.sd_targets, pname)) && continue
        throw(ArgumentError(
            "BetaBinomial `sd($grp)` supports only `mu`, fixed `sigma ~ 1`, and one `sd($grp)` formula."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "BetaBinomial random-effect SD regression requires observed responses; remove incomplete rows before fitting."))

    successes = Float64.(getproperty(data, f.response))
    failures = Float64.(getproperty(data, f.response2))
    (all(isfinite, successes) && all(isfinite, failures) &&
     all(x -> x >= 0 && isinteger(x), successes) &&
     all(x -> x >= 0 && isinteger(x), failures)) || throw(ArgumentError(
        "BetaBinomial() requires finite non-negative integer successes and failures"))
    trials = successes .+ failures
    all(>(0), trials) || throw(ArgumentError(
        "BetaBinomial random-effect SD regression requires positive trial counts"))
    _, Xmu, nmmu = _design(f.response, fixed_mu, data)
    _, Xsigma, nmsigma = _design(f.response, sigma_rhs, data)
    gidx, G = _group_index(getproperty(data, grp))
    Wsd, nmsd = _sd_group_design(f.response, sd_rhs, data, gidx, G, grp;
                                 schema_cache = f.schema_cache,
                                 schema_key = target.key)
    size(Wsd, 2) <= G && rank(Wsd) == size(Wsd, 2) || throw(ArgumentError(
        "BetaBinomial `sd($grp)` group-level design is rank deficient; use predictors that vary independently across groups."))
    Zre = if kind === :intercept
        ones(length(successes), 1)
    else
        hasproperty(data, slope) || throw(ArgumentError(
            "BetaBinomial random-slope predictor `$slope` is absent from data."))
        values = getproperty(data, slope)
        values isa AbstractVector{<:Real} || throw(ArgumentError(
            "BetaBinomial random-slope predictor `$slope` must be numeric."))
        length(values) == length(successes) && all(isfinite, values) || throw(ArgumentError(
            "BetaBinomial random-slope predictor `$slope` must contain one finite value per response."))
        reshape(Float64.(values), :, 1)
    end
    return _withformula(_fit_betabinomial_sd_regression(fam, successes, trials,
        Xmu, Xsigma, Zre, Wsd, gidx, G, nmmu, nmsigma, nmsd, grp, g_tol;
        nAGQ, se), f)
end

function _fit_betabinomial_sd_regression(fam::BetaBinomial, successes, trials,
                                         Xmu, Xsigma, Zre, Wsd, gidx, G,
                                         nmmu, nmsigma, nmsd, grp, g_tol;
                                         nAGQ::Int = _SD_IID_AGHQ_K,
                                         se::Bool = true)
    n = length(successes)
    pmu, psigma, psd = size(Xmu, 2), size(Xsigma, 2), size(Wsd, 2)
    sint, nint = round.(Int, successes), round.(Int, trials)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, nAGQ)
    bcache = zeros(1, G)
    function nll(theta)
        beta_mu = @view theta[1:pmu]
        beta_sigma = @view theta[(pmu+1):(pmu+psigma)]
        alpha = @view theta[(pmu+psigma+1):(pmu+psigma+psd)]
        eta_mu = Xmu * beta_mu
        eta_sigma = clamp.(Xsigma * beta_sigma, -15.0, 15.0)
        sd_group = exp.(Wsd * alpha)
        ll = (i, eta) -> _betabinomial_logit_ll(
            nint[i], sint[i], eta, exp(-2 * eta_sigma[i]))
        total = zero(eltype(theta))
        for gi in 1:G
            Lg = reshape([sd_group[gi]], 1, 1)
            value, bhat = _aghq_group_logint(ll, members[gi], eta_mu,
                Zre, Lg, rule, @view(bcache[:, gi]))
            bcache[1, gi] = isfinite(bhat[1]) ? bhat[1] : 0.0
            total += value
        end
        return -total
    end
    pbar = clamp(sum(successes) / max(sum(trials), 1), 1e-3, 1 - 1e-3)
    theta0 = zeros(pmu + psigma + psd)
    theta0[1] = log(pbar / (1 - pbar))
    theta0[pmu+1] = -0.5 * log(10.0)
    theta0[(pmu+psigma+1):end] .= Wsd \ fill(log(0.5), G)
    result = Optim.optimize(nll, theta0, Optim.LBFGS(),
        Optim.Options(g_tol = g_tol); autodiff = :forward)
    theta_hat = Optim.minimizer(result)
    V = se ? _vcov_from_hessian(ForwardDiff.hessian(nll, theta_hat)) :
             fill(NaN, length(theta_hat), length(theta_hat))
    blocks = [:mu => 1:pmu, :sigma => (pmu+1):(pmu+psigma),
              :sd => (pmu+psigma+1):(pmu+psigma+psd)]
    names = [:mu => nmmu, :sigma => nmsigma, :sd => nmsd]
    sd_group = exp.(Wsd * theta_hat[(pmu+psigma+1):end])
    means = Dict(:mu => _logistic.(Xmu * theta_hat[1:pmu]))
    observed = Dict(:mu => successes ./ trials)
    scales = Dict(:sigma => exp.(Xsigma * theta_hat[(pmu+1):(pmu+psigma)]),
                  :sd_group => sd_group, :sd => sd_group[gidx],
                  :sd_group_index => Float64.(gidx), :trials => Float64.(nint),
                  :aghq_nodes => [Float64(nAGQ)])
    return _withiterations(_withnll(DrmFit(fam, blocks, names, theta_hat, V,
        -nll(theta_hat), n, Optim.converged(result), means, observed, scales), nll),
        Optim.iterations(result))
end

# Beta-binomial GLMM with a random intercept (1|g) on the logit mean. b_g ~ N(0,σ_b²)
# integrated out per group by 32-node Gauss–Hermite quadrature (b = √2 σ_b z); the
# precision φ = 1/σ² stays a fixed effect. Same scheme as the Gamma/count GLMMs.
function _fit_betabinomial_ranef(fam::BetaBinomial, s, ntr, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol; K::Int = _RANEF1D_AGHQ_K)
    n = length(s); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    sint = round.(Int, s); nint = round.(Int, ntr)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, K); Zre = ones(n, 1); bcache = zeros(1, G)   # #719: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; σb = exp(θ[pμ+pσ+1])
        η0 = Xμ * βμ; ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        ll = (i, η) -> _betabinomial_logit_ll(nint[i], sint[i], η, exp(-2 * ησ[i]))
        L = reshape([σb], 1, 1)
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    p̄ = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)
    θ0 = zeros(pμ + pσ + 1)
    θ0[1] = log(p̄ / (1 - p̄))                                # logit p̄
    θ0[pμ+1] = -0.5 * log(10.0)                             # moderate precision init (φ ≈ 10)
    θ0[pμ+pσ+1] = log(0.5)
    res = _optimize_with_fallback(nll, θ0, g_tol)
    θ̂ = Optim.minimizer(res); V = _vcov_or_nan(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+1)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp)]]
    means = Dict(:mu => _logistic.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => s ./ ntr)   # population μ (b=0)
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]), :trials => Float64.(nint))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Beta-binomial GLMM with a correlated random intercept+slope (1 + x | g) on the
# logit mean. Per group (b0,b1) ~ N(0, Σ); logit μ_i = Xμ_iᵀβ + b0_g + b1_g·x_i.
# Because groups are disjoint the per-group 2-D integral factorises; it is done by
# per-group ADAPTIVE Gauss–Hermite quadrature
# (`_aghq_marginal_loglik`, #834: nodes b̂_g + √2 C z at each group's mode), `nq` nodes per axis. Σ is the log-Cholesky
# parameterisation L = [exp(a) 0; cc exp(b)] (the `vc` convention), so vc(fit)
# reconstructs Σ = L Lᵀ. The precision φ = 1/σ² stays a fixed effect.
function _fit_betabinomial_corr_ranef(fam::BetaBinomial, s, ntr, Xμ, Xσ, xs, gidx, G, nmμ, nmσ, grp, g_tol; nq::Int = _CORR_RANEF_AGHQ_K)
    n = length(s); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    sint = round.(Int, s); nint = round.(Int, ntr)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(2, nq); Zre = hcat(ones(n), Float64.(xs)); bcache = zeros(2, G)   # #834: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        L = _corr_ranef_L(θ[pμ+pσ+1], θ[pμ+pσ+2], θ[pμ+pσ+3])
        η0 = Xμ * βμ; ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        ll = (i, η) -> _betabinomial_logit_ll(nint[i], sint[i], η, exp(-2 * ησ[i]))
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    p̄ = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)
    θ0 = zeros(pμ + pσ + 3)
    θ0[1] = log(p̄ / (1 - p̄))                                # logit p̄
    θ0[pμ+1] = -0.5 * log(10.0)                             # moderate precision init (φ ≈ 10)
    θ0[pμ+pσ+1] = log(0.4); θ0[pμ+pσ+2] = log(0.4); θ0[pμ+pσ+3] = 0.0
    res = _optimize_with_fallback(nll, θ0, g_tol)
    θ̂ = Optim.minimizer(res); V = _vcov_or_nan(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :recov => (pμ+pσ+1):(pμ+pσ+3)]
    names = [:mu => nmμ, :sigma => nmσ, :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"]]
    means = Dict(:mu => _logistic.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => s ./ ntr)
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]), :trials => Float64.(nint))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

function _fit_betabinomial(fam::BetaBinomial, s, ntr, Xμ, Xσ, nmμ, nmσ, g_tol; offset = nothing)
    n = length(s); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    sint = round.(Int, s); nint = round.(Int, ntr)
    w = _ctl_weights(n)                  # prior weights: row log-lik multipliers (#931)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        ημ = _eta_with_offset(Xμ, βμ, offset)                             # unclamped: see `_betabinomial_logit_ll`
        ησ = clamp.(Xσ * βσ, -15.0, 15.0)        # φ = exp(-2ησ) > 0 finite
        v = zero(eltype(θ))
        @inbounds for i in 1:n
            v -= w[i] * (_betabinomial_logit_ll(nint[i], sint[i], ημ[i], exp(-2 * ησ[i])))
        end
        return v
    end
    p̄ = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)   # overall success rate
    θ0 = zeros(pμ + pσ)
    θ0[1] = log(p̄ / (1 - p̄))                                # logit p̄
    θ0[pμ+1] = -0.5 * log(10.0)                             # moderate precision init (φ ≈ 10)
    θ0[1:pμ] .= _offset_start(θ0[1:pμ], Xμ, offset)   # #938
    res = _ctl_optimize((x0, opts, method) -> Optim.optimize(nll, x0, method, opts; autodiff = :forward),
                       θ0; g_tol = g_tol)   # control: maxit / start / multi_start (#948 #994)
    θ̂ = Optim.minimizer(res); V = _ctl_se(() -> _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂)), θ̂)
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ)]
    names = [:mu => nmμ, :sigma => nmσ]
    means = Dict(:mu => _logistic.(_eta_with_offset(Xμ, θ̂[1:pμ], offset)))           # fitted mean success probability
    obs = Dict(:mu => s ./ ntr)                             # observed proportion (for residuals)
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]), :trials => Float64.(nint))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end
