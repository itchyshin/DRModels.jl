# rescale_response.jl: fit the simple Gaussian / Student-t model on y / sd(y) (#936).
#
# The unit-dependent flags in `convergence_status` / `is_converged` (absolute SD
# boundary, inflated-SE screen, singular-Hessian ratio) read the response's units, so
# the same fit of `y` and `1e6 * y` could differ in status. Owner ruling
# SHINICHI-0900-1119 (same as drmTMB #1751 / PR #1786): rescale the response inside
# the fitter. First slice, as in drmTMB: Gaussian and Student, fixed effects plus at
# most one ordinary `(1 | g)` on the mean, ML, default controls, and sd(y) outside
# [0.01, 100]. Every other route is untouched and never refused.
#
# Fit y* = y / s (s = sd(y)); the original-scale fit follows from y = s y*:
#   beta_mu      = s * beta_mu*                 (every mean coefficient)
#   beta_sigma_0 = beta_sigma_0* + log(s)       (intercept column only)
#   log_sd_g     = log_sd_g* + log(s)           (the `:resd` block, log SD of (1 | g))
#   beta_nu      = beta_nu*                     (Student df is unit-free)
#   u_g          = s * u_g*;  mean, sigma, conditional eta scale with s
#   vcov         = M V* M,  M = diag(s on the mean block, 1 elsewhere)
#   logLik       = logLik* - n log(s)           (original-scale density)
#   nll(theta)   = nll*(theta*) + n log(s),  theta* the inverse map above

const _RESCALE_WINDOW = (1e-2, 1e2)
const _RESCALE_SCALE_KEYS = (:sigma, :nu, :cond_fit_eta_re)

# The data with `y / s` and the scale, or `nothing` when the slice does not apply.
function _y_rescale_plan(f::DrmFormula, fam, kwargs, allowed::Tuple)
    try
        _active_control_scope() === nothing || return nothing   # control / weights keep the ordinary path
        f.response2 === nothing || return nothing
        all(k -> k in allowed, keys(kwargs)) || return nothing
        haskey(kwargs, :data) || return nothing
        get(kwargs, :method, :ML) === :ML || return nothing
        get(kwargs, :marginal, :LA) === :LA || return nothing
        all(p -> first(p) in (:mu, :sigma, :nu), f.forms) || return nothing
        any(p -> first(p) === :mu, f.forms) || return nothing
        nbar = 0
        for (p, rhs) in f.forms
            terms = rhs isa Tuple ? collect(rhs) : Any[rhs]
            for t in terms
                if t isa FunctionTerm && t.f === (|)
                    (p === :mu && t.args[1] isa ConstantTerm && t.args[1].n == 1 &&
                     t.args[2] isa Term) || return nothing
                    nbar += 1
                elseif !(t isa Term || t isa ConstantTerm || t isa InteractionTerm)
                    return nothing                  # offset, meta_V, sd(), phylo(), log(x), ...
                end
            end
        end
        nbar <= 1 || return nothing
        # A right-hand side that reads the response column would be built on y / s, and
        # the restore map only shifts the sigma intercept: keep the ordinary path.
        any(p -> _rhs_mentions(last(p), f.response), f.forms) && return nothing
        data = kwargs[:data]
        raw = _table_column(data, f.response)
        (raw isa AbstractVector && all(v -> v isa Real && isfinite(v), raw) && length(raw) >= 3) ||
            return nothing
        yv = Float64.(raw)
        s = std(yv)
        (isfinite(s) && s > 0) || return nothing
        _RESCALE_WINDOW[1] <= s <= _RESCALE_WINDOW[2] && return nothing
        return (; s, n = length(yv), data = _replace_table_column(data, f.response, yv ./ s))
    catch
        return nothing
    end
end

# Whether a formula right-hand side (a term, a tuple of terms, or a nested term)
# refers to the column `sym`.
function _rhs_mentions(t, sym::Symbol)
    t isa Term && return t.sym === sym
    (t isa Tuple || t isa AbstractVector) && return any(x -> _rhs_mentions(x, sym), t)
    t isa FunctionTerm && return _rhs_mentions(t.args, sym)
    hasproperty(t, :terms) && return _rhs_mentions(t.terms, sym)   # InteractionTerm
    return false
end

# The original-scale fit built from the fit on y / s, or `nothing` when its layout is
# not the expected one (the caller then fits the original data as before).
function _y_rescale_restore(fit, plan)
    fit isa DrmFit || return nothing
    s, n = plan.s, fit.nobs          # rows the fit used (missing-predictor rows are dropped)
    ls = log(s)
    names = Dict(fit.coefnames)
    blocks = Dict(fit.blocks)
    haskey(blocks, :mu) || return nothing
    all(b -> first(b) in (:mu, :sigma, :nu, :resd), fit.blocks) || return nothing
    haskey(blocks, :sigma) && haskey(names, :sigma) || return nothing
    j0 = findfirst(==("(Intercept)"), names[:sigma])
    j0 === nothing && return nothing
    all(k -> k in _RESCALE_SCALE_KEYS, keys(fit.scales)) || return nothing
    all(==(:mu), keys(fit.means)) && all(==(:mu), keys(fit.obs)) || return nothing
    (fit.ranef === nothing || (fit.ranef isa AbstractDict &&
        all(v -> v isa AbstractVector{<:Real}, values(fit.ranef)))) || return nothing
    (haskey(blocks, :resd) ? length(blocks[:resd]) == 1 : true) || return nothing
    (fit.estim_method === :ML && fit.nllgrad === nothing && length(fit.theta) == length(fit.vcov[:, 1])) ||
        return nothing

    p = length(fit.theta)
    mu_idx = collect(blocks[:mu])
    shift_idx = Int[first(blocks[:sigma]) + j0 - 1]
    haskey(blocks, :resd) && push!(shift_idx, first(blocks[:resd]))
    mscale = ones(p); mscale[mu_idx] .= s            # theta = M theta* (on the mean block) ...
    to_orig(θs) = begin
        θ = collect(θs) .* mscale
        θ[shift_idx] .+= ls
        θ
    end
    to_int(θ) = begin
        θs = collect(θ)
        θs[shift_idx] .-= ls
        θs ./ mscale
    end
    nll_s = fit.nll
    nll = nll_s === nothing ? nothing : (θ -> nll_s(to_int(θ)) + n * ls)
    θ = to_orig(fit.theta)
    V = fit.vcov .* (mscale * mscale')
    scl(d) = Dict{Symbol,Vector{Float64}}(k => (k === :nu ? copy(v) : v .* s) for (k, v) in d)
    means = Dict{Symbol,Vector{Float64}}(k => v .* s for (k, v) in fit.means)
    obs = Dict{Symbol,Vector{Float64}}(k => v .* s for (k, v) in fit.obs)
    ranef = fit.ranef === nothing ? nothing :
            Dict{Symbol,Vector{Float64}}(k => v .* s for (k, v) in fit.ranef)
    return DrmFit(fit.family, fit.blocks, fit.coefnames, θ, V, fit.loglik - n * ls, fit.nobs,
        fit.converged, means, obs, scl(fit.scales), fit.formula, nll, fit.nllgrad, ranef,
        fit.estim_method, fit.reml_loglik, fit.ml_loglik - n * ls, fit.marginal,
        fit.phylo_penalty, fit.penalty, fit.iterations, fit.phylo_scale, fit.control_info,
        fit.uncertainty, fit.geometry)
end
