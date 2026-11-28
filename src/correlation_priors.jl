using QuadGK: quadgk

# Correlation support is route-specific: residual rho12 and ordinary q=2
# random effects follow the guards used by drmTMB's corresponding links.
const DRM_RHO12_PRIOR_GUARD = 0.999999
const DRM_Q2_CORRELATION_PRIOR_GUARD = 0.99999999

"""A normalized LKJ density on one named natural-correlation target."""
struct CorrelationLKJPrior <: AbstractDrmPrior
    dpar::Symbol
    group::String
    coef::String
    from_coef::String
    to_coef::String
    eta::Float64
    scale::Symbol
end

"""`drm_prior_lkj(; dpar, eta, ...)` specifies a normalized LKJ prior on natural correlation.

Use `dpar=:rho12` for a constant bivariate residual correlation, or
`dpar=:mu, group, from_coef, to_coef` for an ordinary Gaussian q=2
random-intercept/slope block. At fit time, the density is normalized over the
resolved correlation guard. The model resolver must verify the selected route
and support before applying it.
"""
function drm_prior_lkj(; dpar, eta, group="", coef=nothing,
                       from_coef="", to_coef="", scale=:natural_correlation)
    d = Symbol(dpar)
    eta isa Real && isfinite(eta) && eta > 1 ||
        throw(ArgumentError("LKJ eta must be finite and greater than 1"))
    e = Float64(eta)
    g = String(group)
    c = coef === nothing ? (d === :rho12 ? "(Intercept)" : "") : String(coef)
    from = String(from_coef)
    to = String(to_coef)
    sc = Symbol(scale)
    isfinite(e) && e > 1 ||
        throw(ArgumentError("LKJ eta is outside the representable range"))
    sc === :natural_correlation ||
        throw(ArgumentError("LKJ prior scale must be natural_correlation"))
    if d === :rho12
        c == "(Intercept)" && isempty(g) && isempty(from) && isempty(to) ||
            throw(ArgumentError("rho12 LKJ prior requires coef=(Intercept) and no group or pair selectors"))
    elseif d === :mu
        isempty(g) && throw(ArgumentError("random-slope LKJ prior requires an exact group name"))
        isempty(c) || throw(ArgumentError("random-slope LKJ prior uses from_coef/to_coef, not coef"))
        !isempty(from) && !isempty(to) && from != to &&
            (from == "(Intercept)" || to == "(Intercept)") ||
            throw(ArgumentError("random-slope LKJ prior requires a distinct (Intercept)/slope coefficient pair"))
        if from != "(Intercept)"
            from, to = to, from
        end
    else
        throw(ArgumentError("LKJ prior dpar must be rho12 or mu"))
    end
    CorrelationLKJPrior(d, g, c, from, to, e, sc)
end

function _lkj_log_normalizer(eta, guard)
    eta isa Real && isfinite(eta) && eta > 1 ||
        throw(ArgumentError("LKJ eta must be finite and greater than 1"))
    guard isa Real && isfinite(guard) && 0 < guard <= 1 ||
        throw(ArgumentError("correlation support bound must be finite and in (0, 1]"))
    e = Float64(eta)
    g = Float64(guard)
    # A guard below one defines a truncated natural-correlation density on
    # (-g, g); guard == 1 uses the ordinary full-support LKJ beta normalizer.
    g == 1.0 && return loggamma(0.5) + loggamma(e) - loggamma(e + 0.5)
    e == 2.0 && return log(2 * (g - g^3 / 3))
    value, _ = quadgk(rho -> (1 - rho^2)^(e - 1), -g, g; rtol=1e-12, atol=1e-14)
    isfinite(value) && value > 0 ||
        throw(ArgumentError("LKJ normalizer is not representable"))
    log(value)
end

function _lkj_log_prior(eta, rho, guard)
    isfinite(rho) && abs(rho) < guard ||
        throw(ArgumentError("correlation lies outside the resolved LKJ support"))
    (Float64(eta) - 1) * log1p(-rho^2) - _lkj_log_normalizer(eta, guard)
end

function _lkj_negative_log_prior(eta, rho, guard)
    abs(rho) < guard || return oftype(rho, 1e18)
    -_lkj_log_prior(eta, rho, guard)
end

function _random_effect_correlation_prior_value(row, theta)
    i = row.index
    cc = theta[i + 2]
    rho = cc / hypot(cc, exp(theta[i + 1]))
    return _lkj_negative_log_prior(row.eta, rho, row.guard)
end
