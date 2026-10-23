# Shared checks for confidence levels and bootstrap replicate counts, plus the
# completion / boundary status the percentile routines report. One helper so
# `confint`, `coeftable`, `bootstrap_*`, profile intervals, and the derived
# ratio accessors reject the same bad `level` / `B` with the same message.

"""
    _validate_ci_level(level; what = "") -> Float64

Require `level` to be one finite number in `(0, 1)`. `what` prefixes the error
when a caller wants its own name in the message (`"confint: level must ..."`).
"""
function _validate_ci_level(level; what::AbstractString="")
    prefix = isempty(what) ? "level" : "$(what): level"
    value_ok = (level isa Real) && !(level isa Bool) && isfinite(float(level)) &&
               (0 < float(level) < 1)
    value_ok || throw(ArgumentError(
        "$prefix must be one finite number in (0, 1), got $(repr(level))"))
    return float(level)
end

"""
    _validate_bootstrap_B(B) -> Int

Require `B` to be an integer ≥ 1 (a whole-valued real such as `1.0` counts).
`Bool` is rejected: `true` would otherwise become `B = 1`.
"""
function _validate_bootstrap_B(B)
    value_ok = (B isa Real) && !(B isa Bool) && isfinite(float(B)) && isinteger(B) &&
               (1 <= float(B) <= typemax(Int))
    value_ok || throw(ArgumentError(
        "B must be an integer ≥ 1, got $(repr(B))"))
    return Int(B)
end

# A percentile interval whose retained draws pile up on a bound is reporting
# the constraint, not the sampling distribution. The cutoffs match the
# calibration used for random-effect SDs (natural scale < 1e-4) and
# correlations (|ρ| > 0.98): at a 5% share the lower endpoint has already
# collapsed onto the bound. Fewer than 20 retained draws is too small for a
# share (one draw of two would read as 50%), so the flag stays silent there.
# Message recorded when a replicate is dropped for non-convergence rather than
# because the refit threw. `failures = :error` aborts only on thrown refits.
const _BOOTSTRAP_UNCONVERGED = "refit did not converge or landed on a degenerate optimum"

const _BOOTSTRAP_SD_BOUNDARY = 1e-4
const _BOOTSTRAP_RHO_BOUNDARY = 0.98
const _BOOTSTRAP_BOUNDARY_SHARE = 0.05
const _BOOTSTRAP_BOUNDARY_MIN_DRAWS = 20

# Log-scale random-effect / structured SDs. Residual and distributional scales
# (`:sigma`, `:sigma1`, `:sigma2`, `:resid`) are regular parameters and are not
# flagged. Cholesky blocks (`:recov`, `:phylocov`) are not one SD or one
# correlation; `_bootstrap_cov_block_at_boundary` unpacks them first.
const _BOOTSTRAP_LOG_SD_PARAMS = (:resd, :sd, :sd_phylo, :resd_mu, :resd_sigma)
const _BOOTSTRAP_ATANH_COR_PARAMS = (:rho12,)

function _bootstrap_share_at_bound(values, pred)
    n = 0
    hits = 0
    for v in values
        isfinite(v) || continue
        n += 1
        pred(v) && (hits += 1)
    end
    n < _BOOTSTRAP_BOUNDARY_MIN_DRAWS && return false
    return hits / n >= _BOOTSTRAP_BOUNDARY_SHARE
end

function _bootstrap_param_at_boundary(param::Symbol, values)
    if param in _BOOTSTRAP_LOG_SD_PARAMS
        return _bootstrap_share_at_bound(values, v -> exp(v) < _BOOTSTRAP_SD_BOUNDARY)
    end
    if param in _BOOTSTRAP_ATANH_COR_PARAMS
        return _bootstrap_share_at_bound(values, v -> abs(tanh(v)) > _BOOTSTRAP_RHO_BOUNDARY)
    end
    # Natural-scale q=4 among-axis SDs (`sd_*`) and correlations (`cor_*`).
    name = String(param)
    if startswith(name, "sd_")
        return _bootstrap_share_at_bound(values, v -> v < _BOOTSTRAP_SD_BOUNDARY)
    end
    if startswith(name, "cor_")
        return _bootstrap_share_at_bound(values, v -> abs(v) > _BOOTSTRAP_RHO_BOUNDARY)
    end
    return false
end

# Cholesky-parameterised covariance blocks. Each retained draw is a whole
# covariance matrix, so the block is turned into its SDs and correlations
# before the scalar cutoffs apply (#1046). A slope variance of 0 in
# `(1 + x | g)` lives inside `:recov`, never in a scalar `:resd`.
const _BOOTSTRAP_COV_BLOCK_PARAMS = (:recov, :phylocov)

# Covariance matrix for one draw of a Cholesky block.
# `:recov` is the 2×2 GHQ / location-scale convention `[log L11, log L22, L21]`
# (`vc(fit)` decodes it the same way). `:phylocov` is the column-major
# log-Cholesky lower triangle of `lc_to_cov` (q = 2: 3 entries; q = 4: 10).
function _bootstrap_cov_block_matrix(param::Symbol, v::AbstractVector)
    if param === :recov
        length(v) == 3 || return nothing
        l11 = exp(v[1]); l22 = exp(v[2]); cc = v[3]
        return [l11^2 cc * l11; cc * l11 cc^2 + l22^2]
    end
    m = length(v)
    q = round(Int, (sqrt(8m + 1) - 1) / 2)
    q * (q + 1) ÷ 2 == m || return nothing
    return lc_to_cov(collect(Float64, v), q)
end

# `draws` is the `used × length(block)` matrix of one Cholesky block. Returns
# true when any implied SD or correlation meets the same share rule as the
# scalar blocks (natural SD < 1e-4, |ρ| > 0.98, ≥ 5% of ≥ 20 finite draws).
function _bootstrap_cov_block_at_boundary(param::Symbol, draws::AbstractMatrix)
    param in _BOOTSTRAP_COV_BLOCK_PARAMS || return false
    nd = size(draws, 1)
    nd < _BOOTSTRAP_BOUNDARY_MIN_DRAWS && return false
    sds = Vector{Vector{Float64}}()
    cors = Vector{Vector{Float64}}()
    for i in 1:nd
        Σ = _bootstrap_cov_block_matrix(param, @view draws[i, :])
        Σ === nothing && return false
        q = size(Σ, 1)
        if isempty(sds)
            append!(sds, [Float64[] for _ in 1:q])
            append!(cors, [Float64[] for _ in 1:(q * (q - 1) ÷ 2)])
        end
        sd = sqrt.(max.(diag(Σ), 0.0))
        k = 0
        for a in 1:q
            push!(sds[a], sd[a])
            for b in (a + 1):q
                k += 1
                # A zero SD leaves the correlation undefined; the SD itself is
                # what trips the boundary rule in that draw.
                push!(cors[k], sd[a] > 0 && sd[b] > 0 ? Σ[a, b] / (sd[a] * sd[b]) : NaN)
            end
        end
    end
    any(col -> _bootstrap_share_at_bound(col, v -> v < _BOOTSTRAP_SD_BOUNDARY), sds) &&
        return true
    return any(col -> _bootstrap_share_at_bound(col, v -> abs(v) > _BOOTSTRAP_RHO_BOUNDARY),
               cors)
end

function _bootstrap_status(used::Integer, failed::Integer, at_boundary::Bool)
    used >= 2 || return "bootstrap_unavailable"
    at_boundary && return "bootstrap_at_boundary"
    failed > 0 && return "bootstrap_incomplete"
    return "bootstrap"
end

function _warn_bootstrap_boundary(names)
    isempty(names) && return nothing
    @warn "Bootstrap intervals for $(join(names, ", ")) are at a variance-component or correlation boundary. Coverage is unreliable there; resampling does not repair a boundary. The status is `bootstrap_at_boundary`."
    return nothing
end

function _warn_bootstrap_incomplete(; status::AbstractString, failed::Integer,
                                    used::Integer, attempted::Integer)
    report = failed > 0 &&
        (status == "bootstrap_incomplete" || status == "bootstrap_at_boundary")
    report || return nothing
    @warn "Bootstrap intervals were computed after dropping $failed of $attempted refits ($used retained). Failed refits are often the hard draws, so the percentile interval can be too narrow or shifted. The status is `$status`, not a clean `bootstrap` interval."
    return nothing
end

function _finish_bootstrap_status(used::Integer, failed::Integer, boundary_params)
    status = _bootstrap_status(used, failed, !isempty(boundary_params))
    _warn_bootstrap_boundary(boundary_params)
    _warn_bootstrap_incomplete(; status, failed, used, attempted = used + failed)
    return status
end

# Status for one bridge row. `param === nothing` reports the whole replicate
# set (any boundary parameter in the run). A clean coefficient is not labelled
# `bootstrap_at_boundary` just because a different coefficient pinned.
function _bridge_bootstrap_status(result, param=nothing, coef=nothing)
    bounds = hasproperty(result, :boundary_params) ? result.boundary_params : String[]
    at_boundary = if param === nothing
        !isempty(bounds)
    else
        key = string(param, ":", coef)
        bare = string(param)
        any(n -> n == key || n == bare, bounds)
    end
    failed = hasproperty(result, :failed) ? Int(result.failed) : 0
    return _bootstrap_status(Int(result.used), failed, at_boundary)
end
