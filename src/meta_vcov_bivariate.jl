# meta_vcov_bivariate.jl — A8: known row-paired sampling covariance for
# bivariate meta-analysis, the Julia twin of drmTMB's `meta_vcov_bivariate()`
# (drmTMB R/meta-vcov.R) *plus the engine path that consumes it*.
#
# The A4d design pass refused to port the constructor alone because DRModels.jl had
# no consumer — an exported function whose output nothing accepts closes an
# export-name gap without closing a capability gap. This file ships both halves:
# the constructor and the per-row known-V extension of the bivariate residual
# engine (`_fit_bivariate_residual`), so the port is a capability, not a name.
#
# MODEL. Per study i with responses (y1, y2):
#
#     (y1, y2)_i ~ N( (mu1, mu2)_i,  V_i + Sigma_het,i )
#     V_i         = [[v1_i, cov12_i], [cov12_i, v2_i]]        (KNOWN)
#     Sigma_het,i = [[s1^2, rho*s1*s2], [rho*s1*s2, s2^2]]    (fitted; per-row
#                    via the sigma1 / sigma2 / rho12 formulas)
#
# exactly drmTMB's bivariate known-V model (test-biv-gaussian.R verifies the
# same sum against a base-R MVN computation; test/test_meta_vcov_bivariate.jl
# ports that check).
#
# SURFACE. drmTMB spells consumption `bf(mu1 = y1 ~ x + meta_V(V = V), ...)`.
# That spelling uses a KEYWORD argument inside a formula, which StatsModels'
# @formula cannot represent (the same macro limitation that blocks `corpair`,
# documented in docs/dev-log/design/2026-08-15-a4d-design.md). DRModels.jl therefore
# takes the object as a fit-call keyword — `drm(bfbiv, Gaussian(); data, V = …)`
# — the same place `tree`/`K`/`A` already live. For the R→Julia bridge this is
# the natural shape anyway: drmTMB parses `meta_V(V = …)` on the R side and
# ships arrays in the payload, never Julia formula syntax.

"""
    MetaVcovBivariate

Known row-paired sampling covariance for bivariate meta-analysis — the object
[`meta_vcov_bivariate`](@ref) builds and `drm(...; V = …)` consumes. Fields
`v1`, `v2`, `cov12`, one entry per study.

`Matrix(V)` materialises drmTMB's dense `2n × 2n` block-diagonal form (stacking
order `y1[1], y2[1], y1[2], …`); the constructor also accepts such a matrix
back, refusing any cross-study (off-block) entry.
"""
struct MetaVcovBivariate
    v1::Vector{Float64}
    v2::Vector{Float64}
    cov12::Vector{Float64}
end

"""
    meta_vcov_bivariate(v1, v2; cov12 = nothing, cor12 = nothing) -> MetaVcovBivariate

Build the known row-paired sampling covariance for a bivariate meta-analysis —
drmTMB's `meta_vcov_bivariate()`. One entry per study: sampling variances `v1`,
`v2` and either sampling covariances `cov12` or sampling correlations `cor12`
(scalar or vector; at most one of the two, default independence).

Consumed by the bivariate Gaussian route as a fit-call keyword:

```julia
V = meta_vcov_bivariate(v1, v2; cor12 = 0.6)
fit = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
             sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
             rho12 = @formula(rho12 ~ 1)),
          Gaussian(); data = dat, V = V)
```

The fitted `sigma1` / `sigma2` are the **between-study heterogeneity** SDs and
`rho12` the **residual (heterogeneity) correlation** — the known sampling
covariance is added on top per row, never absorbed into them. (This mirrors the
univariate `meta_V` contract: `sigma` is heterogeneity alone, `V` separate.)

!!! note "drmTMB spelling"
    drmTMB writes `bf(mu1 = y1 ~ x + meta_V(V = V), …)`. A keyword argument
    inside a formula is not representable in StatsModels' `@formula`, so DRModels.jl
    takes the object via the `V =` keyword instead — the same place `tree` /
    `K` / `A` live. Writing `meta_V(...)` inside a bivariate formula errors
    with a pointer to this keyword.
"""
function meta_vcov_bivariate(v1::AbstractVector{<:Real}, v2::AbstractVector{<:Real};
                             cov12 = nothing, cor12 = nothing)
    n = length(v1)
    n > 0 || throw(ArgumentError(
        "meta_vcov_bivariate: `v1` and `v2` must be non-empty"))
    length(v2) == n || throw(ArgumentError(
        "meta_vcov_bivariate: `v1` and `v2` must have equal length " *
        "(got $(n) and $(length(v2)))"))
    all(x -> isfinite(x) && x > 0, v1) || throw(ArgumentError(
        "meta_vcov_bivariate: every sampling variance in `v1` must be a positive finite number"))
    all(x -> isfinite(x) && x > 0, v2) || throw(ArgumentError(
        "meta_vcov_bivariate: every sampling variance in `v2` must be a positive finite number"))
    (cov12 === nothing || cor12 === nothing) || throw(ArgumentError(
        "meta_vcov_bivariate: supply at most one of `cov12` and `cor12`"))

    c = if cor12 !== nothing
        r = cor12 isa Real ? fill(Float64(cor12), n) : Vector{Float64}(cor12)
        length(r) == n || throw(ArgumentError(
            "meta_vcov_bivariate: `cor12` must be a scalar or length-$(n) vector"))
        all(x -> isfinite(x) && abs(x) <= 1, r) || throw(ArgumentError(
            "meta_vcov_bivariate: every `cor12` must be a finite number in [-1, 1]"))
        r .* sqrt.(Float64.(v1) .* Float64.(v2))
    elseif cov12 !== nothing
        cc = cov12 isa Real ? fill(Float64(cov12), n) : Vector{Float64}(cov12)
        length(cc) == n || throw(ArgumentError(
            "meta_vcov_bivariate: `cov12` must be a scalar or length-$(n) vector"))
        all(isfinite, cc) || throw(ArgumentError(
            "meta_vcov_bivariate: every `cov12` must be finite"))
        cc
    else
        zeros(n)
    end
    # Each 2x2 sampling block must be PSD on its own — a cov12 exceeding
    # sqrt(v1*v2) is not a covariance matrix, and refusing here beats letting a
    # later fit wander into an indefinite total covariance.
    for i in 1:n
        c[i]^2 <= v1[i] * v2[i] + 1e-12 || throw(ArgumentError(
            "meta_vcov_bivariate: study $(i) has |cov12| > sqrt(v1*v2) " *
            "($(c[i]) vs $(sqrt(v1[i]*v2[i]))) — not a valid sampling covariance block"))
    end
    return MetaVcovBivariate(Float64.(v1), Float64.(v2), c)
end

Base.length(V::MetaVcovBivariate) = length(V.v1)

# drmTMB's dense shape: 2n x 2n block-diagonal, stack order y1[i], y2[i].
function Base.Matrix(V::MetaVcovBivariate)
    n = length(V)
    M = zeros(2n, 2n)
    @inbounds for i in 1:n
        r = 2i - 1
        M[r, r] = V.v1[i]
        M[r+1, r+1] = V.v2[i]
        M[r, r+1] = M[r+1, r] = V.cov12[i]
    end
    return M
end

# Dense drmTMB-shaped matrix -> compact form. Refuses cross-study entries: the
# engine supports row-paired blocks only, and silently dropping off-block
# covariance would fit a DIFFERENT model than the matrix describes.
function MetaVcovBivariate(M::AbstractMatrix{<:Real})
    size(M, 1) == size(M, 2) || throw(ArgumentError(
        "MetaVcovBivariate: the dense form must be square (got $(size(M)))"))
    iseven(size(M, 1)) || throw(ArgumentError(
        "MetaVcovBivariate: the dense form must be 2n × 2n (got $(size(M, 1)) rows)"))
    n = size(M, 1) ÷ 2
    v1 = Vector{Float64}(undef, n); v2 = similar(v1); c = similar(v1)
    @inbounds for i in 1:n
        r = 2i - 1
        v1[i] = M[r, r]; v2[i] = M[r+1, r+1]
        M[r, r+1] ≈ M[r+1, r] || throw(ArgumentError(
            "MetaVcovBivariate: block $(i) is not symmetric"))
        c[i] = M[r, r+1]
    end
    # Everything outside the row-paired 2×2 blocks must be exactly zero —
    # silently dropping cross-study covariance would fit a DIFFERENT model than
    # the matrix describes. (Read-only scan; the caller's matrix is never touched.)
    @inbounds for j in 1:2n, i in 1:2n
        same_block = (i - 1) ÷ 2 == (j - 1) ÷ 2
        if !same_block && !iszero(M[i, j])
            throw(ArgumentError(
                "MetaVcovBivariate: non-zero entry at ($(i), $(j)) outside the " *
                "row-paired 2×2 blocks; cross-study sampling covariance is not supported"))
        end
    end
    return meta_vcov_bivariate(v1, v2; cov12 = c)
end

# Normalise what `drm(...; V = ...)` accepts.
_resolve_biv_meta_v(V::Nothing, n::Int) = nothing
function _resolve_biv_meta_v(V::MetaVcovBivariate, n::Int)
    length(V) == n || throw(ArgumentError(
        "drm: `V` has $(length(V)) studies but the data have $(n) rows"))
    return V
end
_resolve_biv_meta_v(V::AbstractMatrix{<:Real}, n::Int) =
    _resolve_biv_meta_v(MetaVcovBivariate(Matrix{Float64}(V)), n)
_resolve_biv_meta_v(V, n::Int) = throw(ArgumentError(
    "drm: `V` must be a `meta_vcov_bivariate(...)` object or the dense 2n×2n " *
    "drmTMB-shaped matrix (got $(typeof(V)))"))
