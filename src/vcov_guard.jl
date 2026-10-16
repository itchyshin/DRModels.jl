# vcov_guard.jl — one guarded Hessian→covariance step for every fitter.
#
# Every family fitter used to end with a bare `V = inv(ForwardDiff.hessian(nll, θ̂))`. That is
# unsafe whenever the MLE sits on a variance/dispersion boundary, where the
# corresponding coordinate is unidentified and the Hessian is singular *by
# construction* — e.g. fitting NB2 to Poisson-generated data drives the size
# parameter to r → ∞, leaving that row/column numerically zero.
#
# Relying on `inv` to throw is not enough, and this is the bug it caused: on
# 2026-08-14 the `anchor (c)` VA test (test/test_variational.jl:105) threw
# SingularException(3) on CI Julia 1.10.11 and passed on 1.10.0 locally and on
# 1.11 in CI — same seeded data, same code. At the optimum θ̂[3] = −19.06
# (r = 3.6e16), diag(H)[3] = 9.9e-16 and cond(H) = 9.1e16: exactly the knife
# edge where LAPACK either reports a zero pivot or returns a finite but
# meaningless inverse. A try/inv/catch/pinv guard makes the *crash* go away but
# keeps the outcome platform-dependent and, worse, silently reports SE ≈ 0 for a
# parameter that is not estimable.
#
# So decide on the conditioning explicitly, identically on every platform, and
# say so out loud when the fit is at a boundary.

"""
Relative eigenvalue tolerance of the covariance guard, applied to the
EQUILIBRATED Hessian (see [`_vcov_scaling`](@ref)).
"""
# HISTORY. Calibrated 2026-08-26 (#513) at 3e-8 on the UNSCALED Hessian's
# min|eig| / max|eig|, from a census of every guard call in the suite: a bimodal
# distribution with a degenerate cluster at 4e-12 .. 5.5e-9 (log-SDs flat at the
# sparse-Laplace clamp, implied SE 2,192-6,136) and the next point at 1.09e-7.
#
# That unscaled ratio is not a property of the fit. It moves with the units of y
# and of the predictors (#936): `y ~ year` with year = 2000 +/- 10 measured
# 4.4e-12, and the same fit with y * 1e4 measured 5.4e-9 -- inside the
# degenerate cluster -- so a well-identified fit was reported as "at a variance
# boundary", and near 1e-16 `pinv` truncated real directions (y * 1e8: intercept
# SE 13,088 for a true 9.6e6). The ratio is now taken on the equilibrated matrix,
# whose eigenvalues do not move with units. RECALIBRATED 2026-10-16 on that
# matrix; the census and the gap it splits are recorded below.
const _VCOV_RTOL = 3e-8

"""
Curvature floor, in natural units, for the coordinates a caller declares
unit-free (see [`_vcov_scaling`](@ref)).
"""
const _VCOV_UNITFREE_FLOOR = 1e3

# Instrumentation hook for recalibrating the guard: `tools/vcov_guard_census.jl`
# sets it to a `(Hs, context, unitfree) -> nothing` recorder and runs the full
# suite. It is `nothing` in normal use and the guard's decision never reads it.
const _VCOV_CENSUS_HOOK = Ref{Any}(nothing)

"""
    _unitfree_columns(X, offset = 0) -> Vector{Int}

Parameter indices (`offset + j`) of the columns `j` of a design matrix `X` whose
coefficients are dimensionless: columns that take only the values 0 and 1 (an
intercept or an indicator). A coefficient of any other column is measured in the
units of its predictor, so its curvature says nothing about identifiability.
Use it for the dispersion / shape / zero-inflation / correlation formulas
(log, logit or atanh link) when building the `unitfree` argument of
[`_vcov_from_hessian`](@ref). Do NOT use it for an identity-link location
formula: there even the intercept carries the units of `y`.
"""
_unitfree_columns(X::AbstractMatrix, offset::Integer = 0) =
    Int[offset + j for j in axes(X, 2) if all(v -> v == 0 || v == 1, view(X, :, j))]

"""
    _vcov_scaling(Hs, unitfree) -> s::Vector{Float64}

Diagonal scaling that makes the guard's decision independent of units (#936).
The guard tests `M = Diagonal(s) * Hs * Diagonal(s)` instead of `Hs`, with

- `s[k] = 1 / sqrt(|Hs[k, k]|)` for an ordinary coordinate (a regression
  coefficient, whose scale is set by the units of `y` and of its predictor).
  Rescaling such a coordinate rescales its row and column of `Hs` and leaves `M`
  unchanged, so neither an uncentred `year` nor `y * 1e4` moves the decision. A
  coordinate with zero curvature keeps `s[k] = 1` (its row of `M` stays zero).
- `s[k] = 1 / sqrt(max(|Hs[k, k]|, _VCOV_UNITFREE_FLOOR))` for a coordinate in
  `unitfree`: a log standard deviation or dispersion, a correlation on the
  atanh scale, an intercept of a log-link dispersion formula, a cut-point.
  Its curvature is already in natural units, so a flat one -- the signature of
  a variance boundary -- keeps its smallness in `M` instead of being normalised
  away. A test that rescaled every coordinate could not see it: at a variance
  boundary the log-SD's whole row of `Hs` decays together, so its correlation
  row tends to zero and the equilibrated matrix looks perfectly conditioned.

`unitfree` lists parameter indices; see [`_unitfree_columns`](@ref).
"""
function _vcov_scaling(Hs::AbstractMatrix, unitfree)
    p = size(Hs, 1)
    natural = falses(p)
    for k in unitfree
        1 <= k <= p || throw(ArgumentError(
            "unitfree index $k is outside the $(p)x$(p) Hessian"))
        natural[k] = true
    end
    s = Vector{Float64}(undef, p)
    for k in 1:p
        c = abs(Hs[k, k])
        c = natural[k] ? max(c, _VCOV_UNITFREE_FLOOR) : (c == 0 ? one(c) : c)
        s[k] = 1 / sqrt(c)
    end
    return s
end

"""
    _vcov_from_hessian(H; context = "", unitfree = ())

Covariance matrix from an observed-information (Hessian) matrix, guarded against
boundary degeneracy.

Symmetrises `H`, then inverts it, unless it is numerically singular or not
positive definite. A clearly negative eigenvalue means the point is not a
minimum: the function warns and returns a `NaN` covariance rather than
`inv(H)` or a pseudo-inverse. A numerically zero eigenvalue (a variance
boundary, or a combination of parameters the data cannot separate) falls back
to a pseudo-inverse and warns, naming the flat coordinates. The decision is made
from eigenvalues rather than from whether `inv` happens to throw, so the result
does not depend on the LAPACK build or CPU.

The eigenvalues are those of the equilibrated matrix of [`_vcov_scaling`](@ref),
so the decision does not change with the units of `y` or of the predictors
(#936). `unitfree` lists the coordinates whose curvature is meaningful in
absolute terms (log-SDs, log-dispersions, atanh-correlations, cut-points, the
0/1 columns of a dispersion formula); only those can be reported as flat at a
boundary. Leave it empty for a matrix of regression coefficients only. When the
matrix is healthy the result is exactly `inv(H)`; the pseudo-inverse of the
fallback is taken on the equilibrated matrix and mapped back, so it too is
equivariant under a change of units.

The pseudo-inverse keeps a fit usable, but standard errors for the flagged
coordinates are not trustworthy: at a variance boundary the sampling
distribution is not the usual normal approximation. Use `method = :profile` or
the bootstrap entry points for those targets (see `src/inference.jl`), or the
χ̄² boundary machinery in `src/chibar.jl`.
"""
function _vcov_from_hessian(H::AbstractMatrix; context::AbstractString = "", unitfree = ())
    Hs = Matrix(H)
    Hs = Matrix(Symmetric((Hs + Hs') / 2))
    isempty(Hs) && return Hs
    (hook = _VCOV_CENSUS_HOOK[]) === nothing || hook(Hs, context, unitfree)

    s = _vcov_scaling(Hs, unitfree)
    M = Matrix(Symmetric(s .* Hs .* s'))
    ev = eigvals(Symmetric(M))
    scale = maximum(abs, ev)
    # A negative eigenvalue is a saddle or a ridge, not a variance boundary.
    # `minimum(abs, ev)` cannot see it, and `inv` of that matrix has a negative
    # diagonal (DRModels.jl#972). Do not pseudo-invert it either. Congruence by
    # the positive scaling keeps the signs of the eigenvalues (Sylvester).
    if scale != 0 && minimum(ev) < -_VCOV_RTOL * scale
        neg = findall(<( -_VCOV_RTOL * scale), ev)
        @warn """
              Hessian is not positive definite at the optimum. Wald standard errors are withheld.
              A negative eigenvalue means this point is not a minimum, so inv(H) is not a covariance.
              """ context negative_eigenvalue_indices = neg
        return fill(NaN, size(Hs))
    end

    if scale == 0 || minimum(abs, ev) <= _VCOV_RTOL * scale
        m = diag(M)
        flat = scale == 0 ? collect(eachindex(m)) : findall(<=(_VCOV_RTOL * scale), m)
        E = eigen(Symmetric(M))
        v = E.vectors[:, argmin(abs.(E.values))]
        null_dir = findall(>=(0.1 * maximum(abs, v)), abs.(v))
        cause = isempty(flat) ?
            "A combination of parameters is not identified (the Hessian is singular after\n" *
            "rescaling every coordinate to unit curvature), so standard errors for the\n" *
            "coordinates in that direction are NOT trustworthy." :
            "The fit is at a variance/dispersion boundary, so standard errors for the\n" *
            "flagged coordinates are NOT trustworthy."
        @warn """
              Hessian is numerically singular at the optimum — using a pseudo-inverse.
              $cause Prefer `confint(fit, method = :profile)`
              or the bootstrap entry points for those targets.
              """ context flat_coordinates = flat null_direction_coordinates = null_dir rcond = scale == 0 ? 0.0 : minimum(abs, ev) / scale
        V0 = s .* pinv(M) .* s'
    else
        V0 = inv(Hs)
    end

    return Matrix(Symmetric((V0 + V0') / 2))
end

"""
    _vcov_from_hessian_checked(H, data_finite; context = "", unitfree = ()) -> (V, hessian_finite)

[`_vcov_from_hessian`](@ref) for a ForwardDiff Hessian at an optimum whose
objective may have run off to a degenerate limit (#968, #961): a residual scale
collapsing towards 0 overflows the curvature to `Inf`/`NaN`. When the DATA are
finite that is a degenerate fit, not a NaN input, so it warns and returns an
all-`NaN` covariance plus `hessian_finite = false` (callers report
`converged = false`) instead of the LAPACK `ArgumentError` the guard raises on a
non-finite matrix. With non-finite data the guard is called unchanged and keeps
throwing (owner decision 14: NaN-predictor fits rely on it).
"""
function _vcov_from_hessian_checked(H::AbstractMatrix, data_finite::Bool;
                                    context::AbstractString = "", unitfree = ())
    hf = all(isfinite, H)
    (hf || !data_finite) &&
        return (_vcov_from_hessian(H; context = context, unitfree = unitfree), hf)
    @warn "The Hessian is not finite at the optimum (a scale parameter has run off " *
          "towards a degenerate limit, e.g. a constant or badly scaled response). Wald " *
          "standard errors are withheld and the fit is reported as not converged." context
    return (fill(NaN, size(H)), false)
end

"""
    _vcov_from_gram(G; context = "")

Inverse of a design Gram / GLS information block `X'WX`. Its eigenvalue ratio
reflects the units of the predictors (an uncentred `year ≈ 2000` gives ~1e-11),
which says nothing about identifiability, so a positive-definite Gram is inverted
directly and silently (as before the REML sites went through the guard, #956).
Only a matrix that is not numerically positive definite is passed to
[`_vcov_from_hessian`](@ref), which warns and falls back.
"""
function _vcov_from_gram(G::AbstractMatrix; context::AbstractString = "")
    Gs = Matrix(Symmetric((Matrix(G) + Matrix(G)') / 2))
    all(isfinite, Gs) || return _vcov_from_hessian(Gs; context = context)
    F = cholesky(Symmetric(Gs); check = false)
    return issuccess(F) ? Matrix(Symmetric(inv(F))) : _vcov_from_hessian(Gs; context = context)
end

_all_finite(xs...) = all(x -> all(isfinite, x), xs)

"""
    _fd_hessian_from_grad(grad_at, θ̂; hstep = 1e-6, retry_step = 1e-4) -> (H, ok)

Central finite-difference Hessian of an analytic gradient `grad_at(θ)`.
The sparse LSS `eval_core` / `_lss_sparse_multi_objective_and_grad` return an
EMPTY gradient when the nll or gradient is not finite at a probe point. Each
column is retried once with `retry_step`; if the retry also fails (empty,
wrong length, or non-finite), `ok = false` is returned instead of throwing a
`DimensionMismatch`. `scaled = false` uses `hstep` as an absolute step (no
`max(|θ̂ₖ|, 1)` scaling). Callers report the NaN-vcov convention when `!ok`.
"""
function _fd_hessian_from_grad(grad_at, θ̂::AbstractVector; hstep::Real = 1e-6,
                               retry_step::Real = 1e-4, scaled::Bool = true)
    np = length(θ̂)
    H = zeros(np, np)
    usable(g) = length(g) == np && all(isfinite, g)
    for k in 1:np
        gp = gm = Float64[]
        step = scaled ? hstep * max(abs(θ̂[k]), 1.0) : Float64(hstep)
        for (attempt, s) in enumerate((step, Float64(retry_step)))
            step = s
            θp = collect(Float64, θ̂); θm = collect(Float64, θ̂)
            θp[k] += step; θm[k] -= step
            gp = grad_at(θp); gm = grad_at(θm)
            usable(gp) && usable(gm) && break
        end
        (usable(gp) && usable(gm)) || return (H, false)
        H[:, k] .= (gp .- gm) ./ (2 * step)
    end
    H .= 0.5 .* (H .+ H')
    return (H, true)
end

"""
    _fd_hessian_from_values(f, θ̂; hstep = 1e-5, sentinel = 1e16) -> (H, ok)

Central finite-difference Hessian of an objective VALUE `f(θ)` (4-point mixed
differences, upper triangle mirrored). Value-based sibling of
[`_fd_hessian_from_grad`](@ref). Several objectives signal a failed evaluation
(non-PD factor, non-finite nll) with a finite `1e18` penalty rather than `NaN`;
differencing that sentinel yields a huge but finite garbage Hessian that then
passes the eigenvalue guard. Any probe that is non-finite or `≥ sentinel`
returns `ok = false` at once, and the caller reports the NaN-vcov convention.
"""
function _fd_hessian_from_values(f, θ̂::AbstractVector; hstep::Real = 1e-5,
                                 sentinel::Real = 1e16)
    np = length(θ̂)
    H = zeros(np, np)
    usable(v) = isfinite(v) && v < sentinel
    for k in 1:np, j in k:np
        sk = hstep * max(abs(θ̂[k]), 1.0)
        sj = hstep * max(abs(θ̂[j]), 1.0)
        θpp = collect(Float64, θ̂); θpm = copy(θpp); θmp = copy(θpp); θmm = copy(θpp)
        θpp[k] += sk; θpp[j] += sj
        θpm[k] += sk; θpm[j] -= sj
        θmp[k] -= sk; θmp[j] += sj
        θmm[k] -= sk; θmm[j] -= sj
        vpp, vpm, vmp, vmm = f(θpp), f(θpm), f(θmp), f(θmm)
        (usable(vpp) && usable(vpm) && usable(vmp) && usable(vmm)) || return (H, false)
        H[k, j] = (vpp - vpm - vmp + vmm) / (4 * sk * sj)
        H[j, k] = H[k, j]
    end
    return (H, true)
end

"""
    _vcov_from_fd_hessian(H; context = "", unitfree = ())

[`_vcov_from_hessian`](@ref) for a Hessian from `_finite_hessian`, which signals a
failed stencil (non-finite probe or the failed-fit sentinel) with an all-NaN
matrix and a warning. That failure is reported as the same-shape all-NaN vcov (the
NaN-vcov convention of the other guarded Hessian helpers) rather than crashing in
`eigvals`. Deliberately NOT folded into `_vcov_from_hessian`: elsewhere a NaN
Hessian (e.g. NaN in a predictor) must keep throwing.
"""
function _vcov_from_fd_hessian(H::AbstractMatrix; context::AbstractString = "", unitfree = ())
    all(isfinite, H) || return fill(NaN, size(H)...)
    return _vcov_from_hessian(H; context = context, unitfree = unitfree)
end
