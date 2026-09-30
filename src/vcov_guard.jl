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
Relative tolerance below which a Hessian eigenvalue counts as numerically zero.
"""
# CALIBRATED 2026-08-26 against the measured distribution, not chosen by feel.
# Was 1e-12, which only caught Hessians singular to MACHINE PRECISION and missed
# merely catastrophic ones -- a fit with a variance of 1.5e+07 in one direction
# reported an SE of ~3937 with no warning at all.
#
# Instrumented every _vcov_from_hessian call in the full suite (1,970 calls) on
# Linux, under both inner-solve tolerances. The distribution is BIMODAL below
# ~1e-7 and a smooth continuum of ordinary condition numbers above it:
#
#   arm            non-singular degenerate cluster    next point    empty gap
#   newton_tol 1e-8    9.84e-09 .. 1.62e-08           1.09e-07      0.83 orders
#   newton_tol 1e-10   4.14e-12 .. 5.53e-09           1.09e-07      1.30 orders
#
# 3e-8 splits that confirmed-empty gap: +0.73 orders above the degenerate cluster
# and +0.56 below the nearest other point. ZERO healthy fits are newly warned on
# in either arm -- the 38 calls that newly fire are 23 fixtures across three
# sparse-Laplace routes, every one with an implied SE of 2,192-6,136.
#
# NOTE this threshold is a fix for a specific hole, NOT a general boundary
# detector: above ~1e-7 the distribution is continuous, so no relative-ratio
# threshold up there could be principled.
#
# ALSO NOTE the hole is PRE-EXISTING, not introduced by tightening newton_tol:
# 6 calls already sit in the missed zone at the OLD tolerance. Tightening the
# inner solve widened it from 6 to 38 by moving fits off exact float zero onto
# the solver's own noise floor -- it made a latent weakness visible.
#
# Falsified by: a genuinely healthy fixture measuring <= ~1e-7; or this cluster's
# top (5.53e-9) creeping toward 3e-8 on another BLAS/LAPACK build -- the margin
# is 0.73 orders, comfortable but not enormous.
const _VCOV_RTOL = 3e-8

"""
    _vcov_from_hessian(H; context = "")

Covariance matrix from an observed-information (Hessian) matrix, guarded against
boundary degeneracy.

Symmetrises `H`, then inverts it — unless it is numerically singular, in which
case it falls back to the Moore–Penrose pseudo-inverse **and warns**, naming the
parameter coordinates that are flat. The decision is made from the eigenvalues
rather than from whether `inv` happens to throw, so the result does not depend
on the LAPACK build or CPU.

The pseudo-inverse keeps a fit usable, but standard errors for the flagged
coordinates are not trustworthy: at a variance boundary the sampling
distribution is not the usual normal approximation. Use `method = :profile` or
the bootstrap entry points for those targets (see `src/inference.jl`), or the
χ̄² boundary machinery in `src/chibar.jl`.
"""
function _vcov_from_hessian(H::AbstractMatrix; context::AbstractString = "")
    Hs = Matrix(H)
    # A non-finite Hessian is the upstream failure signal (see `_finite_hessian`,
    # `_fd_hessian_from_*`): report the NaN-vcov convention rather than crash in
    # `eigvals` (the failure was already warned about upstream).
    all(isfinite, Hs) || return fill(NaN, size(Hs)...)
    Hs = Matrix(Symmetric((Hs + Hs') / 2))
    isempty(Hs) && return Hs

    ev = eigvals(Symmetric(Hs))
    scale = maximum(abs, ev)

    if scale == 0 || minimum(abs, ev) <= _VCOV_RTOL * scale
        d = abs.(diag(Hs))
        dmax = maximum(d)
        flat = dmax == 0 ? collect(eachindex(d)) : findall(<=(_VCOV_RTOL * dmax), d)
        @warn """
              Hessian is numerically singular at the optimum — using a pseudo-inverse.
              The fit is at a variance/dispersion boundary, so standard errors for the
              flagged coordinates are NOT trustworthy. Prefer `confint(fit, method = :profile)`
              or the bootstrap entry points for those targets.
              """ context flat_coordinates = flat rcond = scale == 0 ? 0.0 : minimum(abs, ev) / scale
        V0 = pinv(Hs)
    else
        V0 = inv(Hs)
    end

    return Matrix(Symmetric((V0 + V0') / 2))
end

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
