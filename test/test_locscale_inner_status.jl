# Inner-mode acceptance is a numerical certificate: positive curvature alone
# cannot turn a nonstationary latent state into `ok=true`.
using DRModels
using Test, LinearAlgebra, SparseArrays

const _INNER_STATUS_KIND = Val(:inner_status_quartic)
const _INNER_STATUS_FINAL_KIND = Val(:inner_status_final_quadratic)
const _INNER_STATUS_NONFINITE_KIND = Val(:inner_status_nonfinite)
const _INNER_STATUS_NONFINITE_GRADIENT_KIND = Val(:inner_status_nonfinite_gradient)
const _INNER_STATUS_OVERFLOW_NORM_KIND = Val(:inner_status_overflow_norm)
const _INNER_STATUS_DAMPING_DIRECTION_KIND = Val(:inner_status_damping_direction)

"""A smooth two-axis data term used to test the estimated-error identity directly."""
struct _EstimatedIdentityKind
    shift::Float64
end

function _tracked_hp_prior(P, a, d)
    setprecision(BigFloat, 256) do
        total = BigFloat(0)
        for j in axes(P, 2), i in axes(P, 1)
            total += BigFloat(d[i]) * BigFloat(P[i, j]) * BigFloat(a[j])
        end
        total
    end
end

function _tracked_prior_tests()
@testset "tracked prior directional arithmetic" begin
    # The tracked result retains the same four product pieces that the source
    # accumulator combines. A deliberately hi-only recurrence misses this
    # cancellation residual, while the tracked pair agrees with BigFloat.
    P = [1e10 0.0; 0.0 1e10]
    a = [1.0, -1.0 + 2eps(1.0)]
    d = [1e-10, 1e-10]
    tracked = DRModels._ls_inner_prior_tracked(P, a, d)
    reference = _tracked_hp_prior(P, a, d)
    @test tracked !== nothing
    finalized = DRModels._ls_tracked_finalize(tracked.hi, tracked.lo, tracked.bound)
    @test finalized !== nothing
    @test abs(BigFloat(finalized[1]) - reference) <= BigFloat(finalized[2])
    @test tracked.bound >= 0

    # Sparse traversal is an exact coordinate-order counterpart, not a dense
    # fallback. Its tracked pair and retained bound match the dense path.
    sparse_tracked = DRModels._ls_inner_prior_tracked(sparse(P), a, d)
    @test sparse_tracked !== nothing
    sparse_finalized = DRModels._ls_tracked_finalize(sparse_tracked.hi, sparse_tracked.lo,
                                                 sparse_tracked.bound)
    @test sparse_finalized == finalized
    @test sparse_tracked.bound == tracked.bound

    # The bound tracks only discarded tails because the hi/lo pair is retained.
    # A deliberately corrupted hi-only consumer loses the retained low component
    # and therefore falls outside that certificate.
    hi = 5.9001541755913926e-5
    lo = -7.399360758696142e-18
    x = -2.0821559432935227e27
    accumulated = DRModels._ls_tracked_add(hi, lo, 0.0, x)
    @test accumulated !== nothing && accumulated[3] > 0
    tracked_sum = DRModels._ls_tracked_finalize(accumulated...)
    exact_sum = setprecision(BigFloat, 256) do
        BigFloat(hi) + BigFloat(lo) + BigFloat(x)
    end
    @test abs(BigFloat(tracked_sum[1]) - exact_sum) <= BigFloat(tracked_sum[2])
    @test abs(BigFloat(accumulated[1]) - exact_sum) > BigFloat(accumulated[3])

    # This one exercises the *discarded-tail* pathway itself.  The hi/lo pair
    # retains 2^-60, while 2^-120 is recorded in B; erasing B invalidates the
    # certificate against the independently summed Float64 inputs.
    tail = DRModels._ls_tracked_add(1.0, 2.0^-60, 0.0, 2.0^-120)
    @test tail !== nothing && tail[3] >= 2.0^-120
    tail_exact = setprecision(BigFloat, 256) do
        BigFloat(1.0) + BigFloat(2.0^-60) + BigFloat(2.0^-120)
    end
    tail_retained_error = abs(BigFloat(tail[1]) + BigFloat(tail[2]) - tail_exact)
    @test tail_retained_error <= BigFloat(tail[3])
    @test !(tail_retained_error <= BigFloat(0)) # forced B = 0 fails

    # Reordering both matrix axes and the matching coordinates preserves the
    # certified value, without relying on an identical accumulation order.
    perm = [2, 1]
    permuted = DRModels._ls_inner_prior_tracked(P[perm, perm], a[perm], d[perm])
    @test permuted !== nothing
    permuted_final = DRModels._ls_tracked_finalize(permuted.hi, permuted.lo, permuted.bound)
    @test abs(BigFloat(permuted_final[1]) - _tracked_hp_prior(P[perm, perm], a[perm], d[perm])) <=
          BigFloat(permuted_final[2])

    # This is deliberately a different dimension and nontrivial coordinate
    # ordering: each representation must certify its own BigFloat value.
    P3 = [3.0 -2.0 0.5; -2.0 4.0 -1.0; 0.5 -1.0 2.0]
    a3 = [0.25, -0.75, 1.5]
    d3 = [-0.5, 1.0, 0.125]
    p3 = [3, 1, 2]
    for (Q, aa, dd) in ((P3, a3, d3), (sparse(P3), a3, d3),
                         (P3[p3, p3], a3[p3], d3[p3]),
                         (sparse(P3[p3, p3]), a3[p3], d3[p3]))
        qtracked = DRModels._ls_inner_prior_tracked(Q, aa, dd)
        @test qtracked !== nothing
        qfinal = DRModels._ls_tracked_finalize(qtracked.hi, qtracked.lo, qtracked.bound)
        @test abs(BigFloat(qfinal[1]) - _tracked_hp_prior(Q, aa, dd)) <= BigFloat(qfinal[2])
    end

    # Exact zeros are permitted, whereas a product whose residual could hide
    # beneath the Float64 subnormal range declines this optional fallback.
    @test DRModels._ls_twoprod_finite(0.0, -1.0) == (0.0, 0.0)
    @test DRModels._ls_twoprod_finite(2.0^-600, 2.0^-600) === nothing
    @test DRModels._ls_twoprod_finite(Inf, 1.0) === nothing
    @test DRModels._ls_twosum_finite(floatmin(Float64) / 2, 0.0) === nothing

    # The separately assembled estimated identity retains the tracked prior
    # through the final sum; its public fields remain finite and it refuses a
    # zero step or an uphill trial.
    kind = _EstimatedIdentityKind(1e12)
    y = zeros(3); eta = zeros(3); psi = zeros(3); gidx = [1, 1, 1]
    Zeta = [1.0 0.0; 0.5 0.5; -0.25 0.75]
    Zpsi = [0.0 1.0; 0.25 0.5; 0.75 -0.5]
    P2 = [1.5 0.2; 0.2 0.9]
    base = [2e-8, -1e-8]
    trial = zeros(2)
    result = DRModels._ls_inner_estimated_change(kind, y, eta, psi, gidx, 1, P2,
                                             Zeta, Zpsi, base, trial)
    sparse_result = DRModels._ls_inner_estimated_change(kind, y, eta, psi, gidx, 1,
                                                    sparse(P2), Zeta, Zpsi, base, trial)
    @test result !== nothing && sparse_result !== nothing
    @test result.estimate == sparse_result.estimate
    @test result.margin == sparse_result.margin
    @test DRModels._ls_inner_estimated_change(kind, y, eta, psi, gidx, 1, P2,
                                          Zeta, Zpsi, base, base) === nothing
    uphill = DRModels._ls_inner_estimated_change(kind, y, eta, psi, gidx, 1, P2,
                                             Zeta, Zpsi, zeros(2), base)
    @test uphill !== nothing && uphill.margin >= 0
end
end

function DRModels._ls_nll(kind::_EstimatedIdentityKind, y, η, ψ)
    0.5 * ((η - y)^2 + ψ^2) + kind.shift
end
DRModels._ls_grad(kind::_EstimatedIdentityKind, y, η, ψ) = (η - y, ψ)
DRModels._ls_hess(::_EstimatedIdentityKind, y, η, ψ) = (1.0, 0.0, 1.0)
# Test-only admission makes the estimator arithmetic independently testable;
# production dispatch admits only `Val(:nb2)` and `Val(:gamma)`.
DRModels._ls_inner_estimated_family(::_EstimatedIdentityKind) = true
DRModels._ls_inner_smooth_endpoints(::_EstimatedIdentityKind, η0, ψ0, gidx, a, trial, Zη, Zψ) = true
DRModels._ls_inner_gradient_envelope(::_EstimatedIdentityKind, y, η, ψ) = (abs(η - y), abs(ψ))

"""Synthetic rounded-objective controls for the guarded Newton polish path."""
struct _InnerRoundingKind
    scenario::Symbol
end

const _INNER_ROUNDING_ULP = eps(1.0)
const _INNER_ROUNDING_CALLS = Ref(0)

function DRModels._ls_joint(kind::_InnerRoundingKind, y, η0, ψ0, gidx, a, P, Zη, Zψ)
    _INNER_ROUNDING_CALLS[] += 1
    s = kind.scenario
    if s === :normal
        return sum(abs2, a) / 2
    elseif s === :backtracked
        return a[1] == 1e-10 ? 1.0 :
               (abs(a[1]) < 1e-20 ? 1.0 + 5 * _INNER_ROUNDING_ULP :
                                1.0 + 3 * _INNER_ROUNDING_ULP)
    else
        return a[1] == (s === :large ? 1.0 : 1e-10) ? 1.0 :
               1.0 + (s === :over ? 5 : 3) * _INNER_ROUNDING_ULP
    end
end

function DRModels._ls_joint_grad(kind::_InnerRoundingKind, y, η0, ψ0, gidx, a, P, Zη, Zψ)
    s = kind.scenario
    if s === :nonstationary
        return a[1] == 1e-10 ? [4e-9, 0.0] : [2e-9, 0.0]
    elseif s === :backtracked
        return a[1] == 1e-10 ? [2e-9, 0.0] : zeros(2)
    elseif s === :damped
        return a[1] == 1e-10 ? [9e-9, 0.0] : zeros(2)
    elseif s === :unchanged
        return [2e-9, 0.0]
    elseif s === :accepted || s === :over || s === :indefinite || s === :nonfinite ||
           s === :backtracked
        return a[1] == 1e-10 ? [2e-9, 0.0] : zeros(2)
    elseif s === :large
        return a[1] == 1.0 ? [3e-9, 0.0] : zeros(2)
    else
        return copy(a)
    end
end

function DRModels._ls_joint_hess(kind::_InnerRoundingKind, y, η0, ψ0, gidx, G, a, P, Zη, Zψ)
    s = kind.scenario
    if s === :nonstationary
        return a[1] == 1e-10 ? 50.0I(2) : 1.0I(2)
    elseif s === :indefinite
        return a[1] == 1e-10 ? 20.0I(2) : -1.0I(2)
    elseif s === :nonfinite
        return a[1] == 1e-10 ? 20.0I(2) : fill(NaN, 2, 2)
    elseif s === :damped
        return a[1] == 1e-10 ? -1.0I(2) : 1.0I(2)
    elseif s === :unchanged
        return 1e20I(2)
    elseif s === :large
        return 3e-9I(2)
    else
        return 20.0I(2)
    end
end

function DRModels._ls_joint(::Val{:inner_status_quartic}, y, η0, ψ0, gidx, a, P, Zη, Zψ)
    return a[1]^4 / 4 + sum(abs2, a) / 2
end
function DRModels._ls_joint_grad(::Val{:inner_status_quartic}, y, η0, ψ0, gidx, a, P, Zη, Zψ)
    return [a[1]^3 + a[1], a[2]]
end
function DRModels._ls_joint_hess(::Val{:inner_status_quartic}, y, η0, ψ0, gidx, G, a, P, Zη, Zψ)
    return [3a[1]^2 + 1 0.0; 0.0 1.0]
end

DRModels._ls_joint(::Val{:inner_status_final_quadratic}, y, η0, ψ0, gidx, a, P, Zη, Zψ) = sum(abs2, a) / 2
DRModels._ls_joint_grad(::Val{:inner_status_final_quadratic}, y, η0, ψ0, gidx, a, P, Zη, Zψ) = copy(a)
DRModels._ls_joint_hess(::Val{:inner_status_final_quadratic}, y, η0, ψ0, gidx, G, a, P, Zη, Zψ) = Matrix{Float64}(I, 2, 2)

# The zero gradient is deliberate: the test isolates the required finite-mode
# guard before the early-convergence return, without involving a response fit.
DRModels._ls_joint(::Val{:inner_status_nonfinite}, y, η0, ψ0, gidx, a, P, Zη, Zψ) = 0.0
DRModels._ls_joint_grad(::Val{:inner_status_nonfinite}, y, η0, ψ0, gidx, a, P, Zη, Zψ) = zeros(2)
DRModels._ls_joint_hess(::Val{:inner_status_nonfinite}, y, η0, ψ0, gidx, G, a, P, Zη, Zψ) = Matrix{Float64}(I, 2, 2)

DRModels._ls_joint(::Val{:inner_status_nonfinite_gradient}, y, η0, ψ0, gidx, a, P, Zη, Zψ) = 0.0
DRModels._ls_joint_grad(::Val{:inner_status_nonfinite_gradient}, y, η0, ψ0, gidx, a, P, Zη, Zψ) = [NaN, 0.0]
DRModels._ls_joint_hess(::Val{:inner_status_nonfinite_gradient}, y, η0, ψ0, gidx, G, a, P, Zη, Zψ) = Matrix{Float64}(I, 2, 2)

DRModels._ls_joint(::Val{:inner_status_overflow_norm}, y, η0, ψ0, gidx, a, P, Zη, Zψ) = sum(abs2, a) / 2
DRModels._ls_joint_grad(::Val{:inner_status_overflow_norm}, y, η0, ψ0, gidx, a, P, Zη, Zψ) = copy(a)
DRModels._ls_joint_hess(::Val{:inner_status_overflow_norm}, y, η0, ψ0, gidx, G, a, P, Zη, Zψ) = Matrix{Float64}(I, length(a), length(a))

const _INNER_DAMPING_K = 2.0^40
const _INNER_DAMPING_A0 = [_INNER_DAMPING_K, 1.0]
const _INNER_DAMPING_G = [2.0^20, 2.0^19]
const _INNER_DAMPING_H = [2 * _INNER_DAMPING_K _INNER_DAMPING_K;
                           _INNER_DAMPING_K 2 * _INNER_DAMPING_K]
function DRModels._ls_joint(::Val{:inner_status_damping_direction}, y, η0, ψ0, gidx, a, P, Zη, Zψ)
    d = a .- _INNER_DAMPING_A0
    return dot(_INNER_DAMPING_G, d) + 0.5 * dot(d, _INNER_DAMPING_H * d)
end
DRModels._ls_joint_grad(::Val{:inner_status_damping_direction}, y, η0, ψ0, gidx, a, P, Zη, Zψ) =
    _INNER_DAMPING_G + _INNER_DAMPING_H * (a .- _INNER_DAMPING_A0)
DRModels._ls_joint_hess(::Val{:inner_status_damping_direction}, y, η0, ψ0, gidx, G, a, P, Zη, Zψ) =
    copy(_INNER_DAMPING_H)

# Objective-flat floor (the Gamma phylo bootstrap stall): every trial ties the
# objective exactly, so `ft <= f0` accepts each step, but the gradient never
# shrinks. Before the stall stop this ran all `maxiter` iterations; counting
# gradient calls makes the iteration budget a deterministic assertion rather
# than a wall-clock one.
const _INNER_FLAT_GRAD_CALLS = Ref(0)
DRModels._ls_joint(::Val{:inner_status_flat_floor}, y, η0, ψ0, gidx, a, P, Zη, Zψ) = 1.0
function DRModels._ls_joint_grad(::Val{:inner_status_flat_floor}, y, η0, ψ0, gidx, a, P, Zη, Zψ)
    _INNER_FLAT_GRAD_CALLS[] += 1
    return [1e-6, 0.0]
end
DRModels._ls_joint_hess(::Val{:inner_status_flat_floor}, y, η0, ψ0, gidx, G, a, P, Zη, Zψ) =
    Matrix{Float64}(I, 2, 2)
# Controls that must still certify: the objective ties exactly on every step,
# but the gradient falls by half (`progress`) or by only 2.5% (`slow`, Hessian
# 40 I) per step. The slow case reaches the bound within `maxiter` from 50x it.
DRModels._ls_joint(::Val{:inner_status_flat_progress}, y, η0, ψ0, gidx, a, P, Zη, Zψ) = 1.0
function DRModels._ls_joint_grad(::Val{:inner_status_flat_progress}, y, η0, ψ0, gidx, a, P, Zη, Zψ)
    _INNER_FLAT_GRAD_CALLS[] += 1
    return 0.5 .* (a .- [1.0, 0.0])
end
DRModels._ls_joint_hess(::Val{:inner_status_flat_progress}, y, η0, ψ0, gidx, G, a, P, Zη, Zψ) =
    Matrix{Float64}(I, 2, 2)
DRModels._ls_joint(::Val{:inner_status_flat_slow}, y, η0, ψ0, gidx, a, P, Zη, Zψ) = 1.0
DRModels._ls_joint_grad(::Val{:inner_status_flat_slow}, y, η0, ψ0, gidx, a, P, Zη, Zψ) =
    a .- [1.0, 0.0]
DRModels._ls_joint_hess(::Val{:inner_status_flat_slow}, y, η0, ψ0, gidx, G, a, P, Zη, Zψ) =
    40.0 * Matrix{Float64}(I, 2, 2)

_inner_status_args(kind) = (
    kind, Float64[], Float64[], Float64[], Int[], 1,
    Matrix{Float64}(I, 2, 2), zeros(0, 2), zeros(0, 2),
)
_inner_status_args4(kind) = (
    kind, Float64[], Float64[], Float64[], Int[], 2,
    Matrix{Float64}(I, 4, 4), zeros(0, 2), zeros(0, 2),
)
_inner_rounding_args(kind) = _inner_status_args(kind)

@testset "location-scale inner-mode stationarity acceptance" begin
    @testset "synthetic strictly convex controls" begin
        args = _inner_status_args(_INNER_STATUS_KIND)
        a0 = [2.0, 0.0]
        a_zero, ch_zero, ok_zero = DRModels._ls_inner_mode(args...; a0=a0, maxiter=0)
        @test !ok_zero
        @test issuccess(ch_zero)
        @test norm([a_zero[1]^3 + a_zero[1], a_zero[2]]) > 1e-9 * (1 + norm(a_zero))

        a_stationary, ch_stationary, ok_stationary = DRModels._ls_inner_mode(
            args...; a0=zeros(2), maxiter=0,
        )
        @test ok_stationary
        @test issuccess(ch_stationary)
        @test all(isfinite, a_stationary)
        @test norm([a_stationary[1]^3 + a_stationary[1], a_stationary[2]]) == 0.0

        a_default, ch_default, ok_default = DRModels._ls_inner_mode(
            args...; a0=[1e50, 0.0],
        )
        @test !ok_default
        @test issuccess(ch_default)
        @test all(isfinite, a_default)
        @test norm([a_default[1]^3 + a_default[1], a_default[2]]) >
              1e-9 * (1 + norm(a_default))

        final_args = _inner_status_args(_INNER_STATUS_FINAL_KIND)
        a_final, ch_final, ok_final = DRModels._ls_inner_mode(final_args...; a0=a0, maxiter=1)
        @test ok_final
        @test issuccess(ch_final)
        @test all(isfinite, a_final)
        @test norm(a_final) <=
              1e-9 * (1 + norm(a_final))

        nonfinite_args = _inner_status_args(_INNER_STATUS_NONFINITE_KIND)
        a_nonfinite, ch_nonfinite, ok_nonfinite = DRModels._ls_inner_mode(
            nonfinite_args...; a0=[Inf, 0.0], maxiter=1,
        )
        @test !ok_nonfinite
        @test issuccess(ch_nonfinite)
        @test !all(isfinite, a_nonfinite)

        nonfinite_gradient_args = _inner_status_args(_INNER_STATUS_NONFINITE_GRADIENT_KIND)
        a_bad_gradient, ch_bad_gradient, ok_bad_gradient = DRModels._ls_inner_mode(
            nonfinite_gradient_args...; a0=zeros(2), maxiter=1,
        )
        @test !ok_bad_gradient
        @test issuccess(ch_bad_gradient)

        overflow_args = _inner_status_args4(_INNER_STATUS_OVERFLOW_NORM_KIND)
        overflow0, ch_overflow0, ok_overflow0 = DRModels._ls_inner_mode(
            overflow_args...; a0=fill(1e308, 4), maxiter=0,
        )
        @test !ok_overflow0
        @test issuccess(ch_overflow0)
        @test all(isfinite, overflow0)
        @test !isfinite(norm(overflow0))

        overflow1, ch_overflow1, ok_overflow1 = DRModels._ls_inner_mode(
            overflow_args...; a0=fill(1e308, 4), maxiter=1,
        )
        @test issuccess(ch_overflow1)
        @test !ok_overflow1 || isfinite(norm(overflow1))
    end

    @testset "guarded ULP-scale Newton polish" begin
        # A full, undamped step moves the represented state, rises by three ULPs,
        # and is stationary under the unchanged certificate. This is the only
        # permitted positive-objective route.
        accepted_args = _inner_rounding_args(_InnerRoundingKind(:accepted))
        accepted, accepted_ch, accepted_ok = DRModels._ls_inner_mode(
            accepted_args...; a0=[1e-10, 0.0], maxiter=1,
        )
        @test accepted_ok
        @test norm(accepted) <= 1e-9 * (1 + norm(accepted))
        @test issuccess(accepted_ch)

        # Each negative control has a finite apparent candidate, but fails one
        # specific clause of the narrow exception.
        for (scenario, start) in ((:over, 1e-10), (:large, 1.0),
                                  (:nonstationary, 1e-10), (:indefinite, 1e-10),
                                  (:damped, 1e-10), (:backtracked, 1e-10))
            @testset "reject $scenario" begin
                args = _inner_rounding_args(_InnerRoundingKind(scenario))
                _, _, ok = DRModels._ls_inner_mode(args...; a0=[start, 0.0], maxiter=1)
                @test !ok
            end
        end

        # A non-finite trial Hessian must be rejected as an uncertified mode,
        # not leak a factorisation error through the inner-mode API.
        bad_hessian_args = _inner_rounding_args(_InnerRoundingKind(:nonfinite))
        _, _, bad_hessian_ok = DRModels._ls_inner_mode(
            bad_hessian_args...; a0=[1e-10, 0.0], maxiter=1,
        )
        @test !bad_hessian_ok

        # Coordinate-identical trial steps are not progress. The guard avoids
        # 200 no-op line-search acceptances while retaining the failure result.
        unchanged_args = _inner_rounding_args(_InnerRoundingKind(:unchanged))
        _INNER_ROUNDING_CALLS[] = 0
        unchanged_one, _, unchanged_one_ok = DRModels._ls_inner_mode(
            unchanged_args...; a0=[1e-10, 0.0], maxiter=1,
        )
        calls_one = _INNER_ROUNDING_CALLS[]
        _INNER_ROUNDING_CALLS[] = 0
        unchanged_many, _, unchanged_many_ok = DRModels._ls_inner_mode(
            unchanged_args...; a0=[1e-10, 0.0], maxiter=200,
        )
        calls_many = _INNER_ROUNDING_CALLS[]
        @test !unchanged_one_ok && !unchanged_many_ok
        @test unchanged_one == unchanged_many == [1e-10, 0.0]
        @test calls_one > 0
        @test calls_many == calls_one
    end

    @testset "unchanged undamped step still tries damping directions" begin
        args = _inner_status_args(_INNER_STATUS_DAMPING_DIRECTION_KIND)
        kind, y, η0, ψ0, gidx, G, P, Zη, Zψ = args
        f0 = DRModels._ls_joint(kind, y, η0, ψ0, gidx, _INNER_DAMPING_A0, P, Zη, Zψ)
        a, ch, ok = DRModels._ls_inner_mode(args...; a0=_INNER_DAMPING_A0, maxiter=1)
        f1 = DRModels._ls_joint(kind, y, η0, ψ0, gidx, a, P, Zη, Zψ)
        @test !ok
        @test issuccess(ch)
        @test a[2] != _INNER_DAMPING_A0[2]
        @test f1 < f0
    end

    @testset "objective-flat steps without gradient progress stop early" begin
        window = DRModels._LS_INNER_FLAT_WINDOW
        _INNER_FLAT_GRAD_CALLS[] = 0
        a, ch, ok = DRModels._ls_inner_mode(
            _inner_status_args(Val(:inner_status_flat_floor))...; a0=zeros(2),
        )
        # `window` loop gradients plus the final certificate's one, not 200 + 1.
        @test _INNER_FLAT_GRAD_CALLS[] == window + 1
        @test !ok
        @test issuccess(ch)
        @test a ≈ [-window * 1e-6, 0.0]

        _INNER_FLAT_GRAD_CALLS[] = 0
        a, ch, ok = DRModels._ls_inner_mode(
            _inner_status_args(Val(:inner_status_flat_progress))...; a0=zeros(2),
        )
        @test ok
        @test _INNER_FLAT_GRAD_CALLS[] > 11
        @test norm(a .- [1.0, 0.0]) <= 1e-8

        # Slow genuine progress through exact ties still certifies, whether it
        # starts just above the bound or far enough above it to count as flat.
        for mult in (1.2, 1.9, 3.0, 50.0)
            a0 = [1.0 - mult * 2e-9, 0.0]
            a, ch, ok = DRModels._ls_inner_mode(
                _inner_status_args(Val(:inner_status_flat_slow))...; a0=a0,
            )
            @test ok
        end
    end

    @testset "actual Gamma kernel reports only a certified mode" begin
        G, m = 2, 3
        gidx = repeat(1:G, inner=m)
        y = [0.9, 1.1, 1.2, 0.8, 1.0, 1.3]
        η0 = zeros(length(y)); ψ0 = zeros(length(y))
        Λ = DRModels._ls_lc_to_Λ([0.0, 0.1, 0.0])
        P = DRModels.prior_precision(sparse(1.0I, G, G), inv(Λ))
        a, ch, ok = DRModels._ls_inner_mode(Val(:gamma), y, η0, ψ0, gidx, G, P)
        gradient = DRModels._ls_joint_grad(Val(:gamma), y, η0, ψ0, gidx, a, P)
        @test ok
        @test all(isfinite, a) && all(isfinite, gradient)
        @test norm(gradient) <= 1e-9 * (1 + norm(a))
        @test issuccess(ch)

        a0 = [1.0, -1.0, 1.0, -1.0]
        a_limited, ch_limited, ok_limited = DRModels._ls_inner_mode(
            Val(:gamma), y, η0, ψ0, gidx, G, P; a0=a0, maxiter=0,
        )
        g_limited = DRModels._ls_joint_grad(Val(:gamma), y, η0, ψ0, gidx, a_limited, P)
        @test !ok_limited
        @test issuccess(ch_limited)
        @test norm(g_limited) > 1e-9 * (1 + norm(a_limited))
    end

    @testset "estimated-error identity fallback" begin
        # The raw objective comparison can be unreliable at this displacement;
        # this unit test calls only the numerical estimator, not a new fit.
        kind = _EstimatedIdentityKind(1e12)
        y = zeros(3)
        η0 = zeros(3); ψ0 = zeros(3); gidx = [1, 1, 1]
        Zη = [1.0 0.0; 0.5 0.5; -0.25 0.75]
        Zψ = [0.0 1.0; 0.25 0.5; 0.75 -0.5]
        P = [1.5 0.2; 0.2 0.9]
        a = [2e-8, -1e-8]
        trial = zeros(2)

        recovered = DRModels._ls_inner_estimated_change(
            kind, y, η0, ψ0, gidx, 1, P, Zη, Zψ, a, trial,
        )
        @test recovered !== nothing
        @test recovered.estimate < 0
        @test recovered.margin < 0
        @test recovered.directional_scale > 0
        @test recovered.prior_scale > 0
        @test recovered.quadrature_scale > 0
        expected = -0.5 * (sum(abs2, Zη * a) + sum(abs2, Zψ * a) + dot(a, P * a))
        @test recovered.estimate ≈ expected rtol=1e-12

        # Row order and additive per-observation constants do not change the
        # smooth identity.  The components are compared independently of raw
        # objective subtraction.
        perm = reverse(eachindex(y))
        reordered = DRModels._ls_inner_estimated_change(
            kind, y[perm], η0[perm], ψ0[perm], gidx[perm], 1, P,
            Zη[perm, :], Zψ[perm, :], a, trial,
        )
        unshifted = DRModels._ls_inner_estimated_change(
            _EstimatedIdentityKind(0.0), y, η0, ψ0, gidx, 1, P, Zη, Zψ, a, trial,
        )
        @test reordered !== nothing && unshifted !== nothing
        @test reordered.estimate ≈ recovered.estimate atol=1e-22 rtol=1e-12
        @test unshifted.estimate ≈ recovered.estimate atol=1e-22 rtol=1e-12

        uphill = DRModels._ls_inner_estimated_change(
            kind, y, η0, ψ0, gidx, 1, P, Zη, Zψ, zeros(2), [2e-8, -1e-8],
        )
        @test uphill !== nothing
        @test !(uphill.margin < 0)

        flat = DRModels._ls_inner_estimated_change(
            _EstimatedIdentityKind(0.0), zeros(1), zeros(1), zeros(1), [1], 1,
            zeros(2, 2), zeros(1, 2), zeros(1, 2), zeros(2), [1e-16, 0.0],
        )
        @test flat !== nothing
        @test !(flat.margin < 0)

        # Unsupported families and any endpoint at/over a clamp boundary do
        # not get a fallback estimate.
        @test DRModels._ls_inner_estimated_change(
            Val(:poisson), [1.0], [0.0], [0.0], [1], 1, zeros(2, 2),
            [1.0 0.0], [0.0 1.0], [0.0, 0.0], [1e-8, 0.0],
        ) === nothing
        @test DRModels._ls_inner_estimated_change(
            Val(:gamma), [1.0], [29.9], [0.0], [1], 1, zeros(2, 2),
            [1.0 0.0], [0.0 1.0], [0.0, 0.0], [0.2, 0.0],
        ) === nothing
        @test DRModels._ls_inner_estimated_change(
            Val(:gamma), [1.0], [0.0], [29.9], [1], 1, zeros(2, 2),
            [1.0 0.0], [0.0 1.0], [0.0, 0.0], [0.0, 0.2],
        ) === nothing
        @test DRModels._ls_inner_estimated_change(
            Val(:nb2), [1.0], [0.0], [-14.9], [1], 1, zeros(2, 2),
            [1.0 0.0], [0.0 1.0], [0.0, 0.0], [0.0, -0.2],
        ) === nothing

        # At y = μ the Gamma eta gradient is exactly zero, but its
        # uncancelled envelope remains positive and protects the directional
        # scale.  There is no prior or psi contribution here.
        gamma_scale = DRModels._ls_inner_estimated_change(
            Val(:gamma), [1.0], [0.0], [0.0], [1], 1, zeros(2, 2),
            [1.0 0.0], zeros(1, 2), zeros(2), [1e-8, 0.0],
        )
        @test gamma_scale !== nothing
        @test gamma_scale.directional_scale >= 2e-8

        # A loading cancellation has zero actual eta displacement, although
        # the two loading products are nonzero before cancellation.
        loading_scale = DRModels._ls_inner_estimated_change(
            Val(:gamma), [1.0], [0.0], [0.0], [1], 1, zeros(2, 2),
            [1.0 1.0], zeros(1, 2), zeros(2), [1e-8, -1e-8],
        )
        @test loading_scale !== nothing
        @test loading_scale.directional_scale >= 4e-8
        @test loading_scale.margin >= 0

        # Both admitted production families use the same arithmetic with
        # noncanonical loadings and a coupled two-group prior.
        gidx2 = [1, 1, 2, 2]
        η2 = [0.15, -0.10, 0.25, -0.20]
        ψ2 = [0.05, 0.10, -0.08, 0.02]
        Zη2 = [1.0 0.0; 0.5 0.25; 1.0 -0.5; -0.25 0.75]
        Zψ2 = [0.0 1.0; 0.2 0.6; 0.0 1.0; 0.5 -0.2]
        P2 = [1.4 0.1 0.2 0.0;
              0.1 1.1 0.0 0.15;
              0.2 0.0 1.3 0.1;
              0.0 0.15 0.1 1.2]
        a2 = [0.10, -0.05, -0.08, 0.04]
        for (family, yy) in ((Val(:gamma), [1.1, 0.9, 1.3, 0.8]),
                             (Val(:nb2), [0.0, 2.0, 1.0, 3.0]))
            g2 = DRModels._ls_joint_grad(family, yy, η2, ψ2, gidx2, a2, P2, Zη2, Zψ2)
            r2 = DRModels._ls_inner_estimated_change(
                family, yy, η2, ψ2, gidx2, 2, P2, Zη2, Zψ2, a2, a2 .- 1e-6 .* g2,
            )
            @test r2 !== nothing
            @test r2.estimate < 0 && r2.margin < 0
        end
    end
end

_tracked_prior_tests()
