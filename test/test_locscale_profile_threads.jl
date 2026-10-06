# Canonical location-scale profile jobs are independent across coefficients, but
# each coefficient retains its serial lower/upper warm-start chain.  This small
# public fit keeps the test below the slow full-vector profile fixture.
using DRModels
using Test, Random, LinearAlgebra
import Distributions

function _locscale_profile_threads_fixture()
    Random.seed!(20_260_831)
    # Gamma has a smooth positive-response likelihood and the deliberately
    # separated fixed and group effects make both mean coefficients identifiable
    # in this compact profile fixture.
    G, m = 4, 8
    n = G * m
    species = repeat(1:G, inner=m)
    x = repeat(range(-1.0, 1.0; length=m), G)
    re_mu = 0.16 .* randn(G)
    re_sigma = 0.10 .* randn(G)
    eta = 0.70 .+ 0.55 .* x .+ re_mu[species]
    psi = 1.05 .+ re_sigma[species]
    y = [begin
        shape = exp(psi[i])
        mu = exp(eta[i])
        Float64(rand(Distributions.Gamma(shape, mu / shape)))
    end for i in 1:n]
    fit = drm(
        bf(
            @formula(y ~ x + (1 | profile_thread | species)),
            @formula(sigma ~ 1 + (1 | profile_thread | species)),
        ),
        Gamma();
        data=(; y, x, species),
    )
    @test fit.nll isa DRModels.LocScaleObjective
    return fit
end

@testset "canonical location-scale profile coefficient threading" begin
    fit = _locscale_profile_threads_fixture()
    # This test is about serial/threaded profile agreement on a fitted model;
    # failed endpoints from a nonconverged point fit are not valid substitutes.
    @test is_converged(fit)
    blas_before = BLAS.get_num_threads()

    # The serial and threaded calls repeat the same two profile intervals through
    # independent scheduling paths, guarding against shared warm starts or scratch
    # state.
    serial = profile_result(fit; parm=:mu, threads=false)
    threaded = profile_result(fit; parm=:mu, threads=true)
    @test BLAS.get_num_threads() == blas_before
    @test serial.threaded == false
    @test serial.worker_threads == 1
    @test serial.attempted == serial.used == 2
    @test threaded.attempted == threaded.used == 2
    @test all(r -> isfinite(r.lower) && isfinite(r.upper), serial.ci)
    @test reduce(vcat, ([r.lower r.upper] for r in threaded.ci)) ≈
          reduce(vcat, ([r.lower r.upper] for r in serial.ci)) rtol=1e-7 atol=1e-8
    @test [(s.lower_unbounded, s.upper_unbounded, s.lower_endpoint_failed,
            s.upper_endpoint_failed, s.lower_nuisance_reason, s.upper_nuisance_reason)
           for s in threaded.stats] ==
          [(s.lower_unbounded, s.upper_unbounded, s.lower_endpoint_failed,
            s.upper_endpoint_failed, s.lower_nuisance_reason, s.upper_nuisance_reason)
           for s in serial.stats]
    fallback_flags(result) = [
        (diag.lower.nuisance !== nothing && diag.lower.nuisance.fallback,
         diag.upper.nuisance !== nothing && diag.upper.nuisance.fallback)
        for diag in result.endpoint_diagnostics
    ]
    @test fallback_flags(threaded) == fallback_flags(serial)
    @test [(s.lower_nuisance_fallback, s.upper_nuisance_fallback)
           for s in serial.stats] == fallback_flags(serial)
    @test [(s.lower_nuisance_fallback, s.upper_nuisance_fallback)
           for s in threaded.stats] == fallback_flags(threaded)

    if Threads.nthreads() > 1
        @test threaded.threaded
        @test threaded.worker_threads == min(2, Threads.nthreads())
    else
        # A process without worker threads falls back to the serial route even
        # when the caller requests threading.
        @test threaded.threaded == false
        @test threaded.worker_threads == 1
    end

    # One selected mean slope. Endpoint arms stay
    # serial, so this call must not claim coefficient parallelism.
    one = profile_result(fit; parm=:mu => "x", threads=true)
    @test one.attempted == one.used == 1
    @test one.threaded == false
    @test one.worker_threads == 1
    @test isfinite(only(one.ci).lower) && isfinite(only(one.ci).upper)
    @test BLAS.get_num_threads() == blas_before

    # The contraction budget is reported per arm on the public diagnostic
    # surface, next to the expansion and Newton counts it belongs with.  The
    # COUNT is deliberately not pinned to zero.  Whether a given trial is
    # evaluable at the last bit is a property of the machine, not of the search
    # -- exactly what the next testset's header says about the #651 failure --
    # and it was measured here on 2026-09-06: this same commit gave 0
    # contractions on one x64 linux Julia 1.10.12 runner (CI run 34034716401,
    # 26/26) and 1 on another (run 34037371365), where these two lines were the
    # ONLY failures, at `1 == 0`, with every finiteness and serial/threaded
    # agreement assertion above still passing.  Pinning the count re-pins the
    # platform difference the #651 contraction guard exists to absorb.
    #
    # What the guard promises -- and what is asserted instead -- is that
    # contraction is BOUNDED and FREE.  Bounded: every contraction is one extra
    # objective evaluation on the same arm, so the count is strictly below the
    # arm's evaluation count, which a miscounted or double-counted diagnostic
    # would break.  Free: an arm that contracts still ACCEPTS a finite endpoint
    # instead of failing, which is precisely what regresses if the guard is
    # removed and an unevaluable trial once again abandons the arm.  Measured
    # locally 2026-09-06 (Julia 1.10.12, aarch64 macOS): all six arms report
    # 0 contractions in 5-6 evaluations, `accepted = true`, `reason = :accepted`.
    for diag in vcat(serial.endpoint_diagnostics, one.endpoint_diagnostics)
        for arm in (diag.lower, diag.upper)
            @test arm.contractions >= 0
            @test arm.contractions < arm.evaluations
            @test arm.accepted && !arm.endpoint_failed && arm.reason === :accepted
        end
    end
end

# The endpoint search must not treat one unevaluable TRIAL as an unresolvable
# ENDPOINT (#651).  These cases are synthetic on purpose: the live failure is a
# last-bit platform difference that does not reproduce on every architecture, so
# the regression is pinned on the search logic itself, which is what changed.
@testset "unevaluable profile trials contract instead of failing the arm" begin
    # h(t) = t^2 - 1 crosses zero at t = 1.  The callback refuses to evaluate a
    # BAND rather than a single point, so merely retrying the same trial cannot
    # escape it; only contraction toward the feasible floor can.  Both the
    # Wald-seeded expansion trial and the first guarded-Newton trial land inside
    # the band, so this exercises both contraction sites.
    banded(t) = (1.1 <= t <= 1.6) ? (NaN, NaN, false) : (t^2 - 1.0, 2.0 * t, true)

    contracted = DRModels._ls_profile_root_result(banded, 0.0; dir=1.0, init=2.0)
    @test contracted.accepted
    @test !contracted.endpoint_failed
    @test !contracted.unbounded
    @test contracted.reason == :accepted
    @test isfinite(contracted.value)
    @test isapprox(contracted.value, 1.0; atol=1e-6, rtol=0)
    @test abs(contracted.residual) < 1e-7
    # The rescue came from the new code path, not from a lucky trial sequence.
    @test contracted.contractions > 0

    # Counterfactual: with the contraction budget removed the identical callback
    # abandons the arm and reports the signed infinity this leaf exists to stop.
    abandoned = DRModels._ls_profile_root_result(banded, 0.0; dir=1.0, init=2.0,
                                            maxcontract=0)
    @test !abandoned.accepted
    @test abandoned.endpoint_failed
    @test abandoned.value == Inf
    @test abandoned.contractions == 0

    # Contraction only POSTPONES failure: an arm that cannot be evaluated
    # anywhere above the fitted optimum still fails closed, and still reports a
    # failed signed-infinity endpoint rather than inventing a finite bound.
    # `evalh` is called with the PARAMETER COORDINATE `x0 + dir * t`, not with
    # the displacement, so refusing every coordinate away from the fitted value
    # covers both directions.
    unreachable(v) = v == 0.0 ? (-1.0, 0.0, true) : (NaN, NaN, false)
    hopeless = DRModels._ls_profile_root_result(unreachable, 0.0; dir=1.0, init=1.0)
    @test !hopeless.accepted
    @test hopeless.endpoint_failed
    @test !hopeless.unbounded
    @test hopeless.value == Inf
    @test hopeless.contractions > 0

    # A negative direction is guarded the same way, and reports -Inf.
    lower_side = DRModels._ls_profile_root_result(unreachable, 0.0; dir=-1.0, init=1.0)
    @test lower_side.endpoint_failed
    @test lower_side.value == -Inf

    # An expansion that had to contract cannot then certify UNBOUNDEDNESS: part
    # of the range was never evaluated, so "no crossing in the searched range"
    # was not established.  That is reported as a failed endpoint, not as an
    # unbounded interval, so `confint` refuses it instead of returning Inf.
    never_crosses(t) = (1.0 <= t <= 1.05) ? (NaN, NaN, false) : (-1.0, 0.0, true)
    incomplete = DRModels._ls_profile_root_result(never_crosses, 0.0; dir=1.0, init=1.0)
    @test !incomplete.unbounded
    @test incomplete.endpoint_failed
    @test incomplete.reason == :infeasible_region
    @test incomplete.contractions > 0

    # A search that never had to contract keeps the documented `:no_crossing`
    # verdict, so the fail-closed branch above cannot mask a genuine one.
    flat(t) = (-1.0, 0.0, true)
    nocross = DRModels._ls_profile_root_result(flat, 0.0; dir=1.0, init=1.0)
    @test nocross.unbounded
    @test !nocross.endpoint_failed
    @test nocross.reason == :no_crossing
    @test nocross.contractions == 0

    # The search stays bounded: contraction is capped per trial, so the total
    # evaluation count cannot exceed (maxexpand + maxnewton + 1) * (1 + maxcontract).
    @test hopeless.evaluations <= (40 + 30 + 1) * (1 + 8)
    @test contracted.evaluations <= (40 + 30 + 1) * (1 + 8)
end
