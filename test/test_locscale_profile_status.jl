# Failure disclosure for canonical location-scale profile roots. These controls
# are deliberately pure: the root finder must not turn an iteration limit or a
# failed trial into a plausible finite confidence-limit coordinate.
using DRModels
using Test, SparseArrays, LinearAlgebra, Random
import Distributions

function _ls_profile_status_smoke_fit()
    Random.seed!(20_260_831)
    G, m = 4, 8
    n = G * m
    species = repeat(1:G, inner=m)
    x = repeat(range(-1.0, 1.0; length=m), G)
    eta = 0.70 .+ 0.55 .* x .+ (0.16 .* randn(G))[species]
    psi = 1.05 .+ (0.10 .* randn(G))[species]
    y = [begin
        shape = exp(psi[i]); mu = exp(eta[i])
        Float64(rand(Distributions.Gamma(shape, mu / shape)))
    end for i in 1:n]
    return drm(
        bf(@formula(y ~ x + (1 | status_smoke | species)),
           @formula(sigma ~ 1 + (1 | status_smoke | species))),
        Gamma(); data=(; y, x, species),
    )
end

@testset "location-scale profile endpoint status" begin
    @testset "legacy and structured CI helpers retain doc bindings" begin
        docs = Base.Docs.meta(DRModels)
        @test haskey(docs, Base.Docs.Binding(DRModels, :_ls_profile_ci))
        @test haskey(docs, Base.Docs.Binding(DRModels, :_ls_profile_ci_result))
    end

    quadratic(t) = (t^2 - 1.0, 2.0 * t, true)

    @testset "iteration limits are not certified endpoints" begin
        # Thirty Newton updates from 1e20 still leave t around 9e10. The old
        # scalar helper returned that unevaluated/non-root value as an endpoint.
        exhausted = DRModels._ls_profile_root_result(quadratic, 0.0; dir=1.0, init=1e20)
        @test !exhausted.accepted
        @test exhausted.endpoint_failed
        @test !exhausted.unbounded
        @test exhausted.reason == :max_iterations
        @test exhausted.value == Inf
        @test exhausted.candidate > 1e9
        @test exhausted.residual > 1e18

        # A forced zero-iteration refinement preserves the last *evaluated*
        # bracket candidate and its residual for diagnostics rather than calling
        # it a root.
        stuck(t) = (3.0, NaN, true)
        forced = DRModels._ls_profile_root_result(
            stuck, 0.0; dir=1.0, init=2.0, maxnewton=0,
        )
        @test !forced.accepted
        @test forced.endpoint_failed
        @test !forced.unbounded
        @test forced.reason == :max_iterations
        @test forced.value == Inf
        @test forced.candidate == 2.0
        @test forced.residual == 3.0
    end

    @testset "valid roots and no-crossing remain distinct" begin
        accepted = DRModels._ls_profile_root_result(quadratic, 0.0; dir=1.0, init=2.0)
        @test accepted.accepted
        @test !accepted.endpoint_failed
        @test !accepted.unbounded
        @test accepted.reason == :accepted
        @test abs(accepted.residual) < 1e-7
        @test DRModels._ls_profile_root(quadratic, 0.0; dir=1.0, init=2.0) == accepted.value

        # Diagnostics expose the evaluated parameter coordinate, rather than the
        # internal positive displacement from the fitted value.
        centered(x) = ((x - 3.0)^2 - 1.0, 2.0 * (x - 3.0), true)
        shifted = DRModels._ls_profile_root_result(centered, 3.0; dir=-1.0, init=2.0)
        @test shifted.accepted
        @test isapprox(shifted.value, 2.0; atol=1e-7, rtol=0)
        @test shifted.candidate == shifted.value
        @test abs(shifted.residual) < 1e-7

        # A callback may provide no usable slope; guarded bisection remains a
        # valid route to a certified evaluated root.
        no_slope(t) = (t^2 - 1.0, NaN, true)
        bisection = DRModels._ls_profile_root_result(no_slope, 0.0; dir=1.0, init=2.0)
        @test bisection.accepted
        @test !bisection.endpoint_failed

        flat(t) = (-1.0, 0.0, true)
        nocross = DRModels._ls_profile_root_result(flat, 0.0; dir=1.0, init=1.0)
        @test !nocross.accepted
        @test !nocross.endpoint_failed
        @test nocross.unbounded
        @test nocross.reason == :no_crossing
        @test nocross.value == Inf

        invalid_budget = DRModels._ls_profile_root_result(flat, 0.0; dir=1.0, init=1.0,
                                                      maxexpand=0)
        @test invalid_budget.endpoint_failed
        @test !invalid_budget.unbounded
        @test invalid_budget.reason == :invalid_search_budget

        failed_trial(t) = (NaN, NaN, false)
        failed = DRModels._ls_profile_root_result(failed_trial, 0.0; dir=1.0, init=1.0)
        @test !failed.accepted
        @test failed.endpoint_failed
        @test !failed.unbounded
        @test failed.reason == :evaluation_failed
        @test failed.reason != :max_iterations

        # The first bracket point is valid; every later refinement trial fails.
        # The refusal covers the whole open bracket rather than a single point,
        # because an isolated bad trial is now contracted away (#651) and would
        # no longer leave the arm unresolved.
        refinement_failure(t) = (0.0 < t < 2.0) ? (NaN, NaN, false) : (t^2 - 1.0, NaN, true)
        refined = DRModels._ls_profile_root_result(refinement_failure, 0.0; dir=1.0, init=2.0)
        @test refined.endpoint_failed
        @test !refined.unbounded
        @test refined.reason == :evaluation_failed
        @test refined.root_iterations == 1
        @test refined.bracket_expansions == 0
        # One bracket evaluation, one refinement trial, then the exhausted
        # contraction budget -- each halving toward the feasible floor.
        @test refined.contractions == 8
        @test refined.evaluations == 2 + refined.contractions

        interrupted(t) = throw(InterruptException())
        @test_throws InterruptException DRModels._ls_profile_root_result(
            interrupted, 0.0; dir=1.0, init=1.0,
        )

        callback_error(t) = throw(DomainError(t, "test evaluation failure"))
        ordinary_error = DRModels._ls_profile_root_result(callback_error, 0.0; dir=1.0, init=1.0)
        @test ordinary_error.endpoint_failed
        @test !ordinary_error.unbounded
        @test ordinary_error.reason == :exception

        zero_gap(t) = (0.0, 0.0, true)
        nonfinite_init = DRModels._ls_profile_root_result(zero_gap, 0.0; dir=1.0, init=Inf)
        @test nonfinite_init.endpoint_failed
        @test !nonfinite_init.accepted
        @test !nonfinite_init.unbounded
        @test nonfinite_init.reason == :nonfinite_initialization
        nonfinite_origin = DRModels._ls_profile_root_result(zero_gap, Inf; dir=1.0, init=1.0)
        @test nonfinite_origin.endpoint_failed
        @test !nonfinite_origin.accepted
        @test nonfinite_origin.reason == :nonfinite_initialization
        overflow = DRModels._ls_profile_root_result(zero_gap, floatmax(Float64);
                                                dir=1.0, init=floatmax(Float64))
        @test overflow.endpoint_failed
        @test !overflow.accepted
        @test overflow.reason == :nonfinite_candidate
    end

    @testset "absolute-NLL cancellation cannot certify a root" begin
        shift = 1e16
        half = 1.920729410347062
        rounded(t) = begin
            reference = DRModels._profile_reference_difference(shift + t^2, shift)
            (gap=reference.difference - half, slope=2.0 * t,
             ok=reference.status === :accepted, cancellation=reference.cancellation)
        end
        reference = DRModels._profile_reference_difference(shift + 1.6^2, shift)
        @test reference.status == :accepted
        @test reference.cancellation > 0
        cancellation = DRModels._ls_profile_root_result(
            rounded, 0.0; dir=1.0, init=1.6, cancellation=reference.cancellation,
        )
        @test cancellation.endpoint_failed
        @test !cancellation.accepted
        @test !cancellation.unbounded
        @test cancellation.reason == :insufficient_precision

        # The CI path must compare the two represented NLL values before
        # subtracting the LR half-threshold, rather than subtracting huge NLLs
        # inside the callback.
        difference = DRModels._profile_reference_difference(shift + 1.6^2, shift)
        @test difference.difference - half != 0.0
    end

    @testset "boundary log-Cholesky diagonal is exempt from the gradient check" begin
        # owner decision 15 (a): a log-diagonal below _LS_PROFILE_BOUNDARY_LOGCHOL
        # is flat, so only that coordinate is dropped from the 1e-7 stationarity
        # test; the exemption is reported, and other coordinates stay strict.
        f = u -> sum(abs2, u)
        gb!(g, u) = (g .= [0.0, 5e-7]; g)             # gradient only on coordinate 2
        args = (f, gb!, [0.0, -9.0], true)
        strict = DRModels._ls_profile_candidate_status(args...)
        @test !strict.accepted && strict.reason == :not_stationary
        ex = DRModels._ls_profile_candidate_status(args...; logchol_diag = [2])
        @test ex.accepted && ex.reason == :accepted_boundary_exempt
        @test ex.gradient_maxabs <= 1e-7
        # Not on the boundary (value above the cutoff): no exemption.
        off = DRModels._ls_profile_candidate_status(f, gb!, [0.0, -3.0], true; logchol_diag = [2])
        @test !off.accepted && off.reason == :not_stationary
        # A non-exempt coordinate above tolerance still rejects.
        gc!(g, u) = (g .= [5e-7, 5e-7]; g)
        other = DRModels._ls_profile_candidate_status(f, gc!, [0.0, -9.0], true; logchol_diag = [2])
        @test !other.accepted && other.reason == :not_stationary
        # L21 of the same row is exempt only when log L22 is on the boundary.
        gl21!(g, u) = (g .= [0.0, 5e-7, 0.0]; g)      # coords: (b, L21, log L22)
        l21 = DRModels._ls_profile_candidate_status(f, gl21!, [0.0, 0.0, -9.0], true;
                                                    logchol_diag = [3], l21_pos = 2, l22_pos = 3)
        @test l21.accepted && l21.reason == :accepted_boundary_exempt
        l21off = DRModels._ls_profile_candidate_status(f, gl21!, [0.0, 0.0, -3.0], true;
                                                       logchol_diag = [3], l21_pos = 2, l22_pos = 3)
        @test !l21off.accepted
        # Already stationary: reported as plain :accepted, not as an exemption.
        g0!(g, u) = (g .= 0.0; g)
        plain = DRModels._ls_profile_candidate_status(f, g0!, [0.0, -9.0], true; logchol_diag = [2])
        @test plain.accepted && plain.reason == :accepted
    end

    @testset "finite exhausted nuisance solution is rejected" begin
        # Optim's termination flag alone is insufficient: the profiler checks the
        # same 1e-7 free-gradient target on a fresh candidate evaluation.
        stationary_check = DRModels._ls_profile_candidate_status(
            u -> sum(abs2, u),
            (g, u) -> (g .= 1.0; g),
            [0.0],
            true,
        )
        @test !stationary_check.accepted
        @test stationary_check.converged
        @test stationary_check.reason == :not_stationary
        @test stationary_check.gradient_maxabs == 1.0

        kind = Val(:gamma)
        y = [1.0, 1.2, 0.9, 1.1]
        Xmu = [ones(4) [-1.0, -0.3, 0.4, 1.0]]
        Xsigma = ones(4, 1)
        gidx = [1, 1, 2, 2]
        Q = sparse(1.0I, 2, 2)
        # [beta_mu(2), beta_sigma(1), log L11, L21, log L22]
        theta = [0.0, 0.1, 0.0, 0.0, 0.0, 0.0]
        result = DRModels._ls_profile_nll_result(
            kind, y, Xmu, Xsigma, gidx, 2, Q, theta, 1, 0.1;
            iterations=0,
        )
        @test isfinite(result.value)
        @test !result.accepted
        @test result.reason == :not_converged
        _, _, accepted = DRModels._ls_profile_nll(
            kind, y, Xmu, Xsigma, gidx, 2, Q, theta, 1, 0.1;
            iterations=0,
        )
        @test !accepted
    end

    @testset "whitened canonical retry is reported and raw route is untouched" begin
        fit = _ls_profile_status_smoke_fit()
        obj = fit.nll::DRModels.LocScaleObjective
        base = size(obj.Xμ, 2) + size(obj.Xψ, 2)
        perm = vcat(collect(1:base), [base + 1, base + 3, base + 2])
        theta = fit.theta[perm]
        idx = 2
        value = theta[idx] + 0.1
        damaged_warm_start = fill(10.0, length(theta) - 1)

        recovered = DRModels._ls_profile_nll_result(
            obj.kind, obj.y, obj.Xμ, obj.Xψ, obj.gidx, obj.G, obj.Q,
            theta, idx, value;
            x0=damaged_warm_start,
            whitened=true,
        )
        @test recovered.accepted
        @test recovered.reason == :accepted
        @test recovered.fallback
        @test recovered.converged
        @test recovered.gradient_maxabs <= 1e-7

        # With a deliberately short budget, the first solve from the fitted
        # coordinates stops unsuccessfully on this locked fixture. Continuing
        # once from that failed endpoint earns strict convergence without
        # relaxing the 1e-7 exact-gradient gate (the GitHub 1.10 regression).
        z = Distributions.quantile(Distributions.Normal(), 0.975)
        endpoint_value = theta[idx] - max(z * stderror(fit)[2], 1e-3)
        endpoint_recovered = DRModels._ls_profile_nll_result(
            obj.kind, obj.y, obj.Xμ, obj.Xψ, obj.gidx, obj.G, obj.Q,
            theta, idx, endpoint_value;
            whitened=true,
            iterations=10,
        )
        @test endpoint_recovered.accepted
        @test endpoint_recovered.reason == :accepted
        @test endpoint_recovered.fallback
        @test endpoint_recovered.converged
        @test endpoint_recovered.gradient_maxabs <= 1e-7

        raw = DRModels._ls_profile_nll_result(
            obj.kind, obj.y, obj.Xμ, obj.Xψ, obj.gidx, obj.G, obj.Q,
            theta, idx, value;
            x0=damaged_warm_start,
            whitened=false,
            iterations=0,
        )
        @test !raw.accepted
        @test !raw.fallback
    end

    @testset "public canonical result propagates failed-arm diagnostics" begin
        fit = _ls_profile_status_smoke_fit()
        result = profile_result(fit; parm=:mu => "x")
        @info "canonical location-scale profile status fixture" attempted=result.attempted failed=result.failed lower_reason=only(result.endpoint_diagnostics).lower.reason upper_reason=only(result.endpoint_diagnostics).upper.reason
        @test length(result.ci) == length(result.stats) ==
              length(result.endpoint_diagnostics) == 1
        expected_failed = count(diag -> diag.lower.endpoint_failed ||
                                          diag.upper.endpoint_failed,
                                result.endpoint_diagnostics)
        @test result.failed == expected_failed
        @test count(stat -> stat.lower_endpoint_failed || stat.upper_endpoint_failed,
                    result.stats) == expected_failed
        stat = only(result.stats)
        diag = only(result.endpoint_diagnostics)
        @test (stat.param, stat.coef) == (diag.param, diag.coef)
        @test stat.evaluations == diag.lower.evaluations + diag.upper.evaluations
        @test stat.gradient_evaluations == diag.lower.gradient_evaluations +
                                          diag.upper.gradient_evaluations
        @test stat.bracket_expansions == diag.lower.bracket_expansions +
                                          diag.upper.bracket_expansions
        @test stat.root_iterations == diag.lower.root_iterations +
                                     diag.upper.root_iterations
        @test stat.lower_endpoint_failed == diag.lower.endpoint_failed
        @test stat.upper_endpoint_failed == diag.upper.endpoint_failed
        @test stat.lower_nuisance_reason == (diag.lower.nuisance === nothing ?
                                             :not_checked : diag.lower.nuisance.reason)
        @test stat.upper_nuisance_reason == (diag.upper.nuisance === nothing ?
                                             :not_checked : diag.upper.nuisance.reason)
        @test any(arm.endpoint_failed for arm in (diag.lower, diag.upper)) ==
              (expected_failed > 0)
        if expected_failed > 0
            # #631: a failed endpoint arm is REFUSED at the user-facing seam
            # rather than warned about and returned as a signed Inf.
            @test_throws ArgumentError confint(fit; method=:profile, parm=:mu => "x")
        else
            @test_logs begin
                confint(fit; method=:profile, parm=:mu => "x")
            end
        end
    end
end
