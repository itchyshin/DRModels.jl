using DRModels, Test, LinearAlgebra

# Analytic profile with known nuisance optimum u = 2v and profile f(v) = v^2 / 2.
_pn_nll(θ) = θ[1]^2 / 2 + (θ[2] - 2θ[1])^2 / 2
function _pn_grad!(g, θ)
    g[1] = θ[1] - 2 * (θ[2] - 2θ[1])
    g[2] = θ[2] - 2θ[1]
    return g
end

function _pn_fit(nll, θ; nllgrad=_pn_grad!)
    p = length(θ)
    base = DrmFit(
        Gaussian(), [:mu => (1:p)], [:mu => ["θ$i" for i in 1:p]], copy(θ),
        Matrix{Float64}(I, p, p), -nll(θ), 1, true,
        Dict{Symbol,Vector{Float64}}(), Dict{Symbol,Vector{Float64}}(),
        Dict{Symbol,Vector{Float64}}(),
    )
    return nllgrad === nothing ? DRModels._withnll(base, nll) : DRModels._withnll(base, nll, nllgrad)
end

@testset "generic profile nuisance acceptance and status" begin
    θ = [0.0, 0.0]

    @testset "accepted solve is reevaluated at the reported minimizer" begin
        result = DRModels._profile_nuisance_result(
            _pn_nll, θ, 1, 1.5, [0.0]; autodiff = :stored, nllgrad = _pn_grad!,
        )
        @test result.accepted
        @test result.method == :lbfgs_stored
        @test !result.fallback
        @test result.reason == :accepted
        @test result.minimizer ≈ [3.0] atol = 1e-8
        @test result.value ≈ 1.5^2 / 2 atol = 1e-10
        @test result.value ≈ _pn_nll([1.5, only(result.minimizer)]) atol = 1e-12
    end

    @testset "one-parameter profile is a valid direct evaluation" begin
        f1(θ) = θ[1]^2 / 2
        direct = DRModels._profile_nuisance_result(f1, [0.0], 1, 1.25, Float64[])
        @test direct.accepted
        @test direct.method == :direct
        @test !direct.fallback
        @test direct.reason == :accepted
        @test isempty(direct.minimizer)
        @test direct.value == 1.25^2 / 2

        nonfinite(θ) = θ[1] > 0 ? Inf : θ[1]^2
        rejected = DRModels._profile_nuisance_result(nonfinite, [0.0], 1, 1.0, Float64[])
        @test !rejected.accepted
        @test rejected.method == :direct
        @test rejected.reason == :nonfinite_objective

        interrupted(θ) = throw(InterruptException())
        @test_throws InterruptException DRModels._profile_nuisance_result(
            interrupted, [0.0], 1, 0.0, Float64[],
        )
        @test_throws InterruptException DRModels._profile_autodiff_mode(
            interrupted, nothing, [0.0],
        )
    end

    @testset "Nelder-Mead fallback is accepted only on successful termination" begin
        recovered = DRModels._profile_nuisance_result(
            _pn_nll, θ, 1, 1.0, [0.0]; autodiff = :finite,
            primary_attempt = (obj, u0, method, autodiff, grad!) ->
                (value=NaN, minimizer=copy(u0), accepted=false, method=method,
                 fallback=false, reason=:exception),
            fallback_iterations = 80,
        )
        @test recovered.accepted
        @test recovered.method == :nelder_mead
        @test recovered.fallback
        @test recovered.reason == :accepted
        @test recovered.value ≈ 0.5 atol = 1e-8

        exhausted = DRModels._profile_nuisance_result(
            _pn_nll, θ, 1, 1.0, [0.0]; autodiff = :finite,
            primary_attempt = (obj, u0, method, autodiff, grad!) ->
                (value=NaN, minimizer=copy(u0), accepted=false, method=method,
                 fallback=false, reason=:exception),
            fallback_iterations = 0,
        )
        @test !exhausted.accepted
        @test exhausted.fallback
        @test exhausted.reason == :fallback_not_converged
        @test isfinite(exhausted.value) # finite is insufficient without Optim convergence
    end

    @testset "failed nuisance arm is not unbounded or warm-started" begin
        failing(θ) = θ[1] > 0.25 ? throw(DomainError(θ[1], "forced nuisance failure")) : _pn_nll(θ)
        endpoint, arm = DRModels._profile_endpoint_result(
            failing, nothing, θ, 1, 0.0, 0.5, 1.0, +1, [0.0], :finite,
        )
        @test endpoint == Inf
        @test arm.endpoint_failed
        @test !arm.unbounded
        @test arm.nuisance_reason == :exception
        @test arm.nuisance_method == :nelder_mead

        # The expansion point t = 1 is valid and brackets the target; the
        # failure occurs at the first bisection refinement t = 0.5.
        refinement_failing(θ) = 0.4 < θ[1] < 0.6 ?
            throw(DomainError(θ[1], "forced refinement failure")) : _pn_nll(θ)
        endpoint2, arm2 = DRModels._profile_endpoint_result(
            refinement_failing, nothing, θ, 1, 0.0, 0.3, 1.0, +1, [0.0], :finite,
        )
        @test endpoint2 == Inf
        @test arm2.endpoint_failed && !arm2.unbounded
        @test arm2.root_iterations == 1
        @test arm2.nuisance_reason == :exception
    end

    @testset "endpoint is the coordinate whose reevaluated profile hits target" begin
        endpoint, arm = DRModels._profile_endpoint_result(
            _pn_nll, _pn_grad!, θ, 1, 0.0, 2.0, 1.0, +1, [0.0], :stored,
        )
        @test !arm.endpoint_failed
        @test endpoint ≈ 2.0 atol = 1e-6
        @test abs(endpoint^2 / 2 - 2.0) < 1e-7

        _, bad_reference = DRModels._profile_endpoint_result(
            _pn_nll, _pn_grad!, θ, 1, 2.0, 0.5, 1.0, +1, [0.0], :stored,
        )
        @test bad_reference.endpoint_failed
        @test bad_reference.nuisance_reason == :below_reference

        shifted(θ) = 1e16 + _pn_nll(θ)
        _, insufficient = DRModels._profile_endpoint_result(
            shifted, nothing, θ, 1, 1e16, 2.0, 1.0, +1, [0.0], :finite,
        )
        @test insufficient.endpoint_failed
        @test insufficient.nuisance_reason == :insufficient_precision

        # Finite-difference bisection reaches the bracket-collapse path. The
        # return value must be a coordinate actually evaluated by the profile.
        seen = Float64[]
        traced(θ) = (push!(seen, θ[1]); _pn_nll(θ))
        collapsed, collapsed_arm = DRModels._profile_endpoint_result(
            traced, nothing, θ, 1, 0.0, 2.0, 1.0, +1, [0.0], :finite,
        )
        @test !collapsed_arm.endpoint_failed
        @test any(==(collapsed), seen)
        @test abs(collapsed^2 / 2 - 2.0) <= max(1e-9, 1e-4 * 2.0)
    end

    @testset "flat direct profile is validly unbounded, not a solver failure" begin
        flat(θ) = 0.0
        flatfit = _pn_fit(flat, [0.0]; nllgrad=nothing)
        result = profile_result(flatfit; parm=:mu, threads=false)
        @test result.failed == 0
        @test result.threaded == false
        @test result.worker_threads == 1
        @test only(result.ci).lower == -Inf
        @test only(result.ci).upper == Inf
        status = only(result.stats)
        @test status.lower_unbounded && status.upper_unbounded
        @test !status.lower_endpoint_failed && !status.upper_endpoint_failed
        @test status.lower_nuisance_method == :direct
        @test status.upper_nuisance_reason == :accepted
        if Threads.nthreads() > 1
            threaded = profile_result(flatfit; parm=:mu, threads=true)
            @test threaded.threaded
            @test threaded.worker_threads == 2
            @test threaded.failed == 0
            @test only(threaded.stats).lower_unbounded
        end
    end

    @testset "failed profile rows are refused by confint, not returned" begin
        failed_direct(θ) = θ[1] > 0.25 ? Inf : θ[1]^2 / 2
        failedfit = _pn_fit(failed_direct, [0.0]; nllgrad=nothing)
        result = profile_result(failedfit; parm=:mu)
        @test result.failed == 1
        @test only(result.ci).upper == Inf
        @test only(result.stats).upper_endpoint_failed
        @test only(result.stats).upper_nuisance_reason == :nonfinite_objective
        # #631: `profile_result` keeps the auditable +/-Inf convention above;
        # `confint` must not pass that Inf off as a confidence limit.
        @test_throws ArgumentError confint(failedfit; method=:profile, parm=:mu)
    end

    @testset "diagnostics and visual data refuse invalid profiled values" begin
        invalid_ref = _pn_fit(_pn_nll, [2.0, 4.0]) # profile values near zero beat this claimed reference
        @test_throws ArgumentError profile_curve(invalid_ref, 1; npoints = 5, span = 3)

        exploding(θ) = abs(θ[1]) > 0.2 ? throw(DomainError(θ[1], "bad profile grid")) : _pn_nll(θ)
        bad_curve = _pn_fit(exploding, [0.0, 0.0]; nllgrad=nothing)
        @test_throws ArgumentError profile_curve(bad_curve, 1; npoints = 5, span = 2)

        shifted_nll(θ) = 1e12 + _pn_nll(θ)
        shifted_bad_ref = _pn_fit(shifted_nll, [2.0, 4.0]; nllgrad=nothing)
        @test_throws ArgumentError profile_curve(shifted_bad_ref, 1; npoints = 5, span = 3)
        shifted_good = _pn_fit(shifted_nll, [0.0, 0.0]; nllgrad=nothing)
        @test all(isfinite, profile_curve(shifted_good, 1; npoints = 3, span = 1).deviance)
        @test_throws ArgumentError DRModels._profile_plot_deviance(
            floatmax(Float64), -floatmax(Float64), "test", "overflow",
        )

        f2(θ) = θ[1]^2 / 2 + θ[2]^2 / 2
        direct_surface = _pn_fit(f2, [0.0, 0.0]; nllgrad=nothing)
        surface = parameter_surface(direct_surface, 1, 2; npoints = 3, span = 1)
        @test size(surface.z) == (3, 3)
        @test all(isfinite, surface.z)

        bad_surface = _pn_fit((θ -> θ[1] > 0.1 ? Inf : f2(θ)), [0.0, 0.0]; nllgrad=nothing)
        @test_throws ArgumentError parameter_surface(bad_surface, 1, 2; npoints = 3, span = 1)

        f3(θ) = sum(abs2, θ) / 2
        nuisance_surface = _pn_fit(f3, [0.0, 0.0, 0.0]; nllgrad=nothing)
        @test all(isfinite, parameter_surface(nuisance_surface, 1, 2; npoints = 3, span = 1).z)
        bad_nuisance_surface = _pn_fit((θ -> θ[1] > 0.1 ? Inf : f3(θ)), [0.0, 0.0, 0.0]; nllgrad=nothing)
        @test_throws ArgumentError parameter_surface(
            bad_nuisance_surface, 1, 2; npoints = 3, span = 1,
        )
    end
end
