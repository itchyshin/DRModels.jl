#!/usr/bin/env julia
# Bounded correctness pilot, not a performance benchmark or full G4 verdict.
using DRModels, Test, LinearAlgebra, TOML
BLAS.set_num_threads(1)
@assert Threads.nthreads() == 4 "pilot requires four Julia threads"
@assert BLAS.get_num_threads() == 1
@assert realpath(pathof(DRModels)) == realpath(joinpath(@__DIR__, "..", "src", "DRModels.jl"))
println("RUNTIME julia=", VERSION, " threads=", Threads.nthreads(),
        " blas=", BLAS.get_num_threads(), " source=", pathof(DRModels))
include(joinpath(@__DIR__, "..", "test", "test_joint_missing_finite_factor_coding.jl"))
include(joinpath(@__DIR__, "..", "test", "test_joint_missing_finite_frontend.jl"))

function finite_parallel_pilot()
    cases = [_finite_frontend_data(_finite_frontend_reference[kind])
             for kind in ("ordinal", "categorical")]
    original = deepcopy(cases)
    form = bf(@formula(y ~ z + mi(x)), @formula(sigma ~ 1))
    control = miss_control(response = "include", predictor = "model")
    function fit_case(index)
        data, levels = cases[index]
        family = index == 1 ? CumulativeLogit() : CategoricalLogit()
        specification = (x = impute_model(@formula(x ~ z); family, levels),)
        drm(form, Gaussian(); data, impute = specification, missing = control, g_tol = 1e-8)
    end
    snapshot(fit) = deepcopy((converged = is_converged(fit), coefficients = coef(fit),
        covariance = vcov(fit), likelihood = loglik(fit), fitted = fitted(fit),
        summary = joint_missing_summary(fit), imputation = imputed(fit; rows = :all)))
    serial = [fit_case(i) for i in 1:2]
    # Freeze every output before concurrency, so corrupting a serial fit's shared
    # storage cannot also move the comparison target.
    baseline = snapshot.(serial)
    results = Vector{Any}(undef, 8)
    thread_ids = zeros(Int, 8)
    Threads.@threads :static for i in 1:8
        thread_ids[i] = Threads.threadid()
        results[i] = fit_case(mod1(i, 2))
    end
    observed = snapshot.(results)
    if get(ENV, "DRM_FINITE_PILOT_DAMAGE", "") == "coefficient"
        observed[1].coefficients[1] += 0.01
        println("FINITE_PARALLEL_DAMAGE_INJECTED coefficient")
    end
    @testset "finite fits in separate thread workspaces" begin
        @test length(unique(thread_ids)) == 4
        @test isequal(cases, original)
        @test isequal(snapshot.(serial), baseline)
        for (i, output) in enumerate(observed)
            reference = baseline[mod1(i, 2)]
            @test reference.converged
            @test output.converged
            @test maximum(abs, output.coefficients - reference.coefficients) <= 1e-10
            @test maximum(abs, output.covariance - reference.covariance) <= 1e-10
            @test abs(output.likelihood - reference.likelihood) <= 1e-10
            @test maximum(abs, output.fitted - reference.fitted) <= 1e-10
            @test output.summary.optimizer_status == reference.summary.optimizer_status
            @test output.summary.covariance_status == reference.summary.covariance_status
            @test maximum(abs, output.summary.conditional_probabilities - reference.summary.conditional_probabilities) <= 1e-10
            @test isequal(output.imputation, reference.imputation)
        end
    end
    println("FINITE_PARALLEL_THREADS ", join(sort(unique(thread_ids)), ","))
    println("FINITE_PARALLEL_PILOT_PASS cases=2 serial=2 concurrent=8")
end
finite_parallel_pilot()
