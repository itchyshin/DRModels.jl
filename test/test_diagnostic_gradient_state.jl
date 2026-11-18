using DRModels, Test, Logging, DelimitedFiles, ForwardDiff

function _diagnostic_fixture(dataset)
    raw, _ = readdlm(joinpath(@__DIR__, "fixtures", "corr_slope_retry", "data.csv"), ','; header = true)
    rows = String.(raw[:, 1]) .== dataset
    return (; y = Float64.(raw[rows, 2]), x1 = Float64.(raw[rows, 3]),
            g = String.(raw[rows, 4]))
end

@testset "correlated-slope diagnostics preserve their objective cache" begin
    f = bf(@formula(y ~ x1 + (1 + x1 | g)), @formula(sigma ~ 1))
    for (dataset, family) in (("gamma3", Gamma()), ("student8", Student()))
        fit = with_logger(NullLogger()) do
            drm(f, family; data = _diagnostic_fixture(dataset))
        end
        @test fit.nll isa DRModels._AGHQDiagnosticObjective
        cache_before = copy(fit.nll.bcache)
        theta_before = copy(fit.theta)
        resets_before = DRModels._AGHQ_MODE_RESETS[]
        first = with_logger(NullLogger()) do; check_drm(fit); end
        displayed = sprint(io -> DRModels._final_gradient_note(io, fit))
        bridge = bridge_diagnostics(fit)
        second = with_logger(NullLogger()) do; check_drm(fit); end
        @test isfinite(first.max_abs_grad)
        @test first.max_abs_grad == second.max_abs_grad == bridge.max_abs_grad
        @test bridge.route == "_AGHQDiagnosticObjective"
        found = match(r"final gradient: max\|grad\| = ([0-9.e+-]+)", displayed)
        @test found !== nothing
        @test parse(Float64, found.captures[1]) ≈ first.max_abs_grad rtol = 1e-3
        @test fit.nll.bcache == cache_before
        @test fit.theta == theta_before
        @test DRModels._AGHQ_MODE_RESETS[] == resets_before
        copied_objective = deepcopy(fit.nll)
        isolated_value = fit.nll.diagnostic(fit.theta, copy(cache_before))
        @test copied_objective(fit.theta) ≈ isolated_value atol = 1e-10 rtol = 1e-10
        copied_gradient = ForwardDiff.gradient(deepcopy(fit.nll), fit.theta)
        isolated_gradient = ForwardDiff.gradient(θ -> fit.nll.diagnostic(θ, copy(cache_before)), fit.theta)
        @test copied_gradient ≈ isolated_gradient atol = 1e-10 rtol = 1e-10
        @test fit.nll.bcache == cache_before
        if !fit.converged
            report = sprint(show, MIME"text/plain"(), fit)
            @test occursin("final gradient:", report)
            @test fit.nll.bcache == cache_before
        end
        @test length(DRModels.gradient(fit)) == length(fit.theta)
        @test size(DRModels.hessian(fit)) == (length(fit.theta), length(fit.theta))
    end
end

@testset "Student random-intercept route remains callable" begin
    f = bf(@formula(y ~ x1 + (1 | g)), @formula(sigma ~ 1))
    fit = with_logger(NullLogger()) do
        drm(f, Student(); data = _diagnostic_fixture("student8"))
    end
    @test !(fit.nll isa DRModels._AGHQDiagnosticObjective)
    @test isfinite(fit.nll(fit.theta))
end

@testset "diagnostic cache resets do not alter fitted-mode tracking" begin
    cache = ones(2, 1)
    counter = DRModels._AGHQ_MODE_RESETS
    original = counter[]
    added = false
    try
        DRModels._aghq_update_cache!(cache, 1, [NaN, 2.0]; count_resets = false)
        @test cache == zeros(2, 1)
        @test counter[] == original
        cache .= 1
        DRModels._aghq_update_cache!(cache, 1, [NaN, 2.0])
        added = true
        @test cache == zeros(2, 1)
        @test counter[] == original + 1
        DRModels._aghq_update_cache!(cache, 1, [0.2, -0.3])
        @test cache[:, 1] == [0.2, -0.3]
        @test counter[] == original + 1
    finally
        added && Threads.atomic_add!(counter, -1)
    end
end

@testset "diagnostic fallback uses one cache snapshot per stencil" begin
    data = _diagnostic_fixture("gamma3")
    f = bf(@formula(y ~ x1 + (1 + x1 | g)), @formula(sigma ~ 1))
    fit = with_logger(NullLogger()) do; drm(f, Gamma(); data = data); end
    diagnostic = function (θ, cache)
        eltype(θ) <: ForwardDiff.Dual && error("force finite differences")
        value = sum(abs2, θ) + cache[1, 1]
        cache[1, 1] += 1
        value
    end
    wrapper = DRModels._AGHQDiagnosticObjective(θ -> sum(abs2, θ), diagnostic, ones(1, 1))
    fallback_fit = DRModels._withnll(fit, wrapper)
    result = DRModels._check_max_abs_grad(fallback_fit)
    @test result.source == :finite
    @test result.magnitude ≈ maximum(abs, 2 .* fit.theta) rtol = 1e-6
    @test wrapper.bcache == ones(1, 1)
    stored = (g, θ) -> fill!(g, 0.125)
    stored_fit = DRModels._withnll(fit, wrapper, stored)
    result = DRModels._check_max_abs_grad(stored_fit)
    @test result.source == :stored
    @test result.magnitude == 0.125
    @test wrapper.bcache == ones(1, 1)
end
@testset "nonfinite AGHQ mode accounting" begin
    counter = DRModels._AGHQ_MODE_RESETS
    original = counter[]
    added = false
    rule = DRModels._AGHQRule(1, 3)
    ll = (i, η) -> zero(η) + NaN
    members = [[1]]; eta = [0.0]; Z = ones(1, 1); L = ones(1, 1)
    try
        cache = fill(NaN, 1, 1)
        result = DRModels._aghq_marginal_loglik(ll, members, eta, Z, L, rule, cache; count_resets=false)
        @test result == -Inf
        @test cache == zeros(1, 1)
        @test counter[] == original
        cache .= NaN
        result = DRModels._aghq_marginal_loglik(ll, members, eta, Z, L, rule, cache)
        added = true
        @test result == -Inf
        @test cache == zeros(1, 1)
        @test counter[] == original + 1
    finally
        added && Threads.atomic_add!(counter, -1)
    end
end
