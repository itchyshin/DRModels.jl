using DRModels, Test, Random, Optim

const _OS_RNG = MersenneTwister(20261018)
const _OS_N = 72
const _OS_X = randn(_OS_RNG, _OS_N)
const _OS_G = repeat(1:12, inner = 6)
const _OS_U = 0.35 .* randn(_OS_RNG, 12)
const _OS_Y = 1 .+ 0.4 .* _OS_X .+ _OS_U[_OS_G] .+
              exp.(-0.2 .+ 0.1 .* _OS_X) .* randn(_OS_RNG, _OS_N)
const _OS_DATA = (y = _OS_Y, x = _OS_X, g = _OS_G)
const _OS_FIXED = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x))
const _OS_RE = bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1 + x))

@testset "optimizer selection" begin
    @test DRMControl().optimizer === :default
    @test DRMControl(nothing, nothing, 1, true, 1, 0.5).optimizer === :default
    legacy_report = try
        DRMControlReport(DRMControl(), Float64[], 0, true, :computed, nothing)
    catch err
        err
    end
    @test legacy_report isa DRMControlReport
    if legacy_report isa DRMControlReport
        @test legacy_report.requested_optimizer === :default
        @test legacy_report.actual_optimizer === :unrecorded
        @test isempty(legacy_report.optimizer_runs)
        @test legacy_report.selected_optimizer_stage === :none
        @test legacy_report.native_controls.maxit === nothing
    end
    legacy_selected = DRMControlReport(DRMControl(optimizer = :bfgs), Float64[],
                                       0, true, :computed, nothing)
    @test legacy_selected.requested_optimizer === :bfgs
    @test legacy_selected.actual_optimizer === :unrecorded
    @test_throws ArgumentError DRMControl(optimizer = :nelder_mead)

    base = drm(_OS_FIXED, Gaussian(); data = _OS_DATA, control = DRMControl(se = false))
    default = drm(_OS_FIXED, Gaussian(); data = _OS_DATA,
                  control = DRMControl(optimizer = :default, se = false))
    @test coef(base) == coef(default)
    @test loglik(base) == loglik(default)
    @test control_info(default).requested_optimizer === :default
    @test control_info(default).actual_optimizer === :lbfgs

    common_start = coef(base)
    for method in (:lbfgs, :bfgs)
        supplied_start = copy(common_start)
        ctl = DRMControl(optimizer = method, start = supplied_start, se = false)
        fit = drm(_OS_FIXED, Gaussian(); data = _OS_DATA,
                  control = ctl)
        info = control_info(fit)
        @test info.requested_optimizer === method
        @test info.actual_optimizer === method
        @test length(info.optimizer_runs) == 1
        @test info.optimizer_runs[1].method === method
        @test info.optimizer_runs[1].start == Tuple(common_start)
        @test info.optimizer_runs[1].iterations isa Int
        @test info.optimizer_runs[1].objective ≈ -loglik(fit)
        @test info.optimizer_runs[1].gradient_inf >= 0
        @test isfinite(loglik(fit))
        @test maximum(abs, coef(fit) .- common_start) < 0.2
        supplied_start[1] += 100
        ctl.start[1] += 100
        @test info.native_controls.start == Tuple(common_start)
        @test info.control.start == common_start
    end

    # This exercises the method object passed to Optim, not just report text.
    for (name, typ) in ((:lbfgs, Optim.LBFGS), (:bfgs, Optim.BFGS))
        scope = DRModels._ControlScope(DRMControl(optimizer = name), nothing, _OS_FIXED,
                                      false, false, false, Float64[], 0, NamedTuple[], :none)
        methods = DataType[]
        run = (x, opts, method) -> begin
            push!(methods, typeof(method))
            Optim.optimize(v -> sum(abs2, v .- 1), x, method, opts; autodiff = :forward)
        end
        Base.task_local_storage(() -> DRModels._ctl_optimize(run, [0.0, 2.0]; g_tol = 1e-8),
                                :DRModels_drm_control_scope, scope)
        @test length(methods) == 1 && methods[1] <: typ
    end

    count_data = (y = Float64.(round.(Int, exp.(0.4 .+ 0.2 .* _OS_X))), x = _OS_X)
    count_formula = bf(@formula(y ~ 1 + x))
    positive_data = (y = exp.(_OS_Y), x = _OS_X)
    positive_formula = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
    for (f, family, data) in ((count_formula, Poisson(), count_data),
                              (positive_formula, Gamma(), positive_data))
        first_fit = drm(f, family; data = data, control = DRMControl(se = false))
        for method in (:lbfgs, :bfgs)
            fit = drm(f, family; data = data,
                      control = DRMControl(optimizer = method, start = coef(first_fit), se = false))
            @test control_info(fit).actual_optimizer === method
            @test isfinite(loglik(fit))
            @test abs(loglik(fit) - loglik(first_fit)) < 1e-5
        end
    end
    single_formula = drm(bf(@formula(y ~ 1 + x)), Gaussian(); data = _OS_DATA,
                         control = DRMControl(optimizer = :bfgs, se = false))
    @test control_info(single_formula).actual_optimizer === :bfgs

    # A bad parm and missing data make an unguarded inference call fail cheaply;
    # the selector-specific error must win before any nuisance or bootstrap refit.
    selected_gaussian = drm(_OS_FIXED, Gaussian(); data = _OS_DATA,
                            control = DRMControl(optimizer = :bfgs))
    selected_poisson = drm(count_formula, Poisson(); data = count_data,
                           control = DRMControl(optimizer = :lbfgs, se = false))
    @test length(confint(selected_gaussian; method = :wald)) == length(coef(selected_gaussian))
    function selector_refused(call)
        err = try
            call()
            nothing
        catch caught
            caught
        end
        @test err isa ArgumentError
        @test err isa ArgumentError && occursin("optimizer", sprint(showerror, err))
    end
    selector_refused(() -> profile_result(selected_gaussian; parm = :not_a_parameter))
    selector_refused(() -> confint(selected_gaussian; method = :profile,
                                  parm = :not_a_parameter))
    for fit in (selected_gaussian, selected_poisson)
        selector_refused(() -> bootstrap_result(fit; data = nothing, B = 1))
        selector_refused(() -> bootstrap_summary(fit; data = nothing, B = 1))
        selector_refused(() -> bootstrap_ci(fit; data = nothing, B = 1))
    end
    default_inference = drm(_OS_FIXED, Gaussian(); data = _OS_DATA,
                            control = DRMControl())
    @test profile_result(default_inference; parm = :mu).attempted == 2
    default_bootstrap = bootstrap_result(default_inference; data = _OS_DATA,
                                         B = 1, rng = MersenneTwister(20261019))
    @test default_bootstrap.attempted == default_bootstrap.used == 1

    re_default = drm(_OS_RE, Gaussian(); data = _OS_DATA)
    re_control_default = drm(_OS_RE, Gaussian(); data = _OS_DATA,
                             control = DRMControl(optimizer = :default))
    @test coef(re_default) == coef(re_control_default)
    @test loglik(re_default) == loglik(re_control_default)

    for method in (:lbfgs, :bfgs)
        re = drm(_OS_RE, Gaussian(); data = _OS_DATA,
                 control = DRMControl(optimizer = method, se = false))
        info = control_info(re)
        @test info.requested_optimizer === method
        @test info.actual_optimizer === method
        @test 1 <= length(info.optimizer_runs) <= 2
        @test all(run -> run.method === method, info.optimizer_runs)
        @test all(run -> run.stage in (:primary, :restart), info.optimizer_runs)
        @test info.selected_optimizer_stage in (:primary, :restart)
        selected = only(filter(run -> run.stage === info.selected_optimizer_stage,
                               info.optimizer_runs))
        @test selected.gradient_inf <= 1e-8
        @test selected.objective ≈ -loglik(re)
        @test isfinite(loglik(re))
    end

    # Flat boundary objective: a restart is required and the less extreme,
    # equally good representative wins the existing interior-tie rule.
    scope = DRModels._ControlScope(DRMControl(optimizer = :bfgs), nothing, _OS_RE,
                                  false, false, false, Float64[], 0, NamedTuple[], :none)
    boundary = Base.task_local_storage(:DRModels_drm_control_scope, scope) do
        DRModels._re_lbfgs_with_restart(v -> (v[1] - 1)^2,
                                       [1.0, -20.0], [1.0, -2.0], 1e-8, 2, 1.0)
    end
    @test length(scope.optimizer_runs) == 2
    @test scope.selected_optimizer_stage === :restart
    @test Optim.minimizer(boundary)[2] == -2.0

    @test_throws ArgumentError drm(_OS_RE, Gaussian(); data = _OS_DATA,
                                   control = DRMControl(optimizer = :bfgs, maxit = 100))
    @test_throws ArgumentError drm(_OS_RE, Gaussian(); data = _OS_DATA,
                                   control = DRMControl(optimizer = :bfgs, seed = 9))
    @test_throws ArgumentError drm(_OS_RE, Gaussian(); data = _OS_DATA, method = :REML,
                                   control = DRMControl(optimizer = :bfgs))
    @test_throws ArgumentError drm(_OS_FIXED, Gaussian(); data = _OS_DATA,
                                   algorithm = :gls, control = DRMControl(optimizer = :bfgs))
    @test_throws ArgumentError drm(_OS_RE, Poisson(); data = _OS_DATA,
                                   control = DRMControl(optimizer = :bfgs))
    sigma_re = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + (1 | g)))
    @test_throws ArgumentError drm(sigma_re, Gaussian(); data = _OS_DATA,
                                   control = DRMControl(optimizer = :bfgs))
    bivariate = bf(mu1 = @formula(y1 ~ 1 + x), mu2 = @formula(y2 ~ 1 + x))
    two_data = (y1 = _OS_Y, y2 = _OS_Y .+ 0.2 .* randn(MersenneTwister(3), _OS_N), x = _OS_X)
    bivariate_err = try
        drm(bivariate, Gaussian(); data = two_data,
            control = DRMControl(optimizer = :bfgs))
        nothing
    catch err
        err
    end
    @test bivariate_err isa ArgumentError
    @test occursin("optimizer", sprint(showerror, bivariate_err))
    bivariate_default = drm(bivariate, Gaussian(); data = two_data,
                            control = DRMControl(se = false))
    @test control_info(bivariate_default).actual_optimizer === :unrecorded
end

println("OPTIMIZER_SELECTION_PASS")
