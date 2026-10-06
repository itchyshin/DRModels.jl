using Test
using DRModels
using LinearAlgebra
using TOML

# Root owns the red/green executions for this file.  The frozen native fixture
# exercises all eight (observed y, observed x1, observed x2) masks.
const _two_frontend_ref = TOML.parsefile(joinpath(@__DIR__, "fixtures", "joint_missing_predictor", "two_gaussian_reference.toml"))

function _two_frontend_data(reference)
    missing_or_value(values, observed) = Union{Missing,Float64}[
        observed[i] ? Float64(values[i]) : missing for i in eachindex(values)
    ]
    return (; y = missing_or_value(reference["y"], reference["y_observed"]),
             x1 = missing_or_value(reference["x1"], reference["x1_observed"]),
             x2 = missing_or_value(reference["x2"], reference["x2_observed"]),
             z = Float64.(reference["z"]))
end

@testset "two Gaussian missing-predictor direct formula frontend" begin
    @test BLAS.get_num_threads() == 1
    @test isdefined(DRModels, :JointTwoDrmFit)

    reference = _two_frontend_ref
    data = _two_frontend_data(reference)
    n = length(data.y)
    states = Set(zip(reference["y_observed"], reference["x1_observed"], reference["x2_observed"]))
    @test states == Set((y, x1, x2) for y in (false, true) for x1 in (false, true) for x2 in (false, true))
    @test count(ismissing, data.y) == n - count(identity, reference["y_observed"])

    form = DRModels.bf(@formula(y ~ z + mi(x1) + mi(x2)), @formula(sigma ~ 1))
    ctl = DRModels.miss_control(response = "include", predictor = "model")
    # Entry order is intentionally the reverse of formula-marker order.
    impute = (x2 = @formula(x2 ~ z), x1 = @formula(x1 ~ z))

    fit = drm(form, Gaussian(); data = data, impute = impute, missing = ctl, g_tol = 1e-8)
    @test fit isa DRModels.JointTwoDrmFit
    @test fit.variables == (:x1, :x2)
    @test fit.prepared.prepared.predictor_variables == (:x1, :x2)
    @test fit.prepared.prepared.original_row == reference["original_row"]
    @test fit.prepared.prepared.observed_x[1] == BitVector(reference["x1_observed"])
    @test fit.prepared.prepared.observed_x[2] == BitVector(reference["x2_observed"])
    @test fit.prepared.fit.nobs == count(identity, reference["y_observed"])
    @test DRModels.nobs(fit) == count(identity, reference["y_observed"])
    @test DRModels.is_converged(fit)
    @test DRModels.niterations(fit) >= 0
    @test DRModels.family(fit) isa DRModels.Gaussian

    # Formula order is x1 then x2, independent of impute NamedTuple order.
    @test length(DRModels.coef(fit)) == length(reference["theta"])
    @test DRModels.coef(fit) ≈ reference["theta"] atol = 4e-6
    @test length(DRModels.coef(fit, :mu)) == 4
    @test length(DRModels.coef(fit, :mi_x1)) == 2
    @test length(DRModels.coef(fit, :mi_x2)) == 2
    @test DRModels.coef(fit, :sigma_mi_x1) ≈ exp.(DRModels.coef(fit, :logsd_mi_x1)) atol = 1e-12
    @test DRModels.coef(fit, :sigma_mi_x2) ≈ exp.(DRModels.coef(fit, :logsd_mi_x2)) atol = 1e-12
    @test all(>(0), DRModels.coef(fit, :sigma_mi_x1))
    @test all(>(0), DRModels.coef(fit, :sigma_mi_x2))
    @test size(DRModels.vcov(fit)) == (11, 11)
    @test DRModels.vcov(fit) == DRModels.vcov(fit.prepared.fit) # raw log-SD coordinates remain untransformed
    @test isposdef(Symmetric(DRModels.vcov(fit)))

    imp1 = DRModels.imputed(fit; variable = "x1", rows = :all)
    imp2 = DRModels.imputed(fit; variable = :x2, rows = :all)
    @test imp1.original_row == reference["original_row"]
    @test imp2.original_row == reference["original_row"]
    @test imp1.observed == BitVector(reference["x1_observed"])
    @test imp2.observed == BitVector(reference["x2_observed"])
    @test imp1.variable == fill("x1", n)
    @test imp2.variable == fill("x2", n)
    @test all(ismissing, imp1.std_error[imp1.observed])
    @test all(ismissing, imp2.std_error[imp2.observed])
    @test DRModels.imputed(fit; variable = :x1, rows = :missing, se = false).std_error |> x -> all(ismissing, x)
    @test_throws ArgumentError DRModels.imputed(fit)
    @test_throws ArgumentError DRModels.imputed(fit; variable = :unknown)

    # Different fixed predictor-design widths stay a two-Gaussian formula fit.
    uneven = drm(form, Gaussian(); data = data,
        impute = (x1 = @formula(x1 ~ z), x2 = @formula(x2 ~ 1)), missing = ctl, g_tol = 1e-8)
    @test uneven isa DRModels.JointTwoDrmFit
    @test size(uneven.prepared.prepared.Xpredictor[1], 2) == 2
    @test size(uneven.prepared.prepared.Xpredictor[2], 2) == 1
    @test length(DRModels.coef(uneven)) == 10
end

@testset "two Gaussian direct frontend refusals" begin
    reference = _two_frontend_ref
    data = _two_frontend_data(reference)
    ctl = DRModels.miss_control(response = "include", predictor = "model")
    form = DRModels.bf(@formula(y ~ z + mi(x1) + mi(x2)), @formula(sigma ~ 1))
    good = (x1 = @formula(x1 ~ z), x2 = @formula(x2 ~ z))
    data3 = merge(data, (; x3 = copy(data.x1)))

    @test_throws ArgumentError drm(DRModels.bf(@formula(y ~ z + mi(x1) + mi(x2) + mi(x3)), @formula(sigma ~ 1)),
        Gaussian(); data = data3, impute = (x1 = @formula(x1 ~ z), x2 = @formula(x2 ~ z), x3 = @formula(x3 ~ z)), missing = ctl)
    @test_throws ArgumentError drm(DRModels.bf(@formula(y ~ z * mi(x1) + mi(x2)), @formula(sigma ~ 1)),
        Gaussian(); data = data, impute = good, missing = ctl)
    @test_throws ArgumentError drm(form, Gaussian(); data = data,
        impute = (x1 = DRModels.impute_model(@formula(x1 ~ z); family = DRModels.Binomial()), x2 = @formula(x2 ~ z)), missing = ctl)
    @test_throws ArgumentError drm(form, Gaussian(); data = data,
        impute = (x1 = @formula(x1 ~ z), x2 = @formula(x2 ~ z), extra = @formula(x1 ~ z)), missing = ctl)
    @test_throws ArgumentError drm(form, Gaussian(); data = data, impute = good, missing = ctl, method = :REML)
    @test_throws ArgumentError drm(DRModels.bf(@formula(y ~ x1 + mi(x1) + mi(x2)), @formula(sigma ~ 1)),
        Gaussian(); data = data, impute = good, missing = ctl)
    @test_throws ArgumentError drm(DRModels.bf(@formula(y ~ z + mi(x1) + mi(x2)), @formula(sigma ~ x2)),
        Gaussian(); data = data, impute = good, missing = ctl)
    @test_throws ArgumentError drm(form, Gaussian(); data = data,
        impute = (x1 = @formula(x1 ~ y), x2 = @formula(x2 ~ z)), missing = ctl)
    @test_throws ArgumentError drm(form, Gaussian(); data = data,
        impute = (x1 = @formula(x1 ~ z), x2 = @formula(x2 ~ log(abs(x1)))), missing = ctl)
    @test_throws ArgumentError drm(form, Gaussian(); data = data, impute = good,
        missing = DRModels.miss_control(response = "include", predictor = "fail"))
    @test_throws ArgumentError drm(form, Gaussian(); data = data, impute = good, missing = ctl, algorithm = :em)

    # These controls have no missing y/x1/x2 values, so exogeneity refusal is
    # structural rather than an incidental missing-data validation failure.
    complete = (; y = Float64.(coalesce.(data.y, 0.0)),
                 x1 = Float64.(coalesce.(data.x1, 0.0)),
                 x2 = Float64.(coalesce.(data.x2, 0.0)), z = data.z)
    @test_throws ArgumentError drm(DRModels.bf(@formula(y ~ y + mi(x1) + mi(x2)), @formula(sigma ~ 1)),
        Gaussian(); data = complete, impute = good, missing = ctl)
    @test_throws ArgumentError drm(DRModels.bf(@formula(y ~ x1 + mi(x1) + mi(x2)), @formula(sigma ~ 1)),
        Gaussian(); data = complete, impute = good, missing = ctl)
    @test_throws ArgumentError drm(DRModels.bf(@formula(y ~ x2 + mi(x1) + mi(x2)), @formula(sigma ~ 1)),
        Gaussian(); data = complete, impute = good, missing = ctl)
    @test_throws ArgumentError drm(DRModels.bf(@formula(y ~ z + mi(x1) + mi(x2)), @formula(sigma ~ y)),
        Gaussian(); data = complete, impute = good, missing = ctl)
    @test_throws ArgumentError drm(DRModels.bf(@formula(y ~ z + mi(x1) + mi(x2)), @formula(sigma ~ x1)),
        Gaussian(); data = complete, impute = good, missing = ctl)
    @test_throws ArgumentError drm(DRModels.bf(@formula(y ~ z + mi(x1) + mi(x2)), @formula(sigma ~ x2)),
        Gaussian(); data = complete, impute = good, missing = ctl)
    @test_throws ArgumentError drm(form, Gaussian(); data = complete,
        impute = (x1 = @formula(x1 ~ y), x2 = @formula(x2 ~ z)), missing = ctl)
    @test_throws ArgumentError drm(form, Gaussian(); data = complete,
        impute = (x1 = @formula(x1 ~ z), x2 = @formula(x2 ~ x1)), missing = ctl)
end

println("TWO_JOINT_FRONTEND_TEST_READY")
