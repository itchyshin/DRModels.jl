using DRModels
using Test
using LinearAlgebra
using TOML

# Direct finite-state prediction is a known-state plug-in contract.  This test
# uses the frozen all-four-mask native fixture only to obtain stable fitted
# objects; expected new-data values are built independently from the documented
# no-intercept coefficient coordinates below.
const _finite_prediction_reference = TOML.parsefile(joinpath(
    @__DIR__, "..", "docs", "dev-log", "evidence", "julia-r-parity",
    "finite-state", "finite-reference-003.toml"))

function _finite_prediction_data(reference)
    levels = String.(reference["levels"])
    y = Union{Missing,Float64}[reference["observed_y"][i] ? Float64(reference["y"][i]) : missing
                               for i in eachindex(reference["y"])]
    x = Union{Missing,String}[reference["observed_x"][i] ? levels[reference["x"][i]] : missing
                              for i in eachindex(reference["x"])]
    n = length(y)
    a = repeat(["a", "b", "c"], outer = cld(n, 3))[1:n]
    return (; y, x, z = Float64.(reference["z"]), a), levels
end

function _finite_prediction_sigma_row(a::AbstractString)
    return [a == "a" ? 1.0 : 0.0,
            a == "b" ? 1.0 : 0.0,
            a == "c" ? 1.0 : 0.0]
end

# `y ~ 0 + mi(x) + a + z`: mi(x) is the first full-rank factor and `a`
# contributes b/c treatment coordinates.  This is deliberately not
# reconstructed from a rendered coefficient name.
function _finite_prediction_ordinal_mu_row(x::AbstractString, a::AbstractString,
                                            z::Real, levels::Vector{String})
    state = findfirst(==(x), levels)
    state === nothing && throw(ArgumentError("unknown test state"))
    row = zeros(6)
    row[state] = 1.0
    a == "b" && (row[4] = 1.0)
    a == "c" && (row[5] = 1.0)
    row[6] = z
    return row
end

# `y ~ 0 + a + mi(x)`: a is the first full-rank factor and mi(x) contributes
# medium/high treatment coordinates.  The two schemas differ despite equal
# width, which is the regression this slice must prevent.
function _finite_prediction_categorical_mu_row(x::AbstractString, a::AbstractString,
                                                levels::Vector{String})
    state = findfirst(==(x), levels)
    state === nothing && throw(ArgumentError("unknown test state"))
    row = zeros(5)
    a == "a" && (row[1] = 1.0)
    a == "b" && (row[2] = 1.0)
    a == "c" && (row[3] = 1.0)
    state == 2 && (row[4] = 1.0)
    state == 3 && (row[5] = 1.0)
    return row
end

function _finite_prediction_se(row, covariance)
    variance = dot(row, covariance, row)
    variance >= 0 || throw(ArgumentError("test oracle received a negative variance"))
    return sqrt(variance)
end

function _finite_prediction_rewrap(fit; covariance = DRModels.vcov(fit),
                                   theta = fit.prepared.fit.theta,
                                   covariance_status = fit.prepared.metadata.covariance_status)
    base = fit.prepared.fit
    revised = DRModels.DrmFit(base.family, base.blocks, base.coefnames, Float64.(theta),
        Matrix{Float64}(covariance), base.loglik, base.nobs, base.converged,
        base.means, base.obs, base.scales, base.formula, base.nll, base.nllgrad,
        base.ranef, base.estim_method, base.reml_loglik, base.ml_loglik,
        base.marginal, base.phylo_penalty, base.penalty, base.iterations)
    metadata = fit.prepared.metadata
    revised_metadata = typeof(metadata)(metadata.predictor, metadata.variable,
        metadata.levels, metadata.original_row, metadata.observed_y,
        metadata.observed_x, metadata.conditional_probabilities,
        metadata.conditional_mean, metadata.conditional_variance,
        metadata.conditional_status, metadata.all_rows,
        metadata.predictor_only_rows, metadata.uncertainty_status,
        metadata.optimizer_status, covariance_status)
    prepared = typeof(fit.prepared)(revised, fit.prepared.prepared, revised_metadata)
    return DRModels.JointFiniteDrmFit(prepared, fit.formula, fit.variable, fit.prediction)
end

@testset "finite-state direct newdata prediction contract" begin
    @test BLAS.get_num_threads() == 1

    ctl = DRModels.miss_control(response = "include", predictor = "model")
    ordinal_reference = _finite_prediction_reference["ordinal"]
    ordinal_data, levels = _finite_prediction_data(ordinal_reference)
    categorical_reference = _finite_prediction_reference["categorical"]
    categorical_data, categorical_levels = _finite_prediction_data(categorical_reference)
    @test categorical_levels != levels

    ordinal_formula = DRModels.bf(@formula(y ~ 0 + mi(x) + a + z), @formula(sigma ~ 0 + a))
    ordinal_spec = DRModels.impute_model(@formula(x ~ z);
        family = DRModels.CumulativeLogit(), levels = levels)
    ordinal_fit = drm(ordinal_formula, DRModels.Gaussian(); data = ordinal_data,
        impute = (x = ordinal_spec,), missing = ctl, g_tol = 1e-8)

    categorical_formula = DRModels.bf(@formula(y ~ 0 + a + mi(x)), @formula(sigma ~ z))
    categorical_spec = DRModels.impute_model(@formula(x ~ 1);
        family = DRModels.CategoricalLogit(), levels = categorical_levels)
    categorical_fit = drm(categorical_formula, DRModels.Gaussian(); data = categorical_data,
        impute = (x = categorical_spec,), missing = ctl, g_tol = 1e-8)

    # With an intercept and a complete factor before `mi(x)`, ordinal marker
    # columns use the fitted polynomial coordinates.  This is distinct from
    # the full-rank marker-first no-intercept fit above.
    ordinal_polynomial_formula = DRModels.bf(@formula(y ~ a + mi(x) + z), @formula(sigma ~ 1))
    ordinal_polynomial_fit = drm(ordinal_polynomial_formula, DRModels.Gaussian();
        data = ordinal_data, impute = (x = ordinal_spec,), missing = ctl, g_tol = 1e-8)

    @test ordinal_fit isa DRModels.JointFiniteDrmFit
    @test categorical_fit isa DRModels.JointFiniteDrmFit
    @test ordinal_polynomial_fit isa DRModels.JointFiniteDrmFit
    @test all(isfinite, DRModels.vcov(ordinal_fit))
    @test all(isfinite, DRModels.vcov(categorical_fit))

    # Expected values are known-state plug-ins.  New y is omitted, then changed
    # to an implausible value; the prediction must not condition on it.
    ordinal_new = (; x = ["low", "medium", "high"], a = ["a", "b", "c"], z = [-0.4, 0.0, 0.6])
    ordinal_rows = reduce(vcat, permutedims.([
        _finite_prediction_ordinal_mu_row(ordinal_new.x[i], ordinal_new.a[i],
                                           ordinal_new.z[i], levels)
        for i in eachindex(ordinal_new.x)
    ]))
    ordinal_expected = ordinal_rows * DRModels.coef(ordinal_fit, :mu)

    # RED: `JointFiniteDrmFit` currently has no predict(newdata) method.
    ordinal_prediction = DRModels.predict(ordinal_fit, ordinal_new)
    @test ordinal_prediction ≈ ordinal_expected atol = 2e-12 rtol = 0
    @test DRModels.predict(ordinal_fit, merge(ordinal_new, (; y = [1.0e6, -1.0e6, 42.0]))) ≈
          ordinal_prediction atol = 0 rtol = 0
    @test DRModels.predict(ordinal_fit, ordinal_new; type = :link) ≈ ordinal_prediction atol = 0 rtol = 0

    ordinal_se = DRModels.predict(ordinal_fit, ordinal_new; se = true)
    ordinal_range = Dict(ordinal_fit.prepared.fit.blocks)[:mu]
    @test ordinal_se.prediction ≈ ordinal_expected atol = 2e-12 rtol = 0
    @test ordinal_se.se ≈ [_finite_prediction_se(view(ordinal_rows, i, :),
        DRModels.vcov(ordinal_fit)[ordinal_range, ordinal_range]) for i in axes(ordinal_rows, 1)] atol = 2e-12 rtol = 0

    # Sigma uses only its own retained no-intercept factor schema. It must not
    # require or recode x, and a singleton batch must retain all three columns.
    sigma_new = (; a = ["b"])
    sigma_rows = reduce(vcat, permutedims.(_finite_prediction_sigma_row.(sigma_new.a)))
    sigma_eta = sigma_rows * DRModels.coef(ordinal_fit, :sigma)
    @test DRModels.predict(ordinal_fit, sigma_new; dpar = :sigma, type = :link) ≈ sigma_eta atol = 2e-12 rtol = 0
    @test DRModels.predict(ordinal_fit, sigma_new; dpar = :sigma) ≈ exp.(sigma_eta) atol = 2e-12 rtol = 0
    sigma_se = DRModels.predict(ordinal_fit, sigma_new; dpar = :sigma, se = true)
    sigma_range = Dict(ordinal_fit.prepared.fit.blocks)[:sigma]
    sigma_link_se = [_finite_prediction_se(view(sigma_rows, i, :),
        DRModels.vcov(ordinal_fit)[sigma_range, sigma_range]) for i in axes(sigma_rows, 1)]
    @test sigma_se.prediction ≈ exp.(sigma_eta) atol = 2e-12 rtol = 0
    @test sigma_se.se ≈ exp.(sigma_eta) .* sigma_link_se atol = 2e-12 rtol = 0

    # A subset/singleton factor must retain the fitted no-intercept coordinate
    # system.  Categorical factor-first and ordinal marker-first have different
    # coefficient order, so both families are exercised.
    categorical_new = (; x = [categorical_levels[2]], a = ["b"], z = [0.25])
    categorical_row = _finite_prediction_categorical_mu_row(categorical_levels[2], "b", categorical_levels)
    categorical_expected = dot(categorical_row, DRModels.coef(categorical_fit, :mu))
    @test DRModels.predict(categorical_fit, categorical_new) ≈ [categorical_expected] atol = 2e-12 rtol = 0
    @test DRModels.predict(categorical_fit, merge(categorical_new, (; y = [-9.0e5]))) ≈
          [categorical_expected] atol = 0 rtol = 0

    polynomial_new = (; x = ["medium"], a = ["b"], z = [0.25])
    polynomial_contrast = DRModels._joint_finite_polynomials(length(levels))
    polynomial_row = [1.0, 1.0, 0.0, polynomial_contrast[2, 1],
                      polynomial_contrast[2, 2], 0.25]
    @test DRModels.predict(ordinal_polynomial_fit, polynomial_new) ≈
          [dot(polynomial_row, DRModels.coef(ordinal_polynomial_fit, :mu))] atol = 2e-12 rtol = 0

    # Callers and fitting inputs must stay immutable across repeated prediction.
    ordinal_data_before = deepcopy(ordinal_data)
    categorical_data_before = deepcopy(categorical_data)
    ordinal_theta_before = copy(ordinal_fit.prepared.fit.theta)
    categorical_theta_before = copy(categorical_fit.prepared.fit.theta)
    ordinal_X_before = copy(ordinal_fit.prepared.prepared.X_mu_state)
    categorical_X_before = copy(categorical_fit.prepared.prepared.X_mu_state)
    ordinal_new_before = deepcopy(ordinal_new)
    mean_plan_before = (levels = copy(ordinal_fit.prediction.mean.levels),
                        permutation = copy(ordinal_fit.prediction.mean.permutation),
                        names = copy(ordinal_fit.prediction.mean.names))
    sigma_names_before = copy(ordinal_fit.prediction.sigma.names)
    DRModels.predict(ordinal_fit, ordinal_new)
    DRModels.predict(categorical_fit, categorical_new)
    @test isequal(ordinal_data, ordinal_data_before)
    @test isequal(categorical_data, categorical_data_before)
    @test ordinal_fit.prepared.fit.theta == ordinal_theta_before
    @test categorical_fit.prepared.fit.theta == categorical_theta_before
    @test ordinal_fit.prepared.prepared.X_mu_state == ordinal_X_before
    @test categorical_fit.prepared.prepared.X_mu_state == categorical_X_before
    @test isequal(ordinal_new, ordinal_new_before)
    @test ordinal_fit.prediction.mean.levels == mean_plan_before.levels
    @test ordinal_fit.prediction.mean.permutation == mean_plan_before.permutation
    @test ordinal_fit.prediction.mean.names == mean_plan_before.names
    @test ordinal_fit.prediction.sigma.names == sigma_names_before

    # Every computed design, linear predictor, response-scale prediction, and
    # standard error must be finite.  These are deliberately not allowed to
    # leak Inf/NaN into a public prediction result.
    @test_throws ArgumentError DRModels.predict(ordinal_fit,
        (; x = ["low"], a = ["a"], z = [Inf]))
    @test_throws ArgumentError DRModels.predict(ordinal_fit,
        (; x = ["low"], a = ["a"], z = [NaN]))
    overflow_theta = copy(categorical_fit.prepared.fit.theta)
    sigma_start = first(Dict(categorical_fit.prepared.fit.blocks)[:sigma])
    overflow_theta[sigma_start] = 1.0e3
    overflow_fit = _finite_prediction_rewrap(categorical_fit; theta = overflow_theta)
    @test_throws ArgumentError DRModels.predict(overflow_fit,
        (; z = [0.0]); dpar = :sigma, type = :response)

    # SEs require a finite positive-definite covariance and the successful
    # observed-information status.  A materially negative quadratic form must
    # be rejected, never truncated to zero by a defensive `max(0, ...)`.
    p = length(categorical_fit.prepared.fit.theta)
    negative_fit = _finite_prediction_rewrap(categorical_fit;
        covariance = -Matrix{Float64}(I, p, p))
    @test_throws ArgumentError DRModels.predict(negative_fit, categorical_new; se = true)
    tiny_negative_fit = _finite_prediction_rewrap(categorical_fit;
        covariance = -1.0e-30 * Matrix{Float64}(I, p, p))
    @test_throws ArgumentError DRModels.predict(tiny_negative_fit, categorical_new; se = true)
    tiny_positive_fit = _finite_prediction_rewrap(categorical_fit;
        covariance = 1.0e-30 * Matrix{Float64}(I, p, p))
    tiny_positive = DRModels.predict(tiny_positive_fit, categorical_new; se = true)
    @test all(isfinite, tiny_positive.se)
    @test all(>(0.0), tiny_positive.se)
    nan_covariance = copy(DRModels.vcov(categorical_fit)); nan_covariance[1, 1] = NaN
    @test_throws ArgumentError DRModels.predict(
        _finite_prediction_rewrap(categorical_fit; covariance = nan_covariance),
        categorical_new; se = true)
    inf_covariance = copy(DRModels.vcov(categorical_fit)); inf_covariance[1, 1] = Inf
    @test_throws ArgumentError DRModels.predict(
        _finite_prediction_rewrap(categorical_fit; covariance = inf_covariance),
        categorical_new; se = true)
    failed_covariance_fit = _finite_prediction_rewrap(categorical_fit;
        covariance_status = :not_positive_definite)
    @test DRModels.predict(failed_covariance_fit, categorical_new) ≈
          [categorical_expected] atol = 0 rtol = 0
    @test_throws ArgumentError DRModels.predict(failed_covariance_fit, categorical_new; se = true)

    # The compatibility constructor intentionally preserves construction but
    # cannot invent applied schemas after the fact.
    manual_fit = DRModels.JointFiniteDrmFit(ordinal_fit.prepared, ordinal_fit.formula,
        ordinal_fit.variable)
    @test_throws ArgumentError DRModels.predict(manual_fit, ordinal_new)

    # Mu requires a known modelled state, whereas sigma never does.  Retained
    # categorical schemas reject unknown fixed-factor levels rather than silently
    # changing design width or coordinate order.
    @test_throws ArgumentError DRModels.predict(ordinal_fit, (; a = ["a"], z = [0.0]))
    @test_throws ArgumentError DRModels.predict(ordinal_fit,
        (; x = Union{Missing,String}[missing], a = ["a"], z = [0.0]))
    @test_throws ArgumentError DRModels.predict(ordinal_fit,
        (; x = ["unknown"], a = ["a"], z = [0.0]))
    @test_throws ArgumentError DRModels.predict(categorical_fit,
        (; x = [categorical_levels[1]], a = ["unknown"], z = [0.0]))
    @test_throws ArgumentError DRModels.predict(ordinal_fit,
        (; a = ["unknown"]); dpar = :sigma)
    @test_throws ArgumentError DRModels.predict(ordinal_fit,
        (; x = ["low", "medium"], a = ["a"], z = [0.0, 0.5]))
    @test_throws ArgumentError DRModels.predict(ordinal_fit, ordinal_new; dpar = :rho12)
    @test_throws ArgumentError DRModels.predict(ordinal_fit, ordinal_new; type = :distribution)
    @test_throws ArgumentError DRModels.predict(ordinal_fit, ordinal_new; se = "yes")
end

println("FINITE_JOINT_PREDICTION_TEST_READY")
