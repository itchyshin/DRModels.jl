using DRModels
using Test

const bridge_corr_residual_data = (
    y1=[0.5, -0.2, 0.9, 1.1, -0.3, 0.4],
    y2=[0.1, 0.4, -0.5, 0.3, 0.8, -0.2],
)
const bridge_corr_q2_data = (
    y=[0.75, -0.15, 1.3, 1.05, 0.2, -0.3, 0.1, 0.8, 0.4, -0.6, -0.9, -0.2],
    x=[-1.0, 1.0, -1.0, 0.0, 1.0, -1.0, -0.2, 0.4, 1.0, -1.0, 0.0, 1.0],
    g=[1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4],
)

@testset "correlation-prior bridge v4 keeps residual and random-effect targets distinct" begin
    rho_spec = Dict("kind"=>"lkj_residual_correlation", "dpar"=>"rho12",
                    "coef"=>"(Intercept)", "eta"=>2.0,
                    "measure"=>"natural_correlation")
    rho_record = Dict("spec_index"=>1, "kind"=>"lkj_residual_correlation",
                      "class"=>"residual_correlation", "dpar"=>"rho12",
                      "block"=>"bivariate_residual", "coef"=>"(Intercept)",
                      "scale"=>"natural_correlation", "eta"=>2.0,
                      "support_guard"=>0.999999,
                      "log_normalizer"=>log(2 * (0.999999 - 0.999999^3 / 3)))
    q2_spec = Dict("kind"=>"lkj_random_effect_correlation", "dpar"=>"mu",
                   "group"=>"g", "from_coef"=>"(Intercept)", "to_coef"=>"x",
                   "eta"=>2.0, "measure"=>"natural_correlation")
    q2_record = Dict("spec_index"=>1, "kind"=>"lkj_random_effect_correlation",
                     "class"=>"random_effect_correlation", "dpar"=>"mu",
                     "block"=>"gaussian_q2", "group"=>"g",
                     "from_coef"=>"(Intercept)", "to_coef"=>"x",
                     "scale"=>"natural_correlation", "eta"=>2.0,
                     "support_guard"=>0.99999999,
                     "log_normalizer"=>log(2 * (0.99999999 - 0.99999999^3 / 3)))

    rho_opts = Dict("prior_contract"=>"parameter_priors_v4",
                    "priors_v4"=>[rho_spec], "prior_expected_v4"=>[rho_record])
    q2_opts = Dict("prior_contract"=>"parameter_priors_v4",
                   "priors_v4"=>[q2_spec], "prior_expected_v4"=>[q2_record])
    rho_formula = Dict("mu1"=>"y1 ~ 1", "mu2"=>"y2 ~ 1",
                       "sigma1"=>"sigma1 ~ 1", "sigma2"=>"sigma2 ~ 1",
                       "rho12"=>"rho12 ~ 1")
    q2_formula = Dict("mu"=>"y ~ x + (1 + x | g)", "sigma"=>"sigma ~ 1")

    @test DRModels._bridge_prior_contract(DRModels._bridge_prior_safe_options(rho_opts)) === :v4
    rho = drm_bridge(; formula=rho_formula, family="gaussian",
                     data=bridge_corr_residual_data, options=rho_opts)
    @test rho["prior_contract"] == "parameter_priors_v4"
    @test length(rho["resolved_priors_v4"]) == 1
    @test all(haskey(rho["resolved_priors_v4"][1], key) for key in keys(rho_record))
    @test Dict(key=>rho["resolved_priors_v4"][1][key] for key in keys(rho_record)) == rho_record
    @test rho["resolved_priors_v4"][1]["index"] == 5
    @test rho["resolved_priors_v4"][1]["guard"] == 0.999999
    @test rho["resolved_priors_v4"][1]["class"] == "residual_correlation"
    @test rho["resolved_priors_v4"][1]["block"] == "bivariate_residual"
    @test rho["log_prior"] ≈ -rho["correlation_prior_penalty"]
    @test rho["penalized_objective"] ≈ rho["data_nll"] + rho["correlation_prior_penalty"] atol=1e-8
    @test rho["penalized_gradient"] ≈ rho["data_gradient"] + rho["correlation_prior_gradient"] atol=1e-6
    @test rho["estim_method"] == "MAP"
    rho_natural = 0.999999 * tanh(rho["working_values"][5])
    rho_log_prior = log1p(-rho_natural^2) - log(2 * (0.999999 - 0.999999^3 / 3))
    @test rho["log_prior"] ≈ rho_log_prior atol=1e-10
    @test rho["penalised"] === true
    @test rho["objective"] ≈ -rho["penalized_objective"] atol=1e-10
    @test rho["objective"] ≈ rho["logLik"] + rho["log_prior"] atol=1e-10
    @test rho["logLik"] ≈ -rho["data_nll"] atol=1e-10
    @test rho["pd_hessian_unpenalised"] isa Union{Bool,Missing}
    @test all(p -> Set(keys(p)) == Set(("name", "kind")), rho["prior_held_params"])
    @test rho["se_type"] == "penalised"
    rho_replay = drm_bridge(; formula=rho_formula, family="gaussian",
                            data=bridge_corr_residual_data, options=rho_opts)
    @test rho_replay["resolved_priors_v4"] == rho["resolved_priors_v4"]
    @test rho_replay["correlation_prior_penalty"] ≈ rho["correlation_prior_penalty"] atol=1e-10

    q2 = drm_bridge(; formula=q2_formula, family="gaussian",
                    data=bridge_corr_q2_data, options=q2_opts)
    @test q2["prior_contract"] == "parameter_priors_v4"
    @test length(q2["resolved_priors_v4"]) == 1
    @test all(haskey(q2["resolved_priors_v4"][1], key) for key in keys(q2_record))
    @test Dict(key=>q2["resolved_priors_v4"][1][key] for key in keys(q2_record)) == q2_record
    @test q2["resolved_priors_v4"][1]["index"] == 4
    @test q2["resolved_priors_v4"][1]["guard"] == 0.99999999
    @test q2["resolved_priors_v4"][1]["class"] == "random_effect_correlation"
    @test q2["resolved_priors_v4"][1]["block"] == "gaussian_q2"
    @test q2["resolved_priors_v4"][1]["group"] == "g"
    @test q2["resolved_priors_v4"][1]["from_coef"] == "(Intercept)"
    @test q2["resolved_priors_v4"][1]["to_coef"] == "x"
    @test q2["log_prior"] ≈ -q2["correlation_prior_penalty"]
    @test q2["penalized_objective"] ≈ q2["data_nll"] + q2["correlation_prior_penalty"] atol=1e-8
    @test q2["penalized_gradient"] ≈ q2["data_gradient"] + q2["correlation_prior_gradient"] atol=1e-6
    @test q2["estim_method"] == "MAP"
    q2_theta = q2["working_values"]
    q2_rho = q2_theta[6] / hypot(q2_theta[6], exp(q2_theta[5]))
    @test q2["log_prior"] ≈ log1p(-q2_rho^2) - log(4 / 3) atol=1e-10
    @test q2["penalised"] === true
    @test q2["objective"] ≈ -q2["penalized_objective"] atol=1e-10
    @test q2["objective"] ≈ q2["logLik"] + q2["log_prior"] atol=1e-10
    @test q2["logLik"] ≈ -q2["data_nll"] atol=1e-10
    @test q2["pd_hessian_unpenalised"] isa Union{Bool,Missing}
    @test all(p -> Set(keys(p)) == Set(("name", "kind")), q2["prior_held_params"])
    @test q2["se_type"] == "penalised"
    q2_replay = drm_bridge(; formula=q2_formula, family="gaussian",
                           data=bridge_corr_q2_data, options=q2_opts)
    @test q2_replay["resolved_priors_v4"] == q2["resolved_priors_v4"]
    @test q2_replay["log_prior"] ≈ q2["log_prior"] atol=1e-10
    @test_throws ArgumentError DRModels.drm_bridge_inference(; formula=q2_formula,
        family="gaussian", data=bridge_corr_q2_data, options=q2_opts)

    # A residual-correlation record cannot be relabelled as a random-effect
    # pair, nor can a pair be swapped while retaining a matching spec.
    wrong_rho = deepcopy(rho_opts)
    wrong_rho["prior_expected_v4"][1]["class"] = "random_effect_correlation"
    @test_throws ArgumentError drm_bridge(; formula=rho_formula, family="gaussian",
        data=bridge_corr_residual_data, options=wrong_rho)
    reversed_pair = deepcopy(q2_opts)
    reversed_pair["priors_v4"][1]["from_coef"] = "x"
    reversed_pair["priors_v4"][1]["to_coef"] = "(Intercept)"
    reversed_fit = drm_bridge(; formula=q2_formula, family="gaussian",
        data=bridge_corr_q2_data, options=reversed_pair)
    @test reversed_fit["resolved_priors_v4"] == q2["resolved_priors_v4"]
    swapped_record = deepcopy(q2_opts)
    swapped_record["prior_expected_v4"][1]["from_coef"] = "x"
    swapped_record["prior_expected_v4"][1]["to_coef"] = "(Intercept)"
    @test_throws ArgumentError drm_bridge(; formula=q2_formula, family="gaussian",
        data=bridge_corr_q2_data, options=swapped_record)
    wrong_group = deepcopy(q2_opts)
    wrong_group["prior_expected_v4"][1]["group"] = "other"
    @test_throws ArgumentError drm_bridge(; formula=q2_formula, family="gaussian",
        data=bridge_corr_q2_data, options=wrong_group)
    supplied_index = deepcopy(q2_opts)
    supplied_index["prior_expected_v4"][1]["index"] = 1
    @test_throws ArgumentError drm_bridge(; formula=q2_formula, family="gaussian",
        data=bridge_corr_q2_data, options=supplied_index)
    supplied_guard = deepcopy(q2_opts)
    supplied_guard["prior_expected_v4"][1]["support_guard"] = 0.5
    @test_throws ArgumentError drm_bridge(; formula=q2_formula, family="gaussian",
        data=bridge_corr_q2_data, options=supplied_guard)
    supplied_normalizer = deepcopy(q2_opts)
    supplied_normalizer["prior_expected_v4"][1]["log_normalizer"] += 0.1
    @test_throws ArgumentError drm_bridge(; formula=q2_formula, family="gaussian",
        data=bridge_corr_q2_data, options=supplied_normalizer)
    mixed_protocol = merge(q2_opts, Dict("priors_v3"=>[Dict("kind"=>"x")]))
    @test_throws ArgumentError DRModels._bridge_prior_contract(
        DRModels._bridge_prior_safe_options(mixed_protocol))

    # The unpenalized route retains its usual output shape and has no v4 payload.
    plain = drm_bridge(; formula=q2_formula, family="gaussian", data=bridge_corr_q2_data)
    @test !haskey(plain, "prior_contract")
    @test !haskey(plain, "correlation_prior_penalty")
    @test all(key -> !haskey(plain, key), ("penalised", "objective", "logLik", "log_prior",
        "pd_hessian_unpenalised", "prior_held_params", "se_type"))
end

@testset "v4 preserves the q2 fixed-effect and correlation prior table" begin
    fixed = Dict("kind"=>"fixed_normal", "dpar"=>"mu", "coef"=>"x",
                 "mean"=>0.25, "sd"=>1.5)
    fixed_record(spec_index) = Dict("spec_index"=>spec_index, "kind"=>"fixed_normal",
        "class"=>"fixed", "dpar"=>"mu", "coef"=>"x", "mean"=>0.25, "sd"=>1.5)
    corr = Dict("kind"=>"lkj_random_effect_correlation", "dpar"=>"mu",
        "group"=>"g", "from_coef"=>"(Intercept)", "to_coef"=>"x",
        "eta"=>2.0, "measure"=>"natural_correlation")
    corr_record(spec_index) = Dict("spec_index"=>spec_index,
        "kind"=>"lkj_random_effect_correlation", "class"=>"random_effect_correlation",
        "dpar"=>"mu", "block"=>"gaussian_q2", "group"=>"g",
        "from_coef"=>"(Intercept)", "to_coef"=>"x",
        "scale"=>"natural_correlation", "eta"=>2.0,
        "support_guard"=>0.99999999,
        "log_normalizer"=>log(2 * (0.99999999 - 0.99999999^3 / 3)))
    formula = Dict("mu"=>"y ~ x + (1 + x | g)", "sigma"=>"sigma ~ 1")

    # First admit the fixed prior and the named q2 LKJ target together. Both
    # wire orders must resolve to the same semantic targets and additive terms.
    mixed_inputs = (
        ([corr, fixed], [corr_record(1), fixed_record(2)]),
        ([fixed, corr], [fixed_record(1), corr_record(2)]),
    )
    for (specs, expected) in mixed_inputs
        opts = Dict("prior_contract"=>"parameter_priors_v4",
            "priors_v4"=>specs, "prior_expected_v4"=>expected)
        fit = drm_bridge(; formula, family="gaussian", data=bridge_corr_q2_data, options=opts)
        @test length(fit["resolved_priors_v4"]) == 2
        @test count(r -> r["class"] == "fixed", fit["resolved_priors_v4"]) == 1
        @test count(r -> r["class"] == "random_effect_correlation", fit["resolved_priors_v4"]) == 1
        @test any(r -> r["class"] == "random_effect_correlation" &&
            r["index"] == 4 && r["guard"] == 0.99999999 &&
            haskey(r, "log_normalizer"), fit["resolved_priors_v4"])
        @test fit["fixed_prior_penalty"] > 0
        @test fit["correlation_prior_penalty"] >= 0
        @test fit["penalized_objective"] ≈ fit["data_nll"] +
            fit["fixed_prior_penalty"] + fit["correlation_prior_penalty"] atol=1e-8
        @test fit["log_prior"] ≈ -(fit["fixed_prior_penalty"] +
            fit["correlation_prior_penalty"]) atol=1e-10
        @test fit["penalized_gradient"] ≈ fit["data_gradient"] +
            fit["fixed_prior_gradient"] + fit["correlation_prior_gradient"] atol=1e-6
        @test fit["objective"] ≈ fit["logLik"] + fit["log_prior"] atol=1e-10
    end

    # The correlation-only q2 v4 route remains supported and reports a zero
    # fixed component, so callers can use one stable decomposition shape.
    q2_opts = Dict("prior_contract"=>"parameter_priors_v4",
        "priors_v4"=>[corr], "prior_expected_v4"=>[corr_record(1)])
    corr_only = drm_bridge(; formula, family="gaussian", data=bridge_corr_q2_data,
                           options=q2_opts)
    @test corr_only["fixed_prior_penalty"] == 0
    @test corr_only["penalized_objective"] ≈ corr_only["data_nll"] +
        corr_only["correlation_prior_penalty"] atol=1e-8

    # Student-t and ridge rows use the same table contract and decomposition.
    for (spec, record) in (
        (Dict("kind"=>"fixed_student_t", "dpar"=>"mu", "coef"=>"x",
              "location"=>0.0, "scale"=>1.2, "df"=>4.0),
         Dict("kind"=>"fixed_student_t", "class"=>"fixed", "dpar"=>"mu",
              "coef"=>"x", "location"=>0.0, "scale"=>1.2, "df"=>4.0)),
        (Dict("kind"=>"fixed_ridge", "dpar"=>"mu", "coef"=>"x", "lambda"=>0.4),
         Dict("kind"=>"fixed_ridge", "class"=>"fixed", "dpar"=>"mu",
              "coef"=>"x", "lambda"=>0.4)),
    )
        for reverse_order in (false, true)
            specs = reverse_order ? [spec, corr] : [corr, spec]
            records = reverse_order ? [merge(record, Dict("spec_index"=>1)), corr_record(2)] :
                                      [corr_record(1), merge(record, Dict("spec_index"=>2))]
            options = Dict("prior_contract"=>"parameter_priors_v4",
                "priors_v4"=>specs, "prior_expected_v4"=>records)
            fit = drm_bridge(; formula, family="gaussian", data=bridge_corr_q2_data,
                             options=options)
            @test fit["fixed_prior_penalty"] >= 0
            @test fit["penalized_objective"] ≈ fit["data_nll"] +
                fit["fixed_prior_penalty"] + fit["correlation_prior_penalty"] atol=1e-8
            @test fit["log_prior"] ≈ -(fit["fixed_prior_penalty"] +
                fit["correlation_prior_penalty"]) atol=1e-10
        end
    end

    # Semantic record mismatches and omissions fail closed; internal packed
    # coordinates remain Julia-derived and are not part of the input contract.
    wrong = Dict("prior_contract"=>"parameter_priors_v4", "priors_v4"=>[corr, fixed],
        "prior_expected_v4"=>[corr_record(1), merge(fixed_record(2), Dict("coef"=>"(Intercept)"))])
    @test_throws ArgumentError drm_bridge(; formula, family="gaussian",
        data=bridge_corr_q2_data, options=wrong)
    missing = Dict("prior_contract"=>"parameter_priors_v4", "priors_v4"=>[corr, fixed],
        "prior_expected_v4"=>[corr_record(1)])
    @test_throws ArgumentError drm_bridge(; formula, family="gaussian",
        data=bridge_corr_q2_data, options=missing)
    duplicate_corr = Dict("prior_contract"=>"parameter_priors_v4",
        "priors_v4"=>[corr, corr],
        "prior_expected_v4"=>[corr_record(1), corr_record(2)])
    @test_throws ArgumentError drm_bridge(; formula, family="gaussian",
        data=bridge_corr_q2_data, options=duplicate_corr)
    unsupported_sd = Dict("kind"=>"ranef_sd_gamma", "dpar"=>"mu", "group"=>"g",
        "coef"=>"(Intercept)", "block_kind"=>"iid_random_intercept",
        "shape"=>2.0, "rate"=>3.0, "measure"=>"natural_sd")
    sd_mix = Dict("prior_contract"=>"parameter_priors_v4",
        "priors_v4"=>[corr, unsupported_sd],
        "prior_expected_v4"=>[corr_record(1), merge(unsupported_sd,
            Dict("spec_index"=>2, "class"=>"ranef_sd"))])
    @test_throws ArgumentError drm_bridge(; formula, family="gaussian",
        data=bridge_corr_q2_data, options=sd_mix)
end

@testset "v4 does not alter prior bridge protocol dispatch" begin
    v1 = Dict("priors"=>[Dict("kind"=>"normal")], "prior_expected"=>[Dict("coef"=>"x")])
    v2 = Dict("prior_contract"=>"parameter_priors_v2", "priors_v2"=>[Dict("kind"=>"x")],
              "prior_expected_v2"=>[Dict("kind"=>"x")])
    v3 = Dict("prior_contract"=>"parameter_priors_v3", "priors_v3"=>[Dict("kind"=>"x")],
              "prior_expected_v3"=>[Dict("kind"=>"x")])
    @test DRModels._bridge_prior_contract(DRModels._bridge_prior_safe_options(v1)) === :v1
    @test DRModels._bridge_prior_contract(DRModels._bridge_prior_safe_options(v2)) === :v2
    @test DRModels._bridge_prior_contract(DRModels._bridge_prior_safe_options(v3)) === :v3
end
