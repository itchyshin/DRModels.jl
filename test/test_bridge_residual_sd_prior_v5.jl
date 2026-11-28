using DRModels
using Test
using Distributions: Normal, TDist, logpdf

@testset "bridge v5 transmits a normalized residual-SD Gamma prior" begin
    data = (y=[0.2, 0.8, 0.5, 1.4, 1.0, 1.6, 2.0, 1.3, 2.2, 1.8, 2.4, 2.1],
            x=repeat([-1.0, 0.0, 1.0], 4))
    formula = Dict("mu"=>"y ~ 1 + x", "sigma"=>"sigma ~ 1")
    gamma = Dict("kind"=>"residual_sd_gamma", "dpar"=>"sigma",
        "coef"=>"(Intercept)", "shape"=>2.0, "rate"=>3.0,
        "measure"=>"natural_sd")
    gamma_record(i) = Dict("spec_index"=>i, "kind"=>"residual_sd_gamma",
        "class"=>"residual_sd", "dpar"=>"sigma", "coef"=>"(Intercept)",
        "block"=>"gaussian_residual", "measure"=>"natural_sd", "shape"=>2.0,
        "rate"=>3.0, "log_normalizer"=>-2log(3.0)+DRModels.loggamma(2.0),
        "index"=>3)
    fixed = Dict("kind"=>"fixed_student_t", "dpar"=>"mu", "coef"=>"x",
        "location"=>0.0, "scale"=>1.5, "df"=>4.0)
    fixed_record(i) = Dict("spec_index"=>i, "kind"=>"fixed_student_t",
        "class"=>"fixed", "dpar"=>"mu", "coef"=>"x", "location"=>0.0,
        "scale"=>1.5, "df"=>4.0, "index"=>2, "matrix_column"=>2)

    for (specs, expected) in (([gamma], [gamma_record(1)]),
                              ([fixed, gamma], [fixed_record(1), gamma_record(2)]),
                              ([gamma, fixed], [gamma_record(1), fixed_record(2)]))
        options = Dict("prior_contract"=>"parameter_priors_v5",
            "priors_v5"=>specs, "prior_expected_v5"=>expected)
        fit = drm_bridge(; formula, family="gaussian", data, options)
        @test fit["prior_contract"] == "parameter_priors_v5"
        @test length(fit["resolved_priors_v5"]) == length(expected)
        @test only(filter(r -> r["class"] == "residual_sd", fit["resolved_priors_v5"])) == gamma_record(findfirst(==(gamma), specs))
        if any(==(fixed), specs)
            fixed_resolved = only(filter(r -> r["class"] == "fixed", fit["resolved_priors_v5"]))
            @test fixed_resolved["index"] == 2
            @test fixed_resolved["matrix_column"] == 2
        end
        @test fit["residual_sd_prior_penalty"] > 0
        @test fit["penalized_objective"] ≈ fit["data_nll"] +
            fit["fixed_prior_penalty"] + fit["residual_sd_prior_penalty"] atol=1e-8
        @test fit["objective"] ≈ fit["logLik"] + fit["log_prior"] atol=1e-10
        @test fit["log_prior"] ≈ -(fit["fixed_prior_penalty"] +
            fit["residual_sd_prior_penalty"]) atol=1e-10
        @test fit["se_type"] == "penalised"
        @test !haskey(fit, "fixed_prior_contract")
    end

    for field in ("index", "matrix_column")
        bad_fixed = fixed_record(1); bad_fixed[field] += 1
        @test_throws ArgumentError drm_bridge(; formula, family="gaussian", data,
            options=Dict("prior_contract"=>"parameter_priors_v5", "priors_v5"=>[fixed, gamma],
                "prior_expected_v5"=>[bad_fixed, gamma_record(2)]))
    end

    bad_target = gamma_record(1); bad_target["coef"] = "x"
    @test_throws ArgumentError drm_bridge(; formula, family="gaussian", data,
        options=Dict("prior_contract"=>"parameter_priors_v5", "priors_v5"=>[gamma],
            "prior_expected_v5"=>[bad_target]))
    bad_normalizer = gamma_record(1); bad_normalizer["log_normalizer"] += 0.1
    @test_throws ArgumentError drm_bridge(; formula, family="gaussian", data,
        options=Dict("prior_contract"=>"parameter_priors_v5", "priors_v5"=>[gamma],
            "prior_expected_v5"=>[bad_normalizer]))
    wrong_measure = merge(copy(gamma), Dict("measure"=>"log_sd"))
    @test_throws ArgumentError drm_bridge(; formula, family="gaussian", data,
        options=Dict("prior_contract"=>"parameter_priors_v5", "priors_v5"=>[wrong_measure],
            "prior_expected_v5"=>[gamma_record(1)]))
    @test_throws ArgumentError DRModels._bridge_prior_contract(
        DRModels._bridge_prior_safe_options(Dict(
            "prior_contract"=>"parameter_priors_v5", "priors_v5"=>[gamma])))
    @test_throws ArgumentError drm_bridge_inference(; formula, family="gaussian", data,
        options=Dict("prior_contract"=>"parameter_priors_v5", "priors_v5"=>[gamma],
            "prior_expected_v5"=>[gamma_record(1)]))
end

@testset "v5 composes residual-SD and random-intercept SD targets" begin
    data = (y=[0.2, 0.8, 0.5, 1.4, 1.0, 1.6, 2.0, 1.3, 2.2, 1.8, 2.4, 2.1],
            x=repeat([-1.0, 0.0, 1.0], 4),
            site=repeat(["s1", "s2", "s3", "s4"], inner=3),
            observer=repeat(["o1", "o2", "o3"], 4))
    formula = Dict("mu"=>"y ~ 1 + x + (1 | site) + (1 | observer)",
                   "sigma"=>"sigma ~ 1")
    fixed = Dict("kind"=>"fixed_normal", "dpar"=>"mu", "coef"=>"x",
        "mean"=>0.0, "sd"=>2.0)
    sd(group) = Dict("kind"=>"ranef_sd_gamma", "dpar"=>"mu", "group"=>group,
        "coef"=>"(Intercept)", "block_kind"=>"iid_random_intercept",
        "shape"=>2.0, "rate"=>3.0, "measure"=>"natural_sd")
    residual = Dict("kind"=>"residual_sd_gamma", "dpar"=>"sigma",
        "coef"=>"(Intercept)", "shape"=>2.0, "rate"=>3.0,
        "measure"=>"natural_sd")
    fixed_row(i) = Dict("spec_index"=>i, "kind"=>"fixed_normal", "class"=>"fixed",
        "dpar"=>"mu", "coef"=>"x", "mean"=>0.0, "sd"=>2.0, "index"=>2, "matrix_column"=>2)
    sd_row(i, group) = Dict("spec_index"=>i, "kind"=>"ranef_sd_gamma",
        "class"=>"ranef_sd", "dpar"=>"mu", "group"=>group,
        "coef"=>"(Intercept)", "block_kind"=>"iid_random_intercept",
        "shape"=>2.0, "rate"=>3.0, "measure"=>"natural_sd", "index"=>(group == "site" ? 4 : 5))
    residual_row(i) = Dict("spec_index"=>i, "kind"=>"residual_sd_gamma",
        "class"=>"residual_sd", "dpar"=>"sigma", "coef"=>"(Intercept)",
        "block"=>"gaussian_residual", "measure"=>"natural_sd", "shape"=>2.0,
        "rate"=>3.0, "log_normalizer"=>-2log(3.0)+DRModels.loggamma(2.0), "index"=>3)
    specs = [fixed, sd("site"), residual, sd("observer")]
    expected = [fixed_row(1), sd_row(2, "site"), residual_row(3), sd_row(4, "observer")]
    bad_expected = deepcopy(expected)
    bad_expected[2]["index"] += 1
    @test_throws ArgumentError drm_bridge(; formula, family="gaussian", data,
        options=Dict("prior_contract"=>"parameter_priors_v5", "priors_v5"=>specs,
                     "prior_expected_v5"=>bad_expected))
    fit = drm_bridge(; formula, family="gaussian", data,
        options=Dict("prior_contract"=>"parameter_priors_v5", "priors_v5"=>specs,
                     "prior_expected_v5"=>expected))
    resolved = fit["resolved_priors_v5"]
    @test count(r -> r["class"] == "fixed", resolved) == 1
    @test count(r -> r["class"] == "ranef_sd", resolved) == 2
    @test count(r -> r["class"] == "residual_sd", resolved) == 1
    @test sort([r["group"] for r in filter(r -> r["class"] == "ranef_sd", resolved)]) == ["observer", "site"]
    @test fit["penalized_objective"] ≈ fit["data_nll"] + fit["prior_penalty"] atol=1e-8
    @test fit["penalized_gradient"] ≈ fit["data_gradient"] + fit["prior_gradient"] atol=1e-6
end

@testset "v5 composes fixed, residual-SD and q2 correlation priors" begin
    data = (y=[0.75, -0.15, 1.3, 1.05, 0.2, -0.3, 0.1, 0.8, 0.4, -0.6, -0.9, -0.2],
            x=[-1.0, 1.0, -1.0, 0.0, 1.0, -1.0, -0.2, 0.4, 1.0, -1.0, 0.0, 1.0],
            g=[1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4])
    formula = Dict("mu"=>"y ~ x + (1 + x | g)", "sigma"=>"sigma ~ 1")
    fixed = Dict("kind"=>"fixed_normal", "dpar"=>"mu", "coef"=>"x",
        "mean"=>0.25, "sd"=>1.5)
    residual = Dict("kind"=>"residual_sd_gamma", "dpar"=>"sigma",
        "coef"=>"(Intercept)", "shape"=>2.0, "rate"=>3.0,
        "measure"=>"natural_sd")
    corr = Dict("kind"=>"lkj_random_effect_correlation", "dpar"=>"mu",
        "group"=>"g", "from_coef"=>"(Intercept)", "to_coef"=>"x",
        "eta"=>2.0, "measure"=>"natural_correlation")
    fixed_row = Dict("spec_index"=>1, "kind"=>"fixed_normal", "class"=>"fixed",
        "dpar"=>"mu", "coef"=>"x", "mean"=>0.25, "sd"=>1.5, "index"=>2, "matrix_column"=>2)
    residual_row = Dict("spec_index"=>2, "kind"=>"residual_sd_gamma",
        "class"=>"residual_sd", "dpar"=>"sigma", "coef"=>"(Intercept)",
        "block"=>"gaussian_residual", "measure"=>"natural_sd", "shape"=>2.0,
        "rate"=>3.0, "log_normalizer"=>-2log(3.0)+DRModels.loggamma(2.0), "index"=>3)
    corr_row = Dict("spec_index"=>3, "kind"=>"lkj_random_effect_correlation",
        "class"=>"random_effect_correlation", "dpar"=>"mu",
        "block"=>"gaussian_q2", "group"=>"g", "from_coef"=>"(Intercept)",
        "to_coef"=>"x", "scale"=>"natural_correlation", "eta"=>2.0,
        "support_guard"=>0.99999999,
        "log_normalizer"=>log(2 * (0.99999999 - 0.99999999^3 / 3)))
    fit = drm_bridge(; formula, family="gaussian", data,
        options=Dict("prior_contract"=>"parameter_priors_v5",
            "priors_v5"=>[fixed, residual, corr],
            "prior_expected_v5"=>[fixed_row, residual_row, corr_row]))
    resolved = fit["resolved_priors_v5"]
    @test [r["class"] for r in resolved] == ["fixed", "residual_sd", "random_effect_correlation"]
    @test fit["correlation_prior_penalty"] > 0
    @test fit["penalized_objective"] ≈ fit["data_nll"] + fit["prior_penalty"] atol=1e-8
    @test fit["log_prior"] ≈ -fit["prior_penalty"] atol=1e-10
    @test fit["penalized_gradient"] ≈ fit["data_gradient"] + fit["prior_gradient"] atol=1e-6

    unsupported_sd = Dict("kind"=>"ranef_sd_gamma", "dpar"=>"mu", "group"=>"g",
        "coef"=>"(Intercept)", "block_kind"=>"iid_random_intercept", "shape"=>2.0,
        "rate"=>3.0, "measure"=>"natural_sd")
    unsupported_record = Dict("spec_index"=>4, "kind"=>"ranef_sd_gamma",
        "class"=>"ranef_sd", "dpar"=>"mu", "group"=>"g", "coef"=>"(Intercept)",
        "block_kind"=>"iid_random_intercept", "shape"=>2.0, "rate"=>3.0,
        "measure"=>"natural_sd")
    @test_throws ArgumentError drm_bridge(; formula, family="gaussian", data,
        options=Dict("prior_contract"=>"parameter_priors_v5",
            "priors_v5"=>[fixed, residual, corr, unsupported_sd],
            "prior_expected_v5"=>[fixed_row, residual_row, corr_row, unsupported_record]))
end

@testset "v5 bivariate mu1/mu2 fixed priors compose with rho12 LKJ" begin
    data = (y1=[0.5, -0.2, 0.9, 1.1, -0.3, 0.4],
            y2=[0.1, 0.4, -0.5, 0.3, 0.8, -0.2])
    formula = Dict("mu1"=>"y1 ~ 1", "mu2"=>"y2 ~ 1",
        "sigma1"=>"sigma1 ~ 1", "sigma2"=>"sigma2 ~ 1", "rho12"=>"rho12 ~ 1")
    specs = [
        Dict("kind"=>"fixed_normal", "dpar"=>"mu1", "coef"=>"(Intercept)", "mean"=>0.0, "sd"=>2.0),
        Dict("kind"=>"fixed_student_t", "dpar"=>"mu2", "coef"=>"(Intercept)", "location"=>0.0, "scale"=>1.5, "df"=>4.0),
        Dict("kind"=>"lkj_residual_correlation", "dpar"=>"rho12", "coef"=>"(Intercept)",
             "eta"=>2.0, "measure"=>"natural_correlation"),
    ]
    expected = [
        Dict("spec_index"=>1, "kind"=>"fixed_normal", "class"=>"fixed", "dpar"=>"mu1",
             "coef"=>"(Intercept)", "mean"=>0.0, "sd"=>2.0, "index"=>1, "matrix_column"=>1),
        Dict("spec_index"=>2, "kind"=>"fixed_student_t", "class"=>"fixed", "dpar"=>"mu2",
             "coef"=>"(Intercept)", "location"=>0.0, "scale"=>1.5, "df"=>4.0, "index"=>2, "matrix_column"=>1),
        Dict("spec_index"=>3, "kind"=>"lkj_residual_correlation", "class"=>"residual_correlation",
             "dpar"=>"rho12", "block"=>"bivariate_residual", "coef"=>"(Intercept)",
             "scale"=>"natural_correlation", "eta"=>2.0, "support_guard"=>0.999999,
             "log_normalizer"=>log(2 * (0.999999 - 0.999999^3 / 3))),
    ]
    resolved_expected = [merge(copy(expected[1]), Dict("index"=>1, "matrix_column"=>1)),
        merge(copy(expected[2]), Dict("index"=>2, "matrix_column"=>1)),
        merge(copy(expected[3]), Dict("guard"=>0.999999, "index"=>5))]
    fit = drm_bridge(; formula, family="gaussian", data,
        options=Dict("prior_contract"=>"parameter_priors_v5", "priors_v5"=>specs,
                     "prior_expected_v5"=>expected))
    @test fit["resolved_priors_v5"] == resolved_expected
    theta = fit["coefficients"]
    expected_fixed = -logpdf(Normal(0, 2), theta[1]) -
        logpdf(TDist(4), theta[2] / 1.5) + log(1.5)
    @test fit["fixed_prior_penalty"] ≈ expected_fixed atol=1e-10
    @test fit["prior_contract"] == "parameter_priors_v5"
    @test fit["penalized_objective"] ≈ fit["data_nll"] + fit["prior_penalty"] atol=1e-8
    @test fit["correlation_prior_penalty"] > 0
    @test fit["log_prior"] ≈ -fit["prior_penalty"] atol=1e-10
    @test fit["penalized_gradient"] ≈ fit["data_gradient"] + fit["prior_gradient"] atol=1e-6

    gamma = Dict("kind"=>"residual_sd_gamma", "dpar"=>"sigma", "coef"=>"(Intercept)",
        "shape"=>2.0, "rate"=>3.0, "measure"=>"natural_sd")
    gamma_record = Dict("spec_index"=>1, "kind"=>"residual_sd_gamma", "class"=>"residual_sd",
        "dpar"=>"sigma", "coef"=>"(Intercept)", "block"=>"gaussian_residual",
        "measure"=>"natural_sd", "shape"=>2.0, "rate"=>3.0,
        "log_normalizer"=>-2log(3.0)+DRModels.loggamma(2.0), "index"=>3)
    @test_throws ArgumentError drm_bridge(; formula, family="gaussian", data,
        options=Dict("prior_contract"=>"parameter_priors_v5", "priors_v5"=>[gamma],
                     "prior_expected_v5"=>[gamma_record]))
end
