using DRModels, Test, Random
import Distributions

@testset "public Bernoulli/binomial random-intercept SD regression" begin
    rng = MersenneTwister(2026101717)
    G, m, trials = 28, 8, 12
    g = repeat(1:G, inner = m)
    zgroup = repeat([0.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    zfactor = ifelse.(z .== 0, "low", "high")
    x = randn(rng, length(g))
    b = exp.(-0.5 .+ 0.5 .* zgroup) .* randn(rng, G)
    p = 1.0 ./ (1.0 .+ exp.(-(0.1 .+ 0.4 .* x .+ b[g])))
    successes = [rand(rng, Distributions.Binomial(trials, pi)) for pi in p]
    data = (; successes = Float64.(successes), failures = Float64.(trials .- successes),
            y = Float64.(successes ./ trials), x, z, zfactor, g)

    fit = drm(bf(@formula(cbind(successes, failures) ~ x + (1 | g)),
                 @formula(sd(g) ~ zfactor)), Binomial(); data = data)
    @test fit.converged
    @test length(coef(fit, :sd)) == 2
    @test isempty(sigma(fit))
    @test !haskey(DRModels._bootstrap_refit_kwargs(fit), :nAGQ)
    bridge_dpars = DRModels._bridge_dpars(fit)
    @test Set(keys(bridge_dpars)) == Set(["mu"])
    bridge = drm_bridge(; formula = "cbind(successes, failures) ~ x + (1 | g); sd(g) ~ zfactor",
                        family = "binomial", data)
    @test Set(keys(bridge["dpars"])) == Set(["mu"])
    @test isempty(bridge["sigma"])
    group_sd = fit.scales[:sd_group]
    group_index = Int.(fit.scales[:sd_group_index])
    @test length(group_sd) == G
    @test all(isfinite, group_sd) && all(group_sd .> 0)
    pred = predict_parameters(fit, data)
    expected_sd = group_sd[group_index]
    @test pred[:sd] ≈ expected_sd
    pred_link = predict_parameters(fit, data; type = :link)[:sd]
    @test pred_link ≈ log.(expected_sd)
    pred_se = predict_parameters(fit, data; se = true)[:sd]
    @test pred_se.value ≈ expected_sd
    @test all(isfinite, pred_se.se) && all(pred_se.se .>= 0)
    @test marginal_parameters(fit)[:sd] ≈ pred[:sd]

    # Prediction must use the fitted SD schema even when newdata has no response,
    # reordered rows/factor levels, or only a subset of the observed SD levels.
    predictor_only = (; x = data.x, zfactor = data.zfactor, g = data.g)
    @test predict_parameters(fit, predictor_only)[:sd] ≈ pred[:sd]
    predictor_dict = Dict(:x => data.x, :zfactor => data.zfactor, :g => data.g)
    @test predict_parameters(fit, predictor_dict)[:sd] ≈ pred[:sd]
    reversed = length(data.g):-1:1
    reversed_data = (; x = data.x[reversed], zfactor = reverse(data.zfactor),
                       g = data.g[reversed])
    @test predict_parameters(fit, reversed_data)[:sd] ≈ reverse(pred[:sd])
    high_rows = findall(==("high"), data.zfactor)
    high_only = (; x = data.x[high_rows], zfactor = data.zfactor[high_rows], g = data.g[high_rows])
    @test predict_parameters(fit, high_only)[:sd] ≈ pred[:sd][high_rows]
    unseen = merge(predictor_only, (; zfactor = fill("unseen", length(data.g))))
    @test_throws ArgumentError predict_parameters(fit, unseen)

    profile_ci = confint(fit; method = :profile, parm = :sd)
    @test length(profile_ci) == 2
    @test all(row -> isfinite(row.lower) && isfinite(row.upper), profile_ci)
    threaded_profile = profile_result(fit; threads = true, parm = :sd)
    @test threaded_profile.threaded == (Threads.nthreads() > 1)
    @test threaded_profile.used == 2

    # Exercise the fit-based marginal bootstrap that redraws fresh group effects.
    # Two replicates cover the public result and CI wrappers without treating this
    # smoke test as interval calibration.
    boot = bootstrap_result(fit; data, B = 2, rng = MersenneTwister(2026101718),
                            failures = :error)
    @test boot.attempted == 2
    @test boot.used == 2
    @test boot.failed == 0
    boot_ci = bootstrap_ci(fit; data, B = 2, rng = MersenneTwister(2026101719),
                            failures = :skip)
    sd_ci = filter(row -> row.param === :sd, boot_ci)
    @test length(sd_ci) == 2
    @test all(row -> isfinite(row.lower) && isfinite(row.upper), sd_ci)

    # Missing-response routing is not safe for the fit-based marginal bootstrap
    # until the full-row SD design and fresh-effect group map are retained.
    masked = Union{Missing,Float64}[data.successes...]
    masked[1] = missing
    masked_data = merge(data, (; successes = masked))
    @test_throws ArgumentError drm(bf(@formula(cbind(successes, failures) ~ x + (1 | g)),
                                      @formula(sd(g) ~ zfactor)), Binomial(); data = masked_data)

    # Generate slope data under the model exercised below, preserving the
    # original intercept fixture for the intercept and Beta checks above.
    slope_rng = MersenneTwister(2026101717)
    slope_x = randn(slope_rng, length(g))
    slope_b = exp.(-0.5 .+ 0.5 .* zgroup) .* randn(slope_rng, G)
    slope_p = 1.0 ./ (1.0 .+ exp.(-(0.1 .+ 0.4 .* slope_x .+ slope_b[g] .* slope_x)))
    slope_successes = [rand(slope_rng, Distributions.Binomial(trials, pi)) for pi in slope_p]
    slope_test_data = merge(data, (; x = slope_x, successes = Float64.(slope_successes),
                                  failures = Float64.(trials .- slope_successes),
                                  y = Float64.(slope_successes ./ trials)))

    selected_slope = bf(@formula(cbind(successes, failures) ~ x + (0 + x | g)),
                        @formula(sd(g, coef(x)) ~ z))
    slope_fit = drm(selected_slope, Binomial(); data = slope_test_data)
    @test slope_fit.converged
    selected_slope_target = only(DRModels._sd_target_parts(selected_slope))
    selected_slope_block = DRModels._sd_target_block(selected_slope_target.target)
    @test all(isfinite, predict_parameters(slope_fit, slope_test_data)[selected_slope_block])
    @test marginal_parameters(slope_fit)[selected_slope_block] ≈
          predict_parameters(slope_fit, slope_test_data)[selected_slope_block]
    slope_simulator = DRModels._marginal_simulator_build(slope_fit, slope_test_data)
    @test slope_simulator !== nothing
    # A malformed random-slope loading must be rejected before a bootstrap
    # simulator is returned, even when it is not part of the fixed predictor.
    slope_data = merge(data, (; w = randn(rng, length(data.g))))
    w_slope = bf(@formula(cbind(successes, failures) ~ x + (0 + w | g)),
                 @formula(sd(g, coef(w)) ~ z))
    w_slope_fit = drm(w_slope, Binomial(); data = slope_data)
    malformed_slope_data = merge(slope_data, (; w = slope_data.w[1:(end - 1)]))
    @test_throws ArgumentError DRModels._marginal_simulator_build(
        w_slope_fit, malformed_slope_data)
    slope_bridge = drm_bridge(; formula =
        "cbind(successes, failures) ~ x + (0 + x | g); sd(g, coef(x)) ~ z",
        family = "binomial", data = slope_test_data)
    @test slope_bridge["coef_label_contract"] == "bridge_formula_labels_v1"
    @test Set(keys(slope_bridge["dpars"])) == Set(["mu"])
    selected_formula = bf(@formula(cbind(successes, failures) ~ x + (0 + x | g)),
                          @formula(sd(g, coef(x)) ~ zfactor))
    selected_target = only(DRModels._sd_target_parts(selected_formula))
    selected_block = DRModels._sd_target_block(selected_target.target)
    @test DRModels._bridge_lss_form_key(selected_target.key, selected_formula) === selected_block
    slope_boot = bootstrap_result(slope_fit; data = slope_test_data, B = 1,
                                  rng = MersenneTwister(2026101720), failures = :skip)
    @test slope_boot.attempted == 1
    @test slope_boot.used == 1
    @test slope_boot.failed == 0
    unsupported_predictor = bf(@formula(cbind(successes, failures) ~ x + (1 | g)),
                              @formula(sd(g) ~ x))
    @test_throws ArgumentError drm(unsupported_predictor, Binomial(); data)
    beta_data = merge(data, (; y_beta = clamp.((data.successes .+ 0.5) ./ 13, 0.01, 0.99)))
    # The public Binomial route must not intercept another family's SD model.
    beta_fit = drm(bf(@formula(y_beta ~ x + (1 | g)), @formula(sd(g) ~ zfactor)),
                   Beta(); data = beta_data)
    @test beta_fit.converged
    @test length(predict_parameters(beta_fit, beta_data)[:sd]) == length(beta_data.y_beta)
    unsupported_marginal = bf(@formula(cbind(successes, failures) ~ x + (1 | g)),
                              @formula(sd(g) ~ z))
    # `marginal = :Laplace` now fits this shape (opt-in drmTMB parity route).
    @test drm(unsupported_marginal, Binomial(); data, marginal = :Laplace).marginal === :Laplace
end
