using DRModels, Test, Random, QuadGK
import Distributions

_hurdle_sd_logistic(x) = inv(1 + exp(-x))

function _hurdle_sd_fixture(; G = 40, m = 12, seed = 20261021)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x, w = randn(rng, length(g)), randn(rng, length(g))
    alpha = [-0.65, 0.38]
    beta_hu = [-1.25, 0.22]
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    mu = exp.(0.45 .+ 0.25 .* x .+ b[g])
    sigma = 0.45
    size = 1 / sigma^2
    pi0 = _hurdle_sd_logistic.(beta_hu[1] .+ beta_hu[2] .* w)
    y = Float64[]
    for i in eachindex(g)
        if rand(rng) < pi0[i]
            push!(y, 0.0)
        else
            while true
                yi = rand(rng, Distributions.NegativeBinomial(
                    size, size / (size + mu[i])))
                if yi > 0
                    push!(y, Float64(yi))
                    break
                end
            end
        end
    end
    # This group contributes to the fixed hurdle part but contains no positive
    # counts, so it must contribute no information about the mean SD regression.
    y[g .== 1] .= 0.0
    return (; data = (; y, x, w, z, g), alpha, beta_hu)
end

function _hurdle_sd_slope_fixture(; G = 40, m = 14, seed = 20261024)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x, w = randn(rng, G * m), randn(rng, G * m)
    alpha = [-0.65, 0.38]
    beta_x = 0.65
    beta_hu = [-1.25, 0.22]
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    mu = exp.(beta_x .* x .+ x .* b[g])
    sigma = 0.45
    size = 1 / sigma^2
    pi0 = _hurdle_sd_logistic.(beta_hu[1] .+ beta_hu[2] .* w)
    y = Float64[]
    for i in eachindex(g)
        if rand(rng) < pi0[i]
            push!(y, 0.0)
        else
            while true
                yi = rand(rng, Distributions.NegativeBinomial(
                    size, size / (size + mu[i])))
                if yi > 0
                    push!(y, Float64(yi))
                    break
                end
            end
        end
    end
    y[g .== 1] .= 0.0
    return (; data = (; y, x, w, z, g), alpha, beta_x, beta_hu)
end

@testset "Hurdle NB2 random-intercept SD regression" begin
    fixture = _hurdle_sd_fixture()
    data, alpha, beta_hu = fixture.data, fixture.alpha, fixture.beta_hu
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                 @formula(hu ~ w), @formula(sd(g) ~ z))
    fit = drm(formula, NegBinomial2(); data, nAGQ = 21)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :hu, :sd]
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.8
    @test coef(fit, :hu) ≈ beta_hu atol = 0.3
    Zg = hcat(ones(40), repeat([-1.0, 1.0], inner = 20))
    @test fit.scales[:sd_group] ≈ exp.(Zg * coef(fit, :sd))
    @test fit.scales[:sd] ≈ fit.scales[:sd_group][data.g]
    @test fit.scales[:hu] ≈ _hurdle_sd_logistic.(coef(fit, :hu)[1] .+
                                                   coef(fit, :hu)[2] .* data.w)
    predicted = predict_parameters(fit, data)
    @test predicted[:sd] ≈ fit.scales[:sd]
    @test predicted[:hu] ≈ fit.scales[:hu]
    @test length(predict_parameters(fit, data; se = true)[:sd].se) == length(data.y)

    # Independent groupwise normal integration. The oracle uses Distributions'
    # NB2 pmf and QuadGK, not the implementation's likelihood or AGHQ helper.
    theta = [0.45, 0.25, log(0.45), beta_hu..., -0.65, 0.35]
    r = exp(-2 * theta[3])
    reference = 0.0
    normal = Distributions.Normal()
    for gi in 1:40
        rows = findall(==(gi), data.g)
        tau = exp(theta[6] + theta[7] * data.z[first(rows)])
        integrand = u -> begin
            ll = sum(rows) do i
                eta_hu = theta[4] + theta[5] * data.w[i]
                pi0 = _hurdle_sd_logistic(eta_hu)
                if iszero(data.y[i])
                    log(pi0)
                else
                    mu_i = exp(theta[1] + theta[2] * data.x[i] + tau * u)
                    nb = Distributions.NegativeBinomial(r, r / (r + mu_i))
                    log1p(-pi0) + Distributions.logpdf(nb, Int(data.y[i])) -
                        log1p(-Distributions.pdf(nb, 0))
                end
            end
            exp(ll) * Distributions.pdf(normal, u)
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-11)
        reference += log(value)
    end
    @test isapprox(-fit.nll(theta), reference; atol = 3e-3)

    # As the latent SD tends to zero, the marginal hurdle likelihood must
    # reduce to the existing fixed-effect hurdle NB2 likelihood.
    fixed = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(hu ~ w)),
                NegBinomial2(); data)
    near_zero_sd = [coef(fixed, :mu); coef(fixed, :sigma); coef(fixed, :hu); -20.0; 0.0]
    @test -fit.nll(near_zero_sd) ≈ loglik(fixed) atol = 2e-5

    # Marginal bootstrap draws must redraw the mean intercept and retain both
    # hurdle zeros and positive NB2 counts. One refit also checks nAGQ routing.
    marginal_sim = DRModels._marginal_simulator(fit, data)
    ysim = marginal_sim(MersenneTwister(20261022))
    @test length(ysim) == length(data.y)
    @test any(iszero, ysim) && any(>(0), ysim)
    @test DRModels._bootstrap_refit_kwargs(fit).nAGQ == 21
    boot = bootstrap_result(fit; data, B = 1, rng = MersenneTwister(20261023),
                            failures = :skip)
    @test boot.attempted == 1 && boot.used == 1 && boot.failed == 0

end

@testset "Hurdle NB2 independent random-slope SD regression" begin
    fixture = _hurdle_sd_slope_fixture()
    data, alpha, beta_x, beta_hu = fixture.data, fixture.alpha,
        fixture.beta_x, fixture.beta_hu
    formula = bf(@formula(y ~ 0 + x + (0 + x | g)), @formula(sigma ~ 1),
                 @formula(hu ~ w), @formula(sd(g, coef(x)) ~ z))
    fit = drm(formula, NegBinomial2(); data, nAGQ = 21)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :hu, :sd]
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.8
    @test coef(fit, :mu)[1] ≈ beta_x atol = 0.25
    @test coef(fit, :hu) ≈ beta_hu atol = 0.3
    Zg = hcat(ones(40), repeat([-1.0, 1.0], inner = 20))
    @test fit.scales[:sd_group] ≈ exp.(Zg * coef(fit, :sd))
    @test fit.scales[:sd] ≈ fit.scales[:sd_group][data.g]
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    @test length(predict_parameters(fit, data; se = true)[:sd].se) == length(data.y)
    bad_selector = bf(@formula(y ~ 0 + x + (0 + x | g)),
        @formula(sigma ~ 1), @formula(hu ~ w),
        @formula(sd(g, coef(w)) ~ z))
    @test_throws ArgumentError drm(bad_selector, NegBinomial2(); data)
    varying_group_predictor = copy(data.z)
    varying_group_predictor[1] += 0.1
    bad_data = merge(data, (; z = varying_group_predictor))
    @test_throws ArgumentError drm(formula, NegBinomial2(); data = bad_data, nAGQ = 21)

    # Independent scalar normal integral. This likelihood oracle uses the
    # Distributions NB2 PMF and QuadGK, not the package's AGHQ helper.
    theta = [beta_x, log(0.45), beta_hu..., alpha...]
    r = exp(-2 * theta[2])
    reference = 0.0
    normal = Distributions.Normal()
    for gi in 1:40
        rows = findall(==(gi), data.g)
        tau = exp(theta[5] + theta[6] * data.z[first(rows)])
        integrand = u -> begin
            ll = sum(rows) do i
                eta_hu = theta[3] + theta[4] * data.w[i]
                pi0 = _hurdle_sd_logistic(eta_hu)
                if iszero(data.y[i])
                    log(pi0)
                else
                    mu_i = exp(theta[1] * data.x[i] + data.x[i] * tau * u)
                    nb = Distributions.NegativeBinomial(r, r / (r + mu_i))
                    log1p(-pi0) + Distributions.logpdf(nb, Int(data.y[i])) -
                        log1p(-Distributions.pdf(nb, 0))
                end
            end
            exp(ll) * Distributions.pdf(normal, u)
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-11)
        reference += log(value)
    end
    @test isapprox(-fit.nll(theta), reference; atol = 3e-3)

    # A constant SD design reproduces the same slope likelihood, and the
    # zero-SD limit reduces to the fixed-effect hurdle model.
    constant_formula = bf(@formula(y ~ 0 + x + (0 + x | g)),
        @formula(sigma ~ 1), @formula(hu ~ w),
        @formula(sd(g, coef(x)) ~ 1))
    constant_fit = drm(constant_formula, NegBinomial2(); data, nAGQ = 21)
    @test constant_fit.converged
    @test all(isapprox.(constant_fit.scales[:sd_group],
                        fill(constant_fit.scales[:sd_group][1], 40); atol = 1e-8))
    constant_z_formula = bf(@formula(y ~ 0 + x + (0 + x | g)),
        @formula(sigma ~ 1), @formula(hu ~ w),
        @formula(sd(g, coef(x)) ~ 1 + z))
    constant_z_fit = drm(constant_z_formula, NegBinomial2(); data, nAGQ = 21)
    theta_no_z_slope = [coef(constant_fit, :mu); coef(constant_fit, :sigma);
                        coef(constant_fit, :hu); coef(constant_fit, :sd)[1]; 0.0]
    @test -constant_z_fit.nll(theta_no_z_slope) ≈ loglik(constant_fit) atol = 2e-5
    fixed = drm(bf(@formula(y ~ 0 + x), @formula(sigma ~ 1),
                    @formula(hu ~ w)), NegBinomial2(); data)
    near_zero_sd = [coef(fixed, :mu); coef(fixed, :sigma); coef(fixed, :hu); -20.0]
    @test -constant_fit.nll(near_zero_sd) ≈ loglik(fixed) atol = 2e-5

    marginal_sim = DRModels._marginal_simulator(fit, data)
    simulated = marginal_sim(MersenneTwister(20261025))
    @test length(simulated) == length(data.y)
    @test any(iszero, simulated) && any(>(0), simulated)
    zero_loading_data = merge(data, (; x = zeros(length(data.x))))
    unit_loading_data = merge(data, (; x = ones(length(data.x))))
    zero_sim = DRModels._marginal_simulator(fit, zero_loading_data)(MersenneTwister(20261026))
    unit_sim = DRModels._marginal_simulator(fit, unit_loading_data)(MersenneTwister(20261026))
    @test zero_sim != unit_sim
    @test DRModels._bootstrap_refit_kwargs(fit).nAGQ == 21
    boot = bootstrap_result(fit; data, B = 1, rng = MersenneTwister(20261027),
                            failures = :skip)
    @test boot.attempted == 1 && boot.used == 1 && boot.failed == 0
end
