using DRModels
using ForwardDiff
using Test

const q2_prior_data = (
    y=[0.75, -0.15, 1.3, 1.05, 0.2, -0.3, 0.1, 0.8, 0.4, -0.6, -0.9, -0.2],
    x=[-1.0, 1.0, -1.0, 0.0, 1.0, -1.0, -0.2, 0.4, 1.0, -1.0, 0.0, 1.0],
    g=[1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4],
)

const q2_lkj_table = (
    prior=["lkj(2)"], class=["correlation"], coef=[""], dpar=["mu"],
    group=["g"], from_coef=["(Intercept)"], to_coef=["x"],
    scale=["natural_correlation"],
)

@testset "natural-correlation LKJ prior-table targets" begin
    rho12 = (prior=["lkj(2)"], class=["correlation"], coef=["(Intercept)"],
             dpar=["rho12"], group=[""], scale=["natural_correlation"])
    slope = (prior=["lkj(2)"], class=["correlation"], coef=[""], dpar=["mu"],
             group=["site"], from_coef=["(Intercept)"], to_coef=["x"],
             scale=["natural_correlation"])

    rr = only(DRModels._parse_prior_table(rho12))
    sr = only(DRModels._parse_prior_table(slope))
    @test rr.class == "correlation"
    @test rr.dpar == "rho12"
    @test rr.coef == "(Intercept)"
    @test rr.scale == "natural_correlation"
    @test rr.arguments == (2.0,)
    @test sr.class == "correlation"
    @test sr.dpar == "mu"
    @test sr.group == "site"
    @test sr.from_coef == "(Intercept)"
    @test sr.to_coef == "x"

    rho_spec = only(DRModels._prior_table_specs([rr], String[]))
    slope_spec = only(DRModels._prior_table_specs([sr], ["site"]))
    @test rho_spec isa DRModels.CorrelationLKJPrior
    @test rho_spec.dpar === :rho12
    @test isempty(rho_spec.group)
    @test slope_spec isa DRModels.CorrelationLKJPrior
    @test slope_spec.dpar === :mu
    @test slope_spec.group == "site"
    @test slope_spec.from_coef == "(Intercept)"
    @test slope_spec.to_coef == "x"

    # Pair orientation in a correlation matrix does not create a new target.
    reversed = only(DRModels._parse_prior_table(merge(slope,
        (from_coef=["x"], to_coef=["(Intercept)"]))))
    @test reversed.from_coef == sr.from_coef
    @test reversed.to_coef == sr.to_coef
    @test_throws ArgumentError DRModels._prior_table_specs([sr, reversed], ["site"])
    # Residual and random-effect correlations are distinct resolved classes.
    @test length(DRModels._prior_table_specs([rr, sr], ["site"])) == 2
end

@testset "random-slope LKJ resolves only the named Gaussian q=2 block" begin
    f = bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1))
    resolved = DRModels.resolve_priors(f, Gaussian(); data=q2_prior_data,
                                       priors=q2_lkj_table)
    corr = only(filter(r -> r.class === :correlation, resolved))
    @test corr.dpar === :mu
    @test corr.group == "g"
    @test !hasproperty(corr, :block)
    @test corr.from_coef == "(Intercept)"
    @test corr.to_coef == "x"
    @test corr.scale === :natural_correlation
    @test corr.guard == DRModels.DRM_Q2_CORRELATION_PRIOR_GUARD == 0.99999999
    constructor_corr = only(DRModels.resolve_priors(f, Gaussian(); data=q2_prior_data,
        priors=drm_priors(drm_prior_lkj(dpar=:mu, eta=2.0, group="g",
            from_coef="(Intercept)", to_coef="x"))))
    @test constructor_corr == corr
    native_corr = only(DRModels._resolve_priors_native(f, Gaussian(); data=q2_prior_data,
        priors=drm_priors(drm_prior_lkj(dpar=:mu, eta=2.0, group="g",
            from_coef="(Intercept)", to_coef="x"))))
    @test native_corr.class === :random_effect_correlation
    @test native_corr.block === :gaussian_q2
    @test DRModels._lkj_log_normalizer(2.0, corr.guard) ≈
          log(2 * (corr.guard - corr.guard^3 / 3)) atol=1e-14
    @test isfinite(DRModels._lkj_log_prior(2.0, prevfloat(corr.guard), corr.guard))
    @test_throws ArgumentError DRModels._lkj_log_prior(2.0, corr.guard, corr.guard)
    @test_throws ArgumentError DRModels._lkj_log_prior(2.0, 0.999999995, corr.guard)

    # A formula with a second slope or a different grouping block is not the
    # same target and must be refused, rather than partially matched.
    f_q3 = bf(@formula(y ~ x + (1 + x + z | g)), @formula(sigma ~ 1))
    data_q3 = merge(q2_prior_data, (z=collect(range(-1, 1; length=12)),))
    @test_throws ArgumentError DRModels.resolve_priors(f_q3, Gaussian();
        data=data_q3, priors=q2_lkj_table)
    @test_throws ArgumentError drm(f_q3, Gaussian();
        data=data_q3, priors=q2_lkj_table)
    @test_throws ArgumentError drm(f_q3, Gaussian(); data=data_q3,
        priors=drm_priors(drm_prior_lkj(dpar=:mu, eta=2.0, group="g",
            from_coef="(Intercept)", to_coef="x")))
end

@testset "random-slope LKJ objective matches the frozen dense oracle" begin
    f = bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1))
    fit = drm(f, Gaussian(); data=q2_prior_data, priors=q2_lkj_table)
    theta_points = [
        [0.1, -0.2, -0.7985076962177716, -0.5978370007556204, -1.2039728043259361, 0.25],
        [-0.15, 0.3, -0.35667494393873245, -1.0498221244986778, -0.6931471805599453, 0.6],
        [0.25, 0.05, -1.0498221244986778, -0.2231435513142097, -1.3862943611198906, -0.4],
        [-0.3, -0.1, -0.5108256237659907, -0.7985076962177716, -0.4307829160924542, 0.15],
        [0.05, 0.25, -0.6931471805599453, -0.35667494393873245, -0.916290731874155, -0.7],
    ]
    oracle_nll = [11.369675682489564, 15.420045845290858, 11.363186045685032,
                  14.085900240080946, 12.321317228672296]
    engine_coords(oracle_theta) = begin
        rho = oracle_theta[6]
        [oracle_theta[1:4]...,
         oracle_theta[5] + 0.5log1p(-rho^2), rho * exp(oracle_theta[5])]
    end
    @test fit.estim_method === :MAP
    @test fit.penalty.kind === :correlation_priors
    @test fit.penalty.correlation_row.block === :gaussian_q2
    @test fit.penalty.correlation_row.from_coef == "(Intercept)"
    @test fit.penalty.correlation_row.to_coef == "x"
    plain = drm(f, Gaussian(); data=q2_prior_data)
    for (oracle_theta, expected) in zip(theta_points, oracle_nll)
        # The dense oracle uses (log sd_intercept, log sd_slope, natural rho);
        # DRModels stores the equivalent lower-Cholesky coordinates.
        engine_theta = engine_coords(oracle_theta)
        @test fit.nll(engine_theta) ≈ expected atol=1e-8
        @test plain.nll(engine_theta) ≈ fit.penalty.data_nll(engine_theta) atol=1e-10

        # Independently finite-difference the package objective in the oracle
        # coordinates; this checks the AD derivatives and covariance cross-terms.
        objective(theta) = fit.nll(engine_coords(theta))
        grad_ad = ForwardDiff.gradient(objective, oracle_theta)
        hess_ad = ForwardDiff.hessian(objective, oracle_theta)
        h = 1e-4
        grad_fd = similar(grad_ad)
        hess_fd = similar(hess_ad)
        for i in eachindex(oracle_theta)
            ei = zeros(length(oracle_theta)); ei[i] = h
            plus = objective(oracle_theta .+ ei)
            minus = objective(oracle_theta .- ei)
            grad_fd[i] = (plus - minus) / (2h)
            hess_fd[i, i] = (plus - 2objective(oracle_theta) + minus) / h^2
            for j in 1:(i - 1)
                ej = zeros(length(oracle_theta)); ej[j] = h
                hess_fd[i, j] = hess_fd[j, i] =
                    (objective(oracle_theta .+ ei .+ ej) -
                     objective(oracle_theta .+ ei .- ej) -
                     objective(oracle_theta .- ei .+ ej) +
                     objective(oracle_theta .- ei .- ej)) / (4h^2)
            end
        end
        @test maximum(abs.(grad_ad .- grad_fd)) <= 1e-6
        @test maximum(abs.(hess_ad .- hess_fd) ./ max.(1, abs.(hess_fd))) <= 1e-5
    end
    @test fit.loglik ≈ -fit.penalty.data_nll(fit.theta) atol=1e-10
    @test fit.penalty.objective ≈ fit.nll(fit.theta) atol=1e-10
    rho = fit.theta[6] / hypot(fit.theta[6], exp(fit.theta[5]))
    expected_log_prior = log1p(-rho^2) - log(4 / 3)
    @test fit.penalty.log_prior ≈ expected_log_prior atol=1e-10

    mixed = drm_priors(
        drm_prior_normal(dpar=:mu, coef="x", mean=0.0, sd=1.0),
        drm_prior_lkj(dpar=:mu, eta=2.0, group="g",
                      from_coef="(Intercept)", to_coef="x"),
    )
    fit_mixed = drm(f, Gaussian(); data=q2_prior_data, priors=mixed)
    @test fit_mixed.penalty.kind === :mixed_priors
    @test fit_mixed.penalty.fixed_penalty > 0
    @test fit_mixed.penalty.correlation_penalty > 0
    theta = engine_coords(theta_points[1])
    @test fit_mixed.nll(theta) ≈ oracle_nll[1] + 0.5 * theta_points[1][2]^2 +
        0.5log(2π) atol=1e-8

    replay = update(fit, f; data=q2_prior_data)
    @test replay.estim_method === :MAP
    @test replay.penalty.specs == fit.penalty.specs
    @test replay.penalty.input_table == q2_lkj_table
    @test replay.theta ≈ fit.theta atol=1e-7
    @test_throws ArgumentError confint(fit)
    @test_throws ArgumentError drm(f, Gaussian(); data=q2_prior_data,
        priors=q2_lkj_table, method=:REML)
    @test_throws ArgumentError drm(f, Gaussian(); data=q2_prior_data,
        priors=q2_lkj_table, weights=fill(2.0, length(q2_prior_data.y)))
    @test_throws ArgumentError drm(f, Gaussian(); data=q2_prior_data,
        priors=q2_lkj_table, V=zeros(length(q2_prior_data.y), length(q2_prior_data.y)))
    bad_finite = merge(q2_prior_data, (x=Union{Missing,Float64}[missing; q2_prior_data.x[2:end]],))
    @test_throws ArgumentError resolve_priors(f, Gaussian(); data=bad_finite,
        priors=q2_lkj_table)
    bad_infinite = merge(q2_prior_data, (x=[Inf; q2_prior_data.x[2:end]],))
    @test_throws ArgumentError resolve_priors(f, Gaussian(); data=bad_infinite,
        priors=q2_lkj_table)
    f_changed_group = bf(@formula(y ~ x + (1 + x | h)), @formula(sigma ~ 1))
    data_changed_group = merge(q2_prior_data, (h=q2_prior_data.g,))
    @test_throws ArgumentError drm(f_changed_group, Gaussian();
        data=data_changed_group, priors=q2_lkj_table)
    @test_throws ArgumentError update(fit, f_changed_group; data=data_changed_group)
    f_multiple = bf(@formula(y ~ x + (1 + x | g) + (1 | h)), @formula(sigma ~ 1))
    data_multiple = merge(q2_prior_data, (h=q2_prior_data.g,))
    @test_throws ArgumentError resolve_priors(f_multiple, Gaussian();
        data=data_multiple, priors=q2_lkj_table)
    f_structured = bf(@formula(y ~ x + phylo(1 + x | g)), @formula(sigma ~ 1))
    @test_throws ArgumentError resolve_priors(f_structured, Gaussian();
        data=q2_prior_data, priors=q2_lkj_table)
    @test_throws ArgumentError resolve_priors(bf(@formula(y ~ x + (1 + x | g))),
        DRModels.Binomial(); data=q2_prior_data, priors=q2_lkj_table)
end

@testset "bivariate residual LKJ resolves only constant rho12" begin
    data = (y1=[0.5, -0.2, 0.9, 1.1], y2=[0.1, 0.4, -0.5, 0.3])
    f = bf(mu1=@formula(y1 ~ 1), mu2=@formula(y2 ~ 1),
           rho12=@formula(rho12 ~ 1))
    table = (prior=["lkj(2)"], class=["correlation"], coef=["(Intercept)"],
             dpar=["rho12"], group=[""], scale=["natural_correlation"])
    resolved = DRModels.resolve_priors(f, Gaussian(); data, priors=table)
    corr = only(filter(r -> r.class === :correlation, resolved))
    @test corr.dpar === :rho12
    @test corr.block === :bivariate_residual
    @test corr.scale === :natural_correlation
    @test corr.guard == DRModels.DRM_RHO12_PRIOR_GUARD
    constructor_resolved = DRModels.resolve_priors(f, Gaussian(); data,
        priors=drm_priors(drm_prior_lkj(dpar=:rho12, eta=2.0)))
    @test only(constructor_resolved) == corr
    @test only(DRModels._resolve_priors_native(f, Gaussian(); data,
        priors=drm_priors(drm_prior_lkj(dpar=:rho12, eta=2.0)))).class === :residual_correlation

    f_varying = bf(mu1=@formula(y1 ~ 1), mu2=@formula(y2 ~ 1),
                   rho12=@formula(rho12 ~ 1 + x))
    data_x = merge(data, (x=[-1.0, -0.1, 0.3, 1.0],))
    @test_throws ArgumentError DRModels.resolve_priors(f_varying, Gaussian();
        data=data_x, priors=table)
end

@testset "bivariate residual LKJ enters the objective once on natural rho" begin
    data = (y1=[0.5, -0.2, 0.9, 1.1], y2=[0.1, 0.4, -0.5, 0.3])
    f = bf(mu1=@formula(y1 ~ 1), mu2=@formula(y2 ~ 1),
           sigma1=@formula(sigma1 ~ 1), sigma2=@formula(sigma2 ~ 1),
           rho12=@formula(rho12 ~ 1))
    table = (prior=["lkj(2)"], class=["correlation"], coef=["(Intercept)"],
             dpar=["rho12"], group=[""], scale=["natural_correlation"])
    fit = drm(f, Gaussian(); data, priors=table)
    @test fit.estim_method === :MAP
    @test fit.penalty.kind === :correlation_priors
    @test fit.penalty.correlation_row.block === :bivariate_residual
    @test fit.loglik ≈ -fit.penalty.data_nll(fit.theta) atol=1e-10
    @test fit.penalty.objective ≈ fit.nll(fit.theta) atol=1e-10

    # Independently evaluate the paired bivariate normal density at interior
    # points, then add one normalized LKJ density on the actual guarded rho.
    guard = fit.penalty.correlation_row.guard
    for theta in ([0.2, -0.1, -0.3, 0.15, 0.25],
                  [-0.2, 0.3, -0.1, -0.4, -0.5],
                  [0.0, 0.0, 0.1, -0.2, 0.0],
                  [0.4, 0.2, -0.5, -0.1, 0.7],
                  [-0.1, -0.2, -0.2, 0.25, -0.3])
        rho = guard * tanh(theta[5])
        om = 1 - rho^2
        data_nll = sum(begin
            z1 = (data.y1[i] - theta[1]) * exp(-theta[3])
            z2 = (data.y2[i] - theta[2]) * exp(-theta[4])
            log(2π) + theta[3] + theta[4] + 0.5log(om) +
                0.5 * (z1^2 - 2rho*z1*z2 + z2^2) / om
        end for i in eachindex(data.y1))
        prior_nll = -log1p(-rho^2) + log(2 * (guard - guard^3 / 3))
        @test fit.nll(theta) ≈ data_nll + prior_nll atol=1e-10
    end
end

@testset "LKJ natural-correlation density has the frozen normalization" begin
    g = 0.99999999
    @test exp(DRModels._lkj_log_normalizer(2.0, g)) ≈ 2 * (g - g^3 / 3) atol=1e-12
    @test exp(DRModels._lkj_log_normalizer(2.0, g)) ≈ 4 / 3 atol=1e-8
    @test DRModels._lkj_log_prior(2.0, 0.0, g) ≈ -log(4 / 3) atol=1e-8
    @test DRModels._lkj_log_prior(2.0, 0.5, g) ≈ log(0.75) - log(4 / 3) atol=1e-8
    @test_throws ArgumentError DRModels._lkj_log_normalizer(1.0, g)
    @test DRModels._lkj_log_normalizer(2.0, 1.0) ≈ log(4 / 3) atol=1e-14
    @test DRModels._lkj_log_prior(2.0, 0.0, 1.0) ≈ -log(4 / 3) atol=1e-14
    @test_throws ArgumentError DRModels._lkj_log_prior(2.0, 1.0, 1.0)
    @test_throws ArgumentError DRModels.drm_prior_lkj(dpar=:mu, eta=1.0,
        group="site", from_coef="(Intercept)", to_coef="x")
    @test_throws ArgumentError DRModels.drm_prior_lkj(dpar=:mu, eta=2.0,
        group="site", from_coef="x", to_coef="z")
    @test_throws ArgumentError DRModels.drm_prior_lkj(dpar=:rho12, eta=2.0,
        group="site")
end

@testset "correlation rows fail closed on malformed targets" begin
    parse_row(; prior="lkj(2)", class="correlation", coef="(Intercept)",
              dpar="rho12", group="", scale="natural_correlation") =
        DRModels._parse_prior_table((; prior=[prior], class=[class], coef=[coef],
                                      dpar=[dpar], group=[group], scale=[scale]))
    @test_throws ArgumentError parse_row(prior="lkj(1)")
    @test_throws ArgumentError parse_row(prior="normal(0,1)")
    @test_throws ArgumentError parse_row(dpar="sigma")
    @test_throws ArgumentError parse_row(coef="x")
    @test_throws ArgumentError parse_row(group="site")
    @test_throws ArgumentError parse_row(scale="link")
    bad_slope = (prior=["lkj(2)"], class=["correlation"], coef=[""], dpar=["mu"],
                 group=["site"], from_coef=["x"], to_coef=["z"],
                 scale=["natural_correlation"])
    @test_throws ArgumentError DRModels._parse_prior_table(bad_slope)
    duplicate = (prior=["lkj(2)", "lkj(2)"], class=["correlation", "correlation"],
                 coef=["", ""], dpar=["mu", "mu"], group=["site", "site"],
                 from_coef=["(Intercept)", "(Intercept)"], to_coef=["x", "x"],
                 scale=["natural_correlation", "natural_correlation"])
    @test_throws ArgumentError DRModels._prior_table_specs(
        DRModels._parse_prior_table(duplicate), ["site"])
end
