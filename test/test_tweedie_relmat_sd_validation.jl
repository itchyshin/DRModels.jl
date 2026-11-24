using DRModels, Test, LinearAlgebra, Random
using DRModels: Optim
import Distributions

function _tweedie_relmat_validation_fixture()
    G, m = 32, 8
    rng = MersenneTwister(20261026)
    order = randperm(rng, G)
    group_labels = ["g$(i)" for i in order]
    group_id = repeat(1:G; inner = m)
    group = group_labels[group_id]
    z_group = [zeros(G ÷ 2); ones(G ÷ 2)][randperm(rng, G)]
    base_R = 0.45 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    kernel_sd = ifelse.(z_group .== 0, 1.0, 2.0)
    K = kernel_sd .* base_R .* kernel_sd'
    z = z_group[group_id]
    x = rand(rng, G * m) .* 2 .- 1
    latent = cholesky(Symmetric(K)).L * randn(rng, G)
    mu = exp.(0.35 .+ 0.2 .* x .+
              exp.(-0.6 .+ 0.4 .* z_group)[group_id] .* latent[group_id])
    sigma, power = 0.65, 1.5
    phi = sigma^2
    y = map(eachindex(mu)) do i
        lambda = mu[i]^(2 - power) / (phi * (2 - power))
        count = rand(rng, Distributions.Poisson(lambda))
        count == 0 && return 0.0
        shape = count * (2 - power) / (power - 1)
        scale = phi * (power - 1) * mu[i]^(power - 1)
        rand(rng, Distributions.Gamma(shape, scale))
    end
    other = ["h$(mod1(i + 1, G))" for i in group_id]
    data = (; y, x, z, group, other)
    return data, K
end

function _tweedie_relmat_formula(; sigma = @formula(sigma ~ 1),
                                 nu = @formula(nu ~ 1),
                                 mean = @formula(y ~ x + relmat(1 | group)))
    return bf(mean, sigma, nu, @formula(sd(group, relmat) ~ z))
end

@testset "Tweedie relmat rejects uncertifiable latent modes" begin
    # A one-step Newton update solves a quadratic exactly, so use an unbounded
    # objective with no positive-definite mode to exercise the failure path.
    objective = x -> sum(x)
    mode = DRModels._tweedie_relmat_inner_mode(objective, zeros(3);
        g_tol = 1e-14, maxiter = 1)
    @test mode === nothing
end

@testset "Tweedie relmat SD preserves raw kernel scale" begin
    labels = ["a", "b", "c"]
    K = [4.0 1.2 0.4; 1.2 1.0 0.3; 0.4 0.3 2.25]
    Q, group_index, kernel_variance =
        DRModels._tweedie_relmat_cov_setup(K, labels)
    @test group_index == [1, 2, 3]
    @test kernel_variance ≈ diag(K)
    @test Matrix(Q) ≈ inv(K)

    log_sd = log.([0.7, 1.1, 1.8])
    Q_scaled = DRModels._tweedie_relmat_scaled_precision(Q, log_sd)
    covariance = Diagonal(exp.(log_sd)) * K * Diagonal(exp.(log_sd))
    @test Matrix(Q_scaled) ≈ inv(covariance)
    @test logdet(Matrix(Q_scaled)) ≈ logdet(inv(K)) - 2sum(log_sd)
end

@testset "Tweedie relmat SD preserves covariance through missing-row filtering" begin
    data, K = _tweedie_relmat_validation_fixture()
    y_missing = Union{Missing, Float64}[data.y...]
    missing_rows = findall(==(first(data.group)), data.group)
    y_missing[missing_rows] .= missing
    data_missing = merge(data, (; y = y_missing))
    fit = drm(_tweedie_relmat_formula(), Tweedie(); data = data_missing, K,
              g_tol = 1e-4)
    # Estimation uses observed responses only, while fitted scale outputs are
    # rebuilt on the full input design so callers retain row alignment.
    @test nobs(fit) == length(data.y) - length(missing_rows)
    @test length(fit.scales[:sd]) == length(data.y)
    @test all(isnan, fit.obs[:mu][missing_rows])
    @test length(fit.scales[:sd_group]) == length(unique(data.group))
    group_index, _ = DRModels._group_index(data.group)
    @test fit.scales[:marginal_sd] ≈
          fit.scales[:marginal_sd_group][group_index]
    @test fit.scales[:sd_group_index] == Float64.(group_index)
end

@testset "Tweedie relmat SD rejects levels seen only in missing groups before fitting" begin
    data, K = _tweedie_relmat_validation_fixture()
    missing_group = first(data.group)
    missing_rows = findall(==(missing_group), data.group)
    y_missing = Union{Missing, Float64}[data.y...]
    y_missing[missing_rows] .= missing
    sdcat = ifelse.(data.group .== missing_group, "only_missing",
                    ifelse.(data.z .== 0, "low", "high"))
    data_missing = merge(data, (; y = y_missing, sdcat))
    formula = bf(@formula(y ~ x + relmat(1 | group)), @formula(sigma ~ 1),
        @formula(nu ~ 1), @formula(sd(group, relmat) ~ sdcat))

    err = try
        drm(formula, Tweedie(); data = data_missing, K, g_tol = 1e-4)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    message = sprint(showerror, err)
    @test occursin("sdcat", message)
    @test occursin("only_missing", message)
    @test occursin("observed responses", message)
end

@testset "Tweedie relmat covariance follows retained first-seen groups" begin
    data, K = _tweedie_relmat_validation_fixture()
    retained = [i for i in eachindex(data.group) if data.group[i] != first(data.group)]
    data_retained = (y = data.y[retained], x = data.x[retained],
                     z = data.z[retained], group = data.group[retained],
                     other = data.other[retained])
    K_retained = DRModels._tweedie_observed_relmat_K(
        _tweedie_relmat_formula(), data, data_retained, K)
    keep = 2:size(K, 1)
    @test K_retained ≈ K[keep, keep]
end

@testset "Tweedie relmat SD route rejects silently ignored structures" begin
    data, K = _tweedie_relmat_validation_fixture()

    @test_throws ArgumentError drm(_tweedie_relmat_formula(
        sigma = @formula(sigma ~ 0 + x)), Tweedie(); data, K)
    @test_throws ArgumentError drm(_tweedie_relmat_formula(
        nu = @formula(nu ~ 0 + x)), Tweedie(); data, K)
    @test_throws ArgumentError drm(_tweedie_relmat_formula(
        mean = @formula(y ~ x + relmat(1 | group) + relmat(1 | other))),
        Tweedie(); data, K)
end

@testset "Tweedie relmat fit rejects malformed covariance before optimization" begin
    data, K = _tweedie_relmat_validation_fixture()
    wrong_size = K[1:end-1, 1:end-1]
    nonfinite = copy(K)
    nonfinite[1, 1] = NaN
    asymmetric = copy(K)
    asymmetric[1, 2] += 0.1
    non_positive_definite = zeros(size(K))
    non_positive_definite[1, 1] = -1.0

    for invalid_K in (wrong_size, nonfinite, asymmetric, non_positive_definite)
        @test_throws ArgumentError drm(_tweedie_relmat_formula(), Tweedie();
            data, K = invalid_K, g_tol = 1e-4)
    end
end

