using DRModels
using Test
using ForwardDiff
using LinearAlgebra

# This specification deliberately loads the public module only.  The module
# include is owned by the integration lane, so this file must fail until that
# lane wires the finite-state kernel.

_finite_logpdf_normal(y, mean, sd) = -0.5 * log(2π) - log(sd) - 0.5 * ((y - mean) / sd)^2
_finite_lse(values) = (m = maximum(values); m + log(sum(exp.(values .- m))))
_finite_logsigmoid(z) = z >= 0 ? -log1p(exp(-z)) : z - log1p(exp(z))

function _finite_logdiffexp(a, b)
    a >= b || throw(ArgumentError("log difference requires a >= b"))
    a == b && return -Inf
    return a + log1p(-exp(b - a))
end

function _finite_ordinal_logprobabilities(eta, cutraw)
    cuts = similar(cutraw)
    cuts[1] = cutraw[1]
    for k in 2:length(cuts)
        cuts[k] = cuts[k - 1] + exp(cutraw[k])
    end
    K = length(cuts) + 1
    output = similar(cuts, K)
    output[1] = _finite_logsigmoid(cuts[1] - eta)
    for k in 2:(K - 1)
        output[k] = _finite_logdiffexp(_finite_logsigmoid(cuts[k] - eta),
                                        _finite_logsigmoid(cuts[k - 1] - eta))
    end
    output[K] = _finite_logsigmoid(eta - cuts[end])
    return output
end

function _finite_categorical_logprobabilities(eta)
    logits = vcat(zero(eltype(eta)), eta)
    return logits .- _finite_lse(logits)
end

function _finite_central_gradient(f, theta; h = 1e-6)
    output = similar(theta)
    for j in eachindex(theta)
        step = h * max(1.0, abs(theta[j]))
        plus, minus = copy(theta), copy(theta)
        plus[j] += step
        minus[j] -= step
        output[j] = (f(plus) - f(minus)) / (2step)
    end
    return output
end

"""Four masks in prepared order: complete, x-missing/y-observed, x-observed/y-missing, both missing."""
function _finite_fixture(predictor::Symbol; levels = ["low", "middle", "high"])
    n, K = 4, 3
    y = Union{Missing,Float64}[0.75, -0.45, missing, missing]
    x = Union{Missing,String}["middle", missing, "high", missing]
    z = [-0.7, 0.25, 0.8, -0.35]
    Xstate = zeros(n, K, 2)
    score = [-0.8, 0.15, 1.1]
    for i in 1:n, k in 1:K
        Xstate[i, k, 1] = 1.0
        Xstate[i, k, 2] = score[k] + 0.12 * z[i]
    end
    Xsigma = ones(n, 1)
    Xpredictor = reshape(copy(z), n, 1)
    model = DRModels.prepared_joint_model(y, x, Xstate, Xsigma, Xpredictor;
        predictor = predictor, levels = levels, variable = :severity,
        mu_names = ["(Intercept)", "state_score"], sigma_names = ["(Intercept)"],
        predictor_names = ["z"], original_row = [31, 7, 88, 19])
    theta = predictor === :ordinal ?
        [0.2, 0.65, log(0.85), 0.45, -0.35, log(1.4)] :
        [0.2, 0.65, log(0.85), -0.4, 0.6]
    return model, theta
end

"""Identified fit fixture; keep the four-row fixture above pure-only."""
function _finite_fit_fixture(predictor::Symbol)
    n, K = 28, 3
    z = collect(range(-1.0, 1.0; length = n))
    x = Union{Missing,String}[i % 3 == 1 ? "low" : i % 3 == 2 ? "middle" : "high" for i in 1:n]
    y = Union{Missing,Float64}[0.25 + 0.55 * z[i] +
        (x[i] == "low" ? -0.55 : x[i] == "middle" ? 0.05 : 0.7) + 0.08 * sin(i) for i in 1:n]
    for i in (5, 12, 21); x[i] = missing; end
    for i in (4, 12, 25); y[i] = missing; end
    Xstate = zeros(n, K, 2)
    for i in 1:n, k in 1:K
        Xstate[i, k, 1] = 1.0
        Xstate[i, k, 2] = (-0.7 + 0.8 * (k - 1)) + 0.2 * z[i]
    end
    return DRModels.prepared_joint_model(y, x, Xstate, ones(n, 1), reshape(z, n, 1);
        predictor = predictor, levels = ["low", "middle", "high"], variable = :severity,
        original_row = collect(501:(500 + n)))
end

"""Construct a fixed-theta fit for uncertainty-status tests, without an optimizer claim."""
function _finite_synthetic_fit(model, theta; covariance_status::Symbol = :observed_information_inverse)
    frozen = DRModels._finite_joint_copy_model(model)
    theta = Float64.(theta)
    moments = DRModels.prepared_joint_conditional_moments(frozen, theta)
    nll = t -> DRModels.prepared_joint_nll(frozen, t)
    p = length(theta)
    covariance = covariance_status === :observed_information_inverse ? Matrix{Float64}(I, p, p) : fill(NaN, p, p)
    blocks, names = DRModels._finite_joint_blocks(frozen)
    ranges = DRModels._finite_joint_ranges(frozen)
    state_means = DRModels._finite_joint_state_means(frozen, @view theta[ranges.beta])
    mu = [sum(moments.probabilities[i, k] * state_means[i, k] for k in eachindex(frozen.levels)) for i in eachindex(frozen.y)]
    base = DRModels.DrmFit(DRModels.PreparedJointGaussian(), blocks, names, copy(theta), covariance, -nll(theta),
        count(frozen.observed_y), true, Dict{Symbol,Vector{Float64}}(:mu => Float64.(mu)),
        Dict{Symbol,Vector{Float64}}(:mu => Float64[v === missing ? NaN : v for v in frozen.y]),
        Dict{Symbol,Vector{Float64}}(:sigma => exp.(frozen.Xsigma * theta[ranges.delta])))
    base = DRModels._withnll(base, nll)
    metadata = DRModels.JointFiniteMissingMetadata(frozen.predictor, frozen.variable, copy(frozen.levels),
        copy(frozen.original_row), copy(frozen.observed_y), copy(frozen.observed_x),
        Float64.(moments.probabilities), Float64.(moments.mean), Float64.(moments.variance), copy(moments.status),
        length(frozen.y), count(i -> !frozen.observed_y[i] && frozen.observed_x[i], eachindex(frozen.y)),
        :not_computed, :converged, covariance_status)
    return DRModels.PreparedFiniteJointFit(base, frozen, metadata)
end

function _finite_reference(model, theta)
    p, r = size(model.X_mu_state, 3), size(model.Xsigma, 2)
    beta = theta[1:p]
    delta = theta[(p + 1):(p + r)]
    alpha = theta[(p + r + 1):end]
    n, K = length(model.y), length(model.levels)
    logprob = Matrix{Float64}(undef, n, K)
    for i in 1:n
        q = size(model.Xpredictor, 2)
        if model.predictor === :ordinal
            eta = q == 0 ? 0.0 : dot(@view(model.Xpredictor[i, :]), @view(alpha[1:q]))
            logprob[i, :] .= _finite_ordinal_logprobabilities(eta, alpha[(q + 1):end])
        else
            eta = [dot(@view(model.Xpredictor[i, :]), @view(alpha[((k - 2) * q + 1):((k - 1) * q)]))
                   for k in 2:K]
            logprob[i, :] .= _finite_categorical_logprobabilities(eta)
        end
    end
    row = zeros(n)
    posterior = Matrix{Float64}(undef, n, K)
    for i in 1:n
        state_means = [dot(@view(model.X_mu_state[i, k, :]), beta) for k in 1:K]
        sigma = exp(dot(@view(model.Xsigma[i, :]), delta))
        if model.observed_x[i]
            k = model.x[i]
            row[i] = logprob[i, k] + (model.observed_y[i] ?
                _finite_logpdf_normal(model.y[i], state_means[k], sigma) : 0.0)
            posterior[i, :] .= 0.0
            posterior[i, k] = 1.0
        elseif model.observed_y[i]
            logweight = logprob[i, :] .+ [_finite_logpdf_normal(model.y[i], state_means[k], sigma) for k in 1:K]
            row[i] = _finite_lse(logweight)
            posterior[i, :] .= exp.(logweight .- row[i])
        else
            row[i] = 0.0
            posterior[i, :] .= exp.(logprob[i, :])
        end
    end
    return row, posterior
end

@testset "finite-state prepared joint API" begin
    @test isdefined(DRModels, :PreparedFiniteJointModel)
    @test isdefined(DRModels, :PreparedFiniteJointFit)
    @test isdefined(DRModels, :prepared_joint_model)
    @test isdefined(DRModels, :prepared_joint_rowloglik)
    @test isdefined(DRModels, :prepared_joint_conditional_moments)
    @test isdefined(DRModels, :fit_prepared_joint)
    @test isdefined(DRModels, :joint_missing_summary)
end

@testset "ordinal finite-state likelihood, all masks, and posterior moments" begin
    model, theta = _finite_fixture(:ordinal)
    @test model.original_row == [31, 7, 88, 19]
    @test model.levels == ["low", "middle", "high"]
    @test model.variable === :severity
    @test model.observed_y == BitVector([true, true, false, false])
    @test model.observed_x == BitVector([true, false, true, false])
    @test DRModels._finite_joint_ntheta(model) == length(theta)

    expected_row, expected_probabilities = _finite_reference(model, theta)
    @test DRModels.prepared_joint_rowloglik(model, theta) ≈ expected_row atol = 1e-12
    @test DRModels.prepared_joint_nll(model, theta) ≈ -sum(expected_row) atol = 1e-12
    @test expected_row[4] == 0.0

    moments = DRModels.prepared_joint_conditional_moments(model, theta)
    @test size(moments.probabilities) == (4, 3)
    @test moments.probabilities ≈ expected_probabilities atol = 1e-12
    @test vec(sum(moments.probabilities; dims = 2)) ≈ ones(4) atol = 1e-12
    expected_mean = expected_probabilities * [1.0, 2.0, 3.0]
    expected_variance = [sum(((1:3) .- expected_mean[i]).^2 .* expected_probabilities[i, :]) for i in 1:4]
    @test moments.mean ≈ expected_mean atol = 1e-12
    @test moments.variance ≈ expected_variance atol = 1e-12
    @test moments.status == [:observed, :ordinal_posterior, :observed, :predictor_only]
    @test moments.mean[1] == 2.0
    @test moments.variance[1] == 0.0
    # Fitted response means must average state-specific rows, never substitute
    # the expected ordinal score into a state contrast.
    weighted_mu = [sum(expected_probabilities[i, k] *
        dot(@view(model.X_mu_state[i, k, :]), theta[1:2]) for k in 1:3) for i in 1:4]
    @test weighted_mu[2] != dot(@view(model.X_mu_state[2, 2, :]), theta[1:2])
end

@testset "categorical finite-state likelihood, label mapping, and no metric SE" begin
    model, theta = _finite_fixture(:categorical)
    expected_row, expected_probabilities = _finite_reference(model, theta)
    @test DRModels.prepared_joint_rowloglik(model, theta) ≈ expected_row atol = 1e-12
    moments = DRModels.prepared_joint_conditional_moments(model, theta)
    @test moments.probabilities ≈ expected_probabilities atol = 1e-12
    @test moments.mean == [2.0, 1.0, 3.0, 2.0]
    @test all(isnan, moments.variance)
    @test moments.status == [:observed, :categorical_posterior, :observed, :predictor_only]
    categorical_blocks, categorical_names = DRModels._finite_joint_blocks(model)
    @test Dict(categorical_blocks)[:mi_severity] == 4:5
    @test Dict(categorical_names)[:mi_severity] == ["middle:z", "high:z"]
    tied = copy(theta); tied[4:5] .= 0.0
    @test DRModels.prepared_joint_conditional_moments(model, tied).mean[4] == 1.0 # first maximum wins

    # Rebase the nominal logits after a level permutation.  The scientific
    # likelihood is invariant and posterior columns follow the declared order.
    perm = [3, 1, 2]
    permuted_xstate = model.X_mu_state[:, perm, :]
    permuted_levels = model.levels[perm]
    permuted = DRModels.prepared_joint_model(model.y, Union{Missing,String}["middle", missing, "high", missing],
        permuted_xstate, model.Xsigma, model.Xpredictor; predictor = :categorical,
        levels = permuted_levels, variable = :severity, mu_names = model.mu_names,
        sigma_names = model.sigma_names, predictor_names = model.predictor_names,
        original_row = model.original_row)
    permuted_theta = [theta[1:3]; -theta[5]; theta[4] - theta[5]]
    @test DRModels.prepared_joint_rowloglik(permuted, permuted_theta) ≈ expected_row atol = 1e-12
    permuted_moments = DRModels.prepared_joint_conditional_moments(permuted, permuted_theta)
    @test permuted_moments.probabilities[:, invperm(perm)] ≈ moments.probabilities atol = 1e-12
end

@testset "finite-state stable derivatives and validation" begin
    for predictor in (:ordinal, :categorical)
        model, theta = _finite_fixture(predictor)
        nll = t -> DRModels.prepared_joint_nll(model, t)
        @test ForwardDiff.gradient(nll, theta) ≈ _finite_central_gradient(nll, theta) atol = 1e-6 rtol = 1e-6
        H = ForwardDiff.hessian(nll, theta)
        direction = collect(range(-0.3, 0.4; length = length(theta)))
        h = 1e-5
        directional_fd = (ForwardDiff.gradient(nll, theta .+ h .* direction) -
                          ForwardDiff.gradient(nll, theta .- h .* direction)) ./ (2h)
        @test H * direction ≈ directional_fd atol = 1e-5 rtol = 1e-5
    end

    ordinal, theta = _finite_fixture(:ordinal)
    extreme = copy(theta); extreme[4] = 1000.0; extreme[5] = 0.0; extreme[6] = log(1e-12)
    @test all(isfinite, DRModels.prepared_joint_rowloglik(ordinal, extreme))
    extreme_moments = DRModels.prepared_joint_conditional_moments(ordinal, extreme)
    @test all(isfinite, extreme_moments.probabilities)
    @test vec(sum(extreme_moments.probabilities; dims = 2)) ≈ ones(4) atol = 1e-12
    extreme_low = copy(extreme); extreme_low[4] = -1000.0
    @test all(isfinite, DRModels.prepared_joint_rowloglik(ordinal, extreme_low))
    @test all(isfinite, ForwardDiff.gradient(t -> DRModels.prepared_joint_nll(ordinal, t), extreme_low))
    tiny_raw_spacing = copy(theta); tiny_raw_spacing[6] = -1000.0
    @test all(isfinite, DRModels.prepared_joint_rowloglik(ordinal, tiny_raw_spacing))
    @test all(isfinite, ForwardDiff.gradient(t -> DRModels.prepared_joint_nll(ordinal, t), tiny_raw_spacing))
    large_spacing = copy(theta); large_spacing[6] = 10.0
    @test all(isfinite, DRModels.prepared_joint_rowloglik(ordinal, large_spacing))

    # `cut2 == cut1` after Float64 rounding here; the raw spacing still gives
    # the middle state positive mass when its log difference is evaluated stably.
    saturated = copy(theta); saturated[4] = 0.0; saturated[5] = 30.0; saturated[6] = log(1e-16)
    saturated_moments = DRModels.prepared_joint_conditional_moments(ordinal, saturated)
    @test all(isfinite, saturated_moments.probabilities)
    @test saturated_moments.probabilities[4, 2] > 0.0

    # A response can dominate an initially tiny state probability: posterior
    # enumeration must use log weights rather than underflowing probabilities.
    dominant = DRModels.prepared_joint_model([44.0], Union{Missing,String}[missing],
        reshape([1.0, 1.0, 1.0, -1.0, 0.0, 1.0], 1, 3, 2), ones(1, 1), zeros(1, 0);
        predictor = :ordinal, levels = ["low", "middle", "high"], variable = :severity)
    dominant_theta = [0.0, 40.0, log(0.1), 0.0, log(40.0)]
    @test DRModels.prepared_joint_conditional_moments(dominant, dominant_theta).probabilities[1, 3] > 1 - 1e-10

    n = 3
    q0_state = zeros(n, 3, 2)
    q0_state[:, :, 1] .= 1.0
    q0_state[:, :, 2] .= reshape([-0.5, 0.0, 0.5], 1, 3)
    q0 = DRModels.prepared_joint_model([1.0, missing, -0.2], Union{Missing,String}["low", missing, "high"],
        q0_state, ones(n, 1), zeros(n, 0);
        predictor = :ordinal, levels = ["low", "middle", "high"], variable = :severity)
    @test DRModels._finite_joint_ntheta(q0) == 5 # beta(2), delta(1), cutraw(2); ordinal permits q = 0.
    @test isfinite(DRModels.prepared_joint_nll(q0, zeros(5)))
    @test_throws ArgumentError DRModels.prepared_joint_model(ordinal.y, ["unknown", missing, "high", missing],
        ordinal.X_mu_state, ordinal.Xsigma, ordinal.Xpredictor; predictor = :ordinal,
        levels = ordinal.levels, variable = :severity)
    @test_throws ArgumentError DRModels.prepared_joint_model(ordinal.y, ordinal.x, ordinal.X_mu_state,
        ordinal.Xsigma, zeros(4, 0); predictor = :categorical, levels = ordinal.levels, variable = :severity)
    @test_throws ArgumentError DRModels.prepared_joint_model(ordinal.y, ordinal.x, ordinal.X_mu_state,
        ordinal.Xsigma, ordinal.Xpredictor; predictor = :ordinal, levels = ["low", "low", "high"], variable = :severity)
    declared_x = Union{Missing,String}["middle", missing, "high", missing]
    intercept_error = try
        DRModels.prepared_joint_model(ordinal.y, declared_x, ordinal.X_mu_state, ordinal.Xsigma, ones(4, 1);
            predictor = :ordinal, levels = ordinal.levels, variable = :severity)
        nothing
    catch err
        err
    end
    @test intercept_error isa ArgumentError
    @test occursin("span an intercept", sprint(showerror, intercept_error))
    @test_throws ArgumentError DRModels.prepared_joint_model(ordinal.y, declared_x, ordinal.X_mu_state,
        ordinal.Xsigma, hcat([1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]);
        predictor = :ordinal, levels = ordinal.levels, variable = :severity)
    @test_throws ArgumentError DRModels.prepared_joint_rowloglik(ordinal, theta[1:end-1])
    bad = copy(theta); bad[1] = Inf
    @test_throws ArgumentError DRModels.prepared_joint_nll(ordinal, bad)
end

@testset "finite-state fit snapshot, covariance status, and copy isolation" begin
    n, K = 28, 3
    z = collect(range(-1.0, 1.0; length = n))
    x = Union{Missing,String}[i % 3 == 1 ? "low" : i % 3 == 2 ? "middle" : "high" for i in 1:n]
    y = Union{Missing,Float64}[0.25 + 0.55 * z[i] + (x[i] == "low" ? -0.55 : x[i] == "middle" ? 0.05 : 0.7) +
        0.08 * sin(i) for i in 1:n]
    for i in (5, 12, 21); x[i] = missing; end
    for i in (4, 12, 25); y[i] = missing; end
    Xstate = zeros(n, K, 2)
    for i in 1:n, k in 1:K
        Xstate[i, k, 1] = 1.0
        Xstate[i, k, 2] = (-0.7 + 0.8 * (k - 1)) + 0.2 * z[i]
    end
    model = DRModels.prepared_joint_model(y, x, Xstate, ones(n, 1), reshape(z, n, 1);
        predictor = :ordinal, levels = ["low", "middle", "high"], variable = :severity,
        original_row = collect(501:(500 + n)))
    fitted = DRModels.fit_prepared_joint(model; g_tol = 1e-7)
    @test fitted isa DRModels.PreparedFiniteJointFit
    @test fitted.fit.nobs == count(model.observed_y)
    @test isfinite(fitted.fit.loglik)
    @test fitted.metadata.optimizer_status in (:converged, :not_converged)
    @test fitted.metadata.covariance_status in (:observed_information_inverse, :hessian_unavailable, :hessian_not_positive_definite)
    @test size(fitted.fit.vcov) == (length(fitted.fit.theta), length(fitted.fit.theta))
    blocks, names = Dict(fitted.fit.blocks), Dict(fitted.fit.coefnames)
    @test blocks[:rawcut_severity] == (length(fitted.fit.theta) - 1):length(fitted.fit.theta)
    @test blocks[:mi_severity] == 4:4
    @test names[:rawcut_severity] == ["cut1", "log_spacing2"]
    gradient = ForwardDiff.gradient(fitted.fit.nll, fitted.fit.theta)
    fitted.fit.converged && @test norm(gradient, Inf) < 1e-5
    before = fitted.fit.nll(copy(fitted.fit.theta))
    model.X_mu_state[1, 1, 1] = 1e6
    model.y[1] = -1e6
    @test fitted.fit.nll(fitted.fit.theta) == before
    summary = DRModels.joint_missing_summary(fitted)
    summary.original_row[1] = -1
    @test fitted.metadata.original_row[1] == 501
    @test size(summary.conditional_probabilities) == (n, K)
    expected_fitted = [sum(summary.conditional_probabilities[i, k] *
        dot(@view(fitted.prepared.X_mu_state[i, k, :]), fitted.fit.theta[1:2]) for k in 1:K) for i in 1:n]
    @test fitted.fit.means[:mu] ≈ expected_fitted atol = 1e-10
end

@testset "finite-state imputation tables retain ordinal SD and categorical refusal" begin
    ordinal_model, ordinal_theta = _finite_fixture(:ordinal)
    categorical_model, categorical_theta = _finite_fixture(:categorical)
    ordinal_fit = _finite_synthetic_fit(ordinal_model, ordinal_theta)
    categorical_fit = _finite_synthetic_fit(categorical_model, categorical_theta)
    ordinal_table = DRModels.imputed(ordinal_fit; rows = :all, se = false)
    @test ordinal_table.variable == fill("severity", length(ordinal_model.y))
    @test ordinal_table.original_row == ordinal_model.original_row
    @test ordinal_table.source[.!ordinal_model.observed_x] == fill("conditional_expected_score", count(.!ordinal_model.observed_x))
    @test all(ismissing, ordinal_table.std_error)
    categorical_table = DRModels.imputed(categorical_fit; rows = :all)
    @test all(ismissing, categorical_table.std_error)
    @test categorical_table.source[.!categorical_model.observed_x] == fill("conditional_modal_category", count(.!categorical_model.observed_x))
    if categorical_fit.metadata.covariance_status === :observed_information_inverse
        missing_rows = .!categorical_model.observed_x
        @test all(==("route_conditional_se_unavailable"), categorical_table.uncertainty_status[missing_rows])
        @test all(==("ok"), categorical_table.uncertainty_status[.!missing_rows])
    else
        @test all(!=("route_conditional_se_unavailable"), categorical_table.uncertainty_status)
    end
    failed = _finite_synthetic_fit(ordinal_model, ordinal_theta; covariance_status = :hessian_unavailable)
    failed_table = DRModels.imputed(failed; rows = :all)
    @test all(==("sdreport_failed"), failed_table.uncertainty_status)
    @test all(ismissing, failed_table.std_error)
    @test_throws ArgumentError DRModels.imputed(categorical_fit; rows = :elsewhere)
end

println("S9_FINITE_JOINT_RED_TO_GREEN")
