using DRModels, Test, Random, LinearAlgebra
import Distributions

function _poisson_animal_q2_fixture(; G = 48, m = 24, seed = 202611071)
    rng = MersenneTwister(seed)
    ids = ["animal_$(i)" for i in randperm(rng, G)]
    R = 0.35 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    scale = collect(range(0.7, 1.4; length = G))
    A = Diagonal(scale) * R * Diagonal(scale)
    group = repeat(ids, inner = m)
    gidx = repeat(1:G, inner = m)
    z0group = ifelse.(randperm(rng, G) .<= G ÷ 2, -1.0, 1.0)
    z1group = collect(range(-1.0, 1.0; length = G))[randperm(rng, G)]
    x = repeat(collect(range(-1.2, 1.2; length = m)), outer = G)
    alpha0 = [-0.9, 0.35]
    alpha1 = [-1.0, 0.45]
    L = cholesky(Symmetric(A)).L
    u0, u1 = L * randn(rng, G), L * randn(rng, G)
    eta = 0.2 .+ 0.3 .* x .+
        (exp.(alpha0[1] .+ alpha0[2] .* z0group) .* u0)[gidx] .+
        x .* (exp.(alpha1[1] .+ alpha1[2] .* z1group) .* u1)[gidx]
    y = Float64.([rand(rng, Distributions.Poisson(exp(v))) for v in eta])
    return (; data = (; y, x, z0 = z0group[gidx], z1 = z1group[gidx], g = group),
            A, gidx, z0group, z1group, alpha0, alpha1)
end

function _poisson_animal_q2_dense_laplace(theta, y, Xmu, x, Z0, Z1, gidx, A)
    G = size(A, 1)
    Q = inv(Symmetric(A))
    pmu = size(Xmu, 2)
    beta = theta[1:pmu]
    p = size(Z0, 2)
    a0 = theta[(pmu + 1):(pmu + p)]
    a1 = theta[(pmu + p + 1):end]
    eta0 = Xmu * beta
    load0 = exp.(Z0 * a0)
    load1 = x .* exp.(Z1 * a1)
    function joint(b)
        b0, b1 = @view(b[1:G]), @view(b[(G + 1):(2G)])
        value = 0.5 * dot(b0, Q * b0) + 0.5 * dot(b1, Q * b1)
        for i in eachindex(y)
            eta = eta0[i] + load0[i] * b0[gidx[i]] + load1[i] * b1[gidx[i]]
            value -= Distributions.logpdf(Distributions.Poisson(exp(eta)), Int(y[i]))
        end
        value
    end
    result = DRModels.Optim.optimize(joint, zeros(2G), DRModels.Optim.LBFGS(),
        DRModels.Optim.Options(g_tol = 1e-10, iterations = 500);
        autodiff = :forward)
    mode = DRModels.Optim.minimizer(result)
    hessian = DRModels.ForwardDiff.hessian(joint, mode)
    Qjoint = [Matrix(Q) zeros(G, G); zeros(G, G) Matrix(Q)]
    return joint(mode) + 0.5 * logdet(Symmetric(hessian)) -
           0.5 * logdet(Symmetric(Qjoint))
end

@testset "Poisson animal independent intercept and slope SD regression" begin
    fixture = _poisson_animal_q2_fixture()
    data = fixture.data
    formula = bf(@formula(y ~ x + animal(1 + x | g)),
        @formula(sd(g, animal, coef(intercept)) ~ z0),
        @formula(sd(g, animal, coef(x)) ~ z1))
    targets = DRModels._sd_target_parts(formula)
    @test length(targets) == 2

    fit = drm(formula, Poisson(); data, A = fixture.A)
    @test fit.converged
    ordered = [only(filter(t -> t.target.member === m, targets))
               for m in (:intercept, :x)]
    blocks = [DRModels._sd_target_block(item.target; disambiguate = true)
              for item in ordered]
    @test first.(fit.blocks[2:3]) == blocks
    @test abs(coef(fit, blocks[1])[2] - fixture.alpha0[2]) < 0.8
    @test abs(coef(fit, blocks[2])[2] - fixture.alpha1[2]) < 0.8

    # Independent oracle: dense Laplace marginal likelihood at the fitted theta.
    Xmu = hcat(ones(length(data.x)), data.x)
    Z0 = hcat(ones(length(data.z0)), data.z0)
    Z1 = hcat(ones(length(data.z1)), data.z1)
    theta = vec(vcat(coef(fit, :mu), coef(fit, blocks[1]), coef(fit, blocks[2])))
    dense = _poisson_animal_q2_dense_laplace(theta, data.y, Xmu, data.x,
        Z0, Z1, fixture.gidx, fixture.A)
    @test -loglikelihood(fit) ≈ dense atol = 1e-4 rtol = 1e-6

    # Refusals stay explicit.
    @test_throws ArgumentError drm(bf(@formula(y ~ x + animal(1 + x | g)),
        @formula(sd(g, animal, coef(intercept)) ~ z0)), Poisson();
        data, A = fixture.A)
    zero_x = merge(data, (; x = zeros(length(data.x))))
    @test_throws ArgumentError drm(formula, Poisson(); data = zero_x,
        A = fixture.A, se = false)
end


# ---------------------------------------------------------------------------
# Hardening: prediction on new data, optimiser movement, message-checked
# refusals, bridge labels.  Prediction is POPULATION-LEVEL: these routes do not
# store conditional modes, so predict()/fitted() are exp(X beta + offset) with
# the random effects at zero (same as the Gamma q2 route).
# ---------------------------------------------------------------------------
function _pan_msg(f)
    try
        f()
    catch e
        e isa ArgumentError && return e.msg
        rethrow()
    end
    return ""
end

@testset "Poisson animal q2: prediction, optimiser movement, refusals, bridge" begin
    fixture = _poisson_animal_q2_fixture()
    data = fixture.data
    n = length(data.x)
    formula = bf(@formula(y ~ x + animal(1 + x | g)),
        @formula(sd(g, animal, coef(intercept)) ~ z0),
        @formula(sd(g, animal, coef(x)) ~ z1))
    fit = drm(formula, Poisson(); data, A = fixture.A)
    @test fit.converged
    targets = DRModels._sd_target_parts(formula)
    blocks = [DRModels._sd_target_block(
                  only(filter(t -> t.target.member === m, targets)).target;
                  disambiguate = true) for m in (:intercept, :x)]
    beta = coef(fit, :mu)
    Xmu = hcat(ones(n), data.x)

    # (a) The optimiser moved: loglik clearly exceeds loglik at the start values.
    theta_hat = vec(vcat(beta,  coef(fit, blocks[1]), coef(fit, blocks[2])))
    theta0 = vcat([log(sum(data.y) / n), 0.0],  [log(0.4), 0.0], [log(0.25), 0.0])
    @test length(theta0) == length(theta_hat)
    @test loglikelihood(fit) > -fit.nll(theta0) + 1.0
    @test maximum(abs.(theta_hat .- theta0)) > 0.05

    # (b) Fitted values == predict at the training data; link == X beta.
    @test predict(fit, data) ≈ fitted(fit; re = :zero) atol = 1e-10
    @test predict(fit, data; type = :link) ≈ Xmu * beta atol = 1e-10
    @test fitted(fit; re = :zero) ≈ exp.(Xmu * beta) atol = 1e-8
    pp = predict_parameters(fit, data)
    @test pp[:mu] ≈ fitted(fit; re = :zero) atol = 1e-10
    # New data: unseen group label, new x values; random effects at zero.
    nd = (; x = [-0.7, 0.0, 0.5, 1.1], g = fill("never_seen", 4),
          z0 = zeros(4), z1 = zeros(4))
    expected = exp.(hcat(ones(4), nd.x) * beta)
    @test predict(fit, nd) ≈ expected atol = 1e-10
    @test predict(fit, nd; type = :link) ≈ log.(expected) atol = 1e-10
    @test predict_parameters(fit, nd)[:mu] ≈ expected atol = 1e-10
    # An existing group level gives the same population-level answer.
    nd_old = merge(nd, (; g = fill(first(data.g), 4)))
    @test predict(fit, nd_old) ≈ expected atol = 1e-10
    # Delta-method SE on the link scale from the mu block of vcov.
    sel = predict(fit, nd; type = :link, se = true)
    Vmu = vcov(fit)[1:2, 1:2]
    Xnd = hcat(ones(4), nd.x)
    @test sel.prediction ≈ log.(expected) atol = 1e-10
    @test sel.se ≈ sqrt.(diag(Xnd * Vmu * Xnd')) atol = 1e-8
    @test all(isfinite, sel.se) && all(>(0), sel.se)
    @test all(isfinite, predict(fit, nd; se = true).se)
    # Bridge with newdata: the same parameters, and label rendering works.
    bridge_fit = drm_bridge(; formula = "y ~ x + animal(1 + x | g); sd(g, animal, coef(intercept)) ~ z0; sd(g, animal, coef(x)) ~ z1", family = "poisson", data, A = fixture.A,
                            newdata = nd)
    @test bridge_fit["converged"]
    @test bridge_fit["coefficients"] ≈ coef(fit) atol = 1e-6
    @test bridge_fit["coef_names"] == bridge_fit["vcov_names"]
    @test length(unique(bridge_fit["coef_names"])) == length(bridge_fit["coef_names"])
    @test length(bridge_fit["coef_names"]) == length(theta_hat)
    @test bridge_fit["dpars_newdata"]["mu"] ≈ expected atol = 1e-6

    # (c) Message-checked refusals.
    one_sd = bf(@formula(y ~ x + animal(1 + x | g)),
        @formula(sd(g, animal, coef(intercept)) ~ z0))
    @test occursin("needs one SD formula for the intercept and one for the slope",
        _pan_msg(() -> drm(one_sd, Poisson(); data, A = fixture.A, se = false)))
    zero_x = merge(data, (; x = zeros(n)))
    @test occursin("is zero for every observation",
        _pan_msg(() -> drm(formula, Poisson(); data = zero_x, A = fixture.A, se = false)))
    @test occursin("`K = …` was supplied but the formula has no",
        _pan_msg(() -> drm(formula, Poisson(); data, K = fixture.A, se = false)))

    # (d) Constant slope + identical SD designs is refused for the right reason;
    # the same SD designs with a varying slope are accepted (control).
    same = bf(@formula(y ~ 1 + animal(1 + x | g)),
        @formula(sd(g, animal, coef(intercept)) ~ z0),
        @formula(sd(g, animal, coef(x)) ~ z0))
    const_x = merge(data, (; x = fill(1.0, n)))
    msg = _pan_msg(() -> drm(same, Poisson(); data = const_x, A = fixture.A, se = false))
    @test occursin("cannot separate the intercept and slope variances", msg)
    @test occursin("is constant and both SD formulas have identical", msg)
    ctrl = drm(same, Poisson(); data, A = fixture.A, se = false)
    @test ctrl.converged
end
