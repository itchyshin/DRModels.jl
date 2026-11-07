using DRModels, Test, Random, LinearAlgebra
import Distributions

function _poisson_relmat_q2_fixture(; G = 48, m = 24, seed = 202611071)
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

function _poisson_relmat_q2_dense_laplace(theta, y, Xmu, x, Z0, Z1, gidx, A)
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

@testset "Poisson relmat independent intercept and slope SD regression" begin
    fixture = _poisson_relmat_q2_fixture()
    data = fixture.data
    formula = bf(@formula(y ~ x + relmat(1 + x | g)),
        @formula(sd(g, relmat, coef(intercept)) ~ z0),
        @formula(sd(g, relmat, coef(x)) ~ z1))
    targets = DRModels._sd_target_parts(formula)
    @test length(targets) == 2

    fit = drm(formula, Poisson(); data, K = fixture.A)
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
    dense = _poisson_relmat_q2_dense_laplace(theta, data.y, Xmu, data.x,
        Z0, Z1, fixture.gidx, fixture.A)
    @test -loglikelihood(fit) ≈ dense atol = 1e-4 rtol = 1e-6

    # Refusals stay explicit.
    @test_throws ArgumentError drm(bf(@formula(y ~ x + relmat(1 + x | g)),
        @formula(sd(g, relmat, coef(intercept)) ~ z0)), Poisson();
        data, K = fixture.A)
    zero_x = merge(data, (; x = zeros(length(data.x))))
    @test_throws ArgumentError drm(formula, Poisson(); data = zero_x,
        K = fixture.A, se = false)
end
