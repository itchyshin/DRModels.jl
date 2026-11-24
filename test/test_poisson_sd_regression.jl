# Poisson random-intercept SD regression on the Julia engine.
using DRModels, Test, Random, QuadGK, LinearAlgebra, StableRNGs
import Distributions

function _poisson_sd_fixture(; G = 32, m = 16)
    rng = StableRNG(20261018)
    g = repeat(1:G, inner = m)
    zgroup = repeat([0.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, length(g))
    alpha = [-0.45, 0.4]
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    lambda = exp.(0.55 .+ 0.25 .* x .+ b[g])
    y = [rand(rng, Distributions.Poisson(v)) for v in lambda]
    return (; y = Float64.(y), x, z, g), alpha
end

function _poisson_sd_slope_fixture(; G = 48, m = 24)
    rng = StableRNG(20261019)
    g = repeat(1:G, inner = m)
    zgroup = repeat([0.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, length(g))
    alpha = [-0.45, 0.4]
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    lambda = exp.(0.55 .+ 0.25 .* x .+ b[g] .* x)
    y = [rand(rng, Distributions.Poisson(v)) for v in lambda]
    return (; y = Float64.(y), x, z, g, zgroup), alpha
end

function _poisson_phylo_sd_test_newick(depth; height = 1.0)
    function node(prefix, d)
        d == 0 && return "$(prefix):$(height / depth)"
        left = node(prefix * "a", d - 1)
        right = node(prefix * "b", d - 1)
        return d == depth ? "($left,$right);" : "($left,$right):$(height / depth)"
    end
    return node("tip", depth)
end

function _poisson_phylo_sd_dense_laplace(theta, y, Xmu, Zobs, leaf_node, Q)
    q = size(Q, 1)
    Qdense = Matrix(Q)
    p_mu = size(Xmu, 2)
    beta = theta[1:p_mu]
    alpha = theta[p_mu+1:end]
    eta0 = Xmu * beta
    loading = exp.(Zobs * alpha)
    b = zeros(q)
    function joint(bb)
        value = 0.5 * dot(bb, Qdense * bb)
        for i in eachindex(y)
            eta = eta0[i] + loading[i] * bb[leaf_node[i]]
            value -= Distributions.logpdf(Distributions.Poisson(exp(eta)), y[i])
        end
        value
    end
    for _ in 1:100
        grad = Qdense * b
        hdiag = zeros(q)
        for i in eachindex(y)
            li = leaf_node[i]
            eta = eta0[i] + loading[i] * b[li]
            mu = exp(eta)
            grad[li] += loading[i] * (mu - y[i])
            hdiag[li] += loading[i]^2 * mu
        end
        H = Qdense + Diagonal(hdiag)
        ch = cholesky(Symmetric(H))
        step = ch \ grad
        # Newton converges quadratically, so a 1e-10 relative step leaves a
        # mode error far below the objective tolerances; a tighter cut-off sits
        # at the gradient round-off floor (~1e-11 here) and never triggers.
        if norm(step) <= 1e-10 * (1 + norm(b))
            return joint(b) + 0.5 * logdet(ch) -
                0.5 * logdet(Symmetric(Qdense))
        end
        f0 = joint(b)
        fraction = 1.0
        accepted = false
        while fraction >= 1e-8
            candidate = b .- fraction .* step
            if joint(candidate) <= f0
                b = candidate
                accepted = true
                break
            end
            fraction *= 0.5
        end
        accepted || error("dense Poisson phylogenetic Laplace mode did not descend")
    end
    error("dense Poisson phylogenetic Laplace mode did not converge")
end

@testset "Poisson latent SD regression: independent random slope" begin
    data, alpha = _poisson_sd_slope_fixture()
    formula = bf(@formula(y ~ x + (0 + x | g)), @formula(sd(g) ~ z))
    fit = drm(formula, Poisson(); data = data, nAGQ = 7)
    @test fit.converged
    @test length(coef(fit, :sd)) == 2
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.55
    @test only(fit.scales[:aghq_nodes]) == 7
    @test length(fit.scales[:sd_group]) == 48
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    @test predict_parameters(fit, data; type = :link)[:sd] ≈ log.(fit.scales[:sd])

    theta = [0.55, 0.25, -0.45, 0.4]
    reference = 0.0
    normal = Distributions.Normal()
    for gi in 1:48
        rows = findall(==(gi), data.g)
        sdg = exp(theta[3] + theta[4] * data.z[first(rows)])
        integrand = u -> begin
            ll = sum(rows) do i
                eta = theta[1] + theta[2] * data.x[i] + sdg * u * data.x[i]
                Distributions.logpdf(Distributions.Poisson(exp(eta)), Int(data.y[i]))
            end
            exp(ll) * Distributions.pdf(normal, u)
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-11)
        reference += log(value)
    end
    @test isapprox(-fit.nll(theta), reference; atol = 2e-3)

end

@testset "Poisson latent SD regression: ordinary random intercept" begin
    data, alpha = _poisson_sd_fixture()
    formula = bf(@formula(y ~ x + (1 | g)),
                 @formula(sd(g, coef(intercept)) ~ z))
    fit = drm(formula, Poisson(); data = data, nAGQ = 3)
    @test fit.converged
    @test length(coef(fit, :sd)) == 2
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.55
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(32), [zeros(16); ones(16)]) * coef(fit, :sd))
    @test Int.(fit.scales[:sd_group_index]) == data.g
    @test only(fit.scales[:aghq_nodes]) == 3
    @test DRModels._bootstrap_refit_kwargs(fit).nAGQ == 3
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    @test predict_parameters(fit, data; type = :link)[:sd] ≈ log.(fit.scales[:sd])
    @test length(fitted(fit; re = :zero)) == length(data.y)

    bridge = drm_bridge(; formula = "y ~ x + (1 | g); sd(g) ~ z",
                        family = "poisson", data)
    @test Set(keys(bridge["dpars"])) == Set(["mu"])
    @test isempty(bridge["sigma"])
    profile_ci = confint(fit; method = :profile, parm = :sd)
    @test length(profile_ci) == 2
    @test all(row -> isfinite(row.lower) && isfinite(row.upper), profile_ci)
    boot = bootstrap_result(fit; data, B = 2, rng = MersenneTwister(2026101819),
                            failures = :skip)
    @test boot.attempted == 2
    @test boot.used + boot.failed == 2

    no_se = drm(formula, Poisson(); data = data, se = false)
    @test all(isnan, no_se.vcov)

    constant = bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ 1))
    # Same node count as the plain `(1 | g)` route, so only the likelihood assembly is compared.
    f_sd = drm(constant, Poisson(); data = data, nAGQ = 5)
    f_old = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = data)
    @test f_sd.converged && f_old.converged
    @test isapprox(f_sd.loglik, f_old.loglik; atol = 1e-5)
    @test isapprox(coef(f_sd, :mu), coef(f_old, :mu); atol = 1e-4)
    @test isapprox(coef(f_sd, :sd)[1], coef(f_old, :resd)[1]; atol = 1e-4)
end

@testset "Poisson latent SD regression refuses unsupported designs" begin
    data, _ = _poisson_sd_fixture(; G = 8, m = 5)
    route(f; kwargs...) = drm(f, Poisson(); data = data, kwargs...)
    @test_throws ArgumentError route(bf(@formula(y ~ x + (1 | g)),
                                        @formula(sd(h) ~ z)))
    @test_throws ArgumentError route(bf(@formula(y ~ x + (1 + x | g)),
                                        @formula(sd(g) ~ z)))
    varying = merge(data, (; z = copy(data.z)))
    varying.z[1] = 1.0 - varying.z[1]
    @test_throws ArgumentError drm(
        bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z)),
        Poisson(); data = varying)
    @test_throws ArgumentError route(bf(@formula(y ~ x + (1 | g)),
                                        @formula(sd(g) ~ z)); method = :REML)

    wrong_dpar = bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z))
    sd_key = only(keys(wrong_dpar.sd_targets))
    wrong_dpar.sd_targets[sd_key] = merge(wrong_dpar.sd_targets[sd_key],
                                           (; dpar = :sigma))
    @test_throws ArgumentError route(wrong_dpar)
end

@testset "Poisson latent SD regression: fixed-parameter integral" begin
    data, _ = _poisson_sd_fixture(; G = 12, m = 5)
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z))
    fit = drm(formula, Poisson(); data = data)
    theta = [0.55, 0.25, -0.45, 0.4]
    reference = 0.0
    normal = Distributions.Normal()
    for gi in 1:12
        rows = findall(==(gi), data.g)
        sdg = exp(theta[3] + theta[4] * data.z[first(rows)])
        integrand = u -> begin
            ll = sum(rows) do i
                eta = theta[1] + theta[2] * data.x[i] + sdg * u
                Distributions.logpdf(Distributions.Poisson(exp(eta)), Int(data.y[i]))
            end
            exp(ll) * Distributions.pdf(normal, u)
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-11)
        reference += log(value)
    end
    @test isapprox(-fit.nll(theta), reference; atol = 2e-3)

end

@testset "Poisson SD regression includes a fixed exposure offset" begin
    rng = StableRNG(20261023)
    G, m = 64, 18
    group = repeat(1:G, inner = m)
    z_group = repeat([0.0, 1.0], inner = G ÷ 2)
    z = z_group[group]
    x = randn(rng, G * m)
    exposure_group = repeat(range(0.6, 1.8; length = G), inner = m)
    log_exposure = log.(exposure_group)
    alpha = [log(0.35), 0.7]
    b = exp.(alpha[1] .+ alpha[2] .* z_group) .* randn(rng, G)
    eta = -0.1 .+ 0.35 .* x .+ log_exposure .+ b[group]
    y = Float64.([rand(rng, Distributions.Poisson(exp(v))) for v in eta])
    data = (; y, x, z, group, log_exposure)
    formula = bf(@formula(y ~ x + offset(log_exposure) + (1 | group)),
                 @formula(sd(group) ~ z))

    fit = drm(formula, Poisson(); data)
    @test fit.converged
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.5
    expected_mean = exp.(coef(fit, :mu)[1] .+
                          coef(fit, :mu)[2] .* x .+ log_exposure)
    @test fitted(fit; re = :zero) ≈ expected_mean
    @test predict_parameters(fit, data)[:mu] ≈ expected_mean
    boot = bootstrap_result(fit; data, B = 1,
                            rng = MersenneTwister(2026102319), failures = :skip)
    @test boot.attempted == 1
    @test boot.used + boot.failed == 1

    # Independent scalar integration checks that the fixed exposure enters
    # each conditional Poisson mean in the marginal likelihood.
    theta = [-0.1, 0.35, alpha[1], alpha[2]]
    reference = 0.0
    normal = Distributions.Normal()
    for gi in 1:G
        rows = findall(==(gi), group)
        sdg = exp(theta[3] + theta[4] * z_group[gi])
        integrand = u -> exp(sum(rows) do i
            Distributions.logpdf(
                Distributions.Poisson(exp(theta[1] + theta[2] * x[i] +
                    log_exposure[i] + sdg * u)), y[i],
            )
        end) * Distributions.pdf(normal, u)
        value, _ = quadgk(integrand, -10.0, 10.0; rtol = 1e-10)
        reference += log(value)
    end
    @test isapprox(-fit.nll(theta), reference; atol = 2e-3)

    # The same offset contract applies when the sole IID effect is an
    # independent numeric slope rather than an intercept.
    b_slope = exp.(alpha[1] .+ alpha[2] .* z_group) .* randn(rng, G)
    y_slope = Float64.([rand(rng, Distributions.Poisson(exp(
        -0.1 + 0.35 * x[i] + log_exposure[i] + b_slope[group[i]] * x[i])))
        for i in eachindex(x)])
    slope_data = merge(data, (; y = y_slope))
    slope_formula = bf(
        @formula(y ~ x + offset(log_exposure) + (0 + x | group)),
        @formula(sd(group) ~ z),
    )
    slope_fit = drm(slope_formula, Poisson(); data = slope_data)
    @test slope_fit.converged
    @test coef(slope_fit, :sd)[2] > 0
    @test fitted(slope_fit; re = :zero) ≈ exp.(coef(slope_fit, :mu)[1] .+
                                    coef(slope_fit, :mu)[2] .* x .+ log_exposure)
end

@testset "Poisson phylogenetic SD Laplace objective matches dense oracle" begin
    rng = StableRNG(20261019)
    tree = _poisson_phylo_sd_test_newick(2)
    phy = DRModels.augmented_phy(tree)
    m = 6
    labels = repeat(phy.leaf_names, inner = m)
    x = repeat(collect(range(-1.0, 1.0; length = m)), outer = phy.n_leaves)
    ztip = [-1.0, 1.0, -1.0, 1.0]
    z = repeat(ztip, inner = m)
    Xmu = hcat(ones(length(x)), x)
    Zobs = hcat(ones(length(z)), z)
    latent = 0.35 .* cholesky(Symmetric(
        DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
    )).L * randn(rng, phy.n_leaves)
    eta = 0.2 .+ 0.15 .* x .+ latent[repeat(1:phy.n_leaves, inner = m)]
    y = Float64.(rand.(Ref(rng), Distributions.Poisson.(exp.(eta))))
    Q, leaf_node, _ = DRModels._poisson_phylo_setup(phy, labels)
    logdetQ = logdet(cholesky(Symmetric(Q)))
    theta = [0.1, 0.2, log(0.45), 0.3]

    actual, _, ok = DRModels._poisson_phylo_sd_laplace(
        y, Xmu, Zobs, leaf_node, Q, logdetQ, theta
    )
    expected = _poisson_phylo_sd_dense_laplace(
        theta, y, Xmu, Zobs, leaf_node, Q
    )

    @test ok
    @test isapprox(actual, expected; atol = 2e-8)

    theta_high = copy(theta)
    theta_high[1] = 31.0
    theta_high[3] = -100.0
    actual_high, _, ok_high = DRModels._poisson_phylo_sd_laplace(
        y, Xmu, Zobs, leaf_node, Q, logdetQ, theta_high
    )
    expected_high = _poisson_phylo_sd_dense_laplace(
        theta_high, y, Xmu, Zobs, leaf_node, Q
    )
    @test ok_high
    @test isapprox(actual_high, expected_high; rtol = 1e-10)

    data = (; y, x, z, species = labels)
    formula = bf(
        @formula(y ~ x + phylo(1 | species)),
        @formula(sd(species, phylogenetic) ~ z),
    )
    fit = drm(formula, Poisson(); data, tree, se = false, g_tol = 1e-6)
    @test fit.converged
    @test isapprox(fit.nll(fit.theta),
                   _poisson_phylo_sd_dense_laplace(
                       fit.theta, y, Xmu, Zobs, leaf_node, Q
                   ); atol = 2e-8)
    dense_gradient = [
        (_poisson_phylo_sd_dense_laplace(
             fit.theta .+ [i == j ? 1e-5 : 0.0 for i in eachindex(fit.theta)],
             y, Xmu, Zobs, leaf_node, Q
         ) - _poisson_phylo_sd_dense_laplace(
             fit.theta .- [i == j ? 1e-5 : 0.0 for i in eachindex(fit.theta)],
             y, Xmu, Zobs, leaf_node, Q
         )) / 2e-5 for j in eachindex(fit.theta)
    ]
    @test norm(dense_gradient, Inf) < 2e-3
end
