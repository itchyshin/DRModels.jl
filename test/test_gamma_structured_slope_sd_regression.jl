# Gamma structured slope-only SD regression: phylo(0 + x | sp), relmat(0 + x | id, K)
# and coordinate spatial(0 + x | site) with `sd(g, level) ~ z`. Model:
#   eta_i = X_i beta + x_i * sd_g(i) * b_g(i),   b ~ N(0, C),
# with C the RAW tree covariance (phylo), the RAW K (relmat; not correlation-
# rescaled, as in native drmTMB) and the native fixed exp(-d / median d) + 1e-6 I
# kernel (spatial). Each fit's Laplace objective is checked against a dense,
# independently coded Laplace oracle.
using DRModels, Test, Random, LinearAlgebra, Statistics
import Distributions

function _gamma_slope_dense_laplace(theta, y, Xmu, x, Zobs, gidx, C)
    Q = inv(Symmetric(C))
    p_mu = size(Xmu, 2)
    beta = theta[1:p_mu]
    shape = 1 / exp(theta[p_mu + 1])^2
    alpha = theta[p_mu + 2:end]
    eta0 = Xmu * beta
    loading = x .* exp.(Zobs * alpha)
    function joint(b)
        value = 0.5 * dot(b, Q * b)
        for i in eachindex(y)
            mu = exp(eta0[i] + loading[i] * b[gidx[i]])
            value -= Distributions.logpdf(Distributions.Gamma(shape, mu / shape), y[i])
        end
        value
    end
    result = DRModels.Optim.optimize(joint, zeros(size(C, 1)),
        DRModels.Optim.LBFGS(), DRModels.Optim.Options(g_tol = 1e-10, iterations = 500);
        autodiff = :forward)
    mode = DRModels.Optim.minimizer(result)
    hessian = DRModels.ForwardDiff.hessian(joint, mode)
    return joint(mode) + 0.5 * logdet(Symmetric(hessian)) - 0.5 * logdet(Symmetric(Q))
end

function _gamma_slope_response(rng, eta, sigma)
    shape = 1 / sigma^2
    return Float64[rand(rng, Distributions.Gamma(shape, exp(v) / shape)) for v in eta]
end

@testset "Gamma relmat slope-only SD regression (raw K, dense oracle)" begin
    rng = MersenneTwister(20261102)
    G, m = 30, 14
    R = 0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    scale = collect(range(0.7, 1.7; length = G))
    K = Diagonal(scale) * R * Diagonal(scale)       # deliberately non-unit diagonal
    ids = ["id_$(i)" for i in 1:G]
    gidx = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    alpha = [-0.6, 0.45]
    b = cholesky(Symmetric(K)).L * randn(rng, G)
    x = repeat(collect(range(-1.2, 1.2; length = m)), outer = G)
    eta = 0.2 .+ 0.25 .* x .+ x .* (exp.(alpha[1] .+ alpha[2] .* zgroup) .* b)[gidx]
    y = _gamma_slope_response(rng, eta, 0.35)
    data = (; y, x, z = zgroup[gidx], g = ids[gidx])
    formula = bf(@formula(y ~ x + relmat(0 + x | g)), @formula(sigma ~ 1),
                 @formula(sd(g, relmat) ~ z))
    fit = drm(formula, Gamma(); data, K, se = false)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.6
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(G), zgroup) * coef(fit, :sd))
    @test predict_parameters(fit, data; se = false)[:sd] ≈ fit.scales[:sd]

    Xmu = hcat(ones(length(y)), x)
    Zobs = hcat(ones(length(y)), data.z)
    theta = vcat(coef(fit, :mu), coef(fit, :sigma), coef(fit, :sd))
    oracle = _gamma_slope_dense_laplace(theta, y, Xmu, x, Zobs, gidx, K)
    @test fit.nll(theta) ≈ oracle atol = 4e-3
    # The kernel is RAW: a correlation-rescaled K would give a different objective.
    Rnorm = K ./ (sqrt.(diag(K)) * sqrt.(diag(K))')
    wrong = _gamma_slope_dense_laplace(theta, y, Xmu, x, Zobs, gidx, Rnorm)
    @test abs(wrong - oracle) > 0.05

    # Refusals: wrong provider argument, missing slope column, member mismatch.
    @test_throws ArgumentError drm(formula, Gamma(); data, A = K, se = false)
    @test_throws ArgumentError drm(formula, Gamma();
        data = (; y, z = data.z, g = data.g), K, se = false)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + relmat(0 + x | g)),
        @formula(sigma ~ 1), @formula(sd(g, relmat, coef(z)) ~ z)), Gamma();
        data, K, se = false)
end

@testset "Gamma coordinate spatial slope-only SD regression (native kernel, dense oracle)" begin
    rng = MersenneTwister(20261103)
    G, m = 30, 14
    coords = hcat(mod.(0:(G - 1), 6), div.(0:(G - 1), 6)) .* 1.3
    D = [sqrt(sum(abs2, coords[i, :] .- coords[j, :])) for i in 1:G, j in 1:G]
    pos = [D[i, j] for i in 1:G for j in (i + 1):G if D[i, j] > 0]
    C = exp.(-D ./ median(pos)) + 1e-6 * I          # native fixed kernel, RAW
    gidx = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)[randperm(rng, G)]
    alpha = [-0.6, 0.45]
    b = cholesky(Symmetric(C)).L * randn(rng, G)
    x = repeat(collect(range(-1.2, 1.2; length = m)), outer = G)
    eta = 0.2 .+ 0.25 .* x .+ x .* (exp.(alpha[1] .+ alpha[2] .* zgroup) .* b)[gidx]
    y = _gamma_slope_response(rng, eta, 0.35)
    data = (; y, x, z = zgroup[gidx], g = gidx)
    formula = bf(@formula(y ~ x + spatial(0 + x | g)), @formula(sigma ~ 1),
                 @formula(sd(g, spatial) ~ z))
    fit = drm(formula, Gamma(); data, coords, se = false)

    @test fit.converged
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.7
    @test predict_parameters(fit, data; se = false)[:sd] ≈ fit.scales[:sd]
    Xmu = hcat(ones(length(y)), x)
    Zobs = hcat(ones(length(y)), data.z)
    theta = vcat(coef(fit, :mu), coef(fit, :sigma), coef(fit, :sd))
    @test fit.nll(theta) ≈ _gamma_slope_dense_laplace(theta, y, Xmu, x, Zobs, gidx, C) atol = 4e-3

    @test_throws ArgumentError drm(formula, Gamma(); data, K = C, se = false)
    @test_throws ArgumentError drm(formula, Gamma(); data, se = false)
end

@testset "Gamma phylogenetic slope-only SD regression (raw tree covariance, dense oracle)" begin
    rng = MersenneTwister(20261104)
    p, m = 16, 16
    tree = random_balanced_tree(p; branch_length = 0.2)
    ztip = repeat([-1.0, 1.0], p ÷ 2)
    alpha = [-0.6, 0.5]
    C = sigma_phy_dense(tree; σ²_phy = 1.0)
    b = cholesky(Symmetric(C)).L * randn(rng, p)
    tip = repeat(1:p; inner = m)
    species = tree.leaf_names[tip]
    x = repeat(collect(range(-1.2, 1.2; length = m)), outer = p)
    eta = 0.2 .+ 0.25 .* x .+ x .* (exp.(alpha[1] .+ alpha[2] .* ztip) .* b)[tip]
    y = _gamma_slope_response(rng, eta, 0.35)
    data = (; y, x, species, z = ztip[tip])
    formula = bf(@formula(y ~ x + phylo(0 + x | species)), @formula(sigma ~ 1),
                 @formula(sd(species, phylogenetic) ~ z))
    fit = drm(formula, Gamma(); data, tree, se = false, g_tol = 1e-5)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd_phylo]
    @test coef(fit, :sd_phylo)[2] > 0
    @test abs(coef(fit, :sd_phylo)[2] - alpha[2]) < 0.7
    @test predict_parameters(fit, data; se = false)[:sd_phylo] ≈ fit.scales[:sd_phylo]

    # Map response rows to the tree's leaf order for the dense oracle.
    leaf_order = Dict(name => i for (i, name) in enumerate(tree.leaf_names))
    gtip = [leaf_order[s] for s in species]
    Xmu = hcat(ones(length(y)), x)
    Zobs = hcat(ones(length(y)), data.z)
    theta = vcat(coef(fit, :mu), coef(fit, :sigma), coef(fit, :sd_phylo))
    oracle = _gamma_slope_dense_laplace(theta, y, Xmu, x, Zobs, gtip, C)
    @test fit.nll(theta) ≈ oracle atol = 4e-3

    @test_throws ArgumentError drm(formula, Gamma(); data, se = false)         # no tree
    @test_throws ArgumentError drm(formula, Gamma();
        data = (; y, species, z = data.z), tree, se = false)                    # no slope column
end
