# test_164_gamma_hetero.jl — covariate dispersion with a mean-only RE for the
# Gamma and Beta phylogenetic Laplace families (#164).
#
# Extends the NB2 covariate-dispersion path (test_164_mean_re_covariate_sigma.jl)
# to Gamma and Beta. Model: bf(y ~ x + phylo(1 | species), sigma ~ x). The MEAN
# carries a phylogenetic random intercept; the dispersion is a per-observation
# linear predictor ησ = Xσ·βσ. The Gamma shape is α = exp(-2·ησ) and the Beta
# precision is φ = exp(-2·ησ), the per-observation analogues of the scalar
# `:gamma_fixed` / `:beta_fixed` nuisance. The generalised path
# (`_phylo_mean_laplace_hetero_fg` + `Val(:gamma_hetero)` / `Val(:beta_hetero)`)
# carries a VECTOR nuisance gradient over βσ.
#
# Three gates per family:
#   1. STANDING FD-vs-exact gate ≤ 1e-6 on the marginal NLL gradient, evaluated
#      OFF the optimum so the implicit db̂/dθ terms are exercised across both βσ
#      and the phylo logσ (warm-started tight inner mode).
#   2. End-to-end `drm()` recovery of βμ / βσ / σ_phylo.
#   3. Reduction invariant: a one-column constant Xσ reproduces the scalar
#      `_phylo_mean_laplace_nuisance_fg` bit-for-bit (no regression on
#      `sigma ~ 1`).

using DRModels
using Test, Random, LinearAlgebra
import Distributions
using SpecialFunctions: loggamma, digamma

_gh_logistic(x) = 1 / (1 + exp(-x))

# Per-observation Gamma hetero aux: ησ ↦ shape vector + lconst.
function _gamma_hetero_aux(yv)
    return ησ -> begin
        α = exp.(clamp.(-2 .* ησ, -8.0, 8.0))
        lconst = [α[i] * log(α[i]) - loggamma(α[i]) + (α[i] - 1) * log(yv[i])
                  for i in eachindex(yv)]
        return (y = yv, shape = α, lconst = lconst)
    end
end

# Per-observation Beta hetero aux: ησ ↦ precision vector + cached gamma terms.
function _beta_hetero_aux(yv, ylogit)
    return ησ -> begin
        φ = exp.(clamp.(-2 .* ησ, -8.0, 8.0))
        return (y = yv, precision = φ, ylogit = ylogit,
                lgammaφ = loggamma.(φ), digammaφ = digamma.(φ))
    end
end

@testset "Gamma covariate dispersion + mean phylo RE — FD-vs-exact gate ≤ 1e-6 (#164)" begin
    Random.seed!(3041)
    p = 12; m = 4
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    Q, leaf_node, _ = DRModels._poisson_phylo_setup(phy, species)
    q = size(Q, 1)
    logdetQ = logdet(cholesky(Symmetric(Q); check = false))
    C = sigma_phy_dense(phy; σ²_phy = 0.45^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    Xμ = hcat(ones(n), x)
    Xσ = hcat(ones(n), x)                        # sigma ~ 1 + x

    # log-σ = -0.5·log(α): pick a moderate, x-dependent shape.
    ησ_true = -0.35 .+ 0.20 .* x
    α = exp.(-2 .* ησ_true)
    μ = exp.(0.20 .+ 0.35 .* x .+ u[species])
    yv = Float64.([rand(Distributions.Gamma(α[i], μ[i] / α[i])) for i in 1:n])
    aux_from = _gamma_hetero_aux(yv)

    NTOL = 1e-10; NMAX = 400; FDH = 1e-4
    # θ = [βμ(2); βσ(2); logσphylo], OFF the optimum.
    θ = [0.10, 0.45, -0.30, 0.25, log(0.55)]
    val0, g_an, b_base, ok = DRModels._phylo_mean_laplace_hetero_fg(
        Val(:gamma_hetero), aux_from, n, Xμ, Xσ, leaf_node, Q, logdetQ, θ;
        grad = true, b0 = zeros(q), newton_tol = NTOL, newton_maxiter = NMAX,
    )
    @test ok && isfinite(val0) && val0 < 1e17

    g_fd = zeros(length(θ))
    for k in eachindex(θ)
        tp = copy(θ); tp[k] += FDH
        tm = copy(θ); tm[k] -= FDH
        fp = DRModels._phylo_mean_laplace_hetero_fg(Val(:gamma_hetero), aux_from, n, Xμ, Xσ,
            leaf_node, Q, logdetQ, tp; grad = false, b0 = copy(b_base),
            newton_tol = NTOL, newton_maxiter = NMAX)[1]
        fm = DRModels._phylo_mean_laplace_hetero_fg(Val(:gamma_hetero), aux_from, n, Xμ, Xσ,
            leaf_node, Q, logdetQ, tm; grad = false, b0 = copy(b_base),
            newton_tol = NTOL, newton_maxiter = NMAX)[1]
        g_fd[k] = (fp - fm) / (2FDH)
    end
    max_abs_diff = maximum(abs, g_an .- g_fd)
    @info "Gamma hetero-σ phylo gradient gate (#164)" max_abs_diff g_an g_fd
    @test max_abs_diff ≤ 1e-6
end

@testset "Gamma covariate dispersion + mean phylo RE — drm() recovery (#164)" begin
    Random.seed!(20260611)
    p = 40; m = 12
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    βμ = [0.20, 0.35]
    βσ = [-0.30, 0.25]                           # log-σ = -0.30 + 0.25 x  (α = exp(-2·log-σ))
    σphy = 0.45
    C = sigma_phy_dense(phy; σ²_phy = σphy^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    α = exp.(-2 .* (βσ[1] .+ βσ[2] .* x))
    μ = exp.(βμ[1] .+ βμ[2] .* x .+ u[species])
    y = Float64.([rand(Distributions.Gamma(α[i], μ[i] / α[i])) for i in 1:n])

    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x)),
              DRModels.Gamma(); data = (; y, x, species), tree = phy, se = false)

    @test fit.converged
    @test length(coef(fit, :sigma)) == 2            # intercept + slope on log-σ
    @test coef(fit, :mu)[2] ≈ βμ[2] atol = 0.20
    @test coef(fit, :sigma)[2] ≈ βσ[2] atol = 0.30  # the heteroscedastic σ slope
    @test re_sd(fit)[:species] > 0.10
    @test isfinite(loglik(fit))
    @test all(fitted(fit) .> 0)

    # Loose public-path FD sanity (the tight ≤1e-6 gate is the dedicated _fg gate
    # above; here the inner mode is at the fit's default tolerance).
    θ = coef(fit)
    g = zeros(length(θ)); fit.nllgrad(g, θ)
    h = 1e-4; fd = similar(g)
    for k in eachindex(θ)
        e = zeros(length(θ)); e[k] = h
        fd[k] = (fit.nll(θ .+ e) - fit.nll(θ .- e)) / (2h)
    end
    @test g ≈ fd rtol = 5e-3 atol = 5e-3
end

@testset "Gamma hetero σ reduces to the scalar nuisance path on a 1-column Xσ (#164)" begin
    Random.seed!(3051)
    p = 10; m = 4
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    Q, leaf_node, _ = DRModels._poisson_phylo_setup(phy, species)
    q = size(Q, 1)
    logdetQ = logdet(cholesky(Symmetric(Q); check = false))
    C = sigma_phy_dense(phy; σ²_phy = 0.40^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    Xμ = hcat(ones(n), x)
    μ = exp.(0.20 .+ 0.35 .* x .+ u[species])
    shapep = 5.0
    yv = Float64.([rand(Distributions.Gamma(shapep, μi / shapep)) for μi in μ])

    aux_scalar(ls) = begin
        α = exp(clamp(-2 * ls, -8.0, 8.0))
        (y = yv, shape = α,
         lconst = [α * log(α) - loggamma(α) + (α - 1) * log(yv[i]) for i in eachindex(yv)])
    end
    aux_hetero = _gamma_hetero_aux(yv)
    Xσ1 = ones(n, 1)

    θ = [0.1, 0.4, -0.5 * log(shapep), log(0.55)]   # same layout: pσ = 1
    NTOL = 1e-10; NMAX = 400
    vs, gs, _, oks = DRModels._phylo_mean_laplace_nuisance_fg(
        Val(:gamma_fixed), aux_scalar, n, Xμ, leaf_node, Q, logdetQ, θ;
        grad = true, b0 = zeros(q), newton_tol = NTOL, newton_maxiter = NMAX)
    vh, gh, _, okh = DRModels._phylo_mean_laplace_hetero_fg(
        Val(:gamma_hetero), aux_hetero, n, Xμ, Xσ1, leaf_node, Q, logdetQ, θ;
        grad = true, b0 = zeros(q), newton_tol = NTOL, newton_maxiter = NMAX)
    @test oks && okh
    @test vs ≈ vh atol = 1e-12
    @test gs ≈ gh atol = 1e-10
end

@testset "Beta covariate dispersion + mean phylo RE — FD-vs-exact gate ≤ 1e-6 (#164)" begin
    Random.seed!(3061)
    p = 12; m = 4
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    Q, leaf_node, _ = DRModels._poisson_phylo_setup(phy, species)
    q = size(Q, 1)
    logdetQ = logdet(cholesky(Symmetric(Q); check = false))
    C = sigma_phy_dense(phy; σ²_phy = 0.35^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    Xμ = hcat(ones(n), x)
    Xσ = hcat(ones(n), x)                        # sigma ~ 1 + x

    # log-σ = -0.5·log(φ): keep φ comfortably > 1 across x so a,b stay well-posed.
    ησ_true = -1.0 .+ 0.15 .* x
    φ = exp.(-2 .* ησ_true)
    μ = _gh_logistic.(-0.10 .+ 0.45 .* x .+ u[species])
    yv = Float64.([rand(Distributions.Beta(μ[i] * φ[i], (1 - μ[i]) * φ[i])) for i in 1:n])
    ylogit = log.(yv) .- log1p.(-yv)
    aux_from = _beta_hetero_aux(yv, ylogit)

    NTOL = 1e-10; NMAX = 400; FDH = 1e-4
    θ = [0.05, 0.40, -0.95, 0.15, log(0.50)]
    val0, g_an, b_base, ok = DRModels._phylo_mean_laplace_hetero_fg(
        Val(:beta_hetero), aux_from, n, Xμ, Xσ, leaf_node, Q, logdetQ, θ;
        grad = true, b0 = zeros(q), newton_tol = NTOL, newton_maxiter = NMAX,
    )
    @test ok && isfinite(val0) && val0 < 1e17

    g_fd = zeros(length(θ))
    for k in eachindex(θ)
        tp = copy(θ); tp[k] += FDH
        tm = copy(θ); tm[k] -= FDH
        fp = DRModels._phylo_mean_laplace_hetero_fg(Val(:beta_hetero), aux_from, n, Xμ, Xσ,
            leaf_node, Q, logdetQ, tp; grad = false, b0 = copy(b_base),
            newton_tol = NTOL, newton_maxiter = NMAX)[1]
        fm = DRModels._phylo_mean_laplace_hetero_fg(Val(:beta_hetero), aux_from, n, Xμ, Xσ,
            leaf_node, Q, logdetQ, tm; grad = false, b0 = copy(b_base),
            newton_tol = NTOL, newton_maxiter = NMAX)[1]
        g_fd[k] = (fp - fm) / (2FDH)
    end
    max_abs_diff = maximum(abs, g_an .- g_fd)
    @info "Beta hetero-σ phylo gradient gate (#164)" max_abs_diff g_an g_fd
    @test max_abs_diff ≤ 1e-6
end

@testset "Beta covariate dispersion + mean phylo RE — drm() recovery (#164)" begin
    Random.seed!(20260612)
    p = 40; m = 14
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    βμ = [-0.10, 0.45]
    βσ = [-1.10, 0.20]                           # log-σ = -1.10 + 0.20 x  (φ = exp(-2·log-σ))
    σphy = 0.35
    C = sigma_phy_dense(phy; σ²_phy = σphy^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    φ = exp.(-2 .* (βσ[1] .+ βσ[2] .* x))
    μ = _gh_logistic.(βμ[1] .+ βμ[2] .* x .+ u[species])
    y = Float64.([rand(Distributions.Beta(μ[i] * φ[i], (1 - μ[i]) * φ[i])) for i in 1:n])

    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x)),
              DRModels.Beta(); data = (; y, x, species), tree = phy, se = false)

    @test fit.converged
    @test length(coef(fit, :sigma)) == 2
    @test coef(fit, :mu)[2] ≈ βμ[2] atol = 0.25
    @test coef(fit, :sigma)[2] ≈ βσ[2] atol = 0.30
    @test re_sd(fit)[:species] > 0.05
    @test isfinite(loglik(fit))
    @test all(0 .< fitted(fit) .< 1)

    θ = coef(fit)
    g = zeros(length(θ)); fit.nllgrad(g, θ)
    h = 1e-4; fd = similar(g)
    for k in eachindex(θ)
        e = zeros(length(θ)); e[k] = h
        fd[k] = (fit.nll(θ .+ e) - fit.nll(θ .- e)) / (2h)
    end
    @test g ≈ fd rtol = 7e-3 atol = 7e-3
end

@testset "Beta hetero σ reduces to the scalar nuisance path on a 1-column Xσ (#164)" begin
    Random.seed!(3071)
    p = 10; m = 5
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    Q, leaf_node, _ = DRModels._poisson_phylo_setup(phy, species)
    q = size(Q, 1)
    logdetQ = logdet(cholesky(Symmetric(Q); check = false))
    C = sigma_phy_dense(phy; σ²_phy = 0.30^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    Xμ = hcat(ones(n), x)
    μ = _gh_logistic.(0.05 .+ 0.35 .* x .+ u[species])
    precp = 14.0
    yv = Float64.([rand(Distributions.Beta(μi * precp, (1 - μi) * precp)) for μi in μ])
    ylogit = log.(yv) .- log1p.(-yv)

    aux_scalar(ls) = begin
        φ = exp(clamp(-2 * ls, -8.0, 8.0))
        (y = yv, precision = φ, ylogit = ylogit,
         lgammaφ = loggamma(φ), digammaφ = digamma(φ))
    end
    aux_hetero = _beta_hetero_aux(yv, ylogit)
    Xσ1 = ones(n, 1)

    θ = [0.1, 0.4, -0.5 * log(precp), log(0.50)]
    NTOL = 1e-10; NMAX = 400
    vs, gs, _, oks = DRModels._phylo_mean_laplace_nuisance_fg(
        Val(:beta_fixed), aux_scalar, n, Xμ, leaf_node, Q, logdetQ, θ;
        grad = true, b0 = zeros(q), newton_tol = NTOL, newton_maxiter = NMAX)
    vh, gh, _, okh = DRModels._phylo_mean_laplace_hetero_fg(
        Val(:beta_hetero), aux_hetero, n, Xμ, Xσ1, leaf_node, Q, logdetQ, θ;
        grad = true, b0 = zeros(q), newton_tol = NTOL, newton_maxiter = NMAX)
    @test oks && okh
    @test vs ≈ vh atol = 1e-12
    @test gs ≈ gh atol = 1e-10
end
