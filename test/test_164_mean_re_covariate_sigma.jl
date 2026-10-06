# test_164_mean_re_covariate_sigma.jl — covariate dispersion with a mean-only RE
# for a non-Gaussian phylogenetic Laplace family (#164).
#
# Model: bf(y ~ x + phylo(1 | species), sigma ~ x), NegBinomial2().
# The MEAN carries a phylogenetic random intercept; the dispersion (log-size) is
# a per-observation linear predictor ησ = Xσ·βσ. This is the sub-case the prior
# audit flagged: the phylo Laplace spine carried a SCALAR θσ nuisance and
# hard-errored on a non-constant Xσ ("supports only sigma ~ 1"). The generalised
# path (`_phylo_mean_laplace_hetero_fg` + `Val(:nb2_hetero)`) carries a VECTOR
# nuisance gradient over βσ.
#
# Three gates:
#   1. STANDING FD-vs-exact gate ≤ 1e-6 on the marginal NLL gradient (#165 recipe:
#      tight inner mode, warm-started, evaluated OFF the optimum so the implicit
#      db̂/dθ terms are exercised across both βσ and the phylo logσ).
#   2. End-to-end `drm()` recovery of βμ / βσ / σ_phylo (loose public FD sanity,
#      same tolerance as the sibling phylo-Laplace tests).
#   3. Reduction invariant: a one-column constant Xσ reproduces the scalar
#      `_phylo_mean_laplace_nuisance_fg` bit-for-bit (no regression on `sigma ~ 1`).

using DRModels
using Test, Random, LinearAlgebra
import Distributions
using SpecialFunctions: loggamma

# Per-observation NB2 hetero aux: ησ ↦ size vector + lconst.
function _nb2_hetero_aux(yint, yf)
    return ησ -> begin
        r = exp.(clamp.(-2 .* ησ, -8.0, 8.0))
        lconst = [loggamma(yf[i] + r[i]) - loggamma(r[i]) - DRModels._logfactorial(yint[i])
                  for i in eachindex(yint)]
        return (y = yf, size = r, lconst = lconst)
    end
end

@testset "NB2 covariate dispersion + mean phylo RE — FD-vs-exact gate ≤ 1e-6 (#164)" begin
    Random.seed!(2041)
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

    logsize = 1.4 .+ 0.5 .* x
    sz = exp.(logsize)
    μ = exp.(0.20 .+ 0.35 .* x .+ u[species])
    yint = [rand(Distributions.NegativeBinomial(sz[i], sz[i] / (sz[i] + μ[i]))) for i in 1:n]
    yf = Float64.(yint)
    aux_from = _nb2_hetero_aux(yint, yf)

    NTOL = 1e-10; NMAX = 400; FDH = 1e-4
    # θ = [βμ(2); βσ(2); logσphylo], OFF the optimum.
    θ = [0.10, 0.45, 1.2, 0.6, log(0.55)]
    val0, g_an, b_base, ok = DRModels._phylo_mean_laplace_hetero_fg(
        Val(:nb2_hetero), aux_from, n, Xμ, Xσ, leaf_node, Q, logdetQ, θ;
        grad = true, b0 = zeros(q), newton_tol = NTOL, newton_maxiter = NMAX,
    )
    @test ok && isfinite(val0) && val0 < 1e17

    g_fd = zeros(length(θ))
    for k in eachindex(θ)
        tp = copy(θ); tp[k] += FDH
        tm = copy(θ); tm[k] -= FDH
        fp = DRModels._phylo_mean_laplace_hetero_fg(Val(:nb2_hetero), aux_from, n, Xμ, Xσ,
            leaf_node, Q, logdetQ, tp; grad = false, b0 = copy(b_base),
            newton_tol = NTOL, newton_maxiter = NMAX)[1]
        fm = DRModels._phylo_mean_laplace_hetero_fg(Val(:nb2_hetero), aux_from, n, Xμ, Xσ,
            leaf_node, Q, logdetQ, tm; grad = false, b0 = copy(b_base),
            newton_tol = NTOL, newton_maxiter = NMAX)[1]
        g_fd[k] = (fp - fm) / (2FDH)
    end
    max_abs_diff = maximum(abs, g_an .- g_fd)
    @info "NB2 hetero-σ phylo gradient gate (#164)" max_abs_diff g_an g_fd
    @test max_abs_diff ≤ 1e-6
end

@testset "NB2 covariate dispersion + mean phylo RE — drm() recovery (#164)" begin
    Random.seed!(20260611)
    p = 40; m = 12
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    βμ = [0.20, 0.35]
    βσ = [1.3, 0.6]                              # log-size = 1.3 + 0.6 x
    σphy = 0.45
    C = sigma_phy_dense(phy; σ²_phy = σphy^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    sz = exp.(βσ[1] .+ βσ[2] .* x)
    μ = exp.(βμ[1] .+ βμ[2] .* x .+ u[species])
    y = Float64.([rand(Distributions.NegativeBinomial(sz[i], sz[i] / (sz[i] + μ[i]))) for i in 1:n])

    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x)),
              NegBinomial2(); data = (; y, x, species), tree = phy, se = false)

    @test fit.converged
    @test length(coef(fit, :sigma)) == 2            # intercept + slope on log-size
    @test coef(fit, :mu)[2] ≈ βμ[2] atol = 0.20
    @test coef(fit, :sigma)[2] ≈ -0.5 * βσ[2] atol = 0.30  # the heteroscedastic σ slope (log σ = -0.5 log size)
    @test re_sd(fit)[:species] > 0.10
    @test isfinite(loglik(fit))
    @test all(fitted(fit) .> 0)

    # Loose public-path FD sanity (the tight ≤1e-6 gate is the dedicated _fg gate
    # above; here the inner mode is at the fit's default tolerance, matching the
    # sibling NB2/Gamma phylo-Laplace tests' rtol/atol).
    θ = coef(fit)
    g = zeros(length(θ)); fit.nllgrad(g, θ)
    h = 1e-4; fd = similar(g)
    for k in eachindex(θ)
        e = zeros(length(θ)); e[k] = h
        fd[k] = (fit.nll(θ .+ e) - fit.nll(θ .- e)) / (2h)
    end
    @test g ≈ fd rtol = 5e-3 atol = 5e-3
end

@testset "NB2 hetero σ reduces to the scalar nuisance path on a 1-column Xσ (#164)" begin
    Random.seed!(2051)
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
    sizep = 4.0
    yint = [rand(Distributions.NegativeBinomial(sizep, sizep / (sizep + μi))) for μi in μ]
    yf = Float64.(yint)

    aux_scalar(ls) = (y = yf, size = exp(clamp(-2 * ls, -8, 8)),
        lconst = [loggamma(yf[i] + exp(clamp(-2 * ls, -8, 8))) - loggamma(exp(clamp(-2 * ls, -8, 8))) -
                  DRModels._logfactorial(yint[i]) for i in eachindex(yint)])
    aux_hetero = _nb2_hetero_aux(yint, yf)
    Xσ1 = ones(n, 1)

    θ = [0.1, 0.4, log(3.0), log(0.55)]           # same layout: pσ = 1
    NTOL = 1e-10; NMAX = 400
    vs, gs, _, oks = DRModels._phylo_mean_laplace_nuisance_fg(
        Val(:nb2_fixed), aux_scalar, n, Xμ, leaf_node, Q, logdetQ, θ;
        grad = true, b0 = zeros(q), newton_tol = NTOL, newton_maxiter = NMAX)
    vh, gh, _, okh = DRModels._phylo_mean_laplace_hetero_fg(
        Val(:nb2_hetero), aux_hetero, n, Xμ, Xσ1, leaf_node, Q, logdetQ, θ;
        grad = true, b0 = zeros(q), newton_tol = NTOL, newton_maxiter = NMAX)
    @test oks && okh
    @test vs ≈ vh atol = 1e-12
    @test gs ≈ gh atol = 1e-10
end
