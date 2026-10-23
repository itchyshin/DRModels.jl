# REML for the Gaussian MEAN-ONLY phylogenetic cell (DRModels.jl #624 item (c)).
#
#   bf(y ~ x + phylo(1 | species), sigma ~ 1),  Gaussian(),  method = :REML
#
# Until now this shape reached the generic univariate REML gate in
# `gaussian_core.jl` and was refused ("a structured mean marker ... without a
# matching sd() submodel"), even though the sparse location-only spine already
# carried a validated restricted objective (`_loconly_reml_components`). This
# file pins the wiring: ML is untouched, REML restricts, and the fit says so.
#
# What is integrated out, and with which constant. `_fit_structured_gaussian_-
# sparse_lbfgs` profiles beta_mu out EXACTLY by GLS at every (log sigma_e,
# log sigma_phylo), and the phylo field u is already integrated out in closed
# form by the Gaussian marginal. REML adds the Patterson-Thompson term
#
#     nll_REML(v) = nll_ML(v, betahat(v)) + 0.5*logdet(Xmu' V^-1 Xmu)
#                                         - 0.5*p_mu*log(2*pi)
#
# so the integrated-out set is exactly {u, beta_mu} -- the same set native
# drmTMB gets by folding `beta_mu` into TMB's `random=` vector, and the same
# additive +0.5*p_mu*log(2*pi) TMB's Laplace approximation contributes. The two
# engines therefore report REML log-likelihoods on ONE convention, with no
# offset to remove (measured tmb-vs-julia agreement 6.56e-10 on the drmTMB
# fixture; see drmTMB docs/dev-log/evidence/julia-r-parity/reml/
# reml-phylo-mean-receipt.md).

using DRModels
using Test, Random, LinearAlgebra, Statistics

@testset "Gaussian mean-only phylo: method = :REML" begin
    Random.seed!(20260905)
    phy = random_balanced_tree(16)
    p = phy.n_leaves
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    d = sqrt.(diag(C))
    C = C ./ (d * d')
    u = cholesky(Symmetric(C + 1e-10I)).L * randn(p) .* 0.6
    species = repeat(1:p, inner = 4)
    n = length(species)
    x = randn(n)
    y = 0.4 .+ 0.7 .* x .+ u[species] .+ 0.5 .* randn(n)
    data = (y = y, x = x, species = species)
    f = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1))

    fit_ml = drm(f, Gaussian(); data = data, tree = phy)
    fit_reml = drm(f, Gaussian(); data = data, tree = phy, method = :REML)

    # (i) REML FITS this cell (it threw an ArgumentError before this change).
    @test fit_reml.converged
    @test all(isfinite, fit_reml.theta)

    # (iv) the fit reports the estimator that actually ran, and ML still says ML.
    @test fit_reml.estim_method === :REML
    @test fit_ml.estim_method === :ML
    @test estimation_method(fit_reml) === :REML

    # (ii) the REML log-likelihood is a DIFFERENT number from the ML one -- a
    # genuine restriction, not an ML fit wearing a REML label. The gap is the
    # Patterson-Thompson correction, which is strictly positive in nll here
    # (p_mu = 2 columns of Xmu), so the REML loglik sits BELOW the ML loglik.
    @test fit_reml.loglik != fit_ml.loglik
    @test fit_reml.loglik < fit_ml.loglik
    @test fit_reml.reml_loglik == fit_reml.loglik
    @test isfinite(fit_reml.ml_loglik)
    # ml_loglik is the plain ML value AT THE REML OPTIMUM, so it is at most the
    # ML maximum and the two differ (the optima are not the same point).
    @test fit_reml.ml_loglik <= fit_ml.loglik + 1e-8
    @test fit_reml.ml_loglik != fit_reml.reml_loglik

    # ML PATH UNTOUCHED: the ML objective is byte-for-byte the one it was, so
    # this value must equal the pre-change measurement recorded in the leaf
    # ledger (-61.46784165162242, same seed/fixture).
    @test fit_ml.estim_method === :ML
    @test isnan(fit_ml.reml_loglik)
    @test fit_ml.loglik ≈ -61.46784165162242 rtol = 1e-10

    # (iii) THE DIRECTION THEORY PREDICTS. REML removes the downward bias ML
    # carries in variance components because ML ignores the p_mu degrees of
    # freedom spent estimating beta_mu. So the REML phylogenetic SD (theta's
    # last entry, on the log scale) must be at least the ML one, while the mean
    # fixed effects -- which are the GLS/BLUE solution at the fitted covariance
    # either way -- barely move.
    pμ = 2
    lσ_ml, lσphy_ml = fit_ml.theta[pμ + 1], fit_ml.theta[pμ + 2]
    lσ_reml, lσphy_reml = fit_reml.theta[pμ + 1], fit_reml.theta[pμ + 2]
    @test lσphy_reml > lσphy_ml
    @test isapprox(fit_reml.theta[1:pμ], fit_ml.theta[1:pμ]; atol = 5e-2)
    @test isfinite(lσ_reml)

    # The reported REML objective agrees with the route's own restricted
    # components evaluated at the REML optimum, and with the DENSE oracle in
    # the same file (a different assembly of the same restricted likelihood).
    prob = DRModels.make_loc_problem(phy, y, hcat(ones(n), x); species = species)
    comp = DRModels._loconly_reml_components(prob, lσ_reml, lσphy_reml)
    @test comp.converged
    @test -comp.nll ≈ fit_reml.reml_loglik rtol = 1e-9
    @test -comp.ml_nll ≈ fit_reml.ml_loglik rtol = 1e-9
    dense = DRModels._loconly_dense_reml_components(prob, lσ_reml, lσphy_reml)
    @test dense.nll ≈ comp.nll rtol = 1e-7 atol = 1e-7

    # The REML optimum really is a stationary point OF THE RESTRICTED objective
    # (not of the ML one): the restricted objective at the REML minimiser is no
    # larger than at the ML minimiser.
    @test DRModels._loconly_reml_nll(prob, lσ_reml, lσphy_reml) <=
          DRModels._loconly_reml_nll(prob, lσ_ml, lσphy_ml) + 1e-8

    # Standard errors are finite on every block (the variance block uses the
    # RESTRICTED curvature under REML), and the mean block is the canonical
    # REML/GLS (Xmu' V^-1 Xmu)^-1.
    # OBJECTIVE HONESTY: `fit.nll` on a REML fit must be the RESTRICTED objective,
    # not the ML one. Attaching the ML objective/score would report an ML gradient
    # at the REML optimum -- measured (1.01, 0.99) on the two variance parameters,
    # which reads as "not converged" for a converged fit -- through the R bridge's
    # `gradient` field. The ML fit keeps its analytic score; the REML fit carries
    # none, because there is no analytic score for the log-determinant term here.
    @test fit_reml.nll !== nothing
    @test fit_reml.nll(fit_reml.theta) ≈ -fit_reml.reml_loglik rtol = 1e-9
    @test fit_reml.nllgrad === nothing
    @test fit_ml.nll(fit_ml.theta) ≈ -fit_ml.loglik rtol = 1e-9
    @test fit_ml.nllgrad !== nothing

    se = sqrt.(abs.(diag(fit_reml.vcov)))
    @test all(isfinite, se)
    @test all(se .> 0)
    σ²_reml = exp(2 * lσ_reml)
    _, _, chM, _ = DRModels.build_M(prob, exp(2 * lσphy_reml), σ²_reml)
    VX = DRModels.Vinv_mul(prob, chM, σ²_reml, prob.X)
    gls_se = sqrt.(diag(inv(Symmetric(0.5 .* (prob.X' * VX .+ VX' * prob.X)))))
    @test se[1:pμ] ≈ gls_se rtol = 1e-8
end

@testset "Gaussian mean-only phylo REML: adjacent shapes still refuse" begin
    # The gate widening is scoped to EXACTLY the shape the sparse route serves.
    # These neighbours have NO restricted objective in the engine and must keep
    # throwing rather than return an ML fit labelled :REML.
    Random.seed!(20260906)
    phy = random_balanced_tree(8)
    p = phy.n_leaves
    species = repeat(1:p, inner = 4)
    n = length(species)
    x = randn(n)
    z = randn(n)
    y = 0.4 .+ 0.7 .* x .+ 0.5 .* randn(n)

    # sigma carries a predictor -> DRModels.jl routes to the DENSE structured fitter.
    data_lss = (y = y, x = x, z = z, species = species)
    f_lss = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ z))
    @test_throws ArgumentError drm(f_lss, Gaussian(); data = data_lss, tree = phy,
                                   method = :REML)

    # relmat(1 | id): same closed-form family, but served by the dense
    # structured fitter, which has no restricted objective.
    G = 6
    idx = repeat(1:G, inner = 4)
    m = length(idx)
    xr = randn(m)
    yr = 0.4 .+ 0.7 .* xr .+ 0.5 .* randn(m)
    Kmat = Matrix{Float64}(I, G, G)
    data_rel = (y = yr, x = xr, id = idx)
    f_rel = bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1))
    @test_throws ArgumentError drm(f_rel, Gaussian(); data = data_rel, K = Kmat,
                                   method = :REML)

    # algorithm = :em is the conjugate-EM variant: no REML objective either.
    data_em = (y = y, x = x, species = species)
    f_em = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1))
    @test_throws ArgumentError drm(f_em, Gaussian(); data = data_em, tree = phy,
                                   method = :REML, algorithm = :em)
end
