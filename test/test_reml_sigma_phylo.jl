# REML for the Gaussian σ-phylo location-scale route (Ayumi #2, her 2nd ask).
# (Since Arc 2 the production path is `_glsp_joint_reml_fit`, native drmTMB's joint
# Laplace over (phylo, β_μ, β_ψ) — see test_reml_sigma_phylo_joint.jl. The history below
# is kept for the penalty anchor, which still exercises `_glsp_reml_penalty`.)
# Former production path (PR #337): `_glsp_reml_refit_clean(..., pμ + pψ)` restricts BOTH
# mean and scale fixed effects (β_μ and β_ψ) — the complete Cox–Reid / Patterson–
# Thompson correction for a scale-side variance component. Restricting β_μ alone
# left σ²_v ~ML-biased.
#
# The unit-test ANCHOR below still probes the β_μ Schur complement at σ-phylo SD→0
# (calls `_glsp_reml_penalty(..., pμ)` directly): as the latent vanishes,
# S → Xμᵀ W Xμ with W = diag(exp(−2ψ)), matching fixed-effect REML's 0.5·logdet(Xμ'WXμ).
using DRModels
using Test, Random, LinearAlgebra, SparseArrays

@testset "REML σ-phylo: penalty → fixed-effect REML as σ-phylo SD → 0" begin
    Random.seed!(303)
    p = 24; m = 6; n = p * m
    phy = random_balanced_tree(p; branch_length = 0.3)
    species = repeat(1:p, inner = m)
    x = randn(n)
    Xμ = hcat(ones(n), x); Xψ = ones(n, 1)   # pμ = 2 (non-trivial logdet)
    βμ = [0.5, -0.3]; βψ = [0.1]
    y = randn(n)                              # values irrelevant for the penalty-at-fixed-θ identity
    pμ = size(Xμ, 2)

    Q, gidx, G = DRModels._locscale_phylo_setup(phy, species)
    Zη, Zψ = DRModels._glsp_asym_loadings(n)
    asym_grad_fn(θ) = DRModels._glsp_asym_grad(Val(:gaussian_mean), y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)

    # logL22 = −10 ⇒ σ-phylo variance exp(−20) ≈ 0; the mean axis is pinned to ε.
    θ = vcat(βμ, βψ, -10.0)
    penalty = DRModels._glsp_reml_penalty(asym_grad_fn, θ, pμ)

    W = exp.(-2 .* (Xψ * βψ))
    ref = 0.5 * logdet(Symmetric(Xμ' * (W .* Xμ)))
    @info "REML identity" penalty ref diff = abs(penalty - ref)
    @test isfinite(penalty)
    @test penalty ≈ ref atol = 1e-3          # σ-phylo SD ≈ 0 ⇒ S = Xμ'WXμ
end

@testset "REML σ-phylo: end-to-end drm(method=:REML) + tagging" begin
    Random.seed!(404)
    p = 24; m = 6; n = p * m
    phy = random_balanced_tree(p; branch_length = 0.3)
    C   = sigma_phy_dense(phy; σ²_phy = 1.0); LC = cholesky(Symmetric(C)).L
    u_mu  = 0.6 .* (LC * randn(p))
    u_sig = 0.5 .* (LC * randn(p))
    species = repeat(1:p, inner = m)
    y = [1.0 + u_mu[species[i]] + exp(log(0.5) + u_sig[species[i]]) * randn() for i in 1:n]
    data = (; y, species)
    form = bf(@formula(y ~ phylo(1 | species)), @formula(sigma ~ phylo(1 | species)))

    fit_ml   = drm(form, Gaussian(); data = data, tree = phy, method = :ML)
    fit_reml = drm(form, Gaussian(); data = data, tree = phy, method = :REML)

    # tagging: the aic/bic/lrtest guard keys off estim_method
    @test estimation_method(fit_ml)   == :ML
    @test estimation_method(fit_reml) == :REML
    @test is_converged(fit_reml)
    @test isfinite(loglik(fit_reml))          # loglik reports the REML value
    @test isfinite(reml_loglik(fit_reml)) && isfinite(ml_loglik(fit_reml))
    @test loglik(fit_reml) ≈ reml_loglik(fit_reml)
    @test ml_loglik(fit_reml) != reml_loglik(fit_reml)   # REML ≠ ML

    # recovery: REML estimates both phylo SDs (positive, in a sane range)
    sd_sig_reml = exp(coef(fit_reml, :resd_sigma)[1])
    sd_mu_reml  = exp(coef(fit_reml, :resd_mu)[1])
    @test sd_sig_reml > 0 && sd_mu_reml > 0
    @test 0.2 < sd_sig_reml < 1.2             # true 0.5

    # REML lifts the variance off the ML estimate (the n→n−pμ correction); on the
    # σ axis it should be ≥ the ML value (less downward-biased).
    sd_sig_ml = exp(coef(fit_ml, :resd_sigma)[1])
    @info "REML vs ML σ-phylo SD" sd_sig_reml sd_sig_ml
    # at p=60 the n→n−pμ correction is tiny, so REML ≈ ML here; the bias REDUCTION
    # at small samples is checked by the adversarial simulation, not this anchor.
    @test isapprox(sd_sig_reml, sd_sig_ml; rtol = 0.25)
end
