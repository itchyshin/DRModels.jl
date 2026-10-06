# test_reml_vcov_curvature.jl — issue #310.
#
# Under method = :REML the reported Wald vcov for the σ-phylo location-scale routes
# must be the inverse Hessian of the RESTRICTED objective `nll_ML + 0.5·logdet S`
# (the true REML observed information), NOT the ML observed information evaluated at
# the REML point (which omits the restricted-penalty curvature and mis-states
# variance-component uncertainty). `_glsp_reml_vcov` forms the restricted-gradient
# FD Hessian; this test anchors that it (a) is finite/PD on a well-identified fit,
# (b) DIFFERS from the ML observed information at the same θ̂ in the variance
# blocks, and (c) the ML/REML curvatures coincide when the penalty is flat.
using DRModels
using Test, Random, LinearAlgebra

@testset "REML vcov includes restricted-penalty curvature (#310)" begin
    Random.seed!(11)
    p = 20; m = 10; n = p * m
    phy = random_balanced_tree(p; branch_length = 0.5)
    C = sigma_phy_dense(phy; σ²_phy = 1.0); LC = cholesky(Symmetric(C)).L
    u_mu = 0.8 .* (LC * randn(p)); u_sig = 0.6 .* (LC * randn(p))
    species = repeat(1:p, inner = m)
    x = randn(n)
    y = [0.5 + 0.4 * x[i] + u_mu[species[i]] +
         exp(log(0.6) + u_sig[species[i]]) * randn() for i in 1:n]
    Xμ = hcat(ones(n), x); Xψ = ones(n, 1)
    pμ = size(Xμ, 2)                                  # non-trivial mean design (pμ = 2)

    Q, gidx, G = DRModels._locscale_phylo_setup(phy, species)
    kind = Val(:gaussian_mean)
    Zη = DRModels._ls_canonical_Zeta(n); Zψ = DRModels._ls_canonical_Zpsi(n)
    sep_grad_fn(θ) = DRModels._glsp_sep_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)
    sep_obj(θ)     = DRModels._glsp_sep_nll(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)

    # ML fit, then REML refit (the production clean-gradient path).
    βμ0 = Xμ \ y
    θ0 = vcat(βμ0, [0.0], log(0.3), log(0.3))
    θ̂_ml, conv = DRModels._glsp_optimise(sep_obj, (g, θ) -> (g .= sep_grad_fn(θ); g), θ0)
    θ̂, _, _, _, _ = DRModels._glsp_reml_refit_clean(sep_obj, sep_grad_fn, θ̂_ml, pμ;
                                               ml_converged = conv)

    # ML observed information at θ̂ (the OLD, incomplete curvature).
    h = 1e-4; np = length(θ̂); Hml = zeros(np, np)
    for j in 1:np
        tp = copy(θ̂); tp[j] += h; tm = copy(θ̂); tm[j] -= h
        Hml[:, j] .= (sep_grad_fn(tp) .- sep_grad_fn(tm)) ./ (2h)
    end
    chml = cholesky(Symmetric(Hml); check = false)
    @test issuccess(chml)                            # well-identified: ML info PD
    Vml = Matrix(inv(chml))

    # REML observed information (the FIX): restricted-objective inverse Hessian.
    Vreml = DRModels._glsp_reml_vcov(sep_grad_fn, θ̂, pμ)
    @test all(isfinite, Vreml)                       # finite/PD on this fit
    @test isposdef(Symmetric((Vreml .+ Vreml') ./ 2))

    se_ml   = sqrt.(diag(Vml))
    se_reml = sqrt.(diag(Vreml))
    # The restricted objective's Hessian differs from the ML observed information by
    # the penalty curvature ∂²(0.5·logdet S)/∂θ², so the REML vcov must NOT equal the
    # ML-curvature-only vcov (the pre-fix behaviour). Anchor on the full diagonal.
    @test !isapprox(se_reml, se_ml; rtol = 1e-3)
    # The correction is a modest, coupled shift (not a boundary blow-up): SEs stay in
    # the same ballpark and remain positive/finite.
    @test all(se_reml .> 0)
    @test all(isfinite, se_reml)
    @test maximum(abs.(se_reml .- se_ml) ./ se_ml) < 0.5   # same order of magnitude

    # pμ = 0 disables the penalty (empty S ⇒ Hpen = 0), so the helper reduces to the
    # ML observed information — a construction check that the penalty is the ONLY
    # difference between the REML and ML curvatures.
    Vml_via_helper = DRModels._glsp_reml_vcov(sep_grad_fn, θ̂, 0)
    @test isapprox(diag(Vml_via_helper), diag(Vml); rtol = 1e-4)
end
