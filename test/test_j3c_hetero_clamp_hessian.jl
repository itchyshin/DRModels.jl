# test_j3c_hetero_clamp_hessian.jl -- PR #1080 / Refs #1073.
#
# `_fit_phylo_mean_laplace_hetero` used to pick its Wald Hessian path from the
# phylogenetic log-SD coordinate alone (`sparse_laplace_glmm.jl` :1198).
# `_phylo_mean_laplace_hetero_fg` also clamps the *per-observation* log-sigma
# linear predictor `ησ = Xσ·βσ` (`:1047`, then `aux_from` to the family's
# [-4, 4] for NB2 / Gamma / Beta). A clamped observation makes the objective
# flat in that direction while the analytic gradient is not, so the
# gradient-difference Hessian is undefined: those fits must take the value path.
using DRModels
using Test, LinearAlgebra, Random
import Distributions

@testset "J3c: hetero per-observation log-sigma clamp uses the value Hessian (Refs #1073)" begin
    # Family bound on ησ is the aux_from clamp, not the fg's wider [-8, 8] box.
    @test DRModels._dispersion_clamp(Val(:nb2_hetero)) == 4.0
    @test DRModels._dispersion_clamp(Val(:gamma_hetero)) == 4.0
    @test DRModels._dispersion_clamp(Val(:beta_hetero)) == 4.0

    pμ, pσ = 1, 2
    # ησ = βσ0 + βσ1 * x. Observation 2 sits on the family's +4 clamp; the
    # phylogenetic log-SD is interior, so an SD-only path decision is :gradient.
    Xσ = [1.0  0.0;
          1.0  2.0;
          1.0 -0.5]
    θ = [0.2, 0.5, 1.75, -0.4]          # ησ = [0.5, 4.0, -0.375]
    ησ = Xσ * θ[pμ+1:pμ+pσ]
    @test any(η -> abs(η) >= 4.0 - 1e-3, ησ)
    @test θ[end] > DRModels._SE_LOGSD_LO
    @test DRModels._se_hessian_path(θ; sd = (pμ + pσ + 1):(pμ + pσ + 1),
                                    sd_lo = DRModels._SE_LOGSD_LO) === :gradient

    # The fitter must also pass the per-observation ησ. On the PR head before
    # this slice that keyword did not exist, so this assertion is the red-proof.
    @test DRModels._se_hessian_path(θ; sd = (pμ + pσ + 1):(pμ + pσ + 1),
                                    sd_lo = DRModels._SE_LOGSD_LO,
                                    eta = ησ,
                                    disp_bound = DRModels._dispersion_clamp(Val(:nb2_hetero))) === :value

    # Interior ησ (3.5 is inside NB2's [-4, 4]; J3a review: do not treat it as a clamp)
    θ_ok = [0.2, 0.5, 1.5, -0.4]        # ησ = [0.5, 3.5, -0.25]
    @test DRModels._se_hessian_path(θ_ok; sd = 4:4, sd_lo = DRModels._SE_LOGSD_LO,
                                    eta = Xσ * θ_ok[2:3], disp_bound = 4.0) === :gradient

    # End to end: generate past the family's ±4 bound so the fitted ησ hits the
    # clamp while the phylo log-SD stays interior. The SE block must warn and
    # fall back (never throw).
    rng = MersenneTwister(1080)
    p, m = 8, 6
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = collect(range(-1.7, 1.7; length = n))
    C = sigma_phy_dense(phy; σ²_phy = 0.35^2)
    u = cholesky(Symmetric(C)).L * randn(rng, p)
    # log σ = 0 + 2.6 x → tails past ±4; size r = exp(-2 log σ)
    ησ_true = 2.6 .* x
    sz = exp.(-2 .* clamp.(ησ_true, -4.0, 4.0))
    μ = exp.(0.15 .+ 0.25 .* x .+ u[species])
    y = Float64.([rand(rng, Distributions.NegativeBinomial(sz[i], sz[i] / (sz[i] + μ[i]))) for i in 1:n])
    fit = @test_logs (:warn, r"falling back to the value-based finite-difference Hessian") match_mode = :any drm(
        bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x)), NegBinomial2();
        data = (; y, x, species), tree = phy)
    ησ_hat = hcat(ones(n), x) * coef(fit, :sigma)
    @test any(η -> abs(η) >= 4.0 - 1e-3, ησ_hat)
    @test fit.theta[end] > DRModels._SE_LOGSD_LO
    @test size(vcov(fit)) == (length(fit.theta), length(fit.theta))
end

