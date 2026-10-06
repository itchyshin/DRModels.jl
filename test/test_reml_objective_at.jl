# test_reml_objective_at.jl — pin #575's diagnostic API: DRModels.reml_objective_at
# evaluates the q4 REML objective at a SUPPLIED phi (variance components +
# beta_rho), with beta_mu/beta_sigma reprofiled by the same conditional-Newton
# machinery `fit_q4_reml` uses internally. Built to let cross-engine mode-finder
# vs objective-translation diagnosis (#575) be reproduced without ad-hoc
# scratch scripts.
#
#   julia --project=. -e 'using DRModels, Test; include("test/test_reml_objective_at.jl")'

module TestReplObjectiveAt

using DRModels
using Test
using TOML
using LinearAlgebra
using DelimitedFiles: readdlm

const FIXTURE = joinpath(@__DIR__, "parity", "q4-reml", "biv-q4-phylo-reml")

function _load_data(dir)
    raw, header = readdlm(joinpath(dir, "data.csv"), ','; header = true)
    cols = Symbol.(strip.(string.(vec(header))))
    numeric = Set((:y1, :y2, :x))
    pairs = map(enumerate(cols)) do (j, name)
        col = raw[:, j]
        if name in numeric
            name => Float64[parse(Float64, string(v)) for v in col]
        else
            name => string.(col)
        end
    end
    return NamedTuple(pairs)
end

const DAT = _load_data(FIXTURE)
const TREE = read(joinpath(FIXTURE, "tree.newick"), String)
const FORM = bf(mu1    = @formula(y1 ~ x + phylo(1 | species)),
                 mu2    = @formula(y2 ~ x + phylo(1 | species)),
                 sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
                 sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
                 rho12  = @formula(rho12 ~ 1))

@testset "reml_objective_at reproduces #575's probe numbers (biv q4 phylo REML)" begin
    rhs = Dict(FORM.forms)
    fixed, marker = DRModels._bivariate_q4_marker(rhs)
    grp = marker[2]
    lc_zero = length(marker) >= 3 ? marker[3] : Int[]
    phy = DRModels._as_augmented_phy(TREE)

    y1, X1, _ = DRModels._design(FORM.response1, fixed[:mu1], DAT)
    y2, X2, _ = DRModels._design(FORM.response2, fixed[:mu2], DAT)
    _, Xs1, _ = DRModels._design(FORM.response1, fixed[:sigma1], DAT)
    _, Xs2, _ = DRModels._design(FORM.response1, fixed[:sigma2], DAT)
    _, Xr, _  = DRModels._design(FORM.response1, fixed[:rho12], DAT)

    obs1 = DRModels._observed_response_mask(y1)
    obs2 = DRModels._observed_response_mask(y2)
    species = DRModels._phylo_species_index(phy, getproperty(DAT, grp))
    prob, Q_cond = DRModels.make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr; species = species)

    β1 = X1[obs1, :] \ y1[obs1]
    β2 = X2[obs2, :] \ y2[obs2]
    res1 = y1[obs1] .- X1[obs1, :] * β1
    res2 = y2[obs2] .- X2[obs2, :] * β2
    β0 = (mu1 = β1, mu2 = β2,
          s1 = DRModels._initial_scale_beta(Xs1, res1), s2 = DRModels._initial_scale_beta(Xs2, res2),
          rho = zeros(size(Xr, 2)))
    Λ0 = Matrix(Symmetric([
        0.30 0.02 0.01 0.010
        0.02 0.30 0.01 0.010
        0.01 0.01 0.08 0.005
        0.01 0.01 0.005 0.080
    ]))
    if !isempty(lc_zero)
        lc0 = DRModels.Λ_to_lc(Λ0); lc0[lc_zero] .= 0.0; Λ0 = DRModels.lc_to_Λ(lc0)
    end

    # Julia's own REML optimum (same call as test_q4_reml_warm_restart.jl).
    g_tol = 1e-3
    rr = DRModels.fit_q4_reml(prob, Q_cond; beta0 = β0, Lambda0 = Λ0,
                          g_tol = g_tol, iterations = 300, n_newton = 40, lc_zero = lc_zero)
    @test rr.converged == true

    obj_at_julia_hat = reml_objective_at(prob, Q_cond, rr.phi;
                                          beta0 = (mu1 = rr.beta.mu1, mu2 = rr.beta.mu2,
                                                   s1 = rr.beta.s1, s2 = rr.beta.s2,
                                                   rho = rr.beta.rho),
                                          u0 = rr.u_hat, n_newton = 40)
    # atol here (2e-4) is NOT numerical slop in reml_objective_at itself: the
    # inner alternation's own documented early-exit criterion
    # (`delta_b < 1e-4 * (1 + ‖β‖)`, reml_q4.jl reml_ll_and_mode) means a
    # SECOND call seeded from a converged (u_hat, beta) pair can settle a
    # fraction of a limit-cycle step away from the first call's exact fixed
    # point, which the Schur-complement logdet amplifies slightly beyond
    # beta-scale. Both #575's frozen probe (a different worktree/commit) and
    # this in-process reproduction sit inside that same noise floor, and it is
    # 150x tighter than the fixture's own cross-optimum atol_loglik = 0.03.
    # HISTORY: on the finite-difference mode-finder this pinned -219.630231 (the
    # #575 "suboptimal basin"). After #579 (exact REML gradient, merged f930e8bf)
    # fit_q4_reml's own optimum is -219.614005 (2e-5 from drmTMB's -219.613986),
    # so the point this test fits to moved; the value AT that point is what is
    # pinned, and it must agree with the fit's own report (next assertion).
    @test isapprox(obj_at_julia_hat.reml_loglik, -219.614005; atol = 2e-4)
    # Self-consistency: evaluating at Julia's OWN optimum must reproduce
    # fit_q4_reml's own reported reml_loglik, not just #575's frozen number.
    @test isapprox(obj_at_julia_hat.reml_loglik, rr.reml_loglik; atol = 2e-4)

    # TMB's fitted point (#575's R refit: expected.toml coefs + report()$phylo_q4_covariance).
    expected = TOML.parsefile(joinpath(FIXTURE, "expected.toml"))
    rho12_tmb = Float64(expected["coef"]["rho12_(Intercept)"])
    Lambda_tmb = [
        0.5288095   0.25509007 -0.1228962  -0.15554224
        0.25509007  0.28551843 -0.1794685  -0.02445802
       -0.1228962  -0.1794685   0.4857264  -0.10269499
       -0.15554224 -0.02445802 -0.10269499  0.16678147
    ]
    phi_tmb = DRModels.pack_phi(prob, [rho12_tmb], Lambda_tmb)
    beta0_tmb = (mu1 = [Float64(expected["coef"]["mu1_(Intercept)"]), Float64(expected["coef"]["mu1_x"])],
                 mu2 = [Float64(expected["coef"]["mu2_(Intercept)"]), Float64(expected["coef"]["mu2_x"])],
                 s1  = [Float64(expected["coef"]["sigma1_(Intercept)"])],
                 s2  = [Float64(expected["coef"]["sigma2_(Intercept)"])],
                 rho = [rho12_tmb])

    obj_at_tmb_hat = reml_objective_at(prob, Q_cond, phi_tmb; beta0 = beta0_tmb)
    @test isapprox(obj_at_tmb_hat.reml_loglik, -219.620508; atol = 2e-4)

    # The point at issue in #575: DRModels.jl's own objective at TMB's theta is
    # BETTER than what DRModels.jl's own solver returned (mode-finder gap, not an
    # objective-translation difference).
    # HISTORY: pre-#579 this read `obj_at_tmb_hat > rr` — DRModels.jl's own objective
    # was BETTER at TMB's point than at its own optimum, the #575 mode-finder
    # diagnosis. Post-#579 the fitter's optimum is at least as good as any
    # probe point, TMB's included; the diagnostic primitive itself is unchanged.
    @test rr.reml_loglik >= obj_at_tmb_hat.reml_loglik - 1e-8
end

end # module TestReplObjectiveAt
