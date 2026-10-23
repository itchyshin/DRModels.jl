# test_q4_reml_warm_restart.jl — pin #484: drm(..., method = :REML) converges on the
# q4 phylo REML cell through public kwargs alone.
#
# Background: the REML LBFGS used to take ZERO accepted line-search steps from the
# ML warm start on this cell (a starting-value problem, not slow convergence — none
# of drm()'s three exposed knobs, q4_g_tol/q4_iterations/q4_n_newton, could move it;
# see docs/dev-log/evidence/2026-08-24-biv-q4-phylo-reml-converged.md). `fit_q4_reml`
# (src/reml_q4.jl) now detects that exact stall automatically (x == phi0 and
# !converged) and retries via a warm-restart schedule, judged at the SAME g_tol the
# caller asked for. This file pins that outcome and its consequences.
#
#   julia --project=. -e 'using DRModels, Test; include("test/test_q4_reml_warm_restart.jl")'

module TestQ4RemlWarmRestart

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

@testset "q4 phylo REML: public drm() converges through public kwargs alone (#484)" begin
    # Defaults only — q4_g_tol/q4_iterations/q4_n_newton untouched, exactly the
    # public-API call the issue's acceptance criterion describes.
    fit = drm(FORM, Gaussian(); data = DAT, tree = TREE, method = :REML, q4_vcov = false)
    @test estimation_method(fit) === :REML
    @test is_converged(fit) == true
    @test isfinite(loglik(fit))
end

@testset "q4 phylo REML: engine-level g_residual < g_tol, not just the flag (#484)" begin
    # Reproduce, at the engine level, exactly the inputs `_fit_bivariate_q4_phylo`
    # builds for this call (same internal helpers, no `src/` edit) so the
    # returned NamedTuple's `g_residual` — not surfaced on `DrmFit` — can be
    # checked directly against the SAME `g_tol` the public default uses.
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

    g_tol = 1e-3   # drm()'s default `q4_g_tol`
    rr = DRModels.fit_q4_reml(prob, Q_cond; beta0 = β0, Lambda0 = Λ0,
                          g_tol = g_tol, iterations = 300, n_newton = 40, lc_zero = lc_zero)

    @test rr.converged == true
    @test rr.g_residual < g_tol   # the residual criterion itself, not just the flag

    # #484: does the restart find the RIGHT optimum, or merely a converged one?
    # The check is the same as before and is now strictly sharper.
    #
    # Until #477 this route reported the UNNORMALISED restricted log-likelihood,
    # so the expected gap against drmTMB's native REML was the integration
    # constant (n_beta/2)*log(2*pi) = 5.513631 with n_beta = 6 (mu1 + mu2 +
    # sigma1 + sigma2 design widths = 2+2+1+1), and this test allowed a 5.5136
    # offset. Both engines now report the normalised scale, so there is no
    # offset left to allow: the gap must be ZERO to within this cell's
    # cross-optimum reml_ll spread. Measured at the #477 change: 0.001938.
    #
    # This assertion is what caught the change propagating — it failed with
    # `isapprox(0.001938, 5.513631)` the moment the constant was added, which is
    # independent confirmation (separate from the parity fixture) that the whole
    # gap really was the constant.
    expected = TOML.parsefile(joinpath(FIXTURE, "expected.toml"))
    tmb_reml = Float64(expected["fit"]["loglik"])
    measured_gap = tmb_reml - rr.reml_loglik
    @test isapprox(measured_gap, 0.0; atol = 0.05)
end

end # module TestQ4RemlWarmRestart
