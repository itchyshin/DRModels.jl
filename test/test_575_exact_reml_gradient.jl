# test_575_exact_reml_gradient.jl — issue #575: the EXACT gradient of the q=4
# REML objective must agree with a tight central-difference reference to the
# engine-quality battery's own bar (1e-6), INCLUDING at TMB's fitted point,
# where the old FD gradient (h = 5e-4, mode re-solved per perturbation) sat at
# or below the noise floor and so could certify convergence in the wrong basin.
#
# Derivation: docs/src/developer-notes/reml-q4-exact-gradient.md
#
#   julia --project=. -e 'using DRModels, Test; include("test/test_575_exact_reml_gradient.jl")'

module Test575ExactRemlGradient

using DRModels
using Test
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

# Rebuild exactly what `_fit_bivariate_q4_phylo` hands the engine (same helpers,
# no src edit) — mirrors test_q4_reml_warm_restart.jl.
function _engine_inputs()
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
    return prob, Q_cond, β0, lc_zero
end

# TMB's fitted point on this fixture (scratchpad/575-mechanism.md §1): the
# `phylo_q4_covariance` report in axis order (mu1, mu2, sigma1, sigma2) and the
# rho12 intercept. This is the point the FD gradient could not resolve.
const LAMBDA_TMB = [ 0.5288095   0.25509007 -0.1228962  -0.15554224;
                     0.25509007  0.28551843 -0.1794685  -0.02445802;
                    -0.1228962  -0.1794685   0.4857264  -0.10269499;
                    -0.15554224 -0.02445802 -0.10269499  0.16678147]
const RHO12_TMB = 0.065606409

@testset "issue #575: exact q4 REML gradient matches a tight central difference" begin
    prob, Q_cond, β0, lc_zero = _engine_inputs()

    Λ_start = Matrix(0.3I(4))
    phi_start = DRModels.pack_phi(prob, [0.0], Λ_start)
    phi_tmb   = DRModels.pack_phi(prob, [RHO12_TMB], Matrix(LAMBDA_TMB))

    # A third point: DRModels.jl's own converged optimum through the engine route.
    rr = DRModels.fit_q4_reml(prob, Q_cond; beta0 = β0, Lambda0 = Λ_start,
                         g_tol = 1e-3, iterations = 300, n_newton = 40,
                         lc_zero = lc_zero)
    phi_opt = Vector{Float64}(rr.phi)

    points = ("ML-scale start" => phi_start,
              "DRModels.jl optimum" => phi_opt,
              "TMB fitted point" => phi_tmb)

    for (label, phi) in points
        # Exact gradient of the NEGATIVE REML log-likelihood (unnormalised).
        val, g_exact, _, _, _, zres = DRModels.reml_nll_and_exact_grad(
            prob, Q_cond, phi; beta0 = β0)
        @test isfinite(val)
        @test all(isfinite, g_exact)
        # The exact gradient is only valid AT a joint stationary point: the
        # routine must certify one.
        @test zres < 1e-8

        # Central-difference reference on the SAME (Newton-certified) objective,
        # with a step scan: take the step whose Richardson pair is most stable.
        nph = length(phi)
        g_fd = zeros(nph)
        for k in 1:nph
            best = NaN; best_gap = Inf; prev = NaN
            for h in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
                pp = copy(phi); pp[k] += h
                pm = copy(phi); pm[k] -= h
                fp = DRModels.reml_nll_exact(prob, Q_cond, pp; beta0 = β0)
                fm = DRModels.reml_nll_exact(prob, Q_cond, pm; beta0 = β0)
                d  = (fp - fm) / (2h)
                if isfinite(prev) && abs(d - prev) < best_gap
                    best_gap = abs(d - prev); best = d
                end
                prev = d
            end
            g_fd[k] = best
        end

        err = maximum(abs.(g_exact .- g_fd))
        scale = max(1.0, maximum(abs, g_fd))
        @info "exact-vs-FD REML gradient" point = label max_abs_err = err rel = err / scale
        @test err / scale <= 1e-6
    end
end

end # module Test575ExactRemlGradient
