module TestQ4REMLObjectiveConvergence

using DRModels
using Test
using LinearAlgebra
using DelimitedFiles: readdlm

const FIXTURE = joinpath(@__DIR__, "parity", "q4-reml", "biv-q4-phylo-reml")
const GLSP_FIXTURE = joinpath(@__DIR__, "..", "docs", "dev-log", "evidence",
                              "arc2-gaussian-sigma-phylo-reml")

function _data()
    raw, header = readdlm(joinpath(FIXTURE, "data.csv"), ','; header = true)
    names = Symbol.(strip.(string.(vec(header))))
    numeric = Set((:y1, :y2, :x))
    return NamedTuple(map(enumerate(names)) do (j, name)
        values = raw[:, j]
        name => (name in numeric ?
            Float64[parse(Float64, string(value)) for value in values] :
            string.(values))
    end)
end

const FORM = bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
                mu2 = @formula(y2 ~ x + phylo(1 | species)),
                sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
                sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
                rho12 = @formula(rho12 ~ 1))

@testset "q4 phylogenetic REML convergence uses the profiled objective" begin
    @test isfile(joinpath(FIXTURE, "data.csv"))
    dat = _data()
    tree = read(joinpath(FIXTURE, "tree.newick"), String)
    fit = drm(FORM, Gaussian(); data = dat, tree = tree,
              method = :REML, q4_vcov = true)

    @test fit.nll isa DRModels._FitObjectiveGradient
    callback = fit.nll
    theta = Vector{Float64}(fit.theta)
    p_beta = sum(length(last(pair)) for pair in fit.blocks
                 if first(pair) in (:mu1, :mu2, :sigma1, :sigma2))
    phi = theta[(p_beta + 1):end]

    exact_reml_nll = DRModels.reml_nll_exact(
        fit.ranef.prob, fit.ranef.Q_cond, phi; n_newton = fit.ranef.n_newton)
    ml_profile_nll = DRModels.marginal_nll(
        fit.ranef.prob, fit.ranef.Q_cond, theta;
        n_newton = fit.ranef.n_newton)[1]

    @test callback.objective(theta) ≈ exact_reml_nll atol = 1e-7 rtol = 1e-9
    @test callback(theta) ≈ ml_profile_nll atol = 1e-9 rtol = 1e-10
    @test callback.profile_objective(theta) ≈ ml_profile_nll atol = 1e-9 rtol = 1e-10
    @test !isapprox(callback.objective(theta), callback(theta); atol = 1e-4, rtol = 1e-6)

    gradient = zeros(length(theta))
    callback.gradient!(gradient, theta)
    _, phi_gradient, _, _, _, _ = DRModels.reml_nll_and_exact_grad(
        fit.ranef.prob, fit.ranef.Q_cond, phi; n_newton = fit.ranef.n_newton)
    @test all(iszero, gradient[1:p_beta])
    @test gradient[(p_beta + 1):end] ≈ phi_gradient atol = 1e-7 rtol = 1e-6

    expected_relative = maximum(abs.(gradient) .* max.(1.0, abs.(theta))) /
                        max(1.0, abs(callback.objective(theta)))
    @test DRModels._relative_objective_gradient(fit) ≈ expected_relative atol = 1e-12
    @test expected_relative <= DRModels._TWIN_RELATIVE_GRADIENT_TOLERANCE

    # q4_vcov=true retains the finite ML marginal covariance at the REML point.
    # It cannot certify REML convergence because the restricted curvature is
    # unavailable; that limitation is explicit on the fit.
    @test fit.uncertainty === :skipped
    @test fit.geometry === :reml_curvature_unavailable
    @test all(isfinite, fit.vcov)
    @test isposdef(Symmetric(fit.vcov))
    @test DRModels.convergence_status(fit) === :degenerate
end

function _glsp_g1_fixture()
    lines = readlines(joinpath(GLSP_FIXTURE, "fixture-G1.csv"))
    rows = [split(line, ",") for line in lines[2:end]]
    data = (y = [parse(Float64, row[1]) for row in rows],
            x = [parse(Float64, row[2]) for row in rows],
            z = [parse(Float64, row[3]) for row in rows],
            sp = [String(strip(row[4], '"')) for row in rows])
    tree = String(strip(read(joinpath(GLSP_FIXTURE, "fixture-G1.nwk"), String)))
    return data, tree
end

@testset "GLSP REML correlation edge has a feasible-tangent score" begin
    dat, tree = _glsp_g1_fixture()
    form = bf(@formula(y ~ x + phylo(1 | sp)),
              @formula(sigma ~ phylo(1 | sp)))
    fit = drm(form, Gaussian(); data = dat, tree = tree,
              method = :REML, phylo_coupled = true, g_tol = 1e-8)

    cap = DRModels._GLSP_COR_CAP
    @test fit.scales[:lambda_cor][1] ≈ cap atol = 1e-9
    @test all(isnan, fit.vcov[end, :])
    @test DRModels.convergence_status(fit) === :boundary

    @test fit.nll isa DRModels._FitObjectiveGradient
    theta = Vector{Float64}(fit.theta)
    p = length(theta) - 3
    v = theta[(p + 1):end]
    edge_sign = sign(v[2])
    offset = 0.5 * log1p(-cap^2)
    w2 = v[3] - offset
    c = edge_sign * cap

    gradient = zeros(length(theta))
    fit.nll.gradient!(gradient, theta)
    tangent = zeros(length(theta))
    tangent[p + 2] = c * exp(w2)
    tangent[p + 3] = 1.0

    h = 1e-4
    theta_plus = copy(theta)
    theta_minus = copy(theta)
    theta_plus[p + 2] = c * exp(w2 + h)
    theta_minus[p + 2] = c * exp(w2 - h)
    theta_plus[p + 3] = w2 + h + offset
    theta_minus[p + 3] = w2 - h + offset
    finite_difference = (fit.nll.objective(theta_plus) - fit.nll.objective(theta_minus)) / (2h)
    @test dot(gradient, tangent) ≈ finite_difference atol = 1e-5 rtol = 1e-4
end

end # module TestQ4REMLObjectiveConvergence
