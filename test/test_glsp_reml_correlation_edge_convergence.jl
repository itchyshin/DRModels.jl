using DRModels
using Test, LinearAlgebra

const _GLSP_G1_FIXTURE = joinpath(@__DIR__, "..", "docs", "dev-log", "evidence",
                                  "arc2-gaussian-sigma-phylo-reml")

function _glsp_g1_data_for_edge_test()
    lines = readlines(joinpath(_GLSP_G1_FIXTURE, "fixture-G1.csv"))
    rows = [split(line, ",") for line in lines[2:end]]
    data = (y = [parse(Float64, row[1]) for row in rows],
            x = [parse(Float64, row[2]) for row in rows],
            z = [parse(Float64, row[3]) for row in rows],
            sp = [String(strip(row[4], '"')) for row in rows])
    tree = String(strip(read(joinpath(_GLSP_G1_FIXTURE, "fixture-G1.nwk"), String)))
    return data, tree
end

@testset "GLSP REML correlation-edge diagnostics use the feasible tangent" begin
    dat, tree = _glsp_g1_data_for_edge_test()
    form = bf(@formula(y ~ x + phylo(1 | sp)),
              @formula(sigma ~ phylo(1 | sp)))
    fit = drm(form, Gaussian(); data = dat, tree, method = :REML,
              phylo_coupled = true, g_tol = 1e-8)

    cap = DRModels._GLSP_COR_CAP
    @test fit.scales[:lambda_cor][1] ≈ cap atol = 1e-9
    @test fit.nll isa DRModels._FitObjectiveGradient
    if fit.nll isa DRModels._FitObjectiveGradient
        theta = Vector{Float64}(fit.theta)
        p = length(theta) - 3
        # Coupled GLSP reports [logL11, logL22, L21], whereas the inner
        # optimizer uses [logL11, L21, logL22]. On the selected edge,
        # w = [logL11, log(sd_sigma)] and L21 = sign(rho)*cap*exp(w[2]),
        # logL22 = w[2] + 0.5log(1-cap^2).
        logL22, L21 = theta[p+2], theta[p+3]
        edge_sign = sign(L21)
        offset = 0.5 * log1p(-cap^2)
        w2 = logL22 - offset
        c = edge_sign * cap

        gradient = zeros(length(theta))
        fit.nll.gradient!(gradient, theta)
        tangent = zeros(length(theta))
        tangent[p+2] = 1.0
        tangent[p+3] = c * exp(w2)

        h = 1e-4
        theta_plus = copy(theta)
        theta_minus = copy(theta)
        theta_plus[p+2] = w2 + h + offset
        theta_minus[p+2] = w2 - h + offset
        theta_plus[p+3] = c * exp(w2 + h)
        theta_minus[p+3] = c * exp(w2 - h)
        finite_difference = (fit.nll.objective(theta_plus) -
                             fit.nll.objective(theta_minus)) / (2h)

        @test all(isfinite, gradient)
        @test dot(gradient, tangent) ≈ finite_difference atol = 1e-5 rtol = 1e-4
        @test DRModels.convergence_status(fit) === :boundary
    end
end
