# test_bridge_objective_at.jl — pin the SUPPORTED bridge entry point
# `DRModels.drm_bridge_objective_at`: the same (formula, family, data, tree,
# options) payload `drm_bridge` takes, plus an outer point (beta, Lambda,
# rho12), routed to DRModels.jl's own `reml_objective_at` (#575) — so the
# drmTMB R shim (`drm_julia_reml_objective_at()`, R/julia-bridge.R) stops
# depending on five private DRModels.jl names (`_bridge_data`, `_bridge_formula`,
# `_bivariate_q4_marker`, `_design`, `_phylo_species_index`) reached by
# qualified name.
#
# Numbers pinned to docs/dev-log/evidence/julia-r-parity/ayumi-target/
# 2026-09-02-a5-cross-engine-receipt.md (DRModels.jl @ dc3ce1908369e4734e92c37
# 220dad951647b4844, the commit this branch stacks on):
#   DRModels.jl objective at TMB's fitted point   = -219.620688
#   DRModels.jl objective at Julia's own point    = -219.630326 (pre-#579; -219.614005 post-#579)
#
#   julia --project=. -e 'using DRModels, Test; include("test/test_bridge_objective_at.jl")'

module TestBridgeObjectiveAt

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
const EXPECTED = TOML.parsefile(joinpath(FIXTURE, "expected.toml"))

# The exact payload shape drmTMB's `drm_julia_bridge_payload()` builds for this
# fixture (`R/julia-bridge.R`) — the SAME keyed formula spelling `drm_bridge`'s
# own tests use for this fixture (test/test_bridge_bivariate_inference.jl).
const BRIDGE_FORMULA = Dict(
    :mu1 => "y1 ~ x + phylo(1 | species)",
    :mu2 => "y2 ~ x + phylo(1 | species)",
    :sigma1 => "sigma1 ~ 1 + phylo(1 | species)",
    :sigma2 => "sigma2 ~ 1 + phylo(1 | species)",
    :rho12 => "rho12 ~ 1",
)
const BRIDGE_FAMILY = "biv_gaussian"
const BRIDGE_OPTIONS = Dict("method" => "REML")

# TMB's fitted point (expected.toml coefs + #575's frozen phylo covariance,
# same numbers test_reml_objective_at.jl pins for the private path).
const RHO12_TMB = Float64(EXPECTED["coef"]["rho12_(Intercept)"])
const LAMBDA_TMB = [
    0.5288095   0.25509007 -0.1228962  -0.15554224
    0.25509007  0.28551843 -0.1794685  -0.02445802
   -0.1228962  -0.1794685   0.4857264  -0.10269499
   -0.15554224 -0.02445802 -0.10269499  0.16678147
]
const BETA_TMB = (
    mu1 = [Float64(EXPECTED["coef"]["mu1_(Intercept)"]), Float64(EXPECTED["coef"]["mu1_x"])],
    mu2 = [Float64(EXPECTED["coef"]["mu2_(Intercept)"]), Float64(EXPECTED["coef"]["mu2_x"])],
    sigma1 = [Float64(EXPECTED["coef"]["sigma1_(Intercept)"])],
    sigma2 = [Float64(EXPECTED["coef"]["sigma2_(Intercept)"])],
)

# Same FORM as test_reml_objective_at.jl — the `@formula` spelling DRModels.jl's
# own bridge-formula parser produces from BRIDGE_FORMULA's string values.
const FORM = bf(mu1    = @formula(y1 ~ x + phylo(1 | species)),
                 mu2    = @formula(y2 ~ x + phylo(1 | species)),
                 sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
                 sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
                 rho12  = @formula(rho12 ~ 1))

# Rebuild the SAME q4 phylo REML problem the bridge builds internally, so test
# (b) below can compare the bridge's answer to the private `reml_objective_at`
# path at an IDENTICAL point.
function _direct_problem()
    rhs = Dict(FORM.forms)
    fixed, marker = DRModels._bivariate_q4_marker(rhs)
    grp = marker[2]
    phy = DRModels._as_augmented_phy(TREE)
    y1, X1, _ = DRModels._design(FORM.response1, fixed[:mu1], DAT)
    y2, X2, _ = DRModels._design(FORM.response2, fixed[:mu2], DAT)
    _, Xs1, _ = DRModels._design(FORM.response1, fixed[:sigma1], DAT)
    _, Xs2, _ = DRModels._design(FORM.response1, fixed[:sigma2], DAT)
    _, Xr, _  = DRModels._design(FORM.response1, fixed[:rho12], DAT)
    species = DRModels._phylo_species_index(phy, getproperty(DAT, grp))
    return DRModels.make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr; species = species)
end

@testset "drm_bridge_objective_at — bivariate q4 phylo REML" begin

    @testset "(a) reproduces #575's cross-engine receipt at TMB's point" begin
        result = DRModels.drm_bridge_objective_at(BRIDGE_FORMULA, BRIDGE_FAMILY, DAT, TREE,
                                              BRIDGE_OPTIONS;
                                              beta = BETA_TMB, Lambda = LAMBDA_TMB,
                                              rho12 = RHO12_TMB)
        @test isapprox(result["reml_loglik"], -219.620688; atol = 2e-4)
        @test result["contract"] == "bridge_objective_at_v1"
    end

    @testset "(a) reproduces #575's cross-engine receipt at Julia's own point" begin
        prob, Q_cond = _direct_problem()
        obs1, obs2 = prob.obs1, prob.obs2
        β1 = prob.X1[obs1, :] \ prob.y1[obs1]
        β2 = prob.X2[obs2, :] \ prob.y2[obs2]
        res1 = prob.y1[obs1] .- prob.X1[obs1, :] * β1
        res2 = prob.y2[obs2] .- prob.X2[obs2, :] * β2
        β0 = (mu1 = β1, mu2 = β2,
              s1 = DRModels._initial_scale_beta(prob.Xs1, res1),
              s2 = DRModels._initial_scale_beta(prob.Xs2, res2),
              rho = zeros(size(prob.Xr, 2)))
        Λ0 = Matrix(Symmetric([
            0.30 0.02 0.01 0.010
            0.02 0.30 0.01 0.010
            0.01 0.01 0.08 0.005
            0.01 0.01 0.005 0.080
        ]))
        rr = DRModels.fit_q4_reml(prob, Q_cond; beta0 = β0, Lambda0 = Λ0,
                              g_tol = 1e-3, iterations = 300, n_newton = 40)
        @test rr.converged == true

        beta_julia = (mu1 = rr.beta.mu1, mu2 = rr.beta.mu2,
                      sigma1 = rr.beta.s1, sigma2 = rr.beta.s2)
        result = DRModels.drm_bridge_objective_at(BRIDGE_FORMULA, BRIDGE_FAMILY, DAT, TREE,
                                              BRIDGE_OPTIONS;
                                              beta = beta_julia, Lambda = rr.Lambda,
                                              rho12 = rr.beta.rho[1])
        # HISTORY: the A5 receipt (DRModels.jl @ dc3ce190, finite-difference mode-finder)
        # recorded -219.630326 at Julia's own point. After #579 (exact REML
        # gradient, merged f930e8bf) fit_q4_reml's optimum on this fixture is
        # -219.614005, so the point moved with it; the TMB-point value above is
        # unchanged. The entry must agree with the fit's own report (next test).
        @test isapprox(result["reml_loglik"], -219.614005; atol = 2e-4)
    end

    @testset "(b) equals the private reml_objective_at path at an identical point" begin
        prob, Q_cond = _direct_problem()
        phi_tmb = DRModels.pack_phi(prob, [RHO12_TMB], LAMBDA_TMB)
        beta0_tmb = (mu1 = BETA_TMB.mu1, mu2 = BETA_TMB.mu2,
                     s1 = BETA_TMB.sigma1, s2 = BETA_TMB.sigma2, rho = [RHO12_TMB])
        direct = DRModels.reml_objective_at(prob, Q_cond, phi_tmb; beta0 = beta0_tmb)

        bridged = DRModels.drm_bridge_objective_at(BRIDGE_FORMULA, BRIDGE_FAMILY, DAT, TREE,
                                               BRIDGE_OPTIONS;
                                               beta = BETA_TMB, Lambda = LAMBDA_TMB,
                                               rho12 = RHO12_TMB)
        @test isapprox(bridged["reml_loglik"], direct.reml_loglik; atol = 1e-8)
    end

    @testset "(c) rejects a non-q4 payload" begin
        uni_formula = "y ~ x"
        uni_data = (; y = DAT.y1, x = DAT.x)
        @test_throws ArgumentError DRModels.drm_bridge_objective_at(uni_formula, "gaussian",
                                                                 uni_data, TREE, Dict{String,Any}();
                                                                 beta = BETA_TMB, Lambda = LAMBDA_TMB,
                                                                 rho12 = RHO12_TMB)
        err = try
            DRModels.drm_bridge_objective_at(uni_formula, "gaussian", uni_data, TREE, Dict{String,Any}();
                                         beta = BETA_TMB, Lambda = LAMBDA_TMB, rho12 = RHO12_TMB)
            nothing
        catch e
            e
        end
        @test occursin("bivariate q=4", sprint(showerror, err))
    end

    @testset "(c) rejects a wrong-length beta" begin
        bad_beta = merge(BETA_TMB, (mu1 = [1.0],))
        err = try
            DRModels.drm_bridge_objective_at(BRIDGE_FORMULA, BRIDGE_FAMILY, DAT, TREE, BRIDGE_OPTIONS;
                                         beta = bad_beta, Lambda = LAMBDA_TMB, rho12 = RHO12_TMB)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("mu1", sprint(showerror, err))
    end

    @testset "(c) rejects a non-4x4 Lambda" begin
        err = try
            DRModels.drm_bridge_objective_at(BRIDGE_FORMULA, BRIDGE_FAMILY, DAT, TREE, BRIDGE_OPTIONS;
                                         beta = BETA_TMB, Lambda = LAMBDA_TMB[1:3, 1:3], rho12 = RHO12_TMB)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("4", sprint(showerror, err)) && occursin("Lambda", sprint(showerror, err))
    end

    @testset "(d) return shape" begin
        result = DRModels.drm_bridge_objective_at(BRIDGE_FORMULA, BRIDGE_FAMILY, DAT, TREE,
                                              BRIDGE_OPTIONS;
                                              beta = BETA_TMB, Lambda = LAMBDA_TMB,
                                              rho12 = RHO12_TMB)
        @test result isa Dict{String,Any}
        for k in ("objective", "reml_loglik", "converged_inner", "contract")
            @test haskey(result, k)
        end
        @test result["contract"] == "bridge_objective_at_v1"
    end

end

end # module TestBridgeObjectiveAt
