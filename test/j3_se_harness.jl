# j3_se_harness.jl -- shared helpers for the #1073 (lane J3) SE-vs-drmTMB tests.
#
# Per route: test/fixtures/j3_se/<route>/{data.csv, ref.toml[, tree.newick]}, produced by
# test/parity/gen_j3_se_reference.R from native drmTMB (numbers and simulated data only).
# ref.toml carries drmTMB's opt$par and sdreport SEs already ordered like DRModels' theta,
# plus TMB's max |gradient| at opt$par.
#
# The checks (each its own @test, so a failure says which one):
#   1. reference usable   -- TMB's max |gradient| at opt$par is < 1e-6
#   2. fitted optimum     -- DRModels' own fit lands on TMB's opt$par (separate from SEs)
#   3. SE at TMB's point  -- the Hessian is built at TMB's opt$par (not at DRModels' optimum),
#                            from the route's exact gradient, and its SEs match sdreport ~1%
#   4. reported SEs       -- stderror(fit), what a user sees, matches sdreport ~1%
# Check 3 is skipped for a route with no exact gradient (value-closure routes); those are
# tested for finiteness only and are labelled known-noisy.
using DRModels, Test, TOML, LinearAlgebra

const J3_DIR = joinpath(@__DIR__, "fixtures", "j3_se")

function j3_read_csv(route)
    lines = readlines(joinpath(J3_DIR, route, "data.csv"))
    hdr = [strip(c, '"') for c in split(lines[1], ',')]
    cols = Dict{Symbol,Vector{String}}()
    for (j, h) in enumerate(hdr)
        cols[Symbol(h)] = String[strip(split(l, ',')[j], '"') for l in lines[2:end]]
    end
    return cols
end

j3_float(c) = [parse(Float64, v) for v in c]
j3_ref(route) = TOML.parsefile(joinpath(J3_DIR, route, "ref.toml"))
j3_tree(route) = DRModels._bridge_tree(read(joinpath(J3_DIR, route, "tree.newick"), String))

# `fit` is the DRModels fit; `rtol_se` is the SE tolerance against sdreport.
function j3_check(route::AbstractString, fit; rtol_se = 0.01, opt_tol = 2e-4,
                  gradient::Bool = true, known_noisy::Bool = false)
    ref = j3_ref(route)
    θ_tmb = Float64.(ref["theta"]); se_tmb = Float64.(ref["se"])
    # `n_shared`: leading coordinates that share a parameterisation with drmTMB (the rest
    # are a different but equivalent parameterisation of a covariance factor).
    ns = get(ref, "n_shared", length(θ_tmb))
    @testset "$route" begin
        @test ref["max_grad"] < 1e-6                              # 1. reference usable
        @test is_converged(fit)
        @test length(fit.theta) >= ns
        @test maximum(abs.(fit.theta[1:ns] .- θ_tmb[1:ns])) < opt_tol   # 2. fitted optimum
        @test loglik(fit) ≈ ref["loglik"] atol = 1e-5
        if gradient && !known_noisy
            # 3. Hessian at TMB's opt$par from the exact gradient
            @test fit.nllgrad !== nothing
            H = DRModels._finite_hessian_from_gradient(fit.nllgrad, θ_tmb; context = route)
            @test all(isfinite, H)
            se_h = sqrt.(diag(inv(Symmetric(H))))
            @test all(abs.(se_h ./ se_tmb .- 1) .< rtol_se)
            # A Hessian that is not positive definite must give NaN and a warning, never
            # a made-up SE. -gradient has Hessian -H, which is negative definite here.
            Hneg = @test_logs (:warn, r"not positive definite") DRModels._finite_hessian_from_gradient(
                (G, θ) -> (fit.nllgrad(G, θ); G .*= -1; G), θ_tmb; context = route)
            @test all(isnan, Hneg)
        end
        se = sqrt.(diag(vcov(fit)))[1:ns]                          # 4. reported SEs
        se_tmb = se_tmb[1:ns]
        @test all(isfinite, se)
        if known_noisy
            # value-closure route: no exact gradient, so the SE is a noisy second
            # difference. Only a loose sanity band is asserted; NOT counted as native parity.
            @test all(abs.(se ./ se_tmb .- 1) .< 0.10)
        else
            @test all(abs.(se ./ se_tmb .- 1) .< rtol_se)
        end
    end
end
