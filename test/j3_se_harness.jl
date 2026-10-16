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
include(joinpath(@__DIR__, "j3_dense_reference.jl"))

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

const J3_LEDGER = NamedTuple[]

function j3_sha()
    try
        return readchomp(`git -C $(dirname(@__DIR__)) rev-parse HEAD`)
    catch
        return "unknown"
    end
end

# Row validity (#1073 Engine A). A NaN (or Inf) SE is dropped like a missing value on BOTH
# engines: the row is "not comparable" (reported, neither passed nor failed), whichever side
# the NaN came from. drmTMB side: `fit$opt$convergence != 0`, `sdr$pdHess` false, or a
# non-finite SE (ref.toml's `comparable`, and the SEs themselves, written as TOML `nan`).
# DRModels side: a Hessian that is not positive definite gives NaN SEs and a warning. Only
# the `ns` compared coordinates decide it (`n_shared`: the leading coordinates that share a
# parameterisation with drmTMB; the rest are an equivalent parameterisation of a covariance
# factor and are not compared). Returns (status, comparable).
function j3_status(ref, se_jl_all, ns)
    se_tmb = Float64.(ref["se"])
    ref_ok = get(ref, "comparable", true) && all(isfinite, se_tmb[1:ns])
    jl_ok = all(isfinite, se_jl_all[1:ns])
    status = !ref_ok ? "not comparable (drmTMB: convergence != 0, pdHess false or a NaN SE)" :
             !jl_ok ? "not comparable (DRModels: Hessian not positive definite, NaN SEs)" : "compared"
    return status, ref_ok && jl_ok
end

# `n_shared` (default: every coordinate), checked against the reference's own length.
function j3_ns(ref)
    ns = get(ref, "n_shared", length(ref["theta"]))
    1 <= ns <= min(length(ref["theta"]), length(ref["se"])) ||
        error("ref.toml: n_shared = $ns outside 1:$(min(length(ref["theta"]), length(ref["se"])))")
    return ns
end

# `fit` is the DRModels fit; `rtol_se` is the SE tolerance against sdreport. Returns `ns`, the
# number of leading coordinates compared, so a caller can pin it (see the REML F4 row).
function j3_check(route::AbstractString, fit; rtol_se = 0.01, opt_tol = 2e-4,
                  gradient::Bool = true, known_noisy::Bool = false, dense_se = nothing)
    ref = j3_ref(route)
    θ_tmb = Float64.(ref["theta"]); se_tmb = Float64.(ref["se"])
    ns = j3_ns(ref)
    reml = get(ref, "reml", false)
    se_jl_all = sqrt.(diag(vcov(fit)))
    status, comparable = j3_status(ref, se_jl_all, ns)
    push!(J3_LEDGER, (route = String(route), drmTMB_commit = get(ref, "drmTMB_commit", "unrecorded"),
                      opt_convergence = get(ref, "opt_convergence", missing),
                      pd_hess = get(ref, "pd_hess", missing), drm_sha = j3_sha(),
                      converged = is_converged(fit), status = status,
                      n_compared = comparable ? "$ns/$(length(θ_tmb))" : "0/$(length(θ_tmb))",
                      max_rel_se = comparable ? maximum(abs.(se_jl_all[1:ns] ./ se_tmb[1:ns] .- 1)) : NaN))
    @testset "$route" begin
        if !comparable
            @test_skip false                        # not comparable: neither pass nor fail
        else
            @test ref["max_grad"] < 1e-6                              # 1. reference usable
            @test is_converged(fit)
            @test length(fit.theta) >= ns
            @test maximum(abs.(fit.theta[1:ns] .- θ_tmb[1:ns])) < opt_tol   # 2. fitted optimum
            # REML fits report the restricted log-likelihood
            @test (reml ? fit.reml_loglik : loglik(fit)) ≈ ref["loglik"] atol = 1e-5
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
            if dense_se !== nothing
                # 5. an independent reference (j3_dense_reference.jl: dense marginal, Hessian by
                #    ForwardDiff): it must agree with the reported SEs at the Julia optimum, and
                #    with drmTMB's sdreport at drmTMB's optimum.
                @test all(abs.(dense_se(fit.theta) ./ se_jl_all .- 1) .< 1e-4)
                reml || @test all(abs.(dense_se(θ_tmb) ./ se_tmb .- 1) .< 1e-3)
            end
            se = se_jl_all[1:ns]                                      # 4. reported SEs
            se_t = se_tmb[1:ns]
            @test all(isfinite, se)
            if known_noisy
                # value-closure route: no exact gradient, so the SE is a noisy second
                # difference. Only a loose sanity band is asserted; NOT counted as native parity.
                @test all(abs.(se ./ se_t .- 1) .< 0.10)
            else
                @test all(abs.(se ./ se_t .- 1) .< rtol_se)
            end
        end
    end
    return ns
end

# Markdown ledger of every row (written when DRM_J3_LEDGER=<path> is set).
function j3_write_ledger(path)
    open(path, "w") do io
        println(io, "| route | drmTMB commit | opt\$convergence | sdr\$pdHess | DRModels SHA | converged | status | coords compared | max rel. SE diff |")
        println(io, "|---|---|---|---|---|---|---|---|---|")
        for r in J3_LEDGER
            println(io, "| ", r.route, " | `", first(r.drmTMB_commit, 7), "` | ", r.opt_convergence, " | ",
                    r.pd_hess, " | `", first(r.drm_sha, 7), "` | ", r.converged, " | ", r.status, " | ",
                    r.n_compared, " | ",
                    isnan(r.max_rel_se) ? "n/a" : string(round(r.max_rel_se; sigdigits = 2)), " |")
        end
    end
end

# unique string levels -> integer index (sorted), and the cov.csv tip covariance in tip order
function j3_levels(v)
    u = sort(unique(v)); m = Dict(x => i for (i, x) in enumerate(u))
    return [m[x] for x in v], length(u)
end
function j3_cov(route)
    lines = readlines(joinpath(J3_DIR, route, "cov.csv"))
    tips = [strip(c, '"') for c in split(lines[1], ',')][2:end]
    C = [parse(Float64, c) for l in lines[2:end] for c in split(l, ',')[2:end]]
    return reshape(C, length(tips), length(tips)) |> permutedims, tips
end
