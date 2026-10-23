# fit_em_natgrad.jl — UPGRADED EM-ML (Julia-1a) using this session's insights.
# The old warm EM (fit_ml_warm.jl) was 9.76s because its Λ-step used finite-diff
# (20 evals). Upgrades, all already verified:
#   (1) EXACT gradient (1 eval, marginal_and_exact_grad) — needs off-diagonal Λ
#       (lc3/lc7 singularity) which the EM stays off of.
#   (2) NATURAL-GRADIENT Λ-step: precondition the log-Cholesky step by the
#       observed information H_lc (demo_natgrad_p20: best-scaled, no overshoot).
#   (3) conditional-Newton β-step (mstep_beta) — the EM's distinctive block.
#   (4) relative-objective stopping (singular boundary).
# This is block-coordinate natural-gradient ascent — distinct from Julia-2's
# joint LBFGS. ML objective (β_μ are parameters, not integrated out).
#
# #13 DECISION GATE (2026-08-01): FAIL on q4_p100 — stalls at logLik ≈ −259.80
# vs sparse-TMB MLE −256.51 (same class as plain EM). Do NOT wire as
# `algorithm = :natgrad`. The Fisher metric `lc_metric` was extracted to
# `src/lc_metric.jl` (public infra). See
# `docs/dev-log/plans/2026-08-01-natgrad-decision-gate.md`.
# This file remains an unwired experimental prototype.
using LinearAlgebra, SparseArrays, ForwardDiff, Random, Statistics, Printf
# Standalone script path (broken relative include historically): prefer
# `using DRModels` + `DRModels.lc_metric` / `fit_q4_sparse_tmb` from the public module.
include(joinpath(@__DIR__, "..", "fit_q4_sparse_tmb.jl"))  # marginal_and_exact_grad, …
include(joinpath(@__DIR__, "..", "lc_metric.jl"))          # extracted Fisher metric (#13 S1b)

function fit_em_natgrad(prob, Q_cond, β0, Λ0; max_em=60, tol=1e-7, refresh=8, verbose=true)
    β = β0; Λ = Matrix(Λ0); θ = pack_theta(β, Λ)
    nll, g, u, _ = marginal_and_exact_grad(prob, Q_cond, θ; n_newton=60); ll = -nll; hist = [ll]
    Hlc = lc_metric(prob, Q_cond, θ, u)
    verbose && @info "init" loglik=round(ll; digits=4)
    it_done = 0
    for it in 1:max_em
        it_done = it; ll0 = ll
        # --- β block: conditional Newton (M-step), accept if marginal improves ---
        βn = mstep_beta(prob, u, β); θn = pack_theta(βn, Λ)
        nlln, _, un, _ = marginal_and_exact_grad(prob, Q_cond, θn; u0=u, n_newton=25)
        if -nlln >= ll; β = βn; ll = -nlln; u = un; θ = θn; end
        # --- Λ block: natural-gradient ascent on lc, line search on true marginal ---
        it % refresh == 0 && (Hlc = lc_metric(prob, Q_cond, θ, u))
        _, g, u, _ = marginal_and_exact_grad(prob, Q_cond, θ; u0=u, n_newton=25)
        dir = -(Hlc \ g[8:17])                      # Newton/natural direction (descend nll)
        lc0 = θ[8:17]; bestll = ll; bestlc = lc0; bestu = u
        for α in (1.0, 0.5, 0.25, 0.1, 0.03)
            lc = lc0 .+ α .* dir
            llt, ut, = marginal_nll(prob, Q_cond, vcat(θ[1:7], lc); u0=u, n_newton=25)
            if -llt > bestll; bestll = -llt; bestlc = lc; bestu = ut; end
        end
        θ = vcat(θ[1:7], bestlc); Λ = lc_to_Λ(bestlc); ll = bestll; u = bestu
        push!(hist, ll)
        verbose && (it <= 6 || it % 10 == 0) && @info "it $it" loglik=round(ll; digits=4) Δ=round(ll - ll0; digits=6)
        abs(ll - ll0) / (abs(ll) + 1e-8) < tol && it > 3 && (verbose && @info "converged" it; break)
    end
    @assert all(diff(hist) .>= -1e-4) "marginal decreased: $(round.(hist; digits=4))"
    βf, lcf = unpack_theta(prob, θ)
    return (β=βf, Λ=lc_to_Λ(lcf), loglik=ll, u=u, iters=it_done, hist=hist)
end

if abspath(PROGRAM_FILE) == @__FILE__
    using CSV, DataFrames
    FIX = normpath(joinpath(@__DIR__, "..", "..", "fixtures"))
    DRMTMB_LL = -256.52; DRMTMB_T = 2.48
    df = CSV.read(joinpath(FIX, "q4_p100.csv"), DataFrame); p = nrow(df); n = p
    phy = augmented_phy(read(joinpath(FIX, "q4_p100_tree.nwk"), String))
    name2row = Dict(String(s) => i for (i, s) in enumerate(df.species)); perm = [name2row[phy.leaf_names[k]] for k in 1:p]
    y1 = Vector{Float64}(df.y1)[perm]; y2 = Vector{Float64}(df.y2)[perm]; x1 = Vector{Float64}(df.x1)[perm]
    X1 = hcat(ones(n), x1); X2 = hcat(ones(n), x1); Xs1 = reshape(ones(n), n, 1); Xs2 = reshape(ones(n), n, 1); Xr = reshape(ones(n), n, 1)
    prob, Q_cond = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr)
    β0 = (mu1=X1\y1, mu2=X2\y2, s1=[log(std(y1 .- X1*(X1\y1)))], s2=[log(std(y2 .- X2*(X2\y2)))], rho=[0.0])
    Λ0 = Matrix(Symmetric(0.3*I(4) + 0.03*(ones(4,4)-I(4))))    # off the singularity
    println("=== upgraded EM-ML (natural-gradient), q4_p100 (p=$p) ===")
    fit_em_natgrad(prob, Q_cond, β0, Λ0; max_em=3, verbose=false)   # warmup
    t = @elapsed r = fit_em_natgrad(prob, Q_cond, β0, Λ0; max_em=200, verbose=false)
    @printf "\nlogLik=%.4f drmTMB=%.4f |Δ|=%.4f\n" r.loglik DRMTMB_LL abs(r.loglik - DRMTMB_LL)
    @printf "wall=%.3fs  iters=%d   (old finite-diff EM=9.76s; Julia-2 TMB-like=1.36s; drmTMB=2.48s)\n" t r.iters
    @printf "speedup vs old EM = %.1fx ; vs drmTMB = %.2fx %s\n" (9.76/t) (DRMTMB_T/t) (t < DRMTMB_T ? "FASTER than drmTMB" : "")
    println("sd_phy(=sqrt diag Λ)=", round.(sqrt.(diag(r.Λ)); digits=3), " (drmTMB [1.70,0.89,0.18,0.29])")
    println("=== done ===")
end
