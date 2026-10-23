# sparse_em_fit.jl — Laplace-EM on the validated sparse augmented foundation
# (sparse_aug_plsm.jl, Checkpoint 3 passed: marginal == dense oracle to 0.0).
#
# M-step: (a) closed-form 4×4 Λ via the matrix-normal MLE + Takahashi trace
# correction; (b) conditional Newton on the 7 fixed effects β at fixed û.
# Monotonicity guard: recompute the true Laplace marginal, backtrack if it
# dropped. Root-conditioned (N = n_keep = 2p-2).
#
# Run a p=20 recovery test:
#   cd .../drm_q4 && julia --project=.. sparse_em_fit.jl

using LinearAlgebra, SparseArrays, ForwardDiff, Random, Statistics, Printf
include(joinpath(@__DIR__, "sparse_aug_plsm.jl"))

# Build a root-conditioned problem + sparse prior Q from a phylogeny and data.
# Returns (prob, Q_cond). leaf_node maps data row k -> position in the
# root-removed node set.
function make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr; species = 1:phy.n_leaves)
    n_total = phy.n_total
    keep = setdiff(1:n_total, [phy.root_index])
    Q_cond = phy.Q_topology[keep, keep]
    pos = Dict(node => i for (i, node) in enumerate(keep))
    # one entry per DATA ROW: species[i] is the leaf (1:n_leaves) of row i. Default
    # = one row per leaf (backward-compatible). >1 row/leaf ⇒ replicate observations,
    # which IDENTIFY the per-leaf scale random effects (fixes the p≥500 degeneracy).
    leaf_node = [pos[phy.leaf_indices[species[i]]] for i in eachindex(species)]
    # #19: per-row observed-response masks (NaN ⇒ missing). Zero the missing y so the
    # value never leaks into a warm-start path; the masked leaf likelihood ignores it.
    y1f = Vector{Float64}(y1); y2f = Vector{Float64}(y2)
    obs1 = .!isnan.(y1f); obs2 = .!isnan.(y2f)
    y1c = ifelse.(obs1, y1f, 0.0); y2c = ifelse.(obs2, y2f, 0.0)
    prob = AugProblem(phy, length(keep), phy.n_leaves, leaf_node,
                      y1c, y2c, X1, X2, Xs1, Xs2, Xr, obs1, obs2)
    return prob, Q_cond
end

# Placeholder AugmentedPhy so AugProblem's typed `phy` field is filled for
# level-indexed (relmat / animal / spatial) q=4 fits. `fit_q4_sparse_tmb` never
# reads `prob.phy` — only `n_total`, `leaf_node`, and the design/response fields.
function _placeholder_aug_phy(G::Int)
    G >= 1 || error("structured Q_cond must have at least one level")
    Q = sparse(1.0 * I, G, G)
    names = ["L$i" for i in 1:G]
    return AugmentedPhy{Float64}(G, G, Q, collect(1:G), names, Float64[], 1)
end

"""
    make_problem_from_Q(Q_cond, y1, y2, X1, X2, Xs1, Xs2, Xr; group) -> (prob, Q)

Build an [`AugProblem`](@ref) from a known G×G group-level precision (relmat /
animal / fixed-range spatial) rather than from an augmented phylogeny. `group[i]`
maps data row `i` to a 1-based level in `1:G`. Returns `(prob, sparse(Q_cond))`
for [`fit_q4_sparse_tmb`](@ref).

This is the #189 level-indexed entry point; the phylo path keeps [`make_problem`](@ref).
"""
function make_problem_from_Q(Q_cond::AbstractMatrix, y1, y2, X1, X2, Xs1, Xs2, Xr;
                             group)
    G = LinearAlgebra.checksquare(Q_cond)
    n = length(y1)
    length(y2) == n || error("y1/y2 length mismatch in make_problem_from_Q")
    length(group) == n || error("group has $(length(group)) entries but y1 has $n rows")
    size(X1, 1) == n && size(X2, 1) == n && size(Xs1, 1) == n &&
        size(Xs2, 1) == n && size(Xr, 1) == n ||
        error("design matrices must have $n rows")
    leaf_node = Int.(collect(group))
    all(1 .<= leaf_node .<= G) ||
        error("group indices must be 1-based integers in 1:$G")
    Qdense = Matrix{Float64}(Q_cond)
    isposdef(Symmetric(Qdense)) ||
        error("known structured precision Q_cond must be positive definite")
    Q = sparse(Symmetric(Qdense))
    y1f = Vector{Float64}(y1); y2f = Vector{Float64}(y2)
    obs1 = .!isnan.(y1f); obs2 = .!isnan.(y2f)
    y1c = ifelse.(obs1, y1f, 0.0); y2c = ifelse.(obs2, y2f, 0.0)
    phy = _placeholder_aug_phy(G)
    # n_total = G (levels); p = G (placeholder leaf count). Engine loops use leaf_node.
    prob = AugProblem(phy, G, G, leaf_node,
                      y1c, y2c,
                      Matrix{Float64}(X1), Matrix{Float64}(X2),
                      Matrix{Float64}(Xs1), Matrix{Float64}(Xs2), Matrix{Float64}(Xr),
                      obs1, obs2)
    return prob, Q
end

# --- M-step (a): closed-form Λ (4×4) ----------------------------------------
# Λ_new = (1/N)( Û Q_cond Û'  +  Σ_{(s,t)} Q_cond[s,t]·Cov(u_s,u_t|y) ),
# N = n_keep. Posterior covariance blocks from Takahashi selected inverse.
#
# NOT reachable from the public `drm()` API (#472). Production (`fit_ml_q4.jl`)
# does not call this: it replaces the Λ step with `lambda_ml_step`, a
# line-searched finite-difference ascent on the true marginal.
#
# The p=100 DESCENT that motivated abandoning this step was an ARTEFACT (#577,
# fixed 2026-09-02). The posterior covariance blocks above come from
# `takahashi_selinv`, which can only return entries the CHOLMOD pattern carries;
# `prior_precision` used to drop Λ⁻¹'s exact zeros, so at a diagonal Λ the
# cross-axis entries were structurally ABSENT at every non-leaf node and this
# function read them as zero covariances. On the q4_p100 fixture that produced
# Λ_em off-diagonals of 1.21 (repaired: 0.0078) and a 55.8-nat DESCENT; with the
# pattern repaired the same step ASCENDS by 0.70 nats, at every step size
# (`test/test_lambda_p100.jl`, which now asserts the ascent). The p=8 sibling
# `test/test_lambda_direction.jl` always asserted ascent and always passed — the
# artefact scales with the number of NON-LEAF nodes.
#
# NOT a licence to re-adopt the closed-form step. On a SYNTHETIC p=100 tree with
# pure-noise responses it still descends (~52 nats) after the repair, and that is
# unexplained. #472 stays OPEN, re-scoped: measure before re-wiring anything.
function mstep_Lambda(prob::AugProblem, Q_cond::SparseMatrixCSC, u::Vector{Float64}, ch_H)
    N = prob.n_total                       # n_keep
    Û = reshape(u, 4, N)                    # 4 × n_keep
    base = Û * Q_cond * Û'                  # 4×4
    V = takahashi_selinv(ch_H)              # selected inverse at H's pattern
    corr = zeros(4, 4)
    rows = rowvals(Q_cond); vals = nonzeros(Q_cond)
    @inbounds for t in 1:N
        for idx in nzrange(Q_cond, t)
            s = rows[idx]; q = vals[idx]
            bs = 4(s-1); bt = 4(t-1)
            for a in 1:4, b in 1:4
                corr[a, b] += q * V[bs+a, bt+b]
            end
        end
    end
    Λ = (base + corr) ./ N
    # PD-floor: keep Λ positive-definite so inv(Λ) in the prior never fails on
    # ill-conditioned real data (the closed-form update can drift non-PD).
    S = Symmetric((Λ + Λ') / 2)
    E = eigen(S)
    λf = max.(E.values, 1e-6)
    return Matrix(Symmetric(E.vectors * Diagonal(λf) * E.vectors'))
end

# --- M-step (b): conditional Newton on β (7 params) at fixed û leaves --------
function mstep_beta(prob::AugProblem, u::Vector{Float64}, β; n_newton=25, tol=1e-10)
    p = prob.p
    uL = [(u[4(prob.leaf_node[i]-1)+1], u[4(prob.leaf_node[i]-1)+2],
           u[4(prob.leaf_node[i]-1)+3], u[4(prob.leaf_node[i]-1)+4]) for i in 1:p]
    k1=size(prob.X1,2); k2=size(prob.X2,2); ks1=size(prob.Xs1,2); ks2=size(prob.Xs2,2); kr=size(prob.Xr,2)
    o1=0;o2=k1;o3=o2+k2;o4=o3+ks1;o5=o4+ks2
    function f(θ)
        bm1=θ[o1+1:o1+k1]; bm2=θ[o2+1:o2+k2]; bs1=θ[o3+1:o3+ks1]; bs2=θ[o4+1:o4+ks2]; br=θ[o5+1:o5+kr]
        tot=zero(eltype(θ))
        @inbounds for i in 1:p
            tot += leaf_nll(uL[i], prob.y1[i], prob.y2[i],
                            prob.X1[i,:]'bm1, prob.X2[i,:]'bm2,
                            prob.Xs1[i,:]'bs1, prob.Xs2[i,:]'bs2, prob.Xr[i,:]'br,
                            prob.obs1[i], prob.obs2[i])
        end
        return tot
    end
    θ = vcat(β.mu1, β.mu2, β.s1, β.s2, β.rho)
    for _ in 1:n_newton
        g = ForwardDiff.gradient(f, θ); H = ForwardDiff.hessian(f, θ)
        # robust solve: escalate a ridge until Cholesky succeeds (the 7×7 β
        # Hessian can go indefinite/singular far from the conditional optimum)
        local step = g
        for λ in (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4)
            ch = cholesky(Symmetric(Matrix(H)) + λ*I; check=false)
            if issuccess(ch); step = ch \ g; break; end
        end
        step = cap_newton_step(step)   # trust radius BEFORE backtracking (same guard as cond_newton_beta)
        f0=f(θ); α=1.0; θn=θ.-α.*step
        for _bt in 1:25
            (f(θn)<=f0 || α<1e-8) && break
            α*=0.5; θn=θ.-α.*step
        end
        θ=θn; norm(α.*step)<tol && break
    end
    return (mu1=θ[o1+1:o1+k1], mu2=θ[o2+1:o2+k2], s1=θ[o3+1:o3+ks1], s2=θ[o4+1:o4+ks2], rho=θ[o5+1:o5+kr])
end

# --- EM driver with monotonicity guard --------------------------------------
# NOT reachable from the public `drm()` API (#472) — this is the closed-form-Λ
# EM route, driven only by this file's own `abspath(PROGRAM_FILE)` demo and by
# `bench/` scripts. Production fits go through `fit_ml_q4.jl` instead, which
# line-searches the Λ step.
#
# Its Λ block (`mstep_Lambda`) DID fail to ascend the true marginal at p=100;
# that failure was the #577 sparsity-pattern artefact and is fixed (see that
# function's comment for the before/after numbers). The general case is NOT
# settled — a synthetic pure-noise p=100 tree still descends — so this route
# stays fenced and #472 stays open. The monotonicity `@assert` below guards
# THIS driver's own bookkeeping (it only accepts a step if `marg` reports an
# improvement), so it cannot mask a failure to ascend the underlying objective;
# that property is unchanged and is why the fence is safe to leave as-is.
function fit_em_aug(prob::AugProblem, Q_cond::SparseMatrixCSC, β0, Λ0;
                    max_em=200, tol=1e-6, verbose=true)
    β = β0; Λ = Matrix(Λ0)
    function marg(βx, Λx, u0)
        P = prior_precision(Q_cond, inv(Λx))
        u, ch, _ = estep_mode(prob, P, βx; u0=u0)
        return laplace_ll(prob, P, βx, u, ch), u, ch
    end
    ll, u, ch = marg(β, Λ, nothing)
    verbose && @info "EM init" loglik=round(ll;digits=4)
    ll_hist=[ll]; conv=false; it_done=0
    for it in 1:max_em
        it_done=it; ll0=ll
        # β block (guarded) — warm-start the inner Newton from the current mode
        βp = mstep_beta(prob, u, β)
        llp,up,chp = marg(βp, Λ, u)
        if llp >= ll; β,ll,u,ch = βp,llp,up,chp; end
        # Λ block (guarded with backtrack toward current)
        Λp = mstep_Lambda(prob, Q_cond, u, ch)
        llq,uq,chq = marg(β, Λp, u)
        if llq < ll
            α=1.0
            while llq < ll && α>1e-3
                α*=0.5; Λt=Matrix(Symmetric(Λ .+ α.*(Λp.-Λ)))
                llt,ut,cht = marg(β, Λt, u)
                if llt>llq; Λp,llq,uq,chq=Λt,llt,ut,cht; end
            end
        end
        if llq >= ll; Λ,ll,u,ch = Λp,llq,uq,chq; end
        push!(ll_hist, ll)
        verbose && (it<=5||it%10==0) && @info "EM iter $it" loglik=round(ll;digits=4) Δ=round(ll-ll0;digits=6)
        if ll-ll0 < tol && it>3; conv=true; verbose && @info "converged" iter=it loglik=round(ll;digits=4); break; end
    end
    @assert all(diff(ll_hist) .>= -1e-7) "marginal decreased"
    return (β=β, Λ=Λ, loglik=ll, u=u, iters=it_done, converged=conv, ll_hist=ll_hist)
end

# ===================== p=20 recovery test ===================================
if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(7); p=20; n=p
    phy = random_balanced_tree(p; branch_length=0.2)
    Σ_phy = sigma_phy_dense(phy; σ²_phy=1.0)
    βt=(mu1=[1.0,0.5],mu2=[-0.3,0.4],s1=[-0.4],s2=[-0.5],rho=[0.3])
    Λt=[0.25 0.10 0.05 0.0;0.10 0.25 0.0 0.04;0.05 0.0 0.09 0.02;0.0 0.04 0.02 0.09]; Λt=(Λt+Λt')/2
    x1=randn(n); X1=hcat(ones(n),x1);X2=hcat(ones(n),x1)
    Xs1=reshape(ones(n),n,1);Xs2=reshape(ones(n),n,1);Xr=reshape(ones(n),n,1)
    U=cholesky(Λt).L*randn(4,p)*cholesky(Symmetric(Σ_phy)).U
    y1=zeros(n);y2=zeros(n)
    for i in 1:n
        m1=(X1[i,:]'βt.mu1)+U[1,i];m2=(X2[i,:]'βt.mu2)+U[2,i]
        s1=exp((Xs1[i,:]'βt.s1)+U[3,i]);s2=exp((Xs2[i,:]'βt.s2)+U[4,i]);ρ=RHO_GUARD*tanh(Xr[i,:]'βt.rho)
        e=cholesky([s1^2 ρ*s1*s2;ρ*s1*s2 s2^2]).L*randn(2);y1[i]=m1+e[1];y2[i]=m2+e[2]
    end
    prob,Q_cond = make_problem(phy,y1,y2,X1,X2,Xs1,Xs2,Xr)
    β0=(mu1=X1\y1,mu2=X2\y2,s1=[log(std(y1.-X1*(X1\y1)))],s2=[log(std(y2.-X2*(X2\y2)))],rho=[0.0])
    t=@elapsed res=fit_em_aug(prob,Q_cond,β0,Matrix(0.3*I(4));max_em=200,tol=1e-6)
    println("\n=== sparse EM p=20 ===")
    @printf "wall=%.3fs iters=%d conv=%s logLik=%.4f\n" t res.iters res.converged res.loglik
    @printf "β_mu1 %s->%s  β_mu2 %s->%s\n" βt.mu1 round.(res.β.mu1;digits=3) βt.mu2 round.(res.β.mu2;digits=3)
    @printf "β_s1 %s->%s  β_s2 %s->%s  β_rho %s->%s\n" βt.s1 round.(res.β.s1;digits=3) βt.s2 round.(res.β.s2;digits=3) βt.rho round.(res.β.rho;digits=3)
    println("Λ diag truth ", round.(diag(Λt);digits=3), " -> ", round.(diag(res.Λ);digits=3))
    println("=== done ===")
end
