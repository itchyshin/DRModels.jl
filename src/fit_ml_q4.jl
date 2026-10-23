# fit_ml_q4.jl — ML fit of the q=4 PLSM by ascending the TRUE ML Laplace
# marginal (laplace_ll = drmTMB's objective). β by conditional Newton; Λ by a
# LINE-SEARCHED gradient ascent on the marginal over log-Cholesky params (fixes
# the overshooting closed-form). Monotone by construction. method=:ML default;
# :REML adds the fixed-effects-information correction (TODO flag).

using LinearAlgebra, SparseArrays, ForwardDiff, Random, Statistics, Printf
include(joinpath(@__DIR__, "sparse_em_fit.jl"))

# log-Cholesky 4×4 <-> 10-vector
function lc_to_Λ(v)
    C=zeros(eltype(v),4,4);k=0
    for j in 1:4,i in j:4; k+=1; C[i,j]= i==j ? exp(v[k]) : v[k]; end
    return C*C'
end
function Λ_to_lc(Λ)
    C=cholesky(Symmetric(Λ)).L;v=Float64[]
    for j in 1:4,i in j:4; push!(v, i==j ? log(C[i,j]) : C[i,j]); end; return v
end

# Marginal as a function of Λ (β fixed), fully-converged E-step.
# Whitened prior (#857 site q4): Λ⁻¹ is never formed; the latent is v = (I⊗L⁻¹)u.
function L_given_Λ(prob,Q_cond,β,Λ; nit=60)
    W=whitened_prior(Q_cond,cholesky(Symmetric(Λ)).L); v,ch,_=estep_mode(prob,W,β;n_newton=nit)
    return laplace_ll(prob,W,β,v,ch)
end

# Λ block: line-searched gradient ascent on the ML marginal (log-chol params).
function lambda_ml_step(prob,Q_cond,β,Λ; h=1e-5)
    v=Λ_to_lc(Λ); L0=L_given_Λ(prob,Q_cond,β,lc_to_Λ(v))
    g=similar(v)
    for k in eachindex(v)
        vp=copy(v);vp[k]+=h; vm=copy(v);vm[k]-=h
        g[k]=(L_given_Λ(prob,Q_cond,β,lc_to_Λ(vp)) - L_given_Λ(prob,Q_cond,β,lc_to_Λ(vm)))/(2h)
    end
    gn=norm(g); gn<1e-8 && return Λ, L0
    bestΛ=Λ; bestL=L0
    for s in (1.0,0.3,0.1,0.03,0.01,0.003,0.001) ./ gn   # normalized step (no overshoot)
        Ls=L_given_Λ(prob,Q_cond,β,lc_to_Λ(v .+ s.*g))
        if Ls>bestL; bestL=Ls; bestΛ=lc_to_Λ(v .+ s.*g); end
    end
    return bestΛ, bestL
end

function fit_ml(prob,Q_cond,β0,Λ0; max_em=100, tol=1e-5, n_lam=3, verbose=true)
    β=β0; Λ=Matrix(Λ0)
    ll=L_given_Λ(prob,Q_cond,β,Λ); hist=[ll]
    verbose && @info "ML init" loglik=round(ll;digits=4)
    it_done=0
    for it in 1:max_em
        it_done=it; ll0=ll
        # β block: conditional Newton, accept if marginal improves
        W=whitened_prior(Q_cond,cholesky(Symmetric(Λ)).L); v,ch,_=estep_mode(prob,W,β;n_newton=60)
        βn=mstep_beta(prob,whitened_to_u(W,v),β)
        lln=L_given_Λ(prob,Q_cond,βn,Λ)
        if lln>=ll; β,ll=βn,lln; end
        # Λ block: a few line-searched ascent steps on the ML marginal
        for _ in 1:n_lam
            Λn,llΛ = lambda_ml_step(prob,Q_cond,β,Λ)
            if llΛ>ll+1e-10; Λ,ll=Λn,llΛ; else break; end
        end
        push!(hist,ll)
        verbose && (it<=8||it%10==0) && @info "ML it $it" loglik=round(ll;digits=4) Δ=round(ll-ll0;digits=6)
        ll-ll0<tol && it>3 && (verbose && @info "converged" it loglik=round(ll;digits=4); break)
    end
    @assert all(diff(hist).>=-1e-6) "marginal decreased"
    return (β=β,Λ=Λ,loglik=ll,iters=it_done,hist=hist)
end

if abspath(PROGRAM_FILE)==@__FILE__
    Random.seed!(7); p=20; n=p
    phy=random_balanced_tree(p;branch_length=0.2); Σ_phy=sigma_phy_dense(phy;σ²_phy=1.0)
    βt=(mu1=[1.0,0.5],mu2=[-0.3,0.4],s1=[-0.4],s2=[-0.5],rho=[0.3])
    Λt=[0.25 0.10 0.05 0.0;0.10 0.25 0.0 0.04;0.05 0.0 0.09 0.02;0.0 0.04 0.02 0.09];Λt=(Λt+Λt')/2
    x1=randn(n);X1=hcat(ones(n),x1);X2=hcat(ones(n),x1);Xs1=reshape(ones(n),n,1);Xs2=reshape(ones(n),n,1);Xr=reshape(ones(n),n,1)
    U=cholesky(Λt).L*randn(4,p)*cholesky(Symmetric(Σ_phy)).U
    y1=zeros(n);y2=zeros(n)
    for i in 1:n
        m1=(X1[i,:]'βt.mu1)+U[1,i];m2=(X2[i,:]'βt.mu2)+U[2,i]
        s1=exp((Xs1[i,:]'βt.s1)+U[3,i]);s2=exp((Xs2[i,:]'βt.s2)+U[4,i]);ρ=RHO_GUARD*tanh(Xr[i,:]'βt.rho)
        e=cholesky([s1^2 ρ*s1*s2;ρ*s1*s2 s2^2]).L*randn(2);y1[i]=m1+e[1];y2[i]=m2+e[2]
    end
    prob,Q_cond=make_problem(phy,y1,y2,X1,X2,Xs1,Xs2,Xr)
    β0=(mu1=X1\y1,mu2=X2\y2,s1=[log(std(y1.-X1*(X1\y1)))],s2=[log(std(y2.-X2*(X2\y2)))],rho=[0.0])
    t=@elapsed r=fit_ml(prob,Q_cond,β0,Matrix(0.3*I(4)))
    println("\n=== ML fit p=20 ===")
    @printf "wall=%.2fs iters=%d logLik=%.4f\n" t r.iters r.loglik
    @printf "β_mu1 %s->%s  β_s1 %s->%s  β_rho %s->%s\n" βt.mu1 round.(r.β.mu1;digits=2) βt.s1 round.(r.β.s1;digits=2) βt.rho round.(r.β.rho;digits=2)
    println("Λ diag truth ",round.(diag(Λt);digits=3)," -> ",round.(diag(r.Λ);digits=3), "  (Λ MOVED from 0.3I?)")
    println("=== done ===")
end
