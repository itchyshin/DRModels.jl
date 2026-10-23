# run_sparse_tmb_nd.jl — same as run_sparse_tmb_q4p100 but starts Λ0 OFF the
# diagonal singularity (lc3=C31, lc7=C42 gradient is a removable singularity at
# the ρ=0 + diagonal-Λ init). Tests whether the TMB-like exact-gradient LBFGS now
# converges and beats drmTMB (LL -256.52, 2.48 s).
import Pkg
Pkg.activate(dirname(@__DIR__))
using DRModels
using LinearAlgebra, Statistics, Printf, DelimitedFiles

const FIX = joinpath(@__DIR__, "fixtures")
const DRMTMB_LL=-256.52; const DRMTMB_T=2.48
raw = readdlm(joinpath(FIX, "q4_p100.csv"), ',', String; header = true)[1]
p = size(raw, 1); n = p
phy=augmented_phy(read(joinpath(FIX,"q4_p100_tree.nwk"),String))
species = raw[:, 4]
name2row=Dict(String(s)=>i for (i,s) in enumerate(species)); perm=[name2row[phy.leaf_names[k]] for k in 1:p]
y1=parse.(Float64, raw[:, 1])[perm];y2=parse.(Float64, raw[:, 2])[perm];x1=parse.(Float64, raw[:, 3])[perm]
X1=hcat(ones(n),x1);X2=hcat(ones(n),x1);Xs1=reshape(ones(n),n,1);Xs2=reshape(ones(n),n,1);Xr=reshape(ones(n),n,1)
prob,Q_cond=make_problem(phy,y1,y2,X1,X2,Xs1,Xs2,Xr)
β0=(mu1=X1\y1,mu2=X2\y2,s1=[log(std(y1.-X1*(X1\y1)))],s2=[log(std(y2.-X2*(X2\y2)))],rho=[0.0])
# Λ0 OFF the singularity: 0.3 diag + small off-diagonals (SPD).
Λ0=[0.30 0.05 0.03 0.03; 0.05 0.30 0.03 0.03; 0.03 0.03 0.30 0.03; 0.03 0.03 0.03 0.30]
Λ0=Matrix(Symmetric(Λ0)); @assert isposdef(Λ0)
println("=== sparse TMB-like LBFGS, q4_p100, Λ0 off-diagonal (p=$p) ===")
fit_q4_sparse_tmb(prob,Q_cond;β0=β0,Λ0=Λ0,g_tol=1e-3,iterations=300,n_newton=40)  # warmup
t1=@elapsed r=fit_q4_sparse_tmb(prob,Q_cond;β0=β0,Λ0=Λ0,g_tol=1e-3,iterations=300,n_newton=40)
t2=@elapsed fit_q4_sparse_tmb(prob,Q_cond;β0=β0,Λ0=Λ0,g_tol=1e-3,iterations=300,n_newton=40)
tmed=min(t1,t2)
println("\n--- result ---")
@printf "logLik Julia=%.4f  drmTMB=%.4f  |Δ|=%.4f\n" r.loglik DRMTMB_LL abs(r.loglik-DRMTMB_LL)
@printf "converged=%s iters=%d g_resid=%.2e f_calls=%d g_calls=%d\n" r.converged r.iterations r.g_residual r.f_calls r.g_calls
@printf "wall Julia=%.3fs (best of 2: %.3f, %.3f)  drmTMB=%.2fs  ratio=%.2fx %s\n" tmed t1 t2 DRMTMB_T (DRMTMB_T/tmed) (tmed<DRMTMB_T ? "JULIA FASTER" : "")
@printf "β_mu1=%s β_mu2=%s β_s1=%s β_s2=%s β_rho=%s\n" round.(r.β.mu1;digits=3) round.(r.β.mu2;digits=3) round.(r.β.s1;digits=3) round.(r.β.s2;digits=3) round.(r.β.rho;digits=3)
println("sd_phy(=sqrt diag Λ)=",round.(sqrt.(diag(r.Λ));digits=3)," (drmTMB [1.70,0.89,0.18,0.29])")
println("=== done ===")
