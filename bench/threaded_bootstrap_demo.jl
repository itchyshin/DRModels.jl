# threaded_bootstrap_demo.jl — the structural DRModels.jl win: a parametric
# bootstrap of the q=4 PLSM run SERIAL vs THREADED. drmTMB's bootstrap is
# serial R; DRModels.jl threads the refits across cores. At per-fit parity the
# threading multiplier is the pipeline-level speedup over drmTMB.
#
# Run (threads matter):
#   cd .../drm_q4
#   /Users/z3437171/.juliaup/bin/julia --project=.. --threads=16 threaded_bootstrap_demo.jl

using LinearAlgebra, SparseArrays, Random, Statistics, Printf, CSV, DataFrames
BLAS.set_num_threads(1)                       # avoid oversubscription under threads
include(joinpath(@__DIR__, "sparse_em_fit.jl"))

const FIX = normpath(joinpath(@__DIR__, "..", "..", "fixtures"))
const DRMTMB_T = 2.585                         # drmTMB single-fit q4_p100 (s)

@info "threads available" n=Threads.nthreads()

# --- load real q4_p100 + fit once for the bootstrap truth -------------------
df = CSV.read(joinpath(FIX, "q4_p100.csv"), DataFrame); p = nrow(df); n = p
newick = read(joinpath(FIX, "q4_p100_tree.nwk"), String); phy = augmented_phy(newick)
name2row = Dict(String(s) => i for (i,s) in enumerate(df.species))
perm = [name2row[phy.leaf_names[k]] for k in 1:p]
x1 = Vector{Float64}(df.x1)[perm]
X1 = hcat(ones(n), x1); X2 = hcat(ones(n), x1)
Xs1 = reshape(ones(n),n,1); Xs2 = reshape(ones(n),n,1); Xr = reshape(ones(n),n,1)
Σ_phy = sigma_phy_dense(phy; σ²_phy=1.0)        # leaf covariance (for simulation)
Lσ = cholesky(Symmetric(Σ_phy)).U

y1 = Vector{Float64}(df.y1)[perm]; y2 = Vector{Float64}(df.y2)[perm]
prob0, Q_cond = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr)
β̂0 = (mu1=X1\y1, mu2=X2\y2, s1=[log(std(y1.-X1*(X1\y1)))], s2=[log(std(y2.-X2*(X2\y2)))], rho=[0.0])
@info "fitting point estimate..."
fit0 = fit_em_aug(prob0, Q_cond, β̂0, Matrix(0.3*I(4)); max_em=80, verbose=false)
β̂ = fit0.β; Λ̂ = fit0.Λ
@printf "point fit logLik=%.3f iters=%d\n" fit0.loglik fit0.iters

# --- one parametric-bootstrap refit (simulate from (β̂,Λ̂), refit) ----------
function boot_refit(b::Int)
    rng = MersenneTwister(4000 + b)
    U = cholesky(Λ̂).L * randn(rng, 4, p) * Lσ          # leaf REs ~ MN(0,Λ̂,Σ_phy)
    yb1 = zeros(p); yb2 = zeros(p)
    for i in 1:p
        m1=(X1[i,:]'β̂.mu1)+U[1,i]; m2=(X2[i,:]'β̂.mu2)+U[2,i]
        s1=exp((Xs1[i,:]'β̂.s1)+U[3,i]); s2=exp((Xs2[i,:]'β̂.s2)+U[4,i]); ρ=RHO_GUARD*tanh(Xr[i,:]'β̂.rho)
        e=cholesky([s1^2 ρ*s1*s2; ρ*s1*s2 s2^2]).L*randn(rng,2); yb1[i]=m1+e[1]; yb2[i]=m2+e[2]
    end
    prob = make_problem(phy, yb1, yb2, X1, X2, Xs1, Xs2, Xr)[1]
    r = fit_em_aug(prob, Q_cond, β̂, Matrix(Λ̂); max_em=40, verbose=false)  # warm from point fit
    return r.β.rho[1]                                  # a bootstrap statistic (ρ12 intercept)
end

B = 48
boot_refit(1)                                          # warm-up (compile)
GC.gc()                                                # clean slate before timing

# serial
ser = zeros(B); t_ser = @elapsed for b in 1:B; ser[b] = boot_refit(b); end
# threaded
thr = zeros(B); t_thr = @elapsed Threads.@threads for b in 1:B; thr[b] = boot_refit(b); end

@printf "\n=== threaded parametric bootstrap (B=%d, q4_p100) ===\n" B
@printf "serial   : %.2f s  (%.3f s/refit)\n" t_ser (t_ser/B)
@printf "threaded : %.2f s  (%.3f s/refit)  on %d threads\n" t_thr (t_thr/B) Threads.nthreads()
@printf "threading speedup: %.2fx\n" (t_ser/t_thr)
@printf "\n--- extrapolate to standard B=199 ---\n"
per = t_ser/B
@printf "drmTMB (serial R, ~%.2fs/refit): ~%.0f s\n" DRMTMB_T (199*DRMTMB_T)
@printf "DRModels.jl serial:   ~%.0f s\n" (199*per)
@printf "DRModels.jl threaded: ~%.0f s  ->  vs drmTMB = %.1fx FASTER\n" (199*t_thr/B) (199*DRMTMB_T/(199*t_thr/B))
println("=== done ===")
