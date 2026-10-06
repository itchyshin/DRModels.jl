# bench/check_p120_match.jl — definitive large-p correctness check: does the safeguarded
# observed-info Newton land on the SAME REML optimum as the (slow) FD-REML refit at p=120?
using DRModels, Random, LinearAlgebra, Printf

Random.seed!(11)
p = 120; m = 4; n = p * m
phy = random_balanced_tree(p; branch_length = 0.3)
C = sigma_phy_dense(phy; σ²_phy = 1.0); LC = cholesky(Symmetric(C)).L
u_mu  = 0.6 .* (LC * randn(p)); u_sig = 0.5 .* (LC * randn(p))
species = repeat(1:p, inner = m)
y = [1.0 + u_mu[species[i]] + exp(log(0.5) + u_sig[species[i]]) * randn() for i in 1:n]
Xμ = ones(n, 1); Xψ = ones(n, 1)

kind = Val(:gaussian_mean)
Q, gidx, G = DRModels._locscale_phylo_setup(phy, species)
Zη = DRModels._ls_canonical_Zeta(n); Zψ = DRModels._ls_canonical_Zpsi(n)
obj(θ)  = DRModels._glsp_sep_nll(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)
grad(θ) = DRModels._glsp_sep_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)
θ0 = vcat(Xμ \ y, [0.0], log(0.3), log(0.3))
θ̂_ml, _ = DRModels._glsp_optimise(obj, (g, θ) -> (g .= grad(θ); g), θ0)

t_fd = @elapsed ((θf, cf, _, _)     = DRModels._glsp_reml_refit(obj, grad, θ̂_ml, 1; ml_converged = true))
t_nw = @elapsed ((θn, cn, _, _, ns) = DRModels._glsp_reml_newton(obj, grad, θ̂_ml, 1, [3, 4]; ml_converged = true))

@printf("\n# p=120 large-p correctness: safeguarded Newton vs FD-REML\n")
@printf("FD-REML : μSD=%.5f σSD=%.5f conv=%s  (%.1fs)\n", exp(θf[3]), exp(θf[4]), string(cf), t_fd)
@printf("Newton  : μSD=%.5f σSD=%.5f conv=%s steps=%d  (%.1fs)\n", exp(θn[3]), exp(θn[4]), string(cn), ns, t_nw)
@printf("Δ(μSD)=%.2e  Δ(σSD)=%.2e  speedup=%.1f×\n\n",
        abs(exp(θn[3]) - exp(θf[3])), abs(exp(θn[4]) - exp(θf[4])), t_fd / t_nw)
