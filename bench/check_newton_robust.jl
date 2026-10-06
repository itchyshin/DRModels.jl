# bench/check_newton_robust.jl — Newton-only robustness probe. Before the safeguarded
# (backtracking) step, the raw observed-info Newton DIVERGED at p≥120 (σSD ≈ 7, conv=false).
# This confirms it now lands near the truth (μSD≈0.6, σSD≈0.5) and converges. No slow
# FD-REML reference here (that is the 14-min part of the full benchmark).
using DRModels, Random, LinearAlgebra, Printf

function gen_sep(p; m = 4, seed = 11)
    Random.seed!(seed)
    n = p * m
    phy = random_balanced_tree(p; branch_length = 0.3)
    C = sigma_phy_dense(phy; σ²_phy = 1.0); LC = cholesky(Symmetric(C)).L
    u_mu  = 0.6 .* (LC * randn(p))
    u_sig = 0.5 .* (LC * randn(p))
    species = repeat(1:p, inner = m)
    y = [1.0 + u_mu[species[i]] + exp(log(0.5) + u_sig[species[i]]) * randn() for i in 1:n]
    return (; y, species, phy, Xμ = ones(n, 1), Xψ = ones(n, 1))
end

println("\n# Newton-only robustness (raw Newton diverged at p≥120 → σSD≈7; truth μ≈0.6, σ≈0.5)")
@printf("%-6s %-7s %-8s %-10s %-10s\n", "p", "steps", "conv", "μSD", "σSD")
for p in [60, 120, 250]
    d = gen_sep(p)
    kind = Val(:gaussian_mean)
    Q, gidx, G = DRModels._locscale_phylo_setup(d.phy, d.species)
    Zη = DRModels._ls_canonical_Zeta(length(d.y)); Zψ = DRModels._ls_canonical_Zpsi(length(d.y))
    obj(θ)  = DRModels._glsp_sep_nll(kind, d.y, d.Xμ, d.Xψ, gidx, G, Q, θ, Zη, Zψ)
    grad(θ) = DRModels._glsp_sep_grad(kind, d.y, d.Xμ, d.Xψ, gidx, G, Q, θ, Zη, Zψ)
    θ0 = vcat(d.Xμ \ d.y, [0.0], log(0.3), log(0.3))
    θ̂_ml, mlc = DRModels._glsp_optimise(obj, (g, θ) -> (g .= grad(θ); g), θ0)
    θn, cn, _, _, ns = DRModels._glsp_reml_newton(obj, grad, θ̂_ml, 1, [3, 4]; ml_converged = mlc)
    @printf("%-6d %-7d %-8s %-10.4f %-10.4f\n", p, ns, string(cn), exp(θn[3]), exp(θn[4]))
end
println()
