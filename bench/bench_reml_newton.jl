# bench/bench_reml_newton.jl
# Wall-clock + step count of the fast observed-information Newton REML vs the FD-REML refit
# on the σ-phylo separate-block model (both axes), across p. Measures the internal speedup
# (the cross-engine ASReml/glmmTMB comparison is pending — σ-phylo location-scale has no
# native R baseline, and glmmTMB here is TMB-version-mismatched). Run:
#   julia --project=. bench/bench_reml_newton.jl
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

function bench_p(p)
    d = gen_sep(p)
    kind = Val(:gaussian_mean)
    Q, gidx, G = DRModels._locscale_phylo_setup(d.phy, d.species)
    Zη = DRModels._ls_canonical_Zeta(length(d.y)); Zψ = DRModels._ls_canonical_Zpsi(length(d.y))
    obj(θ)  = DRModels._glsp_sep_nll(kind, d.y, d.Xμ, d.Xψ, gidx, G, Q, θ, Zη, Zψ)
    grad(θ) = DRModels._glsp_sep_grad(kind, d.y, d.Xμ, d.Xψ, gidx, G, Q, θ, Zη, Zψ)
    θ0 = vcat(d.Xμ \ d.y, [0.0], log(0.3), log(0.3))
    θ̂_ml, mlc = DRModels._glsp_optimise(obj, (g, θ) -> (g .= grad(θ); g), θ0)
    vidx = [3, 4]
    # warm-up (exclude compilation from the timing)
    DRModels._glsp_reml_newton(obj, grad, θ̂_ml, 1, vidx; ml_converged = mlc)
    DRModels._glsp_reml_refit(obj, grad, θ̂_ml, 1; ml_converged = mlc)
    t_nw = @elapsed ((θn, cn, _, _, ns) = DRModels._glsp_reml_newton(obj, grad, θ̂_ml, 1, vidx; ml_converged = mlc))
    t_fd = @elapsed ((θf, cf, _, _)     = DRModels._glsp_reml_refit(obj, grad, θ̂_ml, 1; ml_converged = mlc))
    return (; p, t_nw, t_fd, ns, speedup = t_fd / t_nw, conv = cn,
            sd_match = abs(exp(θn[4]) - exp(θf[4])))
end

println("\n# Observed-info Newton REML vs FD-REML (σ-phylo separate block, m=4 reps/species)")
@printf("%-6s %-12s %-12s %-9s %-7s %-7s %-10s\n",
        "p", "Newton(s)", "FD-REML(s)", "speedup", "steps", "conv", "σSD-Δ")
for p in [24, 60, 120, 250]
    r = bench_p(p)
    @printf("%-6d %-12.3f %-12.3f %-9.2f %-7d %-7s %-10.2e\n",
            r.p, r.t_nw, r.t_fd, r.speedup, r.ns, string(r.conv), r.sd_match)
end
println()
