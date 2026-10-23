# ML vs REML: σ-phylo SD recovery + boundary behaviour across signal levels.
using DRModels, Random, LinearAlgebra, Printf
println("\n# ML vs REML σ-phylo SD by true signal (mean over 3 seeds), p=20 species, m=4 reps")
@printf("%-22s %-12s %-12s %-10s %-16s\n", "true σ_phylo", "ML σ-SD", "REML σ-SD", "REML/ML", "REML profile CI")
for σtrue in [0.50, 0.15, 0.03]
    mls = Float64[]; rmls = Float64[]; los = Float64[]; his = Float64[]
    for seed in 1:3
        Random.seed!(seed); p=20; m=4; n=p*m
        phy = random_balanced_tree(p; branch_length=0.3)
        C = sigma_phy_dense(phy; σ²_phy=1.0); LC = cholesky(Symmetric(C)).L
        u = σtrue .* (LC*randn(p)); species = repeat(1:p, inner=m); x = randn(n)
        y = [1.0 + 0.3*x[i] + exp(log(0.5)+u[species[i]])*randn() for i in 1:n]
        data = (; y, x, species); form = bf(@formula(y~x), @formula(sigma~phylo(1|species)))
        fm = drm(form, Gaussian(); data=data, tree=phy, method=:ML)
        fr = drm(form, Gaussian(); data=data, tree=phy, method=:REML, profile_ci=true)
        push!(mls, exp(coef(fm,:resd_sigma)[1])); push!(rmls, exp(coef(fr,:resd_sigma)[1]))
        ci = get(fr.scales, :profile_ci_sd_sigma, [NaN,NaN]); push!(los, ci[1]); push!(his, ci[2])
    end
    mu(v)=sum(v)/length(v)
    @printf("%-22.2f %-12.4f %-12.4f %-10.3f [%.3f, %.3f]\n", σtrue, mu(mls), mu(rmls), mu(rmls)/mu(mls), mu(los), mu(his))
end
println()
