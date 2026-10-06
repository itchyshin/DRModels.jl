using DRModels, Random, LinearAlgebra, Statistics
# UNIT-HEIGHT tree, so the simulated sd and the estimated sd are on the SAME scale.
function unit_tree(G)
    p0 = random_balanced_tree(G; branch_length=1.0)
    h  = maximum(diag(sigma_phy_dense(p0; σ²_phy=1.0)))
    random_balanced_tree(G; branch_length=1.0/h)
end
function study(G, m; nrep=30, β0=0.3, β1=0.8, sdt=0.7)
    b0=Float64[]; b1=Float64[]; sd=Float64[]; conv=Ref(0)
    for r in 1:nrep
        rng = MersenneTwister(9000+r)
        phy = unit_tree(G)
        C = sigma_phy_dense(phy; σ²_phy=1.0); d=sqrt.(diag(C)); K=C./(d*d')
        L = cholesky(Symmetric(K)).L
        species = repeat(1:G, inner=m); n=G*m; x=randn(rng,n)
        u = sdt .* (L*randn(rng,G))
        y = Float64.(rand(rng,n) .< 1 ./(1 .+exp.(-(β0 .+ β1 .*x .+ u[species]))))
        try
            f = drm(bf(@formula(y ~ x + phylo(1 | species))), Binomial();
                    data=(; y, x, species), tree=phy)
            c = coef(f,:mu); s = re_sd(f)[:species]
            if f.converged && all(isfinite,c) && isfinite(s)
                push!(b0,c[1]); push!(b1,c[2]); push!(sd,s); conv[] += 1
            end
        catch; end
    end
    (; G, conv=conv[], nrep, b0, b1, sd)
end
println("UNIT-HEIGHT tree (h=1) -- simulated and estimated sd on the same scale")
for G in (40, 80, 160)
    r = study(G,12)
    println("G=$(r.G) n=$(r.G*12):  converged $(r.conv)/$(r.nrep)")
    for (v,t,nm) in ((r.b0,0.3,"beta0"),(r.b1,0.8,"beta1"),(r.sd,0.7,"sd_phylo"))
        println("   ", rpad(nm,9), " mean=", rpad(round(mean(v);digits=4),8),
                " bias=", rpad(round(mean(v)-t;digits=4),9),
                " rel=", rpad(round(100*(mean(v)-t)/t;digits=1),6), "%",
                "  mcse=", round(std(v)/sqrt(length(v));digits=4))
    end
end
