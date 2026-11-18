using Test, DRModels, DelimitedFiles

@testset "phylogenetic priors with current-main fitting contracts" begin
    raw = readdlm(joinpath(@__DIR__, "fixtures", "phylo_fixed_priors", "fit-data.csv"), ',', Float64; skipstart=1)
    labels = ["t" * lpad(string(i), 2, '0') for i in 1:16]
    quartets = ["(($(labels[k]):0.5,$(labels[k+1]):0.5):0.25,($(labels[k+2]):0.5,$(labels[k+3]):0.5):0.25):0.25" for k in 1:4:16]
    phy = DRModels.augmented_phy("(" * join(quartets, ",") * ");")
    d = (y=raw[:,1], x=raw[:,2], sp=labels[Int.(raw[:,3])], z=raw[:,4])
    f = bf(@formula(y ~ x + phylo(1 | sp)), @formula(sigma ~ x),
           @formula(sd(sp, phylogenetic) ~ z))
    pri = (prior=["normal(.1,.7)","normal(-.2,.5)"],
           class=["fixef","fixef"],coef=["x","x"],dpar=["mu","sigma"])
    fit = drm(f, Gaussian(); data=d, tree=phy, priors=pri)
    index = indexin(d.sp, String.(phy.leaf_names))
    @test fitted(fit) ≈ fit.means[:mu] .+ ranef(fit)[:sp][index]
    @test residuals(fit) ≈ d.y .- fitted(fit)
    @test_throws ArgumentError drm(f, Gaussian(); data=d, tree=phy, priors=pri,
                                   weights=fill(2.,length(d.y)))
    @test_throws ArgumentError drm(f, Gaussian(); data=d, tree=phy, priors=pri,
                                   weights=ones(length(d.y)))
    withoff = merge(d, (o=zeros(length(d.y)),))
    fo = bf(@formula(y ~ x + offset(o) + phylo(1 | sp)), @formula(sigma ~ x),
            @formula(sd(sp, phylogenetic) ~ z))
    @test_throws ArgumentError drm(fo, Gaussian(); data=withoff, tree=phy, priors=pri)
    refit = update(fit, f; data=d, tree=phy)
    @test refit.estim_method == :MAP
    @test [(r.dpar,r.coef) for r in refit.penalty.rows] == [(:mu,"x"),(:sigma,"x")]
    @test refit.nll(refit.theta) ≈ fit.nll(fit.theta) atol=1e-7
end
