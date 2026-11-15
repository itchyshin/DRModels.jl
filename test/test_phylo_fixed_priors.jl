using Test, DRModels, LinearAlgebra, ForwardDiff, DelimitedFiles

@testset "normal priors with phylogenetic SD regression" begin
    phy = DRModels.augmented_phy("((a:0.5,b:0.5):0.5,(c:0.5,d:0.5):0.5);")
    dat = (y=[.2,.7,.35,1.1,.8,1.25,1.4,1.1,1.9,1.5,2.15,.5],
           x=[-1.2,-.8,-.7,-.2,.1,.15,.4,.6,.8,1.,1.2,-.3],
           sp=["a","a","b","b","b","c","c","d","d","d","d","a"],
           z=[-1.,-1.,-.35,-.35,-.35,.4,.4,1.,1.,1.,1.,-1.])
    f = bf(@formula(y ~ x + phylo(1 | sp)), @formula(sigma ~ x),
           @formula(sd(sp, phylogenetic) ~ z))
    p = drm_priors(drm_prior_normal(dpar="mu",coef="x",mean=.1,sd=.7),
                   drm_prior_normal(dpar="sigma",coef="x",mean=-.2,sd=.5))
    varied_z = merge(dat, (z=[-1.,-.8,dat.z[3:end]...],))
    @test_throws ArgumentError resolve_priors(f, Gaussian(); data=varied_z, priors=p)
    nan_z = merge(dat, (z=[NaN, dat.z[2:end]...],))
    @test_throws ArgumentError resolve_priors(f, Gaussian(); data=nan_z, priors=p)
    @test_throws ArgumentError drm(f, Gaussian(); data=nan_z, tree=phy, priors=p)
    preview = resolve_priors(f, Gaussian(); data=dat, priors=p)
    @test [(r.dpar,r.coef) for r in preview] == [(:mu,"x"),(:sigma,"x")]
    table = (prior=["normal(.1,.7)", "normal(-.2,.5)"],
             class=["fixef", "fixef"], coef=["x", "x"], dpar=["mu", "sigma"])
    @test [(r.dpar,r.coef) for r in resolve_priors(f, Gaussian(); data=dat, priors=table)] ==
          [(:mu,"x"),(:sigma,"x")]
    fit = drm(f, Gaussian(); data=dat, tree=phy, priors=p)
    double = merge(dat, (other=copy(dat.sp),))
    bad = bf(@formula(y ~ x + phylo(1 | sp) + phylo(1 | other)),
             @formula(sigma ~ x), @formula(sd(sp, phylogenetic) ~ z))
    @test_throws ArgumentError drm(bad, Gaussian(); data=double, tree=phy, priors=p)
    sigma_meta = bf(@formula(y ~ x + phylo(1 | sp)),
                    @formula(sigma ~ x + meta_V(v)), @formula(sd(sp, phylogenetic) ~ z))
    meta_data = merge(dat, (v=fill(.1,length(dat.y)),))
    @test_throws ArgumentError drm(sigma_meta, Gaussian(); data=meta_data, tree=phy, priors=p)
    @test_throws ArgumentError aic(fit)
    @test_throws ArgumentError confint(fit; method=:wald)
    refit = update(fit, f; data=dat, tree=phy)
    @test refit.estim_method == :MAP
    @test [(r.dpar,r.coef) for r in refit.penalty.rows] == [(:mu,"x"),(:sigma,"x")]
    @test fit.penalty.data_nll isa Function
    @test all(isfinite, ForwardDiff.gradient(fit.penalty.data_nll, [.8,.3,-.4,.15,-.3,.2]))
    @test length(fit.theta) == 6
    @test [(r.dpar,r.coef) for r in fit.penalty.rows] == [(:mu,"x"),(:sigma,"x")]
    K = [1. .5 0 0; .5 1 0 0; 0 0 1 .5; 0 0 .5 1]
    grp = [1,1,2,2,2,3,3,4,4,4,4,1]
    X = hcat(ones(length(dat.y)), dat.x); Z = hcat(ones(4), [-1.,-.35,.4,1.])
    function dense(q)
        a = exp.(Z*q[5:6]); sig = exp.(X*q[3:4])
        C = (a*a') .* K; V = C[grp,grp] + Diagonal(sig.^2)
        L = cholesky(Symmetric(V)); r = dat.y - X*q[1:2]
        .5*(length(r)*log(2π)+logdet(L)+dot(r,L\r)) +
            .5*((q[2]-.1)/.7)^2 + log(.7) + .5log(2π) +
            .5*((q[4]+.2)/.5)^2 + log(.5) + .5log(2π)
    end
    points = ([.8,.3,-.4,.15,-.3,.2],[.4,.7,-.8,-.2,-.5,.35],
              [1.,-.1,.1,.2,-.9,-.2],[.6,.6,-.1,-.3,.2,.1],
              [1.2,.4,-.6,.4,-.1,-.4])
    tablefit = drm(f, Gaussian(); data=dat, tree=phy, priors=table)
    plain_reml = drm(f, Gaussian(); data=dat, tree=phy, method=:REML)
    # Independently computed public-tree oracle; see development evidence.
    reference_obj = [11.3440779905637, 9.14530839622453, 15.6716932003145, 14.8468076770793, 11.215463299799]
    reference_grad = [-0.636593951095403 -1.034862477404889 7.415567591797730 1.926684071596260 2.574044520824935 -0.151920121082818;
        -4.134219706342567 1.862044078261477 4.700612008022679 -0.727287069369394 1.027779272889617 0.728921696344921;
        -0.166474366182474 -3.269045124778116 8.764914843669930 2.531456876508034 0.456779539602011 -0.147143780496628;
        -0.655913155588067 0.708955022388450 7.742923841196141 -0.376278876929146 2.919385631360480 0.169978774196428;
        0.264670646768146 -0.560190140108327 6.684085965717813 3.909170192878195 3.117209709057500 -0.252099105319253]
    for (i,q) in enumerate(points)
        @test isapprox(fit.nll(q), dense(q); rtol=1e-8)
        @test isapprox(fit.nll(q), reference_obj[i]; rtol=1e-8)
        @test maximum(abs.(ForwardDiff.gradient(fit.nll,q)-reference_grad[i,:])) < 1e-6
        @test isapprox(tablefit.nll(q), dense(q); rtol=1e-8)
        pvalue = .5*((q[2]-.1)/.7)^2 + log(.7) + .5log(2π) +
                 .5*((q[4]+.2)/.5)^2 + log(.5) + .5log(2π)
        @test isapprox(plain_reml.nll(q), dense(q)-pvalue; rtol=1e-8)
        @test maximum(abs.(ForwardDiff.gradient(fit.nll,q)-ForwardDiff.gradient(dense,q))) < 1e-6
        H1=ForwardDiff.hessian(fit.nll,q); H2=ForwardDiff.hessian(dense,q)
        @test maximum(abs.(H1-H2)./max.(1,abs.(H2))) < 1e-5
    end
end

@testset "phylogenetic prior fit matches independent interior optimum" begin
    raw = readdlm(joinpath(@__DIR__, "fixtures", "phylo_fixed_priors", "fit-data.csv"), ',', Float64; skipstart=1)
    labels = ["t" * lpad(string(i), 2, '0') for i in 1:16]
    quartets = ["(($(labels[k]):0.5,$(labels[k+1]):0.5):0.25,($(labels[k+2]):0.5,$(labels[k+3]):0.5):0.25):0.25" for k in 1:4:16]
    phy = DRModels.augmented_phy("(" * join(quartets, ",") * ");")
    dat = (y=raw[:,1], x=raw[:,2], sp=labels[Int.(raw[:,3])], z=raw[:,4])
    f = bf(@formula(y ~ x + phylo(1 | sp)), @formula(sigma ~ x),
           @formula(sd(sp, phylogenetic) ~ z))
    p = drm_priors(drm_prior_normal(dpar="mu",coef="x",mean=.1,sd=.7),
                   drm_prior_normal(dpar="sigma",coef="x",mean=-.2,sd=.5))
    fit = drm(f, Gaussian(); data=dat, tree=phy, priors=p)
    expected = [1.1304557328, .4343248334, -.3068621170,
                .2124665887, -.5879083171, 1.1086827482]
    @test maximum(abs.(fit.theta-expected)) < 1e-5
    @test isapprox(fit.nll(fit.theta), 104.69371002665; rtol=1e-8)
    @test maximum(abs.(ForwardDiff.gradient(fit.nll,fit.theta))) < 1e-5
    @test fit.converged
end
