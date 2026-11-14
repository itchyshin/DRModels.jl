using Test, DRModels

@testset "prior table preview" begin
    d = (y=[.2,.8,.5,1.4,1.,1.6,2.,1.3,2.2,1.8,2.4,2.1],
         x=repeat([-1.,0.,1.],4), site=repeat(["a","b","c","d"],inner=3),
         observer=repeat(["u","v","w"],4))
    f = bf(@formula(y ~ x + (1 | site) + (1 | observer)))
    tab = (prior=["normal(0,2)","gamma(0.6666666666666666,2)"],
           class=["fixef","ranef_sd"], coef=["x",""])
    expected = drm_priors(drm_prior_normal(dpar="mu",coef="x",sd=2),
                          drm_prior_sd_gamma(dpar="mu",group="site",shape=2,rate=3),
                          drm_prior_sd_gamma(dpar="mu",group="observer",shape=2,rate=3))
    @test resolve_priors(f,Gaussian();data=d,priors=tab) == resolve_priors(f,Gaussian();data=d,priors=expected)
    @test_throws ArgumentError resolve_priors(f,Gaussian();data=d,priors=(prior=["gamma(1,2)","gamma(1,2)"],class=["ranef_sd","ranef_sd"],coef=["","site"]))
    @test_throws ArgumentError resolve_priors(f,Gaussian();data=d,priors=(prior=["normal(0,2)"],class=["fixef"],coef=["missing"]))
end

@testset "prior table input metadata" begin
    d=(y=[.2,.8,.5,1.4,1.,1.6],x=[-1.,0.,1.,-1.,0.,1.])
    f=bf(@formula(y~x))
    table=(prior=["normal(0,2)"],class=["fixef"],coef=["x"])
    fit=drm(f,Gaussian();data=d,priors=table)
    @test haskey(fit.penalty,:input_table)
    @test fit.penalty.input_table == table
    table.prior[1] = "ridge(0)"
    @test fit.penalty.input_table.prior[1] == "normal(0,2)"
    refit = update(fit, f; data=d)
    @test haskey(refit.penalty,:input_table)
end

@testset "update keeps the resolved target" begin
    d=(y=[.2,.8,.5,1.4,1.,1.6],x=[-1.,0.,1.,-1.,0.,1.],z=[0.,0.,1.,1.,0.,1.])
    f=bf(@formula(y~x))
    broad=(prior=["normal(0,2)"],class=["fixef"],coef=[""])
    fit=drm(f,Gaussian();data=d,priors=broad)
    refit=update(fit,bf(@formula(y~x+z));data=d)
    @test [r.coef for r in refit.penalty.rows] == ["x"]
    @test refit.penalty.input_table == broad
end

@testset "reserved literal all cannot be selected by a broad rule" begin
    d=(y=[.2,.8,.5,1.4,1.,1.6],all=[-1.,0.,1.,-1.,0.,1.])
    f=bf(@formula(y~all))
    broad=(prior=["normal(0,2)"],class=["fixef"],coef=[""])
    @test_throws ArgumentError resolve_priors(f,Gaussian();data=d,priors=broad)
end
