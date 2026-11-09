using Test, DRModels

@testset "prior bridge version dispatch fails closed" begin
    @test isdefined(DRModels, :_bridge_prior_contract)
    if isdefined(DRModels, :_bridge_prior_contract)
        dispatch = DRModels._bridge_prior_contract
        @test dispatch(Dict{Symbol,Any}()) === :none
        @test dispatch(Dict{Symbol,Any}(:priors => [], :prior_expected => [])) === :v1
        good = Dict{Symbol,Any}(:prior_contract => "parameter_priors_v2", :priors_v2 => [1], :prior_expected_v2 => [1])
        @test dispatch(good) === :v2
        @test_throws ArgumentError dispatch(merge(good, Dict(:priors => [])))
        @test_throws ArgumentError dispatch(Dict{Symbol,Any}(:prior_contract => "parameter_priors_v2"))
        @test_throws ArgumentError dispatch(merge(good, Dict(:prior_contract => "unknown")))
        @test_throws ArgumentError dispatch(merge(good, Dict(:priors_v2 => [])))
        @test_throws ArgumentError dispatch(merge(good, Dict(:prior_expected_v2 => nothing)))
    end
end

@testset "v2 preparation resolves SD identity before fitting" begin
    @test isdefined(DRModels, :_bridge_prepare_priors_v2)
    if isdefined(DRModels, :_bridge_prepare_priors_v2)
        data = (y=[1.,2.,3.,4.,5.,6.], x=[-1.,0.,1.,-1.,0.,1.], site=["a","a","b","b","c","c"])
        form, dat, labels = DRModels._bridge_formula(Dict("mu"=>"y ~ x + (1 | site)", "sigma"=>"sigma ~ 1"), "gaussian", DRModels._bridge_data(data); labels=true)
        spec = Dict("kind"=>"ranef_sd_gamma", "dpar"=>"mu", "group"=>"site", "coef"=>"(Intercept)", "block_kind"=>"iid_random_intercept", "shape"=>2., "rate"=>3., "measure"=>"natural_sd")
        expected = Dict("spec_index"=>1, "kind"=>"ranef_sd_gamma", "class"=>"ranef_sd", "dpar"=>"mu", "group"=>"site", "coef"=>"(Intercept)", "block_kind"=>"iid_random_intercept", "shape"=>2., "rate"=>3., "measure"=>"natural_sd")
        prepare = DRModels._bridge_prepare_priors_v2
        priors = prepare(form, DRModels.Gaussian(), dat, labels, [spec], [expected])
        @test length(priors) == 1
        bad = copy(expected); bad["rate"] = 4.
        @test_throws ArgumentError prepare(form, DRModels.Gaussian(), dat, labels, [spec], [bad])
        bad = copy(expected); bad["group"] = "wrong"
        @test_throws ArgumentError prepare(form, DRModels.Gaussian(), dat, labels, [spec], [bad])
        badspec = copy(spec); badspec["shape"] = true
        badrecord = copy(expected); badrecord["shape"] = true
        @test_throws ArgumentError prepare(form, DRModels.Gaussian(), dat, labels, [badspec], [badrecord])
        badspec = copy(spec); badspec["extra"] = 1
        @test_throws ArgumentError prepare(form, DRModels.Gaussian(), dat, labels, [badspec], [expected])
        duplicated = Dict{Any,Any}(pairs(spec)); duplicated[:group] = "site"
        @test_throws ArgumentError prepare(form, DRModels.Gaussian(), dat, labels, [duplicated], [expected])
        badspec = copy(spec); badspec["rate"] = 1.0
        badrecord = copy(expected); badrecord["rate"] = true
        @test_throws ArgumentError prepare(form, DRModels.Gaussian(), dat, labels, [badspec], [badrecord])
    end
end

@testset "v2 preparation preserves mixed user order" begin
    data = (y=[1.,2.,3.,4.,5.,6.], x=[-1.,0.,1.,-1.,0.,1.], site=["a","a","b","b","c","c"])
    form, dat, labels = DRModels._bridge_formula(Dict("mu"=>"y ~ x + (1 | site)", "sigma"=>"sigma ~ 1"), "gaussian", DRModels._bridge_data(data); labels=true)
    sd = Dict("kind"=>"ranef_sd_gamma", "dpar"=>"mu", "group"=>"site", "coef"=>"(Intercept)", "block_kind"=>"iid_random_intercept", "shape"=>2., "rate"=>3., "measure"=>"natural_sd")
    ridge = Dict("kind"=>"fixed_ridge", "dpar"=>"mu", "coef"=>"all", "intercept"=>"include", "lambda"=>0.5)
    rows = [merge(sd, Dict("spec_index"=>1, "class"=>"ranef_sd")),
            Dict("spec_index"=>2, "kind"=>"fixed_ridge", "class"=>"fixed", "dpar"=>"mu", "coef"=>"(Intercept)", "lambda"=>0.5),
            Dict("spec_index"=>2, "kind"=>"fixed_ridge", "class"=>"fixed", "dpar"=>"mu", "coef"=>"x", "lambda"=>0.5)]
    priors = DRModels._bridge_prepare_priors_v2(form, DRModels.Gaussian(), dat, labels, [sd, ridge], rows)
    @test length(priors) == 2
    @test_throws ArgumentError DRModels._bridge_prepare_priors_v2(form, DRModels.Gaussian(), dat, labels, [sd, ridge], reverse(rows))
    bad = deepcopy(rows); bad[2]["lambda"] = 0.7
    @test_throws ArgumentError DRModels._bridge_prepare_priors_v2(form, DRModels.Gaussian(), dat, labels, [sd, ridge], bad)
end

@testset "v2 response reports separately auditable prior components" begin
    @test isdefined(DRModels, :_bridge_attach_priors_v2!)
    if isdefined(DRModels, :_bridge_attach_priors_v2!)
        data=(y=[1.,1.4,2.3,2.8,2.1,2.6,3.2,3.4,4.3], x=[-1.,0.,1.,-1.,0.,1.,-1.,0.,1.], site=repeat(["a","b","c"], inner=3))
        form, dat, labels = DRModels._bridge_formula(Dict("mu"=>"y ~ x + (1 | site)", "sigma"=>"sigma ~ 1"), "gaussian", DRModels._bridge_data(data); labels=true)
        p=DRModels.drm_priors(DRModels.drm_prior_sd_gamma(dpar="mu", group="site", coef="(Intercept)", shape=2., rate=3.))
        fit=DRModels.drm(form, DRModels.Gaussian(); data=dat, priors=p)
        out=Dict{String,Any}()
        rec=[Dict("spec_index"=>1, "kind"=>"ranef_sd_gamma", "class"=>"ranef_sd", "dpar"=>"mu", "group"=>"site", "coef"=>"(Intercept)", "block_kind"=>"iid_random_intercept", "shape"=>2., "rate"=>3., "measure"=>"natural_sd")]
        DRModels._bridge_attach_priors_v2!(out, fit, rec, labels)
        @test out["data_logLik"] ≈ -out["data_nll"]
        @test out["penalized_objective"] ≈ out["data_nll"] + out["fixed_prior_penalty"] + out["sd_prior_penalty"]
        @test out["penalized_gradient"] ≈ out["data_gradient"] + out["fixed_prior_gradient"] + out["sd_prior_gradient"]
        @test length(out["working_coordinates"]) == length(out["working_values"]) == length(fit.theta)
    end
end

@testset "ordinary bridge admits validated SD v2 request" begin
    data=(y=[1.,1.4,2.3,2.8,2.1,2.6,3.2,3.4,4.3], x=[-1.,0.,1.,-1.,0.,1.,-1.,0.,1.], site=repeat(["a","b","c"], inner=3))
    sd=Dict("kind"=>"ranef_sd_gamma", "dpar"=>"mu", "group"=>"site", "coef"=>"(Intercept)", "block_kind"=>"iid_random_intercept", "shape"=>2., "rate"=>3., "measure"=>"natural_sd")
    row=merge(sd,Dict("spec_index"=>1,"class"=>"ranef_sd"))
    opts=Dict("prior_contract"=>"parameter_priors_v2", "priors_v2"=>[sd], "prior_expected_v2"=>[row])
    formula=Dict("mu"=>"y ~ x + (1 | site)", "sigma"=>"sigma ~ 1")
    bad = deepcopy(opts); bad["prior_expected_v2"][1]["rate"] = 4.
    @test_throws ArgumentError DRModels.drm_bridge(;formula=formula,family="gaussian",data=data,options=bad)
    bad = merge(opts,Dict("method"=>"REML"))
    @test_throws ArgumentError DRModels.drm_bridge(;formula=formula,family="gaussian",data=data,options=bad)
    out=DRModels.drm_bridge(;formula=formula,family="gaussian",data=data,options=opts)
    @test out["prior_contract"] == "parameter_priors_v2"
    @test out["estim_method"] == "MAP"
    @test out["infocrit_basis"] == "unavailable"
end

@testset "phylogenetic MAP retains existing bridge information criteria" begin
    using Random, LinearAlgebra
    rng = MersenneTwister(244)
    tree = random_balanced_tree(8; branch_length=.3)
    C = sigma_phy_dense(tree; σ²_phy=1.0)
    d = sqrt.(diag(C)); K = C ./ (d*d')
    species = repeat(1:8, inner=3); x = randn(rng, 24)
    u = .9 .* (cholesky(Symmetric(K)).L * randn(rng, 8))
    y = .2 .+ .5 .* x .+ u[species] .+ .4 .* randn(rng, 24)
    form = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1))
    fit = drm(form, Gaussian(); data=(; y,x,species), tree=tree, penalty=drm_phylo_penalty(sd_u=.6))
    @test fit.estim_method === :MAP
    out = DRModels._bridge_flatten(fit; family="gaussian")
    @test out["aic"] == aic(fit)
    @test out["bic"] == bic(fit)
    @test out["infocrit_basis"] == "ml"
end
