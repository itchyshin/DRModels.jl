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
        # Reject unsupported selectors even when the expected row repeats the
        # same invalid target, before any call to the fitting engine.
        for (key, value) in (("block_kind", "correlated"), ("dpar", "sigma"),
                             ("coef", "x"), ("measure", "log_sd"))
            badspec = copy(spec); badspec[key] = value
            badrecord = copy(expected); badrecord[key] = value
            @test_throws ArgumentError prepare(form, DRModels.Gaussian(), dat, labels, [badspec], [badrecord])
        end
        @test_throws ArgumentError prepare(form, DRModels.Gaussian(), dat, labels, [spec], [])
        @test_throws ArgumentError prepare(form, DRModels.Gaussian(), dat, labels, [spec], [expected, copy(expected)])
        badrecord = copy(expected); delete!(badrecord, "coef")
        @test_throws ArgumentError prepare(form, DRModels.Gaussian(), dat, labels, [spec], [badrecord])
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

@testset "v2 string bridge rejects duplicated random intercept before fitting" begin
    data=(y=[1.,1.4,2.3,2.8,2.1,2.6,3.2,3.4,4.3], x=[-1.,0.,1.,-1.,0.,1.,-1.,0.,1.], site=repeat(["a","b","c"], inner=3))
    sd=Dict("kind"=>"ranef_sd_gamma", "dpar"=>"mu", "group"=>"site", "coef"=>"(Intercept)", "block_kind"=>"iid_random_intercept", "shape"=>2., "rate"=>3., "measure"=>"natural_sd")
    row=merge(sd,Dict("spec_index"=>1,"class"=>"ranef_sd"))
    opts=Dict("prior_contract"=>"parameter_priors_v2", "priors_v2"=>[sd], "prior_expected_v2"=>[row])
    duplicated=Dict("mu"=>"y ~ x + (1 | site) + (1 | site)", "sigma"=>"sigma ~ 1")
    err = try
        DRModels.drm_bridge(;formula=duplicated, family="gaussian", data=data, options=opts)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test err isa ArgumentError && occursin("duplicated random intercept", sprint(showerror, err))

    # The R embedded conditional route uses this same preparation sequence,
    # before it calls its fitter. A counter proves failure happens beforehand.
    calls = Ref(0)
    function conditional_prepare_then_fit(formula)
        dat=DRModels._bridge_data(data)
        bundle, dat, labels=DRModels._bridge_formula(formula, "gaussian", dat; labels=true)
        parsed=DRModels._bridge_prior_safe_options(opts)
        DRModels._bridge_enable_priors_v2!(parsed, bundle, DRModels.Gaussian(), dat, labels)
        calls[] += 1
    end
    for f in (duplicated, "y ~ x + ((1 | site) + (1 | site)); sigma ~ 1",
              ["y ~ x + (1|site) + (1|site)", "sigma ~ 1"],
              (mu="y ~ x + (1 | site) + (1.0 | site)", sigma="sigma ~ 1"))
        @test_throws ArgumentError conditional_prepare_then_fit(f)
        @test calls[] == 0
    end
    conditional_prepare_then_fit(Dict("mu"=>"y ~ x + (1 | site)", "sigma"=>"sigma ~ 1"))
    @test calls[] == 1
    # Different grouping variables are not mistaken for duplicate spellings;
    # the native target resolver separately rejects multiple blocks.
    @test isnothing(DRModels._bridge_reject_duplicate_intercept_source(["y ~ x + (1 | site) + (1 | site2)"]))
    # Existing non-v2 calls retain their prior parser behavior.
    plain = DRModels.drm_bridge(;formula=duplicated, family="gaussian", data=data)
    @test plain["estim_method"] == "ML"
    v1 = Dict("priors"=>[Dict("kind"=>"normal", "dpar"=>"mu", "coef"=>"x", "mean"=>0., "sd"=>2.)],
              "prior_expected"=>[Dict("dpar"=>"mu", "coef"=>"x")])
    fixed = DRModels.drm_bridge(;formula=duplicated, family="gaussian", data=data, options=v1)
    @test fixed["fixed_prior_contract"] == "fixed_effect_priors_v1"
end

@testset "v2 refuses random-bar algebra before fitting" begin
    # Pure fixed-effect algebra and additive random-intercept grouping remain valid.
    check = DRModels._bridge_reject_duplicate_intercept_source
    for f in ("y ~ (x^2) + (1 | site)", "y ~ (x * z) + (1 | site)",
              "y ~ (x / 2) + (1 | site)", "y ~ (x - 1) + (1 | site)",
              "y ~ x + (1 | site) - 1", "y ~ x + (1 | site) - x",
              "y ~ ifelse((x > 0) | (x < -0.5), 1.0, 0.0) + (1 | site)",
              "y ~ ifelse(flag1 | flag2, 1.0, 0.0) + (1 | site)",
              "y ~ ifelse(((1 | site) > 1), 1.0, 0.0) + (1 | site)",
              "y ~ identity(1 | site) + (1 | site)",
              "y ~ scale(x) + ((1 | site) + z)")
        @test isnothing(check([f]))
    end
    for f in ("y ~ x + ((1 | site)^2)", "y ~ x + ((1 | site) * x)",
              "y ~ x + ((1 | site) & x)", "y ~ x + ((1 | site) / 2)",
              "y ~ x - (1 | site)")
        @test_throws ArgumentError check([f])
    end
    data=(y=[1.,1.4,2.3,2.8,2.1,2.6,3.2,3.4,4.3], x=[-1.,0.,1.,-1.,0.,1.,-1.,0.,1.], site=repeat(["a","b","c"], inner=3))
    sd=Dict("kind"=>"ranef_sd_gamma", "dpar"=>"mu", "group"=>"site", "coef"=>"(Intercept)", "block_kind"=>"iid_random_intercept", "shape"=>2., "rate"=>3., "measure"=>"natural_sd")
    row=merge(sd,Dict("spec_index"=>1,"class"=>"ranef_sd"))
    opts=Dict("prior_contract"=>"parameter_priors_v2", "priors_v2"=>[sd], "prior_expected_v2"=>[row])
    logical=Dict("mu"=>"y ~ ifelse((x > 0) | (x < -0.5), 1.0, 0.0) + (1 | site)", "sigma"=>"sigma ~ 1")
    @test DRModels.drm_bridge(;formula=logical, family="gaussian", data=data)["estim_method"] == "ML"
    @test DRModels.drm_bridge(;formula=logical, family="gaussian", data=data, options=opts)["estim_method"] == "MAP"
    flags=merge(data,(flag1=[true,false,false,true,false,false,true,false,false], flag2=[false,true,false,false,true,false,false,true,false]))
    logical_symbols=Dict("mu"=>"y ~ ifelse(flag1 | flag2, 1.0, 0.0) + (1 | site)", "sigma"=>"sigma ~ 1")
    @test DRModels.drm_bridge(;formula=logical_symbols, family="gaussian", data=flags)["estim_method"] == "ML"
    @test DRModels.drm_bridge(;formula=logical_symbols, family="gaussian", data=flags, options=opts)["estim_method"] == "MAP"
    numeric_site=merge(data,(site=repeat([1,2,3], inner=3),))
    fixed_bitwise=Dict("mu"=>"y ~ ifelse(((1 | site) > 1), 1.0, 0.0) + (1 | site)", "sigma"=>"sigma ~ 1")
    @test DRModels.drm_bridge(;formula=fixed_bitwise, family="gaussian", data=numeric_site)["estim_method"] == "ML"
    @test DRModels.drm_bridge(;formula=fixed_bitwise, family="gaussian", data=numeric_site, options=opts)["estim_method"] == "MAP"
    nointercept=Dict("mu"=>"y ~ x + (1 | site) - 1", "sigma"=>"sigma ~ 1")
    @test DRModels.drm_bridge(;formula=nointercept, family="gaussian", data=data, options=opts)["estim_method"] == "MAP"
    ambiguous=Dict("mu"=>"y ~ x + ((1 | site)^2)", "sigma"=>"sigma ~ 1")
    @test_throws ArgumentError DRModels.drm_bridge(;formula=ambiguous, family="gaussian", data=data, options=opts)
    calls = Ref(0)
    @test_throws ArgumentError begin
        dat=DRModels._bridge_data(data)
        bundle, dat, labels=DRModels._bridge_formula(ambiguous, "gaussian", dat; labels=true)
        parsed=DRModels._bridge_prior_safe_options(opts)
        DRModels._bridge_enable_priors_v2!(parsed, bundle, DRModels.Gaussian(), dat, labels)
        calls[] += 1
    end
    @test calls[] == 0
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
