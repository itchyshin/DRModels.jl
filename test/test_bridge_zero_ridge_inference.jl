using DRModels, Test, Random

@testset "bridge inference decodes v1 zero-ridge priors" begin
    rng = MersenneTwister(56329)
    n = 80
    x = randn(rng, n)
    y = 0.4 .+ 0.7 .* x .+ 0.6 .* randn(rng, n)
    data = (; y, x)
    formula = "y ~ x; sigma ~ 1"
    form = bf(@formula(y ~ x), @formula(sigma ~ 1))
    table = (prior=["ridge(0)"],class=["fixef"],coef=["x"])
    specs = (DRModels._table_priors_for_model(form, Gaussian(), data, table),
             drm_priors(drm_prior_ridge(dpar="mu",coef="x",lambda=0)))
    payload(p) = Dict("priors"=>[Dict("kind"=>"ridge", "dpar"=>String(only(p).dpar),
                                      "coef"=>only(p).coef, "lambda"=>only(p).precision)],
                      "prior_expected"=>[Dict("dpar"=>"mu","coef"=>"x")])
    options = payload.(specs)
    for method in ("profile", "bootstrap")
        kwargs = (; formula, family="gaussian", data, method, level=0.9, B=5, seed=2026,
                  parm="fixef:mu:x")
        plain = drm_bridge_inference(;kwargs...)
        for opts in options
            out = drm_bridge_inference(;kwargs...,options=opts)
            @test out["coef"] == plain["coef"] == "x"
            @test out["estimate"] ≈ plain["estimate"] atol=1e-7
            @test out["lower"] ≈ plain["lower"] atol=1e-6
            @test out["upper"] ≈ plain["upper"] atol=1e-6
            @test out["attempted"] == plain["attempted"]
            @test out["used"] == plain["used"]
            @test out["failed"] == plain["failed"]
        end
        for empty in (DRModels.FixedPrior[], DRModels.AbstractDrmPrior[])
            empty_fit = drm_bridge_inference(;kwargs...,options=Dict("priors"=>empty))
            @test empty_fit["estimate"] ≈ plain["estimate"] atol=1e-7
            @test empty_fit["lower"] ≈ plain["lower"] atol=1e-6
            @test empty_fit["upper"] ≈ plain["upper"] atol=1e-6
        end
        # An empty Any[] could be an omitted wire payload. A v1 expected record
        # paired with typed prior objects also has no validated wire meaning.
        @test_throws ArgumentError drm_bridge_inference(;kwargs...,options=Dict("priors"=>Any[]))
        @test_throws ArgumentError drm_bridge_inference(;kwargs...,options=Dict("priors"=>specs[2],"prior_expected"=>options[1]["prior_expected"]))
        typed = drm_bridge_inference(;kwargs...,options=Dict("priors"=>specs[2]))
        @test typed["estimate"] ≈ plain["estimate"] atol=1e-7
        @test typed["lower"] ≈ plain["lower"] atol=1e-6
        @test typed["upper"] ≈ plain["upper"] atol=1e-6
        for version in (2,3)
            unsupported = Dict("prior_contract"=>"parameter_priors_v$version",
                               "priors_v$version"=>[Dict("kind"=>"ranef_sd_gamma")],
                               "prior_expected_v$version"=>[Dict("group"=>"site")])
            err = try
                drm_bridge_inference(;kwargs...,options=unsupported)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("SD-prior inference is not supported", sprint(showerror,err))
        end
        missing_expected = Dict("priors"=>options[1]["priors"])
        wrong_expected = merge(options[1],Dict("prior_expected"=>[Dict("dpar"=>"mu","coef"=>"(Intercept)")]))
        orphan_expected = Dict("prior_expected"=>options[1]["prior_expected"])
        for bad in (missing_expected, orphan_expected)
            @test_throws ArgumentError drm_bridge_inference(;kwargs...,options=bad)
        end
        wrong_error = try
            drm_bridge_inference(;kwargs...,options=wrong_expected)
            nothing
        catch e
            e
        end
        @test wrong_error isa ArgumentError
        @test wrong_error isa ArgumentError && occursin("resolved prior coordinates differ", sprint(showerror,wrong_error))
        active = Dict("priors"=>[Dict("kind"=>"normal","dpar"=>"mu","coef"=>"x","mean"=>0.,"sd"=>2.)],
                      "prior_expected"=>[Dict("dpar"=>"mu","coef"=>"x")])
        active_error = try
            drm_bridge_inference(;kwargs...,options=active)
            nothing
        catch e
            e
        end
        @test active_error isa ArgumentError
        @test active_error isa ArgumentError && occursin(r"not (yet )?qualified", sprint(showerror,active_error))
    end
end
