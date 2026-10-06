using DRModels, Test, LinearAlgebra

@testset "bridge materialized columns never overwrite user data" begin
    x = collect(range(-1.5,1.5;length=12))
    user = sin.(collect(1.0:12.0))
    dat = (y=cos.(x), x=x, __bridge_I_1=user, __bridge_I_2=reverse(user))
    bundle, augmented = DRModels._bridge_formula("y ~ __bridge_I_1 + I(x^2); sigma ~ 1", "gaussian", dat)
    @test augmented.__bridge_I_1 == user
    @test augmented.__bridge_I_2 == reverse(user)
    _,X,names = DRModels._design(bundle.response,Dict(bundle.forms)[:mu],augmented)
    oracle = hcat(ones(12),user,x.^2)
    @test size(X) == size(oracle)
    @test rank(X) == 3
    @test X ≈ oracle atol=1e-14
    @test dat.__bridge_I_1 == user

    # Every materialization kind uses the same allocator, and cached identical
    # expressions must reuse a single generated column across parameters.
    for (kind,formula) in (("scale","y ~ scale(x); sigma ~ scale(x)"),
                           ("factor","y ~ factor(x); sigma ~ 1"),
                           ("poly2c1","y ~ poly(x, 2); sigma ~ 1"))
        reserved = Symbol("__bridge_",kind,"_1")
        input = merge((y=cos.(x),x=x),NamedTuple{(reserved,)}((user,)))
        f,aug = DRModels._bridge_formula(formula,"gaussian",input)
        @test getproperty(aug,reserved) == user
        @test getproperty(input,reserved) == user
        if kind == "scale"
            @test length(keys(aug)) == length(keys(input))+1
            _,Xm,_ = DRModels._design(f.response,Dict(f.forms)[:mu],aug)
            _,Xs,_ = DRModels._design(f.response,Dict(f.forms)[:sigma],aug)
            @test Xm == Xs
        end
    end
end
