using Test, DRModels

@testset "variance-ratio intervals refuse active fixed-prior MAP fits" begin
    dat = (y=[1.0, 1.4, 2.0, 2.7, 1.2, 1.9, 3.2, 3.4],
           x=[0., 1., 0., 1., 0., 1., 0., 1.], id=repeat(1:4, inner=2))
    form = bf(@formula(y ~ x + (1 | id)))
    active = drm_priors(drm_prior_normal(dpar="mu", coef="x", mean=0.2, sd=2))
    zero = drm_priors(drm_prior_ridge(dpar="mu", coef="x", lambda=0))
    mapfit = drm(form, Gaussian(); data=dat, priors=active)
    plain = drm(form, Gaussian(); data=dat)
    zerofit = drm(form, Gaussian(); data=dat, priors=zero)
    for accessor in (heritability, icc, repeatability), method in (:delta, :profile)
        err = try
            accessor(mapfit; method=method)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("variance-ratio intervals are not qualified for fixed-prior MAP fits", sprint(showerror, err))
    end
    for accessor in (heritability, icc, repeatability)
        @test accessor(zerofit; method=:delta).estimate ≈ accessor(plain; method=:delta).estimate atol=1e-10
    end
    @test icc(zerofit; method=:profile).estimate ≈ icc(plain; method=:profile).estimate atol=1e-10
    @test haskey(ranef(mapfit), :id)
    @test isfinite(loglik(mapfit))
    @test summary(mapfit) !== nothing
end

@testset "other MAP penalties keep existing variance-ratio status" begin
    phy = random_balanced_tree(8; branch_length=0.3)
    species = repeat(1:8, inner=3)
    x = repeat([-1., 0., 1.], 8)
    y = 0.4 .+ 0.3 .* x .+ 0.15 .* sin.(species) .+ 0.1 .* cos.(1:24)
    fit = drm(bf(@formula(y ~ x + phylo(1 | species))), Gaussian();
              data=(; y, x, species), tree=phy, penalty=drm_phylo_penalty())
    @test fit.estim_method === :MAP
    @test !DRModels._is_fixed_prior_fit(fit)
    ratio = @test_logs (:warn, r"penalized \(MAP\)") heritability(fit; method=:delta)
    @test ratio.status === :penalized_map
end
