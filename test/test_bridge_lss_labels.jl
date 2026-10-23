using DRModels, Test, Random, LinearAlgebra

@testset "actual LSS transformed scale coefficient labels" begin
    rng = MersenneTwister(563317)
    group = repeat(1:16, inner=6)
    gx = collect(range(0.2,1.6;length=16))
    x = gx[group]
    z = randn(rng,length(group))
    u = exp.(0.1 .+ 0.6 .* gx.^2) .* randn(rng,16)
    y = 0.3 .+ 0.25 .* z .+ u[group] .+ 0.4 .* randn(rng,length(group))
    data = (; y,x,z,g=string.(group))
    out = drm_bridge(formula="y ~ z + (1 | g); sigma ~ 1; sd(g) ~ I(x^2)",
                     family="gaussian",data=data)
    # A direct-Julia fit to the explicitly computed covariate is independent of
    # the R-formula materializer and its label reconstruction.
    direct = drm(bf(@formula(y ~ z + (1 | g)), @formula(sigma ~ 1),
                    @formula(sd(g) ~ x2)), Gaussian(); data=merge(data,(x2=x.^2,)))
    @test out["converged"]
    @test is_converged(direct)
    @test "sd_I(x^2)" in out["coef_names"]
    @test out["coef_names"] == out["vcov_names"]
    @test out["coef_name_map"]["sd_I(x^2)"] == "sd___bridge_I_1"
    @test out["coefficients"] ≈ coef(direct) atol=1e-9
    @test out["loglik"] ≈ loglik(direct) atol=1e-10
    V = out["vcov"]; oracle = vcov(direct)
    @test size(V) == size(oracle)
    @test isequal(isfinite.(V),isfinite.(oracle))
    @test V[isfinite.(V)] ≈ oracle[isfinite.(oracle)] atol=1e-8
end
