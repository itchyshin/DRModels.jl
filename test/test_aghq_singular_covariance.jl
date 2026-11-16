using DRModels, Test, LinearAlgebra

@testset "adaptive quadrature at a nearly singular covariance" begin
    y = [0.2, -0.5, 0.8, 0.1, -0.3]
    x = [-1.2, -0.5, 0.0, 0.7, 1.3]
    Z = hcat(ones(length(x)), x)
    η = 0.4 .- 0.2 .* x
    σ = 0.7
    ll = (i, z) -> -0.5 * log(2π * σ^2) - 0.5 * ((y[i] - z) / σ)^2
    rule = DRModels._AGHQRule(2, 5)
    for logdiag in (-14.0, -21.17672954245707)
        L = [exp(-1.6947434146926605) 0.0; 0.031332581013492455 exp(logdiag)]
        V = σ^2 * I + Z * L * L' * Z'
        F = cholesky(Symmetric(V))
        residual = y - η
        expected = -0.5 * (length(y)*log(2π) + 2sum(log, diag(F.L)) + dot(residual, F \ residual))
        members = [collect(eachindex(y))]
        actual = DRModels._aghq_marginal_loglik(ll, members, η, Z, L, rule, zeros(2, 1))
        @test isapprox(actual, expected; atol=1e-8, rtol=0)
        warm_cache = reshape([0.2, -0.02], 2, 1)
        warm_value = DRModels._aghq_marginal_loglik(ll, members, η, Z, L, rule, warm_cache)
        expected_mode = L * L' * Z' * (F \ residual)
        @test isapprox(warm_value, expected; atol=1e-8, rtol=0)
        @test isapprox(vec(warm_cache), expected_mode; atol=1e-8, rtol=0)
    end
end

@testset "adaptive quadrature derivatives match a dense Gaussian" begin
    using ForwardDiff
    y = [0.2, -0.5, 0.8, 0.1, -0.3]
    x = [-1.2, -0.5, 0.0, 0.7, 1.3]
    Z = hcat(ones(length(x)), x)
    members = [collect(eachindex(y))]
    rule = DRModels._AGHQRule(2, 5)
    function dense(θ)
        η = θ[1] .+ θ[2] .* x
        σ = exp(θ[3])
        L = [exp(θ[4]) zero(θ[4]); θ[6] exp(θ[5])]
        F = cholesky(Symmetric(σ^2 * I + Z * L * L' * Z'))
        residual = y - η
        return -0.5 * (length(y)*log(2π) + 2sum(log, diag(F.L)) + dot(residual, F \ residual))
    end
    function aghq(θ)
        η = θ[1] .+ θ[2] .* x
        σ = exp(θ[3])
        L = [exp(θ[4]) zero(θ[4]); θ[6] exp(θ[5])]
        ll = (i, z) -> -0.5 * log(2π * σ^2) - 0.5 * ((y[i] - z) / σ)^2
        return DRModels._aghq_marginal_loglik(ll, members, η, Z, L, rule, zeros(2, 1))
    end
    for logdiag in (-2.0, -10.0, -21.17672954245707)
        θ = [0.4, -0.2, log(0.7), -1.6947434146926605, logdiag, 0.031332581013492455]
        @test isapprox(aghq(θ), dense(θ); atol=1e-8, rtol=0)
        @test isapprox(ForwardDiff.gradient(aghq, θ), ForwardDiff.gradient(dense, θ); atol=1e-7, rtol=1e-7)
        @test isapprox(ForwardDiff.hessian(aghq, θ), ForwardDiff.hessian(dense, θ); atol=1e-6, rtol=1e-6)
    end
end

@testset "correlated Gamma quadrature retains its interior orientation" begin
    using DelimitedFiles, Distributions
    raw, _ = readdlm(joinpath(@__DIR__, "fixtures", "corr_slope_retry", "data.csv"), ','; header=true)
    raw = raw[String.(raw[:, 1]) .== "gamma3", :]
    y, x, g = Float64.(raw[:, 2]), Float64.(raw[:, 3]), String.(raw[:, 4])
    members = [findall(==(id), g) for id in unique(g)]
    Z = hcat(ones(length(x)), x)
    η = 0.8260536407198629 .+ 0.2926324303589577 .* x
    shape = exp(-2 * -0.7438908025861597)
    ll = (i, z) -> logpdf(Distributions.Gamma(shape, exp(z) / shape), y[i])
    rule = DRModels._AGHQRule(2, 5)
    # Baseline values from cbd870be; tensor-product GH rules are sensitive to rotation.
    for (d, expected) in ((-1.0, -183.82153308561422), (-2.0, -178.26242187487054),
                          (-3.0, -177.07100526116702), (-5.0, -176.90161407931805))
        L = [exp(-1.6947434146926605) 0.0; 0.031332581013492455 exp(d)]
        actual = DRModels._aghq_marginal_loglik(ll, members, η, Z, L, rule, zeros(2, length(members)))
        @test isapprox(actual, expected; atol=1e-8, rtol=0)
    end
end

@testset "zero covariance scale remains an impossible point" begin
    ll = (i,z) -> -0.5*z^2
    value = try
        first(DRModels._aghq_group_logint(ll, [1], zeros(1), ones(1,2), [1.0 0.0;0.1 0.0], DRModels._AGHQRule(2,5), zeros(2)))
    catch e
        e
    end
    @test value === -Inf
end
@testset "group quadrature allocation does not scale with unrelated rows" begin
    n=100_000; Z=ones(n,2); eta=zeros(n); L=[0.4 0.0;0.1 0.3]
    ll=(i,z)->-0.5*z^2
    rule=DRModels._AGHQRule(2,5); idx=collect(1:5); start=zeros(2)
    f=()->DRModels._aghq_group_logint(ll,idx,eta,Z,L,rule,start)
    f()
    bytes=@allocated f()
    @test bytes < 500_000
end
