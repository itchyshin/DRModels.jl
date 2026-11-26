# cov lane 1125n8 — lognormal Jacobian shifts (src/lognormal.jl, src/bivariate_lognormal.jl).
using DRModels
using Test, LinearAlgebra

function _mini_gaussian_fit(y; bivariate = false)
    n = length(y)
    if bivariate
        blocks = [:mu1 => 1:1, :sigma1 => 2:2, :rho12 => 3:3]
        names = [:mu1 => ["(Intercept)"], :sigma1 => ["(Intercept)"], :rho12 => ["(Intercept)"]]
        θ = [0.0, -0.5, 0.0]
        means = Dict(:mu1 => fill(θ[1], n), :mu2 => fill(θ[1], n))
        obs = Dict(:mu1 => y, :mu2 => y)
        scales = Dict(:sigma1 => fill(exp(θ[2]), n), :sigma2 => fill(exp(θ[2]), n),
                      :rho12 => fill(0.0, n))
    else
        blocks = [:mu => 1:1, :sigma => 2:2]
        names = [:mu => ["(Intercept)"], :sigma => ["(Intercept)"]]
        θ = [0.0, -0.5]
        means = Dict(:mu => fill(θ[1], n))
        obs = Dict(:mu => y)
        scales = Dict(:sigma => fill(exp(θ[2]), n))
    end
    V = Matrix{Float64}(I, length(θ), length(θ))
    ll = -10.0
    DrmFit(Gaussian(), blocks, names, θ, V, ll, n, true, means, obs, scales)
end

@testset "cov 1125n8: univariate _lognormal_jacobian_shift" begin
    y = [1.5, 2.0, 3.5]
    gfit = _mini_gaussian_fit(y)
    out = DRModels._lognormal_jacobian_shift(LogNormal(), gfit, Float64.(y))
    jac = sum(log, y)
    @test out.loglik ≈ gfit.loglik - jac
    @test out.ml_loglik ≈ gfit.ml_loglik - jac
    @test out.means[:mu] ≈ exp.(gfit.means[:mu])
    @test out.obs[:mu] == y
    @test out.family isa LogNormal
    gnll = θ -> sum(abs2, θ)
    gfit2 = DrmFit(Gaussian(), gfit.blocks, gfit.coefnames, gfit.theta, gfit.vcov,
                   gfit.loglik, gfit.nobs, true, gfit.means, gfit.obs, gfit.scales,
                   nothing, gnll, nothing, nothing, :ML, 5.0, gfit.loglik, :LA)
    out2 = DRModels._lognormal_jacobian_shift(LogNormal(), gfit2, Float64.(y))
    @test out2.reml_loglik ≈ 5.0 - jac
    @test out2.nll !== nothing
    @test out2.nll(gfit2.theta) ≈ gnll(gfit2.theta) + jac
end

@testset "cov 1125n8: bivariate _lognormal_jacobian_shift and logging helpers" begin
    y1 = [1.0, 2.0, 3.0, 4.0]
    y2 = [1.5, 0.5, 2.0, 3.0]
    obs1 = trues(4)
    obs2 = trues(4)
    obs1[3] = false
    @test_throws ArgumentError DRModels._check_positive_response(
        [1.0, 0.0, 2.0], trues(3), :y1)
    DRModels._check_positive_response(Float64[1.0, 2.0], trues(2), :y1)
    data = (; a = y1, b = y2, x = 1:4)
    logd = DRModels._with_logged_responses(data, :a, :b, y1, y2, obs1, obs2)
    @test logd.a[1] ≈ log(1.0)
    @test !obs1[3] && logd.a[3] == y1[3]
    @test logd.b[2] ≈ log(0.5)
    n = 4
    gfit = _mini_gaussian_fit(rand(n); bivariate = true)
    out = DRModels._lognormal_jacobian_shift(LogNormal(), gfit, y1, y2, obs1, obs2)
    jac = sum(i -> obs1[i] ? log(y1[i]) : 0.0, 1:n) +
          sum(i -> obs2[i] ? log(y2[i]) : 0.0, 1:n)
    @test out.loglik ≈ gfit.loglik - jac
    @test haskey(out.means, :mu1)
end