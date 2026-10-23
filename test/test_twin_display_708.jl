# #708 (duplicate: #763) — `re_sd(fit)` returned an empty Dict for a correlated
# random-effect block `(1 + x | g)`, while `vc(fit)` already exposed the same
# information (as a 2x2 covariance matrix, keyed by grouping factor). This
# guards that `re_sd` now returns the intercept/slope SDs for a correlated
# block, consistent with `sqrt.(diag(vc(fit)[:g]))`.
using DRModels
using Test, Random, LinearAlgebra

@testset "#708/#763 — re_sd on a correlated random-effect block" begin
    Random.seed!(20270905)
    n_groups = 12
    n_per = 8
    g = repeat(1:n_groups, inner = n_per)
    x = Float64.(repeat(0:n_per-1, outer = n_groups))
    b0 = 0.5 .* randn(n_groups)
    b1 = 0.2 .* randn(n_groups)
    y = 0.5 .+ 0.15 .* x .+ b0[g] .+ b1[g] .* x .+ 0.5 .* randn(length(g))
    data = (; y, x, g)

    fit = drm(bf(@formula(y ~ 1 + x + (1 + x | g))), Gaussian(); data = data)

    rs = re_sd(fit)
    @test !isempty(rs)
    @test haskey(rs, :g_intercept)
    @test haskey(rs, :g_slope)

    # Must be consistent with the already-working vc(fit).
    Σ = vc(fit)[:g]
    @test rs[:g_intercept] ≈ sqrt(Σ[1, 1]) atol = 1e-10
    @test rs[:g_slope] ≈ sqrt(Σ[2, 2]) atol = 1e-10
end
