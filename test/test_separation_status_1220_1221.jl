using Test, DRModels

# #1220 / #1221: separated BetaBinomial and CumulativeLogit fits must not read :converged
# (converged rule, SHINICHI-1753-1118: separation is :degenerate).
@testset "separation status BetaBinomial / CumulativeLogit (#1220 #1221)" begin
    n = 40; x = collect(range(-2, 2, length = n))
    s = Float64.(x .> 0) .* 10; fl = 10 .- s
    # se = true still throws the informative error (test_1220_betabinomial_separation_error.jl)
    f = drm(bf(@formula(cbind(s, fl) ~ x), @formula(sigma ~ 1)), BetaBinomial();
            data = (; s, fl, x), se = false)
    @test convergence_status(f) === :degenerate
    y = [x[i] < -0.7 ? 1 : x[i] < 0.7 ? 2 : 3 for i in 1:n]
    f = drm(bf(@formula(y ~ x)), CumulativeLogit(); data = (; y, x))
    @test convergence_status(f) === :degenerate
    # control: non-separated BetaBinomial still converges
    s2 = [3, 5, 4, 6, 7, 5, 4, 6, 5, 5]; fl2 = 10 .- s2; x2 = collect(1.0:10.0)
    g = drm(bf(@formula(cbind(s2, fl2) ~ x2), @formula(sigma ~ 1)), BetaBinomial();
            data = (; s2, fl2, x2))
    @test g.geometry === :ok
end
