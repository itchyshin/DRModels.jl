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
    # well-identified: overdispersed, n = 60, 20 trials each (converges cleanly)
    x2 = collect(range(-1, 1, length = 60))
    pr = 1 ./ (1 .+ exp.(-(0.2 .+ 0.8 .* x2 .+ 0.9 .* sin.(7 .* (1:60)))))
    s2 = round.(Int, 20 .* pr); fl2 = 20 .- s2
    g = drm(bf(@formula(cbind(s2, fl2) ~ x2), @formula(sigma ~ 1)), BetaBinomial();
            data = (; s2, fl2, x2))
    @test g.geometry === :ok
    @test convergence_status(g) === :converged

    # zero-weight rows must not block the separation screen (prereview #1435)
    xz = vcat(x, [-1.5, 1.5]); sz = vcat(s, [5.0, 5.0]); flz = 10 .- sz
    wz = vcat(ones(n), zeros(2))
    fz = drm(bf(@formula(cbind(sz, flz) ~ xz), @formula(sigma ~ 1)), BetaBinomial();
             data = (; sz, flz, xz), weights = wz, se = false)
    @test fz.geometry === :betabinomial_separation
    yz = vcat(y, [1, 3]); xo = vcat(x, [1.5, -1.5]); wo = vcat(ones(n), zeros(2))
    fo = drm(bf(@formula(yz ~ xo)), CumulativeLogit(); data = (; yz, xo), weights = wo, se = false)
    @test fo.geometry === :ordinal_separation

    # check_drm must not call a separated fit ok
    f2 = drm(bf(@formula(y ~ x)), CumulativeLogit(); data = (; y, x))
    @test !check_drm(f2).ok
end
