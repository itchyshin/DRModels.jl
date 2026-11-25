using DRModels, Test, Random, Statistics

# #1496. `simulate(fit; data = data)` draws FRESH random slopes for an ordinary correlated `(1 + x | g)`
# and an independent `(1 | g) + (0 + x | g)` on the mean, as drmTMB's default `re.form = NA` does. The
# draw uses the fitted covariance `vc(fit)`. Gaussian is checked against the exact marginal variance
# Var(y_i) = sigma^2 + S00 + 2 x_i S01 + x_i^2 S11.
function _slope1496_data(seed)
    r = Random.Xoshiro(seed); G = 40; n = G * 10
    g = repeat(1:G, inner = 10); x = randn(r, n)
    b0 = 0.8 .* randn(r, G); b1 = 0.3 .* b0 .+ 0.4 .* randn(r, G)
    eta = 0.3 .+ 0.4 .* x .+ b0[g] .+ b1[g] .* x
    return (; x, g = string.("g", g), eta, y = eta .+ 0.5 .* randn(r, n))
end

@testset "simulate draws fresh slope effects (#1496)" begin
    d = _slope1496_data(11)
    for (label, f) in (("correlated", @formula(y ~ x + (1 + x | g))),
                       ("independent", @formula(y ~ x + (1 | g) + (0 + x | g))))
        fit = drm(bf(f), Gaussian(); data = d)
        Y = simulate(fit; nsim = 3000, rng = Random.Xoshiro(5), data = d)
        @test size(Y) == (length(d.y), 3000)
        v = vc(fit)
        S = label == "correlated" ? Matrix(v[:g]) :
            [only(v[:g]) 0.0; 0.0 only(v[Symbol("g:x")])]
        s2 = DRModels._scale_vector(fit, :sigma) .^ 2
        theory = s2 .+ S[1, 1] .+ 2 .* d.x .* S[1, 2] .+ d.x .^ 2 .* S[2, 2]
        emp = vec(var(Y; dims = 2))
        @test mean(emp ./ theory) ≈ 1.0 atol = 0.05
        # the draw is not the b = 0 draw: its variance exceeds the zero-effect draw's
        Z = simulate(fit; nsim = 300, rng = Random.Xoshiro(5), data = d, re = :zero)
        @test mean(var(Y; dims = 2)) > 1.2 * mean(var(Z; dims = 2))
    end
end

@testset "non-Gaussian slope draws carry between-group signal (#1496)" begin
    d = _slope1496_data(12)
    yp = Float64.(rand.(Random.Xoshiro(3), DRModels.Distributions.Poisson.(exp.(d.eta))))
    dp = merge(d, (; y = yp))
    fit = drm(bf(@formula(y ~ x + (1 + x | g))), Poisson(); data = dp)
    Y = simulate(fit; nsim = 400, rng = Random.Xoshiro(9), data = dp)
    Z = simulate(fit; nsim = 400, rng = Random.Xoshiro(9), data = dp, re = :zero)
    @test all(isfinite, Y) && all(>=(0), Y)
    @test mean(var(Y; dims = 2)) > 1.3 * mean(var(Z; dims = 2))
end
