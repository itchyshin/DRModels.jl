# gen_skewnormal_restarts_ridge.jl -- regenerates ridge_sim235.csv, ridge_sim25.csv (n = 50 each)
# and ridge_rev23.csv (n = 54), the refusal fixtures of test_skewnormal_restarts.jl.
#
#   julia --project=test test/fixtures/skewnormal_restarts/gen_skewnormal_restarts_ridge.jl
#
# sim235 and sim25: the restart study design. x ~ N(0, 1); mu = 1 + 0.5 x; sigma = exp(0.2 (+ 0.3 x));
# slant = a (+ 0.7 x); moment form; Azzalini draw. StableRNG seed 9000 + id; a = -1 (id 235, sigma ~ 1)
# and a = -2 (id 25, sigma ~ x); the slant depends on x in both.
# rev23: an independent design (StableRNG 123000 + 23): n = 54, true slant -10, y ~ x1, sigma ~ x1,
# nu ~ 1, 1 to 3% outliers of 8 SD, no ties; only the columns y and x1 are written (x1 renamed x).
using StableRNGs, Printf, Random
function study(id, n, a, sx)
    rng = StableRNG(9000 + id)
    x = randn(rng, n); y = zeros(n)
    for i in 1:n
        mu = 1.0 + 0.5 * x[i]; sigma = exp(0.2 + (sx ? 0.3 * x[i] : 0.0)); alpha = a + 0.7 * x[i]
        delta = alpha / sqrt(1 + alpha^2)
        omega = sigma / sqrt(1 - 2delta^2 / pi)
        xi = mu - omega * delta * sqrt(2 / pi)
        u0 = abs(randn(rng)); u1 = randn(rng)
        y[i] = xi + omega * (delta * u0 + sqrt(1 - delta^2) * u1)
    end
    return y, x
end
function rev23()
    id = 23; rng = StableRNG(123000 + id)
    A = [-10.0, -6.0, -3.0, -1.5, -0.7, -0.3, -0.1, -0.03, 0.0, 0.0, 0.03, 0.1, 0.3, 0.7, 1.5, 3.0, 6.0, 10.0]
    n = round(Int, exp(log(30) + rand(rng) * (log(3000) - log(30))))
    a = A[mod1(id * 7 + rand(rng, 0:3), length(A))]
    v = rand(rng, 1:8); ties = rand(rng) < 0.25; outl = rand(rng) < 0.2
    x1 = randn(rng, n); x2 = rand(rng) < 0.5 ? 2 .* rand(rng, n) .- 1 : Float64.(rand(rng, Bool, n))
    y = zeros(n)
    for i in 1:n
        mu = 1.0 + 0.5 * x1[i]; sigma = exp(0.1 + 0.3 * x1[i]); alpha = a
        delta = alpha / sqrt(1 + alpha^2); omega = sigma / sqrt(1 - 2delta^2 / pi); xi = mu - omega * delta * sqrt(2 / pi)
        y[i] = xi + omega * (delta * abs(randn(rng)) + sqrt(1 - delta^2) * randn(rng))
    end
    if ties; y .= round.(y ./ 0.4) .* 0.4; end
    if outl
        k = max(1, round(Int, n * (0.01 + 0.02 * rand(rng)))); idx = rand(rng, 1:n, k)
        s = sqrt(sum(abs2, y .- sum(y) / n) / n); for j in idx; y[j] += (rand(rng, Bool) ? 1 : -1) * 8 * s; end
    end
    return y, x1
end
function write_csv(name, y, x)
    open(joinpath(@__DIR__, name), "w") do io
        println(io, "y,x")
        for i in eachindex(y); @printf(io, "%.17g,%.17g\n", y[i], x[i]); end
    end
end
write_csv("ridge_sim235.csv", study(235, 50, -1.0, false)...)
write_csv("ridge_sim25.csv", study(25, 50, -2.0, true)...)
write_csv("ridge_rev23.csv", rev23()...)
