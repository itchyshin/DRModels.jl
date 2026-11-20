# refs #1389 — ZeroOneBeta with missing responses must finish after row rebuild.
module TestIssue1389ZeroOneBetaMissing

using DRModels
using Test

@testset "#1389 ZeroOneBeta missing response" begin
    n = 60
    i = 1:n
    x = [sin(k) for k in i]
    g = [string("g", 1 + (k - 1) % 6) for k in i]
    y = Union{Missing,Float64}[
        (k % 6 == 0) ? missing :
        (k % 10 == 3) ? 0.0 :
        (k % 13 == 5) ? 1.0 :
        clamp(0.5 + 0.25 * sin(k) + 0.12 * cos(5k) + 0.1 * (((k - 1) % 6) - 2.5) / 2.5, 0.05, 0.95) for k in i
    ]
    d = (; y, x, g)
    fit = drm(bf(@formula(y ~ x)), ZeroOneBeta(); data = d)
    @test nobs(fit) == 50
    @test isfinite(loglikelihood(fit))
end

end