# refs #1353 — Binomial `sigma(fit)` must not leak internal `:trials` bookkeeping.
module TestIssue1353SigmaBinomial

using DRModels
using Test

@testset "#1353 Binomial sigma() hides :trials" begin
    x = [sin(i) for i in 1:60]
    y = [sin(3 * i) + x[i] + 1.2 * cos(11 * i) > 0 ? 1 : 0 for i in 1:60]
    fit = drm(bf(@formula(y ~ x)), Binomial(); data = (; y, x))
    s = sigma(fit)
    @test isempty(s)
    @test sigma(drm(bf(@formula(c ~ x)), Poisson(); data = (; c = round.(Int, 3 .+ x), x))) == Dict()
end

end