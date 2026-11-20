# refs #1353 — Binomial sigma() must not expose internal :trials bookkeeping.
module TestIssue1353SigmaBinomial

using DRModels
using Test

@testset "#1353 sigma(Binomial)" begin
    x = [sin(i) for i in 1:60]
    y = [sin(3i) + x[i] + 1.2cos(11i) > 0 ? 1 : 0 for i in 1:60]
    fit = drm(bf(@formula(y ~ x)), Binomial(); data = (; y, x))
    @test sigma(fit) isa Dict
    @test isempty(sigma(fit))
end

end