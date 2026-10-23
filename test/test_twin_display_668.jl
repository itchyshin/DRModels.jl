# #668 — `StatsBase.loglikelihood(::DrmFit)` was missing (DRM.loglik worked),
# so anything dispatching on the StatsAPI/StatsBase generic (e.g. code written
# against the StatsAPI.StatisticalModel interface) errored on a DrmFit.
using DRModels
using Test

@testset "#668 — StatsAPI loglikelihood(fit) matches loglik(fit)" begin
    n = 40
    x = Float64.(1:n)
    y = 0.5 .+ 0.2 .* x .+ randn(n)
    fit = drm(bf(@formula(y ~ x)), Gaussian(); data = (; x, y))

    @test loglikelihood(fit) == loglik(fit)
    @test loglikelihood(fit) isa Float64
end
