# A NaN in `vcov` at a flagged boundary coordinate must not remove the standard errors of
# `repeatability(fit, newdata)` / `icc` / `heritability` rows that do not depend on it
# (DRModels.jl#1383). `0 * NaN` is NaN, so the product with the full matrix lost every row.

using DRModels
using Test
using Random
using LinearAlgebra
using Logging

@testset "conditional ratio standard errors with a NaN in vcov (#1383)" begin
    rng = Random.MersenneTwister(20260715)
    n_id, n_each = 80, 6
    sex = repeat([0.0, 1.0], inner = n_id ÷ 2)
    b = randn(rng, n_id) .* [0.65, 0.40][Int.(sex) .+ 1]
    id = repeat(1:n_id, inner = n_each)
    sexl = sex[id]
    y = [0.35, 0.70][Int.(sexl) .+ 1] .+ b[id] .+
        randn(rng, n_id * n_each) .* [0.35, 0.60][Int.(sexl) .+ 1]
    dat = (; y, sex = sexl, id)
    fit = Logging.with_logger(Logging.NullLogger()) do
        drm(bf(@formula(y ~ sex + (1 | id)), @formula(sigma ~ sex),
               @formula(sd(id) ~ sex)), Gaussian(); data = dat)
    end
    nd = (; sex = [0.0, 1.0, 0.5])
    V0 = copy(fit.vcov)
    r0 = repeatability(fit, nd)
    @test all(isfinite, r0.se_logit) && r0.status === :ok
    n = length(fit.theta)
    # Coordinates 1 and 2 are the mean intercept and slope; the ratio does not use them.
    @test n == 6
    try
        fit.vcov[1, :] .= NaN
        fit.vcov[:, 1] .= NaN
        r = repeatability(fit, nd)
        @test r.se_logit == r0.se_logit
        @test all(isfinite, r.lower) && all(isfinite, r.upper)
        @test r.status === :ok
        # Both public aliases take the same path.
        @test icc(fit, nd).se_logit == r0.se_logit
        # A flagged coordinate the ratio does depend on keeps its NaN (no invented value).
        fit.vcov .= V0
        # Coordinate 3 is the sigma intercept, which every row uses.
        fit.vcov[3, :] .= NaN
        fit.vcov[:, 3] .= NaN
        rd = repeatability(fit, nd)
        @test all(isnan, rd.se_logit)
        @test rd.status === :vcov_unavailable
    finally
        fit.vcov .= V0
    end
    @test repeatability(fit, nd).se_logit == r0.se_logit
end
