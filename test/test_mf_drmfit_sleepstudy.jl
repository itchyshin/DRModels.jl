# Issue #1529: exported `mf_*` post-fit helpers must work on a standard `DrmFit`
# (sleepstudy Gaussian mixed model), delegating to StatsAPI accessors.

using Test
using DRModels

function _read_csv_lme4(name)
    lines = readlines(joinpath(@__DIR__, "fixtures", "lme4", name))
    hdr = [strip(c, '"') for c in split(lines[1], ',')]
    cols = [String[strip(split(l, ',')[j], '"') for l in lines[2:end]] for j in eachindex(hdr)]
    return Dict(Symbol(h) => c for (h, c) in zip(hdr, cols))
end

@testset "mf_* on DrmFit (sleepstudy, refs #1529)" begin
    ss = _read_csv_lme4("sleepstudy.csv")
    data = (; Reaction = parse.(Float64, ss[:Reaction]), Days = parse.(Float64, ss[:Days]),
            Subject = ss[:Subject])
    fit = drm(bf(@formula(Reaction ~ Days + (1 | Subject))), Gaussian(); data = data)
    @test fit.converged

    @test mf_aic(fit) ≈ aic(fit)
    @test mf_bic(fit) ≈ bic(fit)
    @test mf_coef(fit) ≈ coef(fit)
    @test mf_fitted(fit) ≈ fitted(fit)

    buf = IOBuffer()
    @test mf_summary(fit; io = buf) === fit
    s = String(take!(buf))
    @test occursin("logLik", s)
    @test occursin("AIC", s)
end