# Plain NB2 fixed effects at the Poisson limit (HSAUR3::bladdercancer).
# Refs #1072 — mu SEs must stay finite and match the Poisson twin; dispersion SE
# is not identified (Inf after vcov_guard, never NaN on the mean block).
using DRModels
using Test

function _bladdercancer_data()
    L = readlines(joinpath(@__DIR__, "fixtures", "bladdercancer_1072", "bladder.csv"))
    rows = [split(l, ",") for l in L[2:end]]
    return (;
        number = parse.(Float64, getindex.(rows, 1)),
        tumorsize = String.(getindex.(rows, 2)),
        time = parse.(Float64, getindex.(rows, 3)),
    )
end

@testset "#1072 bladdercancer: NB2 Poisson-limit mu SEs (refs #1072)" begin
    bl = _bladdercancer_data()
    forms = (@formula(number ~ 1 + tumorsize + time),)
    fit_nb = drm(bf(forms...), NegBinomial2(); data = bl)
    fit_po = drm(bf(forms...), Poisson(); data = bl)

    @test loglik(fit_nb) ≈ loglik(fit_po) atol = 1e-9
    @test coef(fit_nb, :mu) ≈ coef(fit_po, :mu) atol = 1e-9

    se_nb = stderror(fit_nb)
    se_po = stderror(fit_po)
    @test length(se_nb) == 4
    @test all(isfinite, se_nb[1:3])
    @test !any(isnan, se_nb[1:3])
    @test se_nb[1:3] ≈ se_po atol = 1e-3
    @test !isfinite(se_nb[4])   # log-dispersion at the Poisson limit
    @test !any(isnan, fit_nb.vcov[1:3, 1:3])
end