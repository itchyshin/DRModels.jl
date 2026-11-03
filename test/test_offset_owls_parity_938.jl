# test_offset_owls_parity_938.jl: #938 Owls parity row (owner decision, 25 Oct 2026).
#
# NegBinomial2 with `offset(lbs)` (log brood size, coefficient fixed at 1) on
# glmmTMB::Owls, against drmTMB main 8aafcc2c (0.7.1), run by
# test/parity/gen_owls_offset_938.R. The Owls data ship with glmmTMB (AGPL) and
# are not vendored: the generator writes the prepared CSV outside the repository
# and this file reads it from DRM_OWLS_CSV. Without that variable it is skipped.
#
#   Rscript test/parity/gen_owls_offset_938.R /tmp/owls_938.csv
#   DRM_OWLS_CSV=/tmp/owls_938.csv julia --project=test -e 'include("test/test_offset_owls_parity_938.jl")'

using DRModels
using Test
using DelimitedFiles: readdlm

function _owls938(path)
    m, h = readdlm(path, ','; header = true)
    cols = Dict(Symbol(strip(string(c))) => m[:, j] for (j, c) in enumerate(vec(h)))
    return (; NCalls = Float64.(cols[:NCalls]), arr = Float64.(cols[:arr]),
              lbs = Float64.(cols[:lbs]), Food = String.(string.(cols[:Food])),
              Sex = String.(string.(cols[:Sex])), Nest = String.(string.(cols[:Nest])))
end

# drmTMB coefficient name -> DRModels.jl coefficient name
const _OWLS_NAMES_938 = Dict(
    "(Intercept)" => "(Intercept)", "FoodSatiated" => "Food: Satiated",
    "SexMale" => "Sex: Male", "arr" => "arr",
    "FoodSatiated:SexMale" => "Food: Satiated & Sex: Male")

function _owls_coef938(fit, p, rnames)
    nm = Dict(fit.coefnames)[p]
    cf = coef(fit, p)
    return [cf[findfirst(==(_OWLS_NAMES_938[r]), nm)] for r in rnames]
end

@testset "#938 Owls parity: NegBinomial2 + offset(lbs) vs drmTMB 8aafcc2c" begin
    path = get(ENV, "DRM_OWLS_CSV", "")
    if isempty(path)
        @info "DRM_OWLS_CSV not set: the Owls offset parity row (#938) is skipped"
        @test_skip false
    else
        d = _owls938(path)
        @test length(d.NCalls) == 599

        # O1: fixed effects, sigma ~ 1 (the issue's first repro)
        o1 = drm(bf(@formula(NCalls ~ Food + offset(lbs)), @formula(sigma ~ 1)),
                 NegBinomial2(); data = d)
        @test o1.converged
        @test loglik(o1) ≈ -1748.96054442 atol = 1e-5
        @test _owls_coef938(o1, :mu, ["(Intercept)", "FoodSatiated"]) ≈
              [0.6526296939, -0.5199636850] atol = 1e-4
        @test coef(o1, :sigma) ≈ [0.1354107930] atol = 1e-4

        # L2: fixed effects, sigma ~ Food
        l2 = drm(bf(@formula(NCalls ~ Food * Sex + arr + offset(lbs)), @formula(sigma ~ Food)),
                 NegBinomial2(); data = d)
        @test l2.converged
        @test loglik(l2) ≈ -1698.79305625 atol = 1e-5
        @test _owls_coef938(l2, :mu, ["(Intercept)", "FoodSatiated", "SexMale", "arr",
                                      "FoodSatiated:SexMale"]) ≈
              [0.68132559130, -0.60857459242, 0.06500446698, -0.12963866612, 0.16544189781] atol = 1e-4
        @test _owls_coef938(l2, :sigma, ["(Intercept)", "FoodSatiated"]) ≈
              [-0.22785821474, 0.66585134254] atol = 1e-4

        # L1: one (1 | Nest), sigma ~ 1, TMB-convention Laplace (drmTMB's integrator)
        l1 = drm(bf(@formula(NCalls ~ Food * Sex + arr + offset(lbs) + (1 | Nest)), @formula(sigma ~ 1)),
                 NegBinomial2(); data = d, marginal = :Laplace)
        @test l1.converged
        @test loglik(l1) ≈ -1731.76688652 atol = 1e-5
        @test _owls_coef938(l1, :mu, ["(Intercept)", "FoodSatiated", "SexMale", "arr",
                                      "FoodSatiated:SexMale"]) ≈
              [0.76055680127, -0.76772540058, -0.01031087744, -0.11540804499, 0.17447045487] atol = 1e-4
        @test coef(l1, :sigma) ≈ [0.06143494307] atol = 1e-4
        @test exp(coef(l1, :resd)[1]) ≈ 0.3302286007 atol = 1e-4
        # the default :LA (GHQ) integrates the same model more exactly: close, not equal
        l1g = drm(bf(@formula(NCalls ~ Food * Sex + arr + offset(lbs) + (1 | Nest)), @formula(sigma ~ 1)),
                  NegBinomial2(); data = d)
        @test l1g.converged
        @test length(coef(l1g, :mu)) == 5
        @test abs(loglik(l1g) - loglik(l1)) < 0.5
    end
end
