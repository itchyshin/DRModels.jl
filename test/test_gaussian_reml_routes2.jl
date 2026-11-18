# Gaussian REML, second campaign: correlated blocks of 3-4 coefficients, a correlated block beside a crossed
# intercept, and relmat / animal / spatial intercepts with an sd() formula. Each shape is compared with
# R drmTMB 0.8.1 `REML = TRUE` on three committed simulated datasets (fixtures/gaussian_reml_routes2/, made by
# gen_ref.R). Thresholds are fixed in docs/dev-log/2026-11-09-gaussian-reml-routes2.md and in compare.jl.
using DRModels, Test, Statistics

include(joinpath(@__DIR__, "fixtures", "gaussian_reml_routes2", "compare.jl"))

@testset "Gaussian REML routes (second campaign) agree with R drmTMB 0.8.1" begin
    T = REML2_THRESH
    skipped = String[]
    for (grp, shapes) in REML2_GROUPS
        @testset "$grp" begin
            for shape in shapes, seed in (21, 22, 23)
                r = reml2_compare(shape, seed)
                if !r.r_converged
                    # rule 6: R itself does not converge on this dataset, so there is nothing to match
                    push!(skipped, "$shape/$seed")
                    continue
                end
                @test r.estim === :REML
                @test r.converged
                @test r.ll_diff <= T.loglik
                @test r.mu_se_units <= T.mu_se_units
                r.boundary || @test r.sd_rel <= T.sd_rel
                r.boundary || @test r.cor_abs <= T.cor_abs
            end
        end
    end
    @test skipped == ["ord4/22"]
end

@testset "REML on the new shapes: labelled equals unlabelled, REML differs from ML, refusals are accurate" begin
    plain, data = reml2_fit("ord3", 21)
    lab, _ = reml2_fit("ord3lab", 21)
    @test plain.loglik ≈ lab.loglik atol = 1e-8
    @test reml_loglik(plain) == plain.loglik
    @test ml_loglik(plain) != reml_loglik(plain)          # not an ML fit under a REML label

    msg(f; kw...) = try
        drm(f, Gaussian(); data = data, method = :REML, kw...); ""
    catch e
        sprint(showerror, e)
    end
    blk = DRModels.@formula(y ~ x1 + x2 + x3 + (1 + x1 + x2 | g))
    s1 = DRModels.@formula(sigma ~ 1)
    # an sd() formula on a correlated block: refused, naming drmTMB
    m1 = msg(bf(blk, s1, DRModels.@formula(sd(g, coef(x1)) ~ z)))
    @test occursin("together with `sd()` formulas", m1) && occursin("drmTMB refuses", m1)
    # a sigma predictor with the block: refused with its own reason
    m2 = msg(bf(blk, DRModels.@formula(sigma ~ x1)))
    @test occursin("constant residual scale", m2)
    # a block beside a random slope term
    m3 = msg(bf(DRModels.@formula(y ~ x1 + (1 + x1 | g) + (0 + x2 | g2)), s1))
    @test occursin("method = :REML is not implemented", m3) && occursin("INTERCEPTS", m3)
    # the bare spatial intercept: drmTMB's fixed median-distance kernel, so REML is admitted and is the
    # same model as the `sd(g, spatial) ~ 1` spelling
    sp, spdata = reml2_fit("spatial_sd1", 21)
    coords = let m = readdlm(joinpath(dirname(@__FILE__), "fixtures", "gaussian_reml_routes2", "coords.csv"), ',', Any; skipstart = 1)
        Dict(String(m[i, 1]) => [Float64(m[i, 2]), Float64(m[i, 3])] for i in 1:size(m, 1))
    end
    bare = drm(bf(DRModels.@formula(y ~ x1 + x2 + x3 + spatial(1 | g)), s1), Gaussian(); data = spdata, coords = coords,
               method = :REML)
    @test bare.converged
    @test bare.loglik ≈ sp.loglik atol = 1e-6
    # a labelled structured q2 block: refused by the structured-term check, which runs before the
    # method is read, so this is the same refusal under ML and does not exercise a REML gate
    m5 = try
        drm(bf(DRModels.@formula(y ~ x1 + x2 + x3 + animal(1 + x1 | p | g)), s1), Gaussian(); data = spdata,
            A = Matrix{Float64}(readdlm(joinpath(dirname(@__FILE__), "fixtures", "gaussian_reml_routes2", "A.csv"), ',', Float64)),
            ids = String.(vec(readdlm(joinpath(dirname(@__FILE__), "fixtures", "gaussian_reml_routes2", "A_ids.csv"), ',', Any; skipstart = 1))),
            method = :REML); ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("is not implemented on the univariate routes", m5)
end
