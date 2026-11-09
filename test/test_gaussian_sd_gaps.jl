# Gaussian ordinary and crossed shapes closed by docs/dev-log/2026-11-09-gaussian-sd-gaps.md, checked
# against drmTMB 0.8.1 at the optimum on three committed datasets (reference: test/data/gaussian_sd_gaps/
# r_reference.tsv, made by r_reference.R). Thresholds are the pre-registered ones of that note:
# |dlogLik| <= 1e-4, fixed effects and SD-formula coefficients within 0.01 R standard errors,
# natural-scale SDs within 0.05 SE on the log scale, correlations within 0.01.
# The full 45-fit table is in docs/dev-log/2026-11-09-gaussian-sd-gaps-compare.tsv
# (regenerate with tools/gaussian_sd_gaps_compare.jl); this file runs a representative subset.
using DRModels, Test
include("gaussian_sd_gaps_helpers.jl")

@testset "Gaussian SD gaps agree with drmTMB 0.8.1" begin
    ref = gsd_reference()
    for ds in ("gsd_s1", "gsd_s2", "gsd_s3")
        d = gsd_read(ds)
        for id in ("ord_q3", "crs_q2_lab", "crs_q2_first", "ind_q3_z")
            @testset "$ds $id" begin
                fit = drm(gsd_build(id), Gaussian(); data = d)
                c = gsd_compare(gsd_extract(fit, id), ref[(ds, id)])
                @test fit.converged
                @test c.dll ≤ 1e-4
                @test c.fe ≤ 0.01
                @test c.sdf ≤ 0.01
                @test c.sdlog ≤ 0.05
                @test c.cor ≤ 0.01
                @test isempty(c.unmatched)
            end
        end
    end
end

@testset "Gaussian SD gap shapes: refusals stay accurate" begin
    d = gsd_read("gsd_s1")
    f(re; extra = (), sigma = "sigma ~ 1") =
        bf(gsd_formula("y ~ x1 + " * re), gsd_formula(sigma), extra...)
    msg(fm; kw...) = try
        drm(fm, Gaussian(); data = d, kw...); ""
    catch e
        sprint(showerror, e)
    end
    # REML: the sd()-free correlated shapes now fit (docs/dev-log/2026-11-09-gaussian-reml-routes2.md);
    # a correlated block with an sd() formula, or beside a slope term, still refuses and says so
    @test msg(f("(1 + x1 | g) + (1 | g2)"); method = :REML) == ""
    @test msg(f("(1 + x2 + x3 | g)"); method = :REML) == ""
    @test occursin("together with `sd()` formulas",
                   msg(f("(1 + x1 | g) + (1 | g2)"; extra = (gsd_formula("sd(g2) ~ z2"),)); method = :REML))
    @test occursin("method = :REML is not implemented", msg(f("(1 + x2 | g) + (0 + x3 | g2)"); method = :REML))
    @test occursin("maximum likelihood only",
                   msg(f("(1 | g) + (0 + x2 | g)"; extra = (gsd_formula("sd(g, coef(x2)) ~ z"),)); method = :REML))
    # a label shared by two terms ties their correlations: refused by name, not dropped
    @test occursin("shared by 2", msg(f("(1 + x2 | p | g) + (1 + x3 | p | g2)")))
    # a correlated block beside a slope term
    @test occursin("only be combined with random intercepts",
                   msg(f("(1 + x2 | g) + (0 + x3 | g2)")))
    # an SD formula naming a coefficient that is not a term
    @test occursin("not a random-effect term",
                   msg(f("(1 | g) + (0 + x2 | g)"; extra = (gsd_formula("sd(g, coef(x3)) ~ z"),))))
    # mean SD formula together with a sigma random effect: the message names what R does
    @test occursin("R drmTMB fits",
                   msg(f("(1 | g)"; sigma = "sigma ~ 1 + (1 | g)", extra = (gsd_formula("sd(g) ~ z"),))))
end
