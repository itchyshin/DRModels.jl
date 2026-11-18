# Gaussian shapes closed by docs/dev-log/2026-11-09-gaussian-sd-groupf.md, checked against drmTMB 0.8.1
# at the optimum on three committed datasets (reference: test/data/gaussian_sd_groupf/r_reference.tsv,
# made by r_reference.R). Thresholds are the pre-registered ones of that note:
# |dlogLik| <= 1e-4, fixed effects and SD-formula coefficients within 0.01 R standard errors,
# natural-scale SDs within 0.05 SE on the log scale, correlations within 0.01.
# The full 30-fit table is in docs/dev-log/2026-11-09-gaussian-sd-groupf-compare.tsv
# (regenerate with tools/gaussian_sd_groupf_compare.jl); this file runs a representative subset.
using DRModels, Test
include("gaussian_sd_groupf_helpers.jl")

@testset "Gaussian Group F shapes agree with drmTMB 0.8.1" begin
    ref = gsf_reference()
    for ds in ("gsf_s1", "gsf_s2", "gsf_s3")
        d = gsf_read(ds)
        for id in ("g_blk_slope_sd", "g_two_blk_sd", "g_blk_int_slope", "g_blk_indep",
                   "f_corr", "f_indep", "f_cross")
            @testset "$ds $id" begin
                fit = drm(gsf_build(id), Gaussian(); data = d)
                c = gsd_compare(gsf_extract(fit, id), ref[(ds, id)])
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

@testset "Gaussian Group F shapes: refusals stay accurate" begin
    d = gsf_read("gsf_s1")
    f(re; extra = (), sigma = "sigma ~ 1", resp = "y ~ x1 + x2 + x3") =
        bf(gsd_formula(resp * " + " * re), gsd_formula(sigma), extra...)
    msg(fm; kw...) = try
        drm(fm, Gaussian(); data = d, kw...); ""
    catch e
        sprint(showerror, e)
    end
    # REML: ML only on every new shape (the generic gate or the route's own message, both name REML)
    reml_refused(m) = occursin("REML", m) &&
        (occursin("not implemented", m) || occursin("maximum likelihood only", m))
    @test reml_refused(msg(f("(1 + x1 | g) + (0 + x2 | g2)"); method = :REML))
    @test reml_refused(msg(f("(1 + x1 | g) + (1 + x2 | g2)"); method = :REML))
    @test reml_refused(msg(f("(1 | g)"; resp = "yf ~ x1", sigma = "sigma ~ 1 + (1 | g)",
                             extra = (gsd_formula("sd(g) ~ z"),)); method = :REML))
    # drmTMB refuses an sd() formula on a grouping that carries a block and another term
    @test occursin("refuses it",
                   msg(f("(1 + x1 | g) + (0 + x3 | g)"; extra = (gsd_formula("sd(g, coef(x3)) ~ z"),))))
    @test occursin("refuses it",
                   msg(f("(1 + x1 | g) + (0 + x3 | g)"; extra = (gsd_formula("sd(g, coef(x1)) ~ z"),))))
    # a 3- or 4-coefficient block beside a slope term: drmTMB's own fit does not converge
    @test occursin("did not converge", msg(f("(1 + x1 + x2 | g) + (0 + x3 | g2)")))
    # two blocks: 2-coefficient blocks only; never two on one grouping
    @test occursin("2-coefficient blocks", msg(f("(1 + x1 + x2 | g) + (1 + x3 | g2)")))
    @test occursin("two correlated blocks", msg(f("(1 + x1 | g) + (1 + x2 | g)")))
    # a coefficient in the block and also a separate term
    @test occursin("overlapping", msg(f("(1 + x1 | g) + (1 | g)")))
    @test occursin("overlapping", msg(f("(1 + x1 | g) + (0 + x1 | g)")))
    # an SD formula naming no term
    @test occursin("names no random-effect term",
                   msg(f("(1 + x1 | g) + (0 + x2 | g2)"; extra = (gsd_formula("sd(g2, coef(x3)) ~ z2"),))))
    # a label shared by two terms on one grouping: R ignores the label; the message says so
    @test occursin("drmTMB ignores such a label", msg(f("(1 | p | g) + (0 + x1 | p | g)")))
    # mean SD formula with a sigma random slope, or with different labels, stays refused
    @test occursin("random intercepts",
                   msg(f("(1 | g)"; resp = "yf ~ x1", sigma = "sigma ~ 1 + (0 + x1 | g)",
                         extra = (gsd_formula("sd(g) ~ z"),))))
    @test occursin("same label",
                   msg(f("(1 | p | g)"; resp = "yf ~ x1", sigma = "sigma ~ 1 + (1 | g)",
                         extra = (gsd_formula("sd(g) ~ z"),))))
    @test occursin("one grouping factor",
                   msg(f("(1 | p | g)"; resp = "yf ~ x1", sigma = "sigma ~ 1 + (1 | p | g2)",
                         extra = (gsd_formula("sd(g) ~ z"),))))
end
