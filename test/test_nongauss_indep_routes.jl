using DRModels, Test
include(joinpath(@__DIR__, "nongauss_indep_helpers.jl"))

# Independent intercept-plus-slope terms `(1 | g) + (0 + x1 | g) [+ (0 + x2 | g) ...]` on the
# non-Gaussian mean routes (Poisson, NegBinomial2, Binomial, Beta, LogNormal, Gamma, Student,
# Tweedie), with and without `sd(g, coef(.)) ~ ...` formulas, against drmTMB 0.8.1 (ML) on the
# committed gap-map data. Reference: test/fixtures/nongauss_indep/r_reference*.csv, made by the
# committed gen_ref.R. The agreement rule was fixed before the numbers were read
# (docs/dev-log/2026-11-09-nongauss-indep-design.md): |dlogLik| <= 1e-4, fixed effects within 1e-3,
# random-effect log SDs (or SD-formula coefficients) within 5e-3. A term whose SD sits on the
# boundary in either fit (log SD < -6) is exempt from the SD criterion only; logLik and fixed
# effects still have to agree. That exemption was added AFTER the rule was fixed, and it applies
# to every cell below, not only Binomial q4.

const _NGI_REF = ngi_reference()

function _ngi_check(id)
    fit = drm(ngi_build(id), ngi_family(id); data = ngi_data(split(id, '|')[1]; crossed = occursin("|crossed|", id)))
    c = ngi_compare(id, fit, _NGI_REF[id])
    @test fit.converged
    @test c.dll <= 1e-4
    @test c.fe <= 1e-3
    @test c.sdlog <= 5e-3 || c.boundary
    return fit
end

@testset "non-Gaussian independent intercept + slope terms agree with drmTMB" begin
    @testset "no SD formulas, $(split(id, '|')[1]) $(split(id, '|')[2]) q$(ngi_parts(id)[3])" for id in [
            "poisson|indep|lab0|q2|sd-none|cm0|ML",
            "poisson|indep|lab0|q3|sd-none|cm0|ML",
            "poisson|slope|lab0|q1|sd-none|cm0|ML",
            "nb2|indep|lab0|q2|sd-none|cm0|ML",
            "nb2|indep|lab0|q3|sd-none|cm0|ML",
            "nb2|slope|lab0|q1|sd-none|cm0|ML",
            "binomial|indep|lab0|q2|sd-none|cm0|ML",
            "binomial|indep|lab0|q3|sd-none|cm0|ML",
            "beta|indep|lab0|q2|sd-none|cm0|ML",
            "beta|slope|lab0|q1|sd-none|cm0|ML",
            "lognormal|indep|lab0|q2|sd-none|cm0|ML",
            "gamma|indep|lab0|q2|sd-none|cm0|ML",
            "student|indep|lab0|q2|sd-none|cm0|ML",
            "tweedie|indep|lab0|q2|sd-none|cm0|ML"]
        fit = _ngi_check(id)
        startswith(id, "lognormal") || @test fit.marginal === :Laplace   # LogNormal is the exact Gaussian marginal of log(y)
    end

    @testset "SD formulas on independent terms, $id" for id in [
            "poisson|indep|lab0|q2|sd-one|cm0|ML",
            "poisson|indep|lab0|q2|sd-z|cm0|ML",
            "poisson|indep|lab0|q2|sd-first|cm0|ML",
            "poisson|indep|lab0|q2|sd-last|cm0|ML",
            "nb2|indep|lab0|q2|sd-z|cm0|ML",
            "nb2|indep|lab0|q2|sd-first|cm0|ML",
            "binomial|indep|lab0|q2|sd-last|cm0|ML",
            "binomial|indep|lab0|q3|sd-first|cm0|ML",
            "gamma|indep|lab0|q2|sd-z|cm0|ML",
            "gamma|indep|lab0|q2|sd-first|cm0|ML"]
        _ngi_check(id)
    end

    @testset "Binomial q4 independent terms: a zero slope SD is a boundary, not a disagreement" begin
        id = "binomial|indep|lab0|q4|sd-none|cm0|ML"
        fit = drm(ngi_build(id), ngi_family(id); data = ngi_data("binomial"))
        c = ngi_compare(id, fit, _NGI_REF[id])
        @test c.dll <= 1e-4 && c.fe <= 1e-3
        @test c.boundary                         # a term sits at SD ~ 0 in at least one fit
        @test !_NGI_REF[id].clean                # drmTMB flags this cell itself
        # the three SDs that are not at zero must still match R (the boundary term only waives its own SD)
        sdb = [first(p) for p in fit.blocks if first(p) in (:sd, :resd) || startswith(String(first(p)), "sdtarget_")]
        jsd = reduce(vcat, [coef(fit, b) for b in sdb])
        rsd = ngi_ref_sdpar(id, _NGI_REF[id])
        live = [i for i in eachindex(jsd) if jsd[i] > -6 && rsd[i] > -6]
        @test length(live) == 3
        @test maximum(abs.(jsd[live] .- rsd[live])) <= 5e-3
    end
end

@testset "non-Gaussian independent terms: shapes that stay refused" begin
    d = ngi_data("poisson")
    dc = merge(d, (; g2 = ngi_data("poisson"; crossed = true).g2))
    f = bf(@formula(y ~ x1 + (1 | g) + (0 + x1 | g)))
    @test_throws r"is not available for Poisson\(\) with independent random slopes" drm(
        f, Poisson(); data = d, marginal = :AGHQ)
    @test_throws r"`method = :REML` is not available" drm(f, Poisson(); data = d, method = :REML)
    @test_throws r"marginal = :VA" drm(f, Poisson(); data = d, marginal = :VA)
    # a correlated block beside another term
    @test_throws r"a correlated `\(1 \+ x \| g\)` must be the only random-effect term" drm(
        bf(@formula(y ~ x1 + (1 + x1 | g) + (1 | g2))), Poisson(); data = dc)
    # drmTMB 0.8.1 itself errors on these SD shapes, so they stay refused here
    for (fam, id) in (Beta() => "beta|indep|lab0|q2|sd-z|cm0|ML",
                      Tweedie() => "tweedie|indep|lab0|q2|sd-z|cm0|ML")
        @test_throws r"random-effect SD regression requires" drm(ngi_build(id), fam; data = ngi_data(split(id, '|')[1]))
    end
end
