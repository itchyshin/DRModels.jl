# test_summary_zp_suppress.jl — issues #320 and #323.2.
#
# #320 SUPERSEDED 2026-09-06. z / p are now reported for EVERY coefficient with a
# finite SE, on the working scale shown in the block heading (log σ for dispersion,
# where a Wald test is symmetric and unbounded). The old blanket suppression was
# inconsistent: the μ intercept null is as unit-dependent as "σ = 1" and was always
# printed, and a slope on log σ is a log-RATIO of SDs — a real, unit-free null.
# What replaces the suppression is a stated null under each block heading.
#
# #323.2: a boundary / singular direction (Inf SE) reports NaN z / p in the table,
# not the misleading z = est/Inf = 0, p = 1 that reads as a confident null.
using DRModels, Test, Random
import Distributions

@testset "coeftable/show z-p suppression (#320, #323.2)" begin
    @testset "sigma block z/p blanked; mu kept (#320)" begin
        Random.seed!(1)
        n = 300
        x = randn(n)
        y = 0.5 .- 0.8 .* x .+ exp.(-0.3 .+ 0.4 .* x) .* randn(n)
        fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data = (; y, x))
        ct = coeftable(fit)
        z = ct.cols[3]; p = ct.cols[4]
        # Row order: mu:(Intercept), mu:x, sigma:(Intercept), sigma:x.
        @test isfinite(z[1]) && isfinite(z[2])          # mu z present
        @test isfinite(p[1]) && isfinite(p[2])          # mu p present
        # sigma z/p are now REPORTED (2026-09-06), on the log σ scale.
        @test isfinite(z[3]) && isfinite(z[4])
        @test isfinite(p[3]) && isfinite(p[4])
        @test z[3] ≈ ct.cols[1][3] / ct.cols[2][3]      # z really is est/se
        @test z[4] ≈ ct.cols[1][4] / ct.cols[2][4]
        @test all(0 .<= [p[3], p[4]] .<= 1)
        # SEs are still reported for the sigma block (only the test is suppressed).
        @test all(isfinite, ct.cols[2])
        # show() states the null under each block heading instead of hiding the test.
        s = sprint(show, MIME("text/plain"), fit)
        @test occursin("H0: coefficient = 0 on log σ", s)
        @test !occursin("NaN", s)
    end

    @testset "random-effect SD (:resd) z/p blanked (#320)" begin
        Random.seed!(3)
        G = 25; nper = 10; n = G * nper
        g = repeat(1:G, inner = nper); x = randn(n)
        b = 0.7 .* randn(G)
        y = 1.0 .+ 0.5 .* x .+ b[g] .+ 0.4 .* randn(n)
        fit = drm(bf(@formula(y ~ x + (1 | g))), Gaussian(); data = (; y, x, g))
        ct = coeftable(fit)
        # Find the :resd row(s) by row name prefix.
        resd_rows = findall(nm -> startswith(nm, "resd:"), ct.rownms)
        @test !isempty(resd_rows)
        for i in resd_rows
            @test isfinite(ct.cols[3][i])               # z now reported
            @test isfinite(ct.cols[4][i])               # p now reported
            @test isfinite(ct.cols[1][i])               # estimate present
        end
        # mu rows keep their z/p.
        mu_rows = findall(nm -> startswith(nm, "mu:"), ct.rownms)
        @test all(isfinite, ct.cols[3][mu_rows])
    end

    @testset "boundary Inf SE -> NaN z/p, not 0/1 (#323.2)" begin
        # Unit-level check of the shared z/p helper: a location block with an Inf SE
        # (singular / boundary direction) must report NaN z and NaN p, NOT z = 0,
        # p = 1 (which reads as a confident non-significant result).
        z, p = DRModels._wald_zp(:mu, 0.7, Inf)
        @test isnan(z) && isnan(p)
        # A finite-SE location coefficient still gets a real test.
        z2, p2 = DRModels._wald_zp(:mu, 2.0, 1.0)
        @test isfinite(z2) && z2 == 2.0
        @test isfinite(p2) && 0.0 <= p2 <= 1.0
        # SUPERSEDED 2026-09-06: these blocks are no longer blanked. With a finite
        # SE every block gets a real Wald test on its own working scale; the null is
        # stated under the heading instead of the test being withheld.
        for (blk, est, se) in ((:sigma, -0.4, 0.05), (:resd, 0.7, 0.2), (:recov, 0.1, 0.3))
            zb, pb = DRModels._wald_zp(blk, est, se)
            @test isfinite(zb) && zb ≈ est / se
            @test isfinite(pb) && 0.0 <= pb <= 1.0
        end
        # The suppression that REMAINS is the boundary one, and it is block-agnostic:
        # a non-finite SE is not a test, on any scale.
        for blk in (:sigma, :resd, :recov, :mu, :rho12)
            @test all(isnan, DRModels._wald_zp(blk, 0.5, Inf))
        end
        # :rho12 keeps a real test (rho12 = 0 is a meaningful null).
        zr, pr = DRModels._wald_zp(:rho12, 0.5, 0.1)
        @test isfinite(zr) && zr == 5.0
    end

    @testset "a real sigma slope is testable, and recovers the simulated ratio" begin
        # The case the book's location-scale chapter ships. Males were simulated
        # 1.6/0.6 = 2.67x as variable; the log σ slope must be a usable test, not NaN.
        Random.seed!(20260906)
        n = 400
        sex = rand(["female", "male"], n); x = randn(n)
        sd = [s == "male" ? 1.6 : 0.6 for s in sex]
        y = 1.0 .+ 2.0 .* x .+ sd .* randn(n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ sex)), Gaussian(); data = (; y, x, sex))
        ct = coeftable(fit)
        i = findfirst(nm -> startswith(nm, "sigma:") && !occursin("(Intercept)", nm), ct.rownms)
        @test i !== nothing
        @test isfinite(ct.cols[3][i]) && isfinite(ct.cols[4][i])   # RED CONTROL: NaN under the old rule
        @test ct.cols[4][i] < 1e-10                                # a 2.67x ratio is not ambiguous
        @test 2.0 < exp(ct.cols[1][i]) < 3.5                       # recovers the simulated ratio
    end
end
