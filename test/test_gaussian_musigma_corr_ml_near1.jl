# Issue #1366: Gaussian `(1 | p | g)` in mu and sigma, ML, true correlation near 1.
# At atanh(rho) ~ 8 the prior precision 1 / (1 - rho^2) is ~1e8, so the inner-mode gradient has a
# roundoff floor (~1e-9) above the absolute 1e-9 test of `_gsl_mode`. Newton sat at the exact mode, the
# mode solve reported failure and the outer objective returned the 1e18 sentinel, ending the fit with
# logLik = -1e18 although drmTMB 0.8.1 reaches logLik -440.4357 on the same data (320 rows, 80 groups
# of 4, fixtures/gaussian_musigma_corr_ml_near1/data.csv, copied from the issue reproducer).
# The reference value -440.4357 is the one quoted in the issue for drmTMB 0.8.1; the tolerance is 5e-3
# because the likelihood is flat in atanh(rho) here (drmTMB stops at 10.7, this route at about 8-9).
using DRModels, Test, DelimitedFiles, Logging

@testset "Gaussian (1|p|g) mu+sigma ML, correlation near 1 (#1366)" begin
    raw, hdr = readdlm(joinpath(@__DIR__, "fixtures", "gaussian_musigma_corr_ml_near1", "data.csv"),
                       ','; header = true)
    d = (; y = Float64.(raw[:, 1]), x1 = Float64.(raw[:, 2]), x2 = Float64.(raw[:, 3]),
         g = String.(raw[:, 4]))
    f = bf(@formula(y ~ x1 + x2 + (1 | p | g)), @formula(sigma ~ 1 + x1 + (1 | p | g)))
    fit = with_logger(NullLogger()) do
        drm(f, Gaussian(); data = d)
    end
    @test fit.loglik > -1e17                       # not the failed-objective sentinel
    @test isapprox(fit.loglik, -440.4357; atol = 5e-3)
    # fixed effects and both log SDs agree with drmTMB to the issue's four digits
    @test isapprox(fit.theta[1:5], [0.5725, 0.4546, -0.3349, -0.2487, 0.385]; atol = 5e-3)
    @test isapprox(fit.theta[6:7], [-0.5249, -1.5076]; atol = 5e-3)
    @test fit.theta[end] > 6                       # correlation at the edge (atanh scale)
end
