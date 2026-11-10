using DRModels, Test
import DelimitedFiles

# G6d: Binomial independent terms, `sd(g, coef(intercept)) ~ z` only (cell `binomial|indep|q2|sd-first`).
# The likelihood is a monotone ridge in the intercept-SD coefficients (log SD = b0 + b1 z; the intercept SD is
# zero except in the group with the largest z). drmTMB 0.8.1 stops at b0 = -54.2, b1 = 56.9 (log SD of the
# groups reaches -111); Julia used to stop at the |log SD| <= 30 wall of the Laplace objective (logLik -253.9952,
# converged = false). With the wall at 300 Julia reaches drmTMB's optimum from its default start.
# Reference: drmTMB 0.8.1 ML on fixtures/binomial_indep_restarts/g12_binomial.csv, logLik -253.551309266,
# is_converged() = FALSE (flat ridge), standard errors from sdreport.

const _G6D_DIR = joinpath(@__DIR__, "fixtures", "binomial_indep_restarts")

function _g6d_data()
    m, h = DelimitedFiles.readdlm(joinpath(_G6D_DIR, "g12_binomial.csv"), ',', Any; header = true)
    h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
    return (; y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")),
            x3 = Float64.(col("x3")), z = Float64.(col("z")), g = String.(col("g")))
end

@testset "G6d: independent-term Laplace objective follows the sd-z ridge past |log SD| = 30" begin
    d = _g6d_data()
    fit = drm(bf(@formula(y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g)),
                 @formula(sd(g, coef(intercept)) ~ z)), Binomial(:logit); data = d, se = false)
    @test fit.converged
    @test abs(loglikelihood(fit) - (-253.551309266)) <= 1e-4
    ref_beta = [0.4726729146, 0.2117444872, 0.2369921639, -0.2725294906]
    ref_se = [0.1113261125, 0.1293936525, 0.1019957937, 0.1009295245]
    @test all(abs.(fit.theta[1:4] .- ref_beta) .<= 1e-3 .* ref_se)
    @test abs(fit.theta[7] - (-0.9965199535)) <= 5e-3        # log SD of the x1 slope term
    # the ridge coefficients are not identified individually: both are far beyond the old wall
    @test fit.theta[5] < -40 && fit.theta[6] > 40
    # the guard itself is still there for absurd values
    th = copy(fit.theta); th[5] = -1000.0
    @test fit.nll(th) >= 1e17
end
