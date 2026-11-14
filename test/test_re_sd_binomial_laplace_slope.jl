# `re_sd` and `vc` on a Binomial correlated block `(1 + x | g)`.
#
# The `marginal = :Laplace` fit and every q >= 3 block store log SDs followed by atanh
# partial correlations in `:recov`. `re_sd` and `vc` used to read those numbers as a
# log-Cholesky triple, so the slope SD and the covariance came out wrong.
#
# Data: test/data/re_sd_laplace_slope/data.csv (300 rows, 30 groups, 4 trials per row).
# R reference (drmTMB 0.8.1, same file, max |gradient| 7e-10):
#   cbind(s, fl) ~ x1 + x2 + (1 + x1 | g), binomial(): logLik -411.139118754,
#   log SD = -0.3545015622, -0.1683012742, atanh(rho) = 0.2488671949.
using Test, DRModels, DelimitedFiles, LinearAlgebra

@testset "Binomial (1 + x | g): re_sd and vc decode the stored parameters" begin
    raw, hdr = readdlm(joinpath(@__DIR__, "data", "re_sd_laplace_slope", "data.csv"), ',';
                       header = true)
    col(nm) = raw[:, findfirst(==(nm), vec(hdr))]
    d = (; g = String.(col("g")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")),
         s = Float64.(col("s")), fl = Float64.(col("fl")))
    fam = DRModels.Binomial(:logit)
    r_logsd = [-0.3545015622, -0.1683012742]
    r_eta = 0.2488671949

    @testset "q = 2, marginal = :Laplace" begin
        fit = drm(bf(@formula(cbind(s, fl) ~ x1 + x2 + (1 + x1 | g))), fam;
                  data = d, marginal = :Laplace)
        @test fit.converged
        th = fit.theta
        # hand computation from the stored parameters (log SD, log SD, atanh rho)
        sd1, sd2, rho = exp(th[4]), exp(th[5]), tanh(th[6])
        sd = re_sd(fit)
        @test sd[:g_intercept] ≈ sd1 rtol = 1e-12
        @test sd[:g_slope] ≈ sd2 rtol = 1e-12
        V = vc(fit)[:g]
        @test V ≈ [sd1^2 rho*sd1*sd2; rho*sd1*sd2 sd2^2] rtol = 1e-12
        @test sqrt.(diag(V)) ≈ [sd[:g_intercept], sd[:g_slope]] rtol = 1e-12
        # against R, log SD scale (5e-3), correlation on the atanh scale
        @test abs(log(sd[:g_intercept]) - r_logsd[1]) < 5e-3
        @test abs(log(sd[:g_slope]) - r_logsd[2]) < 5e-3
        @test abs(atanh(V[1, 2] / sqrt(V[1, 1] * V[2, 2])) - r_eta) < 5e-3
    end

    @testset "q = 2, default route is unchanged (log-Cholesky)" begin
        fit = drm(bf(@formula(cbind(s, fl) ~ x1 + x2 + (1 + x1 | g))), fam; data = d)
        a, b, c = fit.theta[4:6]
        @test re_sd(fit)[:g_intercept] ≈ exp(a) rtol = 1e-12
        @test re_sd(fit)[:g_slope] ≈ sqrt(c^2 + exp(b)^2) rtol = 1e-12
        @test vc(fit)[:g] ≈ [exp(a)^2 c*exp(a); c*exp(a) c^2 + exp(b)^2] rtol = 1e-12
    end

    @testset "q = 3 block ($m)" for m in (:Laplace, :LA)
        fit = drm(bf(@formula(cbind(s, fl) ~ x1 + x2 + (1 + x1 + x2 | g))), fam;
                  data = d, marginal = m)
        th = fit.theta
        sds = exp.(th[4:6])
        V = vc(fit)[:g]
        # variances from the stored log SDs (a third partial correlation can sit on the
        # boundary, so only the first two correlations are pinned by hand)
        @test diag(V) ≈ sds .^ 2 rtol = 1e-12
        @test V[2, 1] / (sds[1] * sds[2]) ≈ tanh(th[7]) rtol = 1e-10
        @test V[3, 1] / (sds[1] * sds[3]) ≈ tanh(th[8]) rtol = 1e-10
        # all three correlations against the fit's own pairwise correlations
        @test [V[2, 1] / (sds[1] * sds[2]), V[3, 1] / (sds[1] * sds[3]),
               V[3, 2] / (sds[2] * sds[3])] ≈ fit.scales[:corpars] rtol = 1e-10
        sd = re_sd(fit)
        @test sd[:g_intercept] ≈ sds[1] rtol = 1e-12
        @test sd[:g_x1] ≈ sds[2] rtol = 1e-12
        @test sd[:g_x2] ≈ sds[3] rtol = 1e-12
    end
end
