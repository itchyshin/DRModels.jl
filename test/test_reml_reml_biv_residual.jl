# test_reml_reml_biv_residual.jl -- #624 / drmTMB #1142: REML on the
# residual-only bivariate Gaussian route
#   bf(mu1 = y1 ~ x, mu2 = y2 ~ x, sigma1 = ~1, sigma2 = ~1, rho12 = ~1)
# which this package refused outright until now ("method = :REML needs random
# effects to restrict"), while native drmTMB/TMB fitted it by handing
# beta_mu1/beta_mu2 to its Laplace approximation.
#
# What is asserted, in the order the leaf ledger asks for it:
#   (i)   REML fits and converges;
#   (ii)  the restricted log-likelihood differs from the ML one;
#   (iii) ML and REML point estimates differ in the way theory predicts FOR THIS
#         cell -- see the derivation below;
#   (iv)  estimation_method(fit) === :REML;
#   plus  the normalisation convention (sign of the log-determinant AND the
#         +(p_beta/2)*log(2*pi) constant), the REML degrees of freedom visible in
#         the scale standard error, and that the ML route is untouched.
using DRModels
using Test, Random, LinearAlgebra, Statistics

@testset "REML on the residual-only bivariate Gaussian route (#624)" begin
    Random.seed!(20260905)
    n = 80
    x = randn(n)
    y1 = 0.4 .+ 0.6 .* x .+ 0.5 .* randn(n)
    y2 = -0.2 .+ 0.3 .* x .+ 0.4 .* y1 .+ 0.5 .* randn(n)
    dat = (; y1, y2, x)
    fform = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
               sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
               rho12 = @formula(rho12 ~ 1))

    fml = drm(fform, Gaussian(); data = dat, method = :ML)
    fre = drm(fform, Gaussian(); data = dat, method = :REML)

    # (i) it fits.
    @test fre.converged
    @test all(isfinite, fre.theta)
    @test isfinite(DRModels.loglik(fre))

    # (iv) and it says so.
    @test estimation_method(fre) === :REML
    @test estimation_method(fml) === :ML
    @test DRModels.loglik(fre) == reml_loglik(fre)      # public loglik IS the restricted one
    @test isnan(reml_loglik(fml))
    @test ml_loglik(fml) == DRModels.loglik(fml)

    # (ii) the two objectives are genuinely different numbers, not one value
    # wearing two labels. The restricted log-likelihood is BELOW the ML maximum
    # (it pays the -0.5*logdet(X' V^-1 X) restriction).
    @test DRModels.loglik(fre) != DRModels.loglik(fml)
    @test DRModels.loglik(fre) < DRModels.loglik(fml)
    @test ml_loglik(fre) != reml_loglik(fre)
    # ml_loglik(fre) is the PLAIN likelihood at the REML estimate, so it cannot
    # beat the ML maximum.
    @test ml_loglik(fre) <= DRModels.loglik(fml) + 1e-8

    # ---- (iii) how the estimates differ, and why ------------------------------
    # mu1 and mu2 share ONE design matrix here, so the seemingly-unrelated
    # regressions have identical regressors: Zellner's result makes the GLS
    # profile of (beta_mu1, beta_mu2) collapse to per-equation OLS, independent
    # of the residual covariance. REML therefore moves NO mean coefficient.
    b_ml = fml.theta[1:4]
    b_re = fre.theta[1:4]
    @test maximum(abs.(b_ml .- b_re)) < 1e-6

    # What REML DOES move is the scale: with p_mu = 2 mean coefficients per
    # response and an intercept-only log-sigma, the restricted estimator uses the
    # (n - p_mu) divisor instead of n, i.e. each log-sigma is shifted UP by
    # exactly 0.5*log(n / (n - p_mu)). This is the defining REML correction --
    # the variance components stop being downward-biased.
    p_mu = 2
    shift = 0.5 * log(n / (n - p_mu))
    @test isapprox(fre.theta[5] - fml.theta[5], shift; atol = 1e-5)
    @test isapprox(fre.theta[6] - fml.theta[6], shift; atol = 1e-5)
    @test fre.theta[5] > fml.theta[5]
    @test fre.theta[6] > fml.theta[6]
    # rho12 is scale-free, and both sigmas move by the same factor, so the
    # residual correlation is unchanged.
    @test isapprox(fre.theta[7], fml.theta[7]; atol = 1e-5)

    # ---- the objective's normalisation, checked independently ----------------
    # l_R = l_ML(beta_hat, phi) - 0.5*logdet(H) + (p_beta/2)*log(2*pi), with
    # H = S^-1 (x) X'X for this shared-design cell. Rebuilding H by hand here
    # pins BOTH the sign of the log-determinant and the presence of the
    # (p_beta/2)*log(2*pi) constant -- the #477 normalised Patterson-Thompson
    # convention TMB/lme4/glmmTMB also report, so `loglik(fit)` is directly
    # comparable to drmTMB's `logLik()` with no leftover constant.
    s1 = exp(fre.theta[5]); s2 = exp(fre.theta[6])
    rho = DRModels.RHO_GUARD * tanh(fre.theta[7])
    S = [s1^2 rho*s1*s2; rho*s1*s2 s2^2]
    X = hcat(ones(n), x)
    H = kron(inv(S), transpose(X) * X)
    p_beta = 4
    @test isapprox(reml_loglik(fre),
                   ml_loglik(fre) - 0.5 * logdet(H) + 0.5 * p_beta * log(2pi);
                   atol = 1e-8)

    # ---- REML degrees of freedom show up in the scale standard error ---------
    # For an intercept-only log-sigma the restricted information is 2*(n - p_mu),
    # not 2*n: SE(log sigma) = 1/sqrt(2*(n - p_mu)). This is the nobs/df side of
    # the same correction, read off the reported vcov rather than the objective.
    se = sqrt.(diag(fre.vcov))
    @test isapprox(se[5], 1 / sqrt(2 * (n - p_mu)); atol = 1e-6)
    @test isapprox(se[6], 1 / sqrt(2 * (n - p_mu)); atol = 1e-6)
    @test all(isfinite, se)
    @test isposdef(Symmetric(fre.vcov))
    @test nobs(fre) == n

    # ---- the ML route is untouched -------------------------------------------
    # Same call, same numbers as before this route learned REML (the ML fit is
    # re-run through the SAME extracted design helper).
    fml2 = drm(fform, Gaussian(); data = dat, method = :ML)
    @test fml2.theta == fml.theta
    @test DRModels.loglik(fml2) == DRModels.loglik(fml)

    @testset "different mu1/mu2 designs: the profile does move with phi" begin
        # With mu2 dropped to an intercept the regressors differ, so the GLS
        # profile is no longer per-equation OLS and d(beta_hat)/d(phi) != 0 --
        # the branch of the vcov assembly that the shared-design cell above
        # leaves exactly zero. Guard that it still produces a finite, PD vcov.
        f2 = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ 1),
                sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                rho12 = @formula(rho12 ~ 1))
        g = drm(f2, Gaussian(); data = dat, method = :REML)
        @test g.converged
        @test estimation_method(g) === :REML
        @test all(isfinite, sqrt.(diag(g.vcov)))
        @test isposdef(Symmetric(g.vcov))
        @test isfinite(reml_loglik(g))
        @test reml_loglik(g) != ml_loglik(g)
    end

    @testset "partially observed rows still fit by REML" begin
        y1m = copy(y1); y2m = copy(y2)
        y1m[3] = NaN; y2m[7] = NaN
        datm = (; y1 = y1m, y2 = y2m, x)
        h = drm(fform, Gaussian(); data = datm, method = :REML)
        @test h.converged
        @test estimation_method(h) === :REML
        @test all(isfinite, h.theta)
        @test isfinite(reml_loglik(h))
        # The two dropped cells cannot leave the fit identical to the complete one.
        @test reml_loglik(h) != reml_loglik(fre)
    end

    @testset "`V` (known sampling covariance) keeps its permanent REML refusal" begin
        Vk = [fill(0.05, n) fill(0.01, n) fill(0.05, n)]
        err = nothing
        try
            drm(fform, Gaussian(); data = dat, method = :REML, V = Vk)
        catch e
            err = e
        end
        @test err isa ArgumentError
        @test occursin("no REML target", sprint(showerror, err))
    end
end
