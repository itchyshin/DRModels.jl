using DRModels
using Test

@testset "convergence checks curvature when standard errors are skipped" begin
    x = collect(range(-1, 1; length=40))
    y = 0.5 .+ 0.7 .* x .+ 0.4 .* sin.(collect(1:40))
    fit = drm(bf(@formula(y ~ x)), Gaussian(); data=(; x, y),
              control=DRMControl(se=false))
    @test fit.control_info.se === :skipped
    @test all(isnan, fit.vcov)
    @test convergence_status(fit) === :converged
    @test all(isnan, fit.vcov) # Checking convergence must not add SEs to the fit.

    # Some routes compute a covariance and discard it for se=false. Their
    # stored curvature result must survive without exposing the covariance.
    info = fit.control_info
    report(curvature) = DRMControlReport(info.control, info.start_logliks,
        info.kept_start, info.starts_agree, :discarded, info.weights,
        info.requested_optimizer, info.actual_optimizer, info.optimizer_runs,
        info.selected_optimizer_stage, info.native_controls, curvature)
    good = (positive_definite=true, finite_se=true, inflated_se=true)
    bad_curvature = (positive_definite=false, finite_se=false, inflated_se=false)
    discarded_good = DRModels._withcontrolinfo(fit, fit.vcov, report(good))
    discarded_bad = DRModels._withcontrolinfo(fit, fit.vcov, report(bad_curvature))
    @test convergence_status(discarded_good) === :converged
    @test convergence_status(discarded_bad) === :degenerate

    priors = drm_priors(drm_prior_normal(dpar=:mu, coef="x", mean=0, sd=2))
    mapfit = drm(bf(@formula(y ~ x)), Gaussian(); data=(; x, y), priors=priors,
                 control=DRMControl(se=false))
    @test mapfit.estim_method === :MAP
    @test convergence_status(mapfit) === :converged
    @test all(isnan, mapfit.vcov)

    theta0 = copy(fit.theta)
    flat = theta -> 20.0 + zero(sum(theta))
    flatgrad = (g, theta) -> fill!(g, 0.0)
    flatfit = DRModels._withnll(fit, flat, flatgrad)
    flatfit = DRModels._withcontrolinfo(flatfit, flatfit.vcov, fit.control_info)
    @test convergence_status(flatfit) === :degenerate

    # An inflated temporary SE must not be reported as a failed SE diagnostic:
    # the caller explicitly requested no standard errors.
    wide = theta -> 20.0 + (1e-8 * (theta[1] - theta0[1])^2 +
        sum(abs2, theta[2:end] .- theta0[2:end])) / 2
    widegrad = (g, theta) -> (g .= [1e-8 * (theta[1] - theta0[1]); theta[2:end] .- theta0[2:end]])
    widefit = DRModels._withnll(fit, wide, widegrad)
    widefit = DRModels._withcontrolinfo(widefit, widefit.vcov, fit.control_info)
    @test convergence_status(widefit) === :converged

    bad = theta -> 20.0 - sum(abs2, theta .- theta0) / 2
    badgrad = (g, theta) -> (g .= -(theta .- theta0))
    badfit = DRModels._withnll(fit, bad, badgrad)
    badfit = DRModels._withcontrolinfo(badfit, badfit.vcov, fit.control_info)
    @test convergence_status(badfit) === :degenerate
end
