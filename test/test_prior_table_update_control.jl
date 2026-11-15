using Test, DRModels

# `update` replays the prior table on the refit. The refit already carries the
# `control` report of its own `drm` call, and replaying the table must not drop it
# (the legacy `drm_priors` route keeps it).
@testset "update of a prior-table fit keeps the control report of the refit" begin
    d = (y=[.2,.8,.5,1.4,1.0,1.6,2.0,1.3,2.2,1.8,2.4,2.1],
         x=[-1.,0.,1.,-1.,0.,1.,-1.,0.,1.,-1.,0.,1.])
    f = bf(@formula(y ~ x))
    ctl = DRMControl(maxit = 500)
    table = (; prior=["normal(0,2)"], class=["fixef"], coef=["x"])
    fit = drm(f, Gaussian(); data=d, priors=table, control=ctl)
    @test control_info(fit) !== nothing
    refit = update(fit, f; data=d, control=ctl)
    @test control_info(refit) !== nothing
    @test refit.theta ≈ fit.theta atol = 1e-8
    # The replayed table is still attached.
    @test haskey(refit.penalty, :input_table)
    legacy = drm_priors(drm_prior_normal(dpar="mu", coef="x", mean=0, sd=2))
    lfit = drm(f, Gaussian(); data=d, priors=legacy, control=ctl)
    @test control_info(update(lfit, f; data=d, control=ctl)) !== nothing
end
