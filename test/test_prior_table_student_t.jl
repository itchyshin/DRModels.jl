using Test
using Distributions: TDist, logpdf
using ForwardDiff
using DRModels

const student_t_data = (
    y = [0.2, 1.1, 1.8, 2.7, 3.3, 4.2, 5.1, 5.6, 6.4, 7.2, 8.0, 8.9],
    x = collect(range(-1.5, 1.5; length=12)),
)
const student_t_form = bf(@formula(y ~ x))
const student_t_table = (
    prior = ["student_t(0.25,1.5,4)"],
    class = ["fixef"],
    coef = ["x"],
    dpar = ["mu"],
    scale = ["link"],
)

@testset "Student-t coefficient prior table" begin
    @testset "normalized density and resolved target" begin
        rows = DRModels._parse_prior_table(student_t_table)
        @test length(rows) == 1
        @test only(rows).kind === :student_t
        @test only(rows).arguments == (0.25, 1.5, 4.0)

        resolved = resolve_priors(student_t_form, Gaussian();
                                  data=student_t_data, priors=student_t_table)
        @test length(resolved) == 1
        row = only(resolved)
        @test row.kind === :student_t
        @test row.dpar === :mu
        @test row.coef == "x"
        @test row.mean == 0.25
        @test row.scale === :link
        @test row.scale_parameter == 1.5
        @test row.df == 4.0

        theta = [0.7, -0.4]
        expected = -logpdf(TDist(4), (theta[row.index] - 0.25) / 1.5) + log(1.5)
        @test isapprox(DRModels._fixed_prior_value(resolved, theta), expected; atol=1e-12)

        h = 1e-5
        fd = (DRModels._fixed_prior_value(resolved, theta .+ [0, h]) -
              DRModels._fixed_prior_value(resolved, theta .- [0, h])) / (2h)
        @test isapprox(DRModels._prior_gradient(row, theta), fd; atol=1e-8)
        hessian_fd = (DRModels._prior_gradient(row, theta .+ [0, h]) -
                      DRModels._prior_gradient(row, theta .- [0, h])) / (2h)
        hessian_ad = ForwardDiff.hessian(t -> DRModels._fixed_prior_value(resolved, t), theta)[2, 2]
        @test isapprox(hessian_ad, hessian_fd; atol=1e-7)
    end

    @testset "fit and mixed fixed-prior rows" begin
        fit = drm(student_t_form, Gaussian(); data=student_t_data, priors=student_t_table)
        @test fit.estim_method === :MAP
        @test fit.penalty.kind === :fixed_priors
        @test any(r -> r.kind === :student_t && r.coef == "x", fit.penalty.rows)
        @test isfinite(fit.nll(fit.theta))

        mixed = merge(student_t_table, (;
            prior = ["normal(0,2)", "student_t(0.25,1.5,4)"],
            class = ["fixef", "fixef"],
            coef = ["(Intercept)", "x"],
            dpar = ["mu", "mu"],
            scale = ["link", "link"],
        ))
        mixed_fit = drm(student_t_form, Gaussian(); data=student_t_data, priors=mixed)
        @test length(mixed_fit.penalty.rows) == 2
        @test count(r -> r.kind === :student_t, mixed_fit.penalty.rows) == 1

        constructor_prior = drm_priors(drm_prior_student_t(
            dpar="mu", coef="x", location=0.25, scale=1.5, df=4))
        direct_fit = drm(student_t_form, Gaussian(); data=student_t_data, priors=constructor_prior)
        @test direct_fit.estim_method === :MAP
        @test only(direct_fit.penalty.specs).kind === :student_t
        replayed = update(fit, student_t_form; data=student_t_data)
        @test replayed.estim_method === :MAP
        @test any(p -> p isa DRModels.FixedPrior && p.kind === :student_t, replayed.penalty.specs)
    end

    @testset "constructor and invalid inputs" begin
        prior = drm_prior_student_t(dpar=:mu, coef="x", location=0.25, scale=1.5, df=4)
        @test prior.kind === :student_t
        @test_throws ArgumentError drm_prior_student_t(dpar=:mu, coef="x", scale=1, df=0)
        @test_throws ArgumentError drm_prior_student_t(dpar=:mu, coef="x", scale=0, df=4)
        @test_throws ArgumentError resolve_priors(student_t_form, Gaussian();
            data=student_t_data,
            priors=(prior=["student_t(0,1,0)"], class=["fixef"], coef=["x"]))
        @test_throws ArgumentError resolve_priors(student_t_form, Gaussian();
            data=student_t_data,
            priors=(prior=["student_t(0,0,4)"], class=["fixef"], coef=["x"]))
    end

    @testset "binomial logit and bridge parity" begin
        data = (y = [0, 0, 0, 1, 1, 1], x = [-3., -2., -1., 1., 2., 3.])
        form = bf(@formula(y ~ x))
        table = (prior = ["normal(0,3)", "student_t(0,2,4)"],
                 class = ["fixef", "fixef"], coef = ["(Intercept)", "x"],
                 dpar = ["mu", "mu"], scale = ["link", "link"])
        fit = drm(form, DRModels.Binomial(); data=data, priors=table)
        @test fit.estim_method === :MAP
        @test any(r -> r.kind === :student_t, fit.penalty.rows)
        bridge = drm_bridge(; formula="y ~ x", family="binomial", data=data,
            options=Dict(
                "priors" => [Dict("kind"=>"normal", "dpar"=>"mu", "coef"=>"(Intercept)", "mean"=>0., "sd"=>3.),
                             Dict("kind"=>"student_t", "dpar"=>"mu", "coef"=>"x",
                                  "location"=>0., "scale"=>2., "df"=>4.)],
                "prior_expected" => [Dict("dpar"=>"mu", "coef"=>"(Intercept)"),
                                     Dict("dpar"=>"mu", "coef"=>"x")]))
        @test bridge["coefficients"] ≈ fit.theta atol=1e-7
        @test bridge["penalized_objective"] ≈ fit.nll(fit.theta) atol=1e-10
        student_row = only(filter(r -> r["kind"] == "student_t", bridge["resolved_priors"]))
        @test student_row["scale"] == 2
        @test student_row["df"] == 4
    end
end
