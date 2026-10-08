# Regression tests for the three triage twins:
#   #1018 level / B validation (drmTMB #1480)
#   #1008 update carries stored fit options (drmTMB #1241)
#   #962  bootstrap check_converged default and incomplete status (drmTMB #1458)
using DRModels
using Test
using Random
using Statistics

function _argmsg(f)
    try
        f()
        error("expected an ArgumentError")
    catch err
        err isa ArgumentError || rethrow()
        return err.msg
    end
end

function _with_theta(fit::DrmFit; theta = fit.theta, converged::Bool = fit.converged,
                      scales = fit.scales)
    return DrmFit(
        fit.family, fit.blocks, fit.coefnames, collect(Float64, theta), fit.vcov,
        fit.loglik, fit.nobs, converged, fit.means, fit.obs, scales,
        fit.formula, fit.nll, fit.nllgrad, fit.ranef, fit.estim_method,
        fit.reml_loglik, fit.ml_loglik, fit.marginal, fit.phylo_penalty,
        fit.penalty, fit.iterations, fit.phylo_scale,
    )
end

@testset "triage G: level, update options, bootstrap status" begin
    Random.seed!(1018)
    n = 60
    x = randn(n)
    y = 1 .+ 0.5 .* x .+ randn(n)
    data = (; y, x)
    form = bf(@formula(y ~ x), @formula(sigma ~ 1))
    fit = drm(form, Gaussian(); data = data)

    @testset "#1018 level and B are rejected before an interval is built" begin
        for bad in (-0.5, 0.0, 1.0, NaN, Inf, 95)
            msg = _argmsg(() -> confint(fit; level = bad))
            @test occursin("level", msg)
            @test occursin(repr(bad), msg)
            @test !occursin("abs(x)", msg)
            @test occursin("level", _argmsg(() -> coeftable(fit; level = bad)))
            @test occursin("level", _argmsg(() -> confint(fit; method = :profile, level = bad)))
            @test occursin("level", _argmsg(() -> profile_result(fit; level = bad)))
            @test occursin("level", _argmsg(() -> profile_curve(fit, 1; level = bad)))
            @test occursin("level", _argmsg(() -> bias_correct(fit, θ -> exp(θ[2]); level = bad)))
        end
        # A valid level still returns lower <= upper (the -0.5 case used to invert).
        row = confint(fit; level = 0.95)[1]
        @test row.lower <= row.estimate <= row.upper

        for badB in (0, -3, 1.5, NaN, Inf, true)
            msg = _argmsg(() -> bootstrap_ci(fit; data = data, B = badB, rng = MersenneTwister(1)))
            @test occursin("B", msg)
            @test occursin(repr(badB), msg)
        end
        # B = 1 is a legal (degenerate) percentile; a whole-valued real is an integer.
        one = bootstrap_result(fit; data = data, B = 1.0, rng = MersenneTwister(2), failures = :skip)
        @test one.attempted == 1
        @test one.status == "bootstrap_unavailable"

        # Sibling entry points share the helper. icc validates before it looks
        # at the model, so a fixed-effect fit is enough.
        @test occursin("icc", _argmsg(() -> icc(fit; level = 95)))
        @test occursin("heritability", _argmsg(() -> heritability(fit; level = -1)))
        @test occursin("level", _argmsg(() -> profile_sigma_a(fit; level = 95)))
        @test DRModels._validate_ci_level(0.95) === 0.95
        @test DRModels._validate_bootstrap_B(4.0) === 4
    end

    @testset "#1008 update repeats stored method, marginal, and penalty" begin
        reml = drm(form, Gaussian(); data = data, method = :REML)
        @test reml.estim_method === :REML
        reduced_f = bf(@formula(y ~ 1), @formula(sigma ~ 1))
        reduced = update(reml, reduced_f; data = data)
        direct = drm(reduced_f, Gaussian(); data = data, method = :REML)
        @test reduced.estim_method === :REML
        @test coef(reduced) ≈ coef(direct)
        overridden = update(reml, reduced_f; data = data, method = :ML)
        @test overridden.estim_method === :ML

        g = repeat(1:6, inner = 5)
        yp = rand(MersenneTwister(3), 1:4, 30)
        xp = randn(MersenneTwister(3), 30)
        dp = (; y = yp, x = xp, g = g)
        pois = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = dp,
                    marginal = :Laplace, se = false)
        @test pois.marginal === :Laplace
        pois_u = update(pois, bf(@formula(y ~ 1 + (1 | g))); data = dp, se = false)
        @test pois_u.marginal === :Laplace
        @test pois_u.estim_method === :ML

        G = 8
        m = 2
        phy = random_balanced_tree(G; branch_length = 0.3)
        species = repeat(1:G, inner = m)
        xp = randn(MersenneTwister(4), G * m)
        yp = 0.2 .+ 0.4 .* xp .+ 0.3 .* randn(MersenneTwister(5), G * m)
        dphy = (; y = yp, x = xp, species = species)
        pen = drm_phylo_penalty(sd_u = 0.8)
        fphy = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1))
        mapfit = drm(fphy, Gaussian(); data = dphy, tree = phy, penalty = pen)
        @test mapfit.estim_method === :MAP
        # `tree` is not stored on the fit, so the caller passes it again.
        # The stored penalty still has to come along without being repeated.
        map_u = update(mapfit, bf(@formula(y ~ 1 + phylo(1 | species)), @formula(sigma ~ 1));
                       data = dphy, tree = phy)
        @test map_u.estim_method === :MAP
        @test map_u.penalty == pen
        # Changing the estimator drops the stored MAP penalty unless one is passed.
        ml_from_map = update(mapfit,
            bf(@formula(y ~ 1 + phylo(1 | species)), @formula(sigma ~ 1));
            data = dphy, tree = phy, method = :ML)
        @test ml_from_map.estim_method === :ML
        @test ml_from_map.penalty === nothing

        # A NamedTuple in this slot is splatted into keywords. The silent-drop
        # case is a real positional extra beside a supplied `data`.
        unnamed = _argmsg(() -> update(reml, reduced_f, "dropped"; data = data))
        @test occursin("named arguments", unnamed)
        famsg = _argmsg(() -> update(reml, reduced_f; data = data, family = Poisson()))
        @test occursin("family", famsg)

        short = drm(form, Gaussian(); data = (y = y[1:30], x = x[1:30]))
        nmsg = _argmsg(() -> lrtest(short, fit))
        @test occursin("nobs", nmsg)
        @test occursin("different samples", nmsg)
    end

    @testset "#962 check_converged defaults to true; dropped replicates warn" begin
        clean = bootstrap_result(fit; data = data, B = 2, rng = MersenneTwister(6),
                                 failures = :skip)
        @test clean.check_converged == true
        @test clean.failed == 0
        @test clean.status == "bootstrap"
        @test isempty(clean.boundary_params)

        calls = Ref(0)
        function drop_second(datab)
            calls[] += 1
            refit = drm(form, Gaussian(); data = datab)
            return calls[] == 2 ? _with_theta(refit; converged = false) : refit
        end
        logs, dropped = Test.collect_test_logs() do
            DRModels._bootstrap_result(
                fit, form, data, 4, 0.95, MersenneTwister(7), false, drop_second;
                failures = :skip,
            )
        end
        @test dropped.check_converged == true
        @test dropped.failed == 1
        @test dropped.used == 3
        @test dropped.status == "bootstrap_incomplete"
        @test any(l -> occursin("bootstrap_incomplete", string(l.message)), logs)
        @test any(l -> occursin("dropping", string(l.message)), logs)

        # Default `failures = :error` still aborts a thrown refit. A replicate
        # that merely did not converge is dropped, warned, and does not abort.
        calls[] = 0
        elogs, edrop = Test.collect_test_logs() do
            DRModels._bootstrap_result(
                fit, form, data, 3, 0.95, MersenneTwister(8), false, drop_second)
        end
        @test edrop.failed == 1
        @test edrop.used == 2
        @test edrop.status == "bootstrap_incomplete"
        @test any(l -> occursin("dropping", string(l.message)), elogs)
        @test_throws ErrorException DRModels._bootstrap_result(
            fit, form, data, 2, 0.95, MersenneTwister(8), false, _ -> error("forced refit failure"))

        g = repeat(1:8, inner = 5)
        yb = 0.3 .+ 0.2 .* randn(MersenneTwister(9), 40)
        xb = randn(MersenneTwister(9), 40)
        db = (; y = yb, x = xb, g = g)
        fre = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))
        fitre = drm(fre, Gaussian(); data = db)
        resd = findfirst(p -> first(p) === :resd, fitre.blocks)
        @test resd !== nothing
        idx = last(fitre.blocks[resd])[1]
        θb = copy(fitre.theta)
        θb[idx] = log(1e-8)
        # This seed can land on a collapsed residual scale (the Hessian warning
        # above). `is_converged` rejects that independently of the log-SD we
        # pin, so restore a non-degenerate residual SD and keep only `:resd` at
        # the boundary. The boundary flag reads the coefficient, not `scales`.
        scales = copy(fitre.scales)
        scales[:sigma] = fill(0.2, length(fitre.scales[:sigma]))
        pinned = _with_theta(fitre; theta = θb, converged = true, scales = scales)
        @test is_converged(pinned)
        blogs, bounded = Test.collect_test_logs() do
            DRModels._bootstrap_result(
                fitre, fre, db, 20, 0.95, MersenneTwister(10), false, _ -> pinned;
                failures = :skip,
            )
        end
        @test bounded.failed == 0
        @test bounded.status == "bootstrap_at_boundary"
        @test any(n -> startswith(n, "resd:"), bounded.boundary_params)
        @test any(l -> occursin("bootstrap_at_boundary", string(l.message)), blogs)

        calls[] = 0
        function drop_and_pin(_)
            calls[] += 1
            return calls[] == 1 ? _with_theta(pinned; converged = false) : pinned
        end
        both, bothres = Test.collect_test_logs() do
            DRModels._bootstrap_result(
                fitre, fre, db, 21, 0.95, MersenneTwister(11), false, drop_and_pin;
                failures = :skip,
            )
        end
        @test bothres.failed == 1
        @test bothres.status == "bootstrap_at_boundary"
        @test any(l -> occursin("dropping", string(l.message)) &&
                      occursin("bootstrap_at_boundary", string(l.message)), both)
    end

    @testset "scaled predictors stay converged and keep every bootstrap replicate" begin
        rng = MersenneTwister(1503)
        n = 240
        G = 24
        g = repeat(1:G, inner = n ÷ G)
        x0 = randn(rng, n)
        b = 0.4 .* randn(rng, G)
        y = 0.3 .+ 0.5 .* x0 .+ b[g] .+ exp.(-0.3 .+ 0.15 .* x0) .* randn(rng, n)
        for scale in (1.0, 1000.0)
            xs = x0 .* scale
            data = (; y, x = xs, g)
            scaled = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x)),
                         Gaussian(); data)
            @test is_converged(scaled)
            res = bootstrap_result(scaled; data, B = 40, rng = MersenneTwister(1503))
            @test res.failed == 0
            @test res.used == 40
            @test res.status == "bootstrap"
        end
        xs = x0 .* 1e5
        fixed = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian();
                    data = (; y, x = xs))
        @test is_converged(fixed)
    end
end
