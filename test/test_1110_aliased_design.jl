# #1110: a rank-deficient fixed-effect design (an aliased column in `mu` or
# `sigma`) must not read as a converged fit with finite SEs. drmTMB reports
# `is_converged` FALSE and a `fixed_effect_rank_deficiency` warning (drmTMB
# #1470 / #1478); the twin does the same. The log-likelihood is unchanged and
# the pseudo-inverse covariance is unchanged: only the flag, the aliased SEs
# and the diagnosis move.
using DRModels
using Test, Random, LinearAlgebra, Logging

# Run `f` and return (result, warnings), where warnings are the @warn messages.
function _capture_warns(f)
    lg = Test.TestLogger(min_level = Logging.Warn)
    res = with_logger(f, lg)
    return res, [string(r.message) for r in lg.logs if r.level == Logging.Warn]
end

_rank_msgs(ws) = filter(w -> occursin("rank-deficient", w), ws)

@testset "#1110 aliased fixed-effect design" begin
    @testset "sigma ~ 1 + x + z, z = 2x + 1" begin
        Random.seed!(1110)
        n = 200
        x = randn(n); z = 2 .* x .+ 1
        y = 1.0 .+ 0.5 .* x .+ exp.(-0.3 .+ 0.2 .* x) .* randn(n)
        fit, ws = _capture_warns(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + x + z)),
                                           Gaussian(); data = (; y, x, z)))
        @test !is_converged(fit)
        r = check_drm(fit)
        @test !r.ok
        @test !r.converged
        rd = r.fixed_effect_rank_deficiency
        @test rd !== nothing
        @test rd.blocks[1].param === :sigma
        @test rd.blocks[1].aliased == ["z"]
        @test rd.blocks[1].rank == 2 && rd.blocks[1].ncol == 3
        # exactly one rank warning, naming the block and the aliased column
        rw = _rank_msgs(ws)
        @test length(rw) == 1
        @test occursin("sigma", rw[1]) && occursin("z", rw[1])
        # the singular Hessian is explained by the aliasing, not blamed on a variance boundary
        @test !any(w -> occursin("variance/dispersion boundary", w), ws)
        # SE of the aliased coordinate is NaN; identified coordinates keep finite SEs
        names = last.(fit.coefnames)
        idx = findfirst(r -> r.first === :sigma, fit.blocks)
        rng = fit.blocks[idx].second
        se = sqrt.(abs.(diag(fit.vcov)))
        @test isnan(se[rng[3]])
        @test all(isfinite, se[1:rng[1]-1])
        @test isfinite(se[rng[1]]) && isfinite(se[rng[2]])
        # the raw optimiser flag keeps its meaning
        @test fit.converged
    end

    @testset "duplicated column in mu" begin
        Random.seed!(1111)
        n = 200
        x = randn(n); x2 = copy(x)
        y = 1.0 .+ 0.5 .* x .+ 0.7 .* randn(n)
        fit, ws = _capture_warns(() -> drm(bf(@formula(y ~ x + x2), @formula(sigma ~ 1)),
                                           Gaussian(); data = (; y, x, x2)))
        @test !is_converged(fit)
        r = check_drm(fit)
        @test !r.ok
        @test r.fixed_effect_rank_deficiency.blocks[1].param === :mu
        @test r.fixed_effect_rank_deficiency.blocks[1].aliased == ["x2"]
        rw = _rank_msgs(ws)
        @test length(rw) == 1 && occursin("mu", rw[1]) && occursin("x2", rw[1])
    end

    @testset "full-rank control stays converged with no rank warning" begin
        Random.seed!(1112)
        n = 200
        x = randn(n); z = randn(n)
        y = 1.0 .+ 0.5 .* x .+ exp.(-0.3 .+ 0.2 .* x) .* randn(n)
        fit, ws = _capture_warns(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + x + z)),
                                           Gaussian(); data = (; y, x, z)))
        @test is_converged(fit)
        @test isempty(_rank_msgs(ws))
        @test all(isfinite, fit.vcov)
        r = check_drm(fit)
        @test r.fixed_effect_rank_deficiency === nothing
        @test r.ok
    end

    @testset "near-collinear but full rank is not flagged" begin
        Random.seed!(1113)
        n = 200
        x = randn(n); z = x .+ 1e-3 .* randn(n)
        y = 1.0 .+ 0.5 .* x .+ 0.7 .* randn(n)
        fit, ws = _capture_warns(() -> drm(bf(@formula(y ~ x + z), @formula(sigma ~ 1)),
                                           Gaussian(); data = (; y, x, z)))
        @test isempty(_rank_msgs(ws))
        @test check_drm(fit).fixed_effect_rank_deficiency === nothing
    end

    @testset "rank helper" begin
        X = [ones(5) collect(1.0:5) 2 .* collect(1.0:5) .+ 1]
        r = DRModels._design_rank(X, ["(Intercept)", "x", "z"])
        @test r.rank == 2 && r.aliased == ["z"]
        @test DRModels._design_rank(X[:, 1:2], ["(Intercept)", "x"]) === nothing
    end
end
