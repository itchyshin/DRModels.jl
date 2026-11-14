# Regression test: the bivariate Gaussian with an ordinary correlated random intercept pair
# `(1 | p | g)` keeps the best of three starts for the group covariance.
#
# On small data sets the default start (group SD 0.5, covariance 0.02) can slide to Sigma_g = 0, a local
# optimum, while drmTMB finds a higher mode with a group correlation of +1 or -1. Two more starts (group SD
# 0.2 with correlation +0.99 and -0.99) reach that mode. The default result is kept unless an extra start
# converged with a higher log-likelihood by more than 1e-6.
#
# Data: test/fixtures/biv_re_starts/d{80,132,135}.csv (make_data.jl, same generator as the domain-fix
# test). Reference: reference.tsv (make_reference.R), a real drmTMB 0.8.1 run. Agreement bounds, fixed
# beforehand: |dLL| <= 1e-4; fixed effects within 1e-3 of the R SE; log SDs of the two group effects within
# 5e-3. drmTMB's group correlation is +1 (d80) or -1 (d132, d135), where atanh is flat, so the correlation
# is checked to be beyond 0.9999 in the same direction. With the default start alone Julia ends at group SDs
# of about 1e-10, logLik lower by 3e-3 to 7e-2: with `_BIV_RE_EXTRA_STARTS = ()` every cell below fails.
using DRModels
using Test, DelimitedFiles, LinearAlgebra, Logging

const _STD = joinpath(@__DIR__, "fixtures", "biv_re_starts")

function _st_data(name)
    m, h = readdlm(joinpath(_STD, name * ".csv"), ',', Any; header = true)
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(String.(vec(h)))
        c = m[:, j]
        push!(cols, Symbol(n) => n == "g" ? String.(c) : Float64.(c))
    end
    return (; cols...)
end

const _ST_REF = let ref = Dict{String,Vector{Tuple{String,Float64,Float64}}}()
    m, _ = readdlm(joinpath(_STD, "reference.tsv"), '\t', Any; header = true)
    for r in eachrow(m)
        push!(get!(ref, String(r[1]), Tuple{String,Float64,Float64}[]),
              (String(r[3]), Float64(r[4]), r[5] == "NA" ? NaN : Float64(r[5])))
    end
    ref
end
_st_one(cell, name) = only(r for r in _ST_REF[cell] if r[1] == name)
_st_rows(cell, prefixes...) = [r for r in _ST_REF[cell] if any(p -> startswith(r[1], p), prefixes)]

const _ST_F = bf(mu1 = @formula(y1 ~ x1 + (1 | p | g)), mu2 = @formula(y2 ~ x2 + (1 | p | g)),
                 sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))

function _st_check(cell, data, method)
    fit = with_logger(NullLogger()) do
        drm(_ST_F, Gaussian(); data = data, method = method)
    end
    @test fit.converged
    # at the higher mode: not lower than drmTMB's logLik by more than 1e-4 (the default start alone: 3e-3 to 7e-2 lower)
    @test fit.loglik >= _st_one(cell, "logLik")[2] - 1e-4
    @test abs(fit.loglik - _st_one(cell, "logLik")[2]) <= 1e-4
    coefs = vcat([_st_rows(cell, p) for p in ("beta_mu1", "beta_mu2", "beta_sigma1", "beta_sigma2", "beta_rho12")]...)
    @test length(coefs) == 7
    for (k, (nm, est, se)) in enumerate(coefs)
        @test abs(fit.theta[k] - est) <= 1e-3 * se
    end
    S = fit.ranef.Sigma_g
    sd = sqrt.(diag(S))
    sdref = _st_rows(cell, "log_sd_mu")
    @test abs(log(sd[1]) - sdref[1][2]) <= 5e-3
    @test abs(log(sd[2]) - sdref[2][2]) <= 5e-3
    rho = S[1, 2] / (sd[1] * sd[2])
    @test sign(rho) == sign(_st_one(cell, "eta_cor_mu")[2])
    @test abs(rho) > 0.9999
    return fit
end

@testset "bivariate ordinary (1 | p | g): extra starts reach the higher mode drmTMB finds" begin
    # Correlation +1 in drmTMB (d80) and -1 (d132, d135): the +0.99 and the -0.99 start each matter.
    @testset "d80 ML" begin _st_check("d80_ml", _st_data("d80"), :ML) end
    @testset "d80 REML" begin _st_check("d80_reml", _st_data("d80"), :REML) end
    @testset "d132 ML" begin _st_check("d132_ml", _st_data("d132"), :ML) end
    @testset "d132 REML" begin _st_check("d132_reml", _st_data("d132"), :REML) end
    @testset "d135 ML" begin _st_check("d135_ml", _st_data("d135"), :ML) end
end

@testset "bivariate ordinary (1 | p | g): start selection rules" begin
    # A double well in the second-to-last coordinate (the `L21` entry of the group Cholesky): the default
    # start and the +0.99 start roll into the well at +1 (value 0), the -0.99 start into the well at -1
    # (value `d`; a quartic well, so that one iteration does not reach it). Starts are compared on `obj`; the helper is the one the fit uses.
    best_start = DRModels._biv_re_best_start
    gain = DRModels._BIV_RE_START_GAIN
    @test gain == 1e-6
    @test DRModels._BIV_RE_EXTRA_STARTS == ((0.2, 0.99), (0.2, -0.99))
    mkobj(d) = x -> min((x[end-1] - 1)^2, (x[end-1] + 1)^4 + d) + sum(abs2, x[1:(end-3)]) + (x[end] + 3)^2
    x0 = vcat([0.3, -0.2], DRModels.cov_to_lc(Matrix(Symmetric([0.25 0.02; 0.02 0.25]))))
    runner(obj; iterations = 1000) = x -> DRModels.Optim.optimize(obj, x, DRModels.Optim.LBFGS(),
        DRModels.Optim.Options(g_tol = 1e-8, iterations = iterations); autodiff = :forward)
    lc2(r) = DRModels.Optim.minimizer(r)[end-1]
    # 1. the other well is deeper by far more than the gain: the extra start is returned
    obj = mkobj(-1e-3)
    r = best_start(runner(obj), obj, x0)
    @test lc2(r) < -0.9
    # 2. deeper by less than the gain: the default result is returned, the same object as a plain run
    obj = mkobj(-1e-8)
    r = best_start(runner(obj), obj, x0)
    @test lc2(r) > 0.9
    @test DRModels.Optim.minimizer(r) == DRModels.Optim.minimizer(runner(obj)(x0))
    # 3. the other well is higher: the default result stays
    obj = mkobj(+0.5)
    @test lc2(best_start(runner(obj), obj, x0)) > 0.9
    # 4. deeper but the extra start did not converge: the default result stays
    obj = mkobj(-1e-3)
    slow(x) = x[end-1] < 0 ? runner(obj; iterations = 1)(x) : runner(obj)(x)
    @test !DRModels.Optim.converged(slow(vcat(x0[1:(end-3)], DRModels.cov_to_lc(Matrix(Symmetric(0.04 .* [1.0 -0.99; -0.99 1.0]))))))
    @test lc2(best_start(slow, obj, x0)) > 0.9
    # 5. the default start fails: a failed start is skipped and the best extra start is used
    obj = mkobj(-1e-3)
    failfirst(x) = x == x0 ? throw(DomainError(-1.0, "log")) : runner(obj)(x)
    @test lc2(best_start(failfirst, obj, x0)) < -0.9
    # 6. an extra start fails: skipped, the default result stays
    failextra(x) = x == x0 ? runner(obj)(x) : throw(SingularException(2))
    @test lc2(best_start(failextra, obj, x0)) > 0.9
    # 7. every start fails: the default start's error is rethrown
    allfail(x) = x == x0 ? throw(DomainError(-1.0, "first")) : throw(SingularException(2))
    @test_throws DomainError best_start(allfail, obj, x0)
    # 8. an error that is not numerical is never swallowed
    @test_throws MethodError best_start(x -> x == x0 ? runner(obj)(x) : nothing + 1, obj, x0)
    # 9. the start values: only the last three entries change, in this order, deterministically
    seen = Vector{Float64}[]
    rec(x) = (push!(seen, copy(x)); runner(obj)(x))
    best_start(rec, obj, x0); first_run = copy(seen); empty!(seen); best_start(rec, obj, x0)
    @test seen == first_run
    @test length(seen) == 3 && seen[1] == x0
    for (k, (sd0, cr0)) in enumerate(DRModels._BIV_RE_EXTRA_STARTS)
        @test seen[k+1][1:(end-3)] == x0[1:(end-3)]
        @test seen[k+1][(end-2):end] == DRModels.cov_to_lc(Matrix(Symmetric(sd0^2 .* [1.0 cr0; cr0 1.0])))
    end
end

@testset "bivariate ordinary (1 | p | g): a start that did not converge is never kept, even when its value is lower" begin
    # Same double well as above, but the well at -1 is deeper (value -0.2; still shallow enough that the
    # default start rolls into the well at +1). The -0.99 start is stopped by `iterations = 1`, so it
    # ends short of the well but already at a value below the default result (0) by far more than the
    # gain: it would win on value alone. Only the convergence requirement keeps it out. The
    # runner is a real Optim run with a real iteration limit; `_biv_re_best_start` is the real selection.
    best_start = DRModels._biv_re_best_start
    gain = DRModels._BIV_RE_START_GAIN
    obj = x -> min((x[end-1] - 1)^2, (x[end-1] + 1)^4 - 0.2) + sum(abs2, x[1:(end-3)]) + (x[end] + 3)^2
    x0 = vcat([0.3, -0.2], DRModels.cov_to_lc(Matrix(Symmetric([0.25 0.02; 0.02 0.25]))))
    runner(; iterations = 1000) = x -> DRModels.Optim.optimize(obj, x, DRModels.Optim.LBFGS(),
        DRModels.Optim.Options(g_tol = 1e-8, iterations = iterations); autodiff = :forward)
    slow(x) = x[end-1] < 0 ? runner(iterations = 1)(x) : runner()(x)
    xs = vcat(x0[1:(end-3)], DRModels.cov_to_lc(Matrix(Symmetric(0.04 .* [1.0 -0.99; -0.99 1.0]))))
    r_short = slow(xs)
    r_default = runner()(x0)
    # the premise: not converged, and clearly lower than the default result
    @test !DRModels.Optim.converged(r_short)
    @test obj(DRModels.Optim.minimizer(r_short)) < obj(DRModels.Optim.minimizer(r_default)) - 100 * gain
    # the same start run to the end does win, so the value rule alone would pick the short run
    @test DRModels.Optim.minimizer(best_start(runner(), obj, x0))[end-1] < -0.9
    # the rule: the not converged start is skipped and the default result is returned unchanged
    r = best_start(slow, obj, x0)
    @test DRModels.Optim.minimizer(r) == DRModels.Optim.minimizer(r_default)
    @test DRModels.Optim.minimizer(r)[end-1] > 0.9
    @test DRModels.Optim.converged(r)
end
