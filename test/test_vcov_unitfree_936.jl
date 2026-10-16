# #936: the covariance guard's singular / variance-boundary decision must not
# depend on the units of `y` or of the predictors.
#
# Before the fix the guard compared min|eig(H)| / max|eig(H)| of the UNSCALED
# Hessian with 3e-8. For `y ~ year` with year = 2000 +/- 10 that ratio is 4.4e-12,
# so a perfectly identified fit was reported as "at a variance/dispersion
# boundary"; with y * 1e4 (centred year) it is 5.4e-9; with y * 1e8 it falls below
# `pinv`'s own cutoff and the pseudo-inverse truncated the mean directions
# (intercept SE 13,088 for a true 9.6e6). The guard now equilibrates the Hessian
# (`_vcov_scaling`): regression coefficients by their own curvature, declared
# unit-free coordinates (log-SDs, log-dispersions, atanh-correlations, 0/1
# columns of a dispersion formula) by max(curvature, floor) in natural units, so a
# flat one is still a boundary.
#
# Data: test/fixtures/vcov_unitfree_936/data.csv, written by Julia with
# StableRNG(20261016): n = 100, year = 2000 + 10 * z, y = 1 + 0.5 (year - 2000) + e.
# drmTMB reference SEs on the same data: expected_drmtmb.toml (gen_drmtmb.R).

using DRModels
using Test
using LinearAlgebra
using Logging
using DelimitedFiles
using StableRNGs
using TOML
using Random
import Distributions

const _UF936 = DRModels
const _UF936_DIR = joinpath(@__DIR__, "fixtures", "vcov_unitfree_936")

function _uf936_data()
    raw, _ = readdlm(joinpath(_UF936_DIR, "data.csv"), ',', Float64; header = true)
    return raw[:, 1], raw[:, 2]
end

function _uf936_fit(y, year)
    lg = Test.TestLogger(min_level = Logging.Warn)
    fit = Logging.with_logger(lg) do
        drm(bf(@formula(y ~ year), @formula(sigma ~ 1)), Gaussian(); data = (; y, year))
    end
    return fit, lg.logs
end

_uf936_guard_warnings(logs) =
    filter(l -> occursin(r"numerically singular|not positive definite", string(l.message)), logs)

_uf936_se(fit) = sqrt.(diag(Matrix(vcov(fit))))

# Closed-form Gaussian ML standard errors, in BigFloat, with no optimiser:
# Var(beta) = sigma2_ML (X'X)^-1 with sigma2_ML = RSS / n, and Var(log sigma) = 1 / (2n).
function _uf936_exact_se(y, year)
    n = length(y)
    X = hcat(ones(BigFloat, n), BigFloat.(year))
    yb = BigFloat.(y)
    XtX = X' * X
    β = XtX \ (X' * yb)
    σ2 = sum(abs2, yb - X * β) / n
    return Float64.(vcat(sqrt.(diag(σ2 * inv(XtX))), sqrt(1 / BigFloat(2n))))
end

@testset "#936 vcov guard: decision is unit-free" begin
    y, year = _uf936_data()
    m = 2000.0
    fc, lc = _uf936_fit(y, year .- m)             # the reference: centred year

    @testset "centred reference fit is healthy and silent" begin
        @test isempty(_uf936_guard_warnings(lc))
        @test all(isfinite, vcov(fc))
    end

    @testset "uncentred year: no false boundary warning" begin
        fu, lu = _uf936_fit(y, year)
        @test isempty(_uf936_guard_warnings(lu))     # warned on f71b292
        # SEs equal the centred fit's after the exact reparametrisation
        # a = a_c - m b:  V_u = J V_c J' with J = [1 -m 0; 0 1 0; 0 0 1].
        J = [1.0 -m 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        Vt = J * Matrix(vcov(fc)) * J'
        @test _uf936_se(fu) ≈ sqrt.(diag(Vt)) rtol = 1e-8
        @test _uf936_se(fu) ≈ _uf936_exact_se(y, year) rtol = 1e-8
    end

    @testset "y * 1e4 and y * 1e8: no warning, SEs scale exactly" begin
        for c in (1e4, 1e8), xs in (year, year .- m)
            f0, _ = _uf936_fit(y, xs)
            fs, ls = _uf936_fit(y .* c, xs)
            @test isempty(_uf936_guard_warnings(ls))  # 1e4 warned on f71b292
            # beta is in the units of y: SE scales by c. log sigma shifts by
            # log c, so its SE does not move.
            @test _uf936_se(fs) ≈ _uf936_se(f0) .* [c, c, 1.0] rtol = 1e-8
            @test _uf936_se(fs) ≈ _uf936_exact_se(y .* c, xs) rtol = 1e-8
        end
        # y * 1e8 used to cross pinv's own cutoff: the intercept SE came back as
        # 13,088 instead of 9.6e6 (truncated, not merely warned).
        f8, _ = _uf936_fit(y .* 1e8, year .- m)
        @test _uf936_se(f8)[1] ≈ 1e8 * _uf936_se(fc)[1] rtol = 1e-8
    end

    @testset "parity with drmTMB on the same data" begin
        ref = TOML.parsefile(joinpath(_UF936_DIR, "expected_drmtmb.toml"))
        cases = Dict("uncentred" => (y, year), "centred" => (y, year .- m),
                     "y1e4_uncentred" => (y .* 1e4, year), "y1e4_centred" => (y .* 1e4, year .- m))
        for (nm, (yy, xx)) in cases
            f, l = _uf936_fit(yy, xx)
            @test isempty(_uf936_guard_warnings(l))
            r = ref[nm]
            @test f.theta ≈ r["coef"] rtol = 1e-6
            # drmTMB stops its optimiser at ~3e-7 relative on these data (its
            # log-sigma SE is 0.07071065 against the exact 1/sqrt(200) =
            # 0.07071068); on the uncentred y * 1e4 fit it reports
            # converged = false and its SEs are 5.3e-6 off the closed form.
            tol = r["converged"] ? 1e-6 : 1e-5
            @test _uf936_se(f) ≈ r["se"] rtol = tol
        end
    end
end

@testset "#936 vcov guard: equivariance of the guard itself" begin
    rng = StableRNG(936)
    p = 5
    A = randn(rng, p, p)
    H = A' * A + 0.5I                 # healthy, well conditioned
    uf = [4, 5]                       # say, a log-sigma and a log-SD
    V = @test_logs DRModels._vcov_from_hessian(H; unitfree = uf)
    for e in (-8, -4, 4, 8)
        # Rescale the three regression coefficients by 10^e (their units); the
        # unit-free coordinates do not change scale.
        d = [10.0^e, 10.0^(-e), 10.0^(e ÷ 2), 1.0, 1.0]
        D = Diagonal(1 ./ d)          # theta' = d .* theta  =>  H' = D H D
        Hd = D * H * D
        Vd = @test_logs DRModels._vcov_from_hessian(Hd; unitfree = uf)
        @test Vd ≈ Diagonal(d) * V * Diagonal(d) rtol = 1e-10
    end
end

@testset "#936 vcov guard: true boundaries still warn" begin
    # A flat unit-free coordinate (a log-SD at its lower boundary: curvature 1e-9 in
    # log units, implied SE ~ 3e4) is a boundary whatever the units of the others.
    Hb = [40.0 -14.5 0.0 3.8
          -14.5 85.6 0.0 0.4
          0.0 0.0 1e-9 0.0
          3.8 0.4 0.0 12.1]
    for e in (-6, 0, 6)
        d = [10.0^e, 10.0^e, 1.0, 1.0]
        Hd = Diagonal(1 ./ d) * Hb * Diagonal(1 ./ d)
        lg = Test.TestLogger(min_level = Logging.Warn)
        V = Logging.with_logger(lg) do
            DRModels._vcov_from_hessian(Hd; unitfree = [3, 4], context = "boundary test")
        end
        @test length(lg.logs) == 1
        msg = string(lg.logs[1].message)
        @test occursin("numerically singular", msg)
        @test occursin("variance/dispersion boundary", msg)
        @test lg.logs[1].kwargs[:flat_coordinates] == [3]
        @test all(isfinite, V)
    end
    # The same curvature on a REGRESSION coefficient is units, not a boundary.
    @test_logs DRModels._vcov_from_hessian(Hb)

    # A clearly negative eigenvalue is still a saddle after any rescaling.
    Hn = [2.0 3.0; 3.0 2.0]
    for e in (-8, 0, 8)
        Dn = Diagonal([10.0^e, 1.0])
        Vn = @test_logs (:warn, r"not positive definite") DRModels._vcov_from_hessian(Dn * Hn * Dn)
        @test all(isnan, Vn)
    end

    # Two regression coefficients the data cannot separate (correlation 1 - 1e-12
    # after equilibration) are still singular, and the warning names them.
    ρ = 1 - 1e-12
    Hc = [1.0 ρ 0.0; ρ 1.0 0.0; 0.0 0.0 1.0]
    Dc = Diagonal([1e6, 1e-3, 1.0])
    lg = Test.TestLogger(min_level = Logging.Warn)
    Logging.with_logger(lg) do
        DRModels._vcov_from_hessian(Dc * Hc * Dc)
    end
    @test length(lg.logs) == 1
    @test occursin("not identified", string(lg.logs[1].message))
    @test isempty(lg.logs[1].kwargs[:flat_coordinates])
    @test lg.logs[1].kwargs[:null_direction_coordinates] == [1, 2]

    # Integration (AD Hessian): NB2 with a random intercept fitted by VA to Poisson
    # data drives the NB2 size to r -> infinity, so the log-sigma coordinate is flat
    # (test_vcov_guard.jl, anchor (c) of test_variational.jl). Rescaling the
    # predictor must neither hide that boundary nor move the slope SE off scale.
    se_slope = Float64[]
    for c in (1.0, 1e3, 1e-3)
        rng = MersenneTwister(1362)
        ng, per = 8, 6
        gidx = repeat(1:ng, inner = per)
        n = ng * per
        x = randn(rng, n)
        X = hcat(ones(n), x)
        bg = 0.5 .* randn(rng, ng)
        yv = Float64.([rand(rng, Distributions.Poisson(exp((X * [0.4, -0.3])[i] + bg[gidx[i]])))
                       for i in 1:n])
        lgv = Test.TestLogger(min_level = Logging.Warn)
        fv = Logging.with_logger(lgv) do
            DRModels._fit_nb2_ranef_va(DRModels.NegBinomial2(), yv, hcat(ones(n), c .* x), ones(n, 1),
                                       gidx, ng, ["(Intercept)", "x"], ["(Intercept)"], :g, 1e-8)
        end
        sing = filter(l -> occursin("numerically singular", string(l.message)), lgv.logs)
        @test length(sing) == 1
        @test occursin("variance/dispersion boundary", string(sing[1].message))
        @test sing[1].kwargs[:flat_coordinates] == [3]        # the NB2 log-sigma
        @test all(isfinite, fv.vcov)
        push!(se_slope, c * sqrt(fv.vcov[2, 2]))
    end
    @test se_slope[2] ≈ se_slope[1] rtol = 1e-6
    @test se_slope[3] ≈ se_slope[1] rtol = 1e-6
end

@testset "#936 _unitfree_columns" begin
    X = hcat(ones(4), [0.0, 1.0, 1.0, 0.0], [0.3, 1.2, -2.0, 4.0], [1.0, 1.0, 1.0, 1.0] .* 2)
    @test DRModels._unitfree_columns(X) == [1, 2]
    @test DRModels._unitfree_columns(X, 3) == [4, 5]
    @test_throws ArgumentError DRModels._vcov_from_hessian([1.0 0.0; 0.0 1.0]; unitfree = [3])
end
