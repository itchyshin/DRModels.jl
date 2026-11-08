# Failed-fit hygiene (2026-11-08 robustness lane):
#   1. NB2 structured sigma `sigma ~ 1 + animal(1 | id)` with `sd(id, animal, sigma) ~ w0` returned
#      a PosDefException out of `drm()`; it now returns a fit with converged = false.
#   2. No sentinel / runaway log-likelihood reaches the user: `loglik`/`loglikelihood` is NaN with a
#      warning for a failed fit, `show` prints NaN, the raw value stays in `fit.loglik`.
#   3. The unlabelled-`A` (and unlabelled `coords`) warning fires on every call.
using DRModels, Test, LinearAlgebra, DelimitedFiles, Logging, StableRNGs
import Distributions

const _FX = joinpath(@__DIR__, "fixtures", "entry_robust")

_quiet(f) = Logging.with_logger(f, Logging.NullLogger())

# The user-facing contract for a failed or degenerate fit, independent of which local optimum (or
# non-optimum) an optimiser reaches on a given platform / Julia version / BLAS: `drm()` returns a
# DrmFit, and `loglik(fit)` is either a finite, plausible number or NaN. A failed-objective
# sentinel (|ll| >= 1e15, or NaN raw) never reaches the user. A converged fit has a finite loglik.
function _check_loglik_contract(fit)
    @test fit isa DRModels.DrmFit
    @test fit.converged isa Bool
    ll_user = _quiet(() -> loglik(fit))
    @test isnan(ll_user) || abs(ll_user) < 1e15                  # no sentinel / runaway value leaks
    @test isnan(_quiet(() -> loglikelihood(fit))) == isnan(ll_user)
    fit.converged && @test isfinite(ll_user)                      # converged => finite
    if !fit.converged && !(isfinite(fit.loglik) && abs(fit.loglik) < 1e15)
        @test isnan(ll_user)                                      # failed raw value => NaN for the user
    end
    @test isnan(ll_user) || ll_user == fit.loglik                 # never altered, only withheld
    return ll_user
end

@testset "NB2 structured sigma animal: no PosDefException, loglik contract on every platform" begin
    raw = readdlm(joinpath(_FX, "nb2_sigma_animal_data.csv"), ','; header = true)[1]
    data = (; y = Float64.(raw[:, 1]), x = Float64.(raw[:, 2]), v = Float64.(raw[:, 3]),
            w0 = Float64.(raw[:, 4]), id = string.(raw[:, 5]))
    A = readdlm(joinpath(_FX, "nb2_sigma_animal_A.csv"), ',')
    ids = unique(data.id)
    f = bf(@formula(y ~ x + v), @formula(sigma ~ 1 + animal(1 | id)), @formula(sd(id, animal, sigma) ~ w0))
    fit_at(d) = drm(f, NegBinomial2(); data = d, A, ids, se = false)
    # as-generated data, and the reproducer (rescaling the SD-formula covariate is a pure
    # reparametrisation). Both are degenerate: whether the optimiser stops at a finite non-optimum
    # (Mac, Julia 1.10.0: -354.5337) or stores NaN (Linux, Julia 1.10.12 and 1.12.7) is platform
    # numerics, so only the contract is asserted. The point of the fix is that nothing throws.
    for d in (data, merge(data, (; w0 = 2.5 .* data.w0)))
        fit = _quiet(() -> fit_at(d))
        _check_loglik_contract(fit)
    end
end

function _failed_copy(f, ll; converged = false)
    names = fieldnames(typeof(f))
    vals = Any[getfield(f, k) for k in names]
    vals[findfirst(==(:loglik), names)] = ll
    vals[findfirst(==(:converged), names)] = converged
    return typeof(f).name.wrapper(vals...)
end

@testset "loglik is NaN with a warning on a failed fit; converged fits unchanged" begin
    rng = StableRNG(7)
    x = randn(rng, 60)
    d = (; y = 1 .+ 0.5 .* x .+ randn(rng, 60), x)
    good = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = d)
    ll = good.loglik
    @test isfinite(ll)
    @test @test_logs(loglikelihood(good)) == ll                     # no warning, same value
    @test loglik(good) == ll
    for bad_ll in (-1e18, -floatmax(Float64), -Inf, NaN)
        bad = _failed_copy(good, bad_ll)
        v = @test_logs (:warn, r"loglik\(fit\) is NaN") match_mode = :any loglikelihood(bad)
        @test isnan(v)
        @test isnan(@test_logs((:warn, r"NaN"), match_mode = :any, loglik(bad)))
        @test isnan(@test_logs((:warn, r"NaN"), match_mode = :any, aic(bad)))
        @test (bad_ll == -1e18 ? bad.loglik == -1e18 : true)         # the raw value stays on the object
    end
    # a runaway positive value on a non-converged fit (Gaussian q2, x scaled by 1e6, was +3.6e76)
    runaway = _failed_copy(good, 3.6e76)
    @test isnan(@test_logs((:warn, r"NaN"), match_mode = :any, loglikelihood(runaway)))
    # a non-converged fit with a plausible log-likelihood is NOT touched
    plausible = _failed_copy(good, ll; converged = false)
    @test @test_logs(loglikelihood(plausible)) == ll
    # the REPL display prints NaN, not the sentinel
    s = sprint(show, MIME"text/plain"(), _failed_copy(good, -1e18))
    @test occursin("logLik = NaN", s)
    @test !occursin("e18", s) && !occursin("1000000000000000000", s)
    @test occursin("logLik = NaN", sprint(show, _failed_copy(good, -1e18)))
end

@testset "real failed fits: Binomial spatial (duplicate coords) and Gaussian q2 (x scaled 1e6)" begin
    function _read(path)
        L = readlines(path); hdr = strip.(split(L[1], ','), '"'); rows = [split(l, ',') for l in L[2:end]]
        cols = Dict{Symbol,Any}()
        for (j, h) in enumerate(hdr)
            raw = [strip(r[j], '"') for r in rows]
            cols[Symbol(h)] = h == "g" ? String.(raw) : parse.(Float64, raw)
        end
        return cols
    end
    c = _read(joinpath(_FX, "binomial_spatial_dupcoords_data.csv"))
    data = (; y = c[:y], x = c[:x], z = c[:z], g = c[:g])
    cc, _ = readdlm(joinpath(_FX, "binomial_spatial_dupcoords_coords.csv"), ','; header = true)
    coords = Dict(String(strip(String(cc[k, 1]), '"')) => [Float64(cc[k, 2]), Float64(cc[k, 3])] for k in 1:size(cc, 1))
    # 24 coordinate rows but only 21 distinct points: a singular spatial kernel. Either a clear
    # ArgumentError, or a fit that obeys the loglik contract. Which local outcome the optimiser reaches
    # is platform numerics (Mac: sentinel -1e18; Linux: finite non-converged -139.55 on 1.10.12,
    # -132.45 on 1.12.7), so only the contract is asserted, never a particular value.
    res = try
        _quiet(() -> drm(bf(@formula(y ~ x + spatial(1 | g)), @formula(sd(g, spatial) ~ z)), Binomial(); data, coords))
    catch e
        e
    end
    if res isa Exception
        @test res isa ArgumentError
        @test !isempty(sprint(showerror, res))
    else
        _check_loglik_contract(res)
        @test isnan(_quiet(() -> aic(res))) == isnan(_quiet(() -> loglik(res)))
        if !res.converged && res.loglik <= -1e15               # the internal sentinel is stored ...
            v = @test_logs (:warn, r"NaN") match_mode = :any loglikelihood(res)
            @test isnan(v)                                     # ... but is never what the user reads
        end
    end

    g = _read(joinpath(_FX, "gaussian_q2_x1e6_data.csv"))
    dq = (; y = g[:y], x = g[:x], z0 = g[:z0], z1 = g[:z1], g = g[:g])
    fq = _quiet(() -> drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1),
                             @formula(sd(g, coef(intercept)) ~ z0), @formula(sd(g, coef(x)) ~ z1)), Gaussian(); data = dq))
    @test fq.converged == false
    @test abs(fq.loglik) >= 1e15                                # stored +3.6e76
    @test isnan(_quiet(() -> loglikelihood(fq)))
end

@testset "unlabelled A / coords: the warning fires on every call, not once per session" begin
    G, m = 6, 5
    rng = StableRNG(5)
    g = repeat(["a$i" for i in 1:G], inner = m)
    x = randn(rng, G * m)
    y = 1 .+ 0.4 .* x .+ repeat(randn(rng, G), inner = m) .+ 0.5 .* randn(rng, G * m)
    d = (; y, x, g)
    R0 = [0.4^abs(i - j) for i in 1:G, j in 1:G]
    f = bf(@formula(y ~ x + animal(1 | g)), @formula(sigma ~ 1))
    for fam_call in (A -> drm(f, Gaussian(); data = d, A),
                     A -> drm(bf(@formula(y ~ x + animal(1 | g))), Poisson(); data = merge(d, (; y = round.(abs.(d.y)))), A))
        for k in 1:3
            Ak = k == 2 ? reverse(R0, dims = (1, 2)) : R0      # a second, wrong-order call must not be silent
            @test_logs (:warn, r"`A` has no labels") match_mode = :any fam_call(Ak)
        end
    end
    # a labelled A stays silent
    @test_logs match_mode = :any min_level = Logging.Warn drm(f, Gaussian(); data = d, A = R0, ids = unique(g))
    # coords: same rule
    fc = bf(@formula(y ~ x + spatial(1 | g)), @formula(sigma ~ 1))
    xy = hcat(Float64.(1:G), zeros(G))
    for k in 1:2
        @test_logs (:warn, r"coords") match_mode = :any drm(fc, Gaussian(); data = d, coords = xy)
    end
end

# The R bridge must not ship the sentinel: a failed fit crosses as NaN plus an explicit flag, a good
# fit crosses unchanged (R reads `loglik` with as.numeric(); NaN and the extra keys are safe).
@testset "bridge: failed fit round-trips as NaN + fit_failed; good fit unchanged" begin
    rng = StableRNG(7)
    x = randn(rng, 60)
    d = (; y = 1 .+ 0.5 .* x .+ randn(rng, 60), x)
    good = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = d)
    flat(f) = _quiet(() -> DRModels._bridge_flatten(f; family = "gaussian"))
    g = flat(good)
    @test g["loglik"] == good.loglik
    @test g["ml_loglik"] == good.ml_loglik
    @test g["fit_failed"] === false
    @test g["fit_failed_message"] == ""
    for bad_ll in (-1e18, -floatmax(Float64), -Inf, NaN)
        b = flat(_failed_copy(good, bad_ll))
        @test isnan(b["loglik"])
        @test isnan(b["ml_loglik"])
        @test b["fit_failed"] === true
        @test occursin("failed fit", b["fit_failed_message"])
        @test isfinite(b["coefficients"][1])        # estimates still ship; only the loglik is withheld
    end
    # runaway positive value on a non-converged fit
    @test isnan(flat(_failed_copy(good, 3.6e76))["loglik"])
    # non-converged but plausible loglik: not failed, value unchanged
    p = flat(_failed_copy(good, good.loglik; converged = false))
    @test p["loglik"] == good.loglik && p["fit_failed"] === false
end
