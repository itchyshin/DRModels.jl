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

@testset "NB2 structured sigma animal: no PosDefException, a fit with converged = false" begin
    raw = readdlm(joinpath(_FX, "nb2_sigma_animal_data.csv"), ','; header = true)[1]
    data = (; y = Float64.(raw[:, 1]), x = Float64.(raw[:, 2]), v = Float64.(raw[:, 3]),
            w0 = Float64.(raw[:, 4]), id = string.(raw[:, 5]))
    A = readdlm(joinpath(_FX, "nb2_sigma_animal_A.csv"), ',')
    ids = unique(data.id)
    f = bf(@formula(y ~ x + v), @formula(sigma ~ 1 + animal(1 | id)), @formula(sd(id, animal, sigma) ~ w0))
    fit_at(d) = drm(f, NegBinomial2(); data = d, A, ids, se = false)
    # the reproducer: rescaling the SD-formula covariate is a pure reparametrisation
    rescaled = merge(data, (; w0 = 2.5 .* data.w0))
    fit = _quiet(() -> fit_at(rescaled))
    @test fit isa DRModels.DrmFit
    @test fit.converged == false
    @test !DRModels.is_converged(fit)
    ll_user = _quiet(() -> loglikelihood(fit))
    @test isnan(ll_user) || ll_user > -1e15                    # never a sentinel
    # the as-generated data still return the same (non-converged, finite) fit as before the change
    fit0 = _quiet(() -> fit_at(data))
    @test fit0 isa DRModels.DrmFit
    @test fit0.converged == false
    @test isapprox(fit0.loglik, -354.53365584985033; atol = 1e-6)
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
    fit = _quiet(() -> drm(bf(@formula(y ~ x + spatial(1 | g)), @formula(sd(g, spatial) ~ z)), Binomial(); data, coords))
    @test fit.converged == false
    @test fit.loglik <= -1e15                                   # the internal sentinel is still stored ...
    v = @test_logs (:warn, r"NaN") match_mode = :any loglikelihood(fit)
    @test isnan(v)                                              # ... but is never what the user reads
    @test isnan(_quiet(() -> loglik(fit)))
    @test isnan(_quiet(() -> aic(fit)))

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
