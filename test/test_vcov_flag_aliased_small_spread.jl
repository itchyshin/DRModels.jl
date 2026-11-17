# The boundary flag of the Hessian guard (DRModels.jl#1383): three cases that a
# curvature-only rule gets wrong.
#
#   1. An aliased design column plus a covariate in large units. The covariance of an
#      aliased fit is rewritten after the guard (the aliased rows and columns are set to
#      NaN), and the flag used to be left as the matrix alone had set it, so identified
#      intercepts got NaN standard errors.
#   2. A very small spread (sigma, CV, 1 / precision). The curvature of a location
#      intercept is n / sigma^2 and that of a log scale is 2n, so a reference taken from
#      the other coordinates flagged a well identified log scale. Each dimensionless
#      coordinate is now judged against the number of observations.
#   3. A coefficient on a covariate in a log or logit block is judged per unit of its own
#      design column, so a flat slope (sigma slope of a negative binomial fitted to
#      Poisson data) is flagged instead of getting a huge finite standard error.

using DRModels
using Test
using Random
using LinearAlgebra
using Logging
using StableRNGs
import Distributions

const _DVF = DRModels

_quiet(f) = Logging.with_logger(f, Logging.NullLogger())
_bnd(fit) = bridge_diagnostics(fit).boundary

@testset "flag: aliased design, small spread, per-column curvature" begin

    @testset "unit: curvature per unit of the design column" begin
        # coordinates: 1 intercept (dimensionless), 2 slope in a log block, 3 log size
        unit = [false, true, false]
        for cn in (1e-10, 1.0, 1e10)            # squared norm of the slope's column
            n = 300.0
            # mean per-observation weight 1e-16: flat, whatever the units of the column
            H = Diagonal([n, 1e-16 * cn, 2n])
            @test _DVF._flat_units_free(H, unit, 300; colnorm2 = [NaN, cn, NaN]) == [2]
            # mean per-observation weight 1e-2: informative, whatever the units
            H = Diagonal([n, 1e-2 * cn, 2n])
            @test _DVF._flat_units_free(H, unit, 300; colnorm2 = [NaN, cn, NaN]) == Int[]
        end
        # without a recorded column the old rule stands: flagged only at exactly zero
        @test _DVF._flat_units_free(Diagonal([300.0, 1e-16, 600.0]), unit, 300) == Int[]
        @test _DVF._flat_units_free(Diagonal([300.0, 0.0, 600.0]), unit, 300) == [2]
        # dimensionless coordinates are judged against the number of observations, not
        # against the largest curvature: a location intercept with curvature 3e10 does not
        # make a log scale with curvature 600 look flat
        unit2 = [false, false]
        @test _DVF._flat_units_free(Diagonal([3.0e10, 600.0]), unit2, 300) == Int[]
        @test _DVF._flat_units_free(Diagonal([3.0e10, 1e-9]), unit2, 300) == [2]
    end

    @testset "unit: a record is matched ignoring the aliased rows and columns" begin
        a = [1.0 0.3 0.5; 0.3 2.0 0.1; 0.5 0.1 2.0]      # what the guard stored
        b = copy(a); b[2, :] .= NaN; b[:, 2] .= NaN       # after the aliased coordinate 2 was set to NaN
        @test !isequal(a, b)
        @test _DVF._same_except(a, b, [2])
        @test !_DVF._same_except(a, b, Int[])
        @test !_DVF._same_except(a, b, [3])                # row 3 of b is not NaN
        c = copy(b); c[1, 1] = 9.0
        @test !_DVF._same_except(a, c, [2])
        @test _DVF._same_except(a, a, Int[])
    end

    # y ~ x + x2 with x2 = 3x (mu block) or sigma ~ x + x2 (sigma block), x in units `u`
    function alias_fit(u; block = :mu)
        rng = StableRNG(11)
        n = 240
        x0 = randn(rng, n)
        y = 1 .+ 0.5 .* x0 .+ 0.3 .* randn(rng, n)
        x = u .* x0
        d = (; y, x, x2 = 3 .* x)
        f = block === :mu ? bf(@formula(y ~ x + x2), @formula(sigma ~ 1)) :
            bf(@formula(y ~ x), @formula(sigma ~ x + x2))
        return _quiet(() -> drm(f, Gaussian(); data = d))
    end

    @testset "aliased design: only the aliased coefficient is flagged, at every unit" begin
        # mu block: coordinates (Intercept), x, x2 | log sigma; x2 is aliased (index 3)
        # sigma block: (Intercept), x | (Intercept), x, x2; x2 is aliased (index 5)
        for (block, alias, ident) in ((:mu, 3, [1, 4]), (:sigma, 5, [1, 3]))
            base = alias_fit(1.0; block)
            se0 = stderror(base)
            @test _bnd(base) == [alias]
            for u in (1.0, 1e5, 1e-5)
                fit = alias_fit(u; block)
                se = stderror(fit)
                V = vcov(fit)
                # the flag is the aliased coordinate and nothing else
                @test _bnd(fit) == [alias]
                @test isnan(V[alias, alias]) && all(isnan, V[alias, :]) && all(isnan, V[:, alias])
                keep = setdiff(1:size(V, 1), alias)
                @test all(isfinite, V[keep, keep])
                @test issymmetric(V[keep, keep])
                # the aliased coefficient keeps the Inf standard error that main gives. drmTMB
                # 0.8.1 measured on the same data: pdHess FALSE and NA for every coefficient
                # (whole table), at covariate units 1, 1e5 and 1e-5; it never gives Inf here.
                @test se[alias] == Inf
                @test all(isfinite, se[keep])
                # identified intercepts do not depend on the units of the covariate
                @test se[ident] ≈ se0[ident] rtol = 1e-3
                @test !is_converged(fit)
            end
        end
    end

    # Spread families. The data are m(x) + spread * noise with one fixed noise draw, so
    # the fit at every spread is the fit at spread 1 up to the known rescaling.
    n = 300
    rng0 = StableRNG(5)
    x0 = randn(rng0, n)
    e0 = randn(rng0, n)
    t0 = rand(StableRNG(9), Distributions.TDist(5), n)
    g0 = repeat(1:15, inner = 20)
    b0 = randn(StableRNG(7), 15)

    function check_spread(fit; unit_scale = 1.0, ref = nothing, nloc = 2, rtol = 5e-3)
        se = stderror(fit)
        @test isempty(_bnd(fit))
        @test all(isfinite, se) && all(isfinite, vcov(fit))
        if ref !== nothing
            @test se[1:nloc] ./ unit_scale ≈ ref[1:nloc] rtol = rtol
            @test se[nloc+1:end] ≈ ref[nloc+1:end] rtol = rtol
        end
        return se
    end

    @testset "spread: Gaussian and Student, sigma in 1e-6 .. 1e4" begin
        ref = nothing
        for s in (1.0, 1e-6, 1e-4, 1e4)
            fit = _quiet(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian();
                                   data = (; y = 1 .+ 0.5 .* x0 .+ s .* e0, x = x0)))
            se = check_spread(fit; unit_scale = s, ref, nloc = 2)
            ref === nothing && (ref = se)
        end
        ref = nothing
        for s in (1.0, 1e-6, 1e-4, 1e4)
            fit = _quiet(() -> drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian();
                                   data = (; y = 1 .+ 0.5 .* x0 .+ s .* (0.5 .* b0[g0] .+ e0),
                                             x = x0, g = string.(g0))))
            se = check_spread(fit; unit_scale = s, ref, nloc = 2)
            ref === nothing && (ref = se)
        end
        ref = nothing
        for s in (1.0, 1e-6, 1e-4, 1e4)
            fit = _quiet(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Student();
                                   data = (; y = 1 .+ 0.5 .* x0 .+ s .* t0, x = x0)))
            se = check_spread(fit; unit_scale = s, ref, nloc = 2)
            ref === nothing && (ref = se)
        end
    end

    @testset "spread: LogNormal and Gamma, sdlog and CV from 1e-6 to 3" begin
        ref = nothing
        for s in (1.0, 1e-6, 1e-4, 1e-2, 3.0)
            fit = _quiet(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), DRModels.LogNormal();
                                   data = (; y = exp.(1 .+ 0.3 .* x0 .+ s .* e0), x = x0)))
            se = check_spread(fit; unit_scale = s, ref, nloc = 2)
            ref === nothing && (ref = se)
        end
        # Gamma: not an exact rescaling, but the standard errors of the log mean per unit
        # of CV, and of the log CV, are the same for every small CV
        ses = Vector{Float64}[]
        for s in (1e-6, 1e-4, 1e-2)
            r = StableRNG(21)
            y = [rand(r, Distributions.Gamma(1 / s^2, exp(1 + 0.3 * x0[i]) * s^2)) for i in 1:n]
            fit = _quiet(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), DRModels.Gamma(); data = (; y, x = x0)))
            se = check_spread(fit)
            push!(ses, vcat(se[1:2] ./ s, se[3]))
        end
        @test ses[1] ≈ ses[2] rtol = 1e-2
        @test ses[2] ≈ ses[3] rtol = 1e-2
    end

    @testset "spread: Beta precision, NB2 size, Tweedie dispersion" begin
        ses = Vector{Float64}[]
        for ph in (1e2, 1e4, 1e6)
            r = StableRNG(31)
            mu = 1 ./ (1 .+ exp.(-(0.2 .+ 0.5 .* x0)))
            y = [rand(r, Distributions.Beta(mu[i] * ph, (1 - mu[i]) * ph)) for i in 1:n]
            fit = _quiet(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), DRModels.Beta(); data = (; y, x = x0)))
            se = check_spread(fit)
            push!(ses, vcat(se[1:2] .* sqrt(ph), se[3]))
        end
        # the three precisions are different draws; the large two agree closely
        @test ses[2] ≈ ses[3] rtol = 2e-2
        for sz in (0.5, 2.0, 10.0, 50.0)
            r = StableRNG(41)
            mu = exp.(1.5 .+ 0.3 .* x0)
            y = Float64[rand(r, Distributions.NegativeBinomial(sz, sz / (sz + mu[i]))) for i in 1:n]
            check_spread(_quiet(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), NegBinomial2(); data = (; y, x = x0))))
        end
        function rtw(r, mu, phi, p)
            lam = mu^(2 - p) / (phi * (2 - p)); gam = phi * (p - 1) * mu^(p - 1); sh = (2 - p) / (p - 1)
            N = rand(r, Distributions.Poisson(lam))
            return N == 0 ? 0.0 : rand(r, Distributions.Gamma(N * sh, gam))
        end
        for ph in (1.0, 10.0)
            r = StableRNG(51)
            y = [rtw(r, exp(0.7 + 0.3 * x0[i]), ph, 1.5) for i in 1:n]
            check_spread(_quiet(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Tweedie();
                                          data = (; y, x = x0))))
        end
    end

    # Gaussian y ~ x + (1 | g) with the group means of the residual exactly zero: the
    # random-effect SD is at zero (index 4), at every spread s and covariate unit u.
    function re_zero_fit(s, u)
        G = 20; m = 8; nn = G * m
        gg = repeat(1:G, inner = m)
        xr = randn(StableRNG(5), nn)
        er = 0.5 .* randn(StableRNG(6), nn)
        for i in 1:G
            r = gg .== i
            er[r] .-= sum(er[r]) / m
        end
        return _quiet(() -> drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian();
                                data = (; y = 0.3 .+ 0.5 .* xr .+ s .* er, x = u .* xr, g = gg)))
    end

    @testset "true boundaries stay flagged at every spread and unit" begin
        # (spread, covariate unit): the spread and the unit are varied one at a time
        for (s, u) in ((1e-6, 1.0), (1.0, 1.0), (1e4, 1.0), (1.0, 1e-5), (1.0, 1e5))
            fit = re_zero_fit(s, u)
            @test _bnd(fit) == [4]
            se = stderror(fit)
            @test isnan(se[4]) && all(isfinite, se[1:3])
        end

        # NB2 fitted to Poisson data: the log size, and its slope, run off. The intercept
        # and the slope of the log size are both flagged, at every unit of the covariate
        for u in (1e-5, 1.0, 1e5)
            r = StableRNG(105)
            nb = 3000
            xs = randn(r, nb)
            yp = Float64.([rand(r, Distributions.Poisson(exp(1.0 + 0.3 * a))) for a in xs])
            f1 = _quiet(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), NegBinomial2(); data = (; y = yp, x = u .* xs)))
            @test _bnd(f1) == [3]
            @test isnan(stderror(f1)[3]) && all(isfinite, stderror(f1)[1:2])
            f2 = _quiet(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ x)), NegBinomial2(); data = (; y = yp, x = u .* xs)))
            @test _bnd(f2) == [3, 4]
            @test all(isnan, stderror(f2)[3:4]) && all(isfinite, stderror(f2)[1:2])
        end

        # ZeroOneBeta without exact ones: the zero-one-inflation coi runs to the logit clamp
        for u in (1e-5, 1.0, 1e5)
            r = StableRNG(20261019)
            nz = 800
            xz = randn(r, nz)
            yz = Vector{Float64}(undef, nz)
            for i in 1:nz
                mu = 1 / (1 + exp(-(log(0.4 / 0.6) + 0.5 * xz[i])))
                yz[i] = rand(r) < 0.2 ? 0.0 : rand(r, Distributions.Beta(mu * 10, (1 - mu) * 10))
            end
            fit = _quiet(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)),
                                   ZeroOneBeta(); data = (; y = yz, x = u .* xz)))
            @test _bnd(fit) == [5]
            @test isnan(stderror(fit)[5]) && all(isfinite, stderror(fit)[1:4])
        end
    end

    @testset "convention: Inf for separation and aliasing, NaN for a flagged boundary" begin
        xs = collect(range(-2, 2; length = 40))
        sep = _quiet(() -> drm(bf(@formula(y ~ x)), Binomial(); data = (; y = Float64.(xs .> 0), x = xs)))
        @test all(==(Inf), stderror(sep))
        @test stderror(alias_fit(1.0))[3] == Inf
        @test isnan(stderror(re_zero_fit(1.0, 1.0))[4])
    end

    @testset "unit: the cut for a dimensionless coordinate is 3e-8 times the number of observations" begin
        unit = [false, false, false]
        # Curvature per observation d/n between 1e-7 and 1e-3 is informative, whatever n is.
        # This rules out a cut that is larger (for example 3e-4 times n), or a reference that
        # grows faster than n (n squared) or slower than n (a constant or sqrt(n)).
        for n in (30, 240, 100_000), q in (1e-7, 1e-6, 1e-5, 1e-3)
            H = Diagonal([n, q * n, 2n])
            @test _DVF._flat_units_free(H, unit, n) == Int[]
        end
        # d/n at or below 1e-8 is flat, again at every n
        for n in (30, 240, 100_000), q in (1e-8, 1e-10, 0.0)
            H = Diagonal([n, q * n, 2n])
            @test _DVF._flat_units_free(H, unit, n) == [2]
        end
    end

    @testset "unit: the final flag is the intersection of the units-free flag and the guard's own flag" begin
        # Gaussian y ~ x with sigma ~ 1: coordinates (Intercept), x, log sigma. Only the log
        # sigma intercept (3) is dimensionless; the other two carry units.
        rng = StableRNG(2406)
        n = 120
        x = randn(rng, n)
        y = 1 .+ 0.5 .* x .+ 0.3 .* randn(rng, n)
        fit = _quiet(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x)))
        V = copy(fit.vcov)
        @test _bnd(fit) == Int[]
        rec(H, flat) = (H = Matrix{Float64}(H), V = copy(V), flat = flat, context = "test", rcond = 1.0, warn = false)
        # The curvature of coordinate 3 is 1e-9, below 3e-8 times n, so the units-free rule
        # alone would flag it. The guard flagged nothing, so the flag must stay empty.
        H = Diagonal([1e3, 1e3, 1e-9])
        @test _DVF._flat_units_free(H, [true, true, false], n) == [3]
        recs = _DVF._GuardRecords()
        push!(recs.items, rec(H, Int[]))
        out = _DVF._reconcile_guard_flags(fit, recs)
        @test _bnd(out) == Int[]
        @test isequal(out.vcov, V)
        # The guard flagged coordinate 3 and the curvature is large: the flag is removed
        # and the covariance is the inverse of the whole matrix.
        H = Diagonal([1e3, 1e3, 1e3])
        recs = _DVF._GuardRecords()
        push!(recs.items, rec(H, [3]))
        out = _DVF._reconcile_guard_flags(fit, recs)
        @test _bnd(out) == Int[]
        @test all(isfinite, out.vcov)
        @test out.vcov ≈ inv(H) rtol = 1e-10
    end
    @testset "unit: a random-effect log SD at or below -8 keeps the guard's flag (P48 j)" begin
        rng = StableRNG(2407)
        G = 30; m = 6; n = G * m
        gi = repeat(1:G, inner = m)
        g = string.(gi); x = randn(rng, n)
        u = 0.8 .* randn(rng, G)
        # Poisson counts without and with a group effect: the SD runs to zero, or is interior.
        y0 = [rand(rng, Distributions.Poisson(exp(0.2 + 0.3 * xi))) for xi in x]
        y1 = [rand(rng, Distributions.Poisson(exp(0.2 + 0.3 * x[i] + u[gi[i]]))) for i in 1:n]
        f(y) = _quiet(() -> drm(bf(@formula(y ~ x + (1 | g))), DRModels.Poisson(); data = (; y, x, g)))
        fit0 = f(y0); fit1 = f(y1)
        k = length(fit0.theta)
        for fit in (fit0, fit1)
            @test _DVF._re_logsd_coords(fit) == [k]
        end
        @test fit0.theta[k] <= -8
        @test fit1.theta[k] > -8
        # A curvature as large as the others would not be flagged by the units-free rule;
        # the guard flagged the SD, and the floor keeps that flag for the SD at zero only.
        H = Diagonal(fill(1e3, k))
        mkrec(fit, flat) = (H = Matrix{Float64}(H), V = copy(fit.vcov), flat = flat, context = "test", rcond = 1.0, warn = false)
        for (fit, kept) in ((fit0, true), (fit1, false))
            recs = _DVF._GuardRecords()
            push!(recs.items, mkrec(fit, [k]))
            @test (_bnd(_DVF._reconcile_guard_flags(fit, recs)) == [k]) == kept
        end
        # The floor never adds a flag the guard did not set.
        recs = _DVF._GuardRecords()
        push!(recs.items, mkrec(fit0, Int[]))
        @test _bnd(_DVF._reconcile_guard_flags(fit0, recs)) == _bnd(fit0)   # unchanged: nothing added
        # A Gaussian fit has none: its SD carries the units of the response.
        rg = StableRNG(2408)
        yg = 1e-6 .* (0.5 .* randn(rg, G)[gi] .+ randn(rg, n))
        fitg = _quiet(() -> drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = (; y = yg, x, g)))
        @test fitg.theta[end] <= -8
        @test _DVF._re_logsd_coords(fitg) == Int[]
    end
end
