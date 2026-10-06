# check_drm(fit): post-fit convergence / identifiability diagnostics,
# mirroring drmTMB's check_drm().
using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import ForwardDiff

@testset "check_drm() — convergence / identifiability report" begin
    @testset "well-identified model → ok" begin
        Random.seed!(20260601)
        n = 500; x = randn(n); z = randn(n)
        y = 0.4 .+ 0.8 .* x .- 0.5 .* z .+ 0.5 .* randn(n)
        fit = drm(bf(@formula(y ~ x + z), @formula(sigma ~ 1)), Gaussian(); data = (; y, x, z))
        r = check_drm(fit)
        @test r.converged
        @test r.vcov_posdef
        @test r.max_abs_grad < 1e-2          # near a clean interior optimum
        @test r.min_eigval > 0
        @test isfinite(r.cond)
        @test r.ok
    end

    @testset "report has the documented fields and types" begin
        Random.seed!(20260602)
        n = 300; x = randn(n)
        y = 1.0 .+ 0.5 .* x .+ 0.6 .* randn(n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = (; y, x))
        r = check_drm(fit)
        @test r isa NamedTuple
        # `penalized_map` added by A4c: a penalized (MAP) fit reports credible-
        # interval-shaped SEs, and its stored UNPENALIZED gradient is non-zero at
        # the optimum by construction, so `ok` is not scored against it.
        # `grad_source` added by the AD-safety fix: `max_abs_grad` alone cannot say
        # whether the number is an exact gradient, a finite-difference stand-in, or
        # absent, and the last of those silently satisfies `ok`.
        @test keys(r) == (:converged, :max_abs_grad, :grad_source, :vcov_complete,
                          :vcov_posdef, :min_eigval, :cond, :penalized_map,
                          :variance_boundary, :temporal_boundary, :ok)
        # `variance_boundary` (#724/#697): `nothing` unless a Gaussian fit has a grouped /
        # structured random effect and a homoscedastic residual
        @test r.variance_boundary === nothing
        # `temporal_boundary` (drmTMB's temporal_boundary rules): `nothing` without a
        # temporal() term; see test/test_temporal_boundary.jl
        @test r.temporal_boundary === nothing
        @test r.vcov_complete isa Bool
        @test r.vcov_complete            # an ordinary fixed-effect fit has a full vcov
        @test r.converged isa Bool
        @test r.penalized_map isa Bool
        @test !r.penalized_map          # an ordinary ML fit is not penalized
        @test r.vcov_posdef isa Bool
        @test r.ok isa Bool
        @test r.max_abs_grad isa Float64
        @test r.grad_source isa Symbol
        # A dual-safe fixed-effect objective: ForwardDiff runs, so the source is
        # `:forward` and the magnitude is a genuine gradient norm.
        @test r.grad_source === :forward
        @test isfinite(r.max_abs_grad)
    end

    @testset "stored objective gradient is used when available" begin
        blocks = [:mu => 1:2]
        coefnames = [:mu => ["(Intercept)", "x"]]
        theta = [0.0, 0.0]
        V = Matrix{Float64}(I, 2, 2)
        empty = Dict{Symbol,Vector{Float64}}()
        nll(_) = error("check_drm should use the stored gradient")
        nllgrad!(g, _) = (fill!(g, 0.0); g)
        base = DRModels.DrmFit(Gaussian(), blocks, coefnames, theta, V, -1.0, 2, true,
                          empty, empty, empty)
        fit = DRModels._withnll(base, nll, nllgrad!)

        r = check_drm(fit)
        @test r.max_abs_grad == 0.0
        @test r.grad_source === :stored
        @test r.ok
    end

    @testset "a partial (NaN) covariance is REPORTED, not raised" begin
        # Regression. `check_drm` used to throw `ArgumentError: matrix contains Infs
        # or NaNs` from `isposdef`/`eigvals` for any fit whose vcov held a NaN —
        # historically the NORMAL state of the sparse phylo route, which computed
        # only the fixed-effect block. #556 fixed that route (its vcov is complete
        # now), so this test synthesises the partial-vcov fixture instead: the
        # check_drm contract — REPORT trouble, never raise — is route-independent.
        rng = MersenneTwister(5); G = 16; m = 5
        phy = random_balanced_tree(G; branch_length = 0.3)
        C = sigma_phy_dense(phy; σ²_phy = 1.0)
        d = sqrt.(diag(C)); K = C ./ (d * d')
        n = G * m; species = repeat(1:G, inner = m); x = randn(rng, n)
        u = 0.9 .* (cholesky(Symmetric(K)).L * randn(rng, G))
        y = 0.2 .+ 0.5 .* x .+ u[species] .+ 0.4 .* randn(rng, n)
        fit0 = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                   Gaussian(); data = (; y, x, species), tree = phy)

        # #556 closed the NATURAL source of a NaN vcov: the sparse phylo route
        # now fills its variance block from the profiled curvature, so a healthy
        # fit here has a COMPLETE covariance — assert that as the positive half.
        @test !any(isnan, fit0.vcov)

        # The check_drm contract under a partial vcov still needs testing, so
        # poison a copy the way the old route used to leave it: mean block
        # finite, variance block NaN.
        Vnan = copy(fit0.vcov)
        Vnan[end-1:end, :] .= NaN
        Vnan[:, end-1:end] .= NaN
        fit = DRModels.DrmFit(fit0.family, fit0.blocks, fit0.coefnames, fit0.theta,
                         Vnan, fit0.loglik, fit0.nobs, fit0.converged,
                         fit0.means, fit0.obs, fit0.scales)
        @test any(isnan, fit.vcov)

        r = check_drm(fit)                      # must not throw
        @test r isa NamedTuple
        @test !r.vcov_complete
        @test !r.vcov_posdef
        @test isnan(r.min_eigval)
        @test r.cond == Inf
        @test !r.ok                             # honestly not-ok, rather than an exception
        @test r.converged                       # ...but the FIT itself converged
    end

    # ---------------------------------------------------------------------------
    # Regression: routes whose OBJECTIVE is not dual-number safe.
    #
    # `_check_max_abs_grad` ended in a bare `maximum(abs, ForwardDiff.gradient(...))`
    # with no guard. Four shipping routes store a bare objective (no gradient
    # callback) that is exact on Float64 but not dual-number safe, so that line
    # THREW and `check_drm` crashed on the very fits it exists to report on --
    # measured 2026-09-05 at origin/main 109b6421c, two distinct exception types:
    #
    #   bivariate q=2 structured and the sparse two-structured Gaussian mean route
    #     (both factorise a sparse matrix through CHOLMOD):
    #     TypeError: in Sparse, in Tv, expected Tv<:Union{Float64, ComplexF64},
    #     got Type{ForwardDiff.Dual{...}}
    #   sparse LSS under REML, single- and multi-component (Float64 work arrays;
    #     both store `reml ? nothing : nllgrad!`, so only the REML arm gets here):
    #     MethodError: no method matching Float64(::ForwardDiff.Dual{...})
    #
    # A diagnostic must REPORT that, not raise it. The assertion below is
    # deliberately in two parts: check_drm must RETURN, and the magnitude it
    # returns must be either a real finite number or an EXPLICITLY flagged
    # unavailable -- a bare NaN would slip through `ok`'s `isnan(mag)` disjunct and
    # make an untested route look identical to a clean one.
    function _assert_reports_not_raises(fit)
        r = check_drm(fit)                                  # must not throw
        @test r isa NamedTuple
        @test r.grad_source in (:finite, :unavailable)       # AD did not survive here
        if r.grad_source === :finite
            @test isfinite(r.max_abs_grad)                   # a usable number...
            @test r.max_abs_grad >= 0
        else
            @test isnan(r.max_abs_grad)                      # ...or an honest NaN,
        end                                                  # never an unlabelled one
        return r
    end

    @testset "AD-hostile route: bivariate q=2 structured (CHOLMOD)" begin
        rng = MersenneTwister(20260905)
        G = 12; nrep = 3
        idx = collect(1:G)
        K = [0.55 ^ abs(i - j) for i in idx, j in idx] + 1e-6I
        beta = [0.20 -0.12; 0.18 0.10]
        Lam = Matrix(Symmetric([0.20 0.05; 0.05 0.17]))
        residual_cov = Matrix(Symmetric([0.10 0.025; 0.025 0.14]))
        Q = sparse(Matrix(inv(cholesky(Symmetric(K)))))
        P = DRModels.prior_precision(Q, inv(Lam))
        u = cholesky(Symmetric(P)).UP \ randn(rng, size(P, 1))
        group = repeat(1:G, inner = nrep); n = length(group)
        x = randn(rng, n); X = hcat(ones(n), x)
        L = cholesky(Symmetric(residual_cov)).L
        Y = zeros(n, 2)
        for i in 1:n
            base = 2 * (group[i] - 1)
            Y[i, 1] = sum(X[i, :] .* beta[:, 1]) + u[base + 1]
            Y[i, 2] = sum(X[i, :] .* beta[:, 2]) + u[base + 2]
            Y[i, :] .+= L * randn(rng, 2)
        end
        dat = (; y1 = Y[:, 1], y2 = Y[:, 2], x = x, grp = ["g$(i)" for i in group])
        fit = drm(bf(mu1 = @formula(y1 ~ x + relmat(1 | grp)),
                     mu2 = @formula(y2 ~ x + relmat(1 | grp)),
                     sigma1 = @formula(sigma1 ~ 1),
                     sigma2 = @formula(sigma2 ~ 1),
                     rho12 = @formula(rho12 ~ 1)),
                  Gaussian(); data = dat, K = K, g_tol = 2e-4)

        # This route reaches the ForwardDiff line: a bare objective, no callback.
        @test fit.nll !== nothing
        @test fit.nllgrad === nothing
        @test !(fit.nll isa DRModels.LocScaleObjective)
        # ...and ForwardDiff genuinely cannot be run through it (the RED condition
        # this regression covers), while the Float64 objective is perfectly fine.
        @test_throws Exception ForwardDiff.gradient(fit.nll, fit.theta)
        @test isfinite(fit.nll(fit.theta))

        r = _assert_reports_not_raises(fit)
        # The finite-difference fallback is what keeps `ok`'s gradient criterion
        # live on this route, so it must be the branch taken here.
        @test r.grad_source === :finite
        # And the diagnostic must SAY the number is not an exact gradient.
        @test_logs (:warn,) match_mode = :any check_drm(fit)
    end

    @testset "AD-hostile route: sparse two-structured Gaussian mean (CHOLMOD)" begin
        Random.seed!(424242)
        G = 20
        _corr(M) = (d = sqrt.(diag(M)); M ./ (d * d'))
        function _banded_corr(g; bw = 2, seed = 0)
            Random.seed!(seed)
            Om = Matrix(2.0I(g))
            for i in 1:g, j in (i+1):min(i + bw, g)
                v = 0.3 / (j - i); Om[i, j] = v; Om[j, i] = v
            end
            Om += (0.1 + g * eps()) * I
            return _corr(inv(Symmetric(Om)))
        end
        C1 = _banded_corr(G; bw = 3, seed = 11)
        C2 = _banded_corr(G; bw = 2, seed = 22)
        m = 5; n = G * m
        species = repeat(1:G, inner = m); id = repeat(1:G, inner = m)
        x = randn(n)
        a1 = 0.8 .* (cholesky(Symmetric(C1)).L * randn(G))
        a2 = 0.5 .* (cholesky(Symmetric(C2)).L * randn(G))
        y = 0.3 .+ 0.5 .* x .+ a1[species] .+ a2[id] .+ 0.4 .* randn(n)
        gidx1, G1 = DRModels._group_index(species)
        gidx2, G2 = DRModels._group_index(id)
        fit = DRModels._fit_two_structured_gaussian_sparse(Gaussian(), y, hcat(ones(n), x),
            gidx1, G1, C1, gidx2, G2, C2, ["(Intercept)", "x"], :species, :id, 1e-7)

        @test fit.nllgrad === nothing
        @test_throws Exception ForwardDiff.gradient(fit.nll, fit.theta)
        r = _assert_reports_not_raises(fit)
        @test r.grad_source === :finite
    end

    @testset "AD-hostile route: sparse LSS under REML (Float64 work arrays)" begin
        # The ML arm of the SAME fitter stores `nllgrad!` and never reaches the
        # ForwardDiff line; only REML stores `nothing` there
        # (`gaussian_sparse_lss.jl:285`). Both are asserted, so the control and the
        # regression sit side by side on one fixture.
        Random.seed!(55801)
        p = 16; m = 4; n = p * m
        phy = random_balanced_tree(p; branch_length = 0.25)
        K = DRModels._phylo_correlation(phy)
        LK = cholesky(Symmetric(K)).L
        species = repeat(1:p, inner = m)
        x = randn(n); zg = randn(p); z = zg[species]
        u_a = exp.(0.3 .+ 0.4 .* zg) .* (LK * randn(p))
        y = 0.5 .+ 0.8 .* x .+ u_a[species] .+ exp.(-0.5 .+ 0.3 .* x) .* randn(n)
        data = (; y, x, z, species)
        f = bf(@formula(y ~ x + phylo(1 | species)),
               @formula(sigma ~ x),
               @formula(sd(species, phylogenetic) ~ z))

        ml = drm(f, Gaussian(); data = data, tree = phy, algorithm = :sparse,
                 method = :ML, g_tol = 1e-8)
        @test ml.nllgrad !== nothing                 # ML: stored callback, AD untouched
        @test check_drm(ml).grad_source === :stored

        rm = drm(f, Gaussian(); data = data, tree = phy, algorithm = :sparse,
                 method = :REML, g_tol = 1e-8)
        @test rm.nllgrad === nothing                 # REML: bare objective, AD path
        @test_throws Exception ForwardDiff.gradient(rm.nll, rm.theta)
        r = _assert_reports_not_raises(rm)
        @test r.grad_source === :finite
    end

    @testset "AD-hostile route: multi-component sparse LSS under REML" begin
        Random.seed!(913)
        p = 10; m = 5; n = p * m
        phy = random_balanced_tree(p; branch_length = 0.25)
        K = DRModels._phylo_correlation(phy)
        LK = cholesky(Symmetric(K)).L
        species = repeat(1:p, inner = m); g = repeat(1:p, inner = m)
        x = randn(n); zg = randn(p); z = zg[species]
        ua = exp.(0.2 .+ 0.3 .* zg) .* (LK * randn(p))
        ub = exp.(0.1 .+ 0.2 .* zg) .* randn(p)
        y = 0.5 .+ 0.7 .* x .+ ua[species] .+ ub[g] .+ exp.(-0.6 .+ 0.2 .* x) .* randn(n)
        f = bf(@formula(y ~ x + phylo(1 | species) + (1 | g)),
               @formula(sigma ~ x),
               @formula(sd(species, phylogenetic) ~ z),
               @formula(sd(g) ~ z))
        fit = drm(f, Gaussian(); data = (; y, x, z, species, g), tree = phy,
                  algorithm = :sparse, method = :REML)

        @test fit.nllgrad === nothing
        @test_throws Exception ForwardDiff.gradient(fit.nll, fit.theta)
        r = _assert_reports_not_raises(fit)
        @test r.grad_source === :finite
    end

    @testset "an UNCHECKABLE gradient is labelled, not silently NaN" begin
        # The distinction this whole fix exists for. Both fits below report
        # `max_abs_grad = NaN` and both pass `ok` through the `isnan(mag)` disjunct;
        # only `grad_source` says that one of them was never gradient-checked.
        blocks = [:mu => 1:2]
        coefnames = [:mu => ["(Intercept)", "x"]]
        theta = [0.0, 0.0]
        V = Matrix{Float64}(I, 2, 2)
        empty = Dict{Symbol,Vector{Float64}}()
        base = DRModels.DrmFit(Gaussian(), blocks, coefnames, theta, V, -1.0, 2, true,
                          empty, empty, empty)

        # (a) No objective at all: nothing to differentiate. This is the ORIGINAL
        # meaning of a NaN magnitude and it must keep reporting `:none`.
        none = check_drm(base)
        @test base.nll === nothing
        @test isnan(none.max_abs_grad)
        @test none.grad_source === :none
        @test none.ok                        # unchanged behaviour on this path

        # (b) An objective that is stored but cannot be evaluated at all, so even
        # the finite-difference fallback has nothing to work with.
        hostile(_) = error("objective is not evaluable on Float64 either")
        dead = DRModels._withnll(base, hostile)
        r = check_drm(dead)                  # must not throw
        @test isnan(r.max_abs_grad)
        @test r.grad_source === :unavailable # NOT :none -- that is the whole point
        @test r.grad_source !== none.grad_source
        # ...and it must SAY so, or `ok = true` here is indistinguishable from a
        # fit whose stationarity was actually verified.
        @test r.ok
        @test_logs (:warn,) match_mode = :any check_drm(dead)
    end

    @testset "a NON-FINITE stored gradient falls back rather than passing free" begin
        # Same defect class one branch over: a gradient callback that fills NaN on
        # failure (`gaussian_sparse_lss.jl`'s `nllgrad!` does exactly this) used to
        # hand `ok` the same silent free pass. The Float64 objective is fine here,
        # so the finite difference recovers a real magnitude.
        blocks = [:mu => 1:1]
        coefnames = [:mu => ["(Intercept)"]]
        theta = [2.0]
        V = Matrix{Float64}(I, 1, 1)
        empty = Dict{Symbol,Vector{Float64}}()
        base = DRModels.DrmFit(Gaussian(), blocks, coefnames, theta, V, -1.0, 2, true,
                          empty, empty, empty)
        quad(t) = 0.5 * (t[1] - 1.0)^2                  # d/dt = t - 1 = 1.0 at t = 2
        broken!(g, _) = (fill!(g, NaN); g)
        fit = DRModels._withnll(base, quad, broken!)

        r = check_drm(fit)
        @test r.grad_source === :finite
        @test isapprox(r.max_abs_grad, 1.0; atol = 1e-6)
        @test !r.ok                                      # honestly NOT stationary
    end
end
