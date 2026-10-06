# test_q2_structured_vcov.jl -- standard errors on the bivariate q=2 structured
# route (leaf-jl-q2-vcov).
#
# The route used to write an all-NaN placeholder into `fit.vcov`, which
# `stderror` then laundered into `Inf` via `_boundary_se` (src/inference.jl) --
# the package's signal for "at a variance boundary, direction unidentified".
# A user could not tell "never computed" from "genuinely flat". The route now
# second-differences the marginal ML NLL it already builds and passes the result
# through the same `_vcov_from_hessian` guard the dense ML bivariate sibling
# uses (src/gaussian_bivariate.jl, `context = "bivariate Gaussian"`).
#
# Deliberately a NEW file rather than assertions bolted onto
# test_reml_q2_structured.jl: that file is REML-scoped by name and header, and
# these cover ML too.

using DRModels
using Test, LinearAlgebra, Random, Statistics, SparseArrays

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

# Same construction as test_reml_q2_structured.jl's `_q2_reml_known_cov_fixture`,
# kept local so the two files stay independent.
function _q2v_sim(K, beta, Lambda, residual_cov; nrep, rng)
    G = size(K, 1)
    Q = sparse(Matrix(inv(cholesky(Symmetric(K)))))
    P = DRModels.prior_precision(Q, inv(Lambda))
    F = cholesky(Symmetric(P))
    u = F.UP \ randn(rng, size(P, 1))
    group = repeat(1:G, inner = nrep)
    n = length(group)
    x = randn(rng, n)
    X = hcat(ones(n), x)
    L = cholesky(Symmetric(residual_cov)).L
    Y = zeros(n, 2)
    for i in 1:n
        base = 2 * (group[i] - 1)
        Y[i, 1] = sum(X[i, :] .* beta[:, 1]) + u[base + 1]
        Y[i, 2] = sum(X[i, :] .* beta[:, 2]) + u[base + 2]
        Y[i, :] .+= L * randn(rng, 2)
    end
    return (; Y, X, group)
end

function _q2v_levelled_fit(provider::Symbol, method::Symbol)
    rng = MersenneTwister(20260824)
    G = 14
    idx = collect(1:G)
    K = [0.55 ^ abs(i - j) for i in idx, j in idx] + 1e-6I
    beta = [0.20 -0.15; 0.25 0.10]
    Lambda = Matrix(Symmetric([0.20 0.05; 0.05 0.17]))
    residual_cov = Matrix(Symmetric([0.10 0.025; 0.025 0.14]))
    sim = _q2v_sim(K, beta, Lambda, residual_cov; nrep = 4, rng = rng)
    dat = (; y1 = sim.Y[:, 1], y2 = sim.Y[:, 2], x = sim.X[:, 2],
             grp = ["g$(i)" for i in sim.group])
    form = provider === :relmat ?
        bf(mu1 = @formula(y1 ~ x + relmat(1 | grp)),
           mu2 = @formula(y2 ~ x + relmat(1 | grp)),
           sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
           rho12 = @formula(rho12 ~ 1)) :
        bf(mu1 = @formula(y1 ~ x + animal(1 | grp)),
           mu2 = @formula(y2 ~ x + animal(1 | grp)),
           sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
           rho12 = @formula(rho12 ~ 1))
    kw = provider === :relmat ? (; K = K) : (; A = K)
    fit = drm(form, Gaussian(); data = dat, kw..., g_tol = 1e-6, method = method)
    return fit, size(sim.Y, 1)
end

function _q2v_phylo_fit(method::Symbol)
    rng = MersenneTwister(20260625)
    phy = DRModels.random_balanced_tree(14; branch_length = 0.2)
    beta = [0.20 -0.15; 0.25 0.10]
    Lambda = Matrix(Symmetric([0.22 0.07; 0.07 0.18]))
    residual_cov = Matrix(Symmetric([0.12 0.04; 0.04 0.16]))
    Q_cond, leaf_pos, _ = DRModels.augmented_tree_precision(phy)
    P = DRModels.prior_precision(Q_cond, inv(Lambda))
    F = cholesky(Symmetric(P))
    u = F.UP \ randn(rng, size(P, 1))
    species = repeat(1:phy.n_leaves, inner = 3)
    n = length(species)
    x = randn(rng, n)
    X = hcat(ones(n), x)
    L = cholesky(Symmetric(residual_cov)).L
    Y = zeros(n, 2)
    for i in 1:n
        base = 2 * (leaf_pos[species[i]] - 1)
        for a in 1:2
            Y[i, a] = sum(X[i, :] .* beta[:, a]) + u[base + a]
        end
        Y[i, :] .+= L * randn(rng, 2)
    end
    dat = (; y1 = Y[:, 1], y2 = Y[:, 2], x,
             species_name = [phy.leaf_names[k] for k in species])
    form = bf(mu1 = @formula(y1 ~ x + phylo(1 | species_name)),
              mu2 = @formula(y2 ~ x + phylo(1 | species_name)),
              sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
              rho12 = @formula(rho12 ~ 1))
    fit = drm(form, Gaussian(); data = dat, tree = phy, g_tol = 1e-4,
              method = method)
    return fit, n
end

# Rebuild (Lambda-hat, D-hat, beta-hat) on the engine's own parameterisation so
# the analytic Schur complement can be evaluated at exactly the fitted point.
function _q2v_pieces(fit)
    b = Dict(fit.blocks)
    th = fit.theta
    beta = hcat(th[b[:mu1]], th[b[:mu2]])
    s1 = exp(th[b[:sigma1]][1])
    s2 = exp(th[b[:sigma2]][1])
    rho = DRModels.RHO_GUARD * tanh(th[b[:rho12]][1])
    D = Matrix(Symmetric([s1^2 rho*s1*s2; rho*s1*s2 s2^2]))
    Lam = DRModels.lc_to_cov(th[b[:phylocov]], 2)
    return (; beta, D, Lam)
end

# ---------------------------------------------------------------------------

@testset "q2 structured vcov: every provider/estimator arm reports usable SEs" begin
    arms = [(:relmat, :ML), (:relmat, :REML), (:animal, :ML), (:animal, :REML)]
    for (prov, meth) in arms
        fit, _ = _q2v_levelled_fit(prov, meth)
        V = vcov(fit)
        @test fit.converged
        # The defect this file exists for: V was fill(NaN, 10, 10).
        @test count(isnan, V) == 0
        @test all(isfinite, V)
        @test isposdef(Symmetric(V))
        se = stderror(fit)
        @test length(se) == length(fit.theta) == 10
        @test all(isfinite, se)
        @test all(>(0), se)
        # stderror mapped the NaN variance to Inf (src/inference.jl `_boundary_se`),
        # so confint returned (-Inf, Inf) on every row. It must not any more.
        ci = confint(fit)
        @test length(ci) == 10
        @test all(r -> isfinite(r.lower) && isfinite(r.upper) && r.lower < r.upper, ci)
    end

    for meth in (:ML, :REML)
        fit, _ = _q2v_phylo_fit(meth)
        V = vcov(fit)
        @test fit.converged
        @test count(isnan, V) == 0
        @test all(isfinite, V)
        @test isposdef(Symmetric(V))
        se = stderror(fit)
        @test length(se) == 10
        @test all(isfinite, se)
        @test all(>(0), se)
        ci = confint(fit)
        # Pin the length first: `all` over an empty collection is vacuously true.
        @test length(ci) == 10
        @test all(r -> isfinite(r.lower) && isfinite(r.upper), ci)
    end
end

@testset "q2 structured vcov: relmat and animal are the same computation" begin
    # Both providers reach `make_coevo_problem_from_covariance` with the same
    # matrix, so the covariance must agree to the last bit, not merely to rtol.
    for meth in (:ML, :REML)
        fr, _ = _q2v_levelled_fit(:relmat, meth)
        fa, _ = _q2v_levelled_fit(:animal, meth)
        @test vcov(fr) == vcov(fa)
        @test stderror(fr) == stderror(fa)
    end
end

@testset "q2 structured vcov: wired through the guard, with the 2n FD step" begin
    # Pins the whole chain, not just the outcome: the stored V must be exactly
    # `_vcov_from_hessian(_finite_hessian(nll, theta; h = _fd_hessian_step(2n)))`.
    # `_vcov_from_hessian` (src/vcov_guard.jl) is the positive-definiteness
    # sentinel -- it eigen-tests against `_VCOV_RTOL` and pseudo-inverts with a
    # warning at a boundary. The step argument is 2n, not n, because two scalar
    # responses enter the objective per row; see the call site's comment.
    fit, n = _q2v_levelled_fit(:relmat, :ML)
    H = DRModels._finite_hessian(fit.nll, fit.theta; h = DRModels._fd_hessian_step(2n))
    expected = DRModels._vcov_from_hessian(H)
    @test vcov(fit) == expected
    # A step of `n` would clamp to the same 1e-4 at this size, so the assertion
    # above cannot by itself distinguish them. Show the two agree HERE, so the
    # test is pinning the wiring rather than silently accepting either step.
    @test DRModels._fd_hessian_step(2n) == DRModels._fd_hessian_step(n) == 1e-4
    @test DRModels._fd_hessian_step(2 * 600) > DRModels._fd_hessian_step(600)
end

# Second-differencing is only "observed information" if the objective really is
# the one that was maximised and theta-hat really is where it was maximised.
# Both are checked here rather than assumed.
function _q2v_fd_gradient(f, x; h = 1e-5)
    g = similar(x)
    for i in eachindex(x)
        e = zeros(length(x)); e[i] = max(h, h * (1 + abs(x[i])))
        g[i] = (f(x .+ e) - f(x .- e)) / (2 * e[i])
    end
    return g
end

@testset "q2 structured vcov: the objective is the fitted one, at the fitted point" begin
    for meth in (:ML, :REML)
        fit, _ = _q2v_levelled_fit(:relmat, meth)
        # The stored closure reproduces the engine's own ML log-likelihood exactly,
        # so its parameterisation is the fitter's and the Hessian is of the right
        # function. Measured 2026-09-05: agrees to rtol 1e-12 on both arms.
        @test isapprox(-fit.nll(fit.theta), fit.ml_loglik; rtol = 1e-10)
    end

    # ML: theta-hat IS a stationary point of this objective, so the curvature there
    # is the observed information. Measured max|grad| = 1.5e-4 against Hessian
    # diagonal entries up to 9.1e2.
    fml, _ = _q2v_levelled_fit(:relmat, :ML)
    @test maximum(abs, _q2v_fd_gradient(fml.nll, fml.theta)) < 1e-3

    # REML: theta-hat is the REML optimum, so the ML gradient there is NOT zero --
    # measured 1.07. This is the documented policy (V is the ML curvature at the
    # REML point, restricted-penalty curvature omitted, as for the q=4 sibling), and
    # it is asserted rather than merely commented so that a future change which
    # silently switched this route to REML curvature would fail here instead of
    # quietly altering every REML interval.
    frm, _ = _q2v_levelled_fit(:relmat, :REML)
    @test maximum(abs, _q2v_fd_gradient(frm.nll, frm.theta)) > 1e-2
    # ... and the two optima really are different points.
    @test frm.theta != fml.theta
end

@testset "q2 structured vcov: beta block matches the EXACT Schur complement" begin
    # Independent oracle, not a finite-difference self-comparison. The beta block
    # of the marginal ML Hessian IS S = X' V^-1 X, which `_q2_profile_and_schur`
    # (src/reml_q2.jl) assembles exactly and by a completely different route.
    for meth in (:ML, :REML)
        fit, n = _q2v_levelled_fit(:relmat, meth)
        p = _q2v_pieces(fit)
        sch = DRModels._q2_profile_and_schur(fit.ranef.prob, fit.ranef.Q_cond,
                                        p.Lam, p.D, p.beta)
        @test sch.ok
        S = Matrix(sch.S)
        nb = size(S, 1)
        @test nb == 4
        H = DRModels._finite_hessian(fit.nll, fit.theta; h = DRModels._fd_hessian_step(2n))
        rel = maximum(abs, H[1:nb, 1:nb] .- S) / maximum(abs, S)
        # Measured 2026-09-05: 3.96e-9 (ML), 9.75e-9 (REML) -- ~3 orders of headroom.
        @test rel < 1e-6

        # The reported beta SEs come from the FULL inverse, so they must EXCEED
        # the conditional inv(S) ones: the full inverse carries the uncertainty in
        # the variance parameters that the conditional block holds fixed.
        full_se = stderror(fit)[1:nb]
        cond_se = sqrt.(diag(inv(Symmetric(S))))
        @test all(full_se .> cond_se)
        @test all(full_se ./ cond_se .< 1.05)   # and not wildly so on a healthy fit
    end
end

@testset "q2 structured vcov: profile likelihood corroborates the rho12 SE" begin
    # SECOND oracle, covering what the Schur complement cannot. That oracle is
    # exact but only for the four beta coordinates; the six variance / correlation
    # coordinates were unvalidated by it. A profile-likelihood interval is built by
    # re-optimising the nuisance parameters at each fixed value -- it never touches
    # the Hessian -- so agreement with the Wald interval is independent evidence
    # that the curvature is right on a NON-beta axis.
    #
    # rho12 (atanh scale) is the cheapest single-coefficient block. Measured
    # 2026-09-05: profile/Wald half-width ratio 1.0082 for rho12 and 1.0050 for
    # sigma1, at 2.5 s and 7.5 s. Only rho12 is asserted here, to keep the file
    # under ~20 s. Profiling works on this route through the `:finite` autodiff
    # fallback in `_profile_autodiff_mode` -- the same CHOLMOD wall that forces the
    # finite-difference Hessian also forces finite-difference nuisance gradients.
    fit, _ = _q2v_levelled_fit(:relmat, :ML)
    se = stderror(fit)
    idx = Dict(fit.blocks)[:rho12][1]
    z = 1.959963984540054
    ci = confint(fit; method = :profile, parm = [:rho12])
    @test length(ci) == 1
    r = ci[1]
    @test isfinite(r.lower) && isfinite(r.upper)
    @test r.lower < r.estimate < r.upper
    ratio = ((r.upper - r.lower) / 2) / (z * se[idx])
    @test 0.9 < ratio < 1.1
end

@testset "q2 structured vcov: a genuinely flat direction is reported as flat" begin
    # DGP with an EXACTLY rank-one among-group covariance: mu1 and mu2 share ONE
    # group effect, so Sigma_a is [s^2 s^2; s^2 s^2] by construction and its
    # second Cholesky diagonal is truly zero. This is the regime the leaf cares
    # about, because before the fix EVERY fit on this route looked like this --
    # `stderror` returned Inf on all ten coordinates and `_boundary_se` made
    # "never computed" indistinguishable from "unidentified". Now Inf, or a very
    # large SE, on the L22 coordinate MEANS something, and the mu block does not.
    rng = MersenneTwister(20260905)
    G = 16; nrep = 4
    K = Matrix{Float64}(I, G, G)          # independent groups: no relatedness signal
    group = repeat(1:G, inner = nrep)
    n = length(group)
    x = randn(rng, n)
    a = 0.6 .* randn(rng, G)              # ONE shared among-group effect
    dat = (; y1 = 0.3 .+ 0.2 .* x .+ a[group] .+ 0.35 .* randn(rng, n),
             y2 = -0.1 .+ 0.4 .* x .+ a[group] .+ 0.35 .* randn(rng, n),
             x, grp = ["g$(i)" for i in group])
    form = bf(mu1 = @formula(y1 ~ x + relmat(1 | grp)),
              mu2 = @formula(y2 ~ x + relmat(1 | grp)),
              sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
              rho12 = @formula(rho12 ~ 1))
    for meth in (:ML, :REML)
        fit = drm(form, Gaussian(); data = dat, K = K, g_tol = 1e-6, method = meth)
        @test fit.converged
        se = stderror(fit)
        # Assert the SCALE-FREE near-singularity of Sigma_a -- 1 - r^2 for the
        # among-axis correlation r, which is det(Sigma_a) divided by the product
        # of its diagonal -- and NOT det(Sigma_a) against an absolute bound.
        # det has units of (variance)^2, and along a direction the data does not
        # identify its residual value is set by wherever the optimiser stopped,
        # so it is Julia-version dependent. Measured on this exact fixture
        # 2026-09-06: det = 9.75e-8 (ML) / 1.98e-8 (REML) on Julia 1.10.12 --
        # the numbers the original `det < 1e-6` was written against -- but
        # 5.58e-7 / 1.15e-6 on 1.12.6, and 1.02e-6 in CI on 1.12.7, which is
        # what made the old bound fail there. 1 - r^2 is stable across both:
        # 6.71e-6 / 1.17e-6 on 1.10.12, 4.37e-6 / 7.88e-6 on 1.12.6.
        # It still DISCRIMINATES, measured the same day: rerun this fixture with
        # an INDEPENDENT second group effect and 1 - r^2 is 0.99983 on both arms
        # under 1.12.6 (0.377 / 0.399 under 1.10.12) -- four to five orders of
        # magnitude above the bound -- and the flat-direction assertions below
        # go with it (se[10] finite at 0.19, argmax(se) = 8, not 10).
        Sigma_a = fit.ranef.Sigma_a
        @test det(Sigma_a) / (Sigma_a[1, 1] * Sigma_a[2, 2]) < 1e-4
        # The placeholder is gone even here: a flat direction must not reintroduce
        # a whole-matrix NaN.
        @test count(isnan, vcov(fit)) == 0
        # The nine identified coordinates are finite and the mu block is tight.
        @test all(isfinite, se[1:9])
        @test all(isfinite, vcov(fit)[1:9, 1:9])
        @test maximum(se[1:4]) < 0.5
        # The tenth (Sigma_a:L22) is the flat one and is reported as flat: 26.9 on
        # the ML arm, Inf on the REML arm, where `_finite_hessian` also finds H is
        # not positive definite and warns. Asserted as `> 10` so the PD/non-PD
        # split, which sits on the solver's noise floor, cannot make this brittle.
        @test argmax(se) == 10
        @test se[10] > 10.0
    end
end

@testset "q2 structured vcov: the sentinel says so rather than going silent" begin
    # `_vcov_from_hessian` is the guard the route is wired through (pinned above).
    # On a singular Hessian it must WARN and pseudo-invert, naming the route via
    # the `context` the call site passes -- not return a plausible-looking number
    # in silence. Deterministic, unlike waiting for a fit to land there.
    Hsing = [2.0 0.0; 0.0 0.0]
    ctx = "bivariate Gaussian q=2 structured (relmat) finite-difference Hessian"
    Vs = @test_logs (:warn,) match_mode = :any DRModels._vcov_from_hessian(Hsing; context = ctx)
    @test all(isfinite, Vs)
    @test Vs[1, 1] == 0.5
    @test Vs[2, 2] == 0.0        # pseudo-inverse: the flat direction gets zero, not junk
    # ... and `stderror`'s boundary map turns a non-positive variance into Inf,
    # which is what the user sees on that coordinate (src/inference.jl).
    @test DRModels._boundary_se(Vs[2, 2]) == Inf
end
