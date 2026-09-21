# test_reml_q2_structured.jl — REML (Patterson–Thompson) for the bivariate
# q=2 structured residual-correlation route (#470).
#
# The q=2 structured route puts a random effect on mu1/mu2 ONLY (Λ is 2x2);
# sigma1, sigma2, rho12 are intercept-only fixed effects with no random-effect
# axis. REML therefore marginalises beta_mu1/beta_mu2 only — see
# `src/reml_q2.jl`'s header for the full argument (mirroring the axis rule
# `reml_q4.jl` uses, generalised to "no u-axis ⇒ stays outer").

using DRModels
using Test, LinearAlgebra, Random, Statistics, SparseArrays

@testset "q2 ML: inadmissible covariance and failed final likelihood" begin
    # No random draws or optimizer trajectory: these states are fixed regressions
    # for the covariance boundary reached by the phylogenetic ML fit on CI.
    X = ones(4, 1)
    Y = [0.1 0.4; -0.2 0.3; 0.3 -0.1; -0.4 -0.2]
    prob, Q = DRModels.make_coevo_problem_from_covariance(
        Matrix{Float64}(I, 4, 4), Y, X; group = collect(1:4))
    β = zeros(1, 2)
    D = Matrix{Float64}(I, 2, 2)
    for lc in ([-25.0, 1.0, -25.0], [-35.0, 0.0, 0.0])
        Λ = DRModels.lc_to_cov(lc, 2)
        @test !DRModels._q2_lambda_admissible(Λ)
        @test first(DRModels.coevo_marginal_cov(prob, Q, β, Λ, D)) == -Inf
    end

    # An unsuccessful factorization already returns -Inf, without throwing.
    # A stationary barrier (zero finite-difference gradient) must never turn
    # that failed final evaluation into a converged fit.
    badQ = sparse(-100.0 * I(4))
    failed = DRModels.fit_coevolution_q2_residual(
        prob, badQ; β0 = β, Λ0 = D, σ0 = ones(2), iterations = 0)
    @test !isfinite(failed.loglik)
    @test !failed.converged

    # The ML driver already retains its log-Cholesky coordinates. Re-factorizing
    # their rounded covariance during front-end packing can throw even when the
    # failed fit should simply be returned as nonconverged.
    θ = [0.0, 0.0, -400.0, 1.0, -400.0, 0.0, 0.0, 0.0]
    βbad, Λbad, Dbad, σbad, ρbad = DRModels.coevo_q2_residual_unpack(prob, θ)
    @test !isposdef(Symmetric(Λbad))
    badfit = (; β = βbad, Λ = Λbad, σ_res = σbad, rho12 = ρbad, θ)
    packed = DRModels._q2_structured_theta(badfit, 1, :ML)
    @test packed == θ[[1, 2, 6, 7, 8, 3, 4, 5]]

    # Healthy ML and REML fits retain the existing public parameter order.
    Λhealthy = [0.3 0.02; 0.02 0.2]
    θhealthy = DRModels.coevo_q2_residual_pack(β, Λhealthy, [0.5, 0.7], 0.2)
    healthy = (; β, Λ = Λhealthy, σ_res = [0.5, 0.7], rho12 = 0.2, θ = θhealthy)
    @test DRModels._q2_structured_theta(healthy, 1, :ML) ≈
          DRModels._q2_structured_theta(healthy, 1, :REML)
end

function _q2_reml_known_cov_fixture(K, β, Λ, residual_cov; nrep, rng)
    G = size(K, 1)
    Q = sparse(Matrix(inv(cholesky(Symmetric(K)))))
    P = DRModels.prior_precision(Q, inv(Λ))
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
        Y[i, 1] = sum(X[i, :] .* β[:, 1]) + u[base + 1]
        Y[i, 2] = sum(X[i, :] .* β[:, 2]) + u[base + 2]
        Y[i, :] .+= L * randn(rng, 2)
    end
    return (; Y, X, group)
end

@testset "q2 REML: direct fit_coevolution_q2_reml matches the ML shape, moves Λ up" begin
    rng = MersenneTwister(20260824)
    G = 14
    idx = collect(1:G)
    K = [0.55 ^ abs(i - j) for i in idx, j in idx] + 1e-6I
    β = [0.20 -0.15; 0.25 0.10]
    Λ = Matrix(Symmetric([0.20 0.05; 0.05 0.17]))
    residual_cov = Matrix(Symmetric([0.10 0.025; 0.025 0.14]))
    sim = _q2_reml_known_cov_fixture(K, β, Λ, residual_cov; nrep = 4, rng = rng)
    prob, Q = DRModels.make_coevo_problem_from_covariance(K, sim.Y, sim.X; group = sim.group)

    fit_ml = DRModels.fit_coevolution_q2_residual(prob, Q; iterations = 300, g_tol = 1e-6)
    fit_reml = DRModels.fit_coevolution_q2_reml(prob, Q; iterations = 300, g_tol = 1e-6)

    @test fit_ml.converged
    @test fit_reml.converged
    @test isfinite(fit_reml.reml_loglik)
    @test isfinite(fit_reml.ml_loglik)
    # Still holds after #477 normalised this route, but for a slightly different
    # reason than the original comment gave, so it is worth restating: the
    # reported value is now `ml_ll − ½logdet(S) + (n_β/2)·log(2π)`, i.e. a
    # NEGATIVE correction plus a POSITIVE constant (n_β = 2 here, so log(2π) ≈
    # 1.838). The inequality survives because ½logdet(S) dominates that constant
    # on these fits — it is not the tautology "the correction is negative" any
    # more. Under the normalised convention REML > ML is possible in general, so
    # if this ever fails, check the two magnitudes before assuming a regression.
    @test fit_reml.reml_loglik < fit_reml.ml_loglik
    @test size(fit_reml.Λ) == (2, 2)
    @test isposdef(Symmetric(fit_reml.Λ))
    @test isposdef(Symmetric(fit_reml.residual_cov))

    # The defining REML property: restricted variance-component estimates are
    # less downward-biased than ML, on BOTH random-effect axes (mu1, mu2) —
    # same direction as reml_q4.jl's all-axes property, checked here for the
    # axes this route actually has a random effect on.
    @test diag(fit_reml.Λ)[1] >= diag(fit_ml.Λ)[1] - 1e-6
    @test diag(fit_reml.Λ)[2] >= diag(fit_ml.Λ)[2] - 1e-6
    @test maximum(abs.(diag(fit_reml.Λ) .- diag(fit_ml.Λ))) > 1e-4   # not a silent no-op
end

@testset "q2 REML: small-G recovery — REML reduces variance-component bias vs ML" begin
    # G = 8 is deliberately small (the REML bias-correction regime). Averaged
    # over 60 seeds; asserts REML's |bias| is smaller than ML's on both
    # random-effect axes (mu1, mu2), matching the same finding reported in the
    # after-task check log.
    G = 8
    K = Matrix(1.0I, G, G)
    β = [0.20 -0.15; 0.25 0.10]
    Λ_true = Matrix(Symmetric([0.30 0.10; 0.10 0.25]))
    residual_cov = Matrix(Symmetric([0.08 0.02; 0.02 0.10]))
    nsim = 60

    ml1 = Float64[]; ml2 = Float64[]; reml1 = Float64[]; reml2 = Float64[]
    for s in 1:nsim
        rng = MersenneTwister(1000 + s)
        sim = _q2_reml_known_cov_fixture(K, β, Λ_true, residual_cov; nrep = 3, rng = rng)
        prob, Q = DRModels.make_coevo_problem_from_covariance(K, sim.Y, sim.X; group = sim.group)
        fit_ml = DRModels.fit_coevolution_q2_residual(prob, Q; iterations = 400, g_tol = 1e-5)
        fit_reml = DRModels.fit_coevolution_q2_reml(prob, Q; iterations = 400, g_tol = 1e-5)
        if fit_ml.converged && all(isfinite, fit_ml.Λ)
            push!(ml1, fit_ml.Λ[1, 1]); push!(ml2, fit_ml.Λ[2, 2])
        end
        if fit_reml.converged && all(isfinite, fit_reml.Λ)
            push!(reml1, fit_reml.Λ[1, 1]); push!(reml2, fit_reml.Λ[2, 2])
        end
    end

    @test length(ml1) >= 50 && length(reml1) >= 50   # most seeds converge cleanly

    bias_ml1 = abs(mean(ml1) - Λ_true[1, 1]); bias_reml1 = abs(mean(reml1) - Λ_true[1, 1])
    bias_ml2 = abs(mean(ml2) - Λ_true[2, 2]); bias_reml2 = abs(mean(reml2) - Λ_true[2, 2])
    @test bias_reml1 < bias_ml1
    @test bias_reml2 < bias_ml2
end

@testset "q2 REML: front-end drm(method = :REML) on the phylo route" begin
    rng = MersenneTwister(20260901)
    phy = DRModels.random_balanced_tree(20; branch_length = 0.2)
    β = [0.20 -0.15; 0.25 0.10]
    Λ = Matrix(Symmetric([0.22 0.07; 0.07 0.18]))
    residual_cov = Matrix(Symmetric([0.12 0.04; 0.04 0.16]))
    Q_cond, leaf_pos, _ = DRModels.augmented_tree_precision(phy)
    P = DRModels.prior_precision(Q_cond, inv(Λ))
    F = cholesky(Symmetric(P))
    u = F.UP \ randn(rng, size(P, 1))
    species = repeat(1:phy.n_leaves, inner = 4)
    n = length(species)
    x = randn(rng, n)
    X = hcat(ones(n), x)
    L = cholesky(Symmetric(residual_cov)).L
    Y = zeros(n, 2)
    for i in 1:n
        base = 2 * (leaf_pos[species[i]] - 1)
        for a in 1:2
            Y[i, a] = sum(X[i, :] .* β[:, a]) + u[base + a]
        end
        Y[i, :] .+= L * randn(rng, 2)
    end
    dat = (; y1 = Y[:, 1], y2 = Y[:, 2], x, species_name = [phy.leaf_names[k] for k in species])

    form = bf(
        mu1 = @formula(y1 ~ x + phylo(1 | species_name)),
        mu2 = @formula(y2 ~ x + phylo(1 | species_name)),
        sigma1 = @formula(sigma1 ~ 1),
        sigma2 = @formula(sigma2 ~ 1),
        rho12 = @formula(rho12 ~ 1),
    )

    # ML baseline, run TWICE: the REML addition must not perturb the ML path
    # at all (verified below by exact float equality, not just "close").
    fit_ml_1 = drm(form, Gaussian(); data = dat, tree = phy, g_tol = 1e-4)
    fit_ml_2 = drm(form, Gaussian(); data = dat, tree = phy, g_tol = 1e-4)
    @test fit_ml_1.theta == fit_ml_2.theta
    @test fit_ml_1.loglik == fit_ml_2.loglik
    @test estimation_method(fit_ml_1) === :ML

    fit_reml = drm(form, Gaussian(); data = dat, tree = phy, g_tol = 1e-4, method = :REML)
    @test estimation_method(fit_reml) === :REML
    @test isfinite(reml_loglik(fit_reml))
    @test isfinite(ml_loglik(fit_reml))
    @test loglik(fit_reml) == reml_loglik(fit_reml)
    @test fit_reml.ranef.axes == (:mu1, :mu2)
    @test size(fit_reml.ranef.Sigma_a) == (2, 2)
    # REML genuinely changes the fit (mirrors the q4 "not a silent no-op" check).
    @test maximum(abs.(diag(fit_reml.ranef.Sigma_a) .- diag(fit_ml_1.ranef.Sigma_a))) > 1e-4

    # Re-running ML AFTER the REML fit must still reproduce the original ML
    # baseline exactly — the REML branch shares no mutable state with :ML.
    fit_ml_3 = drm(form, Gaussian(); data = dat, tree = phy, g_tol = 1e-4)
    @test fit_ml_3.theta == fit_ml_1.theta
end

@testset "q2 spatial(coords) is admitted, and is relmat(K) under another name" begin
    # Parity leaf jl-q2-spatial. drmTMB fits this cell natively AND through the
    # bridge, which rewrites `spatial(...)` to `relmat(...) + K` on the R side
    # (R/julia-bridge.R); DRModels.jl used to refuse it outright here. `spatial` now
    # supplies a covariance exactly as `relmat`/`animal` do -- the exponential
    # kernel at a FIXED range -- so the two markers must be ONE model reached by
    # two names, and this testset holds them to bit-for-bit equality.
    rng = MersenneTwister(20260905)
    G = 12
    coords = hcat(range(0.0, 1.0; length = G), sin.(range(0.0, pi; length = G)))
    Ddist = [sqrt(sum(abs2, coords[k, :] .- coords[l, :])) for k in 1:G, l in 1:G]
    rho_mean = sum(Ddist) / (G^2 - G)          # the route's fixed-range rule
    Ksp = exp.(-Ddist ./ rho_mean) + 1e-8 * I

    # The matrix the route builds is EXACTLY this one, bit for bit, and
    # `spatial_range` overrides the default rule.
    @test DRModels._spatial_covariance_from_coords(:site, G, coords, nothing) == Ksp
    @test DRModels._spatial_covariance_from_coords(:site, G, coords, 0.5) ==
        exp.(-Ddist ./ 0.5) + 1e-8 * I

    # Signal-bearing fixture: simulate FROM the covariance the route will build.
    beta = [0.20 -0.12; 0.18 0.10]
    Lam = Matrix(Symmetric([0.20 0.05; 0.05 0.17]))
    residual_cov = Matrix(Symmetric([0.10 0.025; 0.025 0.14]))
    sim = _q2_reml_known_cov_fixture(Ksp, beta, Lam, residual_cov; nrep = 3, rng = rng)
    labels = ["s$(i)" for i in sim.group]
    dat = (; y1 = sim.Y[:, 1], y2 = sim.Y[:, 2], x = sim.X[:, 2], site = labels)

    spatial_form = bf(
        mu1 = @formula(y1 ~ x + spatial(1 | site)),
        mu2 = @formula(y2 ~ x + spatial(1 | site)),
        sigma1 = @formula(sigma1 ~ 1),
        sigma2 = @formula(sigma2 ~ 1),
        rho12 = @formula(rho12 ~ 1),
    )
    relmat_form = bf(
        mu1 = @formula(y1 ~ x + relmat(1 | site)),
        mu2 = @formula(y2 ~ x + relmat(1 | site)),
        sigma1 = @formula(sigma1 ~ 1),
        sigma2 = @formula(sigma2 ~ 1),
        rho12 = @formula(rho12 ~ 1),
    )

    # ADMISSION -- the cell fits instead of refusing.
    fit_sp = drm(spatial_form, Gaussian(); data = dat, coords = coords, g_tol = 2e-4)
    @test fit_sp.converged
    @test isfinite(loglik(fit_sp))
    @test fit_sp.ranef.axes == (:mu1, :mu2)
    @test size(fit_sp.ranef.Sigma_a) == (2, 2)

    # The `:spatial` sentinel reaches the public accessor, so the q2 bridge
    # export target is built from it with no change to src/bridge.jl.
    @test fit_sp.ranef.structured_type === :spatial
    @test DRModels._bridge_q2_point_export(fit_sp; family = "biv_gaussian")["target"] ==
        "gaussian_q2_mu1_mu2_spatial_residual_correlation"

    # The FIXED range is the one free choice this cell makes and its default is
    # data-dependent, so it must be readable off the fit, not inferred -- the
    # same field the q=4 route records. A relmat fit has no range and says so.
    @test fit_sp.ranef.spatial_range == rho_mean
    @test drm(spatial_form, Gaussian(); data = dat, coords = coords,
              spatial_range = 0.5, g_tol = 2e-4).ranef.spatial_range == 0.5

    # SAME TARGET, EXACTLY -- one matrix, two marker names, one fit.
    fit_rm = drm(relmat_form, Gaussian(); data = dat, K = Ksp, g_tol = 2e-4)
    @test loglik(fit_sp) == loglik(fit_rm)
    @test fit_sp.theta == fit_rm.theta
    @test maximum(abs.(fit_sp.theta .- fit_rm.theta)) == 0.0
    @test fit_rm.ranef.spatial_range === nothing

    # REML rides the same path: admitted, and exact against relmat there too.
    fit_sp_reml = drm(spatial_form, Gaussian(); data = dat, coords = coords,
                      g_tol = 2e-4, method = :REML)
    fit_rm_reml = drm(relmat_form, Gaussian(); data = dat, K = Ksp,
                      g_tol = 2e-4, method = :REML)
    @test estimation_method(fit_sp_reml) === :REML
    @test isfinite(reml_loglik(fit_sp_reml))
    @test loglik(fit_sp_reml) == loglik(fit_rm_reml)
    @test fit_sp_reml.theta == fit_rm_reml.theta
    @test loglik(fit_sp_reml) != loglik(fit_sp)      # not a silent no-op

    # REPAIR -- the deleted route-wide guard refused ANY q=2 structured fit that
    # merely carried a `coords =` kwarg, spatial or not. An ordinary relmat fit
    # now ignores the stray kwarg instead of erroring, and is unchanged by it.
    fit_rm_coords = drm(relmat_form, Gaussian(); data = dat, K = Ksp,
                        coords = coords, g_tol = 2e-4)
    @test fit_rm_coords.theta == fit_rm.theta

    # PRESERVED REFUSALS -- still fail-closed on every shape out of scope.
    @test_throws ErrorException drm(spatial_form, Gaussian(); data = dat)
    @test_throws ErrorException drm(spatial_form, Gaussian(); data = dat,
                                    coords = coords[1:(G - 1), :])
    @test_throws ErrorException drm(spatial_form, Gaussian(); data = dat,
                                    coords = zeros(G, 2))
    @test_throws ErrorException drm(spatial_form, Gaussian(); data = dat,
                                    coords = coords, spatial_range = -1.0)
    mixed_form = bf(
        mu1 = @formula(y1 ~ x + spatial(1 | site)),
        mu2 = @formula(y2 ~ x + relmat(1 | site)),
        sigma1 = @formula(sigma1 ~ 1),
        sigma2 = @formula(sigma2 ~ 1),
        rho12 = @formula(rho12 ~ 1),
    )
    @test_throws ErrorException drm(mixed_form, Gaussian(); data = dat,
                                    coords = coords, K = Ksp)
end

@testset "q2 REML: V + REML is permanently rejected (not the vague 'in this slice')" begin
    n = 20
    dat = (; y1 = randn(n), y2 = randn(n), x = randn(n))
    v1 = fill(0.1, n); v2 = fill(0.1, n)
    Vm = meta_vcov_bivariate(v1, v2; cor12 = 0.0)
    form = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x))

    err = try
        drm(form, Gaussian(); data = dat, V = Vm, method = :REML)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test !occursin("in this slice", err.msg)
    @test occursin("no REML target", err.msg)

    # V + ML is unaffected by the REML message change.
    fit_v_ml = drm(form, Gaussian(); data = dat, V = Vm, method = :ML)
    @test isfinite(loglik(fit_v_ml))
end

# SUPERSEDED 2026-09-05 by #624 / drmTMB #1142. This testset used to assert
# that the residual-only bivariate route REFUSES REML ("rejection is
# untouched"). That refusal was wrong: the route has no random effects, but it
# does have mean fixed effects to restrict, and the closed-form
# Patterson-Thompson objective in `_fit_bivariate_residual_reml` now fits it
# (test/test_reml_reml_biv_residual.jl carries the full contract and the
# drmTMB same-target receipt). The #470 boundary this file cares about is the
# `V` + REML rejection, which is UNCHANGED and asserted in the testset above.
@testset "q2 REML: residual-only route now FITS by REML (#624)" begin
    n = 20
    dat = (; y1 = randn(n), y2 = randn(n), x = randn(n))
    form = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x))
    fit = drm(form, Gaussian(); data = dat, method = :REML)
    @test estimation_method(fit) === :REML
    @test isfinite(reml_loglik(fit))
    @test reml_loglik(fit) != ml_loglik(fit)
end

# The spatial covariance helper `_spatial_covariance_from_coords` is SHARED by the q=2 and
# q=4 spatial routes. PR #658 merged while this branch was open -- it both admitted q=2
# spatial AND extracted the helper -- so the fix lives in the helper and BOTH routes gain it
# in one change, which is what makes this q=2 testset real coverage rather than a pin on
# inherited behaviour: a 1-D transect written as a plain Vector is read as a G-by-1 column,
# and the Matrix path is unchanged.
@testset "q2 spatial (inherited helper): coords may be a length-G vector" begin
    G = 3
    v = [0.0, 2.0, 5.0]
    Qv = DRModels._q4_structured_precision(:spatial, :site, G;
                                      K = nothing, A = nothing, coords = v, spatial_range = 2.0)
    Qm = DRModels._q4_structured_precision(:spatial, :site, G;
                                      K = nothing, A = nothing, coords = reshape(v, G, 1),
                                      spatial_range = 2.0)
    @test size(Qv) == (G, G)
    @test isposdef(Symmetric(Qv))
    @test Qv == Qm
end
