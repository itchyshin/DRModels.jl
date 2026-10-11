# Lane J2 (convergence, optimiser and vcov) regressions.
#
# One @testset per issue. Each would fail on main before this slice: either the
# behaviour was wrong (a cap exit reported as success, a LAPACK error, a raw flag
# shown for a degenerate fit, a sentinel read as a CI endpoint) or the recorded
# status the fix adds did not exist.
using DRModels
using Test, Random, LinearAlgebra, SparseArrays

# Rebuild a fit with selected fields replaced (all 23 positional fields).
function _j2_rebuild(fit; converged = fit.converged, scales = fit.scales, ranef = fit.ranef)
    return DRModels.DrmFit(fit.family, fit.blocks, fit.coefnames, fit.theta, fit.vcov,
        fit.loglik, fit.nobs, converged, fit.means, fit.obs, scales, fit.formula, fit.nll,
        fit.nllgrad, ranef, fit.estim_method, fit.reml_loglik, fit.ml_loglik, fit.marginal,
        fit.phylo_penalty, fit.penalty, fit.iterations, fit.phylo_scale)
end

# An AbstractDict whose lookups throw, to make a boundary diagnostic fail.
struct _J2ThrowingDict <: AbstractDict{Symbol,Any} end
Base.get(::_J2ThrowingDict, k, d) = error("boom (test)")
Base.length(::_J2ThrowingDict) = 0
Base.iterate(::_J2ThrowingDict, s...) = nothing

@testset "#1024 sparse Laplace inner mode: a cap exit is not success" begin
    y = [0.0, 3, 10, 40, 200]
    Z = sparse(1.0I, 5, 5); cid = ones(Int, 5)
    _, _, iters2, ok2 = DRModels._poisson_laplace_mode(y, zeros(5), Z, cid, [0.0];
                                                       maxiter = 2, tol = 1e-10)
    @test iters2 == 2
    @test ok2 == false                       # was `true` (ch !== nothing)
    _, _, iters60, ok60 = DRModels._poisson_laplace_mode(y, zeros(5), Z, cid, [0.0];
                                                         maxiter = 60, tol = 1e-10)
    @test ok60 == true
    @test iters60 < 60

    # coordinate-spatial Poisson mode (poisson.jl), same contract
    ys = [0.0, 1, 2, 3, 4]
    Cinv = Matrix(1.0I, 5, 5)
    _, _, ok1 = DRModels._poisson_spatial_mode(ys, zeros(5), collect(1:5), Cinv, 1.0, zeros(5);
                                               maxiter = 1)
    @test ok1 == false                       # was `issuccess(Hch)` = true
    _, _, okd = DRModels._poisson_spatial_mode(ys, zeros(5), collect(1:5), Cinv, 1.0, zeros(5))
    @test okd == true
end

@testset "#1047 EM: converged is the EM's own stop, not iterations < cap" begin
    Random.seed!(20261011)
    G = 32
    phy = random_balanced_tree(G; branch_length = 0.3)
    m = 3; n = G * m
    sp = repeat(1:G; inner = m)
    x = randn(n)
    y = 0.2 .+ 0.5 .* x .+ 0.8 .* randn(G)[sp] .+ 0.4 .* randn(n)
    X = hcat(ones(n), x)
    prob = DRModels.make_loc_problem(phy, y, X; species = sp)
    r = DRModels.em_fit(prob; max_iter = 500, reltol = 1e-10)
    @test r.converged
    @test r.iterations < 500
    # Stopped by the cap after one step: not converged.
    r1 = DRModels.em_fit(prob; max_iter = 1, reltol = 1e-10)
    @test r1.converged == false
    # A run whose tolerance is met exactly ON the cap is converged (the old
    # `iterations < max_iter` rule called it unconverged).
    rk = DRModels.em_fit(prob; max_iter = r.iterations, reltol = 1e-10)
    @test rk.iterations == r.iterations
    @test rk.converged
end

@testset "#964 temporal starts are compared at the minimizer, not Optim.minimum" begin
    src = read(joinpath(pkgdir(DRModels), "src", "temporal.jl"), String)
    @test !occursin("Optim.minimum(res) < Optim.minimum(best)", src)
    @test count("_better_restart(nll, best, res)", src) == 2
end

@testset "#956 raw inv(H) Wald sites go through the guarded constructor" begin
    # The issue's REML repro (an exactly collinear `sigma ~ z + 2z` design) now
    # fails earlier, inside the REML profile solve, before any vcov is built, so
    # it cannot exercise these sites; it is a design-validation case. Assert the
    # sites themselves no longer carry a raw inverse.
    root = joinpath(pkgdir(DRModels), "src")
    rd(f) = read(joinpath(root, f), String)
    @test !occursin("inv(Symmetric(Ĥ))", rd("gaussian_bivariate.jl"))
    @test !occursin("inv(Symmetric(ForwardDiff.hessian(nll_reml", rd("gaussian_bivariate.jl"))
    @test !occursin("inv(Symmetric(Hmat))", rd("gaussian_structured.jl"))
    @test !occursin("inv(Symmetric(H))", rd("locscale_corr.jl"))
    @test !occursin("inv(Symmetric(Hσ))", rd("gaussian_core.jl"))
    @test !occursin(r"^\s*inv\(H\)\s*$"m, rd("mixed_family.jl"))
end

@testset "#968 constant response: no LAPACK error from the vcov guard" begin
    x = [1.516365964624, -1.10216883287986, 0.0437081628396838, -0.244139880505117,
         0.169596556966056, -1.38014579564091, 1.16924063206502, -0.557166958025678,
         -1.26976096398019, 0.830998928705942, 1.86507051250845, 0.486796504665075,
         0.385539985582033, -0.881224314754194, -0.408360147832616, 0.918153970571253,
         -0.371338689847125, -0.89956867119498, 0.438905790739674, -0.131006067818851,
         1.18634583166084, -2.75107957857718, -0.256516548283282, -2.64865209710159,
         0.917823126833453, 0.589095863716824, 0.447281877160813, 0.556833678557327,
         -0.118680559418778, 0.229224768530565, 0.424604544036726, -1.01548501306329,
         -0.54043985839703, -0.281543124928349, -0.832622801434502, 0.914207139798737,
         -0.21280631274447, -0.489806565230964, 1.34017304442836, -0.0463818196819419,
         -2.319887738207, 0.718828972512561, -1.04081181692415, -0.946085919924027,
         0.301509651894001, 0.749154143997904, -0.39593399940496, 0.0645792052553393,
         0.266520606549011, -0.0991708046336379]
    y = fill(3, length(x))   # Int64, as read from a CSV
    fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
    @test fit isa DrmFit
    # A non-finite curvature is never reported as a converged fit.
    all(isfinite, vcov(fit)) || @test !fit.converged
end

@testset "#949 show / check_drm / bridge_diagnostics use is_converged" begin
    Random.seed!(949)
    n = 60; x = randn(n); y = 1 .+ 0.5 .* x .+ 0.3 .* randn(n)
    good = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
    @test is_converged(good)
    # Same fit with a collapsed residual scale: the raw flag stays true.
    bad = _j2_rebuild(good; converged = true,
                      scales = Dict(:sigma => fill(1e-12, n)))
    @test bad.converged && !is_converged(bad)
    txt = sprint(show, MIME"text/plain"(), bad)
    @test occursin("converged = false", txt)
    @test occursin("degenerate", txt)
    rep = check_drm(bad)
    @test rep.converged == false
    @test rep.ok == false
    @test bridge_diagnostics(bad).converged == false
    # Healthy fit: unchanged.
    @test check_drm(good).converged == true
    @test bridge_diagnostics(good).converged == good.converged
end

@testset "#1003 a failing boundary diagnostic warns instead of reading as clean" begin
    fit = DRModels.DrmFit(Gaussian(), [:mu => 1:1, :sigma => 2:2, :resd => 3:3],
        [:mu => ["(Intercept)"], :sigma => ["(Intercept)"], :resd => ["g"]],
        zeros(3), Matrix(1.0I, 3, 3), -10.0, 5, true,
        Dict(:mu => zeros(5)), Dict(:mu => collect(1.0:5.0)), Dict(:sigma => ones(5)),
        nothing, nothing, nothing, _J2ThrowingDict(), :ML, NaN, -10.0, :LA)
    @test_logs (:warn, r"diagnostic failed") match_mode = :any DRModels._warn_variance_boundary(fit)
end

@testset "#1005 Nelder-Mead locscale results need an exact-gradient certificate" begin
    θ = [0.1, 0.2]
    @test DRModels._ls_gradient_certified((g, t) -> (g .= 0.0; g), θ, 1e-6)
    @test !DRModels._ls_gradient_certified((g, t) -> (g .= 1e-3; g), θ, 1e-6)
    @test !DRModels._ls_gradient_certified((g, t) -> (g .= NaN; g), θ, 1e-6)
    @test !DRModels._ls_gradient_certified((g, t) -> error("boom"), θ, 1e-6)
end

@testset "#1007 variance ratios carry the fit's status" begin
    Random.seed!(42)
    G = 25; m = 6; nn = G * m
    g = repeat(1:G, inner = m)
    b1 = 0.9 .* randn(G)
    y = 1.0 .+ b1[g] .+ 0.5 .* randn(nn)
    fit = drm(bf(@formula(y ~ 1 + (1 | g))), Gaussian(); data = (; y, g))
    h = heritability(fit)
    @test h.status in (:ok, :vcov_unavailable)
    @test isfinite(h.se) == (h.status === :ok)
    bad = _j2_rebuild(fit; converged = false)
    hb = @test_logs (:warn, r"not converged") match_mode = :any heritability(bad)
    @test hb.status === :not_converged
    ib = @test_logs (:warn, r"not converged") match_mode = :any icc(bad)
    @test ib.status === :not_converged
end

@testset "#1023 resd profile: the sentinel is not an endpoint, nuisance exit recorded" begin
    Random.seed!(3)
    tr = "((a:1,b:1):1,(c:1,d:1):1,(e:1,f:1):1);"
    sp = ["a", "b", "c", "d", "e", "f"]
    g = repeat(sp, inner = 3); n = length(g); x = randn(n); y = 1 .+ 0.3 .* x .+ randn(n)
    f = drm(bf(@formula(y ~ x + phylo(1 | g)), @formula(sigma ~ 1)), Gaussian();
            data = (y = y, x = x, g = g), tree = tr)
    obj = f.nll
    if obj isa DRModels.LocOnlyObjective
        th = f.theta
        # At the domain bound the objective is the 1e18 sentinel: not accepted.
        v, _, _, nu = DRModels._loconly_profiled_resd_nll(obj, th[obj.pμ + 1], -50.0)
        @test v >= 1e16
        @test nu.accepted == false
        @test nu.reason === :nonfinite_objective
        # Interior point: accepted.
        _, _, _, nu2 = DRModels._loconly_profiled_resd_nll(obj, th[obj.pμ + 1], th[obj.pμ + 2])
        @test nu2.accepted

        r = profile_result(f; parm = :resd)
        @test r.autodiff === :loconly
        s = r.stats[1]; row = r.ci[1]
        # No finite bound at the sentinel edge (was -49.999999999136).
        @test !(isfinite(row.lower) && row.lower <= -49)
        @test !(isfinite(row.upper) && row.upper >= 49)
        # Flags are measured, not constants.
        @test s.lower_nuisance_reason !== :not_checked
        @test s.upper_nuisance_reason !== :not_checked
        @test r.failed == count(t -> t.lower_endpoint_failed || t.upper_endpoint_failed, r.stats)
        if isinf(row.lower)
            @test s.lower_unbounded || s.lower_endpoint_failed
        end
    else
        @test_skip "fit did not route through LocOnlyObjective"
    end
end

@testset "#961 non-finite Hessian on finite data: NaN vcov, not a LAPACK error" begin
    H = [1.0 0.0; 0.0 Inf]
    V, hf = @test_logs (:warn, r"not finite") match_mode = :any DRModels._vcov_from_hessian_checked(H, true)
    @test hf == false
    @test size(V) == (2, 2) && all(isnan, V)
    # NaN data: the guard is called unchanged and still throws (owner decision 14).
    @test_throws Exception DRModels._vcov_from_hessian_checked([1.0 0.0; 0.0 NaN], false)
    # Healthy matrix: identical to the guard.
    V2, hf2 = DRModels._vcov_from_hessian_checked([2.0 0.0; 0.0 4.0], true)
    @test hf2 && V2 ≈ [0.5 0.0; 0.0 0.25]
end

@testset "#952 a fit ending on the sentinel warns" begin
    fit = DRModels.DrmFit(Gaussian(), [:mu => 1:1, :sigma => 2:2],
        [:mu => ["(Intercept)"], :sigma => ["(Intercept)"]],
        zeros(2), fill(NaN, 2, 2), -1e18, 5, false,
        Dict(:mu => zeros(5)), Dict(:mu => collect(1.0:5.0)), Dict(:sigma => ones(5)))
    @test_logs (:warn, r"FAILED") match_mode = :any DRModels._warn_variance_boundary(fit)
end

@testset "#978 sparse phylo-mean: mean SEs from the joint observed information" begin
    Random.seed!(978)
    G = 48
    phy = random_balanced_tree(G; branch_length = 0.3)
    m = 2; n = G * m
    species = repeat(1:G; inner = m)
    z = randn(G)
    x = z[species] .+ 0.3 .* randn(n)          # covariate aligned with the tree structure
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    u = 0.8 .* (cholesky(Symmetric(C ./ maximum(diag(C)))).L * randn(G))
    y = 0.2 .+ 0.5 .* x .+ u[species] .+ 0.5 .* randn(n)
    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)), Gaussian();
              data = (; y, x, species), tree = phy)
    V = vcov(fit)
    if fit.nll isa DRModels.LocOnlyObjective && all(isfinite, V)
        # Cross block is no longer zero by construction.
        @test any(!iszero, V[1:2, 3:4])
        # Mean SEs match the inverse of a finite-difference Hessian of the full objective.
        Hfd = DRModels._finite_hessian(fit.nll, fit.theta; h = DRModels._fd_hessian_step(n))
        Vfd = inv(Symmetric(Hfd))
        @test sqrt.(diag(V)[1:2]) ≈ sqrt.(diag(Vfd)[1:2]) rtol = 5e-3
    else
        @test_skip "fit did not route through the sparse phylo-mean objective"
    end
end

@testset "#959 a fit ending at a likelihood clamp is reported" begin
    hits = @test_logs (:warn, r"clamp inside the likelihood") match_mode = :any DRModels._warn_at_clamp(
        "test", ["log sigma" => ([-1.0, 15.0], -15.0, 15.0)])
    @test length(hits) == 1 && occursin("upper clamp", hits[1])
    # Interior values: no warning, no hits.
    @test isempty(@test_logs DRModels._warn_at_clamp("test", ["log sigma" => ([-1.0, 2.0], -15.0, 15.0)]))
    # Quiet inside bootstrap / profile refits, but still detected.
    quiet = @test_logs DRModels._without_boundary_warnings(() ->
        DRModels._warn_at_clamp("test", ["x" => ([-8.0], -8.0, 3.0)]))
    @test length(quiet) == 1
    # Sparse-Laplace layout: random-effect log SD at -8 and dispersion log σ at 4.
    fit = DRModels.DrmFit(Poisson(), [:mu => 1:1, :sigma => 2:2, :resd => 3:3],
        [:mu => ["(Intercept)"], :sigma => ["(Intercept)"], :resd => ["g"]],
        [0.0, 4.0, -8.0], Matrix(1.0I, 3, 3), -10.0, 5, true,
        Dict(:mu => ones(5)), Dict(:mu => ones(5)), Dict{Symbol,Vector{Float64}}())
    @test_logs (:warn, r"random-effect log SD.*dispersion log sigma") match_mode = :any DRModels._warn_laplace_clamp(
        fit; logsigma = [4.0])
end

@testset "review: PD design Gram blocks invert silently; singular ones still warn" begin
    X = hcat(ones(50), 2000 .+ 10 .* randn(MersenneTwister(1), 50))   # uncentred predictor
    G = X' * X
    @test cond(G) > 1e10
    V = @test_nowarn DRModels._vcov_from_gram(G)
    @test V ≈ inv(G) rtol = 1e-6
    Gs = [1.0 2.0; 2.0 4.0]                                            # exactly singular
    @test_logs (:warn, r"singular|positive definite") match_mode = :any DRModels._vcov_from_gram(Gs)
end

@testset "review: profile_edge fires near the upper bound" begin
    # the upper arm of a ratio profile ends near 1 - 5e-7 (nll_at_ratio(1) = Inf)
    src = read(joinpath(pkgdir(DRModels), "src", "heritability.jl"), String)
    @test occursin("hi >= 1 - 1e-6", src)
end
