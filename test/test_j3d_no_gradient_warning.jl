# test_j3d_no_gradient_warning.jl -- #1073 lane J3d: a route with NO exact gradient falls
# back to the value-based finite-difference Hessian (`_finite_hessian`), and that fallback
# must warn, never pass silently (Code Reviewer, #1082). The sites:
#   1. `_ordinary_laplace_vcov` with `grad! === nothing` (every production caller passes the
#      exact gradient today, but the branch is reachable and must not be silent);
#   2. the bivariate Gaussian q=2 structured route (gaussian_bivariate.jl), whose objective
#      factorises through CHOLMOD and so has no gradient at all: every fit on it warns,
#      ML and REML, whichever structured provider;
#   3. the Gaussian phylo-mean variance block (location_only.jl): since J3e (#1091) BOTH arms
#      difference an exact score (REML: `_loconly_reml_fg`), so neither takes the no-gradient
#      path; on J3d (#1082) the REML arm had only a finite-difference gradient and warned.
# The clamp fallback (a gradient exists but a log-SD has collapsed) keeps its own warning,
# `_warn_value_hessian`; both are checked to be distinct below.
using DRModels
using Test, LinearAlgebra, Random, SparseArrays
import Distributions

const _J3D_NOGRAD = r"this route has no exact gradient"
const _J3D_CLAMP = r"falling back to the value-based finite-difference Hessian"

# Log records of `f()`, and its value, without asserting on them.
_j3d_logs(f) = Test.collect_test_logs(f)
_j3d_has(logs, re) = any(l -> l.level == Base.CoreLogging.Warn && occursin(re, string(l.message)), logs)

@testset "_warn_no_gradient_hessian: names the route, quiet only inside bootstrap refits" begin
    @test_logs (:warn, r"^myroute vcov: this route has no exact gradient") DRModels._warn_no_gradient_hessian("myroute")
    @test_logs DRModels._without_boundary_warnings(() -> DRModels._warn_no_gradient_hessian("myroute"))
    # the quiet switch is restored afterwards
    @test_logs (:warn, _J3D_NOGRAD) DRModels._warn_no_gradient_hessian("myroute")
    # distinct from the clamp fallback's message
    logs, _ = _j3d_logs(() -> DRModels._warn_value_hessian("myroute"))
    @test _j3d_has(logs, _J3D_CLAMP) && !_j3d_has(logs, _J3D_NOGRAD)
end

@testset "_ordinary_laplace_vcov: no-gradient branch warns; gradient branch is silent" begin
    A = [4.0 0.5 0.1; 0.5 3.0 0.2; 0.1 0.2 2.0]
    nll = θ -> 0.5 * dot(θ, A * θ)
    grad! = (G, θ) -> (G .= A * θ; G)
    θ̂ = [0.1, -0.2, 0.3]                 # last coordinate is log sigma_b, interior (> -7)
    # 1. no gradient: the value-based Hessian, with the no-gradient warning (and only that)
    V0 = @test_logs (:warn, r"^ordinary test vcov: this route has no exact gradient") DRModels._ordinary_laplace_vcov(
        nll, θ̂, 100; se = true, context = "ordinary test")
    @test V0 ≈ inv(A) rtol = 1e-6
    # 2. exact gradient, interior point: gradient-difference Hessian, no log record at all
    Vg = @test_logs DRModels._ordinary_laplace_vcov(nll, θ̂, 100; se = true,
                                                   context = "ordinary test", grad! = grad!)
    @test Vg ≈ inv(A) rtol = 1e-8
    # 3. exact gradient but log sigma_b collapsed: the CLAMP warning, not the no-gradient one
    θc = [0.1, -0.2, -7.5]
    Vc = @test_logs (:warn, _J3D_CLAMP) DRModels._ordinary_laplace_vcov(nll, θc, 100; se = true,
                                                                       context = "ordinary test", grad! = grad!)
    @test Vc ≈ inv(A) rtol = 1e-6
    # 4. se = false builds no Hessian, so there is nothing to warn about
    Vn = @test_logs DRModels._ordinary_laplace_vcov(nll, θ̂, 100; se = false, context = "ordinary test")
    @test size(Vn) == (3, 3) && all(isnan, Vn)
end

@testset "ordinary (1 | g) Laplace fit: production caller passes the gradient, so no warning" begin
    rng = MersenneTwister(1082)
    G = 30; m = 8; n = G * m
    g = repeat(1:G, inner = m); x = randn(rng, n); b = 0.5 .* randn(rng, G)
    λ = exp.(0.3 .+ 0.4 .* x .+ b[g])
    y = Float64.([rand(rng, Distributions.Poisson(l)) for l in λ])
    logs, fit = _j3d_logs(() -> drm(bf(@formula(y ~ x + (1 | g))), Poisson();
                                    data = (; y, x, g = string.("g", g)), marginal = :Laplace, se = true))
    @test !_j3d_has(logs, _J3D_NOGRAD)
    @test all(isfinite, vcov(fit))
end

# q=2 data: two responses with a shared 2x2 structured mean effect.
function _j3d_q2_phylo(; p = 16, nrep = 3, seed = 1082)
    rng = MersenneTwister(seed)
    phy = random_balanced_tree(p; branch_length = 0.25)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    P = prior_precision(phy.Q_topology[keep, keep], inv([0.30 0.08; 0.08 0.25]))
    u = cholesky(Symmetric(P)).UP \ randn(rng, size(P, 1))
    pos = Dict(node => i for (i, node) in enumerate(keep))
    leaf = [pos[phy.leaf_indices[k]] for k in 1:p]
    sp = repeat(1:p, inner = nrep); n = length(sp); x = randn(rng, n)
    y1 = [0.5 + 0.4 * x[i] + u[2 * leaf[sp[i]] - 1] + 0.40 * randn(rng) for i in 1:n]
    y2 = [-0.2 + 0.3 * x[i] + u[2 * leaf[sp[i]]] + 0.35 * randn(rng) for i in 1:n]
    return (; y1, y2, x, species = [phy.leaf_names[k] for k in sp]), phy
end

function _j3d_q2_relmat(; G = 24, nrep = 5, seed = 1082)
    rng = MersenneTwister(seed)
    K = [0.55^abs(i - j) for i in 1:G, j in 1:G] + 1e-6I
    P = prior_precision(sparse(Matrix(inv(cholesky(Symmetric(K))))), inv([0.50 0.15; 0.15 0.40]))
    u = cholesky(Symmetric(P)).UP \ randn(rng, size(P, 1))
    grp = repeat(1:G, inner = nrep); n = length(grp); x = randn(rng, n)
    y1 = [0.2 + 0.25 * x[i] + u[2 * grp[i] - 1] + 0.30 * randn(rng) for i in 1:n]
    y2 = [-0.15 + 0.10 * x[i] + u[2 * grp[i]] + 0.35 * randn(rng) for i in 1:n]
    return (; y1, y2, x, grp = ["g$(i)" for i in grp]), K
end

_j3d_q2_formula(term) = bf(mu1 = term(:y1), mu2 = term(:y2),
                           sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                           rho12 = @formula(rho12 ~ 1))

@testset "bivariate Gaussian q=2 structured: every fit warns (no gradient on this route)" begin
    dat, phy = _j3d_q2_phylo()
    fphy = _j3d_q2_formula(y -> y === :y1 ? @formula(y1 ~ x + phylo(1 | species)) :
                                            @formula(y2 ~ x + phylo(1 | species)))
    for method in (:ML, :REML)
        fit = @test_logs (:warn, r"^bivariate Gaussian q=2 structured \(phylo\) vcov: this route has no exact gradient") match_mode = :any drm(
            fphy, Gaussian(); data = dat, tree = phy, method = method)
        @test all(isfinite, vcov(fit))
    end
    datK, K = _j3d_q2_relmat()
    frel = _j3d_q2_formula(y -> y === :y1 ? @formula(y1 ~ x + relmat(1 | grp)) :
                                            @formula(y2 ~ x + relmat(1 | grp)))
    fitK = @test_logs (:warn, r"^bivariate Gaussian q=2 structured \(relmat\) vcov: this route has no exact gradient") match_mode = :any drm(
        frel, Gaussian(); data = datK, K = K, method = :ML)
    @test all(isfinite, vcov(fitK))
    # the warning comes from the vcov build, so it is emitted once per fit
    logs, _ = _j3d_logs(() -> drm(frel, Gaussian(); data = datK, K = K, method = :ML))
    @test count(l -> occursin(_J3D_NOGRAD, string(l.message)), logs) == 1
    # a bootstrap refit (`_without_boundary_warnings`) discards its vcov and stays quiet
    logs, fitq = _j3d_logs(() -> DRModels._without_boundary_warnings(
        () -> drm(frel, Gaussian(); data = datK, K = K, method = :ML)))
    @test !_j3d_has(logs, _J3D_NOGRAD)
    @test vcov(fitq) ≈ vcov(fitK) rtol = 1e-12  # same fit, same SEs: only the message is muted
end

@testset "Gaussian phylo mean: both arms have an exact score, so neither warns about a missing gradient" begin
    rng = MersenneTwister(1082)
    p = 40
    phy = random_balanced_tree(p; branch_length = 0.2)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    u = cholesky(Symmetric(prior_precision(phy.Q_topology[keep, keep], fill(1 / 0.4, 1, 1)))).UP \
        randn(rng, length(keep))
    pos = Dict(node => i for (i, node) in enumerate(keep))
    x = randn(rng, p)
    y = [0.4 + 0.6 * x[k] + u[pos[phy.leaf_indices[k]]] + 0.4 * randn(rng) for k in 1:p]
    dat = (; y, x, species = copy(phy.leaf_names))
    f = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1))
    for method in (:ML, :REML)
        logs, fit = _j3d_logs(() -> drm(f, Gaussian(); data = dat, tree = phy, method = method))
        @test !_j3d_has(logs, _J3D_NOGRAD)   # ML: profiled score; REML: exact restricted score
        @test fit.nllgrad !== nothing
        @test all(isfinite, vcov(fit))
    end
end
