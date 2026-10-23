# Arc 2 `structured_with_ordinary_bar`: ONE structured marker (phylo / relmat /
# animal) PLUS ordinary `(1 | h)` / `(0 + x | h)` bars on the Gaussian mean.
# Before this route the dispatcher sent the formula to the single-structured
# fitter, which silently DROPPED every ordinary bar (measured through drmTMB's
# bridge at da8b3f871: df 4 and the marker-only logLik, vs native df 5). The
# model is drmTMB's: independent blocks, V = D + Σ_k σ_k² Z_k K_k Z_kᵀ with
# K = I for an ordinary bar. The same-target comparison against native drmTMB
# lives in docs/dev-log/evidence/arc2-structured-ordinary-bar/ (committed
# R + Julia scripts); these tests are in-process relationships (D-277).
using DRModels
using Test, Random, LinearAlgebra

# Independent dense Gaussian marginal log-likelihood, written out here from the
# model definition (not from the fitter), for the objective check.
function _sob_dense_loglik(y, X, βμ, σ_resid, comps)
    n = length(y)
    V = Matrix(Diagonal(σ_resid .^ 2 .* ones(n)))
    for (σk, w, gidx, K) in comps
        for i in 1:n, j in 1:n
            V[i, j] += σk^2 * w[i] * w[j] * K[gidx[i], gidx[j]]
        end
    end
    r = y .- X * βμ
    ch = cholesky(Symmetric(V))
    return -0.5 * (logdet(ch) + dot(r, ch \ r) + n * log(2π))
end

function _sob_phylo_fixture(seed; ntip = 16, nh = 6, σp = 0.8, σh = 0.7, σe = 0.5)
    Random.seed!(seed)
    phy = random_balanced_tree(ntip; branch_length = 0.3)
    C = DRModels._phylo_correlation(phy)
    species = repeat(phy.leaf_names, outer = nh)
    h = repeat(["h$k" for k in 1:nh], inner = ntip)
    n = length(species)
    x = randn(n)
    a = σp .* (cholesky(Symmetric(C)).L * randn(ntip))
    b = σh .* randn(nh)
    leaf = Dict(nm => i for (i, nm) in enumerate(phy.leaf_names))
    hix = Dict("h$k" => k for k in 1:nh)
    y = [1.0 + 0.5 * x[i] + a[leaf[species[i]]] + b[hix[h[i]]] + σe * randn() for i in 1:n]
    return phy, C, (; y, x, species, h)
end

@testset "phylo(1 | sp) + (1 | h): the ordinary bar is kept (was dropped)" begin
    phy, C, data = _sob_phylo_fixture(20260924)
    f = bf(@formula(y ~ x + phylo(1 | species) + (1 | h)), @formula(sigma ~ 1))
    fit = drm(f, Gaussian(); data = data, tree = phy)
    marker_only = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                      Gaussian(); data = data, tree = phy)
    @test is_converged(fit)
    # df = 2 mean + 1 sigma + 2 SDs (native drmTMB df 5; the dropped fit had 4).
    @test dof(fit) == 5
    @test dof(marker_only) == 4
    @test first(n for (p, n) in fit.coefnames if p === :resd) == ["h", "species"]
    # The marker-only model is the σ_h → 0 boundary of this one, so the full
    # fit cannot be worse, and on this fixture (σ_h = 0.7) it is far better.
    @test loglik(fit) > loglik(marker_only) + 5
    # The reported logLik IS the Gaussian marginal of the stated model at θ̂.
    θ = coef(fit)
    sds = re_sd(fit)
    leaf = Dict(nm => i for (i, nm) in enumerate(phy.leaf_names))
    gsp = [leaf[s] for s in data.species]
    gh, _ = DRModels._group_index(data.h)
    X = hcat(ones(length(data.y)), data.x)
    ll = _sob_dense_loglik(data.y, X, θ[1:2], exp(θ[3]),
        [(sds[:h], ones(length(data.y)), gh, Matrix(1.0I, 6, 6)),
         (sds[:species], ones(length(data.y)), gsp, C)])
    @test loglik(fit) ≈ ll atol = 1e-8
    # vc / ranef carry both components under distinct keys.
    @test Set(keys(vc(fit))) == Set([:h, :species])
    @test length(ranef(fit)[:h]) == 6
    @test length(ranef(fit)[:species]) == 16
    # The marker SD is on the CORRELATION scale (drmTMB's `ape::vcv(tree, corr =
    # TRUE)`), so it does not move when every branch is doubled. (The marker-only
    # sparse route reports the RAW branch-length scale instead; an R caller must
    # not apply its sqrt(tree height) rescaling to this route's `resd` block.)
    phy2 = random_balanced_tree(16; branch_length = 0.6)
    @test phy2.leaf_names == phy.leaf_names
    fit2 = drm(f, Gaussian(); data = data, tree = phy2)
    @test loglik(fit2) ≈ loglik(fit) atol = 1e-8
    @test re_sd(fit2)[:species] ≈ sds[:species] rtol = 1e-6
end

@testset "phylo(1 | sp) + (1 | h): rows go to tips BY NAME (shuffled rows, absent tip)" begin
    # A first-seen level order would put rows on the wrong tips whenever the
    # data's species order differs from the tree's. Drop every row of one tip
    # and shuffle the rest: the fit must be invariant to row order and equal the
    # dense marginal with rows mapped to tips by name (G = all 16 tips).
    phy, C, full = _sob_phylo_fixture(20260925)
    keep = findall(!=(phy.leaf_names[3]), full.species)
    sub = map(v -> v[keep], full)
    perm = randperm(Random.MersenneTwister(7), length(keep))
    shuf = map(v -> v[perm], sub)
    @test unique(shuf.species) != filter(!=(phy.leaf_names[3]), phy.leaf_names)
    f = bf(@formula(y ~ x + phylo(1 | species) + (1 | h)), @formula(sigma ~ 1))
    fit_sub = drm(f, Gaussian(); data = sub, tree = phy)
    fit_shuf = drm(f, Gaussian(); data = shuf, tree = phy)
    @test is_converged(fit_shuf) && dof(fit_shuf) == 5
    @test loglik(fit_shuf) ≈ loglik(fit_sub) atol = 1e-8
    @test coef(fit_shuf) ≈ coef(fit_sub) rtol = 1e-5
    @test length(ranef(fit_shuf)[:species]) == 16
    θ = coef(fit_shuf); sds = re_sd(fit_shuf); n = length(shuf.y)
    leaf = Dict(nm => i for (i, nm) in enumerate(phy.leaf_names))
    gh, Gh = DRModels._group_index(shuf.h)
    @test loglik(fit_shuf) ≈ _sob_dense_loglik(shuf.y, hcat(ones(n), shuf.x), θ[1:2], exp(θ[3]),
        [(sds[:h], ones(n), gh, Matrix(1.0I, Gh, Gh)),
         (sds[:species], ones(n), [leaf[s] for s in shuf.species], C)]) atol = 1e-8
    # The phylo SD is on the tip-correlation scale, recorded for the bootstrap.
    @test fit_shuf.phylo_scale === :correlation
end

@testset "bootstrap refuses marker + ordinary bar (the simulator draws one field)" begin
    phy, _, data = _sob_phylo_fixture(20260926)
    fit = drm(bf(@formula(y ~ x + phylo(1 | species) + (1 | h)), @formula(sigma ~ 1)),
              Gaussian(); data = data, tree = phy)
    @test_throws ArgumentError DRModels._marginal_simulator(fit, data; tree = phy)
    @test_throws ArgumentError bootstrap_result(fit; data = data, tree = phy, B = 5)
    Random.seed!(1); M = randn(6, 6); K = M * M' / 6 + I
    rfit = drm(bf(@formula(y ~ x + relmat(1 | h) + (1 | species)), @formula(sigma ~ 1)),
               Gaussian(); data = data, K = K)
    @test_throws ArgumentError DRModels._marginal_simulator(rfit, data; K = K)
end

@testset "relmat(K = I) + (1 | h) is the two-bar (1 | id) + (1 | h) model" begin
    Random.seed!(924001)
    nid, nh = 20, 5
    id = repeat(["i$k" for k in 1:nid], outer = nh)
    h = repeat(["h$k" for k in 1:nh], inner = nid)
    n = length(id)
    x = randn(n)
    u = 0.8 .* randn(nid); b = 0.6 .* randn(nh)
    idx = Dict("i$k" => k for k in 1:nid); hx = Dict("h$k" => k for k in 1:nh)
    y = [1.0 + 0.5 * x[i] + u[idx[id[i]]] + b[hx[h[i]]] + exp(-0.6 + 0.3 * x[i]) * randn()
         for i in 1:n]
    data = (; y, x, id, h)
    # heteroscedastic residual: the route honours `sigma ~ x` (D -> diag)
    structured = drm(bf(@formula(y ~ x + relmat(1 | id) + (1 | h)), @formula(sigma ~ x)),
                     Gaussian(); data = data, K = Matrix(1.0I, nid, nid))
    two_bars = drm(bf(@formula(y ~ x + (1 | id) + (1 | h)), @formula(sigma ~ x)),
                   Gaussian(); data = data)
    @test dof(structured) == dof(two_bars) == 6
    @test loglik(structured) ≈ loglik(two_bars) atol = 1e-6
    @test coef(structured, :mu) ≈ coef(two_bars, :mu) rtol = 1e-4
    @test coef(structured, :sigma) ≈ coef(two_bars, :sigma) rtol = 1e-4
    @test re_sd(structured)[:id] ≈ re_sd(two_bars)[:id] rtol = 1e-4
    @test re_sd(structured)[:h] ≈ re_sd(two_bars)[:h] rtol = 1e-4
    # animal(A = K) is the same engine with the same matrix.
    animal_fit = drm(bf(@formula(y ~ x + animal(1 | id) + (1 | h)), @formula(sigma ~ x)),
                 Gaussian(); data = data, A = Matrix(1.0I, nid, nid))
    @test loglik(animal_fit) ≈ loglik(structured) atol = 1e-10
end

@testset "(1 | sp) + phylo(1 | sp) is the sd() router with intercept-only sd() parts" begin
    Random.seed!(924002)
    ntip, reps = 16, 6
    phy = random_balanced_tree(ntip; branch_length = 0.3)
    C = DRModels._phylo_correlation(phy)
    species = repeat(phy.leaf_names, inner = reps)
    n = length(species)
    x = randn(n)
    a = 0.8 .* (cholesky(Symmetric(C)).L * randn(ntip)); u = 0.5 .* randn(ntip)
    leaf = Dict(nm => i for (i, nm) in enumerate(phy.leaf_names))
    y = [1.0 + 0.5 * x[i] + a[leaf[species[i]]] + u[leaf[species[i]]] + 0.5 * randn() for i in 1:n]
    data = (; y, x, species)
    fit = drm(bf(@formula(y ~ x + (1 | species) + phylo(1 | species)), @formula(sigma ~ 1)),
              Gaussian(); data = data, tree = phy)
    router = drm(bf(@formula(y ~ x + (1 | species) + phylo(1 | species)), @formula(sigma ~ 1),
                    @formula(sd(species) ~ 1), @formula(sd(species, phylogenetic) ~ 1)),
                 Gaussian(); data = data, tree = phy)
    # the ordinary intercept sharing the marker's grouping is keyed `<g>_iid`
    @test first(n for (p, n) in fit.coefnames if p === :resd) == ["species_iid", "species"]
    @test dof(fit) == dof(router) == 5
    @test loglik(fit) ≈ loglik(router) atol = 1e-6
    @test coef(fit, :mu) ≈ coef(router, :mu) rtol = 1e-4
    @test re_sd(fit)[:species_iid] ≈ exp(only(coef(router, :sd))) rtol = 1e-4
    @test re_sd(fit)[:species] ≈ exp(only(coef(router, :sd_phylo))) rtol = 1e-4
end

@testset "phylo(1 | sp) + (0 + x | h) and two ordinary bars" begin
    phy, C, data0 = _sob_phylo_fixture(924003)
    n = length(data0.y)
    g = ["g$(mod1(i, 5))" for i in 1:n]
    # add a real h-level slope and a real g effect, so neither SD sits on the
    # zero boundary (a boundary fit would weaken the objective check)
    hx = Dict("h$k" => k for k in 1:6)
    bslope = 0.6 .* randn(6); ug = 0.6 .* randn(5)
    y = data0.y .+ [bslope[hx[data0.h[i]]] * data0.x[i] + ug[mod1(i, 5)] for i in 1:n]
    data = merge(data0, (; y, g))
    slope = drm(bf(@formula(y ~ x + phylo(1 | species) + (0 + x | h)), @formula(sigma ~ 1)),
                Gaussian(); data = data, tree = phy)
    @test dof(slope) == 5
    @test first(n for (p, n) in slope.coefnames if p === :resd) == ["h:x", "species"]
    θ = coef(slope); sds = re_sd(slope)
    leaf = Dict(nm => i for (i, nm) in enumerate(phy.leaf_names))
    gh, Gh = DRModels._group_index(data.h)
    X = hcat(ones(n), data.x)
    @test loglik(slope) ≈ _sob_dense_loglik(data.y, X, θ[1:2], exp(θ[3]),
        [(sds[Symbol("h:x")], data.x, gh, Matrix(1.0I, Gh, Gh)),
         (sds[:species], ones(n), [leaf[s] for s in data.species], C)]) atol = 1e-8
    two = drm(bf(@formula(y ~ x + phylo(1 | species) + (1 | h) + (1 | g)), @formula(sigma ~ 1)),
              Gaussian(); data = data, tree = phy)
    @test dof(two) == 6
    @test first(n for (p, n) in two.coefnames if p === :resd) == ["h", "g", "species"]
end

@testset "drm_bridge: the R entry point returns the full model" begin
    phy, _, data = _sob_phylo_fixture(924004)
    direct = drm(bf(@formula(y ~ x + phylo(1 | species) + (1 | h)), @formula(sigma ~ 1)),
                 Gaussian(); data = data, tree = phy)
    out = drm_bridge(; formula = "y ~ x + phylo(1 | species) + (1 | h); sigma ~ 1",
                     family = "gaussian", data = data, tree = phy)
    @test out["df"] == 5
    @test out["raw_coef_names"] == ["mu_(Intercept)", "mu_x", "sigma_(Intercept)",
                                    "resd_h", "resd_species"]
    @test out["loglik"] ≈ loglik(direct) atol = 1e-10
    @test out["coefficients"] ≈ coef(direct) atol = 1e-10
end

@testset "variants this route does not fit are refused by name, never dropped" begin
    phy, _, data0 = _sob_phylo_fixture(924005)
    n = length(data0.y)
    data = merge(data0, (; v = fill(0.1, n)))
    fit_err(f; kw...) = drm(f, Gaussian(); data = data, tree = phy, kw...)
    @test_throws ArgumentError fit_err(bf(@formula(y ~ x + phylo(1 | species) + (1 + x | h)),
                                          @formula(sigma ~ 1)))
    @test_throws ArgumentError fit_err(bf(@formula(y ~ x + phylo(1 | species) + (1 | h)),
                                          @formula(sigma ~ 1)); method = :REML)
    @test_throws ArgumentError fit_err(bf(@formula(y ~ x + phylo(1 | species) + (1 | h)),
                                          @formula(sigma ~ 1)); algorithm = :sparse)
    # With `meta_V(v)` the combination is FITTED, not refused: #814's
    # `_fit_meta_gaussian_re` route is dispatched first and keeps both fields.
    metafit = fit_err(bf(@formula(y ~ x + phylo(1 | species) + (1 | h) + meta_V(v)),
                         @formula(sigma ~ 1)))
    @test length(coef(metafit)) == 5
    @test Dict(metafit.coefnames)[:resd] == ["h", "species"]
    @test_throws ArgumentError fit_err(bf(@formula(y ~ x + phylo(1 | species) + (1 | h)),
                                          @formula(sigma ~ 1)); penalty = drm_phylo_penalty())
    coords = randn(16, 2)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + spatial(1 | species) + (1 | h)),
                                      @formula(sigma ~ 1)), Gaussian(); data = data, coords = coords)
end

@testset "neighbouring routes are untouched (D-273)" begin
    phy, _, data = _sob_phylo_fixture(924006)
    n = length(data.y)
    X = hcat(ones(n), data.x); Xσ = ones(n, 1)
    nmμ = ["(Intercept)", "x"]; nmσ = ["(Intercept)"]
    # marker only -> the sparse phylo-mean route, byte-identical
    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              Gaussian(); data = data, tree = phy)
    gphy = DRModels._phylo_mean_leaf_index(phy, data.species)
    direct = DRModels._fit_structured_gaussian_sparse_lbfgs(Gaussian(), Float64.(data.y), X, Xσ,
        gphy, phy.n_leaves, phy, nmμ, nmσ, :species, 1e-8; penalty = nothing, reml = false)
    @test coef(fit) == coef(direct) && loglik(fit) == loglik(direct)
    # relmat only -> the closed-form single-structured fitter, byte-identical
    Random.seed!(1); M = randn(6, 6); K = M * M' / 6 + I
    fit = drm(bf(@formula(y ~ x + relmat(1 | h)), @formula(sigma ~ 1)), Gaussian(); data = data, K = K)
    gh, Gh = DRModels._group_index(data.h)
    direct = DRModels._fit_structured_gaussian(Gaussian(), Float64.(data.y), X, Xσ, gh, Gh, K,
        nmμ, nmσ, :h, 1e-8)
    @test coef(fit) == coef(direct) && loglik(fit) == loglik(direct)
    # ordinary bar only -> the single random-intercept fitter, byte-identical
    fit = drm(bf(@formula(y ~ x + (1 | h)), @formula(sigma ~ 1)), Gaussian(); data = data)
    direct = DRModels._fit_ranef_gaussian(Gaussian(), Float64.(data.y), X, Xσ, gh, Gh, ones(n),
        nmμ, nmσ, :h, 1e-8)
    @test coef(fit) == coef(direct) && loglik(fit) == loglik(direct)
    # the multi-component sd() router still owns sd()-carrying shapes
    router = drm(bf(@formula(y ~ x + phylo(1 | species) + (1 | h)), @formula(sigma ~ 1),
                    @formula(sd(h) ~ 1), @formula(sd(species, phylogenetic) ~ 1)),
                 Gaussian(); data = data, tree = phy)
    @test any(p -> first(p) === :sd_phylo, router.blocks)
    @test !any(p -> first(p) === :resd, router.blocks)
end
