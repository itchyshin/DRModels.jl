# Gaussian meta-analysis with known sampling variances PLUS random intercepts on
# the mean: drmTMB's `meta_V(V = v) + (1 | study)`, `+ phylo(1 | sp)`,
# `+ relmat(1 | id)` and sums of these (Arc 2, `_fit_meta_gaussian_re`).
#
# Before this route the Gaussian router returned the meta_V-only fit for
# `meta_V + (1 | g)` (random effect dropped) and the dense structured fit for
# `meta_V + phylo/relmat` (known variances dropped). The tests below are
# relationships checked in-process (D-277), plus one same-target comparison
# against the numbers drmTMB (engine = "tmb") produced with the committed
# script docs/dev-log/evidence/arc2-metav-random-effect/native-fit.R.
using DRModels
using Test, Random, LinearAlgebra, Statistics, StableRNGs

const _MRE_DIR = joinpath(@__DIR__, "..", "docs", "dev-log", "evidence", "arc2-metav-random-effect")

# Dense reference: -0.5 [logdet Ω + rᵀΩ⁻¹r + n log 2π], Ω = diag(v + σ²) + Σ s_k² Z_k C_k Z_kᵀ.
function _mre_dense_loglik(y, X, β, v, σ2, comps)
    n = length(y)
    Ω = Matrix(Diagonal(v .+ σ2))
    for (gidx, C, s) in comps
        Z = zeros(n, size(C, 1))
        for i in 1:n
            Z[i, gidx[i]] = 1.0
        end
        Ω .+= s^2 .* (Z * C * Z')
    end
    F = cholesky(Symmetric(Ω))
    r = y .- X * β
    return -0.5 * (logdet(F) + dot(r, F \ r) + n * log(2π))
end

function _mre_study_data(rng; S = 30, m = 5, sd_study = 0.6, tau = 0.3)
    n = S * m
    study = repeat(1:S, inner = m)
    x = randn(rng, n)
    v = 0.05 .+ 0.25 .* rand(rng, n)
    u = sd_study .* randn(rng, S)
    y = 0.3 .+ 0.5 .* x .+ u[study] .+ tau .* randn(rng, n) .+ sqrt.(v) .* randn(rng, n)
    return (; y, x, v, study = string.("s", study))
end

@testset "meta_V + random effect: the random effect is fitted, not dropped" begin
    d = _mre_study_data(StableRNG(20260924))
    f_re = bf(@formula(y ~ x + meta_V(v) + (1 | study)), @formula(sigma ~ 1))
    f_meta = bf(@formula(y ~ x + meta_V(v)), @formula(sigma ~ 1))
    fit = drm(f_re, Gaussian(); data = d)
    only_meta = drm(f_meta, Gaussian(); data = d)
    @test fit.converged
    @test length(coef(fit)) == 4                      # mu (2) + sigma + sd(study)
    @test Dict(fit.coefnames)[:resd] == ["study"]
    @test length(coef(only_meta)) == 3
    # Nested models: adding the study field can only raise the likelihood, and
    # on this draw (sd(study) = 0.6) it raises it a lot.
    @test loglik(fit) > loglik(only_meta) + 10

    # The objective IS the dense marginal N(Xβ, diag(v + σ²) + s² ZZᵀ).
    θ = coef(fit)
    gidx, G = DRModels._group_index(d.study)
    X = hcat(ones(length(d.y)), d.x)
    ref = _mre_dense_loglik(d.y, X, θ[1:2], d.v, exp(2θ[3]), [(gidx, Matrix(1.0I, G, G), exp(θ[4]))])
    @test loglik(fit) ≈ ref atol = 1e-9

    # The reported scale is the conditional observation SD √(v + σ²), the
    # meta_V-only route's convention, so the bridge still recovers τ and V_known.
    @test fit.scales[:sigma] ≈ sqrt.(d.v .+ exp(2θ[3]))
    b = drm_bridge(; formula = Dict(:mu => "y ~ x + meta_V(v) + (1 | study)", :sigma => "sigma ~ 1"),
                   family = "gaussian", data = d)
    @test b["loglik"] ≈ loglik(fit) atol = 1e-10
    @test b["df"] == 4
    @test maximum(abs.(b["V_known"] .- d.v)) < 1e-10
    @test all(b["dpars"]["sigma"] .≈ exp(θ[3]))

    # BLUPs are E[u | y] = s² Zᵀ Ω⁻¹ r.
    Z = zeros(length(d.y), G)
    for i in eachindex(gidx)
        Z[i, gidx[i]] = 1.0
    end
    Ω = Diagonal(d.v .+ exp(2θ[3])) + exp(2θ[4]) .* (Z * Z')
    @test ranef(fit)[:study] ≈ exp(2θ[4]) .* (Z' * (Symmetric(Matrix(Ω)) \ (d.y .- X * θ[1:2]))) atol = 1e-8
end

@testset "meta_V + random effect: sigma ~ x keeps the known variances additive" begin
    d = _mre_study_data(StableRNG(7); S = 24, m = 6)
    fit = drm(bf(@formula(y ~ x + meta_V(v) + (1 | study)), @formula(sigma ~ x)), Gaussian(); data = d)
    @test fit.converged
    @test length(coef(fit)) == 5
    θ = coef(fit)
    gidx, G = DRModels._group_index(d.study)
    X = hcat(ones(length(d.y)), d.x)
    σ2 = exp.(2 .* (θ[3] .+ θ[4] .* d.x))
    ref = _mre_dense_loglik(d.y, X, θ[1:2], d.v, σ2, [(gidx, Matrix(1.0I, G, G), exp(θ[5]))])
    @test loglik(fit) ≈ ref atol = 1e-9
end

@testset "meta_V + random effect: v → 0 recovers the verified no-meta routes" begin
    # With a negligible known variance the model IS the ordinary Gaussian random-
    # effect model, so the new route must land on the existing routes' optimum.
    rng = StableRNG(11)
    d = _mre_study_data(rng)
    d0 = merge(d, (; v = fill(1e-12, length(d.y))))
    a = drm(bf(@formula(y ~ x + meta_V(v) + (1 | study)), @formula(sigma ~ 1)), Gaussian(); data = d0)
    b = drm(bf(@formula(y ~ x + (1 | study)), @formula(sigma ~ 1)), Gaussian(); data = d0)
    @test loglik(a) ≈ loglik(b) atol = 1e-6
    @test coef(a) ≈ coef(b) rtol = 1e-5

    # phylo(1 | sp): the default phylo-mean route (sparse, RAW branch-length
    # scale). Agreement of the SD here pins the reporting scale too.
    phy = random_balanced_tree(16; branch_length = 0.25)
    sp = repeat(phy.leaf_names, inner = 4)
    n = length(sp)
    C = DRModels._phylo_correlation(phy)
    a_sp = cholesky(Symmetric(C)).L * (0.7 .* randn(rng, 16))
    lidx = DRModels._phylo_mean_leaf_index(phy, sp)
    x = randn(rng, n)
    y = 0.2 .+ 0.4 .* x .+ a_sp[lidx] .+ 0.3 .* randn(rng, n)
    dp = (; y, x, sp, v = fill(1e-12, n))
    pa = drm(bf(@formula(y ~ x + meta_V(v) + phylo(1 | sp)), @formula(sigma ~ 1)), Gaussian();
             data = dp, tree = phy)
    pb = drm(bf(@formula(y ~ x + phylo(1 | sp)), @formula(sigma ~ 1)), Gaussian();
             data = dp, tree = phy)
    @test loglik(pa) ≈ loglik(pb) atol = 1e-6
    @test coef(pa, :mu) ≈ coef(pb, :mu) rtol = 1e-5
    @test re_sd(pa)[:sp] ≈ re_sd(pb)[:sp] rtol = 1e-4

    # relmat(1 | id): the dense structured route.
    K = C                                    # any PD relatedness matrix will do
    dr = (; y, x, id = sp, v = fill(1e-12, n))
    ra = drm(bf(@formula(y ~ x + meta_V(v) + relmat(1 | id)), @formula(sigma ~ 1)), Gaussian();
             data = dr, K = K)
    rb = drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1)), Gaussian(); data = dr, K = K)
    @test loglik(ra) ≈ loglik(rb) atol = 1e-6
    @test coef(ra) ≈ coef(rb) rtol = 1e-4
end

@testset "meta_V + phylo + (1 | study): two fields, dense reference" begin
    rng = StableRNG(5)
    phy = random_balanced_tree(12; branch_length = 0.2)
    sp = repeat(phy.leaf_names, inner = 6)
    n = length(sp)
    study = string.("s", repeat(1:18, outer = 4))
    x = randn(rng, n); v = 0.05 .+ 0.25 .* rand(rng, n)
    y = 0.1 .+ 0.5 .* x .+ 0.4 .* randn(rng, n) .+ sqrt.(v) .* randn(rng, n)
    d = (; y, x, v, sp, study)
    fit = drm(bf(@formula(y ~ x + meta_V(v) + phylo(1 | sp) + (1 | study)), @formula(sigma ~ 1)),
              Gaussian(); data = d, tree = phy)
    @test length(coef(fit)) == 5
    @test Dict(fit.coefnames)[:resd] == ["study", "sp"]   # ordinary bars first, then structured
    θ = coef(fit)
    gs, Gs = DRModels._group_index(study)
    gp = DRModels._phylo_mean_leaf_index(phy, sp)
    Craw = sigma_phy_dense(phy; σ²_phy = 1.0)
    ref = _mre_dense_loglik(y, hcat(ones(n), x), θ[1:2], v, exp(2θ[3]),
        [(gs, Matrix(1.0I, Gs, Gs), exp(θ[4])), (gp, Craw, exp(θ[5]))])
    @test loglik(fit) ≈ ref atol = 1e-9
end

@testset "meta_V + random effect: bootstrap draws the marginal model" begin
    d = _mre_study_data(StableRNG(21); S = 12, m = 4)
    fit = drm(bf(@formula(y ~ x + meta_V(v) + (1 | study)), @formula(sigma ~ 1)), Gaussian(); data = d)
    sim = DRModels._marginal_simulator(fit, d)
    @test sim !== nothing
    θ = coef(fit)
    Y = reduce(hcat, [sim(StableRNG(k)) for k in 1:4000])
    # Per-row variance of a draw is v + σ² + s²; the study field makes rows of
    # one study covary by s². Both must hold (MC tolerance, 4000 draws).
    @test mean(vec(var(Y; dims = 2)) ./ (d.v .+ exp(2θ[3]) .+ exp(2θ[4]))) ≈ 1 atol = 0.05
    @test cov(Y[1, :], Y[2, :]) ≈ exp(2θ[4]) rtol = 0.15      # rows 1, 2 share study s1
    @test abs(cov(Y[1, :], Y[5, :])) < 0.1 * exp(2θ[4])      # rows 1, 5 do not

    phy = random_balanced_tree(8; branch_length = 0.2)
    sp = repeat(phy.leaf_names, inner = 6)
    n = length(sp)
    d2 = (; y = randn(StableRNG(4), n), x = randn(StableRNG(6), n),
          v = fill(0.1, n), sp, study = string.("s", repeat(1:12, outer = 4)))
    two = drm(bf(@formula(y ~ x + meta_V(v) + phylo(1 | sp) + (1 | study)), @formula(sigma ~ 1)),
              Gaussian(); data = d2, tree = phy)
    @test_throws ArgumentError DRModels._marginal_simulator(two, d2; tree = phy)
end

# The phylo field in a bootstrap draw must sit on the tree leaves the FIT used:
# rows map to leaves by name (`_phylo_mean_leaf_index`), not by the order species
# first appear in the data. Before the fix the simulator used first-seen order,
# so with species listed L1, L3, L5, … the sister tips L1/L2 drew covariance
# -0.005 against 0.294 in the model, and a tree with tips absent from the data
# returned `nothing` (conditional fallback: no phylo field at all on meta_V).
function _mre_phylo_data(phy, sp; seed = 3, sd_phy = 0.8, v = 0.05)
    rng = StableRNG(seed)
    n = length(sp)
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    a = cholesky(Symmetric(C)).L * (sd_phy .* randn(rng, phy.n_leaves))
    leaf = DRModels._phylo_mean_leaf_index(phy, sp)
    x = randn(rng, n)
    y = 0.2 .+ 0.3 .* x .+ a[leaf] .+ 0.2 .* randn(rng, n) .+ sqrt(v) .* randn(rng, n)
    return (; y, x, v = fill(v, n), sp)
end

@testset "phylo bootstrap draws the field on the fitted tree leaves" begin
    phy = random_balanced_tree(8; branch_length = 0.5)     # height 1.5
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    L = phy.leaf_names
    first_row(sp, name) = findfirst(==(name), sp)
    fmeta = bf(@formula(y ~ x + meta_V(v) + phylo(1 | sp)), @formula(sigma ~ 1))
    fplain = bf(@formula(y ~ x + phylo(1 | sp)), @formula(sigma ~ 1))
    # (a) every tip present, species first seen in a NON-tip order;
    # (b) two tips absent from the data (the fit keeps them in the prior).
    shapes = (permuted = repeat(L[[1, 3, 5, 7, 2, 4, 6, 8]], inner = 10),
              subset = repeat(L[[3, 1, 6, 2, 5, 4]], inner = 10))
    for (label, sp) in pairs(shapes), (route, f) in ((:meta, fmeta), (:plain, fplain))
        d = _mre_phylo_data(phy, sp)
        fit = drm(f, Gaussian(); data = d, tree = phy)
        s2 = re_sd(fit)[:sp]^2
        sim = DRModels._marginal_simulator(fit, d; tree = phy)
        @test sim !== nothing
        Y = reduce(hcat, [sim(StableRNG(k)) for k in 1:4000])
        i, j, k = first_row(sp, L[1]), first_row(sp, L[2]), first_row(sp, L[3])
        # Model covariances: sisters L1/L2 share 1.0 of the 1.5 height, L1/L3 0.5.
        # MC SE ≈ 0.01 at 4000 draws; the first-seen-order bug missed by ≥ 0.15.
        @test cov(Y[i, :], Y[j, :]) ≈ s2 * C[1, 2] atol = 0.05
        @test cov(Y[i, :], Y[k, :]) ≈ s2 * C[1, 3] atol = 0.05
        # Row variance: phylo field + residual (√(v + σ²) on meta_V, σ plain).
        @test var(Y[i, :]) ≈ s2 * C[1, 1] + fit.scales[:sigma][i]^2 rtol = 0.1
    end

    # A meta_V + phylo fit whose simulator cannot be built REFUSES: returning
    # `nothing` would send `bootstrap` to the conditional `simulate`, which on this
    # route has no random field at all. The plain phylo route is unchanged
    # (no tree → `nothing`, its pre-existing conditional fallback).
    sp = shapes.permuted
    d = _mre_phylo_data(phy, sp)
    fm = drm(fmeta, Gaussian(); data = d, tree = phy)
    @test_throws ArgumentError DRModels._marginal_simulator(fm, d)
    fp = drm(fplain, Gaussian(); data = d, tree = phy)
    @test DRModels._marginal_simulator(fp, d) === nothing

    # Relabelling rows with integer tip indices (the second mapping tier) is
    # the same model, so it must give the SAME draws, value for value.
    leaf = DRModels._phylo_mean_leaf_index(phy, sp)
    dint = merge(d, (; sp = leaf))
    fmi = drm(fmeta, Gaussian(); data = dint, tree = phy)
    @test loglik(fmi) ≈ loglik(fm) atol = 1e-8
    @test DRModels._marginal_simulator(fmi, dint; tree = phy)(StableRNG(7)) ≈
          DRModels._marginal_simulator(fm, d; tree = phy)(StableRNG(7)) atol = 1e-6

    # End to end: a small bootstrap on the subset-tip tree runs on the marginal
    # simulator and brackets the estimate.
    ds = _mre_phylo_data(phy, shapes.subset)
    fs = drm(fmeta, Gaussian(); data = ds, tree = phy)
    res = bootstrap_result(fs; data = ds, tree = phy, B = 8, rng = StableRNG(11),
                           failures = :skip, check_converged = false)
    @test all(r -> isfinite(r.lower) && isfinite(r.upper), res.summary)
end

@testset "meta_V + random effect: same target as drmTMB (committed native-fit.R numbers)" begin
    function rd(name)
        lines = filter(!isempty, readlines(joinpath(_MRE_DIR, "fixtures", name * ".tsv")))
        hdr = split(lines[1], '\t')
        cols = Dict(h => [split(l, '\t')[j] for l in lines[2:end]] for (j, h) in enumerate(hdr))
        nt = (; y = parse.(Float64, cols["y"]), x = parse.(Float64, cols["x"]), v = parse.(Float64, cols["v"]))
        haskey(cols, "study") && (nt = merge(nt, (; study = String.(cols["study"]))))
        haskey(cols, "sp") && (nt = merge(nt, (; sp = String.(cols["sp"]))))
        return nt
    end
    native = let lines = filter(!isempty, readlines(joinpath(_MRE_DIR, "native.tsv")))
        hdr = split(lines[1], '\t')
        [Dict(zip(hdr, split(l, '\t'))) for l in lines[2:end]]
    end
    nat(fx, q) = only(filter(r -> r["fixture"] == fx && r["quantity"] == q, native))

    fit = drm(bf(@formula(y ~ x + meta_V(v) + (1 | study)), @formula(sigma ~ 1)), Gaussian();
              data = rd("study-a"))
    @test length(coef(fit)) == parse(Int, nat("study-a", "mu:x")["df"])
    @test loglik(fit) ≈ parse(Float64, nat("study-a", "mu:x")["logLik"]) atol = 1e-6
    @test coef(fit, :mu)[2] ≈ parse(Float64, nat("study-a", "mu:x")["estimate"]) rtol = 1e-5
    @test coef(fit, :sigma)[1] ≈ parse(Float64, nat("study-a", "sigma:(Intercept)")["estimate"]) rtol = 1e-5
    @test re_sd(fit)[:study] ≈ parse(Float64, nat("study-a", "sd:(1 | study)")["estimate"]) rtol = 1e-5

    tree = read(joinpath(_MRE_DIR, "fixtures", "phylo-b.nwk"), String)   # height 3
    pfit = drm(bf(@formula(y ~ x + meta_V(v) + phylo(1 | sp)), @formula(sigma ~ 1)), Gaussian();
               data = rd("phylo-b"), tree = tree)
    @test loglik(pfit) ≈ parse(Float64, nat("phylo-b", "mu:x")["logLik"]) atol = 1e-6
    @test coef(pfit, :mu)[2] ≈ parse(Float64, nat("phylo-b", "mu:x")["estimate"]) rtol = 1e-5
    # RAW-scale SD × sqrt(tree height) = drmTMB's correlation-scale SD.
    @test re_sd(pfit)[:sp] * sqrt(3.0) ≈
          parse(Float64, nat("phylo-b", "sd:phylo(1 | sp)")["estimate"]) rtol = 1e-5
end

@testset "meta_V neighbours are unchanged (D-273)" begin
    d = _mre_study_data(StableRNG(3))
    X = hcat(ones(length(d.y)), d.x)
    nm = ["(Intercept)", "x"]
    # meta_V alone and with sigma ~ x still dispatch to `_fit_meta_gaussian`.
    for (fσ, Xσ, nmσ) in ((@formula(sigma ~ 1), ones(length(d.y), 1), ["(Intercept)"]),
                          (@formula(sigma ~ x), X, nm))
        viadrm = drm(bf(@formula(y ~ x + meta_V(v)), fσ), Gaussian(); data = d)
        direct = DRModels._fit_meta_gaussian(Gaussian(), d.y, X, Xσ, d.v, nm, nmσ, 1e-8)
        @test coef(viadrm) == coef(direct)
        @test loglik(viadrm) == loglik(direct)
    end
    # (1 | g) without meta_V still takes the verified Woodbury route.
    gidx, G = DRModels._group_index(d.study)
    viadrm = drm(bf(@formula(y ~ x + (1 | study)), @formula(sigma ~ 1)), Gaussian(); data = d)
    direct = DRModels._fit_ranef_gaussian(Gaussian(), d.y, X, ones(length(d.y), 1), gidx, G,
        ones(length(d.y)), nm, ["(Intercept)"], :study, 1e-8)
    @test coef(viadrm) == coef(direct)
    @test loglik(viadrm) == loglik(direct)
end

@testset "meta_V + random effect: unsupported shapes refuse loudly" begin
    d = _mre_study_data(StableRNG(9))
    d2 = merge(d, (; g2 = d.study))
    mv(f) = drm(f, Gaussian(); data = d2)
    @test_throws ArgumentError mv(bf(@formula(y ~ x + meta_V(v) + (0 + x | study)), @formula(sigma ~ 1)))
    @test_throws ArgumentError mv(bf(@formula(y ~ x + meta_V(v) + (1 + x | study)), @formula(sigma ~ 1)))
    @test_throws ArgumentError drm(bf(@formula(y ~ x + meta_V(v) + (1 | study)), @formula(sigma ~ 1)),
        Gaussian(); data = d, method = :REML)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + meta_V(v) + (1 | study)), @formula(sigma ~ 1)),
        Gaussian(); data = d, algorithm = :sparse)
    # Unchanged pre-existing refusals: sd() submodel and a sigma random effect.
    @test_throws Exception drm(bf(@formula(y ~ x + meta_V(v) + (1 | study)), @formula(sigma ~ 1),
        @formula(sd(study) ~ 1)), Gaussian(); data = d)
    @test_throws Exception mv(bf(@formula(y ~ x + meta_V(v)), @formula(sigma ~ 1 + (1 | study))))
    # A phylo field and an ordinary intercept on the SAME grouping column.
    phy = random_balanced_tree(10; branch_length = 0.2)
    sp = repeat(phy.leaf_names, inner = 5)
    n = length(sp)
    ds = (; y = randn(StableRNG(1), n), x = randn(StableRNG(2), n), v = fill(0.1, n), sp)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + meta_V(v) + phylo(1 | sp) + (1 | sp)),
        @formula(sigma ~ 1)), Gaussian(); data = ds, tree = phy)
end
