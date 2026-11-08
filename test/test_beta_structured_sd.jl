# Beta random-effect SD regression on phylogenetic, related-matrix, animal and
# coordinate-spatial structures:
#   q1   y ~ x + phylo|relmat|spatial(1 | g),      sd(g, level) ~ z
#   q2   y ~ x + phylo|relmat|spatial|animal(1 + x | g)
#        with sd(g, level, coef(intercept)) ~ z0 and sd(g, level, coef(x)) ~ z1
# The q1 route reuses the Beta animal-intercept SD Laplace objective with the kernel
# swapped; the q2 route is the Gamma/NB2 two-field engine (`_fit_twofield_sd_laplace`)
# with the Beta kernel. Kernels follow native drmTMB: relmat/animal enter RAW, phylo is
# the tip correlation of an ultrametric tree, spatial is exp(-d / median distance) + 1e-6 I.
# The oracle shares no fitter code: a dense ForwardDiff Laplace with the Beta density
# written out, in the latent b ~ N(0, C) coordinates.
# Tolerance (fixed before the first run): |loglik - oracle| < 1e-5; fit converged.
# The committed datasets (test/fixtures/beta_structured_sd/*.tsv) are what the tests fit;
# `_bss_generate` (StableRNG, Beta draws by quantile transform) regenerates them.
using DRModels, Test, Random, LinearAlgebra
import Distributions, ForwardDiff
using StableRNGs: StableRNG
using SpecialFunctions: loggamma

const _BSS_BITS = 4                        # 16 tips / sites / levels

function _bss_tree()
    G = 2^_BSS_BITS
    ids = ["t$(lpad(i, 2, '0'))" for i in 1:G]
    nw(ix, root) = length(ix) == 1 ? "$(ids[ix[1]]):1.5" :
        "(" * nw(ix[1:end÷2], false) * "," * nw(ix[end÷2+1:end], false) * ")" * (root ? ";" : ":0.1")
    height = 1.5 + 0.1 * (_BSS_BITS - 1)
    bits(i) = reverse(digits(i - 1; base = 2, pad = _BSS_BITS))
    C = [i == j ? 1.0 : (findfirst(bits(i) .!= bits(j)) - 1) * 0.1 / height for i in 1:G, j in 1:G]
    return nw(collect(1:G), true), ids, C
end

function _bss_grid(G)
    side = round(Int, sqrt(G))
    xy = [(Float64(i), Float64(j)) for j in 0:side-1 for i in 0:side-1]
    D = [hypot(xy[k][1] - xy[l][1], xy[k][2] - xy[l][2]) for k in 1:G, l in 1:G]
    rng = Distributions.median([D[k, l] for k in 1:G for l in k+1:G])
    return xy, exp.(-D ./ rng) + 1e-6 * I
end

# Raw (non-correlation) covariance, used as `K` (relmat) and `A` (animal).
function _bss_raw_cov(G; rho, lo, hi)
    R = [rho^abs(i - j) for i in 1:G, j in 1:G]
    s = collect(range(lo, hi; length = G))
    return Matrix(Diagonal(s) * R * Diagonal(s))
end

function _bss_kernels()
    nwk, tips, Cphy = _bss_tree()
    xy, Csp = _bss_grid(length(tips))
    G = length(tips)
    return (; nwk, tips, Cphy, Csp, xy,
            coords = Dict(tips[k] => [xy[k]...] for k in 1:G),
            K = _bss_raw_cov(G; rho = 0.5, lo = 0.7, hi = 1.4),
            A = _bss_raw_cov(G; rho = 0.4, lo = 0.8, hi = 1.3))
end

# ---- generator (regeneration only) ----
function _bss_generate(C, ids; q, shuffle, m = 10, seed = 20261108)
    rng = StableRNG(seed)
    G = length(ids)
    z0 = repeat([-1.0, 1.0], G ÷ 2)
    z1 = collect(range(-1.0, 1.0; length = G))[randperm(rng, G)]
    L = cholesky(Symmetric(C)).L
    u0 = exp.(log(0.5) .+ 0.35 .* z0) .* (L * randn(rng, G))
    u1 = q == 2 ? exp.(log(0.35) .- 0.3 .* z1) .* (L * randn(rng, G)) : zeros(G)
    gi = repeat(1:G, inner = m)
    x = repeat(collect(range(-1.2, 1.2; length = m)), G)
    eta = 0.2 .+ 0.4 .* x .+ u0[gi] .+ x .* u1[gi]
    mu = 1 ./ (1 .+ exp.(-eta))
    phi = 1 / 0.3^2
    y = [Distributions.quantile(Distributions.Beta(mu[i] * phi, (1 - mu[i]) * phi), rand(rng))
         for i in eachindex(mu)]
    y = clamp.(y, nextfloat(0.0), prevfloat(1.0))
    ord = shuffle ? randperm(rng, length(y)) : collect(eachindex(y))
    return (; y = y[ord], x = x[ord], z0 = z0[gi][ord], z1 = z1[gi][ord],
            id = ids[gi][ord], gi = gi[ord])
end
# ---- end generator ----

function _bss_fixture(name)
    lines = readlines(joinpath(@__DIR__, "fixtures", "beta_structured_sd", "$name.tsv"))
    rows = [split(l, '\t') for l in lines[2:end]]
    col(j, T) = T === String ? String[r[j] for r in rows] : parse.(T, [r[j] for r in rows])
    return (; y = col(1, Float64), x = col(2, Float64), z0 = col(3, Float64),
              z1 = col(4, Float64), id = col(5, String), gi = col(6, Int))
end

function _bss_pieces(fit)
    b = fit.blocks
    beta = fit.theta[last(b[findfirst(p -> first(p) === :mu, b)])]
    logsig = only(fit.theta[last(b[findfirst(p -> first(p) === :sigma, b)])])
    sdb = [fit.theta[last(p)] for p in b if first(p) !== :mu && first(p) !== :sigma]
    return beta, logsig, sdb
end

# Dense Laplace oracle. `zs` are the group-level SD predictors (one per field) in
# kernel order; `ws` the per-row loadings of each field (1 for the intercept, x for the slope).
function _bss_oracle(fit, d, C, zs, ws)
    beta, logsig, sdb = _bss_pieces(fit)
    G = size(C, 1); gi = d.gi; nf = length(zs)
    sds = [exp.(sdb[k][1] .+ sdb[k][2] .* zs[k]) for k in 1:nf]
    Ci = inv(Symmetric(C))
    phi = exp(-2logsig)
    function joint(b)
        eta = beta[1] .+ beta[2] .* d.x
        for k in 1:nf
            bk = b[((k - 1) * G + 1):(k * G)]
            eta = eta .+ ws[k] .* (sds[k] .* bk)[gi]
        end
        mu = 1 ./ (1 .+ exp.(-eta))
        a, bb = mu .* phi, (1 .- mu) .* phi
        lp = sum(loggamma(phi) .- loggamma.(a) .- loggamma.(bb) .+
                 (a .- 1) .* log.(d.y) .+ (bb .- 1) .* log1p.(-d.y))
        pen = sum(dot(b[((k - 1) * G + 1):(k * G)], Ci * b[((k - 1) * G + 1):(k * G)]) for k in 1:nf)
        return lp - 0.5 * pen
    end
    b = zeros(nf * G)
    for _ in 1:200
        g = ForwardDiff.gradient(joint, b); H = ForwardDiff.hessian(joint, b)
        step = -(Symmetric(H) \ g)
        f0, t = joint(b), 1.0
        while !(joint(b .+ t .* step) >= f0) && t > 1e-8      # damped Newton ascent
            t /= 2
        end
        b .+= t .* step
        maximum(abs, t .* step) < 1e-12 && break
    end
    H = -ForwardDiff.hessian(joint, b)
    return joint(b) - 0.5 * nf * logdet(C) - 0.5 * logdet(Symmetric(H))
end

const _BSS_SHAPES = [
    # name, kind, q, level
    ("relmat_q1", :relmat, 1, :relmat), ("spatial_q1", :spatial, 1, :spatial),
    ("phylo_q1", :phylo, 1, :phylogenetic), ("relmat_q2", :relmat, 2, :relmat),
    ("spatial_q2", :spatial, 2, :spatial), ("phylo_q2", :phylo, 2, :phylogenetic),
    ("animal_q2", :animal, 2, :animal)]

function _bss_formula(kind, q, level)
    mu = Dict((:relmat, 1) => @formula(y ~ x + relmat(1 | id)),
              (:spatial, 1) => @formula(y ~ x + spatial(1 | id)),
              (:phylo, 1) => @formula(y ~ x + phylo(1 | id)),
              (:relmat, 2) => @formula(y ~ x + relmat(1 + x | id)),
              (:spatial, 2) => @formula(y ~ x + spatial(1 + x | id)),
              (:phylo, 2) => @formula(y ~ x + phylo(1 + x | id)),
              (:animal, 2) => @formula(y ~ x + animal(1 + x | id)))[(kind, q)]
    sdf = Dict(:relmat => (@formula(sd(id, relmat) ~ z0), @formula(sd(id, relmat, coef(intercept)) ~ z0), @formula(sd(id, relmat, coef(x)) ~ z1)),
               :spatial => (@formula(sd(id, spatial) ~ z0), @formula(sd(id, spatial, coef(intercept)) ~ z0), @formula(sd(id, spatial, coef(x)) ~ z1)),
               :phylo => (@formula(sd(id, phylogenetic) ~ z0), @formula(sd(id, phylogenetic, coef(intercept)) ~ z0), @formula(sd(id, phylogenetic, coef(x)) ~ z1)),
               :animal => (nothing, @formula(sd(id, animal, coef(intercept)) ~ z0), @formula(sd(id, animal, coef(x)) ~ z1)))[kind]
    return q == 1 ? bf(mu, @formula(sigma ~ 1), sdf[1]) :
                    bf(mu, @formula(sigma ~ 1), sdf[2], sdf[3])
end

@testset "Beta structured SD regression (phylo / relmat / animal / spatial)" begin
    kn = _bss_kernels()
    Cof = Dict(:relmat => kn.K, :animal => kn.A, :phylo => kn.Cphy, :spatial => kn.Csp)
    for (name, kind, q, level) in _BSS_SHAPES
        @testset "$name" begin
            d = _bss_fixture(name)
            kw = kind === :relmat ? (; K = kn.K) : kind === :animal ? (; A = kn.A, ids = unique(d.id)) :
                 kind === :phylo ? (; tree = kn.nwk) : (; coords = kn.coords)
            fit = drm(_bss_formula(kind, q, level), Beta(); data = d, kw...)
            @test fit.converged
            @test length(fit.theta) == (q == 1 ? 5 : 7)
            @test all(isfinite, stderror(fit))
            # Group-level SD predictors in kernel order.
            G = size(Cof[kind], 1)
            gi = d.gi
            z0g = zeros(G); z0g[gi] .= d.z0
            z1g = zeros(G); z1g[gi] .= d.z1
            zs = q == 1 ? [z0g] : [z0g, z1g]
            ws = q == 1 ? [ones(length(d.y))] : [ones(length(d.y)), d.x]
            @test loglikelihood(fit) ≈ _bss_oracle(fit, d, Cof[kind], zs, ws) atol = 1e-5
        end
    end

    @testset "simulate draws the structured effects (q2)" begin
        for (name, kind, q, level) in _BSS_SHAPES
            q == 2 || continue
            d = _bss_fixture(name)
            kw = kind === :relmat ? (; K = kn.K) : kind === :animal ? (; A = kn.A, ids = unique(d.id)) :
                 kind === :phylo ? (; tree = kn.nwk) : (; coords = kn.coords)
            fit = drm(_bss_formula(kind, q, level), Beta(); data = d, kw...)
            ysim = simulate(fit; data = d, Base.structdiff(kw, (; ids = nothing))..., rng = StableRNG(1))
            @test length(ysim) == length(d.y)
            @test all(y -> 0 < y < 1, ysim)
        end
    end

    @testset "refusals and scope" begin
        d = _bss_fixture("phylo_q2")
        fq2 = _bss_formula(:phylo, 2, :phylogenetic)
        fq1 = bf(@formula(y ~ x + phylo(1 | id)), @formula(sigma ~ 1), @formula(sd(id, phylogenetic) ~ z0))
        # Non-ultrametric tree: drmTMB's correlation scale requires ultrametricity.
        bad = replace(kn.nwk, "t01:1.5" => "t01:2.5")
        @test_throws r"ultrametric" drm(fq2, Beta(); data = d, tree = bad)
        @test_throws r"ultrametric" drm(fq1, Beta(); data = d, tree = bad)
        # Wrong provider keyword (the entry guard refuses an unused `K`).
        @test_throws r"`K = …` was supplied" drm(fq2, Beta(); data = d, tree = kn.nwk, K = kn.K)
        # A tip with no rows.
        keep = d.id .!= "t07"
        sub = map(v -> v[keep], d)
        @test_throws r"every tip" drm(fq2, Beta(); data = sub, tree = kn.nwk)
        # relmat without its covariance matrix.
        dk = _bss_fixture("relmat_q1")
        fk = bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1), @formula(sd(id, relmat) ~ z0))
        @test_throws Exception drm(fk, Beta(); data = dk)
        # One coefficient formula is not enough for an intercept-plus-slope block.
        one = bf(@formula(y ~ x + phylo(1 + x | id)), @formula(sigma ~ 1),
                 @formula(sd(id, phylogenetic, coef(x)) ~ z1))
        @test_throws ArgumentError drm(one, Beta(); data = d, tree = kn.nwk)
        # A non-constant `sigma` submodel is refused on both routes.
        fs = bf(@formula(y ~ x + phylo(1 | id)), @formula(sigma ~ x), @formula(sd(id, phylogenetic) ~ z0))
        @test_throws ArgumentError drm(fs, Beta(); data = d, tree = kn.nwk)
        # The labelled (correlated) block stays refused, as in drmTMB.
        lab = bf(@formula(y ~ x + relmat(1 + x | p | id)), @formula(sigma ~ 1),
                 @formula(sd(id, relmat, coef(intercept)) ~ z0), @formula(sd(id, relmat, coef(x)) ~ z1))
        @test_throws Exception drm(lab, Beta(); data = merge(_bss_fixture("relmat_q2"), (; p = fill("a", 160))), K = kn.K)
        # Responses must lie strictly inside (0, 1).
        @test_throws ArgumentError drm(fq1, Beta(); data = merge(d, (; y = d.y .+ 1.0)), tree = kn.nwk)
    end
end
