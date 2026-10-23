# test/test_lss_sparse_gradient_scaling.jl
# DRModels.jl#627: the single-component sparse LSS gradient must stay O(p).
#
# `_fit_phylo_gaussian_lss_sparse` used to accumulate the sd-block quadratic
# term with a GATHER — for each of the G groups, rescan all n observations —
# which is O(G*n).  The objective stayed O(p), so ordinary fitting only looked
# mildly slow, but profile confidence intervals are gradient-bound and became
# impractical at whole-tree scale: 380.0 ms per gradient at G = 16,384 against
# an 8.5 ms objective, growing 4x per doubling of G.  The scatter form now used
# is O(n + G) and, because each group still sums the same observations in the
# same ascending-`i` order, is bit-for-bit identical to the gather.
#
# Two guards below, and they answer different questions:
#   1. the ENDPOINT PINS say the answer did not move;
#   2. the SCALING TRIPWIRE says the cost did not come back.
# A pure speed regression would leave (1) green, and a subtle numerical change
# would leave (2) green, so neither substitutes for the other.

using Test
using DRModels
using StableRNGs
using LinearAlgebra
using Statistics: median

function _grad_scaling_newick(d)
    function node(prefix, depth)
        depth == 0 && return "$(prefix):$(1/d)"
        l = node(prefix * "a", depth - 1); r = node(prefix * "b", depth - 1)
        return depth == d ? "($l,$r);" : "($l,$r):$(1/d)"
    end
    node("t", d)
end

# Brownian-motion draw built by level doubling: O(G), so the fixture itself
# never materialises a dense G x G correlation matrix (which is what made the
# older sparse-engine fixtures unusable above a few thousand tips).
function _grad_scaling_bm(rng, d)
    u = [0.0]
    for lvl in 1:d
        u = repeat(u, inner = 2) .+ randn(rng, 2^lvl) .* sqrt(1 / d)
    end
    return u
end

function _grad_scaling_leaf_names(d)
    names = ["t"]
    for _ in 1:d
        names = reduce(vcat, [[nm * "a", nm * "b"] for nm in names])
    end
    return names
end

function _grad_scaling_fit(depth; seed = 882)
    phy = DRModels.augmented_phy(_grad_scaling_newick(depth))
    G = phy.n_leaves
    rng = StableRNG(seed)
    lr = _grad_scaling_leaf_names(depth)
    uvec = _grad_scaling_bm(rng, depth)
    umap = Dict(lr[i] => uvec[i] for i in eachindex(lr))
    sp_names = String.(phy.leaf_names)
    u_phylo = [umap[s] for s in sp_names]
    n_per_group = 2
    n = G * n_per_group
    gidx = repeat(1:G, inner = n_per_group)
    species = sp_names[gidx]
    x1 = randn(rng, n)
    z1_g = randn(rng, G)
    z1 = z1_g[gidx]
    sda = exp.(-0.4 .+ 0.35 .* z1_g)
    sde = exp.(-1.1 .- 0.3 .* x1)
    y = 1.2 .+ 0.5 .* x1 .+ (sda .* u_phylo)[gidx] .+ sde .* randn(rng, n)
    dat = (y = y, x1 = x1, z1 = z1, species = species)
    f = bf(@formula(y ~ x1 + phylo(1 | species)),
           @formula(sigma ~ x1),
           @formula(sd(species, phylogenetic) ~ z1))
    fit = drm(f, Gaussian(); data = dat, tree = phy,
              algorithm = :sparse, method = :ML, g_tol = 1e-8)
    return phy, fit
end

@testset "#627 sparse LSS profile endpoints are unchanged" begin
    _, fit = _grad_scaling_fit(9)          # G = 512
    @test fit.converged
    @test DRModels._profile_autodiff_mode(fit.nll, fit.nllgrad, fit.theta) === :stored

    prof = profile_result(fit)
    @test prof.autodiff === :stored
    @test prof.attempted == prof.used == 6
    @test prof.failed == 0

    # Pinned against the pre-#627 gather implementation on this exact fixture.
    # These are the values the O(G*n) code produced; the scatter reproduces them
    # in the last bit, so the tolerance is set for build-to-build noise, not to
    # absorb a real change.  A moved endpoint here is a wrong answer, not a
    # tolerance problem — investigate before loosening.
    expected = Dict(
        (:mu, "(Intercept)")       => (1.0328082386307584, 1.2303670034100371),
        (:mu, "x1")                => (0.4745632795511828, 0.5253247826677572),
        (:sigma, "(Intercept)")    => (-1.1679840700742194, -1.0470227959184628),
        (:sigma, "x1")             => (-0.35227375632029173, -0.22388062766295708),
        (:sd_phylo, "(Intercept)") => (-0.5184601600529245, -0.3269246391449383),
        (:sd_phylo, "z1")          => (0.29384911433549216, 0.3895388646822981),
    )
    for row in prof.ci
        lo, hi = expected[(row.param, row.coef)]
        @test isfinite(row.lower) && isfinite(row.upper)
        @test row.lower ≈ lo rtol = 1e-8
        @test row.upper ≈ hi rtol = 1e-8
    end
    for st in prof.stats
        @test st.lower_nuisance_reason === :accepted
        @test st.upper_nuisance_reason === :accepted
        @test !st.lower_endpoint_failed && !st.upper_endpoint_failed
    end

    # Endpoint-search work ceiling.  This is a TRIPWIRE, not a benchmark: the
    # measured cost on this fixture is 14 objective calls and 8 root iterations
    # per coefficient (both arms combined), stable across sizes and seeds.  The
    # ceilings sit far above that so ordinary solver drift cannot trip them,
    # while the runaway this issue is about — a search that re-solves the
    # nuisance problem tens of times more often — still would.  (DRModels.jl#622: a
    # ceiling pinned near the observed value is a flake, not a guard.)
    for st in prof.stats
        @test st.evaluations <= 60
        @test st.root_iterations <= 40
        @test st.bracket_expansions <= 20
    end
end

@testset "#627 sparse LSS gradient cost stays linear in G" begin
    # Quadrupling G must quadruple the gradient cost, not multiply it by ~16.
    # Timing is a blunt instrument, so the separation is deliberately huge: the
    # O(G*n) gather gave ratios near 16 and the O(n+G) scatter gives ratios near
    # 4, and the gate sits at 9 — roughly halfway on a log scale, and above any
    # plausible cache-effect penalty for the honest linear implementation.
    # Prepare BOTH fits before timing. Five single calls measured in separate
    # windows let a short scheduling/GC disturbance affect only one size (CI
    # once reported 0.489 ms versus 5.481 ms for unchanged linear code).
    # Use a fixed set of paired batches, alternating order, and retain the
    # median pair ratio. This measures the same 4x-size scaling and keeps the
    # same <9 boundary; there is no retry conditional on whether it passes.
    function grad_fixture(depth)
        _, fit = _grad_scaling_fit(depth)
        @test fit.converged
        θ = copy(fit.theta)
        g = zeros(length(θ))
        fit.nllgrad(g, θ)                  # compile
        @test all(isfinite, g)
        return (; fit, θ, g)
    end
    fixtures = (grad_fixture(10), grad_fixture(12))  # G = 1,024 and 4,096
    samples = zeros(9, 2)
    function grad_batch(fixture)
        @elapsed for _ in 1:20
            fixture.fit.nllgrad(fixture.g, fixture.θ)
        end
    end
    # Compile the measurement wrapper outside the retained samples too.
    foreach(grad_batch, fixtures)
    for round in axes(samples, 1)
        for size in (isodd(round) ? (1, 2) : (2, 1))
            samples[round, size] = grad_batch(fixtures[size]) / 20
        end
    end
    ratios = samples[:, 2] ./ samples[:, 1]
    ratio = median(ratios)
    @info "#627 gradient cost ratio for 4x G" samples=repr(samples) ratios=repr(ratios) ratio
    @test ratio < 9.0
end
