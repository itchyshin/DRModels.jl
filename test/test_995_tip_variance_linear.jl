# #995 / #996 follow-up: the per-row tip variance C_ii that the Gaussian
# `phylo(1 | g)` quantile residual needs must cost O(p) in the number of tips,
# not O(p²), so a large tree does not stall `residuals(fit; type = :quantile)`.
#
# The first version solved Q_cond x = e_j once per distinct tip: p sparse
# solves of length 2p − 2, each allocating its result, so the allocation grew
# as p². The Takahashi selected inverse returns diag(Q_cond⁻¹) exactly in
# O(nnz(L)). The forcing check below is allocation SCALING (bytes at 4p vs at
# p tips), which is deterministic, rather than wall-clock time: linear work
# gives a ratio near 4, the per-tip solve gives a ratio near 16.
#
# The values are also pinned against the dense root-to-tip covariance on trees
# whose C_ii differ by tip, with rows in a shuffled order and unequal
# replication, so the row -> tip mapping is exercised (the balanced trees in
# test_995_phylo_pit_centre.jl have one C_ii for every tip).
using DRModels
using Test, Random, LinearAlgebra, Statistics

const _tv995 = DRModels._loconly_row_tip_variance

function _tv995_problem(phy, species)
    n = length(species)
    return DRModels.make_loc_problem(phy, zeros(n), ones(n, 1); species = species)
end

# Bytes allocated by one call, measured inside a function after a warm-up call
# so compilation and global-scope boxing are not counted.
function _tv995_bytes(prob)
    _tv995(prob)
    return @allocated _tv995(prob)
end

# A binary tree with p tips and random branch lengths in [0.05, 1.05]:
# root-to-tip distances differ by tip (not ultrametric).
function _tv995_random_tree(p; seed)
    rng = Random.MersenneTwister(seed)
    edges = Tuple{Int,Int,Float64}[]
    level = collect(1:p)
    next_id = p + 1
    while length(level) > 1
        shuffle!(rng, level)
        a, b = pop!(level), pop!(level)
        push!(edges, (next_id, a, 0.05 + rand(rng)))
        push!(edges, (next_id, b, 0.05 + rand(rng)))
        push!(level, next_id)
        next_id += 1
    end
    return DRModels.make_phy(edges, p; root_index = level[1])
end

@testset "#995 tip variance C_ii: exact per row" begin
    rng = Random.MersenneTwister(995)
    trees = (
        caterpillar = random_caterpillar_tree(40; branch_length = 0.3),
        random = _tv995_random_tree(37; seed = 9951),
    )
    for (nm, phy) in pairs(trees)
        p = phy.n_leaves
        # Unequal replication (1 to 4 rows per tip), three tips left without
        # data, rows shuffled.
        present = sort(randperm(rng, p)[1:(p - 3)])
        species = shuffle(rng, reduce(vcat, [fill(k, rand(rng, 1:4)) for k in present]))
        Cii = diag(sigma_phy_dense(phy; σ²_phy = 1.0))
        @testset "$nm tree" begin
            @test maximum(Cii) / minimum(Cii) > 2          # C_ii genuinely varies by tip
            @test _tv995(_tv995_problem(phy, species)) ≈ Cii[species] rtol = 1e-10
        end
    end
end

@testset "#995 tip variance C_ii: linear, not quadratic, in the number of tips" begin
    small = random_balanced_tree(1024; branch_length = 0.1)
    large = random_balanced_tree(4096; branch_length = 0.1)
    b_small = _tv995_bytes(_tv995_problem(small, 1:1024))
    b_large = _tv995_bytes(_tv995_problem(large, 1:4096))
    # 4x the tips: about 4x the bytes when linear, about 16x for one solve per tip
    # (measured on Julia 1.12: 4.0x vs 15.9x).
    @test b_large / b_small < 8
    # The same 20 kB-per-tip ceiling as test_sparse_precision_storage.jl. The
    # per-tip solve allocated about 460 kB per tip at 4096 tips (1.9 GB in all);
    # the selected inverse about 1 kB.
    @test b_large < 20_000 * 4096
end

# The PIT on a caterpillar tree, where C_ii runs from 0.15 to 7.05, so
# σ̂_b √C_ii / σ̂_e reaches about 6. The Gaussian marginal is closed form; the
# 32-node Gauss–Hermite mixture the residual used before misses it by up to
# 0.031 here (its error grows with that ratio: 6e-3 at 4, 0.12 at 10), so the
# tolerance is tight.
@testset "#995 phylo PIT on a tree with tip-varying C_ii" begin
    phy = random_caterpillar_tree(48; branch_length = 0.15)
    p = phy.n_leaves
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    Random.seed!(20261016)
    u = 0.8 .* (cholesky(Symmetric(C)).L * randn(p))
    species = shuffle(repeat(1:p, inner = 3))
    n = length(species)
    x = randn(n)
    y = 0.4 .+ 0.7 .* x .+ u[species] .+ 0.4 .* randn(n)
    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)), Gaussian();
              data = (; y, x, species), tree = phy)
    @test fit.ranef isa DRModels.LocOnlyRanef
    bl = Dict(fit.blocks)
    β = fit.theta[bl[:mu]]
    σe = exp(fit.theta[first(bl[:sigma])])
    σb = exp(fit.theta[first(bl[:resd])])
    ref = (y .- β[1] .- β[2] .* x) ./ sqrt.(σe^2 .+ σb^2 .* diag(C)[species])
    @test residuals(fit; type = :quantile) ≈ ref atol = 1e-10
end
