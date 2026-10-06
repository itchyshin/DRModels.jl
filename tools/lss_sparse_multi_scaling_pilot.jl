# tools/lss_sparse_multi_scaling_pilot.jl
# Issue #563, sub-slice S7b.5: p = 10,000 scaling pilot for the sparse
# multi-component LSS route (design note
# docs/src/developer-notes/lss-sparse-multi-component.md §5 oracle 5, §6, §7
# -- "pilot on Totoro first, report wall time and nnz(L)/p before committing
# to the full p=10,000 fit"). D-139: pilot only, not a benchmark.
#
# Fixture: one phylogenetic sd() component (balanced tree, p tips, ONE
# observation-pair per tip) plus one iid sd() component whose G_c = p/10
# groups are COARSER than the tips (10 species per site, contiguous blocks)
# -- deliberately NOT the S7b.1 "sites nested finer within species" shape,
# so this pilot exercises the router's "small" eligibility branch
# (`_lss_multi_sparse_eligible`, `c.G <= 0.1 * phy.G`) at scale, not just the
# strict single-parent nested branch S7b.1/S7b.4 already cover. n = 2p rows
# (two observations per tip).
#
# Usage: julia --project=. tools/lss_sparse_multi_scaling_pilot.jl [p1 p2 ...]
# Default ladder: 1000 2500 5000 10000

using DRModels
using Random
using LinearAlgebra
using SparseArrays

# -----------------------------------------------------------------------
# Fixture builder
# -----------------------------------------------------------------------
# Deterministic replica of `random_balanced_tree`'s own level-by-level
# construction (src/sparse_phy.jl), kept LOCAL so we can also walk the tree
# top-down for an O(p) Brownian-motion leaf simulation -- `sigma_phy_dense`
# (O(p^3), the S7b.1/S7b.4 fixtures' approach) is infeasible at p = 10,000.
function _balanced_edges(p::Int; branch_length::Real = 0.15)
    edges = Tuple{Int,Int,Float64}[]
    current_level = collect(1:p)
    next_id = p + 1
    while length(current_level) > 1
        new_level = Int[]
        i = 1
        while i + 1 <= length(current_level)
            parent = next_id; next_id += 1
            push!(edges, (parent, current_level[i], Float64(branch_length)))
            push!(edges, (parent, current_level[i + 1], Float64(branch_length)))
            push!(new_level, parent)
            i += 2
        end
        if i == length(current_level)
            push!(new_level, current_level[i])
        end
        current_level = new_level
    end
    root_idx = current_level[1]
    return edges, root_idx, next_id - 1
end

# O(p) exact BM draw over the tree (root value 0, each edge adds an
# independent Normal(0, branch_length) increment); root-descending order is
# guaranteed by construction because `parent id > child id` always holds
# (parent ids are assigned by a single monotonically increasing counter,
# level by level, from `_balanced_edges` above), so sorting edges by
# DECREASING parent id is a valid topological (root-to-tip) order.
function _simulate_bm_leaves(rng, p::Int, edges, root_idx::Int, n_total::Int)
    values = zeros(n_total)
    values[root_idx] = 0.0
    for (parent, child, bl) in sort(edges; by = e -> -e[1])
        values[child] = values[parent] + sqrt(bl) * randn(rng)
    end
    return values[1:p]
end

# p tips, one phylo sd() component (`sd(species, phylogenetic) ~ 1`) and one
# iid sd() component (`sd(site) ~ 1`) with G_c = p/10 groups, 10 tips/site
# (contiguous blocks -- deterministic partition, so every observation's site
# membership is a function of its species, satisfying the router's "small"
# branch at exactly the 0.1x boundary). n = 2p rows (2 obs/tip).
function _s7b5_pilot_fixture(p::Int; branch_length = 0.15, seed = 20260902)
    p % 10 == 0 || error("p must be a multiple of 10 for G_iid = p/10 to be exact; got $p")
    rng = MersenneTwister(seed)

    edges, root_idx, n_total = _balanced_edges(p; branch_length = branch_length)
    phy = random_balanced_tree(p; branch_length = branch_length)
    phy.n_leaves == p || error("random_balanced_tree built $(phy.n_leaves) leaves, expected $p")

    h = phylo_tree_height(phy)                 # O(p); unit-tip-variance scale
    u_phy_raw = _simulate_bm_leaves(rng, p, edges, root_idx, n_total)
    u_phy = u_phy_raw ./ sqrt(h)                # standardised to unit tip variance

    G_iid = div(p, 10)
    site_of_species = [div(t - 1, 10) + 1 for t in 1:p]   # 10 species/site, contiguous

    n = 2p
    species_idx = repeat(1:p, inner = 2)
    site_idx = site_of_species[species_idx]

    x = randn(rng, n)

    α_phy0 = -0.5   # sd(species, phylogenetic) ~ 1
    α_iid0 = -0.8   # sd(site) ~ 1
    βμ = [0.4, 0.2]
    βσ0 = -0.6      # sigma ~ 1

    a_phy = exp(α_phy0) .* u_phy
    a_iid = exp(α_iid0) .* randn(rng, G_iid)

    σ_e = exp(βσ0)
    y = βμ[1] .+ βμ[2] .* x .+ a_phy[species_idx] .+ a_iid[site_idx] .+ σ_e .* randn(rng, n)

    sp_names = String.(phy.leaf_names)   # "L1".."Lp", leaf t <-> species t
    dat = (y = y, x = x,
           species = sp_names[species_idx],
           site = ["site$(g)" for g in site_idx])

    return (phy = phy, dat = dat, n = n, p = p, G_iid = G_iid,
            species_idx = species_idx, site_idx = site_idx, x = x)
end

# -----------------------------------------------------------------------
# nnz(L)/dim via the S7b.1 assembler, at theta = 0 (structural fill only --
# the pattern of H does not depend on theta's VALUE at theta=0, exp(0)=1
# scales nothing to a structural zero; same convention as
# test_lss_sparse_multi_public.jl's own oracle-4 fill check).
# -----------------------------------------------------------------------
function _fill_ratio(fx)
    n = fx.n
    Xmu = hcat(ones(n), fx.x)
    Xsigma = ones(n, 1)
    site_gidx, G_iid = DRModels._group_index(fx.dat.site)
    comp_iid = DRModels._sparse_lss_iid_comp(site_gidx, G_iid, ones(G_iid, 1))
    comp_phy = DRModels._sparse_lss_phylo_comp(fx.species_idx, fx.p, ones(fx.p, 1), fx.phy)
    sp_comps = [comp_iid, comp_phy]
    theta = zeros(2 + 1 + 1 + 1)
    asm = DRModels._lss_sparse_multi_assemble(theta, fx.dat.y, Xmu, Xsigma, sp_comps)
    H = asm.H
    dim = size(H, 1)
    ch = cholesky(Symmetric(H); check = false)
    issuccess(ch) || return (dim = dim, nnzL = -1, ratio = NaN)
    nnzL = nnz(sparse(ch.L))
    return (dim = dim, nnzL = nnzL, ratio = nnzL / dim)
end

# -----------------------------------------------------------------------
# One fit + measurement
# -----------------------------------------------------------------------
function _run_one(fx, method::Symbol)
    phy, dat = fx.phy, fx.dat
    f = bf(@formula(y ~ x + (1 | site) + phylo(1 | species)),
           @formula(sigma ~ 1),
           @formula(sd(site) ~ 1),
           @formula(sd(species, phylogenetic) ~ 1))

    local fit
    wall = @elapsed begin
        fit = drm(f, Gaussian(); data = dat, tree = phy, algorithm = :sparse, method = method)
    end
    rss_mb = Sys.maxrss() / 1024^2

    ll = method === :ML ? loglik(fit) : reml_loglik(fit)
    route = DRModels._lss_multi_route(fit)
    fr = _fill_ratio(fx)

    return (p = fx.p, n = fx.n, G_iid = fx.G_iid, method = method, wall = wall,
            converged = fit.converged, iterations = niterations(fit), loglik = ll,
            dim = fr.dim, nnzL = fr.nnzL, ratio = fr.ratio, rss_mb = rss_mb, route = route)
end

function main()
    plist = isempty(ARGS) ? [1000, 2500, 5000, 10000] : parse.(Int, ARGS)

    println("p\tn\tG_iid\tmethod\twall_s\tconverged\titerations\tloglik\tnnzL\tdim\tratio\trss_mb\troute")

    ok = true
    for p in plist
        fx = _s7b5_pilot_fixture(p)
        for method in (:ML, :REML)
            r = _run_one(fx, method)
            println(join((r.p, r.n, r.G_iid, r.method, round(r.wall; digits = 3),
                          r.converged, r.iterations, r.loglik, r.nnzL, r.dim,
                          round(r.ratio; digits = 3), round(r.rss_mb; digits = 1),
                          r.route), "\t"))
            if !(r.converged) || !isfinite(r.loglik)
                ok = false
            end
        end
    end

    if ok
        println("PILOT_OK")
        exit(0)
    else
        println("PILOT_FAILED: non-converged or non-finite result present")
        exit(1)
    end
end

main()
