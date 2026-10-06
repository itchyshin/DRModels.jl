# head_to_head_q4_scaling.jl — #376 Julia arm + fixture export
#
# Paired wall-clock vs drmTMB on the q=4 PLSM biological per-dimension-variance
# model (nrep=4). Reuses the same Λ / β / sampler contract as bench/run_scaling.jl.
# Does NOT touch src/. Exports CSV + Newick for the R public-API arm.
#
# Run from repo root:
#   julia --project=. bench/head_to_head_q4_scaling.jl
#
# Env:
#   DRM_376_PS     comma-separated tip counts (default: 100,1000,5000,10000)
#   DRM_376_NREP   observations per species (default: 4)
#   DRM_376_REPS   timed reps after one warmup (default: 3)
#   DRM_376_SEED   base RNG seed (default: 37600)
#   DRM_376_EXPORT_ONLY  if "1", write fixtures and skip Julia timing

import Pkg
Pkg.activate(dirname(@__DIR__))

using Dates
using LinearAlgebra
using Printf
using Random
using SparseArrays
using Statistics
using TOML

BLAS.set_num_threads(1)

using DRModels

const OUT_DIR = joinpath(@__DIR__, "results", "q4_scaling_h2h_376")
const FIX_DIR = joinpath(OUT_DIR, "fixtures")
const JULIA_TOML = joinpath(OUT_DIR, "julia_q4_scaling.toml")

const βT = (
    mu1 = [1.0, 0.5],
    mu2 = [-0.3, 0.4],
    s1 = [-0.4],
    s2 = [-0.5],
    rho = [0.3],
)

const ΛT = Matrix(Symmetric([
    0.25 0.10 0.05 0.00
    0.10 0.25 0.00 0.04
    0.05 0.00 0.09 0.02
    0.00 0.04 0.02 0.09
]))

const Λ0 = Matrix(Symmetric([
    0.30 0.02 0.01 0.010
    0.02 0.30 0.01 0.010
    0.01 0.01 0.08 0.005
    0.01 0.01 0.005 0.080
]))

function _parse_int_list(s)
    vals = Int[]
    for token in split(s, ",")
        stripped = strip(token)
        isempty(stripped) && continue
        push!(vals, parse(Int, stripped))
    end
    isempty(vals) && error("empty integer list: $s")
    return vals
end

function _balanced_edges(p::Integer; branch_length::Real = 0.2)
    edges = Tuple{Int,Int,Float64}[]
    current_level = collect(1:p)
    next_id = p + 1
    while length(current_level) > 1
        next_level = Int[]
        i = 1
        while i <= length(current_level)
            if i == length(current_level)
                push!(next_level, current_level[i])
                break
            end
            parent = next_id
            next_id += 1
            push!(edges, (parent, current_level[i], Float64(branch_length)))
            push!(edges, (parent, current_level[i + 1], Float64(branch_length)))
            push!(next_level, parent)
            i += 2
        end
        current_level = next_level
    end
    root = only(current_level)
    return _ultrametricize(edges, p, root), root
end

"""Extend terminal branches so every leaf has the same root-to-tip height.

drmTMB's `phylo()` requires ultrametric trees. The near-balanced constructor
with equal branch lengths is ultrametric only when `p` is a power of 2."""
function _ultrametricize(edges::Vector{Tuple{Int,Int,Float64}}, n_leaves::Integer, root::Integer)
    parent_of = Dict{Int,Tuple{Int,Float64}}()
    for (parent, child, blen) in edges
        parent_of[child] = (parent, blen)
    end
    function depth(node::Int)
        node == root && return 0.0
        par, blen = parent_of[node]
        return depth(par) + blen
    end
    depths = [depth(t) for t in 1:n_leaves]
    target = maximum(depths)
    # Rebuild edges with extended terminal branches.
    out = Tuple{Int,Int,Float64}[]
    for (parent, child, blen) in edges
        if child <= n_leaves
            push!(out, (parent, child, blen + (target - depths[child])))
        else
            push!(out, (parent, child, blen))
        end
    end
    return out
end

function _edges_to_newick(edges, n_leaves::Integer, root_index::Integer, leaf_names)
    children = Dict{Int,Vector{Tuple{Int,Float64}}}()
    for (parent, child, blen) in edges
        push!(get!(() -> Tuple{Int,Float64}[], children, parent), (child, blen))
    end
    function rec(node::Int)
        if node <= n_leaves
            return String(leaf_names[node])
        end
        kids = get(children, node, Tuple{Int,Float64}[])
        isempty(kids) && error("internal node $node has no children")
        parts = String[]
        for (child, blen) in kids
            push!(parts, string(rec(child), ":", blen))
        end
        return "(" * join(parts, ",") * ")"
    end
    return rec(root_index) * ";"
end

function _sample_augmented_state(rng::AbstractRNG, phy, Q_cond)
    P = prior_precision(Q_cond, inv(ΛT))
    F = cholesky(Symmetric(P))
    return F.UP \ randn(rng, size(P, 1))
end

"""Build one synthetic case; return fit inputs + export payloads."""
function _make_case(p::Integer; seed::Integer, nrep::Integer)
    rng = MersenneTwister(seed)
    edges, root = _balanced_edges(p; branch_length = 0.2)
    leaf_names = ["L$t" for t in 1:p]
    phy = DRModels.make_phy(edges, p; root_index = root, leaf_names = leaf_names)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    Q_cond = phy.Q_topology[keep, keep]
    u_aug = _sample_augmented_state(rng, phy, Q_cond)

    pos = Dict(node => i for (i, node) in enumerate(keep))
    leaf_pos = [pos[phy.leaf_indices[t]] for t in 1:p]
    U = Matrix{Float64}(undef, 4, p)
    @inbounds for k in 1:p, a in 1:4
        U[a, k] = u_aug[4 * (leaf_pos[k] - 1) + a]
    end

    species = repeat(1:p, inner = nrep)
    n = length(species)
    x1 = randn(rng, n)
    X1 = hcat(ones(n), x1)
    X2 = hcat(ones(n), x1)
    Xs1 = reshape(ones(n), n, 1)
    Xs2 = reshape(ones(n), n, 1)
    Xr = reshape(ones(n), n, 1)
    y1 = Vector{Float64}(undef, n)
    y2 = Vector{Float64}(undef, n)

    @inbounds for i in 1:n
        k = species[i]
        m1 = dot(@view(X1[i, :]), βT.mu1) + U[1, k]
        m2 = dot(@view(X2[i, :]), βT.mu2) + U[2, k]
        s1 = exp(dot(@view(Xs1[i, :]), βT.s1) + U[3, k])
        s2 = exp(dot(@view(Xs2[i, :]), βT.s2) + U[4, k])
        ρ = DRModels.RHO_GUARD * tanh(dot(@view(Xr[i, :]), βT.rho))
        e = cholesky(Symmetric([s1^2 ρ * s1 * s2; ρ * s1 * s2 s2^2])).L * randn(rng, 2)
        y1[i] = m1 + e[1]
        y2[i] = m2 + e[2]
    end

    prob, Q = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr; species = species)
    β0 = (
        mu1 = X1 \ y1,
        mu2 = X2 \ y2,
        s1 = [log(std(y1 .- X1 * (X1 \ y1)))],
        s2 = [log(std(y2 .- X2 * (X2 \ y2)))],
        rho = [0.0],
    )
    newick = _edges_to_newick(edges, p, root, leaf_names)
    sp_labels = [leaf_names[k] for k in species]
    return (; prob, Q, β0, phy, y1, y2, x1, sp_labels, newick, nrep, p, n)
end

function _export_fixture(case, dest_dir::AbstractString)
    mkpath(dest_dir)
    csv_path = joinpath(dest_dir, "data.csv")
    nwk_path = joinpath(dest_dir, "tree.nwk")
    open(csv_path, "w") do io
        println(io, "y1,y2,x1,species")
        for i in 1:case.n
            @printf(io, "%.16g,%.16g,%.16g,%s\n",
                case.y1[i], case.y2[i], case.x1[i], case.sp_labels[i])
        end
    end
    write(nwk_path, case.newick * "\n")
    meta = Dict(
        "p" => case.p,
        "nrep" => case.nrep,
        "n" => case.n,
        "shape" => "balanced",
        "model" => "q4_plsm_per_dim_variance",
        "csv" => "data.csv",
        "tree" => "tree.nwk",
    )
    open(joinpath(dest_dir, "meta.toml"), "w") do io
        TOML.print(io, meta)
    end
    return csv_path, nwk_path
end

function _fit_case(prob, Q, β0)
    return fit_q4_sparse_tmb(
        prob, Q;
        β0 = β0,
        Λ0 = Λ0,
        g_tol = 1e-3,
        iterations = 400,
        n_newton = 40,
    )
end

function time_case(p::Integer; seed::Integer, nrep::Integer, reps::Integer, export_only::Bool)
    case = _make_case(p; seed = seed, nrep = nrep)
    dest = joinpath(FIX_DIR, "p$(p)_nrep$(nrep)")
    _export_fixture(case, dest)

    if export_only
        return Dict{String,Any}(
            "p" => p,
            "nrep" => nrep,
            "n" => case.n,
            "fixture_dir" => dest,
            "ok" => true,
            "export_only" => true,
            "note" => "fixtures written; Julia timing skipped",
        )
    end

    warm = _fit_case(case.prob, case.Q, case.β0)
    times = Float64[]
    last = warm
    for _ in 1:reps
        t = @elapsed last = _fit_case(case.prob, case.Q, case.β0)
        push!(times, t)
    end
    return Dict{String,Any}(
        "p" => p,
        "nrep" => nrep,
        "n" => case.n,
        "fixture_dir" => dest,
        "reps" => reps,
        "warmup_discarded" => true,
        "times_s" => times,
        "median_s" => median(times),
        "min_s" => minimum(times),
        "max_s" => maximum(times),
        "logLik" => Float64(last.loglik),
        "converged" => Bool(last.converged),
        "iterations" => Int(last.iterations),
        "g_residual" => Float64(last.g_residual),
        "ok" => true,
        "note" => "",
    )
end

function main()
    ps = _parse_int_list(get(ENV, "DRM_376_PS", "100,1000,5000,10000"))
    nrep = parse(Int, get(ENV, "DRM_376_NREP", "4"))
    reps = max(1, parse(Int, get(ENV, "DRM_376_REPS", "3")))
    seed0 = parse(Int, get(ENV, "DRM_376_SEED", "37600"))
    export_only = get(ENV, "DRM_376_EXPORT_ONLY", "0") == "1"
    nrep >= 2 || error("nrep must be at least 2 for scale RE identifiability")

    mkpath(OUT_DIR)
    rows = Dict{String,Any}[]

    # Tiny compile warmup (not reported).
    if !export_only
        c0 = _make_case(16; seed = seed0 - 1, nrep = 2)
        _fit_case(c0.prob, c0.Q, c0.β0)
    end

    println("=== #376 Julia q4 scaling head-to-head ===")
    @printf "ps=%s nrep=%d reps=%d export_only=%s\n" join(ps, ",") nrep reps export_only
    for p in ps
        seed = seed0 + p
        @info "Julia cell" p nrep reps seed
        try
            push!(rows, time_case(p; seed = seed, nrep = nrep, reps = reps, export_only = export_only))
        catch err
            push!(rows, Dict{String,Any}(
                "p" => p,
                "nrep" => nrep,
                "ok" => false,
                "note" => sprint(showerror, err),
            ))
            @warn "Julia cell failed" p err
        end
    end

    payload = Dict{String,Any}(
        "issue" => 376,
        "arm" => "julia",
        "model" => "q4_plsm_per_dim_variance_nrep4",
        "shape" => "balanced",
        "timestamp_utc" => string(Dates.now(Dates.UTC)),
        "hostname" => gethostname(),
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "nrep" => nrep,
        "reps" => reps,
        "seed_base" => seed0,
        "rows" => rows,
    )
    open(JULIA_TOML, "w") do io
        TOML.print(io, payload)
    end
    println("wrote ", JULIA_TOML)
    return payload
end

main()
