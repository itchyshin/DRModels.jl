# run_scaling.jl -- Workflow Q multi-shape scaling gate for the q=4 PLSM engine.
#
# Default gate:
#   julia --project=bench bench/run_scaling.jl
#
# The default runs balanced + caterpillar trees at p in {100, 1000, 10000}
# with nrep = 4 observations per species. For quick smoke checks:
#   DRM_QGATE_PS=20,40 DRM_QGATE_NREP=2 julia --project=bench bench/run_scaling.jl

import Pkg
Pkg.activate(dirname(@__DIR__))

using DRModels
using LinearAlgebra, Printf, Random, SparseArrays, Statistics

BLAS.set_num_threads(1)

const OUT = joinpath(@__DIR__, "..", "report", "qgate-multishape-scaling.md")

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

# Tight scale-axis starts avoid the known one-observation scale-field collapse.
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

function _parse_symbol_list(s)
    vals = Symbol[]
    for token in split(s, ",")
        stripped = strip(token)
        isempty(stripped) && continue
        push!(vals, Symbol(stripped))
    end
    isempty(vals) && error("empty shape list: $s")
    return vals
end

function _caterpillar_tree(p::Integer; branch_length::Real)
    p >= 2 || error("caterpillar tree needs p >= 2; got $p")
    edges = Tuple{Int,Int,Float64}[]
    next_internal = p + 1
    current = next_internal
    next_internal += 1
    push!(edges, (current, 1, Float64(branch_length)))
    push!(edges, (current, 2, Float64(branch_length)))
    for leaf in 3:p
        parent = next_internal
        next_internal += 1
        push!(edges, (parent, current, Float64(branch_length)))
        push!(edges, (parent, leaf, Float64(branch_length)))
        current = parent
    end
    return DRModels.make_phy(edges, p; root_index = current)
end

function _phy(shape::Symbol, p::Integer)
    balanced_branch = 0.2
    if shape == :balanced
        return random_balanced_tree(p; branch_length = balanced_branch)
    elseif shape == :caterpillar
        # Match the maximum root-to-tip height of the balanced gate at this p.
        # A fixed caterpillar branch length would make the oldest tips' Brownian
        # variance grow as O(p), turning this into a data-scale stress test
        # rather than a sparse-topology scaling gate.
        max_balanced_edges = ceil(Int, log2(p))
        target_height = balanced_branch * max_balanced_edges
        return _caterpillar_tree(p; branch_length = target_height / (p - 1))
    else
        error("unknown shape $shape; expected :balanced or :caterpillar")
    end
end

function _sample_augmented_state(rng::AbstractRNG, phy, Q_cond)
    P = prior_precision(Q_cond, inv(ΛT))
    F = cholesky(Symmetric(P))
    return F.UP \ randn(rng, size(P, 1))
end

function _make_case(shape::Symbol, p::Integer; seed::Integer, nrep::Integer)
    rng = MersenneTwister(seed)
    phy = _phy(shape, p)
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
        e = cholesky(Symmetric([s1^2 ρ*s1*s2; ρ*s1*s2 s2^2])).L * randn(rng, 2)
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
    return prob, Q, β0
end

function _fit_case(prob, Q, β0)
    fit_q4_sparse_tmb(
        prob, Q;
        β0 = β0,
        Λ0 = Λ0,
        g_tol = 1e-3,
        iterations = 400,
        n_newton = 40,
    )
end

function _scaling_exponent(rows)
    ps = Float64[row.p for row in rows]
    ts = Float64[row.wall_s for row in rows]
    length(ps) >= 2 || return NaN
    x = log.(ps)
    y = log.(ts)
    xbar = mean(x)
    ybar = mean(y)
    return sum((x .- xbar) .* (y .- ybar)) / sum(abs2, x .- xbar)
end

function _run_one(shape::Symbol, p::Integer; seed::Integer, nrep::Integer)
    prob, Q, β0 = _make_case(shape, p; seed = seed, nrep = nrep)
    t = @elapsed r = _fit_case(prob, Q, β0)
    nobs = length(prob.y1)
    return (
        shape = shape,
        p = p,
        nobs = nobs,
        wall_s = t,
        iterations = r.iterations,
        loglik = r.loglik,
        per_obs = r.loglik / nobs,
        ms_per_node = 1000 * t / (2p - 1),
        converged = r.converged,
        g_residual = r.g_residual,
        f_calls = r.f_calls,
        g_calls = r.g_calls,
    )
end

function _write_report(rows, ps, shapes, nrep, exponents, passed)
    mkpath(dirname(OUT))
    open(OUT, "w") do io
        println(io, "# Q-gate multi-shape q4 scaling")
        println(io)
        println(io, "Command: `julia --project=bench bench/run_scaling.jl`")
        println(io, "Julia threads: $(Threads.nthreads()); BLAS threads: $(BLAS.get_num_threads()).")
        println(io, "Shapes: $(join(string.(shapes), ", ")); p grid: $(join(ps, ", ")); nrep: $nrep.")
        println(io)
        println(io, "The sampler draws the full augmented state from `P^{-1}` with sparse CHOLMOD (`P = kron(Q_cond, inv(Λ))`), then attaches `nrep` observations per species. Caterpillar branch lengths are scaled so their maximum root-to-tip height matches the balanced tree at the same `p`; this keeps the gate focused on sparse-topology scaling rather than changing the Brownian data scale.")
        println(io)
        println(io, "| shape | p | nobs | wall/s | iterations | logLik | logLik/nobs | ms/node | converged | g_resid |")
        println(io, "|:------|--:|-----:|-------:|-----------:|-------:|------------:|--------:|:----------|--------:|")
        for row in rows
            @printf(io, "| %s | %d | %d | %.3f | %d | %.2f | %.3f | %.3f | %s | %.2e |\n",
                row.shape, row.p, row.nobs, row.wall_s, row.iterations, row.loglik,
                row.per_obs, row.ms_per_node, row.converged, row.g_residual)
        end
        println(io)
        println(io, "| shape | empirical k in wall ~ p^k |")
        println(io, "|:------|--------------------------:|")
        for shape in shapes
            @printf(io, "| %s | %.2f |\n", shape, exponents[shape])
        end
        println(io)
        println(io, "Gate verdict: **$(passed ? "PASS" : "FAIL")**.")
        println(io)
        println(io, "Gate criteria:")
        println(io, "- every row has a finite wall time and finite log-likelihood;")
        println(io, "- every fit reports `converged = true`;")
        println(io, "- each shape's empirical exponent is at most 1.6 when at least three `p` values are run.")
    end
end

function main()
    ps = _parse_int_list(get(ENV, "DRM_QGATE_PS", "100,1000,10000"))
    shapes = _parse_symbol_list(get(ENV, "DRM_QGATE_SHAPES", "balanced,caterpillar"))
    nrep = parse(Int, get(ENV, "DRM_QGATE_NREP", "4"))
    nrep >= 2 || error("nrep must be at least 2 for the q4 scale random effects")

    # Compile warmup outside the reported grid.
    prob0, Q0, β00 = _make_case(:balanced, 20; seed = 20260605, nrep = 2)
    _fit_case(prob0, Q0, β00)

    rows = NamedTuple[]
    println("=== Workflow Q #16: multi-shape q4 scaling gate ===")
    @printf "%12s %8s %8s %10s %8s %12s %12s %10s\n" "shape" "p" "nobs" "wall(s)" "iters" "logLik" "per-obs" "ms/node"
    for shape in shapes
        for p in ps
            row = _run_one(shape, p; seed = 71000 + p + 1000 * findfirst(==(shape), shapes), nrep = nrep)
            push!(rows, row)
            @printf "%12s %8d %8d %10.3f %8d %12.2f %12.3f %10.3f\n" row.shape row.p row.nobs row.wall_s row.iterations row.loglik row.per_obs row.ms_per_node
        end
    end

    exponents = Dict{Symbol,Float64}()
    passed = true
    for shape in shapes
        srows = [row for row in rows if row.shape == shape]
        exponents[shape] = _scaling_exponent(srows)
        passed &= all(row -> isfinite(row.wall_s) && isfinite(row.loglik) && row.converged, srows)
        if length(srows) >= 3
            passed &= exponents[shape] <= 1.6
        end
    end

    println()
    for shape in shapes
        @printf "%s empirical k = %.2f\n" shape exponents[shape]
    end
    println("gate verdict: ", passed ? "PASS" : "FAIL")
    _write_report(rows, ps, shapes, nrep, exponents, passed)
    println("wrote ", OUT)
    passed || error("multi-shape scaling gate failed")
    return rows
end

main()
