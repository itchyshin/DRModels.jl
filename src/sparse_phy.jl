# Augmented-state sparse phylogenetic precision.
#
# Standard phylogenetic comparative methods supply a dense (p × p) Brownian-
# motion covariance Σ_phy over species at the tips of a tree. For p = 10_000
# the dense Cholesky is already 16+ seconds per evaluation. The Felsenstein
# (1981) / Hadfield (2010) / Bates (2015) workaround: augment the state with
# internal ancestral nodes, then represent the tree by a SPARSE precision
# matrix Q over all leaves and internal nodes. Internal nodes get marginalised inside the
# sparse linear solves.
#
# Each tree edge (parent → child) with branch length b contributes
#     Q[parent, parent] += 1 / b
#     Q[child,  child ] += 1 / b
#     Q[parent, child ] -= 1 / b
#     Q[child,  parent] -= 1 / b
# i.e. a 2 × 2 block (1 / b) · [[1, -1], [-1, 1]] on rows/cols (parent, child).
# A rooted tree with E edges has 4E input triplets before duplicate diagonal
# accumulation. Binary trees have E = 2p − 2, hence about 8p input triplets
# but about 6p stored nonzeros; multifurcations use fewer internal nodes and
# edges. Q is symmetric and rank-deficient by
# one: the constant-shift direction `z ≡ 1` lies in its null space (Brownian
# motion is identified only up to a common offset = root value).
#
# This file provides:
#   * `AugmentedPhy`   — container for the sparse topology precision plus
#                        identifying which augmented rows are leaves.
#   * `augmented_phy`  — Newick string parser → `AugmentedPhy`.
#   * `make_phy`       — same, but from a (parent, child, length) triple list.
#
# We deliberately do NOT depend on Phylo.jl or any other ecology package —
# the parser is ~80 lines of recursive descent and matches the minimal
# Newick grammar at
#     https://evolution.genetics.washington.edu/phylip/newicktree.html.
# Leaves use either old-compatible unquoted names or lossless single-quoted
# Newick labels. Internal labels are parsed but discarded. The delimiter is
# `;`; whitespace is ignored only between grammar tokens.

using SparseArrays
using LinearAlgebra

"""
    AugmentedPhy{T}

Augmented-state sparse phylogenetic precision for a rooted tree, including
multifurcations.

Fields
------
* `n_leaves::Int`               – number of tip species (p).
* `n_total::Int`                – leaves + internal ancestor nodes.
* `Q_topology::SparseMatrixCSC` – (n_total × n_total) topology contribution
  to the sparse precision. The actual phylogenetic precision is
  `Q_topology / σ²_phy`. A positive-length tree stores `3n_total - 2`
  nonzeros.
* `leaf_indices::Vector{Int}`   – maps a leaf k ∈ 1:p to its row/col in the
  augmented state. Ordering matches the order leaves were encountered in
  the Newick string (left-to-right).
* `leaf_names::Vector{String}`  – species names parsed from the Newick.
* `branch_lengths::Vector{T}`   – the `n_total - 1` branch lengths in the order
  the parser walked the tree.
* `root_index::Int`             – which augmented row is the root.

`Q_topology` is positive **semi**-definite (rank `n_total - 1`). The all-ones
vector is its sole zero eigenvector — fixing the root removes the
degeneracy. The sparse log-likelihood path adds a positive contribution
to the leaf diagonals (proportional to `λ_phy² / d_total`) which renders
the active solve matrix positive definite without any explicit ridge.
"""
struct AugmentedPhy{T}
    n_leaves::Int
    n_total::Int
    Q_topology::SparseMatrixCSC{T,Int}
    leaf_indices::Vector{Int}
    leaf_names::Vector{String}
    branch_lengths::Vector{T}
    root_index::Int
end

# ---------------------------------------------------------------------------
# Newick parsing
# ---------------------------------------------------------------------------
# Grammar (subset of the Felsenstein Newick standard):
#
#     tree    := node ";"
#     node    := leaf | internal
#     leaf    := name [":" length]
#     internal:= "(" node ("," node)* ")" [name] [":" length]
#     label   := unquoted | "'" quoted "'"
#     quoted  := any character, with "''" representing one apostrophe
#     length  := [0-9]+ ( "." [0-9]+ )? ( [eE] [+-]? [0-9]+ )?
#
# Whitespace outside labels is ignored. It remains literal inside a quoted
# label. Length defaults to 0.0 if omitted.

mutable struct _NewickCursor
    s::String
    i::Int
end

@inline _peek(c::_NewickCursor) = c.i > lastindex(c.s) ? '\0' : c.s[c.i]
@inline function _advance(c::_NewickCursor)
    ch = _peek(c)
    c.i = nextind(c.s, c.i)
    return ch
end

@inline _newick_whitespace(ch::Char) = isspace(ch)

function _skip_newick_whitespace!(c::_NewickCursor)
    while _newick_whitespace(_peek(c))
        _advance(c)
    end
    return nothing
end

function _parse_number!(c::_NewickCursor)
    j = c.i
    while j <= lastindex(c.s)
        ch = c.s[j]
        if ch in '0':'9' || ch == '.' || ch == 'e' || ch == 'E' || ch == '+' || ch == '-'
            j = nextind(c.s, j)
        else
            break
        end
    end
    j == c.i && error("expected number at position $(c.i)")
    val = parse(Float64, c.s[c.i:prevind(c.s, j)])
    c.i = j
    return val
end

"""Parse one Newick label, retaining quoted Unicode/control characters exactly."""
function _parse_label!(c::_NewickCursor)
    ch = _peek(c)
    if ch == '\0' || ch == ',' || ch == ')' || ch == ':' || ch == ';' || ch == '('
        throw(ArgumentError("expected Newick label at position $(c.i)"))
    end
    if ch == '\''
        _advance(c)
        label = IOBuffer()
        while true
            ch = _peek(c)
            ch == '\0' && throw(ArgumentError(
                "unterminated quoted label beginning before position $(c.i)"))
            if ch == '\''
                _advance(c)
                if _peek(c) == '\''
                    write(label, '\'')
                    _advance(c)
                else
                    return String(take!(label))
                end
            else
                write(label, ch)
                _advance(c)
            end
        end
    end

    first = c.i
    while true
        ch = _peek(c)
        if ch == '\0' || ch == ',' || ch == ')' || ch == ':' || ch == ';' || ch == '('
            break
        elseif _newick_whitespace(ch)
            # Whitespace separates grammar tokens, so `A : 1` retains the
            # old accepted spelling. It cannot split an unquoted label such
            # as `two words`.
            label = c.s[first:prevind(c.s, c.i)]
            _skip_newick_whitespace!(c)
            next = _peek(c)
            if next == ':' || next == ',' || next == ')' || next == ';' || next == '\0'
                return label
            end
            throw(ArgumentError(
                "unquoted label cannot contain whitespace at position $(c.i); quote the label"))
        elseif ch == '\''
            throw(ArgumentError(
                "single quote in an unquoted label at position $(c.i); quote the whole label"))
        elseif ch == '[' || ch == ']'
            throw(ArgumentError(
                "Newick comments are unsupported at position $(c.i); quote literal brackets in a label"))
        end
        _advance(c)
    end
    first == c.i && throw(ArgumentError("expected Newick label at position $(c.i)"))
    return c.s[first:prevind(c.s, c.i)]
end

# Internal builder: walks the Newick string and writes nodes + edges into
# the supplied vectors. Returns the index of the node it just consumed.
function _parse_node!(c::_NewickCursor,
                      node_parent::Vector{Int},
                      node_is_leaf::Vector{Bool},
                      node_name::Vector{String},
                      node_length::Vector{Float64},
                      leaf_indices::Vector{Int},
                      leaf_names::Vector{String})
    _skip_newick_whitespace!(c)
    children_local = Int[]
    if _peek(c) == '('
        _advance(c)                       # consume "("
        _skip_newick_whitespace!(c)
        # parse comma-separated children
        push!(children_local, _parse_node!(c, node_parent, node_is_leaf,
                                           node_name, node_length,
                                           leaf_indices, leaf_names))
        _skip_newick_whitespace!(c)
        while _peek(c) == ','
            _advance(c)
            push!(children_local, _parse_node!(c, node_parent, node_is_leaf,
                                               node_name, node_length,
                                               leaf_indices, leaf_names))
            _skip_newick_whitespace!(c)
        end
        _peek(c) == ')' ||
            throw(ArgumentError("expected delimiter ',' or ')' at position $(c.i) in Newick string"))
        _advance(c)
        _skip_newick_whitespace!(c)
        # internal node label (optional, discarded — minimal grammar)
        name = ""
        if _peek(c) != ':' && _peek(c) != ',' && _peek(c) != ')' && _peek(c) != ';'
            name = _parse_label!(c)
        end
        _skip_newick_whitespace!(c)
        # branch length (to PARENT, optional)
        blen = 0.0
        if _peek(c) == ':'
            _advance(c)
            _skip_newick_whitespace!(c)
            blen = _parse_number!(c)
        end
        _skip_newick_whitespace!(c)
        # allocate this internal node and patch children's parents
        push!(node_parent, 0)             # parent set by caller
        push!(node_is_leaf, false)
        push!(node_name, name)
        push!(node_length, blen)
        my_idx = length(node_parent)
        for c_idx in children_local
            node_parent[c_idx] = my_idx
        end
        return my_idx
    else
        # leaf
        name = _parse_label!(c)
        _skip_newick_whitespace!(c)
        blen = 0.0
        if _peek(c) == ':'
            _advance(c)
            _skip_newick_whitespace!(c)
            blen = _parse_number!(c)
        end
        _skip_newick_whitespace!(c)
        push!(node_parent, 0)             # parent set by caller
        push!(node_is_leaf, true)
        push!(node_name, name)
        push!(node_length, blen)
        my_idx = length(node_parent)
        push!(leaf_indices, my_idx)
        push!(leaf_names, name)
        return my_idx
    end
end

function _phy_branch_length(value, label::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "branch length for $label must be a real number"))
    length = Float64(value)
    isfinite(length) && length > 0 || throw(ArgumentError(
        "branch length for $label must be finite and > 0 (got $value)"))
    isfinite(inv(length)) || throw(ArgumentError(
        "branch length for $label is too small for a finite sparse precision"))
    return length
end

function _phy_validate_leaf_names(names::AbstractVector{<:AbstractString}, p::Int)
    length(names) == p || throw(ArgumentError(
        "leaf_names must contain exactly $p names"))
    result = String.(names)
    all(!isempty, result) || throw(ArgumentError("leaf names must be nonempty"))
    length(unique(result)) == p || throw(ArgumentError("leaf names must be unique"))
    return result
end

"""Validate a rooted acyclic tree and return `(root, children)` without Q assembly."""
function _phy_validate_topology(parent::Vector{Int}, is_leaf::Vector{Bool};
                                root_index::Union{Nothing,Int} = nothing)
    n = length(parent)
    n > 0 && length(is_leaf) == n || throw(ArgumentError(
        "phylogenetic topology must contain aligned nonempty nodes"))
    children = [Int[] for _ in 1:n]
    for child in 1:n
        ancestor = parent[child]
        ancestor == 0 && continue
        1 <= ancestor <= n || throw(ArgumentError(
            "phylogenetic parent id $ancestor is outside 1:$n"))
        ancestor != child || throw(ArgumentError("phylogenetic edges cannot be self edges"))
        push!(children[ancestor], child)
    end
    roots = findall(==(0), parent)
    length(roots) == 1 || throw(ArgumentError(
        "phylogenetic topology must have exactly one root (found $(length(roots)))"))
    root = only(roots)
    root_index === nothing || root_index == root || throw(ArgumentError(
        "root_index $root_index is not the unique topology root $root"))
    for node in 1:n
        if is_leaf[node]
            isempty(children[node]) || throw(ArgumentError(
                "leaf node $node cannot be a parent"))
        elseif length(children[node]) < 2
            throw(ArgumentError(
                "internal node $node must have at least two children; unary nodes are currently unsupported"))
        end
    end
    seen = falses(n)
    queue = [root]
    cursor = 1
    while cursor <= length(queue)
        node = queue[cursor]
        cursor += 1
        seen[node] && throw(ArgumentError("phylogenetic topology contains a cycle"))
        seen[node] = true
        append!(queue, children[node])
    end
    all(seen) || throw(ArgumentError(
        "phylogenetic topology must be connected and acyclic from its root"))
    return root, children
end

function _phy_topology_precision(edges::AbstractVector{<:Tuple}, n_total::Int)
    I = Int[]; J = Int[]; V = Float64[]
    branch_lengths = Float64[]
    for (parent, child, length) in edges
        b = _phy_branch_length(length, "edge $parent -> $child")
        inv_b = inv(b)
        push!(branch_lengths, b)
        push!(I, parent); push!(J, parent); push!(V, inv_b)
        push!(I, child);  push!(J, child);  push!(V, inv_b)
        push!(I, parent); push!(J, child);  push!(V, -inv_b)
        push!(I, child);  push!(J, parent); push!(V, -inv_b)
    end
    Q = sparse(I, J, V, n_total, n_total)
    all(isfinite, nonzeros(Q)) || throw(ArgumentError(
        "branch precisions overflow the assembled sparse topology diagonal"))
    return Q, branch_lengths
end

"""
    augmented_phy(newick::AbstractString) :: AugmentedPhy{Float64}

Parse a minimal Newick string and return the augmented-state sparse
precision representation.

Restrictions
------------
* Rooted multifurcating trees are admitted: every internal node must have at
  least two children. Unary nodes are currently unsupported.
* Leaf names may be old-compatible unquoted labels or lossless single-quoted
  labels. Doubled apostrophes inside a quoted label represent one apostrophe.
  Internal labels are tolerated but discarded.
* Literal NUL bytes are invalid Newick input and are rejected before parsing.
* Non-root branch lengths must be finite, > 0, and have finite reciprocal.
* The root has no parent branch; the optional root length in
  `(…):0.0;` is read but does not enter Q.

Example
-------
```julia
phy = augmented_phy("((A:0.1,B:0.2):0.3,C:0.5);")
phy.n_leaves      # 3
phy.n_total       # 5  (3 leaves + 2 internal)
length(phy.branch_lengths)   # 4
nnz(phy.Q_topology)          # 13  (5 diagonal + 8 off-diagonal entries)
```
"""
function augmented_phy(newick::AbstractString)
    s = String(newick)
    occursin('\0', s) && throw(ArgumentError(
        "Newick strings cannot contain literal NUL bytes"))
    c = _NewickCursor(s, firstindex(s))
    _skip_newick_whitespace!(c)

    node_parent = Int[]
    node_is_leaf = Bool[]
    node_name = String[]
    node_length = Float64[]
    leaf_indices = Int[]
    leaf_names = String[]

    root_idx = _parse_node!(c, node_parent, node_is_leaf, node_name,
                            node_length, leaf_indices, leaf_names)

    _skip_newick_whitespace!(c)
    _peek(c) == ';' || throw(ArgumentError(
        "Newick string must end with ';' at position $(c.i)"))
    _advance(c)
    _skip_newick_whitespace!(c)
    _peek(c) == '\0' || throw(ArgumentError(
        "extra characters after end of tree at position $(c.i)"))

    # Reindex: place leaves first (1:p) in the order they were encountered,
    # then internal nodes in the order they were added (post-order, so the
    # root is last). This is the convention the likelihood code uses.
    n_total = length(node_parent)
    p = length(leaf_indices)
    p > 0 || throw(ArgumentError("phylogenetic tree must contain at least one leaf"))
    leaf_names = _phy_validate_leaf_names(leaf_names, p)
    root_idx, _ = _phy_validate_topology(node_parent, node_is_leaf)
    perm = Vector{Int}(undef, n_total)   # perm[new_idx] = old_idx
    new_idx_of = Vector{Int}(undef, n_total)
    for (new_i, old_i) in enumerate(leaf_indices)
        perm[new_i] = old_i
        new_idx_of[old_i] = new_i
    end
    next_new = p + 1
    for old_i in 1:n_total
        node_is_leaf[old_i] && continue
        perm[next_new] = old_i
        new_idx_of[old_i] = next_new
        next_new += 1
    end
    next_new == n_total + 1 ||
        error("internal indexing error: expected $next_new == $(n_total + 1)")

    # Build sparse Q: every non-root node has one parent edge whose supplied
    # branch length is retained exactly. No binary resolution or normalization
    # is introduced for multifurcating trees.
    edges = Tuple{Int,Int,Float64}[]
    new_root_idx = new_idx_of[root_idx]
    for old_child in 1:n_total
        parent_old = node_parent[old_child]
        parent_old == 0 && continue       # root has no parent edge
        new_child  = new_idx_of[old_child]
        new_parent = new_idx_of[parent_old]
        push!(edges, (new_parent, new_child, node_length[old_child]))
    end
    Q, branch_lengths = _phy_topology_precision(edges, n_total)
    length(branch_lengths) == n_total - 1 ||
        throw(ArgumentError("expected $(n_total - 1) edges, got $(length(branch_lengths))"))

    leaf_idx_new = collect(1:p)           # by construction
    leaf_names_new = leaf_names           # already in encounter order
    return AugmentedPhy{Float64}(p, n_total, Q, leaf_idx_new, leaf_names_new,
                                 branch_lengths, new_root_idx)
end

"""
    make_phy(edges::AbstractVector{<:Tuple}, n_leaves::Integer;
             root_index::Integer = -1) :: AugmentedPhy{Float64}

Convenience constructor: build an `AugmentedPhy` from a list of edges
`(parent_id, child_id, branch_length)` with contiguous integer node ids
`1:n_total`. Leaves must be exactly `1:n_leaves`; internal nodes follow them.
The root may have any internal id. Every internal node must have at least two
children, and every non-root node must have one parent.

If `root_index < 0` it is auto-detected as the unique node that is not a
child in any edge.

This bypasses the Newick parser — useful for tests and for trees that
arrive from another tool already as edge lists.
"""
function make_phy(edges::AbstractVector, n_leaves::Integer;
                  root_index::Integer = -1,
                  leaf_names::Union{Nothing,AbstractVector{<:AbstractString}} = nothing)
    n_leaves isa Bool && throw(ArgumentError("n_leaves must be an integer, not Bool"))
    n_leaves > 1 || throw(ArgumentError(
        "make_phy edge-list trees require at least two leaves"))
    root_index isa Bool && throw(ArgumentError("root_index must be an integer, not Bool"))

    # Validate values and contiguity from the edge records themselves before
    # allocating a node-indexed vector. In particular an accidental id such as
    # one billion must fail as a non-contiguous topology, not reserve memory.
    normalized = Tuple{Int,Int,Float64}[]
    node_ids = Int[]
    seen_edges = Set{Tuple{Int,Int}}()
    for edge in edges
        edge isa Tuple && length(edge) == 3 || throw(ArgumentError(
            "each phylogenetic edge must be `(parent, child, branch_length)`"))
        parent, child, branch = edge
        parent isa Integer && !(parent isa Bool) || throw(ArgumentError(
            "phylogenetic parent ids must be positive integers"))
        child isa Integer && !(child isa Bool) || throw(ArgumentError(
            "phylogenetic child ids must be positive integers"))
        parent > 0 && child > 0 || throw(ArgumentError(
            "phylogenetic node ids must be positive"))
        parent == child && throw(ArgumentError("phylogenetic edges cannot be self edges"))
        parent <= typemax(Int) && child <= typemax(Int) || throw(ArgumentError(
            "phylogenetic node id is not representable as Int"))
        parent_i, child_i = Int(parent), Int(child)
        (parent_i, child_i) in seen_edges && throw(ArgumentError(
            "phylogenetic edges must be unique"))
        push!(seen_edges, (parent_i, child_i))
        length_value = _phy_branch_length(branch, "edge $parent_i -> $child_i")
        push!(normalized, (parent_i, child_i, length_value))
        push!(node_ids, parent_i, child_i)
    end
    isempty(normalized) && throw(ArgumentError("make_phy requires at least one edge"))
    ids = unique(node_ids)
    n_total = length(ids)
    all(id -> 1 <= id <= n_total, ids) || throw(ArgumentError(
        "phylogenetic node ids must be contiguous integers 1:$n_total"))
    n_total > n_leaves || throw(ArgumentError(
        "phylogenetic topology must contain internal nodes beyond leaves 1:$n_leaves"))

    parent_of = zeros(Int, n_total)
    is_leaf = [node <= n_leaves for node in 1:n_total]
    for (parent, child, _) in normalized
        is_leaf[parent] && throw(ArgumentError("leaf node $parent cannot be a parent"))
        parent_of[child] == 0 || throw(ArgumentError(
            "non-root node $child must have exactly one parent"))
        parent_of[child] = parent
    end
    requested_root = if root_index < 0
        nothing
    elseif 1 <= root_index <= n_total
        Int(root_index)
    else
        throw(ArgumentError("root_index $root_index is outside 1:$n_total"))
    end
    root, _ = _phy_validate_topology(parent_of, is_leaf; root_index = requested_root)
    names = leaf_names === nothing ? ["L$(t)" for t in 1:n_leaves] :
        _phy_validate_leaf_names(leaf_names, Int(n_leaves))
    Q, branch_lengths = _phy_topology_precision(normalized, n_total)
    length(normalized) == n_total - 1 || throw(ArgumentError(
        "phylogenetic tree must have $(n_total - 1) edges, got $(length(normalized))"))
    return AugmentedPhy{Float64}(Int(n_leaves), n_total, Q, collect(1:n_leaves), names,
                                 branch_lengths, root)
end

"""
    sigma_phy_dense(phy::AugmentedPhy; σ²_phy::Real = 1.0) :: Matrix

Build the dense (p × p) leaf covariance `Σ_phy = σ²_phy · (S Q_cond⁻¹ S')`
where `Q_cond` is `phy.Q_topology` with the root row/col removed and `S`
selects leaves. This is what the existing dense path expects; used by
verification tests to compare sparse vs. dense.

This is **O(p³)** in storage and time — only intended for small trees in
tests. Do NOT call it on the real workload; the entire point of
`AugmentedPhy` is to avoid materialising Σ_phy.
"""
function sigma_phy_dense(phy::AugmentedPhy; σ²_phy::Real = 1.0)
    p = phy.n_leaves
    keep = setdiff(1:phy.n_total, [phy.root_index])
    Q_cond = Matrix(phy.Q_topology[keep, keep])
    Σ_full = inv(Symmetric(Q_cond))
    leaf_pos = [findfirst(==(phy.leaf_indices[t]), keep) for t in 1:p]
    return σ²_phy .* Σ_full[leaf_pos, leaf_pos]
end

"""
    random_balanced_tree(p::Integer; branch_length::Real = 0.1) :: AugmentedPhy

Build a near-balanced binary tree with `p` leaves. All branch lengths
equal `branch_length`. Used in benchmarks and scaling tests.

When `p` is a power of 2 this is perfectly balanced. Otherwise the
left-over leaf at each level is carried up one extra step (so the tree
remains binary, just with slightly uneven depths). Branch lengths stay
uniform — the goal is a representative sparse-tree topology, not an
ultrametric one.
"""
function random_balanced_tree(p::Integer; branch_length::Real = 0.1)
    p > 0 || error("p must be > 0; got $p")
    edges = Tuple{Int,Int,Float64}[]
    current_level = collect(1:p)
    next_id = p + 1
    while length(current_level) > 1
        new_level = Int[]
        i = 1
        while i + 1 <= length(current_level)
            parent = next_id; next_id += 1
            push!(edges, (parent, current_level[i],     Float64(branch_length)))
            push!(edges, (parent, current_level[i + 1], Float64(branch_length)))
            push!(new_level, parent)
            i += 2
        end
        # leftover odd node (if length(current_level) is odd) carries up
        if i == length(current_level)
            push!(new_level, current_level[i])
        end
        current_level = new_level
    end
    root_idx = current_level[1]
    return make_phy(edges, p; root_index = root_idx)
end

"""
    random_caterpillar_tree(p::Integer; branch_length::Real = 0.1) :: AugmentedPhy

Build a **caterpillar** (maximally unbalanced "ladder") binary tree with `p`
leaves: leaves 1 and 2 form the first cherry, that node joins leaf 3 at the next
internal node, and so on — a chain of `p-1` internal nodes of depth `p-1`.

The shape is the worst case for sparse-Cholesky fill-in: O(p) on a *balanced*
tree (log-depth) does not by itself prove O(p) on a deep caterpillar, so this
generator is useful for testing scaling across tree shapes. All branch lengths equal
`branch_length`.
"""
function random_caterpillar_tree(p::Integer; branch_length::Real = 0.1)
    p > 0 || error("p must be > 0; got $p")
    b = Float64(branch_length)
    if p == 1
        # Degenerate: a single leaf is its own root (n_total = 1, no edges).
        Q = sparse([1], [1], [1.0 / b], 1, 1)
        return AugmentedPhy{Float64}(1, 1, Q, [1], ["L1"], [b], 1)
    end
    edges = Tuple{Int,Int,Float64}[]
    next_id = p + 1
    cur = next_id; next_id += 1            # first cherry over leaves 1 and 2
    push!(edges, (cur, 1, b))
    push!(edges, (cur, 2, b))
    for leaf in 3:p                        # ladder: each leaf gets a new node above
        parent = next_id; next_id += 1
        push!(edges, (parent, cur, b))
        push!(edges, (parent, leaf, b))
        cur = parent
    end
    return make_phy(edges, p; root_index = cur)
end

"""
    augmented_tree_precision(phy::AugmentedPhy) -> (Q, leaf_pos, q)

Return the **root-conditioned augmented topology precision** `Q =
Q_topology[keep, keep]` — a sparse, O(p)-nnz, positive-definite matrix over the
`q = n_total - 1` non-root augmented nodes — together with the map `leaf_pos[t]` from
leaf `t ∈ 1:p` to its row/column in `Q`, and `q`. This is the sparse precision
the end-to-end O(p) Gaussian path feeds DIRECTLY as `Qₖ`, bypassing the dense
leaf-correlation inversion.
"""
function augmented_tree_precision(phy::AugmentedPhy)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    q = length(keep)
    Q = phy.Q_topology[keep, keep]
    pos = Dict(node => i for (i, node) in enumerate(keep))
    leaf_pos = [pos[phy.leaf_indices[t]] for t in 1:phy.n_leaves]
    return Q, leaf_pos, q
end

"""
    phylo_tree_height(phy::AugmentedPhy) -> Float64

Maximum root-to-tip path length of `phy`. For an ultrametric tree this is the
common **tip variance** its covariance implies when `σ²_phy = 1`; for a
non-ultrametric tree it is the largest tip variance.

O(p): a breadth-first walk over the sparse topology, where an edge's length is
recovered as `-1/Q_topology[i, j]`. `sigma_phy_dense` would give the same number
but inverts a dense matrix, so it is unusable as a routine check.

**Why this matters.** DRModels.jl builds its phylogenetic covariance from the branch
lengths **as supplied**. On an ultrametric tree of height `h`, the fitted
`sd_phylo` carries a factor `sqrt(h)` relative to unit-tip-variance correlation
scale. R's drmTMB instead standardises via `ape::vcv(tree, corr = TRUE)`, whose
tips always have variance 1. A non-ultrametric conversion is tip-wise, not one
scalar. See [`drm_phylo_penalty`](@ref), where these choices change what `sd_u`
*means*.

For an **ultrametric** tree, divide all branch lengths by `h` before fitting to
match drmTMB's unit-tip-variance correlation scale. A non-ultrametric tree
needs the tip-wise standardization `D^{-1/2} Σ D^{-1/2}`, not one scalar branch
rescaling; this raw-branch constructor deliberately performs neither transform.
"""
function phylo_tree_height(phy::AugmentedPhy)
    Q = phy.Q_topology
    n = phy.n_total
    depth = fill(-1.0, n)
    depth[phy.root_index] = 0.0
    queue = [phy.root_index]
    cursor = 1
    @inbounds while cursor <= length(queue)
        i = queue[cursor]
        cursor += 1
        for k in nzrange(Q, i)
            j = rowvals(Q)[k]
            j == i && continue
            qij = nonzeros(Q)[k]
            qij == 0 && continue
            len = -1.0 / qij            # off-diagonal is -1/branch_length
            len > 0 || continue
            if depth[j] < 0
                depth[j] = depth[i] + len
                push!(queue, j)
            end
        end
    end
    h = 0.0
    @inbounds for t in 1:phy.n_leaves
        d = depth[phy.leaf_indices[t]]
        d > h && (h = d)
    end
    return h
end

# One warning per tree per session: the scale is a REPORTING convention, not an
# error, so it must not shout on every fit in a loop.
const _PHYLO_HEIGHT_WARNED = Set{UInt64}()

function _warn_if_tree_not_unit_height(phy::AugmentedPhy)
    h = try
        phylo_tree_height(phy)
    catch
        return nothing       # a diagnostic must never break a fit
    end
    (isfinite(h) && h > 0) || return nothing
    isapprox(h, 1.0; rtol = 1e-6) && return nothing
    key = hash((phy.n_leaves, round(h; digits = 12)))
    key in _PHYLO_HEIGHT_WARNED && return nothing
    push!(_PHYLO_HEIGHT_WARNED, key)
    @warn "drm: this tree's maximum tip height is $(round(h; digits = 6)), not 1, so " *
          "`sd_phylo` is on the RAW branch-length scale. For an ultrametric tree it is a factor " *
          "$(round(sqrt(h); digits = 6)) = sqrt($(round(h; digits = 6))) away from the correlation " *
          "scale R's drmTMB reports (via `ape::vcv(tree, corr = TRUE)`); rescale branches by " *
          "1/$(round(h; digits = 6)) to match. A non-ultrametric tree requires tip-wise correlation " *
          "standardization, not this scalar rescaling. " *
          "`phylo_tree_height(tree)` reports this."
    return nothing
end
