# test_qgate_alloc_inner.jl — STANDING engine-quality Q-gate for issue #15.
#
# Zero-allocation gate on the inner Newton mode-finder's PURE-JULIA arithmetic.
#
# Honest scope (per report/qgates-design.md §#15): the CHOLMOD sparse
# factorisation / triangular solve allocate INSIDE SuiteSparse, which is not
# Julia-controllable, so this gate EXCLUDES the factor object and targets the
# pure-Julia arithmetic the inner loop repeats every iteration — here the
# prior-coupling gradient term g .= P·u (`aug_prior_grad!`, an in-place sparse
# matvec). The gate asserts:
#   (1) after warm-up it allocates at most a tiny p-INDEPENDENT budget (the
#       in-place sparse matvec is allocation-free in principle; the small budget
#       absorbs SparseArrays/CHOLMOD-version internals), and
#   (2) that is FLAT across p ∈ {100, 1000} (a p-scaling allocation regression in
#       the inner arithmetic would trip it).
# It also DEMONSTRATES THE GATE HAS TEETH: the allocating reference
# `g .= P * u` (a fresh temporary) allocates > 0 and the same magnitude grows
# with p — exactly the regression class the gate guards against.

using DRModels
using Test, LinearAlgebra, SparseArrays, Random

# An allocating variant of the prior-coupling term (the regression we guard
# against): `P * u` materialises a temporary every call.
_alloc_prior_grad(P, u) = (g = P * u; g)

@testset "Q-gate (#15): inner-loop pure-Julia arithmetic is zero-alloc + flat" begin
    function prior_and_state(p)
        # Real augmented prior precision P = kron(Q_cond, Λ⁻¹), the exact object
        # the inner mode-finder multiplies by u every Newton iteration.
        phy = random_balanced_tree(p; branch_length = 0.2)
        keep = setdiff(1:phy.n_total, [phy.root_index])
        Q_cond = phy.Q_topology[keep, keep]
        Λ = [0.30 0.06 0.02 0.0; 0.06 0.28 0.0 0.03;
             0.02 0.0 0.14 0.01; 0.0 0.03 0.01 0.16]
        Λ = Matrix(Symmetric((Λ + Λ') / 2))
        P = prior_precision(Q_cond, inv(Λ))
        u = randn(size(P, 1))
        g = similar(u)
        return P, u, g
    end

    Random.seed!(15)
    allocs = Int[]
    for p in (100, 1000)
        P, u, g = prior_and_state(p)
        # Warm up (compile) the in-place kernel, then measure a SINGLE call.
        aug_prior_grad!(g, P, u)
        a = @allocated aug_prior_grad!(g, P, u)
        push!(allocs, a)
        # Correctness: the in-place result equals the allocating reference.
        @test g ≈ P * u
    end

    # (1) pure-Julia arithmetic allocates a p-INDEPENDENT amount after warm-up.
    #     The in-place sparse matvec is allocation-free in principle; we allow a
    #     tiny fixed budget for SparseArrays/CHOLMOD-version internals (the gate's
    #     teeth come from p-independence + the regression check below, not the
    #     exact byte count). `budget` is p-independent: a p-scaling allocation
    #     regression in the inner arithmetic would blow past it.
    budget = 64
    @test allocs[1] <= budget
    @test allocs[2] <= budget
    # (2) flat across p — the result for p=1000 (10× larger) is NOT larger than
    #     for p=100. A p-scaling inner allocation would break this.
    @test allocs[2] <= max(allocs[1], budget)

    # TEETH: the allocating reference trips the gate — > budget AND p-growing.
    P1, u1, _ = prior_and_state(100)
    P2, u2, _ = prior_and_state(1000)
    _alloc_prior_grad(P1, u1); _alloc_prior_grad(P2, u2)   # warm up
    a_bad_small = @allocated _alloc_prior_grad(P1, u1)
    a_bad_big   = @allocated _alloc_prior_grad(P2, u2)
    @test a_bad_small > budget            # the gate FAILS on this variant
    @test a_bad_big > a_bad_small         # ...and the regression scales with p
end
