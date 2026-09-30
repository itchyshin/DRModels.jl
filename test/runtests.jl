using DRModels
using Test, LinearAlgebra, SparseArrays, Random
# --- Deterministic Julia-suite sharding (DRM_TEST_SHARD="k/N") -------------
# Splits the auto-discovered, sharded include calls below across N shards by
# file position: shard k (1-based) gets file i when (i - 1) % N == k - 1.
# Unset or empty means "run everything", so a local `julia test/runtests.jl`
# is unchanged. Helpers live in shard_util.jl and are exercised standalone in
# test_shard_selection.jl, which proves the N shards partition the file list
# exactly -- disjoint AND complete, so sharding cannot silently drop a file.
#
# Deliberately NOT sharded: the JET gate (which has an else-branch) and the
# three parity suites behind DRM_PARITY_TESTS. Those are guarded by their own
# conditions; putting a shard test in front of a conditional would change
# what the guard means.
#
# CONFLICT-FREE REGISTRATION: this file used to hand-list every test_*.jl via
# a literal `_shard_include("...")` call, so almost every PR touched the same
# few lines here (and the matching NEWS.md line -- see news/README.md). Two
# PRs landing close together flipped every OTHER open PR to CONFLICTING on
# GitHub even though `.gitattributes` marks both files `merge=union` --
# GitHub's mergeability check does not honor local/custom merge drivers.
# Adding a new test_*.jl file now needs NO edit to this file at all: it is
# picked up by readdir() below. Only two lists stay hand-maintained, both
# short and rarely touched: _TEST_ORDER (files that must run before the
# general suite) and _TEST_EXCLUDE (files intentionally not run through the
# discovery loop, e.g. because they are included some other way).
include("shard_util.jl")

_shard_spec = get(ENV, "DRM_TEST_SHARD", "")
_SHARD = isempty(_shard_spec) ? nothing : _parse_shard_spec(_shard_spec)

# Dry-run mode: print the shard plan (which file lands in which shard) without
# loading or running anything. `DRM_TEST_LIST_ONLY=1 julia --project=test -e
# 'include("test/runtests.jl")'` (optionally with DRM_TEST_SHARD set too).
const _LIST_ONLY = get(ENV, "DRM_TEST_LIST_ONLY", "0") == "1"

# Files that must run before the general suite (packaging/shape guards) --
# kept explicit and small; everything else is discovered.
const _TEST_ORDER = [
    "test_shard_selection.jl",         # sharding helper self-test
    "test_runtests_include_list.jl",   # this file's own shape (see below)
    "test_load_contract.jl",
    "test_aqua.jl",                    # General-registry hygiene; early so packaging regressions surface first
]

# Files present in test/ that match the test_*.jl discovery pattern but are
# deliberately NOT run through the discovery loop.
const _TEST_EXCLUDE = [
    "test_qgate_jet.jl",           # included manually below, gated on the JET package being present
    "test_analytic_grad.jl",       # #465: investigated, NOT wired -- superseded by test_qgate_fd_gradient.jl
    "test_q4_laplace.jl",          # #465: investigated, NOT wired -- superseded by an obsolete bench POC
    "test_corr_locscale_equiv.jl", # deferred cluster-① follow-up (Laplace-vs-GHQ optimum gap; not a σ-phylo blocker)
]

_discovered = sort(filter(f -> occursin(r"^test_.*\.jl$", f) &&
                                !(f in _TEST_ORDER) && !(f in _TEST_EXCLUDE),
                           readdir(@__DIR__)))

# --- BLAS thread count: pin once, then guard it per file --------------------
# The suite's numerical pins were all measured at one BLAS thread (the repo's
# stated invariant), and ten test files assert or set BLAS=1 themselves. CI
# does not set OPENBLAS_NUM_THREADS, so without this pin a shard starts at 2
# and silently switches to 1 partway through (the first test_joint_missing_*
# file sets it and never restores). Pin it here so every file, on every
# platform, runs at the same count.
_LIST_ONLY || BLAS.set_num_threads(1)

# Test-isolation guard: the BLAS thread count is process-global, so every
# file must leave `BLAS.get_num_threads()` where the suite set it and leave
# no `_with_pinned_blas` scope open. A different count changes the summation
# order inside dense kernels, which perturbs every tight-tolerance numerical
# test that runs after the leak. Checked per file so a failure names the
# leaking file, not a victim far downstream. Measured 2026-09-19 (bisect over
# the full include order): no file leaks today. The in-suite failures of
# test_q4_perf_identities.jl's rtol=1e-8 vcov pins that motivated this guard
# were NOT a leak: `Pkg.test()` runs with `--check-bounds=yes`, and that
# codegen change alone reproduces them bit-for-bit in a pristine session.
const _BLAS_THREADS_AT_START = _LIST_ONLY ? 1 : BLAS.get_num_threads()
function _check_blas_restored(path::AbstractString)
    nt = BLAS.get_num_threads()
    scopes = DRModels._blas_pin_scopes[]
    if nt != _BLAS_THREADS_AT_START || scopes != 0
        @testset "BLAS state restored after $path" begin
            @test nt == _BLAS_THREADS_AT_START
            @test scopes == 0
        end
    end
end

_shard_pos = Ref(0)

function _shard_include(path::AbstractString)
    _shard_pos[] += 1
    _in_shard = _SHARD === nothing || (_shard_pos[] - 1) % _SHARD[2] == _SHARD[1] - 1
    if _LIST_ONLY
        if _in_shard
            shard_label = _SHARD === nothing ? "-" : "$(_SHARD[1])/$(_SHARD[2])"
            println("  [shard $shard_label] $path")
        end
        return
    end
    if _in_shard
        include(path)
        _check_blas_restored(path)
    end
end

_n_test_files = length(_TEST_ORDER) + length(_discovered)
_n_selected = _SHARD === nothing ? _n_test_files :
    length(_shard_indices(_n_test_files, _SHARD[1], _SHARD[2]))
println(_SHARD === nothing ?
    "DRModels tests: all $_n_test_files files (unsharded)" :
    "DRModels tests: shard $(_SHARD[1])/$(_SHARD[2]) - $_n_selected of $_n_test_files files")

if !_LIST_ONLY
    @testset "DRModels.jl — engine loads + phylo foundation" begin
        @testset "public API present" begin
            @test DRModels.DRM === DRModels
            for f in (:fit_q4_sparse_tmb, :marginal_and_exact_grad, :make_problem,
                      :estep_mode, :prior_precision, :augmented_phy,
                      :random_balanced_tree, :sigma_phy_dense, :takahashi_selinv,
                      :lc_metric)
                @test isdefined(DRModels, f)
            end
        end

        @testset "sparse augmented phylo precision (p=8)" begin
            Random.seed!(1); p = 8
            phy = random_balanced_tree(p; branch_length = 0.2)
            Σ = sigma_phy_dense(phy; σ²_phy = 1.0)          # dense leaf covariance
            @test size(Σ) == (p, p)
            @test isposdef(Symmetric(Σ))                    # well-conditioned tree cov
            # kron(Q_cond, Λ⁻¹) prior precision is sparse + PD (the O(p) engine core)
            Λ = Matrix(Symmetric(0.3I(4) + 0.02 * (ones(4, 4) - I(4))))
            keep = setdiff(1:phy.n_total, [phy.root_index])
            P = prior_precision(phy.Q_topology[keep, keep], inv(Λ))
            @test issparse(P)
            @test isposdef(Symmetric(Matrix(P)))
        end
    end
end

# Julia General-registry hygiene (Aqua.jl) + the shape guards run first, so
# packaging regressions surface before the numerical suite.
for f in _TEST_ORDER
    _shard_include(f)
end

# Standing Workflow Q JET gate (Karpinski): type-stability of hot lc↔Λ kernels.
# JET lives in test/Project.toml — skip gracefully when absent (bare
# `julia --project=. test/runtests.jl`). Macro body is in a separate file so it
# is only parsed when JET is present (same pattern as GLLVM.jl). Deliberately
# not part of the discovery loop above (excluded via _TEST_EXCLUDE): it needs
# its own else-branch instead of a plain shard slot.
if _LIST_ONLY
    println("  [unsharded, gated on JET] test_qgate_jet.jl")
else
    const _HAS_JET = Base.find_package("JET") !== nothing
    @testset "Q-gate: JET type-stability (lc_to_Λ / Λ_to_lc)" begin
        if _HAS_JET
            @eval using JET
            include("test_qgate_jet.jl")
        else
            @info "JET not in this environment — run `Pkg.test()` for the Workflow Q JET gate"
            @test_skip false
        end
    end
end

# Every other test_*.jl file in test/, sorted for a deterministic and
# reviewable shard assignment, not hand-listed (see the CONFLICT-FREE
# REGISTRATION note above).
for f in _discovered
    _shard_include(f)
end

# Gated real-parity suite vs committed drmTMB fixtures (off by default).
# Native `drm()` path (#17) plus `drm_bridge` marshalling path (#370).
# Deliberately not part of the discovery loop above: these three live under
# test/parity/ (not top-level test/), so readdir(@__DIR__) never finds them,
# and they are guarded behind DRM_PARITY_TESTS rather than sharded.
if _LIST_ONLY
    println("  [unsharded, gated on DRM_PARITY_TESTS] parity/runparity.jl, parity/runparity_bridge.jl, parity/runparity_bridge_formula.jl")
elseif get(ENV, "DRM_PARITY_TESTS", "0") == "1"
    @testset "R-parity vs drmTMB 0.6.0" begin
        include("parity/runparity.jl")
    end
    @testset "R-parity via drm_bridge vs drmTMB 0.6.0" begin
        include("parity/runparity_bridge.jl")
    end
    @testset "R-parity via drm_bridge R-formula constructs vs drmTMB 0.7.0 (#467)" begin
        include("parity/runparity_bridge_formula.jl")
    end
else
    @info "R-parity suite skipped (set DRM_PARITY_TESTS=1 to run)"
end
