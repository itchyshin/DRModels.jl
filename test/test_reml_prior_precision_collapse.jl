# test_reml_prior_precision_collapse.jl — #563: `_reml_prior_precision`
# (src/reml_q4.jl, added by #579 as a LOCAL guard against `prior_precision`'s
# `sparse(Λinv)` zero-dropping at an exactly diagonal Λ) was redundant now
# that #577 fixed `prior_precision` itself (src/sparse_aug_plsm.jl) to store
# the full q×q axis block unconditionally, and has been deleted (its callers
# now call `prior_precision` directly).
#
# PROOF (recorded on the pre-collapse head before deleting the helper): both
# builders produced bit-identical sparse P for the biv-q4-phylo-reml
# fixture's Q_cond, at BOTH an exactly-diagonal Λ (the case #579 was
# guarding) and a non-diagonal Λ — same sparsity pattern, same nzval, same
# nnz (1376 in both cases). See .unlazy/563-fu-a/gates/leaf-collapse.md.
#
# What remains as a live regression guard (the actual bug #575/#579 found):
# at an exactly-diagonal Λ, `prior_precision` must still store the FULL 4×4
# axis block per tree node, not just the diagonal — i.e. its nnz at a
# diagonal Λ must equal its nnz at a dense Λ.
#
#   julia --project=. -e 'using DRModels, Test; include("test/test_reml_prior_precision_collapse.jl")'

module TestRemlPriorPrecisionCollapse

using DRModels
using Test
using SparseArrays
using LinearAlgebra
using DelimitedFiles: readdlm

const FIXTURE = joinpath(@__DIR__, "parity", "q4-reml", "biv-q4-phylo-reml")

function _load_data(dir)
    raw, header = readdlm(joinpath(dir, "data.csv"), ','; header = true)
    cols = Symbol.(strip.(string.(vec(header))))
    numeric = Set((:y1, :y2, :x))
    pairs = map(enumerate(cols)) do (j, name)
        col = raw[:, j]
        if name in numeric
            name => Float64[parse(Float64, string(v)) for v in col]
        else
            name => string.(col)
        end
    end
    return NamedTuple(pairs)
end

const DAT = _load_data(FIXTURE)
const TREE = read(joinpath(FIXTURE, "tree.newick"), String)
const FORM = bf(mu1    = @formula(y1 ~ x + phylo(1 | species)),
                mu2    = @formula(y2 ~ x + phylo(1 | species)),
                sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
                sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
                rho12  = @formula(rho12 ~ 1))

# Rebuild exactly what `_fit_bivariate_q4_phylo` hands the engine — mirrors
# test_575_exact_reml_gradient.jl's `_engine_inputs`.
function _engine_Q_cond()
    rhs = Dict(FORM.forms)
    fixed, marker = DRModels._bivariate_q4_marker(rhs)
    grp = marker[2]
    phy = DRModels._as_augmented_phy(TREE)

    y1, X1, _ = DRModels._design(FORM.response1, fixed[:mu1], DAT)
    y2, X2, _ = DRModels._design(FORM.response2, fixed[:mu2], DAT)
    _, Xs1, _ = DRModels._design(FORM.response1, fixed[:sigma1], DAT)
    _, Xs2, _ = DRModels._design(FORM.response1, fixed[:sigma2], DAT)
    _, Xr, _  = DRModels._design(FORM.response1, fixed[:rho12], DAT)

    obs1 = DRModels._observed_response_mask(y1)
    obs2 = DRModels._observed_response_mask(y2)
    species = DRModels._phylo_species_index(phy, getproperty(DAT, grp))
    _, Q_cond = DRModels.make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr; species = species)
    return Q_cond
end

const Q_COND = _engine_Q_cond()

# The exactly-diagonal case #579 was guarding against (fit_q4_reml's own
# default warm start).
const LAM_DIAG = Matrix(0.3I(4))

# A non-diagonal case (TMB's fitted point on this fixture; test_575's
# `LAMBDA_TMB`).
const LAM_DENSE = [ 0.5288095   0.25509007 -0.1228962  -0.15554224;
                    0.25509007  0.28551843 -0.1794685  -0.02445802;
                   -0.1228962  -0.1794685   0.4857264  -0.10269499;
                   -0.15554224 -0.02445802 -0.10269499  0.16678147]

@testset "issue #563: _reml_prior_precision is redundant after #577" begin
    # `_reml_prior_precision` no longer exists — confirm the collapse.
    @test !isdefined(DRModels, :_reml_prior_precision)

    # The regression #575/#579 actually found: at an exactly-diagonal Λ, the
    # shared `prior_precision` must still store the FULL 4×4 axis block per
    # tree node (not just the diagonal) — i.e. the diagonal case's nnz must
    # equal the dense case's nnz. This is the guard that survives the helper
    # deletion. Expected nnz (1376) pinned by the pre-collapse proof against
    # the now-deleted `_reml_prior_precision` — see the module docstring.
    nnz_dense = nnz(DRModels.prior_precision(Q_COND, inv(LAM_DENSE)))
    nnz_diag  = nnz(DRModels.prior_precision(Q_COND, inv(LAM_DIAG)))
    @test nnz_diag == nnz_dense
    @test nnz_diag == 1376

    @info "prior_precision collapse guard" nnz_diag nnz_dense
end

end # module TestRemlPriorPrecisionCollapse
