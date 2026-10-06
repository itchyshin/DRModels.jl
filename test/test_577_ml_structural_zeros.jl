# test_577_ml_structural_zeros.jl — issue #577.
#
# `prior_precision(Q, Λ⁻¹) = kron(Q, sparse(Λinv))` and `sparse` DROPS EXACT
# ZEROS. At an exactly diagonal Λ — including `fit_q4_sparse_tmb`'s own default
# warm start `Λ0 = 0.3I(4)` — the cross-axis entries of `H_uu` are therefore
# structurally ABSENT at non-leaf nodes (which carry no data block to supply
# them). The CHOLMOD pattern, and with it the Takahashi selected inverse, then
# cannot furnish the logdet-H trace terms that `marginal_and_exact_grad`
# genuinely needs, and the exact gradient is silently wrong in exactly those
# components.
#
# #575 fixed this on the REML path (`_reml_prior_precision`) and deliberately
# left the ML path alone; #577 is that residue. The fix taken here is at the
# ROOT — `prior_precision` itself now stores the full block — so every caller
# that feeds P into a selected inverse is covered, not just this one.
#
# MEASURED on the biv-q4-phylo-reml fixture at Λ = 0.3I, before the fix:
#   per-lc abs err  [1.6e-7, 3.7e-7, 0.751, 6.5e-6, 1.2e-5, 2.9e-5, 12.1,
#                    8.4e-9, 3.1e-8, 8.7e-6]        <- components 3 and 7
#   with a 1e-6 off-diagonal added (structurally full P): max err 3.0e-5,
#   i.e. the central-difference reference's own noise floor.
# The 1e-3 absolute bar below therefore sits ~1.5 orders above the clean state
# and ~4 orders below the defect.
#
#   julia --project=. -e 'using DRModels, Test; include("test/test_577_ml_structural_zeros.jl")'

module Test577MLStructuralZeros

using DRModels
using Test
using LinearAlgebra
using SparseArrays
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

# Same construction the engine hands `marginal_and_exact_grad`; mirrors
# test_q4_reml_warm_restart.jl so no src edit is needed to reach it.
function _engine_inputs()
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
    prob, Q_cond = DRModels.make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr; species = species)

    β1 = X1[obs1, :] \ y1[obs1]
    β2 = X2[obs2, :] \ y2[obs2]
    res1 = y1[obs1] .- X1[obs1, :] * β1
    res2 = y2[obs2] .- X2[obs2, :] * β2
    β0 = (mu1 = β1, mu2 = β2,
          s1 = DRModels._initial_scale_beta(Xs1, res1), s2 = DRModels._initial_scale_beta(Xs2, res2),
          rho = zeros(size(Xr, 2)))
    return prob, Q_cond, β0
end

@testset "issue #577: prior_precision keeps structural zeros" begin
    # Unit-level, no fit: the block must be structurally full whatever Λ is.
    # This is the guard that fails fast if anyone reintroduces `sparse(Λinv)`.
    Q = sparse([1, 2, 1, 2], [1, 1, 2, 2], [2.0, -1.0, -1.0, 2.0], 2, 2)
    for (label, Λinv) in ("diagonal"     => Matrix(1.0I(4)),
                          "block-diag"   => Matrix([1.0 0.2 0 0; 0.2 1.0 0 0;
                                                    0 0 1.0 0.2; 0 0 0.2 1.0]),
                          "dense"        => Matrix(1.0I(4)) .+ 0.1)
        P = DRModels.prior_precision(Q, Λinv)
        @test nnz(P) == nnz(Q) * 16          # every 4x4 axis entry stored
        @test Matrix(P) ≈ kron(Matrix(Q), Λinv)   # values unchanged
    end
end

@testset "issue #577: ML exact gradient at an exactly diagonal Lambda" begin
    prob, Q_cond, β0 = _engine_inputs()

    # The engine's own default warm start. Exactly diagonal, so this is the
    # worst case for the dropped-zeros pattern, and it is the point every
    # ML fit actually starts from.
    Λ_start = Matrix(0.3 * I(4))
    θ = DRModels.pack_theta(β0, Λ_start)
    nθ = length(θ)
    o6 = nθ - 10                      # the 10 log-Cholesky entries

    # u0 = nothing on every evaluation: a shared warm-start cache would let one
    # evaluation's mode contaminate the next one's objective (#575 heuristic 4).
    nll_at(t) = first(DRModels.marginal_and_exact_grad(prob, Q_cond, Vector{Float64}(t);
                                                  u0 = nothing, n_newton = 40))

    nll, g, _, _ = DRModels.marginal_and_exact_grad(prob, Q_cond, Vector{Float64}(θ);
                                               u0 = nothing, n_newton = 40)
    @test isfinite(nll)
    @test all(isfinite, g)

    # Step-scanned central difference on the same objective: keep the step whose
    # Richardson pair is most stable, rather than trusting one fixed h.
    g_fd = zeros(10)
    for k in 1:10
        best = NaN; best_gap = Inf; prev = NaN
        for h in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
            tp = copy(Vector{Float64}(θ)); tp[o6 + k] += h
            tm = copy(Vector{Float64}(θ)); tm[o6 + k] -= h
            d = (nll_at(tp) - nll_at(tm)) / (2h)
            if isfinite(prev) && abs(d - prev) < best_gap
                best_gap = abs(d - prev); best = d
            end
            prev = d
        end
        g_fd[k] = best
    end

    err = abs.(g[o6+1:o6+10] .- g_fd)
    scale = max(1.0, maximum(abs, g_fd))
    @info "issue #577 exact-vs-FD ML gradient (lc block)" max_abs_err = maximum(err) rel = maximum(err) / scale
    # Absolute is the honest statement of the defect (12.1 before the fix,
    # 3.0e-5 after); relative is the engine-quality restatement.
    @test maximum(err) <= 1e-3
    @test maximum(err) / scale <= 1e-4
end

end # module Test577MLStructuralZeros
