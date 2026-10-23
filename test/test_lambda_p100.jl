# test_lambda_p100.jl — the #472 descent was an ARTEFACT of the dropped-zeros
# sparsity pattern (#577). This file now asserts the REPAIRED behaviour.
#
# HISTORY, kept because the negative result was real when it was measured.
# From 2026-08-25 this file characterised a measured defect: the sparse-EM Λ
# M-step (`mstep_Lambda`) moved DOWNHILL on the real q4_p100 data for every step
# size in (1.0, 0.5, 0.25, 0.1, 0.01) — ~56 nats at the full step — and that is
# why production (`src/fit_ml_q4.jl`) abandoned the closed-form step for
# line-searched ascent. Its header named the tripwire condition explicitly:
# "if someone repairs `mstep_Lambda`, these assertions fail LOUDLY … a repair
# must also revisit the fence text here, in src/sparse_em_fit.jl, and in
# HANDOVER.md." On 2026-09-02 the tripwire fired; this is that revisit. The
# assertions are INVERTED to the newly measured truth, never loosened.
#
# THE CAUSE. `mstep_Lambda` reads its posterior covariance blocks from
# `takahashi_selinv`, which can only supply entries the CHOLMOD pattern carries.
# `prior_precision` built P through `sparse(Λinv)`, which DROPS EXACT ZEROS, so
# at this file's own exactly-diagonal `Λ0 = 0.3I` the cross-axis entries were
# structurally ABSENT at every non-leaf node — the M-step was reading structural
# zeros as if they were posterior covariances. #577 fixed that at the root.
#
# MEASURED on the q4_p100 fixture; L0 = -282.4242946 to 1e-12 either way:
#              Λ_em diagonal            max|off-diag|   ΔL at α=1
#   before    [0.650 1.212 0.633 1.209]     1.210         -55.80   (descent)
#   after     [0.324 0.300 0.298 0.295]     0.0078        +0.70    (ascent)
# After the fix ΔL is positive and monotone in α at every step size tested. The
# old Λ_em was not a near-miss: off-diagonals of 1.21 where the repaired step
# finds 0.008.
#
# WHY p=8 NEVER SAW IT. `test_lambda_direction.jl` asks the identical question
# at p=8 with the identical `Λ0 = 0.3I` and has always asserted ASCENT — and has
# always passed. The missing entries live on NON-LEAF nodes, so the artefact is
# negligible at p=8 and dominant at p=100. The two files contradicted each other
# for eight days and now agree.
#
# STILL OPEN — do NOT read this as "#472 is closed". On a SYNTHETIC p=100
# balanced tree with pure-noise responses the M-step still descends (~52 nats)
# with the pattern repaired, essentially unchanged. Whether that is a second,
# independent defect or the closed-form step overshooting under gross
# misspecification (Λ_true ≈ 0 there) is NOT established here. #472 wants
# re-measuring and re-scoping, not closing on this evidence.
# `mstep_Lambda`/`fit_em_aug` remain NOT reachable from the public `drm()` API.

using DRModels
using Test, LinearAlgebra, Statistics
using DelimitedFiles: readdlm

const D = DRModels
const FIX = joinpath(@__DIR__, "..", "bench", "fixtures")

@testset "sparse-EM Λ M-step ascends the true Laplace marginal (q4_p100; was #472's descent, an artefact fixed in #577)" begin
    raw, header = readdlm(joinpath(FIX, "q4_p100.csv"), ','; header = true)
    cols = Symbol.(strip.(string.(vec(header))))
    col(name) = raw[:, findfirst(==(name), cols)]
    y1 = Float64.(col(:y1)); y2 = Float64.(col(:y2)); x1 = Float64.(col(:x1))
    species = String.(strip.(string.(col(:species))))

    phy = augmented_phy(read(joinpath(FIX, "q4_p100_tree.nwk"), String))
    p = length(y1); n = p
    name2row = Dict(String(s) => i for (i, s) in enumerate(species))
    perm = [name2row[phy.leaf_names[k]] for k in 1:p]
    y1 = y1[perm]; y2 = y2[perm]; x1 = x1[perm]

    X1 = hcat(ones(n), x1); X2 = hcat(ones(n), x1)
    Xs1 = reshape(ones(n), n, 1); Xs2 = reshape(ones(n), n, 1); Xr = reshape(ones(n), n, 1)
    prob, Q_cond = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr)
    β = (mu1 = X1 \ y1, mu2 = X2 \ y2, s1 = [log(std(y1 .- X1 * (X1 \ y1)))],
         s2 = [log(std(y2 .- X2 * (X2 \ y2)))], rho = [0.0])

    # FULLY-converged E-step (no warm start, many Newton iters) for accurate marginals
    function L_of_Λ(Λ; nit = 60)
        P = prior_precision(Q_cond, inv(Λ))
        u, ch, _ = estep_mode(prob, P, β; u0 = nothing, n_newton = nit)
        return D.laplace_ll(prob, P, β, u, ch)
    end
    Λ0 = Matrix(0.3 * I(4)); L0 = L_of_Λ(Λ0)
    P0 = prior_precision(Q_cond, inv(Λ0)); u0, ch0, _ = estep_mode(prob, P0, β; n_newton = 60)
    Λem = D.mstep_Lambda(prob, Q_cond, u0, ch0)

    # REPAIRED behaviour (see header). The full closed-form step now ASCENDS the
    # true marginal by +0.70 nats, and every partial step along the update
    # direction ascends too, monotonically in α. This is what EM theory requires
    # of an M-step, and what the p=8 sibling has asserted all along.
    #
    # The bound is 0.3 rather than the measured 0.70: it must clear the ±0.1
    # inner-Newton platform noise the old header documented, with margin, while
    # staying far below the measured gain. It is a TRIPWIRE IN BOTH DIRECTIONS —
    # if the dropped-zeros pattern ever returns, ΔL drops to −55.8 and this fails
    # as loudly as the descent assertions did when the pattern was repaired.
    @test L_of_Λ(Λem) > L0 + 0.3
    for α in (1.0, 0.5, 0.25, 0.1, 0.01)
        Λα = Matrix(Symmetric(Λ0 .+ α .* (Λem .- Λ0)))
        @test L_of_Λ(Λα) > L0          # ascent at every step size
    end
    # The old Λ_em's signature failure was absurd off-diagonals (1.21 against a
    # repaired 0.0078). Assert the shape directly, so a future regression is
    # legible as "the pattern broke" rather than only as a likelihood number.
    @test maximum(abs, Λem - Diagonal(diag(Λem))) < 0.05
end
