# test/test_profile_infinite_bound.jl
# Issue #631: a profile confidence interval must never hand a user an infinite
# bound that came from a FAILED endpoint search.
#
# The reported symptom was size-dependent: at 200 tips `confint(..., method =
# "profile")` returned finite bounds, and at 600 / 1000 tips it returned
# `-Inf` / `Inf` in a few seconds. The size threshold was real but incidental --
# `G > 500` is where `drm(...)` auto-dispatches to the sparse LSS engine
# (`gaussian_lss.jl:402`), and that engine's objective was the one that was not
# thread-safe. `confint(...; method = :profile, threads = true)` runs the lower
# and upper endpoint arms concurrently, both arms wrote the same `Core.Box`-ed
# `â` (gaussian_sparse_lss.jl), and the corrupted objective starved the nuisance
# solve of convergence, which the endpoint search reports as a failed arm --
# i.e. as +/-Inf.
#
# Two independent guarantees are tested here, and the FIRST must hold even if
# the second is ever broken again:
#   1. finite-or-raise: `confint` never returns a non-finite bound for a failed
#      endpoint. (`profile_result` keeps the auditable +/-Inf convention.)
#   2. the sparse objective is thread-safe, so the threaded profile agrees with
#      the serial profile and with an independent TMB oracle.

using Test
using DRModels
using StableRNGs
using LinearAlgebra
using Statistics
using Random

# Kingman coalescent newick with `n` ultrametric tips (mirrors `ape::rcoal`,
# the shape the reporter used).
function _p631_coal_newick(n::Int, rng)
    nodes = ["t$(i)" for i in 1:n]
    heights = zeros(n)
    h = 0.0
    while length(nodes) > 1
        k = length(nodes)
        h += randexp(rng) / (k * (k - 1) / 2)
        i = rand(rng, 1:k); j = rand(rng, 1:k-1); j >= i && (j += 1)
        a, b = min(i, j), max(i, j)
        s = "($(nodes[a]):$(h - heights[a]),$(nodes[b]):$(h - heights[b]))"
        deleteat!(nodes, [a, b]); deleteat!(heights, [a, b])
        push!(nodes, s); push!(heights, h)
    end
    return nodes[1] * ";"
end

# Location-scale-scale DGP: quadratic mean term, log-linear residual sigma, and
# a log-linear phylogenetic SD submodel -- the reporter's model shape.
function _p631_simulate(n::Int; seed = 42)
    rng = StableRNG(seed)
    phy = DRModels.augmented_phy(_p631_coal_newick(n, rng))
    G = phy.n_leaves
    K0 = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
    dK = sqrt.(diag(K0)); K = K0 ./ (dK * dK')
    sp = String.(phy.leaf_names)
    t = 2 .* rand(rng, G) .- 1
    t = (t .- mean(t)) ./ std(t)
    sda = exp.(log(0.6) .+ 0.40 .* t)
    sde = exp.(log(0.5) .- 0.25 .* t)
    z = cholesky(Symmetric(K + 1e-8I)).L * randn(rng, G)
    y = 1.0 .+ 0.5 .* t .- 0.30 .* t .^ 2 .+ sda .* z .+ sde .* randn(rng, G)
    return phy, (; y = y, temp_z = t, temp_z2 = t .^ 2, species = sp)
end

_p631_formula() = bf(@formula(y ~ temp_z + temp_z2 + phylo(1 | species)),
                     @formula(sigma ~ temp_z),
                     @formula(sd(species, phylogenetic) ~ temp_z))

@testset "#631 profile CI never returns an infinite bound from a failed endpoint" begin
    # ---- the broken side of the reported break (sparse route, G > 500) -------
    phy, dat = _p631_simulate(600)
    fit = drm(_p631_formula(), Gaussian(); data = dat, tree = phy)

    # THE INVARIANT. Whatever the numerics do, the user-facing routine either
    # answers with finite bounds or refuses; it must not hand back +/-Inf.
    for threads in (false, true)
        local rows
        raised = false
        try
            rows = confint(fit; method = :profile, parm = :mu => "temp_z2", threads = threads)
        catch err
            raised = true
            @test err isa ArgumentError
            @test occursin("did not converge", sprint(showerror, err))
        end
        if !raised
            @test all(r -> isfinite(r.lower) && isfinite(r.upper), rows)
        end
    end

    # ---- the sparse objective is thread-safe (the actual defect) -------------
    if Threads.nthreads() > 1
        θ = fit.theta; np = length(θ)
        θa = copy(θ); θa[1] += 0.004
        θb = copy(θ); θb[2] += 0.004
        va = fit.nll(θa); vb = fit.nll(θb)
        ga = zeros(np); gb = zeros(np)
        fit.nllgrad(ga, θa); fit.nllgrad(gb, θb)
        for _ in 1:50
            ta = Threads.@spawn (o = zeros(np); fit.nllgrad(o, θa); (fit.nll(θa), o))
            tb = Threads.@spawn (o = zeros(np); fit.nllgrad(o, θb); (fit.nll(θb), o))
            xa = fetch(ta); xb = fetch(tb)
            # Bit-for-bit: the serial and concurrent evaluations run identical code.
            @test xa[1] == va
            @test xb[1] == vb
            @test xa[2] == ga
            @test xb[2] == gb
        end
    end

    # ---- the threaded profile now succeeds and matches the serial one -------
    serial = profile_result(fit; parm = :mu => "temp_z2", threads = false)
    @test serial.failed == 0
    srow = only(serial.ci)
    @test isfinite(srow.lower) && isfinite(srow.upper)
    if Threads.nthreads() > 1
        for _ in 1:3
            threaded = profile_result(fit; parm = :mu => "temp_z2", threads = true)
            @test threaded.failed == 0
            trow = only(threaded.ci)
            @test trow.lower == srow.lower
            @test trow.upper == srow.upper
        end
    end

    # ---- cross-engine oracle: drmTMB v0.7.0, engine = "tmb", tmbprofile ------
    # Same tree and data exported to R; `confint(fit, parm = "fixef:mu:temp_z2",
    # method = "profile")` on the TMB engine (67.2 s) returned:
    #     lower = -0.477129166981   upper = -0.242574309249
    # DRModels.jl agrees to 2.8e-6 (lower) and 2.0e-6 (upper); the residual is
    # `tmbprofile`'s grid-and-spline resolution, not a disagreement about the
    # interval. The tolerance below is deliberately looser than that measured
    # gap, and far tighter than any difference that would matter to a user.
    @test isapprox(srow.lower, -0.477129166981; atol = 1e-4)
    @test isapprox(srow.upper, -0.242574309249; atol = 1e-4)

    # ---- the working side of the reported break is unchanged ----------------
    # 200 tips routes to the DENSE engine (G <= 500) and never had the defect.
    phy200, dat200 = _p631_simulate(200)
    fit200 = drm(_p631_formula(), Gaussian(); data = dat200, tree = phy200)
    rows200 = confint(fit200; method = :profile, parm = :mu => "temp_z2")
    r200 = only(rows200)
    @test isfinite(r200.lower) && isfinite(r200.upper)
    @test isapprox(r200.lower, -0.4013017; atol = 1e-5)
    @test isapprox(r200.upper, -0.1355998; atol = 1e-5)
end

@testset "#631 the R bridge cannot present an infinite bound as a profile" begin
    # The bridge flattener is the last thing between a failed endpoint and R's
    # `confint()` data frame. It must refuse a non-finite bound outright.
    failed_row = (param = :mu, coef = "x", estimate = 0.5, lower = -Inf, upper = 1.0)
    @test_throws ArgumentError DRModels._bridge_inference_flatten(
        failed_row; method = "profile", status = "profile_failed", attempted = 1,
        used = 1, failed = 1, elapsed = 0.1, threaded = false, worker_threads = 1,
        julia_threads = 1, blas_threads = 1, message = "profile endpoint solve failed: lower")

    # A genuinely UNBOUNDED profile (the likelihood never crosses the LR
    # threshold in the searched range) is a different, honest answer and keeps
    # the "profile" status -- it must still pass through.
    ok = DRModels._bridge_inference_flatten(
        failed_row; method = "profile", status = "profile", attempted = 1,
        used = 1, failed = 0, elapsed = 0.1, threaded = false, worker_threads = 1,
        julia_threads = 1, blas_threads = 1,
        message = "profile did not cross threshold within searched range")
    @test ok["status"] == "profile"
    @test ok["lower"] == -Inf
end
