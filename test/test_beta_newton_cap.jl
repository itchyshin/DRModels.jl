# test_beta_newton_cap.jl — trust-radius cap on the conditional beta Newton
# step shared by `cond_newton_beta` (src/reml_q4.jl, the production bivariate
# q4-phylo REML path) and `mstep_beta` (src/sparse_em_fit.jl).
#
# FINDING (audit, no ticket yet): both functions take a Newton step on the
# profiled fixed effects from an exact ForwardDiff Hessian, with a ridge
# ladder for indefiniteness and up to 25 halvings accepted only on decrease —
# but neither capped the RAW step's size, so the FIRST trial point is
# evaluated wherever the raw step lands. On a near-collinear mean design
# (x2 = x1 + 1e-8·noise, cond(H) up to 5.6e16): the raw step reached
# ‖step‖∞ ≈ 3.25e5, a step of about 1.4e5 was accepted, and β for the
# collinear pair diverged to ±602. In drmTMB the same unguarded-step pattern
# made the density evaluation hang (fixed in drmTMB draft PR #1441 with a
# trust radius and backtracking).
#
# Fix: `DRModels.cap_newton_step(step, trust)` scales the WHOLE raw step so
# ‖step‖∞ ≤ trust, applied BEFORE backtracking, in both `cond_newton_beta`
# and `mstep_beta`. `trust = 5.0` matches the existing guarded pattern in
# `_estep_robust` (src/sparse_aug_plsm.jl): `sc = min(1, trust/‖step‖∞)`,
# `trust = 5.0`.
#
#   julia --project=. -e 'using DRModels, Test; include("test/test_beta_newton_cap.jl")'

module TestBetaNewtonCap

using DRModels
using Test
using LinearAlgebra
using Random
using ForwardDiff
using DelimitedFiles: readdlm

# ---------------------------------------------------------------------------
# (a) Ill-conditioned fixture: near-collinear mean design, reproducing the
#     audit's demo (scratchpad/newton-audit/demo_cond_newton_beta.jl) but
#     wrapped as a real AugProblem via make_problem_from_Q, so this exercises
#     the ACTUAL `cond_newton_beta` / `mstep_beta` code paths, not a copy.
# ---------------------------------------------------------------------------
function _illconditioned_problem()
    Random.seed!(1)
    n = 40
    x1 = randn(n)
    x2 = x1 .+ 1e-8 .* randn(n)          # near-collinear with x1
    X1 = hcat(ones(n), x1, x2)            # k1 = 3: [intercept, x1, x2]
    X2 = hcat(ones(n))                    # k2 = 1
    Xs1 = hcat(ones(n))                   # ks1 = 1
    Xs2 = hcat(ones(n))                   # ks2 = 1
    Xr  = hcat(ones(n))                   # rho intercept (held fixed, not profiled)
    y1 = 1.0 .+ 0.5 .* x1 .+ 0.3 .* randn(n)
    y2 = -0.3 .+ 0.4 .* randn(n)

    group = collect(1:n)                  # one level per row: G = n
    Q_cond = Matrix{Float64}(I, n, n)     # arbitrary PD; unused by cond_newton_beta/mstep_beta
    prob, _ = DRModels.make_problem_from_Q(Q_cond, y1, y2, X1, X2, Xs1, Xs2, Xr; group = group)

    beta0 = (mu1 = zeros(3), mu2 = zeros(1), s1 = zeros(1), s2 = zeros(1), rho = [0.0])
    u_hat = zeros(4 * n)                  # fixed latent state (u = 0 at every leaf)
    return prob, u_hat, beta0
end

# Instrumented copy of `cond_newton_beta`'s loop that records the inf-norm of
# EVERY trial step it actually evaluates (i.e. norm(step, Inf) for alpha = 1,
# the FIRST point tried each iteration, before backtracking can only shrink
# it further) — `cap = false` reproduces the pre-fix algorithm exactly;
# `cap = true` applies `DRModels.cap_newton_step` exactly where the fix does.
# This directly answers "was any trial point ever evaluated above the cap?"
# without needing to touch cond_newton_beta's own closure.
function _trace_newton_steps(prob, u_hat, beta_full; n_newton::Int = 10,
                             tol::Float64 = 1e-10, cap::Bool)
    k1 = size(prob.X1, 2); k2 = size(prob.X2, 2)
    ks1 = size(prob.Xs1, 2); ks2 = size(prob.Xs2, 2)
    o1 = 0; o2 = k1; o3 = k1 + k2; o4 = k1 + k2 + ks1
    etas_r = prob.Xr * beta_full.rho
    function f_beta(bv)
        bm1 = bv[o1+1:o1+k1];  bm2 = bv[o2+1:o2+k2]
        bs1 = bv[o3+1:o3+ks1]; bs2 = bv[o4+1:o4+ks2]
        eta1  = prob.X1  * bm1; eta2  = prob.X2  * bm2
        etas1 = prob.Xs1 * bs1; etas2 = prob.Xs2 * bs2
        tot = zero(eltype(bv))
        @inbounds for i in eachindex(prob.leaf_node)
            t = prob.leaf_node[i]; base = 4 * (t - 1)
            ublk = (u_hat[base+1], u_hat[base+2], u_hat[base+3], u_hat[base+4])
            tot += DRModels.leaf_nll(ublk, prob.y1[i], prob.y2[i],
                                     eta1[i], eta2[i], etas1[i], etas2[i], etas_r[i])
        end
        return tot
    end
    bv = vcat(beta_full.mu1, beta_full.mu2, beta_full.s1, beta_full.s2)
    trace = Float64[]
    for _ in 1:n_newton
        g = ForwardDiff.gradient(f_beta, bv)
        H = ForwardDiff.hessian(f_beta, bv)
        local step = g
        for lam in (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2)
            ch = cholesky(Symmetric(H + lam * I); check = false)
            if issuccess(ch); step = ch \ g; break; end
        end
        cap && (step = DRModels.cap_newton_step(step, DRModels.BETA_NEWTON_TRUST))
        push!(trace, maximum(abs, step))    # the FIRST trial point's step size
        f0 = f_beta(bv); alpha = 1.0; bvn = bv .- alpha .* step
        for _ in 1:25
            (f_beta(bvn) <= f0 || alpha < 1e-8) && break
            alpha *= 0.5; bvn = bv .- alpha .* step
        end
        bv = bvn
        norm(alpha .* step) < tol && break
    end
    return trace, (mu1 = bv[o1+1:o1+k1],  mu2 = bv[o2+1:o2+k2],
                   s1  = bv[o3+1:o3+ks1], s2  = bv[o4+1:o4+ks2])
end

@testset "beta Newton trust radius: ill-conditioned design" begin
    prob, u_hat, beta0 = _illconditioned_problem()

    # (a1) reproduce the audit finding: the UNCAPPED trace has at least one
    # trial step whose inf-norm is enormous, and the final β diverges.
    trace_uncapped, bv_uncapped = _trace_newton_steps(prob, u_hat, beta0; n_newton = 10, cap = false)
    @test maximum(trace_uncapped) > 1e4
    @test maximum(abs, bv_uncapped.mu1) > 50.0    # the audit's uncapped run reached ±602

    # (a2) the step function directly: cap_newton_step scales the worst
    # observed raw step down to exactly the trust radius, and an
    # already-small step is returned UNCHANGED (bit-for-bit).
    worst = trace_uncapped[argmax(trace_uncapped)]
    big_step = fill(worst, 3)
    capped = DRModels.cap_newton_step(big_step, DRModels.BETA_NEWTON_TRUST)
    @test maximum(abs, capped) ≈ DRModels.BETA_NEWTON_TRUST atol = 1e-10

    small_step = [0.1, -0.2, 0.05]
    @test DRModels.cap_newton_step(small_step, DRModels.BETA_NEWTON_TRUST) === small_step

    # (a3) with the SAME cap applied at every iteration (cap = true, exactly
    # what cond_newton_beta now does): NO trial point's step ever exceeds the
    # trust radius, and the final β stays bounded.
    trace_capped, bv_capped = _trace_newton_steps(prob, u_hat, beta0; n_newton = 10, cap = true)
    @test all(s -> s <= DRModels.BETA_NEWTON_TRUST + 1e-9, trace_capped)
    @test maximum(abs, bv_capped.mu1) < 50.0

    # (a4) integration: the ACTUAL (patched) `cond_newton_beta` reproduces the
    # instrumented (cap = true) trace EXACTLY — the same algorithm, not a
    # parallel copy — so (a3)'s "no trial point exceeds the trust radius"
    # guarantee transfers to the real function. The NOTE below is intentional:
    # the collinear (x1, x2) pair in this design is UNIDENTIFIED regardless of
    # the cap (only their sum is), so a per-axis magnitude bound would be
    # meaningless; what the cap actually guarantees is a PER-ITERATION bound
    # of `trust` on how far any accepted step can move β, hence the
    # `n_newton * trust` envelope used below (never the audit's ±602, at any
    # n_newton).
    fit1 = DRModels.cond_newton_beta(prob, u_hat, beta0; n_newton = 10)
    @test all(isfinite, fit1.mu1) && all(isfinite, fit1.mu2)
    @test all(isfinite, fit1.s1) && all(isfinite, fit1.s2)
    for ax in (:mu1, :mu2, :s1, :s2)
        @test maximum(abs, getfield(fit1, ax) .- getfield(bv_capped, ax)) < 1e-8
    end
    envelope = 10 * DRModels.BETA_NEWTON_TRUST + 1.0
    @test maximum(abs, fit1.mu1) < envelope
    @test maximum(abs, fit1.mu2) < envelope

    # (a5) same guard, same fixture, on `mstep_beta` (src/sparse_em_fit.jl) —
    # not reachable from `drm()` (#472) but shares the identical unguarded
    # Newton-step shape and is fixed identically. `mstep_beta` also profiles
    # rho (7 params, not 4), so it need not match `cond_newton_beta`'s trace
    # bit-for-bit; check the same envelope bound and finiteness instead.
    beta0_full = (mu1 = beta0.mu1, mu2 = beta0.mu2, s1 = beta0.s1, s2 = beta0.s2, rho = beta0.rho)
    fit2 = DRModels.mstep_beta(prob, u_hat, beta0_full; n_newton = 10)
    @test all(isfinite, fit2.mu1) && all(isfinite, fit2.mu2)
    @test maximum(abs, fit2.mu1) < envelope
    @test maximum(abs, fit2.mu2) < envelope
end

# ---------------------------------------------------------------------------
# (b) Well-conditioned fixture: the existing #575 parity fixture. The trust
#     radius must be a NO-OP here — the fitted β and REML logLik must be
#     unchanged to 1e-8 relative to an uncapped run.
# ---------------------------------------------------------------------------
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

function _wellconditioned_inputs()
    dat = _load_data(FIXTURE)
    tree = read(joinpath(FIXTURE, "tree.newick"), String)
    form = bf(mu1    = @formula(y1 ~ x + phylo(1 | species)),
              mu2    = @formula(y2 ~ x + phylo(1 | species)),
              sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
              sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
              rho12  = @formula(rho12 ~ 1))
    rhs = Dict(form.forms)
    fixed, marker = DRModels._bivariate_q4_marker(rhs)
    grp = marker[2]
    phy = DRModels._as_augmented_phy(tree)

    y1, X1, _ = DRModels._design(form.response1, fixed[:mu1], dat)
    y2, X2, _ = DRModels._design(form.response2, fixed[:mu2], dat)
    _, Xs1, _ = DRModels._design(form.response1, fixed[:sigma1], dat)
    _, Xs2, _ = DRModels._design(form.response1, fixed[:sigma2], dat)
    _, Xr, _  = DRModels._design(form.response1, fixed[:rho12], dat)

    obs1 = DRModels._observed_response_mask(y1)
    obs2 = DRModels._observed_response_mask(y2)
    species = DRModels._phylo_species_index(phy, getproperty(dat, grp))
    prob, Q_cond = DRModels.make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr; species = species)

    β1 = X1[obs1, :] \ y1[obs1]
    β2 = X2[obs2, :] \ y2[obs2]
    res1 = y1[obs1] .- X1[obs1, :] * β1
    res2 = y2[obs2] .- X2[obs2, :] * β2
    beta0 = (mu1 = β1, mu2 = β2,
             s1 = DRModels._initial_scale_beta(Xs1, res1),
             s2 = DRModels._initial_scale_beta(Xs2, res2),
             rho = zeros(size(Xr, 2)))
    return prob, Q_cond, beta0
end

# PRE-FIX reference: `cond_newton_beta` exactly as it read on main before this
# fix (src/reml_q4.jl:109-151, no trust-radius cap) — copied here so the
# well-conditioned regression check has something uncapped to compare
# against, without needing to check out a second branch.
function _cond_newton_beta_reference(prob, u_hat::Vector{Float64}, beta_full;
                                     n_newton::Int = 20, tol::Float64 = 1e-10)
    k1 = size(prob.X1, 2); k2 = size(prob.X2, 2)
    ks1 = size(prob.Xs1, 2); ks2 = size(prob.Xs2, 2)
    o1 = 0; o2 = k1; o3 = k1 + k2; o4 = k1 + k2 + ks1
    etas_r = prob.Xr * beta_full.rho
    function f_beta(bv)
        bm1 = bv[o1+1:o1+k1];  bm2 = bv[o2+1:o2+k2]
        bs1 = bv[o3+1:o3+ks1]; bs2 = bv[o4+1:o4+ks2]
        eta1  = prob.X1  * bm1; eta2  = prob.X2  * bm2
        etas1 = prob.Xs1 * bs1; etas2 = prob.Xs2 * bs2
        tot = zero(eltype(bv))
        @inbounds for i in eachindex(prob.leaf_node)
            t = prob.leaf_node[i]; base = 4 * (t - 1)
            ublk = (u_hat[base+1], u_hat[base+2], u_hat[base+3], u_hat[base+4])
            tot += DRModels.leaf_nll(ublk, prob.y1[i], prob.y2[i],
                                     eta1[i], eta2[i], etas1[i], etas2[i], etas_r[i])
        end
        return tot
    end
    bv = vcat(beta_full.mu1, beta_full.mu2, beta_full.s1, beta_full.s2)
    for _ in 1:n_newton
        g = ForwardDiff.gradient(f_beta, bv)
        H = ForwardDiff.hessian(f_beta, bv)
        local step = g
        for lam in (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2)
            ch = cholesky(Symmetric(H + lam * I); check = false)
            if issuccess(ch); step = ch \ g; break; end
        end
        # NO trust-radius cap here — this is the pre-fix algorithm.
        f0 = f_beta(bv); alpha = 1.0; bvn = bv .- alpha .* step
        for _ in 1:25
            (f_beta(bvn) <= f0 || alpha < 1e-8) && break
            alpha *= 0.5; bvn = bv .- alpha .* step
        end
        bv = bvn
        norm(alpha .* step) < tol && break
    end
    return (mu1 = bv[o1+1:o1+k1],  mu2 = bv[o2+1:o2+k2],
            s1  = bv[o3+1:o3+ks1], s2  = bv[o4+1:o4+ks2])
end

@testset "beta Newton trust radius: well-conditioned fixture is a no-op" begin
    prob, Q_cond, beta0 = _wellconditioned_inputs()

    Λ0 = Matrix(0.3I, 4, 4)
    P = DRModels.prior_precision(Q_cond, inv(Λ0))
    beta_full = (mu1 = beta0.mu1, mu2 = beta0.mu2, s1 = beta0.s1, s2 = beta0.s2, rho = beta0.rho)
    u_hat, _, _ = DRModels.estep_mode(prob, P, beta_full)
    u_hat = Vector{Float64}(u_hat)

    fit_capped   = DRModels.cond_newton_beta(prob, u_hat, beta_full; n_newton = 20)   # trust = 5.0 (module default)
    fit_uncapped = _cond_newton_beta_reference(prob, u_hat, beta_full; n_newton = 20)

    for ax in (:mu1, :mu2, :s1, :s2)
        @test maximum(abs, getfield(fit_capped, ax) .- getfield(fit_uncapped, ax)) < 1e-8
    end

    # REML logLik unchanged too, at these two (should-be-identical) beta_hat.
    ll_capped   = DRModels.reml_nll_exact(prob, Q_cond,
                      DRModels.pack_phi(prob, beta_full.rho, Λ0);
                      beta0 = (mu1 = fit_capped.mu1, mu2 = fit_capped.mu2,
                               s1 = fit_capped.s1, s2 = fit_capped.s2, rho = beta_full.rho))
    ll_uncapped = DRModels.reml_nll_exact(prob, Q_cond,
                      DRModels.pack_phi(prob, beta_full.rho, Λ0);
                      beta0 = (mu1 = fit_uncapped.mu1, mu2 = fit_uncapped.mu2,
                               s1 = fit_uncapped.s1, s2 = fit_uncapped.s2, rho = beta_full.rho))
    @test abs(ll_capped - ll_uncapped) < 1e-8
end

end # module TestBetaNewtonCap
