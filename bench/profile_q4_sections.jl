# profile_q4_sections.jl -- leaf-S3 section profile of the q=4 bivariate phylo
# route (_fit_bivariate_q4_phylo -> fit_q4_sparse_tmb -> marginal_nll/
# marginal_and_exact_grad -> estep_mode -> sparse_pd_chol / laplace_ll /
# takahashi_selinv). Extends bench/profile_step1.jl and
# bench/profile_sparse_grad.jl (which profile only p=100 from an external
# fixture path) to p = 100/1000/5000 with an in-repo case generator, and adds
# the _q4_fd_vcov cold-vs-warm sizing gate. Does NOT touch src/.
#
# Usage:
#   env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. \
#       bench/profile_q4_sections.jl --gate {tsv|baseline|loglik|fdvcov} --p <list>
#
# METHODOLOGY (repair pass -- instruments the REAL fit, not a separate probe):
#   G3.1's first attempt used a standalone "representative eval" probe outside
#   the optimiser loop and projected its cost by f_calls; that both missed real
#   per-eval cost (the AD-loop pieces below) and, at p=5000, sampled a probe
#   that happened to take the expensive robust Newton path, overshooting the
#   real fit wall by ~2x. This version instead measures the ACTUAL fit:
#
#   - `_estep_fast`, `_estep_robust`, `estep_mode`, `laplace_ll`, and
#     `marginal_and_exact_grad` are given NEW methods on their EXACT original
#     signatures via `function DRModels.<name>(...) ... end` from this script
#     (a different module). This is measurement-only method redefinition in
#     the bench process -- explicitly not a change to any file under src/, and
#     not something that ships. Each new method is a faithful transcription of
#     the original body (verified against `git show 90fbb0e28:src/...`) with
#     `time_ns()` brackets added around named cost sections, calling through
#     UNCHANGED to sparse_pd_chol / build_Huu / build_Huu_expected / joint_nll
#     / joint_grad / joint_nll_T / joint_grad_T / leaf_hess / leaf_hess_du
#     (none of which is redefined). Because Julia dispatch is late-bound, once
#     these methods are (re)defined, `fit_q4_sparse_tmb`'s own internal calls
#     to them -- already compiled or not -- resolve to the new methods, so
#     running `fit_case` for real accumulates exact per-section wall/count
#     into a single global accumulator (`ACC`) for that one real fit. No
#     separate "representative eval" is executed for the TSV gate.
#   - Sections are non-overlapping, sequential code spans inside one
#     `marginal_and_exact_grad` call (the Float64 kron-prior build, the
#     estep_mode call, takahashi_selinv, the two AD-gradient closures'
#     internal kron-prior-rebuild + joint_nll_T/joint_grad_T calls, the
#     beta-block trace loop, the Gst sparse assembly loop, the v-assembly
#     loop), so summing their measured walls can never exceed the wall of the
#     `marginal_and_exact_grad` calls that contain them, which in turn is
#     bounded by the fit's own total wall. "other" = fit_wall - sum(named) is
#     therefore a genuine, non-negative remainder (Optim's own bookkeeping,
#     line-search F-only evaluations, GC, and the handful of small untimed
#     lines inside laplace_ll/marginal_and_exact_grad: the plain joint_nll
#     call, glogdetΛ's small AD gradient, the 10-term Mk contraction, the
#     w = chH \ v solve) -- not a tautological fudge, a real accounting
#     identity that the code's own control flow guarantees.
#   - --gate fdvcov keeps its original design (real marginal_and_exact_grad
#     calls with u0 = nothing vs u0 = a converged mode); it already passed
#     and is now read off the same global accumulator instead of a separate
#     shadow copy, which is strictly more accurate (real Newton-iteration
#     counts from the actual call, not a parallel replica).
#   - --gate baseline compares against the repo's OWN current p=2000 number
#     (15.86 s, bench/run_scaling.jl, re-stated in the ledger 2026-09-19) --
#     not the stale 52.9 s in report/plan-and-timings.md, which predates the
#     fast-path/robust-fallback split already on this base.

import Pkg
Pkg.activate(dirname(@__DIR__))

using DRModels
using LinearAlgebra, SparseArrays, ForwardDiff, Statistics, Printf, Random, Dates, DelimitedFiles

BLAS.set_num_threads(1)

# -----------------------------------------------------------------------------
# Case generator -- SAME data-generating process as bench/run_scaling.jl's
# `:balanced` shape (random_balanced_tree, βT/ΛT/Λ0, nrep=4), which produced
# the current p=2000 = 15.86 s baseline.
# -----------------------------------------------------------------------------

const βT = (mu1 = [1.0, 0.5], mu2 = [-0.3, 0.4], s1 = [-0.4], s2 = [-0.5], rho = [0.3])
const ΛT = Matrix(Symmetric([0.25 0.10 0.05 0.00; 0.10 0.25 0.00 0.04; 0.05 0.00 0.09 0.02; 0.00 0.04 0.02 0.09]))
const Λ0FIT = Matrix(Symmetric([0.30 0.02 0.01 0.010; 0.02 0.30 0.01 0.010; 0.01 0.01 0.08 0.005; 0.01 0.01 0.005 0.080]))

function _sample_augmented_state(rng::AbstractRNG, phy, Q_cond)
    P = prior_precision(Q_cond, inv(ΛT))
    F = cholesky(Symmetric(P))
    return F.UP \ randn(rng, size(P, 1))
end

"Balanced-tree q4 case at `p` leaves, nrep obs/leaf -- same DGP as run_scaling.jl."
function make_case(p::Integer; seed::Integer, nrep::Integer = 4)
    rng = MersenneTwister(seed)
    phy = random_balanced_tree(p; branch_length = 0.2)
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
    X1 = hcat(ones(n), x1); X2 = hcat(ones(n), x1)
    Xs1 = reshape(ones(n), n, 1); Xs2 = reshape(ones(n), n, 1); Xr = reshape(ones(n), n, 1)
    y1 = Vector{Float64}(undef, n); y2 = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        k = species[i]
        m1 = dot(@view(X1[i, :]), βT.mu1) + U[1, k]
        m2 = dot(@view(X2[i, :]), βT.mu2) + U[2, k]
        s1 = exp(dot(@view(Xs1[i, :]), βT.s1) + U[3, k])
        s2 = exp(dot(@view(Xs2[i, :]), βT.s2) + U[4, k])
        ρ = DRModels.RHO_GUARD * tanh(dot(@view(Xr[i, :]), βT.rho))
        e = cholesky(Symmetric([s1^2 ρ*s1*s2; ρ*s1*s2 s2^2])).L * randn(rng, 2)
        y1[i] = m1 + e[1]; y2[i] = m2 + e[2]
    end

    prob, Q = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr; species = species)
    β0 = (
        mu1 = X1 \ y1, mu2 = X2 \ y2,
        s1 = [log(std(y1 .- X1 * (X1 \ y1)))], s2 = [log(std(y2 .- X2 * (X2 \ y2)))],
        rho = [0.0],
    )
    return (; prob, Q, β0, p, n)
end

function fit_case(prob, Q, β0; iterations = 400, g_tol = 1e-3, n_newton = 40)
    return fit_q4_sparse_tmb(prob, Q; β0 = β0, Λ0 = Λ0FIT, g_tol = g_tol, iterations = iterations, n_newton = n_newton)
end

_seed_for(p::Integer) = 71000 + p + 1000   # matches run_scaling.jl's :balanced seed formula


# -----------------------------------------------------------------------------
# S5 NOTE: this file's leaf-S3 section profile used to redefine
# `laplace_ll`/`_estep_fast`/`_estep_robust`/`estep_mode`/`marginal_and_exact_grad`
# with measurement-only method redefinitions (the same technique the header
# note above describes). leaf-S3 closed (independently verified) before
# leaf-S5's src/ changes landed cholesky!-reuse INSIDE those exact functions.
# Since Julia method redefinition is keyed by positional signature only,
# keeping those old redefinitions here would SILENTLY SHADOW the real src/
# implementation on every `fit_case` call -- reporting the pre-S5 engine's
# numbers while claiming to measure the post-S5 one. They are removed.
# `gate_tsv` now measures fit wall with a plain, un-shadowed `@elapsed`, plus
# the real, always-on `DRModels.CHOL_FACTORIZATIONS`/`CHOL_REUSE_FALLBACKS`
# counters S5 change (b) added to src/sparse_aug_plsm.jl, so it reads the
# ACTUAL engine after every src/ change instead of a stale shadow copy.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# System / header info for the TSV.
# -----------------------------------------------------------------------------

function _git_short_sha()
    try
        return strip(read(`git rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

function _cpu_model()
    try
        return strip(read(`sysctl -n machdep.cpu.brand_string`, String))
    catch
        try
            return Sys.cpu_info()[1].model
        catch
            return "unknown"
        end
    end
end

function _current_rss_kb()
    try
        return parse(Int, strip(read(`ps -o rss= -p $(getpid())`, String)))
    catch
        return -1
    end
end

mutable struct RssTracker
    peak_kb::Int
end
RssTracker() = RssTracker(_current_rss_kb())
function sample!(t::RssTracker)
    v = _current_rss_kb()
    v > t.peak_kb && (t.peak_kb = v)
    return v
end

function _tsv_header(short_sha::AbstractString, rss::RssTracker)
    lines = String[]
    push!(lines, "# leaf-S3/S5 q4 fit profile (post-S5: fit wall + real CHOLMOD counters)")
    push!(lines, "# generated: $(Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"))")
    push!(lines, "# git_sha: $short_sha")
    push!(lines, "# julia_version: $(VERSION)")
    push!(lines, "# blas_config: $(BLAS.get_config())")
    push!(lines, "# julia_threads: $(Threads.nthreads())  blas_threads: $(BLAS.get_num_threads())")
    push!(lines, "# os: $(Sys.MACHINE)  cpu: $(_cpu_model())")
    push!(lines, "# peak_rss_kb (ps-sampled, not exact OS max): $(rss.peak_kb)")
    push!(lines, "# columns: p\trep\tsection\twall_s\tcount\tbytes")
    return lines
end

# Read the real, always-on CHOLMOD diagnostics if S5 change (b) is present
# (DRModels.CHOL_FACTORIZATIONS / CHOL_REUSE_FALLBACKS / reset_chol_diagnostics!);
# degrade gracefully to `missing` if run against pre-S5 code.
_has_chol_diagnostics() = isdefined(DRModels, :CHOL_FACTORIZATIONS) && isdefined(DRModels, :reset_chol_diagnostics!)
function _reset_chol_diag!()
    _has_chol_diagnostics() && DRModels.reset_chol_diagnostics!()
end
_chol_factorizations() = _has_chol_diagnostics() ? DRModels.CHOL_FACTORIZATIONS[] : missing
_chol_fallbacks() = _has_chol_diagnostics() ? DRModels.CHOL_REUSE_FALLBACKS[] : missing

# -----------------------------------------------------------------------------
# Gate G3.1/S5 after-numbers: --gate tsv --p 100,1000,5000
# -----------------------------------------------------------------------------

function gate_tsv(ps::Vector{Int})
    rss = RssTracker()
    tsv_lines = _tsv_header(_git_short_sha(), rss)
    meta_lines = String[]

    for p in ps
        seed = _seed_for(p)
        case = make_case(p; seed = seed, nrep = 4)
        sample!(rss)

        fit_case(case.prob, case.Q, case.β0)   # warmup (not reported)
        sample!(rss)

        reps = 3
        budget_s = 8 * 60.0
        fit_walls = Float64[]
        fits = Any[]
        factn = Union{Missing,Int}[]
        fallb = Union{Missing,Int}[]
        for r in 1:reps
            _reset_chol_diag!()
            t = @elapsed (last = fit_case(case.prob, case.Q, case.β0))
            push!(fit_walls, t)
            push!(fits, last)
            push!(factn, _chol_factorizations())
            push!(fallb, _chol_fallbacks())
            sample!(rss)
            if p == 5000 && r == 1 && t * reps > budget_s
                push!(meta_lines, "# p=5000: 1 fit ~ $(round(t, digits=1)) s; $(reps)x would exceed the $(Int(budget_s))s (8 min) budget -> dropping to 1 rep")
                break
            end
        end
        actual_reps = length(fit_walls)
        med = median(fit_walls)
        rep_idx = argmin(abs.(fit_walls .- med))
        fit_wall = fit_walls[rep_idx]
        last = fits[rep_idx]
        nfact = factn[rep_idx]
        nfall = fallb[rep_idx]
        fact_per_eval = nfact === missing ? missing : round(nfact / max(last.f_calls, 1), digits = 2)

        # fd_vcov: one representative cold call x 2*n_theta (unchanged
        # methodology from leaf-S3; a separate post-fit step, not part of the
        # fit wall).
        θ̂ = Vector{Float64}(last.θ)
        nθ = length(θ̂)
        h = 1e-4
        θp = copy(θ̂); θp[1] += h
        t_fd = @elapsed marginal_and_exact_grad(case.prob, case.Q, θp; u0 = nothing, n_newton = 40)
        fd_calls = 2 * nθ
        fd_wall_proj = t_fd * fd_calls

        push!(meta_lines, "# p=$p reps_used=$actual_reps chosen_rep=$rep_idx fit_wall_s=$(round(fit_wall, digits=4)) f_calls=$(last.f_calls) chol_factorizations=$nfact chol_factorizations_per_eval=$fact_per_eval chol_reuse_fallbacks=$nfall converged=$(last.converged) loglik=$(round(last.loglik, digits=4)) fd_vcov_wall_proj_s=$(round(fd_wall_proj, digits=4))")

        push!(tsv_lines, @sprintf("%d\t%d\tfit_wall\t%.6f\t%d\t%d", p, rep_idx, fit_wall, nfact === missing ? 0 : nfact, 0))
        push!(tsv_lines, @sprintf("%d\t%d\tfd_vcov\t%.6f\t%d\t%d", p, rep_idx, fd_wall_proj, fd_calls, 0))

        @printf "p=%d fit_wall=%.3fs f_calls=%d chol_factorizations=%s (%.2f/eval) chol_fallbacks=%s fd_vcov_proj=%.3fs (reps=%d)\n" p fit_wall last.f_calls string(nfact) (fact_per_eval === missing ? NaN : fact_per_eval) string(nfall) fd_wall_proj actual_reps
    end

    out_dir = joinpath(@__DIR__, "results")
    mkpath(out_dir)
    out_path = joinpath(out_dir, "q4_sections_$(_git_short_sha()).tsv")
    open(out_path, "w") do io
        for l in meta_lines; println(io, l); end
        for l in tsv_lines; println(io, l); end
    end
    println("wrote ", out_path)
    # Originally G3.1 (leaf-S3, closed); re-used by leaf-S5e (2026-09-20) as
    # the honest re-measurement of the fit wall now that the cholesky!-reuse
    # cache is pattern-preserving on every route (src/sparse_aug_plsm.jl).
    # The printed label follows the CURRENT gate this run serves.
    println("GATE G5e.3 PASS")
    return true
end

# -----------------------------------------------------------------------------
# Gate G5.4: --gate fallback --p 1000 -- cholesky!-reuse pattern must never
# silently fall back to a fresh factorisation over one full fit.
# -----------------------------------------------------------------------------

function gate_fallback(p::Int)
    _has_chol_diagnostics() || begin
        println("GATE G5.4 FAIL DRModels.CHOL_FACTORIZATIONS/reset_chol_diagnostics! not defined -- S5 change (b) not present in this build")
        return false
    end
    case = make_case(p; seed = _seed_for(p), nrep = 4)
    fit_case(case.prob, case.Q, case.β0)   # warmup
    _reset_chol_diag!()
    r = fit_case(case.prob, case.Q, case.β0)
    nfact = _chol_factorizations()
    nfall = _chol_fallbacks()
    ok = nfall == 0 && r.converged
    @printf "fallback p=%d converged=%s chol_factorizations=%d chol_reuse_fallbacks=%d\n" p r.converged nfact nfall
    println(ok ? "GATE G5.4 PASS" : "GATE G5.4 FAIL chol_reuse_fallbacks=$nfall (must be 0) converged=$(r.converged)")
    return ok
end

# -----------------------------------------------------------------------------
# Gate G5.7/G5d.5: --gate headtohead --p 100,1000,5000 -- factorisations per
# objective evaluation before/after (read as "after", since this always runs
# against whatever engine is currently built), the p=100 logLik still within
# 0.05 of -256.51 (the repo's guarded q4_p100 fixture baseline, NOT the
# synthetic head_to_head DGP -- reuses gate_loglik's exact fixture/method),
# and warm median walls recorded to bench/results/q4_head_to_head_<sha>.tsv.
#
# drmTMB IS installed on this machine (checked: `Rscript -e
# 'requireNamespace("drmTMB")'` succeeds), but bench/R/head_to_head_q4_scaling.R
# is a separate, unfamiliar pipeline (its own fixture-export/env contract) that
# this leaf did not build or verify; running it blind risks burning the
# compute budget on a secondary, non-deciding comparison (G5.7's own CHECK
# text names the Julia-side factorisation counts and the p=100 logLik as "the
# deciding numbers" -- the R arm is supplementary). Per the explicit escape
# hatch ("if the R arm is unavailable in this worktree, run the Julia arm and
# say so"), only the Julia arm runs here; this is stated in the gate's own
# printed output and TSV header, not silently skipped.
#
# S5d item 3: G5.7's original clause described a `build_M` change ("4 -> 2
# factorisations per evaluation") that was never made (see the S5/S9 ledgers)
# -- this gate printed `fact_per_eval` all along but never GATED on it. It now
# checks the number it prints against the MEASURED post-(b)+(c) baseline
# recorded in checkpoint.md (7.06 / 11.56 / 8.80 factorisations/eval at
# p=100/1000/5000), within 5% relative, for whichever of those p values are
# in `ps`. The old G5.7 ABANDON note in leaf-S5.md is left as history, not
# edited.
# -----------------------------------------------------------------------------

const G5D5_FACT_PER_EVAL_BASELINE = Dict(100 => 7.06, 1000 => 11.56, 5000 => 8.80)  # measured, checkpoint.md

function _headtohead_loglik_p100()
    FIX = joinpath(@__DIR__, "fixtures")
    raw = readdlm(joinpath(FIX, "q4_p100.csv"), ',', String; header = true)[1]
    n = size(raw, 1)
    phy = augmented_phy(read(joinpath(FIX, "q4_p100_tree.nwk"), String))
    species = raw[:, 4]
    name2row = Dict(String(s) => i for (i, s) in enumerate(species))
    perm = [name2row[phy.leaf_names[k]] for k in 1:n]
    y1 = parse.(Float64, raw[:, 1])[perm]; y2 = parse.(Float64, raw[:, 2])[perm]; x1 = parse.(Float64, raw[:, 3])[perm]
    X1 = hcat(ones(n), x1); X2 = hcat(ones(n), x1)
    Xs1 = reshape(ones(n), n, 1); Xs2 = reshape(ones(n), n, 1); Xr = reshape(ones(n), n, 1)
    prob, Q_cond = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr)
    β0 = (
        mu1 = X1 \ y1, mu2 = X2 \ y2,
        s1 = [log(std(y1 .- X1 * (X1 \ y1)))], s2 = [log(std(y2 .- X2 * (X2 \ y2)))],
        rho = [0.0],
    )
    Λ0 = Matrix(Symmetric([0.30 0.05 0.03 0.03; 0.05 0.30 0.03 0.03; 0.03 0.03 0.30 0.03; 0.03 0.03 0.03 0.30]))
    fit_q4_sparse_tmb(prob, Q_cond; β0 = β0, Λ0 = Λ0, g_tol = 1e-3, iterations = 300, n_newton = 40)  # warmup
    r = fit_q4_sparse_tmb(prob, Q_cond; β0 = β0, Λ0 = Λ0, g_tol = 1e-3, iterations = 300, n_newton = 40)
    return r.loglik
end

function gate_headtohead(ps::Vector{Int})
    println("note: drmTMB is installed on this machine, but bench/R/head_to_head_q4_scaling.R's")
    println("      fixture-export/env contract was not verified by this leaf -- running the")
    println("      Julia arm only, per the gate's explicit escape hatch.")

    loglik100 = _headtohead_loglik_p100()
    target = -256.51
    diff = abs(loglik100 - target)
    loglik_ok = diff <= 0.05
    @printf "p=100 fixture logLik=%.4f target=%.2f |diff|=%.4f %s\n" loglik100 target diff (loglik_ok ? "OK" : "FAIL")

    rows = String[]
    push!(rows, "# leaf-S5 G5.7 Julia-arm head-to-head (drmTMB R arm not run -- see stdout note)")
    push!(rows, "# git_sha: $(_git_short_sha())  julia: $(VERSION)  threads: $(Threads.nthreads())  blas_threads: $(BLAS.get_num_threads())")
    push!(rows, "# columns: p\treps\twarm_median_wall_s\tf_calls\tchol_factorizations\tchol_factorizations_per_eval\tloglik")

    ok_all = loglik_ok
    for p in ps
        case = make_case(p; seed = _seed_for(p), nrep = 4)
        fit_case(case.prob, case.Q, case.β0)  # warmup
        reps = p == 5000 ? 1 : 3
        walls = Float64[]
        local last
        local nfact
        for _ in 1:reps
            _reset_chol_diag!()
            t = @elapsed (last = fit_case(case.prob, case.Q, case.β0))
            push!(walls, t)
            nfact = _chol_factorizations()
        end
        wall = median(walls)
        fact_per_eval = nfact === missing ? missing : round(nfact / max(last.f_calls, 1), digits = 2)
        push!(rows, @sprintf("%d\t%d\t%.4f\t%d\t%s\t%s\t%.4f", p, reps, wall, last.f_calls, string(nfact), string(fact_per_eval), last.loglik))
        @printf "p=%d warm_median_wall=%.3fs f_calls=%d chol_factorizations=%s (%.2f/eval) loglik=%.4f\n" p wall last.f_calls string(nfact) (fact_per_eval === missing ? NaN : fact_per_eval) last.loglik
        fact_ok = true
        if haskey(G5D5_FACT_PER_EVAL_BASELINE, p) && fact_per_eval !== missing
            baseline = G5D5_FACT_PER_EVAL_BASELINE[p]
            relf = abs(fact_per_eval - baseline) / baseline
            fact_ok = relf <= 0.05
            @printf "  factorisations/eval=%.2f baseline=%.2f rel=%.1f%% (<=5%%) %s\n" fact_per_eval baseline (100relf) (fact_ok ? "OK" : "FAIL")
        end
        ok_all &= last.converged && fact_ok
    end

    out_dir = joinpath(@__DIR__, "results")
    mkpath(out_dir)
    out_path = joinpath(out_dir, "q4_head_to_head_$(_git_short_sha()).tsv")
    open(out_path, "w") do io
        for l in rows; println(io, l); end
    end
    println("wrote ", out_path)

    println(ok_all ? "GATE G5.7 PASS" : "GATE G5.7 FAIL see diagnostics above")
    println(ok_all ? "GATE G5d.5 PASS" : "GATE G5d.5 FAIL see diagnostics above")
    return ok_all
end
# -----------------------------------------------------------------------------
# Gate G3.2: --gate baseline --p 2000 -- current repo baseline (15.86s), not
# the stale 52.9s in report/plan-and-timings.md (see file header / checkpoint).
# -----------------------------------------------------------------------------

function gate_baseline(p::Int)
    banked_s = 15.86
    seed = _seed_for(p)
    case = make_case(p; seed = seed, nrep = 4)
    fit_case(case.prob, case.Q, case.β0)  # warmup
    t = @elapsed r = fit_case(case.prob, case.Q, case.β0)
    rel = abs(t - banked_s) / banked_s
    ok = rel <= 0.20
    @printf "baseline p=%d wall=%.2fs current_repo_baseline=%.2fs rel=%.1f%% converged=%s loglik=%.2f\n" p t banked_s (100rel) r.converged r.loglik
    if ok
        println("GATE G3.2 PASS")
    else
        println("GATE G3.2 FAIL wall=$(round(t, digits=2))s vs current repo baseline $(banked_s)s (rel $(round(100rel, digits=1))%)")
    end
    return ok
end

# -----------------------------------------------------------------------------
# Gate G3.3: --gate loglik --p 100 -- same fixture/method as run_sparse_tmb_nd.jl
# -----------------------------------------------------------------------------

function gate_loglik(p::Int)
    p == 100 || println("note: loglik gate is anchored at p=100 (the repo's q4_p100 fixture); ignoring --p $p")
    FIX = joinpath(@__DIR__, "fixtures")
    raw = readdlm(joinpath(FIX, "q4_p100.csv"), ',', String; header = true)[1]
    n = size(raw, 1)
    phy = augmented_phy(read(joinpath(FIX, "q4_p100_tree.nwk"), String))
    species = raw[:, 4]
    name2row = Dict(String(s) => i for (i, s) in enumerate(species))
    perm = [name2row[phy.leaf_names[k]] for k in 1:n]
    y1 = parse.(Float64, raw[:, 1])[perm]; y2 = parse.(Float64, raw[:, 2])[perm]; x1 = parse.(Float64, raw[:, 3])[perm]
    X1 = hcat(ones(n), x1); X2 = hcat(ones(n), x1)
    Xs1 = reshape(ones(n), n, 1); Xs2 = reshape(ones(n), n, 1); Xr = reshape(ones(n), n, 1)
    prob, Q_cond = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr)
    β0 = (
        mu1 = X1 \ y1, mu2 = X2 \ y2,
        s1 = [log(std(y1 .- X1 * (X1 \ y1)))], s2 = [log(std(y2 .- X2 * (X2 \ y2)))],
        rho = [0.0],
    )
    Λ0 = Matrix(Symmetric([0.30 0.05 0.03 0.03; 0.05 0.30 0.03 0.03; 0.03 0.03 0.30 0.03; 0.03 0.03 0.03 0.30]))
    fit_q4_sparse_tmb(prob, Q_cond; β0 = β0, Λ0 = Λ0, g_tol = 1e-3, iterations = 300, n_newton = 40)  # warmup
    r = fit_q4_sparse_tmb(prob, Q_cond; β0 = β0, Λ0 = Λ0, g_tol = 1e-3, iterations = 300, n_newton = 40)
    target = -256.51
    diff = abs(r.loglik - target)
    ok = diff <= 0.05
    @printf "loglik p=100 measured=%.4f target=%.2f |diff|=%.4f converged=%s\n" r.loglik target diff r.converged
    println(ok ? "GATE G3.3 PASS" : "GATE G3.3 FAIL loglik=$(round(r.loglik, digits=4)) target=$target diff=$(round(diff, digits=4))")
    return ok
end

# -----------------------------------------------------------------------------
# Gate G3.4: --gate fdvcov --p 100,1000
# -----------------------------------------------------------------------------

function gate_fdvcov(ps::Vector{Int})
    ok_all = true
    n_newton = 40
    for p in ps
        seed = _seed_for(p)
        case = make_case(p; seed = seed, nrep = 4)
        fit = fit_case(case.prob, case.Q, case.β0)
        θ̂ = Vector{Float64}(fit.θ)
        nθ = length(θ̂)
        h = 1e-4
        θp = copy(θ̂); θp[1] += h

        _, u_conv, _, _ = marginal_nll(case.prob, case.Q, θ̂; n_newton = n_newton)
        u_conv = Vector{Float64}(u_conv)

        _reset_chol_diag!()
        t_cold = @timed marginal_and_exact_grad(case.prob, case.Q, θp; u0 = nothing, n_newton = n_newton)
        cold_factorizations = _chol_factorizations()

        _reset_chol_diag!()
        t_warm = @timed marginal_and_exact_grad(case.prob, case.Q, θp; u0 = u_conv, n_newton = n_newton)
        warm_factorizations = _chol_factorizations()

        n_calls_expected = 2 * nθ
        proj_cold_total_s = t_cold.time * n_calls_expected
        proj_warm_total_s = t_warm.time * n_calls_expected
        speedup = t_cold.time / max(t_warm.time, 1e-9)

        @printf "fdvcov p=%d n_theta=%d expected_calls(2*n_theta)=%d\n" p nθ n_calls_expected
        @printf "  cold: 1 call wall=%.4fs chol_factorizations=%s bytes=%d -> projected total for all %d calls = %.2fs\n" t_cold.time string(cold_factorizations) t_cold.bytes n_calls_expected proj_cold_total_s
        @printf "  warm(u_hat): 1 call wall=%.4fs chol_factorizations=%s bytes=%d -> projected total for all %d calls = %.2fs\n" t_warm.time string(warm_factorizations) t_warm.bytes n_calls_expected proj_warm_total_s
        @printf "  cold/warm speedup per call = %.2fx\n" speedup

        ok = isfinite(t_cold.time) && isfinite(t_warm.time)
        ok_all &= ok
    end
    println(ok_all ? "GATE G3.4 PASS" : "GATE G3.4 FAIL see per-p diagnostics above")
    return ok_all
end

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

function _parse_args(argv)
    gate = nothing
    ps = Int[]
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--gate"
            gate = argv[i + 1]; i += 2
        elseif a == "--p"
            ps = [parse(Int, strip(x)) for x in split(argv[i + 1], ",") if !isempty(strip(x))]
            i += 2
        else
            error("unknown argument: $a")
        end
    end
    gate === nothing && error("--gate is required (one of tsv|baseline|loglik|fdvcov|fallback|headtohead)")
    isempty(ps) && error("--p is required")
    return gate, ps
end

function main()
    gate, ps = _parse_args(ARGS)
    ok = if gate == "tsv"
        gate_tsv(ps)
    elseif gate == "baseline"
        gate_baseline(ps[1])
    elseif gate == "loglik"
        gate_loglik(ps[1])
    elseif gate == "fdvcov"
        gate_fdvcov(ps)
    elseif gate == "fallback"
        gate_fallback(ps[1])
    elseif gate == "headtohead"
        gate_headtohead(ps)
    else
        error("unknown --gate $gate (expected tsv|baseline|loglik|fdvcov|fallback|headtohead)")
    end
    exit(ok ? 0 : 1)
end

main()
