#!/usr/bin/env julia
# tools/warm_timing.jl — S12 (#563), root gate G5.
#
# Native-Julia warm-workflow timing harness. Reads the fixtures written by
# tools/warm_timing_fixtures.jl and, for each registered workflow, fits it via
# DRModels.jl's own public front end (`drm(...)`), never through the R bridge --
# this is a pure Julia process, timed independently of tools/warm_timing.R.
#
# Registry: docs/dev-log/evidence/julia-r-parity/warm-workflow-registry.md
# (the workflow list, fixture sizes, and the drmTMB call each is paired
# against are defined there; this script and warm_timing.R must both match
# it, not redefine it).
#
# Threading:
#   - Threads.nthreads() is fixed at Julia STARTUP, not by a CLI flag here --
#     start Julia with `-t N` matching --threads N (documented in the
#     registry doc's usage section; this script errors if they disagree).
#   - BLAS threads ARE settable at runtime: `LinearAlgebra.BLAS.set_num_threads(N)`
#     -- OMP_NUM_THREADS alone does not reliably change Julia's active BLAS
#     thread count (see docs/dev-log root-gate note referenced in the scout).
#
# Timer floor (fixed 2026-09-02, see registry doc §1): a naive single-call
# `time_ns()` measurement is unreliable for legs that finish in microseconds
# (predict, uncertainty on small fixtures) -- system-clock/scheduler noise
# dominates. Each leg is instead timed as a LOOP: repeat the call until
# cumulative wall time >= 0.25s (minimum 1 call), then per-call = total/calls.
# `k` (`--reps`) such loops are run per leg; median_s/min_s are the
# median/min of the k per-call numbers, and `calls` (new TSV column) is the
# mean number of raw invocations per loop, rounded.
#
# Uncertainty leg (fixed 2026-09-02, see registry doc §1): `vcov(fit)` /
# `confint(fit)` just READ `fit.vcov`, computed ONCE eagerly inside `drm()`
# and cached in the DrmFit struct (`vcov(fit::DrmFit) = fit.vcov`,
# src/gaussian_core.jl) -- timing `confint(fit)` measures a cheap field read,
# not the actual covariance computation. Where the fitted objective closure
# is available (`fit.nll !== nothing`) and differentiable through ForwardDiff,
# this harness FORCES real recomputation: `ForwardDiff.hessian(fit.nll,
# fit.theta)` then `DRModels._vcov_from_hessian(...)` -- the exact steps the
# fitter itself ran to produce `fit.vcov` (verified byte-for-byte identical
# on 7/10 registered workflows, 2026-09-02). Two workflow families cannot be
# recomputed this way: the q4 bivariate phylogenetic route
# (`biv_q4_phylo_ml`/`_reml`) and the O(p) sparse LSS route
# (`large_sparse_lss_p2000`) build a Float64-only sparse Cholesky INSIDE
# `fit.nll` (SuiteSparse does not accept ForwardDiff.Dual), so
# `ForwardDiff.hessian(fit.nll, ...)` throws a TypeError/MethodError for
# those three. For those, `vcov` is genuinely inseparable from `fit()` itself
# (computed unconditionally, not as a skippable extra step) -- the `fit` leg
# ALREADY IS the fit+uncertainty combined cost, so no separate `uncertainty`
# row is written for them (not a zero/cached artifact -- an honest N/A;
# tools/warm_timing_compare.jl reports the missing row as such).
#
# Usage (run once per thread count, host idle, same Julia minor version both
# times against R, see registry doc "Usage"):
#   julia --project=. -t N tools/warm_timing.jl --threads N --reps 3 \
#       --fixtures docs/dev-log/evidence/julia-r-parity/warm-fixtures \
#       --out docs/dev-log/evidence/julia-r-parity/warm-timing-julia-tN.tsv
#   (add --workflow NAME to run a single workflow, e.g. for the smoke pre-run)

using DRModels
using ForwardDiff
using LinearAlgebra
using Printf

# ---- CLI ----------------------------------------------------------------

function parse_args(argv)
    threads = nothing
    reps = 3
    out = nothing
    fixtures = "docs/dev-log/evidence/julia-r-parity/warm-fixtures"
    workflow = nothing
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--threads"
            threads = parse(Int, argv[i + 1]); i += 2
        elseif a == "--reps"
            reps = parse(Int, argv[i + 1]); i += 2
        elseif a == "--out"
            out = argv[i + 1]; i += 2
        elseif a == "--fixtures"
            fixtures = argv[i + 1]; i += 2
        elseif a == "--workflow"
            workflow = argv[i + 1]; i += 2
        else
            error("warm_timing.jl: unknown arg $a")
        end
    end
    threads === nothing && error("warm_timing.jl: --threads N is required")
    out === nothing && error("warm_timing.jl: --out FILE is required")
    return (; threads, reps, out, fixtures, workflow)
end

# ---- CSV / Newick readers (manual -- main Project.toml has no CSV.jl) -----

function read_csv(path)
    lines = readlines(path)
    header = split(lines[1], ",")
    ncol = length(header)
    cols = [String[] for _ in 1:ncol]
    for l in lines[2:end]
        isempty(l) && continue
        parts = split(l, ",")
        for k in 1:ncol
            push!(cols[k], parts[k])
        end
    end
    return Dict(String(h) => c for (h, c) in zip(header, cols))
end

num(v) = parse.(Float64, v)
ints(v) = parse.(Int, v)

read_tree(path) = DRModels.augmented_phy(read(path, String))

# ---- per-workflow definitions ---------------------------------------------
# Each entry: fit() -> DrmFit, uncertainty (Function or `nothing` if the
# route cannot be forced -- see the header note), predict_leg(fit) -> preds.

# Forces a genuine covariance recomputation from the stored objective closure
# (NOT a cached-field read -- see header note). `nothing` if this workflow's
# route cannot be recomputed this way (checked once at registration below,
# not silently retried per call).
force_vcov(fit) = DRModels._vcov_from_hessian(ForwardDiff.hessian(fit.nll, fit.theta))

# True iff `force_vcov` actually works for a representative fit from this
# route (probed once when building the registry, not per timed call).
function vcov_recomputable(fit)
    fit.nll === nothing && return false
    try
        force_vcov(fit)
        return true
    catch
        return false
    end
end

struct Workflow
    name::String
    fit::Function
    uncertainty::Union{Function,Nothing}   # nothing => N/A, see header note
    predict_leg::Function
end

function build_workflows(fixdir)
    W = Workflow[]

    # 1. Gaussian mixed + phylo mean
    let d = read_csv(joinpath(fixdir, "gauss_mixed_phylo_mean.csv")),
        phy = read_tree(joinpath(fixdir, "gauss_mixed_phylo_mean.nwk"))
        dat = (y = num(d["y"]), x = num(d["x"]), species = d["species"], study = ints(d["study"]))
        f = bf(@formula(y ~ x + (1 | study) + phylo(1 | species)), @formula(sigma ~ 1))
        fitfn = () -> drm(f, Gaussian(); data = dat, tree = phy)
        push!(W, Workflow("gauss_mixed_phylo_mean", fitfn,
            vcov_recomputable(fitfn()) ? force_vcov : nothing,
            fit -> predict(fit, dat)))
    end

    # 2. Gaussian LSS sd(g) -- sd() predictor (zg) is GROUP-level (constant
    # within g); z (observation-level) drives sigma only.
    let d = read_csv(joinpath(fixdir, "gauss_lss_sd_group.csv"))
        dat = (y = num(d["y"]), x = num(d["x"]), z = num(d["z"]), zg = num(d["zg"]), g = ints(d["g"]))
        f = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ z), @formula(sd(g) ~ zg))
        fitfn = () -> drm(f, Gaussian(); data = dat)
        push!(W, Workflow("gauss_lss_sd_group", fitfn,
            vcov_recomputable(fitfn()) ? force_vcov : nothing,
            fit -> predict(fit, dat)))
    end

    # 3. Gaussian LSS sd_phylo -- sd() predictor (xs) is SPECIES-level;
    # x (observation-level) drives the mean/sigma.
    let d = read_csv(joinpath(fixdir, "gauss_lss_sd_phylo.csv")),
        phy = read_tree(joinpath(fixdir, "gauss_lss_sd_phylo.nwk"))
        dat = (y = num(d["y"]), x = num(d["x"]), xs = num(d["xs"]), species = d["species"])
        f = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x),
               @formula(sd(species, phylogenetic) ~ xs))
        fitfn = () -> drm(f, Gaussian(); data = dat, tree = phy)
        push!(W, Workflow("gauss_lss_sd_phylo", fitfn,
            vcov_recomputable(fitfn()) ? force_vcov : nothing,
            fit -> predict(fit, dat)))
    end

    # 4/5. Bivariate q4 phylo, ML and REML (same fixture). vcov is NOT
    # recomputable via ForwardDiff.hessian(fit.nll, ...) for this route (the
    # sparse q4 Cholesky is Float64-only) -- uncertainty is N/A for both, see
    # header note; `fit` already includes the (inseparable) vcov cost.
    let d = read_csv(joinpath(fixdir, "biv_q4_phylo.csv")),
        phy = read_tree(joinpath(fixdir, "biv_q4_phylo.nwk"))
        dat = (species = d["species"], x = num(d["x"]), y1 = num(d["y1"]), y2 = num(d["y2"]))
        f = bf(mu1 = @formula(y1 ~ x + phylo(1 | p | species)),
               mu2 = @formula(y2 ~ x + phylo(1 | p | species)),
               sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
               rho12 = @formula(rho12 ~ 1))
        fitfn_ml = () -> drm(f, Gaussian(); data = dat, tree = phy, method = :ML)
        fitfn_reml = () -> drm(f, Gaussian(); data = dat, tree = phy, method = :REML)
        @assert !vcov_recomputable(fitfn_ml()) "biv_q4_phylo_ml: vcov IS now recomputable -- update the header note and stop forcing N/A"
        push!(W, Workflow("biv_q4_phylo_ml", fitfn_ml, nothing, fit -> predict(fit, dat)))
        push!(W, Workflow("biv_q4_phylo_reml", fitfn_reml, nothing, fit -> predict(fit, dat)))
    end

    # 6. Bernoulli mixed
    let d = read_csv(joinpath(fixdir, "bernoulli_mixed.csv"))
        dat = (y = num(d["y"]), x = num(d["x"]), g = ints(d["g"]))
        f = bf(@formula(y ~ x + (1 | g)))
        fitfn = () -> drm(f, Binomial(); data = dat)
        push!(W, Workflow("bernoulli_mixed", fitfn,
            vcov_recomputable(fitfn()) ? force_vcov : nothing,
            fit -> predict(fit, dat)))
    end

    # 7. Poisson mixed
    let d = read_csv(joinpath(fixdir, "poisson_mixed.csv"))
        dat = (y = ints(d["y"]), x = num(d["x"]), g = ints(d["g"]))
        f = bf(@formula(y ~ x + (1 | g)))
        fitfn = () -> drm(f, Poisson(); data = dat)
        push!(W, Workflow("poisson_mixed", fitfn,
            vcov_recomputable(fitfn()) ? force_vcov : nothing,
            fit -> predict(fit, dat)))
    end

    # 8. LogNormal location-scale
    let d = read_csv(joinpath(fixdir, "lognormal_locscale.csv"))
        dat = (y = num(d["y"]), x = num(d["x"]), z = num(d["z"]))
        f = bf(@formula(y ~ x), @formula(sigma ~ z))
        fitfn = () -> drm(f, LogNormal(); data = dat)
        push!(W, Workflow("lognormal_locscale", fitfn,
            vcov_recomputable(fitfn()) ? force_vcov : nothing,
            fit -> predict(fit, dat)))
    end

    # 9. Meta-analysis meta_V
    let d = read_csv(joinpath(fixdir, "meta_analysis.csv"))
        dat = (y = num(d["y"]), x = num(d["x"]), v = num(d["v"]))
        f = bf(@formula(y ~ x + meta_V(v)), @formula(sigma ~ 1))
        fitfn = () -> drm(f, Gaussian(); data = dat)
        push!(W, Workflow("meta_analysis_meta_V", fitfn,
            vcov_recomputable(fitfn()) ? force_vcov : nothing,
            fit -> predict(fit, dat)))
    end

    # 10. Large sparse LSS, p ~ 2048. Same non-recomputable sparse Cholesky
    # reason as biv_q4_phylo above -- uncertainty is N/A, `fit` already
    # includes it.
    let d = read_csv(joinpath(fixdir, "large_sparse_lss_p2000.csv")),
        phy = read_tree(joinpath(fixdir, "large_sparse_lss_p2000.nwk"))
        dat = (y = num(d["y"]), x = num(d["x"]), xs = num(d["xs"]), species = d["species"])
        f = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x),
               @formula(sd(species, phylogenetic) ~ xs))
        fitfn = () -> drm(f, Gaussian(); data = dat, tree = phy)   # algorithm=:auto -> sparse (G>500)
        push!(W, Workflow("large_sparse_lss_p2000", fitfn, nothing, fit -> predict(fit, dat)))
    end

    return W
end

# ---- timing ----------------------------------------------------------------
# Loop-until-0.25s-cumulative timer (fixes the sub-microsecond floor a single
# `time_ns()` call cannot resolve reliably) -- see header note.

const TIMER_FLOOR_S = 0.25

function timed_loop(f; floor_s::Real = TIMER_FLOOR_S)
    calls = 0
    total = 0.0
    local val
    while true
        t0 = time_ns()
        val = f()
        total += (time_ns() - t0) / 1e9
        calls += 1
        total >= floor_s && break
    end
    return val, total / calls, calls
end

function time_reps(f, k::Int; floor_s::Real = TIMER_FLOOR_S)
    per_call = Float64[]
    calls_vec = Int[]
    local val
    for _ in 1:k
        val, pc, c = timed_loop(f; floor_s = floor_s)
        push!(per_call, pc)
        push!(calls_vec, c)
    end
    return val, per_call, calls_vec
end

function median_(xs)
    s = sort(xs); n = length(s)
    isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end
mean_calls(cs) = round(Int, sum(cs) / length(cs))

fit_converged(fit) = try
    is_converged(fit)
catch
    "NA"
end
fit_loglik(fit) = try
    Float64(loglik(fit))
catch
    NaN
end

function main()
    args = parse_args(ARGS)
    if Threads.nthreads() != args.threads
        error("warm_timing.jl: Julia was started with -t $(Threads.nthreads()) but --threads $(args.threads) was requested. Start Julia with `-t $(args.threads)` to match.")
    end
    BLAS.set_num_threads(args.threads)

    commit = try
        strip(read(`git -C $(dirname(@__DIR__)) rev-parse --short HEAD`, String))
    catch
        "unknown"
    end

    workflows = build_workflows(args.fixtures)
    if args.workflow !== nothing
        workflows = filter(w -> w.name == args.workflow, workflows)
        isempty(workflows) && error("warm_timing.jl: unknown --workflow $(args.workflow)")
    end

    rows = String[]
    header = "engine\tworkflow\tleg\tthreads\treps\tcalls\tmedian_s\tmin_s\tconverged\tloglik\tjulia_version\tblas_threads\tcommit"
    push!(rows, header)

    for w in workflows
        println("== $(w.name) ==")
        print("  warm-up (untimed) ... ")
        fit0 = w.fit()
        w.uncertainty !== nothing && w.uncertainty(fit0)
        w.predict_leg(fit0)
        println("done")

        fitref, fit_times, fit_calls = time_reps(w.fit, args.reps)
        conv = fit_converged(fitref); ll = fit_loglik(fitref)
        @printf("  fit:         median=%.6fs min=%.6fs calls~%d conv=%s loglik=%.4f\n",
                median_(fit_times), minimum(fit_times), mean_calls(fit_calls), conv, ll)
        push!(rows, join(["julia", w.name, "fit", args.threads, args.reps, mean_calls(fit_calls),
                           @sprintf("%.6f", median_(fit_times)), @sprintf("%.6f", minimum(fit_times)),
                           conv, @sprintf("%.6f", ll), string(VERSION), BLAS.get_num_threads(), commit], "\t"))

        if w.uncertainty === nothing
            println("  uncertainty: N/A (vcov computed eagerly & inseparably inside fit() for this route -- see script header note; fit leg above already includes it)")
        else
            _, unc_times, unc_calls = time_reps(() -> w.uncertainty(fitref), args.reps)
            @printf("  uncertainty: median=%.6fs min=%.6fs calls~%d (forced recompute, not cached read)\n",
                    median_(unc_times), minimum(unc_times), mean_calls(unc_calls))
            push!(rows, join(["julia", w.name, "uncertainty", args.threads, args.reps, mean_calls(unc_calls),
                               @sprintf("%.6f", median_(unc_times)), @sprintf("%.6f", minimum(unc_times)),
                               conv, @sprintf("%.6f", ll), string(VERSION), BLAS.get_num_threads(), commit], "\t"))
        end

        _, pred_times, pred_calls = time_reps(() -> w.predict_leg(fitref), args.reps)
        @printf("  predict:     median=%.6fs min=%.6fs calls~%d\n",
                median_(pred_times), minimum(pred_times), mean_calls(pred_calls))
        push!(rows, join(["julia", w.name, "predict", args.threads, args.reps, mean_calls(pred_calls),
                           @sprintf("%.6f", median_(pred_times)), @sprintf("%.6f", minimum(pred_times)),
                           conv, @sprintf("%.6f", ll), string(VERSION), BLAS.get_num_threads(), commit], "\t"))
    end

    mkpath(dirname(args.out))
    open(args.out, "w") do io
        for r in rows
            println(io, r)
        end
    end
    println("wrote ", args.out)
end

main()
