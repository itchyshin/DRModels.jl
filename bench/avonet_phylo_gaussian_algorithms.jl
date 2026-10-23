# avonet_phylo_gaussian_algorithms.jl -- full AVONET/Hackett Gaussian phylo scout.
#
# Run from the repository root:
#   julia --project=. bench/avonet_phylo_gaussian_algorithms.jl
#
# Optional:
#   julia --project=. bench/avonet_phylo_gaussian_algorithms.jl --g-tols=1e-4,1e-6 --reps=2
#   julia --project=. --threads=4 bench/avonet_phylo_gaussian_algorithms.jl --bootstrap-B=20 --bootstrap-mode=both
#   julia --project=. --threads=4 bench/avonet_phylo_gaussian_algorithms.jl --profile --profile-mode=both

import Pkg
Pkg.activate(dirname(@__DIR__))

using DRModels
using DelimitedFiles
using LinearAlgebra
using Printf
using Random
using Statistics

BLAS.set_num_threads(1)

const DEFAULT_REPORT = joinpath(@__DIR__, "..", "report", "avonet-phylo-gaussian-algorithms.md")
const ZERO_BRANCH = r":0+(?:\.0+)?(?=[,);])"

Base.@kwdef struct BenchOptions
    avonet_path::Union{Nothing,String} = nothing
    tree_path::Union{Nothing,String} = nothing
    out_path::String = DEFAULT_REPORT
    g_tols::Vector{Float64} = [1e-4, 1e-6, 1e-8]
    algorithms::Vector{Symbol} = [:auto, :sparse_lbfgs]
    reps::Int = 1
    bootstrap_B::Int = 0
    bootstrap_mode::Symbol = :serial
    profile::Bool = false
    profile_mode::Symbol = :serial
    profile_parm::Symbol = :resd
end

function parse_options(args)
    opts = BenchOptions()
    avonet_path = opts.avonet_path
    tree_path = opts.tree_path
    out_path = opts.out_path
    g_tols = copy(opts.g_tols)
    algorithms = copy(opts.algorithms)
    reps = opts.reps
    bootstrap_B = opts.bootstrap_B
    bootstrap_mode = opts.bootstrap_mode
    profile = opts.profile
    profile_mode = opts.profile_mode
    profile_parm = opts.profile_parm

    for arg in args
        if startswith(arg, "--avonet=")
            avonet_path = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--tree=")
            tree_path = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--out=")
            out_path = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--g-tols=")
            raw = split(split(arg, "=", limit = 2)[2], ",")
            g_tols = parse.(Float64, strip.(raw))
        elseif startswith(arg, "--algorithms=")
            raw = split(split(arg, "=", limit = 2)[2], ",")
            algorithms = Symbol.(strip.(raw))
        elseif startswith(arg, "--reps=")
            reps = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--bootstrap-B=")
            bootstrap_B = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--bootstrap-mode=")
            bootstrap_mode = Symbol(strip(split(arg, "=", limit = 2)[2]))
        elseif arg == "--profile"
            profile = true
        elseif startswith(arg, "--profile=")
            raw = lowercase(strip(split(arg, "=", limit = 2)[2]))
            profile = raw in ("1", "true", "t", "yes", "y")
        elseif startswith(arg, "--profile-mode=")
            profile_mode = Symbol(strip(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--profile-parm=")
            profile_parm = Symbol(strip(split(arg, "=", limit = 2)[2]))
        elseif arg in ("-h", "--help")
            println("""
            AVONET/Hackett Gaussian phylo benchmark

            Options:
              --avonet=PATH       AVONET CSV path.
              --tree=PATH         Hackett Newick tree path.
              --out=PATH          Markdown report path.
              --g-tols=A,B,C      EM convergence tolerances to time.
              --algorithms=A,B    Algorithms: auto, em, sparse, sparse_lbfgs.
              --reps=N            Timed repetitions per tolerance.
              --bootstrap-B=N     Optional bootstrap replicate count.
              --bootstrap-mode=M  Bootstrap mode: serial, threaded, or both.
              --profile[=true]     Optional profile benchmark.
              --profile-mode=M     Profile mode: serial, threaded, or both.
              --profile-parm=P     Profile parameter block, default resd.
            """)
            exit(0)
        else
            throw(ArgumentError("unknown option: $arg"))
        end
    end

    reps >= 1 || throw(ArgumentError("--reps must be >= 1"))
    bootstrap_B >= 0 || throw(ArgumentError("--bootstrap-B must be >= 0"))
    bootstrap_mode in (:serial, :threaded, :both) ||
        throw(ArgumentError("--bootstrap-mode must be serial, threaded, or both"))
    profile_mode in (:serial, :threaded, :both) ||
        throw(ArgumentError("--profile-mode must be serial, threaded, or both"))
    isempty(g_tols) && throw(ArgumentError("--g-tols must contain at least one value"))
    allowed = Set([:auto, :em, :sparse, :sparse_lbfgs])
    bad = setdiff(algorithms, collect(allowed))
    isempty(bad) || throw(ArgumentError("--algorithms contains unsupported entries: $(join(bad, ", "))"))
    return BenchOptions(; avonet_path, tree_path, out_path, g_tols, reps,
        algorithms, bootstrap_B, bootstrap_mode, profile, profile_mode,
        profile_parm)
end

function first_existing(paths)
    for path in paths
        isfile(path) && return path
    end
    return nothing
end

function default_avonet_path()
    first_existing([
        "/Users/z3437171/Dropbox/Github Local/pigauto/avonet/AVONET3_BirdTree.csv",
        "/Users/z3437171/Dropbox/Github Local/BACE/dev/testing_data/AVONET.csv",
    ])
end

function default_tree_path()
    first_existing([
        "/Users/z3437171/Dropbox/Github Local/pigauto/avonet/Stage2_Hackett_MCC_no_neg.tre",
        "/Users/z3437171/Dropbox/Github Local/BACE/dev/testing_data/Hackett_tree.tre",
    ])
end

clean_header(name) = replace(String(name), "\ufeff" => "")
clean_species(name) = replace(strip(String(name)), " " => "_")

function parse_float_cell(x)
    sx = strip(String(x))
    (isempty(sx) || sx == "NA" || sx == "NaN") && return missing
    return try
        parse(Float64, sx)
    catch
        missing
    end
end

function read_avonet(path)
    raw, header = readdlm(path, ',', String; header = true)
    names = clean_header.(vec(header))
    col = Dict(name => i for (i, name) in enumerate(names))
    needed = ["Species3", "Mass", "Hand-Wing.Index", "Beak.Length_Culmen"]
    missing_cols = [name for name in needed if !haskey(col, name)]
    isempty(missing_cols) ||
        throw(ArgumentError("AVONET file is missing columns: $(join(missing_cols, ", "))"))

    rows = Dict{String,NamedTuple{(:log_mass, :hand_wing, :beak),Tuple{Float64,Float64,Float64}}}()
    skipped = 0
    for i in axes(raw, 1)
        sp = clean_species(raw[i, col["Species3"]])
        mass = parse_float_cell(raw[i, col["Mass"]])
        hand = parse_float_cell(raw[i, col["Hand-Wing.Index"]])
        beak = parse_float_cell(raw[i, col["Beak.Length_Culmen"]])
        if isempty(sp) || any(ismissing, (mass, hand, beak))
            skipped += 1
            continue
        end
        rows[sp] = (log_mass = log(Float64(mass)),
                    hand_wing = Float64(hand),
                    beak = Float64(beak))
    end
    return rows, size(raw, 1), skipped
end

standardize(x) = (x .- mean(x)) ./ std(x)

function load_hackett_tree(path)
    newick = read(path, String)
    zero_count = length(collect(eachmatch(ZERO_BRANCH, newick)))
    positive_newick = replace(newick, ZERO_BRANCH => ":1e-8")
    phy = DRModels.augmented_phy(positive_newick)
    return positive_newick, phy, zero_count
end

function avonet_data_for_tree(rows, phy)
    missing_tips = [name for name in phy.leaf_names if !haskey(rows, name)]
    if !isempty(missing_tips)
        preview = join(first(missing_tips, min(length(missing_tips), 8)), ", ")
        throw(ArgumentError("AVONET has no complete row for $(length(missing_tips)) tree tips; first missing: $preview"))
    end
    log_mass = [rows[name].log_mass for name in phy.leaf_names]
    hand_wing = standardize([rows[name].hand_wing for name in phy.leaf_names])
    beak = standardize([rows[name].beak for name in phy.leaf_names])
    species = copy(phy.leaf_names)
    return (; log_mass, hand_wing_z = hand_wing, beak_z = beak, species)
end

function fit_avonet(data, tree; g_tol, algorithm)
    form = bf(@formula(log_mass ~ hand_wing_z + beak_z + phylo(1 | species)),
              @formula(sigma ~ 1))
    return drm(form, Gaussian(); data, tree, algorithm, g_tol)
end

function fit_summary(fit)
    beta = coef(fit, :mu)
    sigma_hat = exp(coef(fit, :sigma)[1])
    sd_phylo = get(re_sd(fit), :species, NaN)
    vdiag = diag(fit.vcov)
    return (
        loglik = loglik(fit),
        converged = fit.converged,
        beta0 = beta[1],
        beta_hand_wing = beta[2],
        beta_beak = beta[3],
        sigma = sigma_hat,
        sd_phylo = sd_phylo,
        finite_vcov = all(isfinite, fit.vcov),
        finite_wald = all(isfinite, vdiag) && all(>(0), vdiag),
        nll_available = fit.nll !== nothing,
        profile_available = fit.nll !== nothing,
    )
end

route_label(algorithm::Symbol) =
    algorithm === :auto ? "auto_sparse_lbfgs" :
    algorithm === :em ? "forced_sparse_em" :
    algorithm === :sparse ? "forced_sparse_em_alias" :
    algorithm === :sparse_lbfgs ? "sparse_lbfgs_profiled" :
    String(algorithm)

algorithm_note(algorithm::Symbol) =
    algorithm in (:auto, :sparse_lbfgs) ?
    "all-node sparse L-BFGS with profiled β" :
    "current all-node sparse EM"

function run_tolerance(data, tree, g_tol, reps, algorithm)
    times = Float64[]
    fit = nothing
    for _ in 1:reps
        GC.gc()
        t = @elapsed fit = fit_avonet(data, tree; g_tol, algorithm)
        push!(times, t)
    end
    s = fit_summary(fit)
    return merge(s, (
        route = route_label(algorithm),
        algorithm = String(algorithm),
        algorithm_note = algorithm_note(algorithm),
        g_tol = g_tol,
        reps = reps,
        min_time_s = minimum(times),
        median_time_s = median(times),
        times = copy(times),
    )), fit
end

function bootstrap_modes(mode::Symbol)
    mode === :serial && return [false]
    mode === :threaded && return [true]
    return [false, true]
end

function run_bootstrap_smoke(fit, data, tree, B; g_tol, threads)
    result = nothing
    try
        total = @elapsed result = bootstrap_result(
            fit; data, B, tree, rng = MersenneTwister(20260608),
            failures = :skip, check_converged = false, threads = threads,
            algorithm = :auto, g_tol = g_tol,
        )
        return (
            ok = true,
            attempted = result.attempted,
            used = result.used,
            failed = result.failed,
            elapsed_s = result.elapsed,
            total_time_s = total,
            threaded = result.threaded,
            requested_threads = threads,
            worker_threads = result.worker_threads,
            julia_threads = result.julia_threads,
            blas_threads = result.blas_threads,
            per_used_s = result.used == 0 ? NaN : result.elapsed / result.used,
            algorithm = "auto_sparse_lbfgs",
            g_tol = g_tol,
            message = "bootstrap refits used explicit Gaussian algorithm/g_tol controls",
        )
    catch err
        return (
            ok = false,
            attempted = B,
            used = 0,
            failed = B,
            elapsed_s = NaN,
            total_time_s = NaN,
            threaded = false,
            requested_threads = threads,
            worker_threads = 0,
            julia_threads = Threads.nthreads(),
            blas_threads = BLAS.get_num_threads(),
            per_used_s = NaN,
            algorithm = "auto_sparse_lbfgs",
            g_tol = g_tol,
            message = sprint(showerror, err),
        )
    end
end

function run_profile_smoke(fit; parm, threads)
    result = nothing
    try
        total = @elapsed result = profile_result(fit; parm, threads = threads)
        ci = isempty(result.ci) ? nothing : result.ci[1]
        return (
            ok = true,
            parm = parm,
            attempted = result.attempted,
            used = result.used,
            failed = result.failed,
            elapsed_s = result.elapsed,
            total_time_s = total,
            threaded = result.threaded,
            requested_threads = threads,
            worker_threads = result.worker_threads,
            julia_threads = result.julia_threads,
            blas_threads = result.blas_threads,
            lower = ci === nothing ? NaN : ci.lower,
            upper = ci === nothing ? NaN : ci.upper,
            autodiff = result.autodiff,
            message = "profile_result completed",
        )
    catch err
        return (
            ok = false,
            parm = parm,
            attempted = 1,
            used = 0,
            failed = 1,
            elapsed_s = NaN,
            total_time_s = NaN,
            threaded = false,
            requested_threads = threads,
            worker_threads = 0,
            julia_threads = Threads.nthreads(),
            blas_threads = BLAS.get_num_threads(),
            lower = NaN,
            upper = NaN,
            autodiff = :unavailable,
            message = sprint(showerror, err),
        )
    end
end

fmt_fixed(x; digits = 3) = isfinite(x) ? @sprintf("%.*f", digits, x) : "NA"
fmt_sci(x) = isfinite(x) ? @sprintf("%.3e", x) : "NA"
fmt_bool(x) = x ? "yes" : "no"

function write_report(path, context, rows, bootstrap_rows, profile_rows)
    mkpath(dirname(path))
    best_loglik = maximum(row.loglik for row in rows)
    ref = rows[argmax([row.loglik for row in rows])]
    open(path, "w") do io
        println(io, "# AVONET phylogenetic Gaussian algorithm scout")
        println(io)
        println(io, "This report times the current Julia route for a real AVONET/Hackett Gaussian phylogenetic mean model with $(context.n_species) tree tips. It is a direct DRModels.jl benchmark, not the R bridge timing table.")
        println(io)
        println(io, "## Data and model")
        println(io)
        println(io, "- AVONET CSV: `$(context.avonet_path)`")
        println(io, "- Hackett tree: `$(context.tree_path)`")
        println(io, "- Tree tips: $(context.n_species); total all-node tree states: $(context.n_total); internal nodes: $(context.n_internal)")
        println(io, "- AVONET input rows: $(context.avonet_rows); skipped incomplete rows: $(context.skipped_rows)")
        println(io, "- Exact zero-length tree branches rewritten to `1e-8` before parsing: $(context.zero_branches)")
        println(io, "- Model: `log(Mass) ~ z(Hand-Wing.Index) + z(Beak.Length_Culmen) + phylo(1 | species)`, `sigma ~ 1`")
        println(io, "- CPU policy: Julia threads = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads()), Julia $(VERSION)")
        println(io)
        println(io, "## Timings")
        println(io)
        println(io, "| route | g_tol | reps | median_s | min_s | converged | logLik | delta_from_best | beta_hand_wing | beta_beak | sigma | sd_phylo | finite_vcov | nll_for_profile |")
        println(io, "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|")
        for row in rows
            println(io, "| $(row.route) | $(fmt_sci(row.g_tol)) | $(row.reps) | $(fmt_fixed(row.median_time_s)) | $(fmt_fixed(row.min_time_s)) | $(fmt_bool(row.converged)) | $(fmt_fixed(row.loglik; digits = 6)) | $(fmt_sci(best_loglik - row.loglik)) | $(fmt_fixed(row.beta_hand_wing; digits = 6)) | $(fmt_fixed(row.beta_beak; digits = 6)) | $(fmt_fixed(row.sigma; digits = 6)) | $(fmt_fixed(row.sd_phylo; digits = 6)) | $(fmt_bool(row.finite_vcov)) | $(fmt_bool(row.nll_available)) |")
        end
        println(io)
        println(io, "Reference row for coefficient deltas is the highest-logLik row (`g_tol = $(fmt_sci(ref.g_tol))`).")
        println(io)
        println(io, "## Inference pipeline status")
        println(io)
        println(io, "| target | current status | implication |")
        println(io, "|---|---|---|")
        println(io, "| Fixed-effect Wald SEs | Sparse L-BFGS stores the profiled sparse-GLS fixed-effect covariance block | Fixed-effect Wald intervals can be read from the `:mu` block; scale and variance-component Wald rows remain deliberately unset in this first slice. |")
        println(io, "| Random-effect / variance-component profile CIs | Sparse L-BFGS attaches the full sparse objective closure | `profile_result(fit; parm = :resd)` is the high-value CI target; it is now mechanically possible and should be benchmarked next. |")
        println(io, "| Parametric bootstrap | `bootstrap_result(fit; ...)` can reuse a fitted object and Gaussian refits now accept `algorithm` / `g_tol` controls | Bootstrap is the natural Julia speed-payoff path because refits are independent and threadable; use the benchmark below to check whether the advantage appears at the requested B. |")
        println(io)
        if isempty(bootstrap_rows)
            println(io, "No bootstrap smoke was requested. Run with `--bootstrap-B=1` or larger to record the current refit behavior.")
        else
            println(io, "## Bootstrap benchmark")
            println(io)
            println(io, "| B | mode | workers | algorithm | g_tol | ok | used | failed | elapsed_s | sec_per_used | total_time_s | message |")
            println(io, "|---:|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|")
            for boot in bootstrap_rows
                mode = boot.requested_threads ? "threaded" : "serial"
                println(io, "| $(context.bootstrap_B) | $(mode) | $(boot.worker_threads) | $(boot.algorithm) | $(fmt_sci(boot.g_tol)) | $(fmt_bool(boot.ok)) | $(boot.used) | $(boot.failed) | $(fmt_fixed(boot.elapsed_s)) | $(fmt_fixed(boot.per_used_s)) | $(fmt_fixed(boot.total_time_s)) | $(replace(boot.message, '|' => '/')) |")
            end
            if length(bootstrap_rows) >= 2
                serial = first(filter(b -> !b.requested_threads, bootstrap_rows))
                threaded = first(filter(b -> b.requested_threads, bootstrap_rows))
                if serial.ok && threaded.ok && isfinite(threaded.elapsed_s) && threaded.elapsed_s > 0
                    println(io)
                    println(io, "Serial/threaded bootstrap speedup on the simulated-refit phase: $(fmt_fixed(serial.elapsed_s / threaded.elapsed_s; digits = 2))x.")
                end
            end
            println(io)
        end
        if isempty(profile_rows)
            println(io, "No profile benchmark was requested. Run with `--profile --profile-mode=both` to record direct Julia profile behavior.")
        else
            println(io, "## Profile benchmark")
            println(io)
            println(io, "| parm | mode | workers | ok | attempted | used | failed | elapsed_s | total_time_s | lower | upper | autodiff | message |")
            println(io, "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
            for prof in profile_rows
                mode = prof.requested_threads ? "threaded" : "serial"
                println(io, "| $(prof.parm) | $(mode) | $(prof.worker_threads) | $(fmt_bool(prof.ok)) | $(prof.attempted) | $(prof.used) | $(prof.failed) | $(fmt_fixed(prof.elapsed_s)) | $(fmt_fixed(prof.total_time_s)) | $(fmt_fixed(prof.lower; digits = 6)) | $(fmt_fixed(prof.upper; digits = 6)) | $(prof.autodiff) | $(replace(prof.message, '|' => '/')) |")
            end
            if length(profile_rows) >= 2
                serial = first(filter(p -> !p.requested_threads, profile_rows))
                threaded = first(filter(p -> p.requested_threads, profile_rows))
                if serial.ok && threaded.ok && isfinite(threaded.elapsed_s) && threaded.elapsed_s > 0
                    println(io)
                    println(io, "Serial/threaded profile speedup on the endpoint phase: $(fmt_fixed(serial.elapsed_s / threaded.elapsed_s; digits = 2))x.")
                end
            end
            println(io)
        end
        println(io, "## Reading")
        println(io)
        println(io, "The single-fit sparse EM route and the sparse L-BFGS route both use the all-node tree representation, so the algorithm question is no longer dense tips versus sparse nodes. The current default for this cell is sparse profiled L-BFGS with exact Takahashi trace gradients and an attached objective for profile/bootstrap workflows. EM remains available as an explicit comparator and can be very fast at loose tolerance, but it does not carry the profile objective or covariance surface in this slice.")
        println(io)
        println(io, "For applied users, the largest Julia advantage is likely the repeated-refit pipeline: profile likelihood or bootstrap confidence intervals for random-effect and variance-component targets. Fixed-effect Wald intervals should be cheap once information is available, but they are not where the dramatic speedup should be sold.")
    end
    return path
end

function main(args = ARGS)
    opts = parse_options(args)
    avonet_path = opts.avonet_path === nothing ? default_avonet_path() : opts.avonet_path
    tree_path = opts.tree_path === nothing ? default_tree_path() : opts.tree_path
    avonet_path === nothing && error("could not find AVONET CSV; pass --avonet=PATH")
    tree_path === nothing && error("could not find Hackett tree; pass --tree=PATH")

    rows_by_species, avonet_rows, skipped_rows = read_avonet(avonet_path)
    tree_string, phy, zero_branches = load_hackett_tree(tree_path)
    data = avonet_data_for_tree(rows_by_species, phy)
    form = bf(@formula(log_mass ~ hand_wing_z + beak_z + phylo(1 | species)),
              @formula(sigma ~ 1))

    println("AVONET rows available: $(length(data.log_mass)); tree tips: $(phy.n_leaves)")
    println("warming requested routes...")
    for algorithm in opts.algorithms
        fit_avonet(data, tree_string; g_tol = first(opts.g_tols), algorithm)
    end

    rows = NamedTuple[]
    fits = Any[]
    for algorithm in opts.algorithms, g_tol in opts.g_tols
        println("timing algorithm=$(algorithm), g_tol=$(g_tol), reps=$(opts.reps)")
        row, fit = run_tolerance(data, tree_string, g_tol, opts.reps, algorithm)
        push!(rows, row)
        push!(fits, fit)
        @printf("  %-22s median %.3fs, logLik %.6f, converged=%s\n",
                row.route, row.median_time_s, row.loglik, string(row.converged))
    end

    bootstrap_index = 1
    bootstrap_rows = NamedTuple[]
    if opts.bootstrap_B > 0
        for threads in bootstrap_modes(opts.bootstrap_mode)
            push!(bootstrap_rows, run_bootstrap_smoke(
                fits[bootstrap_index], data, tree_string, opts.bootstrap_B;
                g_tol = rows[bootstrap_index].g_tol, threads = threads,
            ))
        end
    end
    profile_rows = NamedTuple[]
    if opts.profile
        for threads in bootstrap_modes(opts.profile_mode)
            push!(profile_rows, run_profile_smoke(
                fits[bootstrap_index]; parm = opts.profile_parm,
                threads = threads,
            ))
        end
    end
    context = (
        avonet_path = avonet_path,
        tree_path = tree_path,
        avonet_rows = avonet_rows,
        skipped_rows = skipped_rows,
        n_species = phy.n_leaves,
        n_total = phy.n_total,
        n_internal = phy.n_total - phy.n_leaves,
        zero_branches = zero_branches,
        bootstrap_B = opts.bootstrap_B,
        formula = form,
    )
    report_path = write_report(opts.out_path, context, rows, bootstrap_rows, profile_rows)
    println("wrote $(report_path)")
    return rows
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
