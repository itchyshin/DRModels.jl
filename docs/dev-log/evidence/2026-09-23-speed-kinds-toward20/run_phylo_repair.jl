# Repair pass: four phylo cells that failed on Distributions name clash.
using Pkg
Pkg.activate(".")
using DRModels
using LinearAlgebra, Random, Statistics, Printf, Dates
import Distributions
BLAS.set_num_threads(1)

const SHA = readchomp(`git rev-parse --short=9 HEAD`)
const HOST = "totoro"
const THR = Threads.nthreads()
const OUTDIR = @__DIR__
const REPS = 3
const BOARD = joinpath(OUTDIR, "board_drm_first_wave_20260923_$(SHA).csv")
const board_rows = String[]

function med_wall(fit_fn; reps=REPS)
    fit_fn()
    times = Float64[]
    ll = NaN
    conv = false
    for _ in 1:reps
        t0 = time()
        fit = fit_fn()
        push!(times, time() - t0)
        ll = try loglik(fit) catch; NaN end
        conv = try fit.converged catch; true end
    end
    return median(times), times, ll, conv
end

function record!(cell_id, kind, metric, wall, times, ll, conv, note="")
    push!(board_rows, join([
        "DRModels", cell_id, kind, metric,
        @sprintf("%.6f", wall),
        join([@sprintf("%.6f", t) for t in times], ";"),
        @sprintf("%.9g", ll),
        string(conv),
        SHA, HOST, string(VERSION), string(THR), "1",
        note, "has_receipt"
    ], ","))
    @printf("[%s] med=%.4fs ll=%.4g conv=%s\n", cell_id, wall, ll, conv)
    flush(stdout)
end

function bal_tree(p; bl=0.25)
    return DRModels.random_balanced_tree(p; branch_length=bl)
end

println("=== DRM phylo-repair @ ", SHA, " threads=", THR, " ===")

# 1) drm-phylo-poisson
try
    Random.seed!(2606101)
    p = 100; m = 5; n = p * m
    tree = bal_tree(p)
    species = repeat(1:p, inner=m)
    x = randn(n)
    C = DRModels.sigma_phy_dense(tree; σ²_phy=1.0)
    d = sqrt.(diag(C)); K = C ./ (d * d')
    u = 0.4 .* (cholesky(Symmetric(K)).L * randn(p))
    η = 0.05 .+ 0.30 .* x .+ u[species]
    y = Float64[rand(Distributions.Poisson(exp(η[i]))) for i in 1:n]
    dat = (; y, x, species)
    form = bf(@formula(y ~ x + phylo(1 | species)))
    wall, ts, ll, conv = med_wall(() -> drm(form, Poisson(); data=dat, tree=tree, g_tol=1e-7, se=false))
    record!("drm-phylo-poisson", "D-phylo-pois", "abs_fit_wall_p100", wall, ts, ll, conv, "Julia abs; tip smoke p=100")
catch e
    @warn "drm-phylo-poisson FAILED" exception=(e, catch_backtrace())
end

# 2) drm-phylo-nb2
try
    Random.seed!(2606201)
    p = 100; m = 5; n = p * m
    tree = bal_tree(p)
    species = repeat(1:p, inner=m)
    x = randn(n)
    C = DRModels.sigma_phy_dense(tree; σ²_phy=1.0)
    d = sqrt.(diag(C)); K = C ./ (d * d')
    u = 0.35 .* (cholesky(Symmetric(K)).L * randn(p))
    μ = exp.(0.1 .+ 0.25 .* x .+ u[species])
    y = Float64[rand(Distributions.NegativeBinomial(3.0, 3.0 / (3.0 + μ[i]))) for i in 1:n]
    dat = (; y, x, species)
    form = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1))
    wall, ts, ll, conv = med_wall(() -> drm(form, NegBinomial2(); data=dat, tree=tree, g_tol=1e-7, se=false))
    record!("drm-phylo-nb2", "D-phylo-nb2", "abs_fit_wall_p100", wall, ts, ll, conv, "Julia abs; tip smoke p=100")
catch e
    @warn "drm-phylo-nb2 FAILED" exception=(e, catch_backtrace())
end

# 3) drm-phylo-binomial
try
    Random.seed!(2606301)
    p = 128; m = 4; n = p * m
    tree = bal_tree(p)
    species = repeat(1:p, inner=m)
    x = randn(n)
    C = DRModels.sigma_phy_dense(tree; σ²_phy=1.0)
    d = sqrt.(diag(C)); K = C ./ (d * d')
    u = 0.5 .* (cholesky(Symmetric(K)).L * randn(p))
    η = -0.2 .+ 0.4 .* x .+ u[species]
    trials = fill(8, n)
    successes = Float64[rand(Distributions.Binomial(trials[i], 1 / (1 + exp(-η[i])))) for i in 1:n]
    failures = Float64.(trials) .- successes
    dat = (; successes, failures, x, species)
    form = bf(@formula(cbind(successes, failures) ~ x + phylo(1 | species)))
    wall, ts, ll, conv = med_wall(() -> drm(form, Binomial(); data=dat, tree=tree, g_tol=1e-7, se=false))
    record!("drm-phylo-binomial", "D-phylo-binom", "abs_fit_wall_p128", wall, ts, ll, conv, "Julia abs; tip smoke p=128")
catch e
    @warn "drm-phylo-binomial FAILED" exception=(e, catch_backtrace())
end

# 4) drm-phylo-gamma
try
    Random.seed!(2606501)
    p = 128; m = 4; n = p * m
    tree = bal_tree(p)
    species = repeat(1:p, inner=m)
    x = randn(n)
    C = DRModels.sigma_phy_dense(tree; σ²_phy=1.0)
    d = sqrt.(diag(C)); K = C ./ (d * d')
    u = 0.4 .* (cholesky(Symmetric(K)).L * randn(p))
    μ = exp.(0.2 .+ 0.3 .* x .+ u[species])
    shape = 7.0
    y = Float64[rand(Distributions.Gamma(shape, μ[i] / shape)) for i in 1:n]
    dat = (; y, x, species)
    form = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1))
    wall, ts, ll, conv = med_wall(() -> drm(form, Gamma(); data=dat, tree=tree, g_tol=1e-7, se=false))
    record!("drm-phylo-gamma", "D-phylo-gamma-beta", "abs_fit_wall_p128", wall, ts, ll, conv,
            "Julia abs; Julia-only fence if TMB cannot")
catch e
    @warn "drm-phylo-gamma FAILED" exception=(e, catch_backtrace())
end

# Append to existing board (dedupe by cell_id)
existing = isfile(BOARD) ? readlines(BOARD) : String[]
header = isempty(existing) ?
    "package,cell_id,kind,metric,wall_med_s,all_times_s,logLik,converged,sha,host,julia,julia_threads,blas_threads,note,status" :
    existing[1]
kept = filter(r -> begin
    startswith(r, "package,") && return false
    isempty(strip(r)) && return false
    cid = split(r, ",")[2]
    !any(startswith(nr, "DRModels,$cid,") for nr in board_rows)
end, existing[2:end])
open(BOARD, "w") do io
    println(io, header)
    for r in kept; println(io, r); end
    for r in board_rows; println(io, r); end
end
n_total = count(l -> startswith(l, "DRModels,"), readlines(BOARD))
open(joinpath(OUTDIR, "DONE.txt"), "w") do io
    println(io, "n_cells=$n_total")
    println(io, "sha=$SHA")
    println(io, "finished=", Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS.sss"))
    println(io, "phylo_repair=true")
end
println("WROTE $BOARD n=$(length(board_rows)) new; board_total=$n_total")
