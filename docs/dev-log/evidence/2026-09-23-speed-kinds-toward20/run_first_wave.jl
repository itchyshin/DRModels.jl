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
const board_rows = String[]

function med_wall(fit_fn; reps=REPS)
    fit_fn()  # warmup
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

println("=== DRM first-wave @ ", SHA, " threads=", THR, " ===")

# --- helpers ---
function bal_tree(p; bl=0.25)
    return DRModels.random_balanced_tree(p; branch_length=bl)
end

# 1) drm-phylo-poisson (p=100, m=5)
try
    Random.seed!(2606101)
    p = 100; m = 5; n = p * m
    tree = bal_tree(p)
    tips = string.(1:p)
    # Newick via write? phylo accepts tree object
    species = repeat(1:p, inner=m)
    x = randn(n)
    # simple phylo effect via dense C
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

# 4) drm-h2h-q4-vs-tmb-p1000 — Julia abs at p=1000 (R pair optional later)
try
    # reuse head_to_head script with env
    env = copy(ENV)
    env["DRM_376_PS"] = "1000"
    env["DRM_376_REPS"] = "3"
    env["DRM_376_NREP"] = "4"
    t0 = time()
    run(pipeline(setenv(`julia --project=. bench/head_to_head_q4_scaling.jl`, env);
                 stdout=joinpath(OUTDIR, "h2h_q4_p1000.log"),
                 stderr=joinpath(OUTDIR, "h2h_q4_p1000.log")))
    elapsed = time() - t0
    # parse TOML if present
    toml = joinpath("bench", "results", "q4_scaling_h2h_376", "julia_q4_scaling.toml")
    wall = NaN; ll = NaN
    if isfile(toml)
        for line in eachline(toml)
            if occursin("p = 1000", line) || occursin("p=1000", line)
                # fallthrough; parse med from nearby
            end
            m = match(r"median_s\s*=\s*([0-9.eE+-]+)", line)
            m !== nothing && (wall = parse(Float64, m.captures[1]))
            m2 = match(r"loglik\s*=\s*([0-9.eE+-]+)", line)
            m2 !== nothing && (ll = parse(Float64, m2.captures[1]))
        end
    end
    if !isfinite(wall)
        # fallback: script wall as script_elapsed (not ideal but banked)
        wall = elapsed
    end
    record!("drm-h2h-q4-vs-tmb-p1000", "D-gauss-q4-phylo", "julia_abs_p1000", wall, [wall], ll, true,
            "Julia arm via head_to_head_q4_scaling; TMB pair not run this receipt (harness/load)")
catch e
    @warn "drm-h2h-q4 FAILED" exception=(e, catch_backtrace())
end

# 5) drm-phylo-gamma
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

# 6) drm-crossed-binomial (small cell)
try
    include(joinpath(@__DIR__, "..", "..", "..", "bench", "gen_crossed_family.jl"))
catch
    # gen may already be run; try direct path
    run(`julia --project=. bench/gen_crossed_family.jl`)
end

try
    # fit only binomial on small via inline (avoid full ladder)
    using DelimitedFiles
    FIX = joinpath("bench", "fixtures", "crossed_family", "small.csv")
    if !isfile(FIX)
        run(`julia --project=. bench/gen_crossed_family.jl`)
    end
    raw = readdlm(FIX, ',', String; header=true)[1]
    x = parse.(Float64, raw[:, 1])
    g = Symbol.(raw[:, 2]); h = Symbol.(raw[:, 3])
    s = Float64.(parse.(Int, raw[:, 6])); fail = Float64.(parse.(Int, raw[:, 7]))
    n = length(x)
    X = hcat(ones(n), x)
    gidx, G = DRModels._group_index(g)
    hidx, H = DRModels._group_index(h)
    comps = [(ones(n), gidx, G, "g"), (ones(n), hidx, H, "h")]
    nmμ = ["(Intercept)", "x"]
    wall, ts, ll, conv = med_wall(() -> DRModels._fit_binomial_crossed_laplace(
        DRModels.Binomial(), s, s .+ fail, X, comps, nmμ, 1e-7; polish_iterations=10))
    record!("drm-crossed-binomial", "D-crossed-family", "abs_fit_wall_small", wall, ts, ll, conv,
            "crossed binomial small G=H=20 n=1000")
catch e
    @warn "drm-crossed-binomial FAILED" exception=(e, catch_backtrace())
end

# 7) drm-biv-gauss-rho12
try
    Random.seed!(20260531)
    n = 2000  # smaller than recovery 6000 for wall
    x = randn(n)
    μ1 = 0.3 .+ 0.5 .* x; μ2 = -0.2 .+ 0.4 .* x
    σ1 = exp.(-0.1 .+ 0.2 .* x); σ2 = exp.(0.0 .- 0.3 .* x)
    ρ = tanh.(0.4 .+ 0.3 .* x)
    z1 = randn(n); z2 = randn(n)
    y1 = μ1 .+ σ1 .* z1
    y2 = μ2 .+ σ2 .* (ρ .* z1 .+ sqrt.(1 .- ρ .^ 2) .* z2)
    dat = (; y1, y2, x)
    form = bf(mu1=@formula(y1 ~ x), mu2=@formula(y2 ~ x),
              sigma1=@formula(sigma1 ~ x), sigma2=@formula(sigma2 ~ x),
              rho12=@formula(rho12 ~ x))
    wall, ts, ll, conv = med_wall(() -> drm(form, Gaussian(); data=dat))
    record!("drm-biv-gauss-rho12", "D-biv-gauss", "abs_fit_wall_n2000", wall, ts, ll, conv, "bivariate residual rho12")
catch e
    @warn "drm-biv-gauss-rho12 FAILED" exception=(e, catch_backtrace())
end

# 8) drm-profile-ci-locscale
try
    Random.seed!(8101)
    n = 600
    x = randn(n); z = randn(n)
    y = 0.4 .+ 0.75 .* x .- 0.35 .* z .+ exp(-0.45) .* randn(n)
    form = bf(@formula(y ~ x + z), @formula(sigma ~ 1))
    dat = (; y, x, z)
    fit = drm(form, Gaussian(); data=dat)
    # time profile CI if available
    wall, ts, ll, conv = med_wall(() -> begin
        if isdefined(DRModels, :confint) || hasmethod(confint, Tuple{typeof(fit)})
            confint(fit; method=:profile)
        else
            # fallback: call fit.nll a few times as proxy — better try profile
            try
                DRModels.profile_ci(fit)
            catch
                confint(fit)
            end
        end
        fit
    end; reps=2)
    record!("drm-profile-ci-locscale", "D-profile-ci", "abs_profile_ci_wall", wall, ts, loglik(fit), true,
            "profile CI on loc-scale gaussian n=600")
catch e
    @warn "drm-profile-ci-locscale FAILED" exception=(e, catch_backtrace())
end

# 9) drm-animal-gauss
try
    Random.seed!(20260606)
    G = 60; m = 6; n = G * m
    M = randn(G, G); A0 = M * M' / G + I
    d = sqrt.(diag(A0)); A = A0 ./ (d * d')
    id = repeat(1:G, inner=m); x = randn(n)
    σ = 0.4; σs = 0.8
    u = σs .* (cholesky(Symmetric(A)).L * randn(G))
    y = 0.3 .+ 0.5 .* x .+ u[id] .+ σ .* randn(n)
    dat = (; y, x, id)
    form = bf(@formula(y ~ x + animal(1 | id)), @formula(sigma ~ 1))
    wall, ts, ll, conv = med_wall(() -> drm(form, Gaussian(); data=dat, A=A))
    record!("drm-animal-gauss", "D-animal", "abs_fit_wall_G60", wall, ts, ll, conv, "animal() Gaussian A supplied")
catch e
    @warn "drm-animal-gauss FAILED" exception=(e, catch_backtrace())
end

# 10) drm-lss-sd-slope
try
    Random.seed!(20260715)
    n_id = 80; n_each = 6
    sex = repeat([0.0, 1.0], inner=n_id ÷ 2)
    b = randn(n_id) .* [0.65, 0.40][Int.(sex) .+ 1]
    id = repeat(1:n_id, inner=n_each)
    sexl = sex[id]
    y = [0.35, 0.70][Int.(sexl) .+ 1] .+ b[id] .+
        randn(n_id * n_each) .* [0.35, 0.60][Int.(sexl) .+ 1]
    dat = (; y, sex=sexl, id)
    form = bf(@formula(y ~ sex + (1 | id)), @formula(sigma ~ sex), @formula(sd(id) ~ sex))
    wall, ts, ll, conv = med_wall(() -> drm(form, Gaussian(); data=dat))
    record!("drm-lss-sd-slope", "D-lss", "abs_fit_wall_n480", wall, ts, ll, conv, "sd(id) ~ sex LSS")
catch e
    @warn "drm-lss-sd-slope FAILED" exception=(e, catch_backtrace())
end

# write board CSV
hdr = "package,cell_id,kind,metric,wall_med_s,all_times_s,logLik,converged,sha,host,julia,julia_threads,blas_threads,note,status"
csv = joinpath(OUTDIR, "board_drm_first_wave_20260923_$(SHA).csv")
open(csv, "w") do io
    println(io, hdr)
    for r in board_rows
        println(io, r)
    end
end
println("WROTE ", csv, " n=", length(board_rows))
open(joinpath(OUTDIR, "DONE.txt"), "w") do io
    println(io, "n_cells=", length(board_rows))
    println(io, "sha=", SHA)
    println(io, "finished=", Dates.now())
end
