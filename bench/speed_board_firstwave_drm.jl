# speed_board_firstwave_drm.jl — Phase B first-wave DRM cells (Julia abs walls)
#
#   SPEED_CELL_ID=drm-phylo-poisson julia --project=. bench/speed_board_firstwave_drm.jl
#   SPEED_ARRAY_INDEX=1 julia --project=. bench/speed_board_firstwave_drm.jl
#
# Julia-only abs walls (no drmTMB H2H on DRAC this wave). Paired × deferred.

using DRModels
using Random
using Statistics
using Printf
using LinearAlgebra
using Distributions: Poisson, Binomial, NegativeBinomial, Gamma, Normal

BLAS.set_num_threads(1)

const REPS = parse(Int, get(ENV, "DRM_FIRSTWAVE_REPS", "3"))
const CELLS = [
    "drm-phylo-poisson",
    "drm-phylo-nb2",
    "drm-phylo-binomial",
    "drm-h2h-q4-vs-tmb-p1000",
    "drm-phylo-gamma",
    "drm-crossed-binomial",
    "drm-biv-gauss-rho12",
    "drm-profile-ci-locscale",
    "drm-animal-gauss",
    "drm-lss-sd-slope",
]

function timed_median(f; reps::Int = REPS)
    f()
    secs = Float64[]
    val = nothing
    for _ in 1:reps
        t = Base.@timed f()
        push!(secs, t.time)
        val = t.value
    end
    return (median(secs), val)
end

function _sha()
    try
        strip(read(`git rev-parse --short HEAD`, String))
    catch
        get(ENV, "DRM_BOARD_SHA", "unknown")
    end
end

_ll(fit) = try
    Float64(loglik(fit))
catch
    NaN
end

function _pick_cell()
    if haskey(ENV, "SPEED_CELL_ID") && !isempty(ENV["SPEED_CELL_ID"])
        return ENV["SPEED_CELL_ID"]
    end
    idx = parse(Int, get(ENV, "SPEED_ARRAY_INDEX", get(ENV, "SLURM_ARRAY_TASK_ID", "1")))
    1 <= idx <= length(CELLS) || error("SPEED_ARRAY_INDEX=$idx out of 1:$(length(CELLS))")
    return CELLS[idx]
end

function _write_row(path, cell_id, kind, detail, wall_s, loglik, note, status)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "cell_id\tkind\tdetail\twall_s\tloglik\tnote\tstatus\tsha\thost\tthreads")
        @printf(io, "%s\t%s\t%s\t%.6f\t%.10g\t%s\t%s\t%s\t%s\t%d\n",
                cell_id, kind, detail,
                isnan(wall_s) ? NaN : wall_s,
                isnan(loglik) ? NaN : loglik,
                replace(note, '\t' => ' ', '\n' => ' '),
                status, _sha(), gethostname(), Threads.nthreads())
    end
    @printf("%-36s  %10.4f s  status=%s  -> %s\n", cell_id, wall_s, status, path)
end

function _phylo_count_dgp(; family::Symbol, p = 64, m = 6, seed = 9401)
    Random.seed!(seed)
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    β = [0.15, 0.35]
    σphy = 0.45
    C = sigma_phy_dense(phy; σ²_phy = σphy^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    η = β[1] .+ β[2] .* x .+ u[species]
    y = if family === :poisson
        Float64.([rand(Poisson(exp(clamp(ηi, -4, 4)))) for ηi in η])
    elseif family === :nb2
        r = 3.0
        Float64.([begin
            μ = exp(clamp(ηi, -4, 4))
            rand(NegativeBinomial(r, r / (r + μ)))
        end for ηi in η])
    elseif family === :binomial
        Float64.([rand(Binomial(10, 1 / (1 + exp(-clamp(ηi, -4, 4))))) for ηi in η])
    elseif family === :gamma
        Float64.([rand(Gamma(2.0, exp(clamp(ηi, -3, 3)) / 2)) for ηi in η])
    else
        error("unknown family $family")
    end
    return phy, (; y, x, species)
end

function run_phylo_poisson()
    cell = "drm-phylo-poisson"
    phy, data = _phylo_count_dgp(; family = :poisson, seed = 9401)
    wall, fit = timed_median(() -> drm(bf(@formula(y ~ x + phylo(1 | species))), Poisson();
                                       data = data, tree = phy, se = false, g_tol = 1e-6))
    return cell, "D-phylo-pois", "drm Poisson phylo p=64", wall, _ll(fit), "julia_abs_only"
end

function run_phylo_nb2()
    cell = "drm-phylo-nb2"
    phy, data = _phylo_count_dgp(; family = :nb2, seed = 9402)
    wall, fit = timed_median(() -> drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                                       NegativeBinomial();
                                       data = data, tree = phy, se = false, g_tol = 1e-6))
    return cell, "D-phylo-nb2", "drm NB2 phylo p=64", wall, _ll(fit), "julia_abs_only"
end

function run_phylo_binomial()
    cell = "drm-phylo-binomial"
    phy, data = _phylo_count_dgp(; family = :binomial, seed = 9403)
    # cbind-style via successes / trials columns
    trials = fill(10.0, length(data.y))
    fails = trials .- data.y
    dat = (; successes = data.y, failures = fails, x = data.x, species = data.species)
    wall, fit = timed_median(() -> drm(bf(@formula(cbind(successes, failures) ~ x + phylo(1 | species))),
                                       Binomial();
                                       data = dat, tree = phy, se = false, g_tol = 1e-6))
    return cell, "D-phylo-binom", "drm binomial phylo p=64", wall, _ll(fit), "julia_abs_only"
end

function run_phylo_gamma()
    cell = "drm-phylo-gamma"
    phy, data = _phylo_count_dgp(; family = :gamma, seed = 9404)
    wall, fit = timed_median(() -> drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                                       Gamma();
                                       data = data, tree = phy, se = false, g_tol = 1e-6))
    return cell, "D-phylo-gamma-beta", "drm Gamma phylo p=64", wall, _ll(fit),
           "julia_abs_only; Julia-only fence if TMB cannot"
end

function run_h2h_q4()
    cell = "drm-h2h-q4-vs-tmb-p1000"
    # Julia-only abs at p=1000 (R TMB arm not on DRAC this wave).
    # Reuse head_to_head script env if present; else small q4 bivariate abs.
    Random.seed!(9405)
    p = 64  # keep array tasks within 2h; label size in note
    phy = random_balanced_tree(p; branch_length = 0.2)
    m = 4
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    y1 = 1.0 .+ 0.5 .* x .+ 0.3 .* randn(n)
    y2 = -0.3 .+ 0.4 .* x .+ 0.3 .* randn(n)
    dat = (; y1, y2, x, species)
    form = bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
              mu2 = @formula(y2 ~ x + phylo(1 | species)),
              sigma1 = @formula(sigma1 ~ 1),
              sigma2 = @formula(sigma2 ~ 1),
              rho12 = @formula(rho12 ~ 1))
    wall, fit = timed_median(() -> drm(form; data = dat, tree = phy, se = false, g_tol = 1e-6))
    return cell, "D-gauss-q4-phylo", "drm biv q2/q4-class phylo julia abs p=$p", wall, _ll(fit),
           "julia_abs_only; TMB pair deferred; not p1000 yet"
end

function run_crossed_binomial()
    cell = "drm-crossed-binomial"
    Random.seed!(9406)
    G, H, n = 20, 20, 1000
    x = randn(n)
    g = [Symbol("g", rand(1:G)) for _ in 1:n]
    h = [Symbol("h", rand(1:H)) for _ in 1:n]
    η = 0.2 .+ 0.4 .* x .+ 0.3 .* randn(n)
    successes = Float64.([rand(Binomial(8, 1 / (1 + exp(-clamp(η[i], -4, 4))))) for i in 1:n])
    failures = 8.0 .- successes
    dat = (; successes, failures, x, g, h)
    wall, fit = timed_median(() -> drm(bf(@formula(cbind(successes, failures) ~ x + (1 | g) + (1 | h))),
                                       Binomial(); data = dat, se = false, g_tol = 1e-6))
    return cell, "D-crossed-family", "drm crossed binomial G=H=20 n=1000", wall, _ll(fit),
           "julia_abs_only"
end

function run_biv_rho12()
    cell = "drm-biv-gauss-rho12"
    Random.seed!(9407)
    n = 800
    x = randn(n)
    z = randn(n)
    e1 = randn(n)
    e2 = 0.5 .* e1 .+ sqrt(1 - 0.25) .* randn(n)
    y1 = 0.3 .+ 0.7 .* x .+ 0.4 .* e1
    y2 = -0.2 .+ 0.5 .* z .+ 0.4 .* e2
    dat = (; y1, y2, x, z)
    form = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ z),
              sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
              rho12 = @formula(rho12 ~ 1))
    wall, fit = timed_median(() -> drm(form; data = dat, se = false, g_tol = 1e-6))
    return cell, "D-biv-gauss", "drm biv gaussian residual rho12 n=800", wall, _ll(fit),
           "julia_abs_only"
end

function run_profile_ci()
    cell = "drm-profile-ci-locscale"
    Random.seed!(9408)
    n = 600
    x = randn(n)
    z = randn(n)
    y = 0.4 .+ 0.75 .* x .- 0.35 .* z .+ exp(-0.45) .* randn(n)
    form = bf(@formula(y ~ x + z), @formula(sigma ~ 1))
    fit = drm(form; data = (; y, x, z), se = false, g_tol = 1e-6)
    wall, _ = timed_median(() -> confint(fit; method = :profile, parm = :mu))
    return cell, "D-profile-ci", "confint profile loc-scale", wall, _ll(fit), "julia_abs_only"
end

function run_animal()
    cell = "drm-animal-gauss"
    Random.seed!(9409)
    p = 48
    phy = random_balanced_tree(p; branch_length = 0.2)
    A = sigma_phy_dense(phy; σ²_phy = 1.0)
    # force PD correlation-ish
    A = Symmetric(A ./ sqrt.(diag(A) * diag(A)'))
    A = Matrix(A) + 1e-6 * I
    m = 4
    id = repeat(1:p, inner = m)
    n = length(id)
    x = randn(n)
    u = cholesky(Symmetric(A)).L * randn(p)
    y = 0.5 .+ 0.4 .* x .+ u[id] .+ 0.5 .* randn(n)
    dat = (; y, x, id)
    wall, fit = timed_median(() -> drm(bf(@formula(y ~ x + animal(1 | id)), @formula(sigma ~ 1));
                                       data = dat, A = A, se = false, g_tol = 1e-6))
    return cell, "D-animal", "drm animal() gaussian p=48", wall, _ll(fit), "julia_abs_only"
end

function run_lss()
    cell = "drm-lss-sd-slope"
    Random.seed!(9410)
    G = 24
    m = 8
    n = G * m
    g = repeat(1:G, inner = m)
    x = randn(n)
    z = randn(G)
    bg = 0.4 .* randn(G)
    σg = exp.(-0.3 .+ 0.25 .* z)
    y = 0.2 .+ 0.5 .* x .+ σg[g] .* bg[g] .+ 0.45 .* randn(n)
    dat = (; y, x, g, z = z[g])
    form = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(sd(g) ~ 1 + z))
    wall, fit = timed_median(() -> drm(form; data = dat, se = false, g_tol = 1e-6))
    return cell, "D-lss", "drm sd(g) ~ 1+z Gaussian", wall, _ll(fit), "julia_abs_only"
end

const DISPATCH = Dict(
    "drm-phylo-poisson" => run_phylo_poisson,
    "drm-phylo-nb2" => run_phylo_nb2,
    "drm-phylo-binomial" => run_phylo_binomial,
    "drm-h2h-q4-vs-tmb-p1000" => run_h2h_q4,
    "drm-phylo-gamma" => run_phylo_gamma,
    "drm-crossed-binomial" => run_crossed_binomial,
    "drm-biv-gauss-rho12" => run_biv_rho12,
    "drm-profile-ci-locscale" => run_profile_ci,
    "drm-animal-gauss" => run_animal,
    "drm-lss-sd-slope" => run_lss,
)

function main()
    cell = _pick_cell()
    out = get(ENV, "DRM_FIRSTWAVE_OUT",
              joinpath(@__DIR__, "results", "firstwave_$(cell)_$(_sha()).tsv"))
    println("=== DRM first-wave @ $(_sha()) cell=$cell ===")
    println("threads=$(Threads.nthreads()) OPENBLAS=$(get(ENV, "OPENBLAS_NUM_THREADS", "?")) reps=$REPS host=$(gethostname())")
    haskey(DISPATCH, cell) || error("unknown cell $cell")
    try
        cid, kind, detail, wall, ll, note = DISPATCH[cell]()
        _write_row(out, cid, kind, detail, wall, ll, note, "ok")
    catch e
        msg = sprint(showerror, e)
        _write_row(out, cell, "ERROR", "ERROR", NaN, NaN, msg, "error")
        @warn "cell failed (receipt written)" cell exception = e
    end
end

main()
