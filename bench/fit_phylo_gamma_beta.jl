# fit_phylo_gamma_beta.jl — Julia timing side of the Gamma/Beta phylo slice.
#
# Run from repo root:
#   julia --project=bench bench/fit_phylo_gamma_beta.jl

import Pkg
Pkg.activate(dirname(@__DIR__))
using DRModels
using LinearAlgebra
using Printf, Random, Statistics
import Distributions

BLAS.set_num_threads(1)

const OUTDIR = joinpath(@__DIR__, "results", "phylo_gamma_beta")
mkpath(OUTDIR)

const CELLS = [
    (id = "phylo_p128", p = 128, m = 4, n = 512, reps = 5),
    (id = "phylo_p512", p = 512, m = 3, n = 1536, reps = 3),
    (id = "phylo_p1024", p = 1024, m = 2, n = 2048, reps = 3),
    (id = "phylo_p2048", p = 2048, m = 2, n = 4096, reps = 2),
]

json_str(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
json_val(x::Bool) = x ? "true" : "false"
json_val(x::Integer) = string(x)
json_val(x::Real) = isfinite(x) ? @sprintf("%.17g", x) : "null"
json_val(x::AbstractString) = json_str(x)
json_val(x::AbstractVector) = "[" * join(json_val.(x), ",") * "]"
json_val(::Nothing) = "null"
function json_obj(pairs)
    "{" * join([json_str(String(k)) * ":" * json_val(v) for (k, v) in pairs], ",") * "}"
end

logistic(x) = 1 / (1 + exp(-x))

function simulate_cell(cell)
    Random.seed!(2606300 + cell.p)
    phy = random_balanced_tree(cell.p; branch_length = 0.20)
    species = repeat(1:cell.p, inner = cell.m)
    x = randn(length(species))
    u_gamma = 0.35 .* randn(cell.p)
    u_beta = 0.30 .* randn(cell.p)

    β_gamma = [0.15, 0.30]
    sigma_gamma = 0.45
    shape = 1 / sigma_gamma^2
    μ_gamma = exp.(β_gamma[1] .+ β_gamma[2] .* x .+ u_gamma[species])
    y_gamma = Float64.([rand(Distributions.Gamma(shape, μi / shape)) for μi in μ_gamma])

    β_beta = [-0.05, 0.40]
    precision = 18.0
    μ_beta = logistic.(β_beta[1] .+ β_beta[2] .* x .+ u_beta[species])
    y_beta = Float64.([rand(Distributions.Beta(μi * precision, (1 - μi) * precision)) for μi in μ_beta])

    (; phy, species, x, y_gamma, y_beta)
end

function fit_gamma(dat)
    drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
        Gamma(); data = (; y = dat.y_gamma, x = dat.x, species = dat.species),
        tree = dat.phy, g_tol = 1e-6, se = false)
end

function fit_beta(dat)
    drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
        Beta(); data = (; y = dat.y_beta, x = dat.x, species = dat.species),
        tree = dat.phy, g_tol = 1e-6, se = false)
end

function run_fit(fitfun, family::String, cell)
    dat = simulate_cell(cell)
    fitfun(dat)
    times = Float64[]
    fits = Any[]
    for _ in 1:cell.reps
        fit = nothing
        t = @elapsed begin
            fit = fitfun(dat)
        end
        push!(times, t)
        push!(fits, fit)
    end
    fit = fits[end]
    rs = re_sd(fit)
    row = [
        :family => family,
        :cell_id => cell.id,
        :engine => "julia_sparse_laplace",
        :p => cell.p,
        :n => cell.n,
        :time_s => mean(times),
        :time_s_med => median(times),
        :times_all => times,
        :logLik => loglik(fit),
        :converged => fit.converged,
        :beta_mu => collect(coef(fit, :mu)),
        :sigma => exp(coef(fit, :sigma)[1]),
        :sd_phylo => get(rs, :species, NaN),
        :julia_threads => Threads.nthreads(),
        :blas_threads => BLAS.get_num_threads(),
    ]
    @printf("[Julia %s %s] p=%d n=%d med=%.4fs logLik=%.3f beta=(%.3f, %.3f) sigma=%.3f sd=%.3f\n",
            family, cell.id, cell.p, cell.n, row[7][2], row[9][2],
            row[11][2][1], row[11][2][2], row[12][2], row[13][2])
    return row
end

rows = Any[]
for cell in CELLS
    push!(rows, run_fit(fit_gamma, "gamma", cell))
    push!(rows, run_fit(fit_beta, "beta", cell))
end

open(joinpath(OUTDIR, "julia_phylo_gamma_beta.json"), "w") do io
    println(io, "[")
    for (i, row) in enumerate(rows)
        comma = i == length(rows) ? "" : ","
        println(io, "  ", json_obj(row), comma)
    end
    println(io, "]")
end
