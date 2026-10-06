# fit_phylo_binomial.jl - Julia timing side of the Binomial phylo slice.
#
# Run from repo root:
#   julia --project=bench bench/fit_phylo_binomial.jl

import Pkg
Pkg.activate(dirname(@__DIR__))
using DRModels
using LinearAlgebra
using Printf, Random, Statistics
import Distributions

BLAS.set_num_threads(1)

const OUTDIR = joinpath(@__DIR__, "results", "phylo_binomial")
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
    Random.seed!(2606400 + cell.p)
    phy = random_balanced_tree(cell.p; branch_length = 0.20)
    species = repeat(1:cell.p, inner = cell.m)
    x = randn(length(species))
    u = 0.35 .* randn(cell.p)
    β = [-0.10, 0.45]
    prob = logistic.(β[1] .+ β[2] .* x .+ u[species])
    trials = fill(8, length(species))
    successes = Float64.([rand(Distributions.Binomial(trials[i], prob[i])) for i in eachindex(trials)])
    failures = Float64.(trials) .- successes
    (; phy, species, x, successes, failures)
end

function fit_binomial(dat)
    drm(bf(@formula(cbind(successes, failures) ~ x + phylo(1 | species))),
        Binomial(); data = (; successes = dat.successes, failures = dat.failures,
                            x = dat.x, species = dat.species),
        tree = dat.phy, g_tol = 1e-6, se = false)
end

function run_cell(cell)
    dat = simulate_cell(cell)
    fit_binomial(dat)
    times = Float64[]
    fits = Any[]
    for _ in 1:cell.reps
        fit = nothing
        t = @elapsed begin
            fit = fit_binomial(dat)
        end
        push!(times, t)
        push!(fits, fit)
    end
    fit = fits[end]
    rs = re_sd(fit)
    row = [
        :family => "binomial",
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
        :sd_phylo => get(rs, :species, NaN),
        :julia_threads => Threads.nthreads(),
        :blas_threads => BLAS.get_num_threads(),
    ]
    @printf("[Julia binomial %s] p=%d n=%d med=%.4fs logLik=%.3f beta=(%.3f, %.3f) sd=%.3f\n",
            cell.id, cell.p, cell.n, row[7][2], row[9][2],
            row[11][2][1], row[11][2][2], row[12][2])
    return row
end

rows = [run_cell(cell) for cell in CELLS]
open(joinpath(OUTDIR, "julia_phylo_binomial.json"), "w") do io
    println(io, "[")
    for (i, row) in enumerate(rows)
        comma = i == length(rows) ? "" : ","
        println(io, "  ", json_obj(row), comma)
    end
    println(io, "]")
end
