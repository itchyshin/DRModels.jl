# fit_phylo_nb2.jl — Julia side of the NB2 phylo benchmark.
#
# Run from repo root after `Rscript bench/R/gen_phylo_nb2.R`:
#   julia --project=bench bench/fit_phylo_nb2.jl

import Pkg
Pkg.activate(dirname(@__DIR__))
using DRModels
using LinearAlgebra
using DelimitedFiles, Printf, Statistics

BLAS.set_num_threads(1)

const FIXDIR = joinpath(@__DIR__, "fixtures", "phylo_nb2")
const OUTDIR = joinpath(@__DIR__, "results", "phylo_nb2")
mkpath(OUTDIR)

const CELLS = [
    (id = "phylo_p100", p = 100, n = 500, reps = 5),
    (id = "phylo_p500", p = 500, n = 1500, reps = 3),
    (id = "phylo_p1000", p = 1000, n = 2000, reps = 3),
    (id = "phylo_p2000", p = 2000, n = 4000, reps = 2),
]

function read_fixture(id)
    raw = readdlm(joinpath(FIXDIR, "$id.csv"), ',', String; header = true)[1]
    y = Float64.(parse.(Int, raw[:, 1]))
    x = parse.(Float64, raw[:, 2])
    species = raw[:, 3]
    tree = read(joinpath(FIXDIR, "$id.nwk"), String)
    return y, x, species, tree
end

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

function fit_cell(cell)
    y, x, species, tree = read_fixture(cell.id)
    drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
        NegBinomial2(); data = (; y, x, species), tree = tree, g_tol = 1e-6, se = false)
end

function run_cell(cell)
    fit_cell(cell)
    times = Float64[]
    fits = Any[]
    for _ in 1:cell.reps
        fit = nothing
        t = @elapsed begin
            fit = fit_cell(cell)
        end
        push!(times, t)
        push!(fits, fit)
    end
    fit = fits[end]
    rs = re_sd(fit)
    theta = exp(coef(fit, :sigma)[1])
    row = [
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
        :theta_nb2 => theta,
        :sd_phylo => get(rs, :species, NaN),
        :julia_threads => Threads.nthreads(),
        :blas_threads => BLAS.get_num_threads(),
    ]
    @printf("[Julia %s] p=%d n=%d med=%.4fs logLik=%.3f beta=(%.3f, %.3f) theta=%.3f sd=%.3f\n",
            cell.id, cell.p, cell.n, row[6][2], row[8][2],
            row[10][2][1], row[10][2][2], row[11][2], row[12][2])
    return row
end

results = [run_cell(cell) for cell in CELLS]
open(joinpath(OUTDIR, "julia_phylo_nb2.json"), "w") do io
    println(io, "[")
    for (i, row) in enumerate(results)
        comma = i == length(results) ? "" : ","
        println(io, "  ", json_obj(row), comma)
    end
    println(io, "]")
end
