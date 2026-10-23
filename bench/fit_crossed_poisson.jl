# fit_crossed_poisson.jl — Julia side of #70 crossed Poisson benchmark.
#
# Run from repo root after `julia --project=bench bench/gen_crossed_poisson.jl`:
#   julia --project=bench bench/fit_crossed_poisson.jl

import Pkg
Pkg.activate(dirname(@__DIR__))
using DRModels
using LinearAlgebra
using DelimitedFiles, Printf, Statistics

BLAS.set_num_threads(1)

const FIXDIR = joinpath(@__DIR__, "fixtures", "crossed_poisson")
const OUTDIR = joinpath(@__DIR__, "results", "crossed_poisson")
mkpath(OUTDIR)

const CELLS = [
    (id = "single_control", kind = "single", G = 50, H = 0, n = 1500, reps = 5),
    (id = "crossed_small", kind = "crossed", G = 20, H = 20, n = 1000, reps = 5),
    (id = "crossed_medium", kind = "crossed", G = 50, H = 50, n = 5000, reps = 5),
    (id = "crossed_large", kind = "crossed", G = 100, H = 100, n = 20000, reps = 3),
    (id = "fixedq_n1000", kind = "crossed", G = 50, H = 50, n = 1000, reps = 5),
    (id = "fixedq_n20000", kind = "crossed", G = 50, H = 50, n = 20000, reps = 3),
]

function read_fixture(id)
    raw = readdlm(joinpath(FIXDIR, "$id.csv"), ',', String; header = true)[1]
    n = size(raw, 1)
    y = Float64.(parse.(Int, raw[:, 1]))
    x = parse.(Float64, raw[:, 2])
    g = Symbol.(raw[:, 3])
    h = Symbol.(raw[:, 4])
    return y, x, g, h
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
    y, x, g, h = read_fixture(cell.id)
    X = hcat(ones(length(y)), x)
    gidx, G = DRModels._group_index(g)
    comps = [(ones(length(y)), gidx, G, "g")]
    if cell.kind == "crossed"
        hidx, H = DRModels._group_index(h)
        push!(comps, (ones(length(y)), hidx, H, "h"))
    end

    DRModels._fit_poisson_crossed_laplace(DRModels.Poisson(), y, X, comps, ["(Intercept)", "x"], 1e-7;
                                     se = false, polish_iterations = 0)
end

function run_cell(cell)
    warm = fit_cell(cell)
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
    row = [
        :cell_id => cell.id,
        :engine => "julia_sparse_laplace",
        :kind => cell.kind,
        :n => cell.n,
        :G => cell.G,
        :H => cell.H,
        :time_s => mean(times),
        :time_s_med => median(times),
        :times_all => times,
        :logLik => loglik(fit),
        :converged => fit.converged,
        :beta_mu => collect(coef(fit, :mu)),
        :sd_g => get(rs, :g, NaN),
        :sd_h => get(rs, :h, NaN),
        :julia_threads => Threads.nthreads(),
        :blas_threads => BLAS.get_num_threads(),
    ]
    @printf("[Julia %s] n=%d med=%.4fs logLik=%.3f beta=(%.3f, %.3f) sd_g=%.3f sd_h=%.3f\n",
            cell.id, cell.n, row[8][2], row[10][2], row[12][2][1], row[12][2][2], row[13][2], row[14][2])
    return row
end

results = [run_cell(cell) for cell in CELLS]
open(joinpath(OUTDIR, "julia_crossed_poisson.json"), "w") do io
    println(io, "[")
    for (i, row) in enumerate(results)
        comma = i == length(results) ? "" : ","
        println(io, "  ", json_obj(row), comma)
    end
    println(io, "]")
end
