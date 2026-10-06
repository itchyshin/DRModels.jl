# fit_crossed_family.jl -- Julia side of the #80 crossed-family benchmark.
#
# Run from repo root after `julia --project=bench bench/gen_crossed_family.jl`:
#   julia --project=bench bench/fit_crossed_family.jl

import Pkg
Pkg.activate(dirname(@__DIR__))

using DRModels
using DelimitedFiles, LinearAlgebra, Printf, Statistics

BLAS.set_num_threads(1)

const FIXDIR = joinpath(@__DIR__, "fixtures", "crossed_family")
const OUTDIR = joinpath(@__DIR__, "results", "crossed_family")
mkpath(OUTDIR)

const CELLS = [
    (id = "small", G = 20, H = 20, n = 1000, reps = 3),
    (id = "medium", G = 50, H = 50, n = 5000, reps = 3),
    (id = "fixedq_n20000", G = 50, H = 50, n = 20000, reps = 2),
]

const FAMILIES = ["poisson", "binomial", "nb2", "gamma", "beta"]

families_for(cell) = cell.n > 5000 ? ["poisson", "binomial", "nb2", "gamma"] : FAMILIES

function read_fixture(id)
    raw = readdlm(joinpath(FIXDIR, "$id.csv"), ',', String; header = true)[1]
    x = parse.(Float64, raw[:, 1])
    g = Symbol.(raw[:, 2])
    h = Symbol.(raw[:, 3])
    y_pois = Float64.(parse.(Int, raw[:, 4]))
    y_nb = Float64.(parse.(Int, raw[:, 5]))
    s = Float64.(parse.(Int, raw[:, 6]))
    fail = Float64.(parse.(Int, raw[:, 7]))
    y_gamma = parse.(Float64, raw[:, 8])
    y_beta = parse.(Float64, raw[:, 9])
    return (; x, g, h, y_pois, y_nb, s, fail, y_gamma, y_beta)
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

function design_parts(dat)
    n = length(dat.x)
    X = hcat(ones(n), dat.x)
    Xσ = ones(n, 1)
    gidx, G = DRModels._group_index(dat.g)
    hidx, H = DRModels._group_index(dat.h)
    comps = [(ones(n), gidx, G, "g"), (ones(n), hidx, H, "h")]
    return X, Xσ, comps
end

function fit_family(family, dat)
    X, Xσ, comps = design_parts(dat)
    nmμ = ["(Intercept)", "x"]
    nmσ = ["(Intercept)"]
    if family == "poisson"
        return DRModels._fit_poisson_crossed_laplace(DRModels.Poisson(), dat.y_pois, X, comps, nmμ, 1e-7;
                                                se = false, polish_iterations = 0)
    elseif family == "binomial"
        return DRModels._fit_binomial_crossed_laplace(DRModels.Binomial(), dat.s, dat.s .+ dat.fail,
                                                 X, comps, nmμ, 1e-7; polish_iterations = 10)
    elseif family == "nb2"
        return DRModels._fit_nb2_crossed_laplace(DRModels.NegBinomial2(), dat.y_nb, X, Xσ,
                                            comps, nmμ, nmσ, 1e-7)
    elseif family == "gamma"
        return DRModels._fit_gamma_crossed_laplace(DRModels.Gamma(), dat.y_gamma, X, Xσ,
                                              comps, nmμ, nmσ, 1e-7)
    elseif family == "beta"
        return DRModels._fit_beta_crossed_laplace(DRModels.Beta(), dat.y_beta, X, Xσ,
                                             comps, nmμ, nmσ, 1e-7)
    end
    error("unknown family: $family")
end

function nuisance_value(family, fit)
    if family == "nb2"
        return exp(coef(fit, :sigma)[1])
    elseif family == "gamma" || family == "beta"
        return exp(-2 * coef(fit, :sigma)[1])
    end
    return NaN
end

function run_one(cell, family)
    dat = read_fixture(cell.id)
    fit_family(family, dat)
    times = Float64[]
    fits = Any[]
    for _ in 1:cell.reps
        fit = nothing
        t = @elapsed fit = fit_family(family, dat)
        push!(times, t)
        push!(fits, fit)
    end
    fit = fits[end]
    rs = re_sd(fit)
    row = [
        :cell_id => cell.id,
        :family => family,
        :engine => "julia_sparse_laplace",
        :n => cell.n,
        :G => cell.G,
        :H => cell.H,
        :time_s => mean(times),
        :time_s_med => median(times),
        :times_all => times,
        :logLik => loglik(fit),
        :converged => fit.converged,
        :beta_mu => collect(coef(fit, :mu)),
        :nuisance => nuisance_value(family, fit),
        :sd_g => get(rs, :g, NaN),
        :sd_h => get(rs, :h, NaN),
        :julia_threads => Threads.nthreads(),
        :blas_threads => BLAS.get_num_threads(),
    ]
    @printf("[Julia %-8s %-13s] n=%d med=%.4fs conv=%s beta1=%.3f sd=(%.3f,%.3f) nuis=%.3f\n",
            family, cell.id, cell.n, row[8][2], row[11][2], row[12][2][2],
            row[14][2], row[15][2], row[13][2])
    return row
end

results = Any[]
for cell in CELLS, family in families_for(cell)
    push!(results, run_one(cell, family))
end

open(joinpath(OUTDIR, "julia_crossed_family.json"), "w") do io
    println(io, "[")
    for (i, row) in enumerate(results)
        comma = i == length(results) ? "" : ","
        println(io, "  ", json_obj(row), comma)
    end
    println(io, "]")
end
