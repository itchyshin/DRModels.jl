# profile_crossed_laplace.jl -- quick CPU-aware family sweep for crossed Laplace.
#
# This is an engine-lane diagnostic, not a public API benchmark. Poisson uses the
# drmTMB-comparison path from #70; Binomial/NB2/Gamma use the internal generic
# crossed-Laplace kernels to prove that the same sparse mode/gradient spine
# carries beyond Poisson. NB2/Gamma/Beta estimate their scalar nuisance
# parameter in the Laplace objective; the fixed-nuisance helpers remain covered
# by unit tests only.
#
# Run:
#   julia --project=bench bench/profile_crossed_laplace.jl

import Pkg
Pkg.activate(dirname(@__DIR__))

using DRModels
using LinearAlgebra, Printf, Random, Statistics
import Distributions

BLAS.set_num_threads(1)

const OUT = joinpath(@__DIR__, "..", "report", "crossed-laplace-family-profile.md")

logistic(x) = 1 / (1 + exp(-x))

function json_escape(s)
    replace(String(s), "\\" => "\\\\", "\"" => "\\\"")
end

function crossed_fixture(seed, n, G, H)
    rng = MersenneTwister(seed)
    x = randn(rng, n)
    gidx = [rand(rng, 1:G) for _ in 1:n]
    hidx = [rand(rng, 1:H) for _ in 1:n]
    X = hcat(ones(n), x)
    β = [0.25, 0.45]
    σg = 0.45
    σh = 0.35
    bg = σg .* randn(rng, G)
    bh = σh .* randn(rng, H)
    η = [β[1] + β[2] * x[i] + bg[gidx[i]] + bh[hidx[i]] for i in 1:n]
    comps = [
        (ones(n), gidx, G, "g"),
        (ones(n), hidx, H, "h"),
    ]
    return (; rng, n, G, H, X, η, comps, β, σg, σh)
end

function median_fit_time(fitfun; reps)
    fitfun()
    times = Float64[]
    fits = Any[]
    for _ in 1:reps
        fit = nothing
        t = @elapsed fit = fitfun()
        push!(times, t)
        push!(fits, fit)
    end
    return median(times), times, fits[end]
end

function family_rows(cell)
    rows = NamedTuple[]
    n = cell.n
    μ = exp.(cell.η)

    ypoi = Float64.([rand(cell.rng, Distributions.Poisson(μ[i])) for i in 1:n])
    t, times, fit = median_fit_time(; reps = cell.n <= 5000 ? 5 : 3) do
        DRModels._fit_poisson_crossed_laplace(DRModels.Poisson(), ypoi, cell.X, cell.comps, ["(Intercept)", "x"], 1e-7;
                                         se = false, polish_iterations = 0)
    end
    push!(rows, (family = "Poisson", nuisance = "none", median_s = t, times = times,
        nuisance_hat = NaN, nuisance_truth = NaN, converged = fit.converged,
        beta1 = coef(fit, :mu)[2], sd_g = re_sd(fit)[:g], sd_h = re_sd(fit)[:h]))

    ntr = fill(8.0, n)
    s = Float64.([rand(cell.rng, Distributions.Binomial(round(Int, ntr[i]), logistic(cell.η[i]))) for i in 1:n])
    t, times, fit = median_fit_time(; reps = cell.n <= 5000 ? 5 : 3) do
        DRModels._fit_binomial_crossed_laplace(DRModels.Binomial(), s, ntr, cell.X, cell.comps, ["(Intercept)", "x"], 1e-7;
                                          polish_iterations = 10)
    end
    push!(rows, (family = "Binomial", nuisance = "none", median_s = t, times = times,
        nuisance_hat = NaN, nuisance_truth = NaN, converged = fit.converged,
        beta1 = coef(fit, :mu)[2], sd_g = re_sd(fit)[:g], sd_h = re_sd(fit)[:h]))

    size = 3.0
    ynb = Float64.([rand(cell.rng, Distributions.NegativeBinomial(size, size / (size + μ[i]))) for i in 1:n])
    t, times, fit = median_fit_time(; reps = cell.n <= 5000 ? 5 : 3) do
        DRModels._fit_nb2_crossed_laplace(DRModels.NegBinomial2(), ynb, cell.X, ones(n, 1),
                                     cell.comps, ["(Intercept)", "x"], ["(Intercept)"], 1e-7)
    end
    push!(rows, (family = "NB2", nuisance = "size estimated", median_s = t, times = times,
        nuisance_hat = exp(coef(fit, :sigma)[1]), nuisance_truth = size,
        converged = fit.converged, beta1 = coef(fit, :mu)[2], sd_g = re_sd(fit)[:g], sd_h = re_sd(fit)[:h]))

    shape = 7.0
    yg = Float64.([rand(cell.rng, Distributions.Gamma(shape, μ[i] / shape)) for i in 1:n])
    t, times, fit = median_fit_time(; reps = cell.n <= 5000 ? 5 : 3) do
        DRModels._fit_gamma_crossed_laplace(DRModels.Gamma(), yg, cell.X, ones(n, 1),
                                       cell.comps, ["(Intercept)", "x"], ["(Intercept)"], 1e-7)
    end
    push!(rows, (family = "Gamma", nuisance = "shape estimated", median_s = t, times = times,
        nuisance_hat = exp(-2 * coef(fit, :sigma)[1]), nuisance_truth = shape,
        converged = fit.converged, beta1 = coef(fit, :mu)[2], sd_g = re_sd(fit)[:g], sd_h = re_sd(fit)[:h]))

    if n <= 5000
        precision = 25.0
        p = logistic.(cell.η)
        ybeta = Float64.([rand(cell.rng, Distributions.Beta(p[i] * precision, (1 - p[i]) * precision)) for i in 1:n])
        t, times, fit = median_fit_time(; reps = cell.n <= 5000 ? 5 : 3) do
            DRModels._fit_beta_crossed_laplace(DRModels.Beta(), ybeta, cell.X, ones(n, 1),
                                          cell.comps, ["(Intercept)", "x"], ["(Intercept)"], 1e-7)
        end
        push!(rows, (family = "Beta", nuisance = "precision estimated", median_s = t, times = times,
            nuisance_hat = exp(-2 * coef(fit, :sigma)[1]), nuisance_truth = precision,
            converged = fit.converged, beta1 = coef(fit, :mu)[2], sd_g = re_sd(fit)[:g], sd_h = re_sd(fit)[:h]))
    end

    return rows
end

cells = [
    (name = "small", n = 1000, G = 20, H = 20),
    (name = "medium", n = 5000, G = 50, H = 50),
    (name = "large", n = 20000, G = 100, H = 100),
    (name = "fixedq_large", n = 20000, G = 50, H = 50),
]

allrows = NamedTuple[]
for (j, c) in enumerate(cells)
    cell = crossed_fixture(9000 + j, c.n, c.G, c.H)
    for row in family_rows(cell)
        @printf("[%-12s %-8s] n=%d G=%d H=%d med=%.4fs conv=%s beta1=%.3f sd=(%.3f,%.3f)\n",
            row.family, c.name, c.n, c.G, c.H, row.median_s, row.converged, row.beta1, row.sd_g, row.sd_h)
        push!(allrows, merge(c, row))
    end
end

mkpath(dirname(OUT))
open(OUT, "w") do io
    println(io, "# Crossed sparse-Laplace family profile")
    println(io)
    println(io, "CPU-aware run: Julia threads = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads()).")
    println(io, "Poisson is drmTMB-comparable through the #70 paired benchmark. Binomial/NB2/Gamma/Beta here are internal Julia engine proofs; NB2/Gamma/Beta estimate one constant nuisance parameter in the Laplace objective with exact implicit nuisance-gradient corrections.")
    println(io, "The generic non-Gaussian path uses fused per-observation derivative kernels so value, mean derivatives, and nuisance derivatives share expensive link/special-function work.")
    println(io, "The crossed Hessian path is adaptive: dense factorisation for q ≤ $(DRModels.CROSSED_SPARSE_Q_THRESHOLD), sparse CHOLMOD + Takahashi selected inverse for larger q.")
    println(io)
    println(io, "| cell | family | n | G | H | median/s | beta1 | sd_g | sd_h | nuisance hat | nuisance truth | converged | nuisance |")
    println(io, "|:-----|:-------|--:|--:|--:|---------:|------:|-----:|-----:|-------------:|---------------:|:----------|:---------|")
    for r in allrows
        nhat = isfinite(r.nuisance_hat) ? @sprintf("%.3f", r.nuisance_hat) : "-"
        ntruth = isfinite(r.nuisance_truth) ? @sprintf("%.3f", r.nuisance_truth) : "-"
        @printf(io, "| %s | %s | %d | %d | %d | %.4f | %.3f | %.3f | %.3f | %s | %s | %s | %s |\n",
            r.name, r.family, r.n, r.G, r.H, r.median_s, r.beta1, r.sd_g, r.sd_h,
            nhat, ntruth, r.converged, r.nuisance)
    end
    println(io)
    println(io, "Interpretation guardrails:")
    println(io, "- Do not compare NB2/Gamma/Beta rows to drmTMB from this report alone; use the paired R benchmark report for parity claims.")
    println(io, "- The fixed-q large cell is the scaling check: n grows while q stays at G+H=100.")
    println(io, "- Beta is skipped for n > 5000 in this quick profile because exact beta d3/nuisance derivatives remain polygamma-heavy even after derivative fusion.")
    println(io, "- Timings are medians from repeated warm-started fits inside this process, not extrapolations.")
end

println("wrote ", OUT)
