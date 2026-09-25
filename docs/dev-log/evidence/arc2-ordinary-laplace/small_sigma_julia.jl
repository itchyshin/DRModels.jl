# Arc 2 small-sigma receipt: DRModels side.
#
# Fits the five small-sigma fixtures in test/fixtures/ordinary_laplace/ with
# `marginal = :Laplace` and with the default `marginal = :LA` (GHQ-32), and
# writes small_sigma_julia.tsv: per cell and route, df, logLik, converged, the
# fitted log sigma, and (against small_sigma_native.tsv) |ΔlogLik| and the max
# relative estimate difference.
#
# Run from the DRModels.jl repo root after small_sigma_native.R:
#   JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia --project=. \
#     docs/dev-log/evidence/arc2-ordinary-laplace/small_sigma_julia.jl
using DRModels
using Printf
const D = DRModels

const DIR = joinpath("docs", "dev-log", "evidence", "arc2-ordinary-laplace")
const FIX = joinpath("test", "fixtures", "ordinary_laplace")

function readfix(path)
    lines = readlines(path)
    hdr = replace.(split(lines[1], ","), "\"" => "")
    rows = [replace.(split(l, ","), "\"" => "") for l in lines[2:end]]
    col(n) = [r[findfirst(==(n), hdr)] for r in rows]
    num(n) = parse.(Float64, col(n))
    return "f" in hdr ? (y = num("y"), x = num("x"), z = num("z"), f = col("f"), g = col("g")) :
           "z" in hdr ? (y = num("y"), x = num("x"), z = num("z"), g = col("g")) :
                        (y = num("y"), x = num("x"), g = col("g"))
end

const CELLS = [
    ("gamma_sigma0012", D.Gamma(), bf(@formula(y ~ x + z + f + (1 | g)), @formula(sigma ~ 1))),
    ("beta_sigma0012", D.Beta(), bf(@formula(y ~ x + z + f + (1 | g)), @formula(sigma ~ 1))),
    ("gamma_sigma0003", D.Gamma(), bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))),
    ("nbinom2_sigma003", NegBinomial2(), bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))),
    ("nbinom2_sigma005_plateau", NegBinomial2(), bf(@formula(y ~ x + z + (1 | g)), @formula(sigma ~ 1)))]

nl = readlines(joinpath(DIR, "small_sigma_native.tsv"))
nh = split(nl[1], '\t')
nat = [Dict(zip(nh, split(l, '\t'))) for l in nl[2:end]]

open(joinpath(DIR, "small_sigma_julia.tsv"), "w") do io
    println(io, join(["cell", "route", "df", "logLik", "converged", "log_sigma",
                      "abs_dlogLik_vs_native", "max_rel_dest_vs_native"], '\t'))
    for (cell, fam, f) in CELLS
        d = readfix(joinpath(FIX, cell * ".csv"))
        nr = filter(r -> r["cell"] == cell, nat)
        llN = parse(Float64, nr[1]["logLik"])
        θN = parse.(Float64, [r["estimate"] for r in nr])
        for route in (:Laplace, :LA)
            fit = route === :LA ? drm(f, fam; data = d, se = false) :
                                  drm(f, fam; data = d, marginal = :Laplace, se = false)
            isg = fit.blocks[findfirst(p -> p.first === :sigma, fit.blocks)].second[1]
            rel = maximum(abs.(fit.theta .- θN) ./ max.(abs.(θN), 1e-8))
            println(io, join([cell, String(route), string(dof(fit)),
                              @sprintf("%.10f", loglik(fit)), string(fit.converged),
                              @sprintf("%.4f", fit.theta[isg]),
                              @sprintf("%.2e", abs(loglik(fit) - llN)),
                              @sprintf("%.2e", rel)], '\t'))
        end
    end
end
print(read(joinpath(DIR, "small_sigma_julia.tsv"), String))
