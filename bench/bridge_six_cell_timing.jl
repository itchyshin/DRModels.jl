# bridge_six_cell_timing.jl — #372 Julia arm (drm_bridge wall-clock)
#
# CPU-aware local timing for the six #370 bridge fixture cells.
# Does NOT touch src/; uses committed fixtures only. Stdlib + DRModels only
# (writes TOML — no JSON.jl dependency).
#
# Run from repo root:
#   julia --project=. bench/bridge_six_cell_timing.jl
#
# Env:
#   DRM_372_REPS   timed reps after one warmup (default 5)
#   DRM_372_CELLS  comma-separated cell ids (overrides cohort default)
#   DRM_BRIDGE_TIMING_COHORT  "six" (default, #372) or "plus5" (#389)

using Dates
using LinearAlgebra
using Statistics
using TOML

BLAS.set_num_threads(1)

using DRModels

include(joinpath(@__DIR__, "..", "test", "parity", "loadfixture.jl"))

const COHORT_SIX = [
    "gaussian-locscale",
    "gaussian-bivariate-rho12",
    "robust-student",
    "count-nbinom2",
    "proportion-beta",
    "meta-analysis-V",
]

const COHORT_PLUS5 = [
    "count-poisson",
    "positive-gamma",
    "binomial-trials",
    "positive-lognormal",
    "nbinom2-dispersion",
]

function _cohort_default()
    c = lowercase(get(ENV, "DRM_BRIDGE_TIMING_COHORT", "six"))
    return c == "plus5" ? copy(COHORT_PLUS5) : copy(COHORT_SIX)
end

function _parse_cells()
    raw = get(ENV, "DRM_372_CELLS", "")
    isempty(raw) && return _cohort_default()
    return String.(split(raw, ","; keepempty = false))
end

function _issue_tag()
    return lowercase(get(ENV, "DRM_BRIDGE_TIMING_COHORT", "six")) == "plus5" ? "#389" : "#372"
end

function _results_subdir()
    return lowercase(get(ENV, "DRM_BRIDGE_TIMING_COHORT", "six")) == "plus5" ?
           "bridge_plus5_389" : "bridge_six_cell_372"
end

function _out_basename()
    return lowercase(get(ENV, "DRM_BRIDGE_TIMING_COHORT", "six")) == "plus5" ?
           "julia_bridge_plus5.toml" : "julia_bridge_six_cell.toml"
end

function _reps()
    return max(1, parse(Int, get(ENV, "DRM_372_REPS", "5")))
end

function _data_n(data)
    isempty(propertynames(data)) && return 0
    return length(getfield(data, first(propertynames(data))))
end

function _bridge_loglik(out)
    if out isa AbstractDict
        for k in ("logLik", "loglik", :logLik, :loglik)
            haskey(out, k) && return Float64(out[k])
        end
    end
    return NaN
end

function time_cell(cell::AbstractString; reps::Int)
    dir = joinpath(@__DIR__, "..", "test", "parity", "fixtures", cell)
    isdir(dir) || error("missing fixture dir: $dir")
    fit_meta = TOML.parsefile(joinpath(dir, "expected.toml"))["fit"]
    formula_text = String(get(fit_meta, "formula", ""))
    family = String(get(fit_meta, "family", ""))
    data = load_data(dir)

    # Warmup (discard) — compile + one fit.
    warm = drm_bridge(; formula = formula_text, family = family, data = data)
    loglik = _bridge_loglik(warm)

    times = Float64[]
    for _ in 1:reps
        push!(times, @elapsed drm_bridge(; formula = formula_text, family = family, data = data))
    end
    return Dict{String,Any}(
        "cell" => cell,
        "family" => family,
        "formula" => formula_text,
        "n" => Int(get(fit_meta, "n", _data_n(data))),
        "reps" => reps,
        "warmup_discarded" => true,
        "times_s" => times,
        "median_s" => median(times),
        "min_s" => minimum(times),
        "max_s" => maximum(times),
        "logLik" => loglik,
        "ok" => true,
        "note" => "",
    )
end

function main()
    cells = _parse_cells()
    reps = _reps()
    rows = Dict{String,Any}[]
    for cell in cells
        @info "timing Julia drm_bridge" cell reps
        try
            push!(rows, time_cell(cell; reps = reps))
        catch err
            push!(rows, Dict{String,Any}(
                "cell" => cell,
                "family" => "",
                "formula" => "",
                "n" => 0,
                "reps" => reps,
                "warmup_discarded" => true,
                "times_s" => Float64[],
                "median_s" => NaN,
                "min_s" => NaN,
                "max_s" => NaN,
                "logLik" => NaN,
                "ok" => false,
                "note" => sprint(showerror, err),
            ))
        end
    end

    out = Dict{String,Any}(
        "arm" => "julia_drm_bridge",
        "issued" => _issue_tag(),
        "timestamp_utc" => string(Dates.now(Dates.UTC)),
        "julia_version" => string(VERSION),
        "blas_threads" => BLAS.get_num_threads(),
        "julia_threads" => Threads.nthreads(),
        "hostname" => gethostname(),
        "reps" => reps,
        "cells" => rows,
    )

    results_dir = joinpath(@__DIR__, "results", _results_subdir())
    mkpath(results_dir)
    out_path = joinpath(results_dir, _out_basename())
    open(out_path, "w") do io
        TOML.print(io, out)
    end
    println("wrote ", out_path)
    for r in rows
        if r["ok"]
            println(r["cell"], " median_s=", r["median_s"], " min_s=", r["min_s"])
        else
            println(r["cell"], " FAILED: ", r["note"])
        end
    end
end

main()
