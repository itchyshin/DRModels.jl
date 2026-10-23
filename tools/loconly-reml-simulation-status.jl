#!/usr/bin/env julia

using DRModels

function _arg_value(args, flag, default)
    idx = findfirst(==(flag), args)
    idx === nothing && return default
    idx == length(args) && error("missing value after $(flag)")
    return args[idx + 1]
end

include_medium = "--with-medium-stress" in ARGS
include_large = "--with-large-stress" in ARGS
large_budget = parse(
    Float64,
    _arg_value(ARGS, "--large-stress-budget-seconds", "0.0"),
)
out = _arg_value(
    ARGS,
    "--output",
    joinpath(@__DIR__, "..", "docs", "dev-log", "validation-status",
             "2026-06-21-loconly-reml-simulation-status.tsv"),
)

result = DRModels._loconly_reml_write_simulation_status_tsv(
    out;
    include_medium_stress = include_medium,
    include_large_stress = include_large,
    large_stress_budget_seconds = large_budget,
)

if !result.validation.ok
    error("simulation-status validation failed: $(join(result.validation.errors, "; "))")
end

println("wrote $(result.n_rows) rows to $(result.path)")
