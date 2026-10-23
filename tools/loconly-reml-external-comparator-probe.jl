#!/usr/bin/env julia

using DRModels

function _arg_value(args, flag, default)
    idx = findfirst(==(flag), args)
    idx === nothing && return default
    idx == length(args) && error("missing value after $(flag)")
    return args[idx + 1]
end

function _r_package_version(package::AbstractString, rscript::AbstractString)
    exe = Sys.which(rscript)
    exe === nothing && return (
        version = "unavailable",
        available = false,
        rscript_status = :rscript_missing,
    )
    code = """
pkg <- commandArgs(TRUE)[1]
if (!requireNamespace(pkg, quietly = TRUE)) quit(status = 17)
cat(as.character(utils::packageVersion(pkg)))
"""
    try
        version = readchomp(`$exe -e $code $package`)
        return (
            version = isempty(version) ? "unavailable" : version,
            available = !isempty(version),
            rscript_status = isempty(version) ? :package_not_installed :
                :package_version_recorded,
        )
    catch
        return (
            version = "unavailable",
            available = false,
            rscript_status = :package_not_installed,
        )
    end
end

candidate_package = _arg_value(ARGS, "--candidate-package", "phylolm")
rscript = _arg_value(ARGS, "--rscript", "Rscript")
skip_rscript = "--skip-rscript" in ARGS
default_output = joinpath(
    @__DIR__, "..", "docs", "dev-log", "validation-status",
    "2026-06-22-loconly-reml-external-comparator-probe.tsv",
)
out = _arg_value(ARGS, "--output", default_output)

probe = if skip_rscript
    (version = "unprobed", available = false, rscript_status = :skipped)
else
    _r_package_version(candidate_package, rscript)
end

result = DRModels._loconly_reml_write_external_comparator_probe_tsv(
    out;
    candidate_package = candidate_package,
    candidate_version = probe.version,
    package_available = probe.available,
    rscript_status = probe.rscript_status,
    evidence = "tools/loconly-reml-external-comparator-probe.jl",
)

if !result.validation.ok
    error("external-comparator probe validation failed: $(join(result.validation.errors, "; "))")
end

println("wrote $(result.n_rows) row to $(result.path)")
println("candidate_package=$(candidate_package)")
println("candidate_version=$(probe.version)")
println("rscript_status=$(probe.rscript_status)")
