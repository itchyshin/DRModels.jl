#!/usr/bin/env julia
# Public-frontend receipt for the two frozen joint missing-predictor fixtures.
# This is a bounded local fit check: no R process, bridge, or interval claim.

using DRModels, ForwardDiff, LinearAlgebra, SHA, TOML

length(ARGS) == 3 || error("usage: check_joint_frontend_fit.jl DATA_REFERENCE_TOML UNCERTAINTY_REFERENCE_TOML NEW_RECEIPT_TOML")
data_path, uncertainty_path, output_path = abspath.(ARGS)
isfile(data_path) || error("data reference does not exist")
isfile(uncertainty_path) || error("uncertainty reference does not exist")
isfile(output_path) && error("refusing stale output: $output_path")

BLAS.set_num_threads(1)
Threads.nthreads() == 1 && BLAS.get_num_threads() == 1 || error("wrong resource budget")

source_root = dirname(pathof(DRModels))
source_manifest() = Dict(
    relpath(joinpath(dir, file), source_root) => bytes2hex(sha256(read(joinpath(dir, file))))
    for (dir, _, files) in walkdir(source_root) for file in files
)
as_rows(A) = [collect(row) for row in eachrow(A)]
as_float_or_zero(values) = [ismissing(value) ? 0.0 : Float64(value) for value in values]
as_available(values) = [!ismissing(value) for value in values]

function table_columns(table)
    return Dict{String,Any}(
        "variable" => table.variable,
        "original_row" => table.original_row,
        "model_row" => table.model_row,
        "observed" => collect(table.observed),
        "estimate" => Float64.(table.estimate),
        "std_error" => as_float_or_zero(table.std_error),
        "std_error_available" => as_available(table.std_error),
        "source" => table.source,
        "uncertainty_status" => table.uncertainty_status,
    )
end

before = source_manifest()
reference = TOML.parsefile(data_path)
uncertainty = TOML.parsefile(uncertainty_path)
receipt = Dict{String,Any}(
    "scope" => "Two public Julia formula fits on frozen 160-row fixtures; native theta comparison is an optional validator gate",
    "data_sha256" => bytes2hex(sha256(read(data_path))),
    "reference_sha256" => bytes2hex(sha256(read(data_path))),
    "uncertainty_reference_sha256" => bytes2hex(sha256(read(uncertainty_path))),
    "runner_sha256" => bytes2hex(sha256(read(@__FILE__))),
    "source_sha256_before" => before,
    "runtime" => Dict(
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "loaded_source" => pathof(DRModels),
    ),
    "cases" => Dict{String,Any}(),
)

started = time()
for kind in ("gaussian", "bernoulli")
    ref = reference[kind]
    native_uncertainty = uncertainty[kind]
    isapprox(Float64.(native_uncertainty["theta"]), Float64.(ref["theta"]); atol = 1e-12, rtol = 0) ||
        error("$kind frozen data and uncertainty references disagree on theta")
    n = length(ref["original_row"])
    n == 160 || error("$kind fixture denominator must be 160")
    y = Union{Missing,Float64}[ref["y_observed"][i] ? Float64(ref["y"][i]) : missing for i in 1:n]
    x = Union{Missing,Float64}[ref["x_observed"][i] ? Float64(ref["x"][i]) : missing for i in 1:n]
    data = (; y, x, z = Float64.(ref["z"]))
    impute_spec = kind == "gaussian" ?
        (x = @formula(x ~ z),) :
        (x = impute_model(@formula(x ~ z); family = Binomial()),)
    fit = drm(bf(@formula(y ~ z + mi(x)), @formula(sigma ~ 1)), Gaussian();
              data, impute = impute_spec,
              missing = miss_control(response = "include", predictor = "model"))
    fit isa JointDrmFit || error("$kind public drm route did not return JointDrmFit")
    prepared = fit.prepared
    raw = Float64.(coef(fit))
    objective = prepared.fit.nll
    gradient = ForwardDiff.gradient(objective, raw)
    hessian = ForwardDiff.hessian(objective, raw)
    table_all = imputed(fit; rows = :all)
    table_missing = imputed(fit; rows = :missing)
    table_no_se = imputed(fit; rows = :all, se = false)
    case = Dict{String,Any}(
        "predictor" => kind,
        "theta" => raw,
        "coef_mu" => Float64.(coef(fit, :mu)),
        "coef_sigma" => Float64.(coef(fit, :sigma)),
        "coef_mi_x" => Float64.(coef(fit, :mi_x)),
        "coef_logsd_mi_x" => kind == "gaussian" ? Float64.(coef(fit, :logsd_mi_x)) : Float64[],
        "sigma_mi_x" => kind == "gaussian" ? Float64.(coef(fit, :sigma_mi_x)) : Float64[],
        "covariance" => as_rows(vcov(fit)),
        "loglik" => loglik(fit),
        "nll" => objective(raw),
        "gradient" => gradient,
        "hessian" => as_rows(hessian),
        "nobs" => nobs(fit),
        "converged" => is_converged(fit),
        "iterations" => niterations(fit),
        "original_row" => copy(prepared.prepared.original_row),
        "x_observed" => collect(prepared.prepared.observed_x),
        "y_observed" => collect(prepared.prepared.observed_y),
        "row_state" => string.(prepared.prepared.row_state),
        "imputed_all" => table_columns(table_all),
        "imputed_missing" => table_columns(table_missing),
        "imputed_no_se" => table_columns(table_no_se),
        "native_theta" => Float64.(ref["theta"]),
        "native_uncertainty_theta" => Float64.(native_uncertainty["theta"]),
        "native_uncertainty_covariance" => native_uncertainty["covariance"],
    )
    case["native_theta_max_abs"] = maximum(abs.(case["theta"] .- case["native_theta"]))
    receipt["cases"][kind] = case
    println("public=", kind, " converged=", case["converged"],
            " gradient=", maximum(abs, gradient),
            " native_theta_max_abs=", case["native_theta_max_abs"])
end
receipt["seconds"] = time() - started
receipt["source_sha256_after"] = source_manifest()
receipt["source_unchanged"] = receipt["source_sha256_before"] == receipt["source_sha256_after"]

open(output_path, "w") do io
    TOML.print(io, receipt; sorted = true)
end
receipt["source_unchanged"] || error("source changed during fit")
println("JOINT_FRONTEND_FITS_EXECUTED cases=2 rows=320")
