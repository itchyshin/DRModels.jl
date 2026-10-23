# Same-parameter comparison against independently integrated native references.
using DRModels, LinearAlgebra, SHA, TOML

length(ARGS) == 2 || error("usage: check_joint_predictor_reference.jl REFERENCE_TOML NEW_RECEIPT_TOML")
reference_path, output_path = abspath.(ARGS)
isfile(output_path) && error("refusing stale output")
BLAS.set_num_threads(1)
Threads.nthreads() == 1 && BLAS.get_num_threads() == 1 || error("wrong resource budget")
source_root = dirname(pathof(DRModels))
source_manifest() = Dict(relpath(joinpath(dir, name), source_root) => bytes2hex(sha256(read(joinpath(dir, name))))
                         for (dir, _, files) in walkdir(source_root) for name in files)
before = source_manifest()
reference = TOML.parsefile(reference_path)
receipt = Dict{String,Any}(
    "scope" => "Prepared exact joint models at frozen native parameters; not full fitting or R bridge parity",
    "reference_sha256" => bytes2hex(sha256(read(reference_path))),
    "source_sha256" => before,
    "runtime" => Dict("julia_version" => string(VERSION), "julia_threads" => Threads.nthreads(),
                       "blas_threads" => BLAS.get_num_threads(), "loaded_source" => pathof(DRModels)),
    "cases" => Dict{String,Any}())
started = time()
for kind in ("gaussian", "bernoulli")
    ref = reference[kind]
    n = length(ref["original_row"])
    x = Union{Missing,Float64}[ref["x_observed"][i] ? ref["x"][i] : missing for i in 1:n]
    y = Union{Missing,Float64}[ref["y_observed"][i] ? ref["y"][i] : missing for i in 1:n]
    X = hcat(ones(n), Float64.(ref["z"]))
    model = prepared_joint_model(y, x, X, ones(n, 1), X; predictor = Symbol(kind),
        mu_names = ["(Intercept)", "z"], sigma_names = ["(Intercept)"],
        predictor_names = ["(Intercept)", "z"], original_row = Int.(ref["original_row"]))
    theta = Float64.(ref["theta"])
    rowloglik = prepared_joint_rowloglik(model, theta)
    moments = prepared_joint_conditional_moments(model, theta)
    receipt["cases"][kind] = Dict(
        "theta" => theta, "original_row" => model.original_row,
        "x_observed" => collect(model.observed_x), "y_observed" => collect(model.observed_y),
        "status" => string.(moments.status), "nobs" => count(model.observed_y),
        "row_loglik" => rowloglik, "nll" => prepared_joint_nll(model, theta),
        "conditional_mean" => moments.mean, "conditional_variance" => moments.variance)
end
receipt["seconds"] = time() - started
receipt["source_unchanged"] = before == source_manifest()
open(output_path, "w") do io
    TOML.print(io, receipt; sorted = true)
end
receipt["source_unchanged"] || error("source changed during evaluation")
println("JOINT_REFERENCE_EXECUTED cases=2 rows=320")
