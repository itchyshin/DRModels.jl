# Bounded positive-fit checks on both frozen, nondegenerate native datasets.
using DRModels, ForwardDiff, LinearAlgebra, SHA, TOML

length(ARGS) == 2 || error("usage: check_joint_predictor_fit.jl REFERENCE_TOML NEW_RECEIPT_TOML")
reference_path, output_path = abspath.(ARGS)
isfile(output_path) && error("refusing stale output")
BLAS.set_num_threads(1)
Threads.nthreads() == 1 && BLAS.get_num_threads() == 1 || error("wrong resource budget")
source_root = dirname(pathof(DRModels))
manifest() = Dict(relpath(joinpath(dir, name), source_root) => bytes2hex(sha256(read(joinpath(dir, name))))
                  for (dir, _, files) in walkdir(source_root) for name in files)
before = manifest()
reference = TOML.parsefile(reference_path)
receipt = Dict{String,Any}("scope" => "Two prepared ML fits; no R bridge or interval parity claim",
    "reference_sha256" => bytes2hex(sha256(read(reference_path))),
    "runner_sha256" => bytes2hex(sha256(read(@__FILE__))), "source_sha256" => before,
    "runtime" => Dict("julia_version" => string(VERSION), "julia_threads" => Threads.nthreads(),
                       "blas_threads" => BLAS.get_num_threads(), "loaded_source" => pathof(DRModels)),
    "cases" => Dict{String,Any}())
started = time()
for kind in ("gaussian", "bernoulli")
    ref = reference[kind]; n = length(ref["original_row"])
    x = Union{Missing,Float64}[ref["x_observed"][i] ? ref["x"][i] : missing for i in 1:n]
    y = Union{Missing,Float64}[ref["y_observed"][i] ? ref["y"][i] : missing for i in 1:n]
    X = hcat(ones(n), Float64.(ref["z"]))
    model = prepared_joint_model(y, x, X, ones(n, 1), X; predictor = Symbol(kind),
        mu_names = ["(Intercept)", "z"], sigma_names = ["(Intercept)"],
        predictor_names = ["(Intercept)", "z"], original_row = Int.(ref["original_row"]))
    initial = DRModels.prepared_joint_initial(model)
    fitted = fit_prepared_joint(model)
    theta = copy(fitted.fit.theta)
    objective = t -> prepared_joint_nll(fitted.prepared, t)
    gradient = ForwardDiff.gradient(objective, theta)
    H = ForwardDiff.hessian(objective, theta)
    V = DRModels.vcov(fitted.fit)
    prior_objective = fitted.fit.nll(theta)
    model.Xmu[1, 1] += 100
    model.x[1] = 999
    isolated = fitted.fit.nll(theta) == prior_objective
    summary = joint_missing_summary(fitted)
    receipt["cases"][kind] = Dict("initial_theta" => initial, "theta" => theta,
        "native_theta" => Float64.(ref["theta"]), "nll" => objective(theta),
        "reported_loglik" => DRModels.loglik(fitted.fit), "gradient" => gradient,
        "row_loglik" => Float64.(prepared_joint_rowloglik(fitted.prepared, theta)),
        "x_observed" => collect(fitted.prepared.observed_x),
        "y_observed" => collect(fitted.prepared.observed_y),
        "hessian" => [collect(row) for row in eachrow(H)],
        "covariance" => [collect(row) for row in eachrow(V)],
        "optimizer_status" => string(summary.optimizer_status),
        "covariance_status" => string(summary.covariance_status),
        "uncertainty_status" => string(summary.uncertainty_status),
        "nobs" => DRModels.nobs(fitted.fit), "all_rows" => summary.all_rows,
        "original_row" => summary.original_row, "snapshot_isolated" => isolated)
    println(kind, " optimizer=", summary.optimizer_status, " gradient=", maximum(abs, gradient))
end
receipt["seconds"] = time() - started
receipt["source_unchanged"] = before == manifest()
open(output_path, "w") do io
    TOML.print(io, receipt; sorted = true)
end
receipt["source_unchanged"] || error("source changed during fit")
println("JOINT_FITS_EXECUTED cases=2")
