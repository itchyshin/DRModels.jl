# Fixed-parameter native-R and independent finite-mixture reference; not fit parity.
using DRModels, ForwardDiff, LinearAlgebra, SHA, TOML, Test
root = dirname(@__DIR__)
reference_path = joinpath(root, "docs/dev-log/evidence/julia-r-parity/finite-state/finite-reference-003.toml")
output_path = isempty(ARGS) ? joinpath(root, "docs/dev-log/evidence/julia-r-parity/finite-state/finite-julia-001.toml") : abspath(ARGS[1])
isfile(output_path) && error("refusing stale receipt")
BLAS.set_num_threads(1)
Threads.nthreads() == 1 && BLAS.get_num_threads() == 1 || error("wrong resource budget")
realpath(dirname(pathof(DRModels))) == realpath(joinpath(root, "src")) || error("wrong loaded source")
manifest() = Dict(relpath(joinpath(dir, name), root) => bytes2hex(sha256(read(joinpath(dir, name))))
    for (dir, _, files) in walkdir(joinpath(root, "src")) for name in files)
before = manifest()
reference = TOML.parsefile(reference_path)
reference["source_json_sha256"] == bytes2hex(sha256(read(joinpath(dirname(reference_path), "finite-native-003.json")))) || error("reference source mismatch")
mat(rows) = reduce(vcat, permutedims.(Float64.(r) for r in rows))
receipt = Dict{String,Any}("scope" => "fixed-parameter likelihood, AD derivatives, conditional states; not full fitting/frontend parity",
    "source_before" => before, "reference_sha256" => bytes2hex(sha256(read(reference_path))),
    "runner_sha256" => bytes2hex(sha256(read(@__FILE__))),
    "runtime" => Dict("julia_version" => string(VERSION), "julia_threads" => Threads.nthreads(),
                      "blas_threads" => BLAS.get_num_threads(), "loaded_source" => pathof(DRModels)),
    "cases" => Dict{String,Any}())
started = time()
@testset "finite-state frozen native reference" begin
    for kind in ("ordinal", "categorical")
        c = reference[kind]; n, K = c["n"], c["K"]; p = length(c["mu_names"])
        states = Array{Float64}(undef, n, K, p)
        for i in 1:n, k in 1:K
            states[i, k, :] .= c["state_design"][(i-1)*K+k]
        end
        y = Union{Missing,Float64}[c["observed_y"][i] ? c["y"][i] : missing for i in 1:n]
        x = Union{Missing,String}[c["observed_x"][i] ? c["levels"][c["x"][i]] : missing for i in 1:n]
        model = prepared_joint_model(y, x, states, mat(c["X_sigma"]), mat(c["X_predictor"]);
            predictor=Symbol(kind), levels=c["levels"], variable=:x,
            mu_names=c["mu_names"], sigma_names=c["sigma_names"], predictor_names=c["predictor_names"], original_row=c["original_row"])
        result = Any[]
        for point in c["points"]
            theta = Float64.(point["theta"]); fn = t -> prepared_joint_nll(model, t)
            value = fn(theta); grad = ForwardDiff.gradient(fn, theta); hess = ForwardDiff.hessian(fn, theta)
            ll = prepared_joint_rowloglik(model, theta); moments = prepared_joint_conditional_moments(model, theta)
            means = DRModels._finite_joint_state_means(model, theta[1:p])
            prediction = vec(sum(moments.probabilities .* means; dims=2))
            nll_error = abs(value-point["nll"]); gradient_error = maximum(abs.(grad-point["gradient"]))
            @test nll_error <= 1e-8
            @test gradient_error <= 1e-6
            @test all(isfinite, hess)
            @test maximum(abs.(ll-point["rowloglik"])) <= 1e-9
            @test maximum(abs.(moments.probabilities-mat(point["probabilities"]))) <= 1e-9
            @test maximum(abs.(moments.mean-point["estimate"])) <= 1e-9
            @test maximum(abs.(prediction-point["prediction"])) <= 1e-9
            if kind == "ordinal"
                @test maximum(abs.(sqrt.(moments.variance)-point["conditional_sd"])) <= 1e-9
            else
                @test all(isnan, moments.variance)
            end
            push!(result, Dict("theta"=>theta,"nll"=>value,"gradient"=>grad,"rowloglik"=>ll,
                "probabilities"=>[collect(r) for r in eachrow(moments.probabilities)],
                "estimate"=>moments.mean,"prediction"=>prediction,"nll_error"=>nll_error,
                "gradient_error"=>gradient_error,"hessian_finite"=>all(isfinite,hess)))
        end
        receipt["cases"][kind] = Dict("points"=>result,"n"=>n,"K"=>K)
    end
end
receipt["source_after"] = manifest()
receipt["source_unchanged"] = before == receipt["source_after"]
receipt["seconds"] = time()-started
open(output_path,"w") do io
    TOML.print(io,receipt;sorted=true)
end
receipt["source_unchanged"] || error("source changed during check")
println("FINITE_JOINT_NATIVE_PASS")
