# Retain native-default fitting discrepancies; never replace the comparator optimum.
using DRModels, LinearAlgebra, ForwardDiff, SHA, TOML
root = dirname(@__DIR__)
length(ARGS)==1 || error("usage: check_finite_joint_fit.jl NEW_RECEIPT_TOML")
isfile(ARGS[1]) && error("refusing stale receipt")
realpath(dirname(pathof(DRModels))) == realpath(joinpath(root,"src")) || error("wrong loaded source")
BLAS.set_num_threads(1)
Threads.nthreads()==1 || error("wrong resource budget")
manifest()=Dict(relpath(joinpath(d,f),root)=>bytes2hex(sha256(read(joinpath(d,f)))) for (d,_,fs) in walkdir(joinpath(root,"src")) for f in fs)
before=manifest()
reference_path=joinpath(root,"docs/dev-log/evidence/julia-r-parity/finite-state/finite-reference-003.toml")
r=TOML.parsefile(reference_path)
mat(rows)=reduce(vcat,permutedims.(Float64.(v) for v in rows))
receipt=Dict{String,Any}("scope"=>"prepared default-fit comparison only; no formula/bridge/performance claim",
    "source_before"=>before,"reference_sha256"=>bytes2hex(sha256(read(reference_path))),
    "runner_sha256"=>bytes2hex(sha256(read(@__FILE__))),"tolerance"=>4e-6,
    "runtime"=>Dict("loaded_source"=>pathof(DRModels),"julia_version"=>string(VERSION),"julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads()),
    "cases"=>Dict{String,Any}())
started=time()
for kind in ("ordinal","categorical")
    c=r[kind];n,K=c["n"],c["K"];p=length(c["mu_names"])
    A=Array{Float64}(undef,n,K,p)
    for i in 1:n,k in 1:K; A[i,k,:].=c["state_design"][(i-1)*K+k]; end
    y=Union{Missing,Float64}[c["observed_y"][i] ? c["y"][i] : missing for i in 1:n]
    x=Union{Missing,String}[c["observed_x"][i] ? c["levels"][c["x"][i]] : missing for i in 1:n]
    m=prepared_joint_model(y,x,A,mat(c["X_sigma"]),mat(c["X_predictor"]);
        predictor=Symbol(kind),levels=c["levels"],mu_names=c["mu_names"],sigma_names=c["sigma_names"],predictor_names=c["predictor_names"])
    fit=fit_prepared_joint(m)
    native=c["points"][1];tab=imputed(fit;rows=:all)
    errs=Dict("theta"=>maximum(abs.(fit.fit.theta-native["theta"])),
        "loglik"=>abs(fit.fit.loglik+native["nll"]),"prediction"=>maximum(abs.(fit.fit.means[:mu]-native["prediction"])),
        "imputation"=>maximum(abs.(tab.estimate-native["estimate"])))
    if kind=="ordinal"
        ids=findall(.!m.observed_x)
        errs["conditional_sd"]=maximum(abs.(Float64.(tab.std_error[ids])-native["conditional_sd"][ids]))
    end
    output=Dict("theta"=>fit.fit.theta,"loglik"=>fit.fit.loglik,"prediction"=>fit.fit.means[:mu],
        "imputation"=>tab.estimate,"imputation_status"=>tab.uncertainty_status,
        "imputation_sd"=>[ismissing(v) ? 0.0 : v for v in tab.std_error],
        "imputation_sd_available"=>[!ismissing(v) for v in tab.std_error],
        "covariance"=>[collect(v) for v in eachrow(fit.fit.vcov)],
        "optimizer_status"=>string(fit.metadata.optimizer_status),"covariance_status"=>string(fit.metadata.covariance_status),
        "gradient_max"=>maximum(abs.(ForwardDiff.gradient(fit.fit.nll,fit.fit.theta))),"errors"=>errs,
        "parity_pass"=>all(<=(4e-6),values(errs)))
    receipt["cases"][kind]=output
    println(kind," parity=",output["parity_pass"]," errors=",errs)
end
receipt["source_after"]=manifest();receipt["source_unchanged"]=before==receipt["source_after"];receipt["seconds"]=time()-started
open(ARGS[1],"w") do io;TOML.print(io,receipt;sorted=true);end
receipt["source_unchanged"] || error("source changed during fit")
println("FINITE_JOINT_FIT_MEASURED")
