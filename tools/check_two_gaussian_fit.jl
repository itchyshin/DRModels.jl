# Same-point and independently fitted evidence; native stopping losses are retained.
using DRModels, ForwardDiff, LinearAlgebra, SHA, TOML
length(ARGS)==2 || error("usage: check_two_gaussian_fit.jl REFERENCE_TOML NEW_TOML")
refpath,outpath=abspath.(ARGS)
isfile(outpath) && error("refusing stale output")
BLAS.set_num_threads(1)
Threads.nthreads()==1 && BLAS.get_num_threads()==1 || error("resource budget")
root=dirname(pathof(DRModels))
manifest()=Dict(relpath(joinpath(d,f),root)=>bytes2hex(sha256(read(joinpath(d,f)))) for (d,_,fs) in walkdir(root) for f in fs)
before=manifest();ref=TOML.parsefile(refpath);n=length(ref["original_row"])
decode(k) = Union{Missing,Float64}[ref[k*"_observed"][i] ? ref[k][i] : missing for i in 1:n]
x=hcat(decode("x1"),decode("x2"));X=hcat(ones(n),Float64.(ref["z"]))
model=prepared_joint_model(decode("y"),x,X,ones(n,1),(X,X);predictor=:gaussian,
 predictor_variables=(:x1,:x2),mu_names=["(Intercept)","z"],sigma_names=["(Intercept)"],
 predictor_names=(["(Intercept)","z"],["(Intercept)","z"]),original_row=Int.(ref["original_row"]))
rows(A)=[collect(r) for r in eachrow(A)]
function point(model,t)
 m=prepared_joint_conditional_moments(model,t)
 Dict("theta"=>t,"nll"=>prepared_joint_nll(model,t),
  "gradient"=>ForwardDiff.gradient(v->prepared_joint_nll(model,v),t),
  "row_loglik"=>prepared_joint_rowloglik(model,t),"mean"=>rows(m.mean),
  "conditional_covariance"=>[rows(m.covariance[i,:,:]) for i in 1:length(model.y)],
  "status"=>[string.(collect(r)) for r in eachrow(m.status)])
end
function table(t)
 d=Dict(String(k)=>collect(v) for (k,v) in pairs(t))
 d["std_error"]=[ismissing(v) ? 0.0 : v for v in t.std_error]
 d["se_available"]=.!ismissing.(t.std_error)
 return d
end
receipt=Dict{String,Any}("schema"=>"two_gaussian_fit_v1",
 "scope"=>"Prepared shared two-Gaussian kernel only; direct formula and R bridge admission remain required",
 "reference_sha256"=>bytes2hex(sha256(read(refpath))),"runner_sha256"=>bytes2hex(sha256(read(@__FILE__))),
 "source_before"=>before,"runtime"=>Dict("julia"=>string(VERSION),"threads"=>Threads.nthreads(),"blas"=>BLAS.get_num_threads(),"source"=>pathof(DRModels)),
 "original_row"=>model.original_row,"observed_y"=>collect(model.observed_y),
 "observed_x1"=>collect(model.observed_x[1]),"observed_x2"=>collect(model.observed_x[2]))
tick=time()
receipt["points"]=[point(model,Float64.(p["theta"])) for p in ref["points"]]
receipt["initial"]=DRModels.prepared_joint_initial(model)
f=fit_prepared_joint(model)
theta=f.fit.theta;H=ForwardDiff.hessian(t->prepared_joint_nll(f.prepared,t),theta)
fitted=point(f.prepared,theta)
merge!(fitted,Dict("loglik"=>DRModels.loglik(f.fit),"covariance"=>rows(f.fit.vcov),"hessian"=>rows(H),
 "converged"=>DRModels.is_converged(f.fit),"optimizer_status"=>String(f.metadata.optimizer_status),
 "covariance_status"=>String(f.metadata.covariance_status),"nobs"=>DRModels.nobs(f.fit),
 "imputed1"=>table(imputed(f;variable=:x1,rows=:all)),"imputed2"=>table(imputed(f;variable=:x2,rows=:all)),
 "imputed1_no_se"=>table(imputed(f;variable=:x1,rows=:all,se=false)),
 "imputed2_no_se"=>table(imputed(f;variable=:x2,rows=:all,se=false))))
receipt["fitted"]=fitted
# Accessor-only evaluation at native theta/V, independent of the Julia optimum.
Vnative=reduce(vcat,permutedims.(Float64.(r) for r in ref["covariance"]))
fixed=Dict{String,Any}("scope"=>"fixed-parameter uncertainty, not a fit")
for j in 1:2
 u=DRModels._two_joint_imputation_uncertainty(f.prepared,Float64.(ref["theta"]),Vnative;predictor_index=j)
 fixed["imputed"*string(j)]=Dict("estimate"=>u.estimate,"std_error"=>[isfinite(v) ? v : 0.0 for v in u.std_error],
  "se_available"=>isfinite.(u.std_error),"conditional_variance"=>u.conditional_variance,
  "parameter_variance"=>u.parameter_variance,"uncertainty_status"=>u.uncertainty_status)
end
receipt["fixed_native_accessor"]=fixed
objective_before=f.fit.nll(theta);model.Xmu[1,1]+=100;model.x[1,1]=999
receipt["snapshot_isolated"]=f.fit.nll(theta)==objective_before
receipt["native_errors"]=Dict("theta"=>maximum(abs.(theta-Float64.(ref["theta"]))),
 "loglik"=>abs(fitted["nll"]-ref["points"][1]["nll"]),
 "imputed1_mean"=>maximum(abs.(fitted["imputed1"]["estimate"].-ref["imputed1_mean"])),
 "imputed2_mean"=>maximum(abs.(fitted["imputed2"]["estimate"].-ref["imputed2_mean"])),
 "imputed1_se"=>maximum(abs.(fitted["imputed1"]["std_error"].-ref["imputed1_se"])),
 "imputed2_se"=>maximum(abs.(fitted["imputed2"]["std_error"].-ref["imputed2_se"])))
receipt["native_status"]=all(v->isfinite(v)&&v<=4e-6,values(receipt["native_errors"])) ? "PASS" : "FAIL"
receipt["seconds"]=time()-tick;receipt["source_after"]=manifest()
receipt["source_unchanged"]=before==receipt["source_after"]
open(outpath,"w") do io
 TOML.print(io,receipt;sorted=true)
end
receipt["source_unchanged"] || error("source changed")
println("TWO_GAUSSIAN_FIT_EXECUTED native=",receipt["native_status"])
