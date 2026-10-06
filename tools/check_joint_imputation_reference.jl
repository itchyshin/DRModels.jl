# Same-parameter uncertainty and accessor-shape oracle. No optimization.
using DRModels, LinearAlgebra, SHA, TOML
length(ARGS)==3 || error("usage: check_joint_imputation_reference.jl DATA_TOML UNCERTAINTY_TOML NEW_RECEIPT_TOML")
data_path,reference_path,output_path=abspath.(ARGS)
isfile(output_path) && error("refusing stale output")
BLAS.set_num_threads(1)
Threads.nthreads()==1 && BLAS.get_num_threads()==1 || error("wrong thread budget")
source_root=dirname(pathof(DRModels))
manifest()=Dict(relpath(joinpath(dir,file),source_root)=>bytes2hex(sha256(read(joinpath(dir,file)))) for (dir,_,files) in walkdir(source_root) for file in files)
before=manifest(); data=TOML.parsefile(data_path); reference=TOML.parsefile(reference_path)
receipt=Dict{String,Any}("scope"=>"Same-parameter native imputation summaries, supplied covariance; no fit/covariance-estimation claim", "source_sha256"=>before,
 "data_sha256"=>bytes2hex(sha256(read(data_path))),"reference_sha256"=>bytes2hex(sha256(read(reference_path))),
 "runner_sha256"=>bytes2hex(sha256(read(@__FILE__))),"julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),"cases"=>Dict{String,Any}())
started=time()
for kind in ("gaussian","bernoulli")
 d=data[kind]; r=reference[kind]; n=length(d["x"])
 x=Union{Missing,Float64}[d["x_observed"][i] ? d["x"][i] : missing for i in 1:n]
 y=Union{Missing,Float64}[d["y_observed"][i] ? d["y"][i] : missing for i in 1:n]
 X=hcat(ones(n),Float64.(d["z"]))
 model=prepared_joint_model(y,x,X,ones(n,1),X;predictor=Symbol(kind),original_row=Int.(d["original_row"]))
 theta=Float64.(r["theta"]);V=reduce(vcat,permutedims.(Float64.(row) for row in r["covariance"]))
 u=DRModels._joint_imputation_uncertainty(model,theta,V)
 # Construct an explicitly non-optimized reference result to exercise the
 # public accessor at the native theta. Optimizer status is not a gate for SEs.
 moments=prepared_joint_conditional_moments(model,theta)
 blocks=Pair{Symbol,UnitRange{Int}}[:mu=>1:3,:sigma=>4:4,:mi_x=>5:6]
 names=Pair{Symbol,Vector{String}}[:mu=>["(Intercept)","z","mi(x)"],:sigma=>["(Intercept)"],:mi_x=>["(Intercept)","z"]]
 if kind=="gaussian";push!(blocks,:logsd_mi_x=>7:7);push!(names,:logsd_mi_x=>["log_sd"]);end
 base=DRModels.DrmFit(Gaussian(),blocks,names,theta,V,-prepared_joint_nll(model,theta),count(model.observed_y),false,
   Dict{Symbol,Vector{Float64}}(),Dict{Symbol,Vector{Float64}}(),Dict{Symbol,Vector{Float64}}())
 meta=DRModels.JointMissingMetadata(Symbol(kind),copy(model.original_row),copy(model.row_state),copy(model.observed_y),copy(model.observed_x),
   Float64.(moments.mean),Float64.(moments.variance),copy(moments.status),n,count(==(:x_observed_y_missing),model.row_state),
   :not_computed,:reference_not_optimized,:observed_information_inverse)
 fitted=PreparedJointFit(base,model,meta)
 table_all=imputed(fitted;rows=:all);selected=imputed(fitted);omitted=imputed(fitted;rows=:all,se=false)
 receipt["cases"][kind]=Dict("theta"=>theta,"covariance"=>r["covariance"],"mean"=>table_all.estimate,
  "std_error"=>[ismissing(v) ? 0.0 : v for v in table_all.std_error],"se_available"=>.!ismissing.(table_all.std_error),
  "observed"=>collect(table_all.observed),"original_row"=>table_all.original_row,"model_row"=>table_all.model_row,
  "source"=>table_all.source,"uncertainty_status"=>table_all.uncertainty_status,"conditional_variance"=>u.conditional_variance,
  "selected_model_row"=>selected.model_row,"no_se_all_missing"=>all(ismissing,omitted.std_error),
  "no_se_status"=>omitted.uncertainty_status,"parameter_variance"=>u.parameter_variance)
end
receipt["seconds"]=time()-started;receipt["source_unchanged"]=before==manifest()
open(output_path,"w") do io;TOML.print(io,receipt;sorted=true);end
receipt["source_unchanged"] || error("source changed during check")
println("JOINT_IMPUTATION_REFERENCE_EXECUTED cases=2 rows=320")
