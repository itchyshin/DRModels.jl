using Test, DRModels, TOML, LinearAlgebra, ForwardDiff

function two_joint_payload()
    d=TOML.parsefile(joinpath(@__DIR__,"fixtures/joint_missing_predictor/two_gaussian_reference.toml"))
    n=length(d["y"])
    Dict{String,Any}("schema"=>"joint_missing_two_gaussian_v1","predictor"=>"gaussian",
        "variable"=>["x1","x2"],"y"=>d["y"],"x"=>hcat(d["x1"],d["x2"]),
        "observed_y"=>d["y_observed"],"observed_x"=>hcat(d["x1_observed"],d["x2_observed"]),
        "X_mu"=>hcat(ones(n),d["x1"],d["x2"],d["z"]),"X_sigma"=>ones(n,1),
        "X_predictor"=>[hcat(ones(n),d["z"]),hcat(ones(n),d["z"])],
        "mu_col"=>[2,3],"mu_names"=>["(Intercept)","mi(x1)","mi(x2)","z"],
        "sigma_names"=>["(Intercept)"],"predictor_names"=>[["(Intercept)","z"],["(Intercept)","z"]],
        "original_row"=>d["original_row"],"options"=>Dict("g_tol"=>1e-8))
end

@testset "two Gaussian primitive preparation" begin
    p=two_joint_payload();prep=DRModels._prepare_joint_bridge(p)
    @test prep.model isa DRModels.PreparedTwoJointGaussianModel
    @test prep.permutation==[1,3,4,2,5,6,7,8,9,10,11]
    @test prep.model.predictor_variables==(:x1,:x2)
    @test prep.model.observed_x==(BitVector(p["observed_x"][:,1]),BitVector(p["observed_x"][:,2]))
    @test prep.model.original_row==p["original_row"]
    changed=deepcopy(p)
    for j=1:2
        ids=findall(.!p["observed_x"][:,j]);changed["x"][ids,j].=999
        changed["X_mu"][ids,p["mu_col"][j]].=-77
    end
    @test isequal(DRModels._prepare_joint_bridge(changed).model.x,prep.model.x)
    for (key,val) in (("variable",["x1","x1"]),("predictor","bernoulli"),
        ("mu_col",[2,2]),("mu_col",[true,3]),("mu_col",[2.0,3.0]),("mu_col",[0,3]),
        ("observed_x",fill(2,160,2)),("x",zeros(160,3)),("X_predictor",[ones(160,1)]),
        ("predictor_names",[["a","a"],["a","b"]]),("options",Dict("REML"=>true)),
        ("original_row",fill(1,160)),("variable",["x1","x2","x3"]))
        bad=deepcopy(p);bad[key]=val
        @test_throws ArgumentError DRModels._prepare_joint_bridge(bad)
    end
    bad=deepcopy(p);bad["X_mu"][1,3]+=1
    @test_throws ArgumentError DRModels._prepare_joint_bridge(bad)
    bad=deepcopy(p);bad["extra"]=1
    @test_throws ArgumentError DRModels._prepare_joint_bridge(bad)
    # Different design widths and a non-monotonic marker order.
    q=deepcopy(p);q["X_predictor"][2]=ones(160,1);q["predictor_names"][2]=["(Intercept)"]
    q["X_mu"]=p["X_mu"][:,[3,4,1,2]];q["mu_names"]=p["mu_names"][[3,4,1,2]];q["mu_col"]=[4,1]
    @test DRModels._prepare_joint_bridge(q).permutation==[4,1,2,3,5,6,7,8,9,10]
end

@testset "two Gaussian primitive fit and native order" begin
    p=two_joint_payload();prep=DRModels._prepare_joint_bridge(p);out=DRModels.drm_bridge_joint(p)
    d=TOML.parsefile(joinpath(@__DIR__,"fixtures/joint_missing_predictor/two_gaussian_reference.toml"))
    @test out["schema"]=="joint_missing_two_gaussian_result_v1"
    @test out["optimizer_status"]=="converged"
    @test out["covariance_status"]=="observed_information_inverse"
    back=invperm(prep.permutation);theta=out["coefficients"][back]
    @test maximum(abs,theta-d["theta"])<4e-6
    H=ForwardDiff.hessian(t->prepared_joint_nll(prep.model,t),theta)
    @test H*out["vcov"][back,back]≈Matrix{Float64}(I,length(theta),length(theta)) atol=1e-8
    @test out["coefficient_blocks"]==["mu","mu","mu","mu","sigma","mi_x1","mi_x1","logsd_mi_x1","mi_x2","mi_x2","logsd_mi_x2"]
    @test out["coefficient_terms"][1:4]==p["mu_names"]
    @test out["observed_x"]==p["observed_x"]
    for j=1:2
        tab=out["imputation"]["x$j"]
        @test tab["original_row"]==p["original_row"]
        @test tab["observed"]==p["observed_x"][:,j]
        @test maximum(abs,tab["estimate"]-d["imputed$(j)_mean"])<4e-6
        ids=findall(.!p["observed_x"][:,j])
        @test maximum(abs,tab["std_error"][ids]-d["imputed$(j)_se"][ids])<4e-6
        @test all(isnan,tab["std_error"][findall(p["observed_x"][:,j])])
    end
    # Fit an actually non-monotonic marker-column layout, then independently
    # reconstruct the raw Hessian; covariance must reorder BOTH axes.
    q=deepcopy(p);q["X_mu"]=p["X_mu"][:,[3,4,1,2]]
    q["mu_names"]=p["mu_names"][[3,4,1,2]];q["mu_col"]=[4,1]
    prepq=DRModels._prepare_joint_bridge(q);outq=DRModels.drm_bridge_joint(q)
    backq=invperm(prepq.permutation);tq=outq["coefficients"][backq]
    Hq=ForwardDiff.hessian(t->prepared_joint_nll(prepq.model,t),tq)
    @test maximum(abs,ForwardDiff.gradient(t->prepared_joint_nll(prepq.model,t),tq))<1e-6
    @test Hq*outq["vcov"][backq,backq]≈Matrix{Float64}(I,length(tq),length(tq)) atol=1e-8
    @test outq["coefficient_terms"][1:4]==q["mu_names"]
    @test size(out["conditional_covariance"])==(160,2,2)
    @test any(!iszero,out["conditional_covariance"][:,1,2])
end
println("TWO_BRIDGE_NEW_PASS")
