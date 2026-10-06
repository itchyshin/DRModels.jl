using Test, DRModels, LinearAlgebra, TOML, ForwardDiff
# BLAS=1 is the suite's numerical invariant; runtests.jl pins it once at start
# and guards it per file, so this file no longer sets it. The check is a real
# precondition (standalone runs need OPENBLAS_NUM_THREADS=1) and is reported,
# not raised, so it cannot abort the files after this one. The Julia thread
# count is deliberately not asserted: this file exercises serial code, and the
# local budget is JULIA_NUM_THREADS=4 (CI runs at 1).
@testset "thread budget" begin
    @test BLAS.get_num_threads() == 1
end
@testset "prepared joint bridge API" begin
    @test isdefined(DRModels,:drm_bridge_joint)
    @test isdefined(DRModels,:_prepare_joint_bridge)
end

ref=TOML.parsefile(joinpath(@__DIR__,"fixtures/joint_missing_predictor/native_reference.toml"))
function joint_payload(kind)
    d=ref[kind];n=length(d["x"])
    Dict{String,Any}("schema"=>"joint_missing_v1","predictor"=>kind,"variable"=>"body_mass",
        "y"=>copy(d["y"]),"x"=>copy(d["x"]),"observed_y"=>copy(d["y_observed"]),"observed_x"=>copy(d["x_observed"]),
        "X_mu"=>hcat(ones(n),d["x"],d["z"]),"X_sigma"=>ones(n,1),"X_predictor"=>hcat(ones(n),d["z"]),
        "mu_col"=>2,"mu_names"=>["(Intercept)","mi(body_mass)","z"],"sigma_names"=>["(Intercept)"],
        "predictor_names"=>["(Intercept)","z"],"original_row"=>copy(d["original_row"]),"options"=>Dict("g_tol"=>1e-8))
end
@testset "prepared joint bridge validation and masks" begin
    payload=joint_payload("gaussian");prepared=DRModels._prepare_joint_bridge(payload)
    @test prepared.model.observed_y==ref["gaussian"]["y_observed"]
    @test prepared.model.observed_x==ref["gaussian"]["x_observed"]
    @test prepared.permutation==[1,3,2,4,5,6,7]
    @test prepared.model.mu_names==["(Intercept)","z"]
    ids=findall(.!prepared.model.observed_x)
    @test all(ismissing,prepared.model.x[ids])
    changed=deepcopy(payload);
    changed["x"][ids]=fill(999.0,length(ids));changed["X_mu"][ids,2].=999
    @test isequal(DRModels._prepare_joint_bridge(changed).model.x,prepared.model.x)
    for (key,value) in (("schema","wrong"),("mu_col",0),("mu_col",1.5),("observed_x",fill(2,160)),
                        ("original_row",fill(1,160)),("options",Dict("method"=>"REML")))
        bad=deepcopy(payload);bad[key]=value
        @test_throws ArgumentError DRModels._prepare_joint_bridge(bad)
    end
    bad=deepcopy(payload);bad["unhandled"]=true
    @test_throws ArgumentError DRModels._prepare_joint_bridge(bad)
    bad=deepcopy(payload);bad["X_mu"][1,2]+=1
    @test_throws ArgumentError DRModels._prepare_joint_bridge(bad)
end
@testset "two joint bridge fits retain native ordering and summaries" begin
    for kind in ("gaussian","bernoulli")
        payload=joint_payload(kind);out=DRModels.drm_bridge_joint(payload)
        @test out["schema"]=="joint_missing_result_v1"
        @test out["converged"]
        @test out["coefficient_terms"][1:3]==payload["mu_names"]
        @test out["coefficient_blocks"][5:6]==fill("mi_body_mass",2)
        # This checks transport ordering at the Julia optimum, not native
        # optimizer parity. Its required native4e-6 gate is kept separate.
        prepared=DRModels._prepare_joint_bridge(payload)
        back=invperm(prepared.permutation)
        theta=out["coefficients"][back]
        H=ForwardDiff.hessian(t->prepared_joint_nll(prepared.model,t),theta)
        @test maximum(abs,ForwardDiff.gradient(t->prepared_joint_nll(prepared.model,t),theta))<1e-6
        @test H*out["vcov"][back,back]≈Matrix{Float64}(I,length(theta),length(theta)) atol=1e-8
        @test out["coefficient_blocks"][1:3]==fill("mu",3)
        @test size(out["vcov"])==(length(out["coefficients"]),length(out["coefficients"]))
        @test out["imputation"]["original_row"]==payload["original_row"]
        @test all(==("body_mass"),out["imputation"]["variable"])
        @test all(==("ok"),out["imputation"]["uncertainty_status"])
        @test all(isnan,out["imputation"]["std_error"][findall(payload["observed_x"])])
        @test out["optimizer_status"]=="converged"
    end
end
println("JOINT_BRIDGE_KERNEL_PASS")
