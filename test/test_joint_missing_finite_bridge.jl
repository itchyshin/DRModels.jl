using Test, DRModels, TOML, LinearAlgebra
function finite_bridge_payload(kind)
    c=TOML.parsefile(joinpath(@__DIR__,"..","docs/dev-log/evidence/julia-r-parity/finite-state/finite-reference-003.toml"))[kind]
    mat(v)=reduce(vcat,permutedims.(Float64.(r) for r in v))
    Xstate=mat(c["state_design"]);Xmu=copy(Xstate[1:c["K"]:end,:])
    for i in 1:c["n"]
        if c["observed_x"][i]; Xmu[i,:].=Xstate[(i-1)*c["K"]+c["x"][i],:];end
    end
    Dict{String,Any}("schema"=>"joint_missing_finite_v1","predictor"=>kind,"variable"=>"x",
        "levels"=>c["levels"],"y"=>c["y"],"x"=>c["x"],"observed_y"=>c["observed_y"],"observed_x"=>c["observed_x"],
        "X_mu"=>Xmu,"X_mu_state"=>Xstate,"state_layout"=>"row_then_state","X_sigma"=>mat(c["X_sigma"]),"X_predictor"=>mat(c["X_predictor"]),
        "mu_names"=>c["mu_names"],"sigma_names"=>c["sigma_names"],"predictor_names"=>c["predictor_names"],"original_row"=>c["original_row"],"options"=>Dict("g_tol"=>1e-8))
end
@testset "finite-state primitive transport" begin
    for kind in ("ordinal","categorical")
        payload=finite_bridge_payload(kind);p=DRModels._prepare_joint_bridge(payload)
        @test p.model isa DRModels.PreparedFiniteJointModel
        @test p.model.levels==payload["levels"]
        @test p.model.observed_x==payload["observed_x"]
        expected=kind=="ordinal" ? [1,2,3,4,5,7,8,6] : collect(1:9)
        @test p.permutation==expected
        bad=deepcopy(payload);bad["state_layout"]="state_then_row"
        @test_throws ArgumentError DRModels._prepare_joint_bridge(bad)
        bad=deepcopy(payload);bad["X_mu_state"][Int(bad["x"][1]),1]+=1
        @test_throws ArgumentError DRModels._prepare_joint_bridge(bad)
        bad=deepcopy(payload);bad["x"][1]=4
        @test_throws ArgumentError DRModels._prepare_joint_bridge(bad)
        bad=deepcopy(payload);bad["levels"][2]=bad["levels"][1]
        @test_throws ArgumentError DRModels._prepare_joint_bridge(bad)
    end
end
@testset "finite-state fit result retains raw order and conditional outputs" begin
    for kind in ("ordinal","categorical")
        p=finite_bridge_payload(kind);r=DRModels.drm_bridge_joint(p)
        @test r["schema"]=="joint_missing_finite_result_v1"
        @test r["predictor_levels"]==p["levels"]
        @test size(r["conditional_probabilities"])==(180,3)
        @test maximum(abs.(sum(r["conditional_probabilities"],dims=2).-1))<1e-12
        @test r["original_row"]==collect(1:180)
        @test r["observed_x"]==p["observed_x"]
        @test length(r["coef_names"])==size(r["vcov"],1)==size(r["vcov"],2)
        @test r["covariance_status"]=="observed_information_inverse"
        if kind=="ordinal"
            @test r["coefficient_blocks"]==vcat(fill("mu",4),["sigma","rawcut_x","rawcut_x","mi_x"])
            raw=r["coefficients"][6:7]
            @test r["ordinal"]["cutpoints"]≈[raw[1],raw[1]+exp(raw[2])]
            @test all(r["imputation"]["se_available"][.!p["observed_x"]])
        else
            @test r["coefficient_terms"][6:end]==["grass:(Intercept)","grass:z","wetland:(Intercept)","wetland:z"]
            @test !any(r["imputation"]["se_available"])
            @test all(==("route_conditional_se_unavailable"),r["imputation"]["uncertainty_status"][.!p["observed_x"]])
        end
    end
end
