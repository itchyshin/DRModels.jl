using Test, DRModels, LinearAlgebra, ForwardDiff
# BLAS=1 is the suite's numerical invariant; runtests.jl pins it once at start
# and guards it per file, so this file no longer sets it. The check is a real
# precondition (standalone runs need OPENBLAS_NUM_THREADS=1) and is reported,
# not raised, so it cannot abort the files after this one. The Julia thread
# count is deliberately not asserted: this file exercises serial code, and the
# local budget is JULIA_NUM_THREADS=4 (CI runs at 1).
@testset "thread budget" begin
    @test BLAS.get_num_threads() == 1
end
@testset "native-shaped imputation uncertainty API" begin
    @test isdefined(DRModels, :imputed)
    @test isdefined(DRModels, :_joint_imputation_uncertainty)
end

@testset "Gaussian prediction error includes parameter uncertainty" begin
    z = [-0.6, 0.3, 0.8, -0.1]
    x = Union{Missing,Float64}[0.8, missing, 1.0, missing]
    y = Union{Missing,Float64}[1.7, -0.2, missing, missing]
    X = hcat(ones(4),z)
    model = prepared_joint_model(y,x,X,ones(4,1),X;predictor=:gaussian)
    theta = [0.2,-0.35,0.7,0.1,0.0,0.25,log(0.9)]
    V = Matrix{Float64}(I,7,7)*0.01
    out = DRModels._joint_imputation_uncertainty(model,theta,V)
    mom = prepared_joint_conditional_moments(model,theta)
    # Independent analytic Jacobian from the Gaussian conditional-mode equation.
    S,T = exp(2theta[4]),exp(2theta[7]); b=theta[3];D=S+b*b*T;g=b*T/D;w=S/D
    r=y[2]-dot(X[2,:],theta[1:2]);u=mom.mean[2];m=dot(X[2,:],theta[5:6])
    J=vcat(-g*X[2,:],(T/D)*(r-2b*u),-2g*(r-b*u),w*X[2,:],2w*(u-m))
    @test out.parameter_variance[2] ≈ dot(J,V*J) atol=1e-12
    @test out.std_error[2]^2 ≈ mom.variance[2]+dot(J,V*J) atol=1e-12
    @test out.std_error[2] > sqrt(mom.variance[2])
    @test out.parameter_variance[4] ≈ dot(X[4,:],V[5:6,5:6]*X[4,:]) atol=1e-12
    @test out.std_error[4]^2 ≈ T+out.parameter_variance[4] atol=1e-12
    @test all(isnan,out.std_error[[1,3]])
    @test all(==("ok"),out.uncertainty_status)
    @test all(isnan,DRModels._joint_imputation_uncertainty(model,theta,V;se=false).std_error)
    bad=DRModels._joint_imputation_uncertainty(model,theta,V;se=false,covariance_status=:hessian_not_positive_definite)
    @test all(==("sdreport_non_pd_hessian"),bad.uncertainty_status)
    @test all(isnan,bad.std_error)
    invalid=DRModels._joint_imputation_uncertainty(model,theta,-V)
    @test all(==("sdreport_non_pd_hessian"),invalid.uncertainty_status)
end

@testset "Bernoulli uncertainty is conditional, with native fit-status gate" begin
    X=ones(4,1)
    model=prepared_joint_model([1.7,-0.2,missing,missing],[1.0,missing,0.0,missing],X,X,X;predictor=:bernoulli)
    theta=[0.2,0.7,0.1,-0.2];V=Matrix{Float64}(I,4,4)*0.5
    out=DRModels._joint_imputation_uncertainty(model,theta,V)
    m=prepared_joint_conditional_moments(model,theta)
    @test out.std_error[2]^2 ≈ m.variance[2] atol=1e-14
    @test out.std_error[4]^2 ≈ m.variance[4] atol=1e-14
    @test out.parameter_variance==zeros(4)
    @test all(isnan,out.std_error[[1,3]])
    bad=DRModels._joint_imputation_uncertainty(model,theta,V;covariance_status=:hessian_unavailable)
    @test all(==("sdreport_failed"),bad.uncertainty_status)
    @test all(isnan,bad.std_error)
end
println("JOINT_UNCERTAINTY_KERNEL_PASS")
