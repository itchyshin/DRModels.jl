module TestPolytomyKernels
using DRModels, Test, LinearAlgebra, SparseArrays, ForwardDiff

@testset "polytomy downstream dimensions and likelihood normalization" begin
    for (tree, C) in [
        ("(A:1,B:2,C:3);", Matrix(Diagonal([1.,2.,3.]))),
        ("((A:1,B:2,C:3):4,D:5,E:6);",
         [5. 4 4 0 0;4 6 4 0 0;4 4 7 0 0;0 0 0 5 0;0 0 0 0 6])]
        phy = augmented_phy(tree)
        # Repeated observations, reordered tips, and an unobserved leaf in mixed.
        species = [phy.n_leaves,1,phy.n_leaves,3,2,1]
        n = length(species); X=ones(n,1)
        Y=hcat(0.1 .* sin.(1:n),0.1 .* cos.(1:n))
        prob2,Q2=DRModels.make_coevo_problem(phy,Y,X;species)
        @test prob2.N == phy.n_total-1
        @test prob2.leaf_node == species
        beta=zeros(1,2); Lambda=[0.3 0.04;0.04 0.2]; D=[0.5 0.05;0.05 0.7]
        ll,u,H,P=DRModels.coevo_marginal_cov(prob2,Q2,beta,Lambda,D)
        @test length(u)==2*(phy.n_total-1)
        @test size(P)==(length(u),length(u))
        # Independent observed covariance assembled from shared-path tip C.
        V=kron(C[species,species],Lambda)+kron(Matrix{Float64}(I,n,n),D)
        residual=vec(Y')
        dense_ll=-0.5*(length(residual)*log(2pi)+logdet(Symmetric(V))+dot(residual,V\residual))
        # Existing engine normalization retains this prior-only ridge.
        ridge=0.5*(logdet(Symmetric(Matrix(P))+1e-10I)-logdet(Symmetric(Matrix(P))))
        @test ll ≈ dense_ll+ridge atol=1e-10 rtol=0

        prob4,Q4=DRModels.make_problem(phy,Y[:,1],Y[:,2],X,X,X,X,X;species)
        @test prob4.n_total==phy.n_total-1
        @test prob4.leaf_node==species
        P4=DRModels.prior_precision(Q4,Matrix{Float64}(I,4,4)*5)
        nlatent=4*(phy.n_total-1); state=zeros(nlatent)
        @test size(P4)==(nlatent,nlatent)
        beta4=(mu1=[0.],mu2=[0.],s1=[-0.2],s2=[0.1],rho=[0.15])
        # Independent bivariate density; no calls to leaf_nll/leaf_hess.
        function joint_oracle(z)
            val=dot(z,Matrix(P4)*z)/2
            rho=DRModels.RHO_GUARD*tanh(0.15)
            for i in 1:n
                k=4*(species[i]-1)
                s1=exp(-0.2+z[k+3]);s2=exp(0.1+z[k+4])
                e1=(Y[i,1]-z[k+1])/s1;e2=(Y[i,2]-z[k+2])/s2
                val += log(2pi)+log(s1)+log(s2)+log1p(-rho^2)/2+
                    (e1^2+e2^2-2rho*e1*e2)/(2*(1-rho^2))
            end
            val
        end
        Hdense=ForwardDiff.hessian(joint_oracle,state)
        Hsparse=DRModels.build_Huu(prob4,P4,state,beta4)
        @test Matrix(Hsparse) ≈ Hdense atol=1e-10 rtol=0
        @test DRModels.joint_nll(prob4,P4,state,beta4) ≈ joint_oracle(state) atol=1e-12 rtol=0
        expected=-joint_oracle(state)-logdet(Symmetric(Hdense))/2+
            logdet(Symmetric(Matrix(P4))+1e-10I)/2
        @test DRModels.laplace_ll(prob4,P4,beta4,state,cholesky(Symmetric(Hsparse))) ≈ expected atol=1e-10 rtol=0
        # Fixed-state formula/normalization check; state is not claimed to be a mode.
    end
end
end
