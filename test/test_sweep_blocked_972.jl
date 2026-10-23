# DRModels.jl#972: _vcov_from_hessian used to invert any Hessian whose
# eigenvalues were merely bounded away from zero in absolute value. A negative
# eigenvalue (a saddle) then produced a negative variance and no warning.
# DRModels.jl#956: two fitters still called inv directly. The bivariate REML
# site in src/gaussian_bivariate.jl is left to open pull request #793.

using DRModels, Test, LinearAlgebra

@testset "indefinite Hessian does not return a Wald variance (#972)" begin
    H = [4.0 0.0; 0.0 -2.0]
    V = @test_logs (:warn, r"not positive definite") DRModels._vcov_from_hessian(H)
    @test all(isnan, V)
    @test !any(d -> isfinite(d) && d < 0, diag(V))

    healthy = [4.0 1.0; 1.0 4.0]
    got = @test_nowarn DRModels._vcov_from_hessian(healthy)
    @test isapprox(got, inv(healthy); rtol = 1e-10)
end
