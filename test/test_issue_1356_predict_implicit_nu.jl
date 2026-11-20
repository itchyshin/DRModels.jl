# refs #1356 — implicit nu ~ 1 must appear in predict_parameters (drmTMB parity).
module TestIssue1356PredictImplicitNu

using DRModels
using Test

@testset "#1356 predict_parameters includes implicit :nu" begin
    N = 150
    x = [sin(i) for i in 1:N]
    u = [mod(0.6180339887i, 1) for i in 1:N]
    y = [0.5x[i] + 0.25tan(1.4 * (2u[i] - 1)) for i in 1:N]
    nd = (; x = [0.0, 1.0])
    data = (; y, x)

    @testset "Student()" begin
        f_impl = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Student(); data = data)
        f_expl = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)),
                     Student(); data = data)
        @test haskey(Dict(fixef(f_impl)), :nu)
        pp_impl = predict_parameters(f_impl, nd; type = :link)
        pp_expl = predict_parameters(f_expl, nd; type = :link)
        @test :nu in keys(pp_impl)
        @test sort(collect(keys(pp_impl))) == [:mu, :nu, :sigma]
        @test pp_impl[:nu] ≈ pp_expl[:nu]
        @test pp_impl[:nu] ≈ fill(coef(f_impl, :nu)[1], length(nd.x))
    end

    @testset "SkewNormal()" begin
        f = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), SkewNormal(); data = data)
        pp = predict_parameters(f, nd; type = :link)
        @test :nu in keys(pp)
        @test pp[:nu] ≈ fill(coef(f, :nu)[1], length(nd.x))
    end

    @testset "Tweedie()" begin
        yt = max.(y, 1e-4)
        f = drm(bf(@formula(yt ~ x), @formula(sigma ~ 1)), Tweedie(); data = (; yt, x))
        pp = predict_parameters(f, nd; type = :link)
        @test :nu in keys(pp)
        @test pp[:nu] ≈ fill(coef(f, :nu)[1], length(nd.x))
    end
end

end