# cov lane 1125n8 — ZeroOneBeta Laplace setup / augmentation (src/zeroonebeta.jl).
using DRModels
using Test, LinearAlgebra

@testset "cov 1125n8: _zeroonebeta_laplace_setup" begin
    n = 20
    y = vcat(zeros(3), ones(2), 0.2 .* rand(15) .+ 0.01)
    Xμ = ones(n, 1)
    aux_from, θβ0, θσ0, zoi, coi = DRModels._zeroonebeta_laplace_setup(y, Xμ)
    @test 0 < zoi < 1
    @test 0 < coi < 1
    @test length(θβ0) == 1
    @test isfinite(θσ0)
    aux = aux_from(θσ0)
    @test aux.y == Float64.(y)
    @test aux.zoi == zoi
    @test aux.coi == coi
    @test length(aux.isatom) == n
    @test sum(aux.isatom) == 5
    @test isfinite(aux.precision)
    all_boundary = vcat(zeros(4), ones(4))
    _, _, σ0, z, c = DRModels._zeroonebeta_laplace_setup(all_boundary, ones(8, 1))
    @test z ≈ 1.0 - 1e-6 atol = 1e-5
    @test 0 < c < 1
    @test isfinite(σ0)
end

@testset "cov 1125n8: _augment_zeroonebeta_fit" begin
    n = 6
    base = DrmFit(
        ZeroOneBeta(),
        [:mu => 1:1, :sigma => 2:2],
        [:mu => ["(Intercept)"], :sigma => ["(Intercept)"]],
        [0.0, -1.0], [1.0 0.1; 0.1 1.0], -8.0, n, true,
        Dict(:mu => fill(0.3, n)),
        Dict(:mu => [0.0, 1.0, 0.2, 0.8, 0.5, 0.1]),
        Dict(:sigma => fill(0.5, n)),
    )
    zoi, coi = 0.2, 0.6
    nb = 2
    aug = DRModels._augment_zeroonebeta_fit(base, zoi, coi, ["(Intercept)"], ["(Intercept)"],
                                            base.obs[:mu], n, nb)
    @test length(aug.theta) == length(base.theta) + 2
    @test haskey(aug.means, :mu)
    @test aug.means[:mu] ≈ (1 - zoi) .* base.means[:mu] .+ zoi * coi
    @test aug.scales[:zoi] == fill(zoi, n)
    @test aug.vcov[end, end] ≈ 1 / (nb * coi * (1 - coi))
    @test any(b -> first(b) == :zoi, aug.blocks)
end