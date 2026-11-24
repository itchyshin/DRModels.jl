using DRModels, Test

@testset "Bernoulli SD formula factor and interaction design" begin
    G, m = 24, 4
    g = repeat(1:G, inner = m)
    band_group = repeat(["control", "treated"], inner = G ÷ 2)
    z_group = collect(range(-1.2, 1.2; length = G))
    band = band_group[g]
    z = z_group[g]
    y = zeros(Float64, length(g))
    data = (; y, band, z, g)

    gidx, n_groups = DRModels._group_index(g)
    rhs = (@formula(y ~ band * z)).rhs
    Zg, names = DRModels._sd_group_design(:y, rhs, data, gidx, n_groups, :g)

    treated = Float64.(band_group .== "treated")
    expected = hcat(ones(G), treated, z_group, treated .* z_group)
    @test n_groups == G
    @test names == ["(Intercept)", "band: treated", "z", "band: treated & z"]
    @test Zg ≈ expected

    alpha = [-0.7, 0.2, 0.3, -0.15]
    expected_sd = exp.(-0.7 .+ 0.2 .* treated .+ 0.3 .* z_group .-
                       0.15 .* treated .* z_group)
    @test exp.(Zg * alpha) ≈ expected_sd

    bad_data = merge(data, (; band = copy(band)))
    bad_data.band[1] = "treated"
    @test_throws ArgumentError DRModels._sd_group_design(
        :y, rhs, bad_data, gidx, n_groups, :g,
    )
end
