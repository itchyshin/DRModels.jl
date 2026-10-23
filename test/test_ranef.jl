# ranef(fit): per-level conditional random-effect estimates (BLUPs) for the
# Gaussian closed-form RE paths. BLUPs shrink toward zero, so we don't expect
# them to equal the simulated truth exactly; instead they must (a) have the right
# shape, (b) correlate strongly with the true per-group effects, and (c) be
# centred near zero. Covers single (1|g), correlated (1+x|g), and multi (1|g)+(1|h).
using DRModels
using Test, Random, LinearAlgebra, Statistics

@testset "ranef() — Gaussian BLUPs" begin
    @testset "single random intercept (1|g)" begin
        Random.seed!(20260601)
        G = 120; m = 25; n = G * m
        g = repeat(1:G, inner = m)
        β0 = 1.0; σ = 0.5; sd_re = 0.8
        b = sd_re .* randn(G)
        y = β0 .+ b[g] .+ σ .* randn(n)
        data = (; y, g)
        fit = drm(bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = data)

        re = ranef(fit)
        @test haskey(re, :g)
        b̂ = re[:g]
        @test length(b̂) == G
        @test cor(b̂, b) > 0.9                 # BLUPs track the true effects
        @test abs(mean(b̂)) < 0.1              # centred near zero
        @test std(b̂) < std(b) + 0.1           # shrinkage: not larger than truth
    end

    @testset "correlated random slope (1+x|g)" begin
        Random.seed!(20260602)
        G = 150; m = 20; n = G * m
        g = repeat(1:G, inner = m); x = randn(n)
        β = [0.4, 0.6]; σ = 0.4
        sd_int = 0.7; sd_slp = 0.5; ρre = 0.3
        Σ = [sd_int^2 ρre*sd_int*sd_slp; ρre*sd_int*sd_slp sd_slp^2]
        B = cholesky(Symmetric(Σ)).L * randn(2, G)
        b0 = B[1, :]; b1 = B[2, :]
        y = β[1] .+ β[2] .* x .+ b0[g] .+ b1[g] .* x .+ σ .* randn(n)
        data = (; y, x, g)
        fit = drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1)), Gaussian(); data = data)

        re = ranef(fit)
        @test haskey(re, :g)
        B̂ = re[:g]
        @test size(B̂) == (G, 2)
        @test cor(B̂[:, 1], b0) > 0.9          # intercept BLUPs
        @test cor(B̂[:, 2], b1) > 0.85         # slope BLUPs
    end

    @testset "multiple components (1|g)+(1|h)" begin
        Random.seed!(20260603)
        G = 80; H = 60; m = 30; n = G * m
        g = repeat(1:G, inner = m)
        # _group_index keys re[:h] by first-seen order; prepend 1:H so first-seen
        # order == natural 1:H and the recovered BLUPs line up with bh below.
        h = vcat(collect(1:H), rand(1:H, n - H))
        β0 = 0.5; σ = 0.4; sd_g = 0.7; sd_h = 0.5
        bg = sd_g .* randn(G); bh = sd_h .* randn(H)
        y = β0 .+ bg[g] .+ bh[h] .+ σ .* randn(n)
        data = (; y, g, h)
        fit = drm(bf(@formula(y ~ 1 + (1 | g) + (1 | h)), @formula(sigma ~ 1)), Gaussian(); data = data)

        re = ranef(fit)
        @test length(re) == 2
        kg, kh = Symbol("g"), Symbol("h")
        @test haskey(re, kg) && haskey(re, kh)
        @test length(re[kg]) == G && length(re[kh]) == H
        @test cor(re[kg], bg) > 0.85
        @test cor(re[kh], bh) > 0.7
    end

    @testset "no random effects → empty" begin
        Random.seed!(20260604)
        n = 200; x = randn(n); y = 1.0 .+ 0.5 .* x .+ 0.3 .* randn(n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
        @test isempty(ranef(fit))
    end
end
