# test_cov_1125n4_gaussian_q34_corr.jl — q3/q4 correlation map (src/gaussian_ordinary_q34_sd.jl).
using DRModels
using Test, LinearAlgebra, Random

_msg(f) = try
    f()
    ""
catch e
    e isa ArgumentError ? e.msg : sprint(showerror, e)
end

@testset "cov 1125n4: _q34_corr_chol" begin
    for q in 2:4
        θ = zeros(q * (q - 1) ÷ 2)
        θ[1] = 0.3
        q == 3 && (θ[2] = -0.1; θ[3] = 0.2)
        q == 4 && (θ .= [0.2, -0.1, 0.05, 0.15, -0.2, 0.1])
        Lc = DRModels._q34_corr_chol(θ, q)
        R = Lc * Lc'
        @test issymmetric(R)
        @test all(≈(1.0; atol = 1e-12), diag(R))
        @test isposdef(Symmetric(R))
        @test all(abs.(R[tril(ones(Bool, q, q), -1)]) .< 1)
    end
    # Inverse-free round trip via from_theta
    Random.seed!(3)
    for _ in 1:15
        q = rand(2:4)
        θ = 0.4 .* randn(q * (q - 1) ÷ 2)
        R1 = DRModels._q34_corr_from_theta(θ, q)
        Lc = DRModels._q34_corr_chol(θ, q)
        @test R1 ≈ Lc * Lc' rtol = 1e-12
    end
end

@testset "cov 1125n4: _re_multi_corr_vars rejects bad lhs" begin
    using StatsModels
    bad = Core.eval(@__MODULE__, Meta.parse("@formula(y ~ 1 + x + z + w)")).rhs
    @test DRModels._re_multi_corr_vars(bad) === nothing
    dup = Core.eval(@__MODULE__, Meta.parse("@formula(y ~ (1 + x + x | g))")).rhs.args[1]
    # duplicate var in block: still parses vars; fit layer rejects duplicates
    v = DRModels._re_multi_corr_vars(dup)
    @test v == [:x, :x]
end

@testset "cov 1125n4: ordinary q3 block SD fit" begin
    Random.seed!(112504)
    n = 150
    g = repeat(1:5, outer = 30)
    x1, x2 = randn(n), randn(n)
    y = 0.1 .+ 0.3 .* x1 .- 0.2 .* x2 .+ randn(n)
    d = (; y, x1, x2, g)
    f = Core.eval(@__MODULE__, Meta.parse("""
        bf(@formula(y ~ 1 + x1 + x2 + (1 + x1 + x2 | g)),
           @formula(sigma ~ 1),
           @formula(sd(g, coef(intercept)) ~ 1),
           @formula(sd(g, coef(x1)) ~ 1),
           @formula(sd(g, coef(x2)) ~ 1))
    """))
    fit = drm(f, Gaussian(); data = d)
    @test isfinite(loglik(fit))
    blks = Dict(fit.blocks)
    @test haskey(blks, :recov) && haskey(blks, :sigma)
    @test count(k -> startswith(string(k), "sdtarget_"), keys(blks)) ≥ 3
end

@testset "cov 1125n4: q34 SD route refusals" begin
    d = (; y = randn(30), x1 = randn(30), x2 = randn(30), x3 = randn(30), x4 = randn(30),
         x5 = randn(30), g = repeat(1:3, 10))
    msg = _msg(() -> drm(bf(@formula(y ~ (1 + x1 + x2 + x3 + x4 + x5 | g)), @formula(sigma ~ 1)),
                                       Gaussian(); data = d))
    @test occursin("2 to 4", msg) || occursin("5", msg)
    msg2 = _msg(() -> drm(bf(@formula(y ~ (1 + x1 + x2 | g) + meta_V(v)),
                           @formula(sigma ~ 1)), Gaussian(); data = (d..., v = abs.(randn(30)))))
    @test occursin("meta_V", msg2) || occursin("only", msg2)
end
