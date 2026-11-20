using DRModels
using Test, Random

function _mixed_column(T, n; miss = Int[], seed = 1)
    rng = MersenneTwister(seed)
    col = Vector{Union{Missing,T}}(undef, n)
    for i in 1:n
        col[i] = T <: Bool ? (rand(rng) < 0.5) : T(rand(rng, 0:3))
    end
    col[filter(<=(n), miss)] .= missing
    return col
end
_expect_observed(raw) = [!(ismissing(v) || (v isa AbstractFloat && isnan(v))) for v in raw]

@testset "bootstrap response-mask helpers (#1374)" begin
    for T in (Int, Int32, Int64, UInt64, Float64), n in (1, 8, 31, 32, 33, 64)
        for miss in (Int[], [3], [3, 20, 77], collect(1:5:n))
            raw = _mixed_column(T, n; miss = miss, seed = n)
            want = _expect_observed(raw)
            ys = rand(MersenneTwister(n + 1), n) .+ 1.0
            out = DRModels._restore_response_mask!(copy(ys), raw)
            @test all(i -> isequal(out[i], want[i] ? ys[i] : NaN), eachindex(out))
            sim = rand(MersenneTwister(n + 2), n) .+ 1.0
            out2 = DRModels._bootstrap_keep_unobserved(raw, sim)
            @test all(i -> want[i] ? out2[i] == sim[i] : ismissing(out2[i]), eachindex(out2))
        end
    end
end