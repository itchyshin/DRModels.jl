# Refs #1374: harden `_restore_response_mask!` and `_bootstrap_keep_unobserved`
# against Julia 1.12 miscompiles (same pattern as #1326 / j112rows).
using DRModels
using Test, Random

_expect_missing(raw) = [ismissing(v) || (v isa AbstractFloat && isnan(v)) for v in raw]

function _mixed_column(T, n; miss = Int[], seed = 1)
    rng = MersenneTwister(seed)
    col = Vector{Union{Missing,T}}(T.(rand(rng, 0:3, n)))
    col[filter(<=(n), miss)] .= missing
    return col
end

@testset "bootstrap mask helpers: Union{Missing,T} columns (#1374)" begin
    for T in (Int, Int32, Int64, UInt64, Float64)
        for n in (1, 8, 29, 31, 32, 33, 35, 64, 160, 1000, 10_001)
            for miss in (Int[], [3], [3, 20, 77], collect(1:5:n), collect(1:n))
                raw = _mixed_column(T, n; miss = miss, seed = n)
                want = _expect_missing(raw)
                rng = MersenneTwister(n + 1)
                ys = rand(rng, n) .+ 1.0
                out = DRModels._restore_response_mask!(copy(ys), raw)
                @test all(isequal.(out, [want[i] ? NaN : ys[i] for i in 1:n]))
                sim = rand(rng, n) .+ 1.0
                out2 = DRModels._bootstrap_keep_unobserved(raw, sim)
                @test all(isequal.(out2, [want[i] ? raw[i] : sim[i] for i in 1:n]))
            end
        end
    end
end