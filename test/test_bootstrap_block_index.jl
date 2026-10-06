# test_bootstrap_block_index.jl — issue #325.3.
#
# `_bootstrap_summary_rows` must index each block's coefficients by the block's
# STORED UnitRange (θ order), not by a private sequential col counter. On the
# current contiguous, θ-ordered layouts both agree; the counter silently misaligns
# names/estimates/SEs/CIs if a fit ever stores non-contiguous or reordered blocks.
# This anchors the range-based indexing against a deliberately non-contiguous layout.
using DRModels, Test, Random, Statistics

@testset "bootstrap summary indexes by stored block range (#325.3)" begin
    # A fit0 mock exposing only .blocks / .coefnames (all the summary reads). The
    # blocks are NON-CONTIGUOUS and out of θ order on purpose:
    #   :mu    -> cols 1:2
    #   :sigma -> cols 4:5   (gap at col 3)
    #   :resd  -> col 3
    fit0 = (
        blocks = Pair{Symbol,UnitRange{Int}}[
            :mu => 1:2, :sigma => 4:5, :resd => 3:3,
        ],
        coefnames = Pair{Symbol,Vector{String}}[
            :mu => ["(Intercept)", "x"], :sigma => ["(Intercept)", "x"], :resd => ["g"],
        ],
    )
    # A distinct estimate + draw distribution per θ column so a misindex is visible.
    est = [10.0, 20.0, 30.0, 40.0, 50.0]                 # col j centred at 10j
    B = 400
    Random.seed!(1)
    # draws[:, j] ~ Normal(10j, j): mean ≈ 10j, sd ≈ j.
    draws = hcat((10.0 * j .+ float(j) .* randn(B) for j in 1:5)...)

    rows = DRModels._bootstrap_summary_rows(fit0, draws, est, 0.95)
    @test length(rows) == 5

    # Row order follows `zip(blocks, coefnames)`: mu(1,2), sigma(4,5), resd(3).
    byname = Dict((r.param, r.coef) => r for r in rows)

    # :mu -> cols 1,2
    @test byname[(:mu, "(Intercept)")].estimate == est[1]
    @test byname[(:mu, "x")].estimate == est[2]
    # :sigma -> cols 4,5 (NOT 3,4 — the counter's misindex)
    @test byname[(:sigma, "(Intercept)")].estimate == est[4]
    @test byname[(:sigma, "x")].estimate == est[5]
    # :resd -> col 3
    @test byname[(:resd, "g")].estimate == est[3]

    # The bootstrap SE of each block reflects its OWN column's spread (sd ≈ col j).
    # sigma:(Intercept) is col 4 (sd≈4), resd:g is col 3 (sd≈3) — a swapped index
    # would put ≈3 on sigma and ≈4 on resd.
    @test byname[(:sigma, "(Intercept)")].std_error > byname[(:resd, "g")].std_error
    @test isapprox(byname[(:resd, "g")].std_error, 3.0; atol = 0.6)
    @test isapprox(byname[(:sigma, "x")].std_error, 5.0; atol = 0.8)
end
