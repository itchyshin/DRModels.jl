# Beta location-scale `y ~ site + gen`, `sigma ~ site` on agridat::wedderburn.barley
# (n = 90, 9 sites x 10 genotypes) (#1225).
#
# KNOWN BAD FIT, recorded as `@test_broken` (no optimizer change in this file).
# The default single start stops at logLik 213.337 with max|grad| ~ 4.9e4 (site S5's
# sigma coefficient runs towards the soft clamp), while `multi_start = 5` reaches
# 241.552. These `@test_broken` lines flip to an "Unexpected Pass" error the moment a
# fix lands -- then change them to plain `@test`.
#
# Data frozen in test/fixtures/beta_locscale_1225 (Rdatasets agridat/wedderburn.barley;
# y is percent leaf-blotch cover, squeezed to (0,1) with the Smithson-Verkuilen
# transform `(y/100*(n-1) + 0.5)/n`, as in the issue).

using DRModels
using Test
using DelimitedFiles
using Logging

function _barley_1225()
    m, h = readdlm(joinpath(@__DIR__, "fixtures", "beta_locscale_1225", "wedderburn_barley.csv"),
                   ',', Any; header = true)
    nm = String.(vec(h))
    col(n) = m[:, findfirst(==(n), nm)]
    y = Float64.(col("y")); n = length(y)
    return (; y = (y ./ 100 .* (n - 1) .+ 0.5) ./ n,
            site = String.(col("site")), gen = String.(col("gen")))
end

@testset "#1225 Beta location-scale on wedderburn.barley (known bad fit)" begin
    d = _barley_1225()
    @test length(d.y) == 90
    fm = bf(@formula(y ~ site + gen), @formula(sigma ~ site))

    fit  = with_logger(NullLogger()) do
        drm(fm, Beta(); data = d)
    end
    fit5 = with_logger(NullLogger()) do
        drm(fm, Beta(); data = d, control = DRMControl(multi_start = 5))
    end

    # The better optimum exists and is reachable by multi-start (guards the reference).
    @test loglik(fit5) > 241.5

    # The default single start should reach it too ...
    @test_broken loglik(fit) > loglik(fit5) - 1e-3
    # ... and stop at a stationary point rather than with max|grad| ~ 5e4.
    @test_broken maximum(abs, gradient(fit)) < 1e-3
end
