# CumulativeLogit `predict` / `predict_parameters` on new data (#1121).
#
# The ordinal fitter drops the location intercept (the cutpoints absorb it;
# src/cumulative.jl) but keeps the user's formula. Rebuilding the `mu` design
# from that formula brought StatsModels' implicit intercept back, so the #1000
# column guard in `_predict_design` threw on every `newdata`, the training data
# included, and blamed factor levels. drmTMB returns the intercept-free X·β,
# the same on the link and the response scale.
#
# Reference values: drmTMB 0.7.1 on the fixtures in
# test/fixtures/cumulative_predict_1121 (see its README), rows 1:9 of nd_main
# (group g01; rows 10:18 repeat them for an unseen group).

using DRModels
using Test
using DelimitedFiles

function _read_1121(name)
    m, h = readdlm(joinpath(@__DIR__, "fixtures", "cumulative_predict_1121", name), ',';
                   header = true)
    cols = map(eachcol(m)) do c
        all(v -> v isa Real, c) ? Float64.(c) : String.(c)
    end
    return NamedTuple{Tuple(Symbol.(vec(h)))}(Tuple(cols))
end

# Intercept-free treatment-coded design for `~ x + f` with levels a < b < c.
_X1121(d) = hcat(d.x, Float64.(d.f .== "b"), Float64.(d.f .== "c"))

const _R1121_FIXED_VALUE = [-0.420934639657746, 0.0, 0.631401959486619,
                            0.00590588692381239, 0.426840526581558, 1.05824248606818,
                            -0.61959784031515, -0.198663200657404, 0.432738758829215]
const _R1121_FIXED_SE = [0.131946023699645, 0.0, 0.197919035549467,
                         0.314289841358566, 0.295717355883263, 0.368430867657332,
                         0.317501110953292, 0.29527844300888, 0.363386211168725]

@testset "CumulativeLogit predict on new data (#1121)" begin
    d = _read_1121("cumulative_logit_fixed.csv")
    nd = _read_1121("nd_main.csv")
    fit = drm(bf(@formula(y ~ x + f)), CumulativeLogit(); data = d)
    @test Dict(fit.coefnames)[:mu] == ["x", "f: b", "f: c"]
    β = coef(fit, :mu)

    @testset "fixed effects: X·β without the intercept, drmTMB values" begin
        η = _X1121(nd) * β
        @test predict(fit, nd; type = :link) ≈ η
        @test predict(fit, nd) ≈ η                       # identity: response == link
        @test predict(fit, d) ≈ _X1121(d) * β            # the training data itself
        @test predict(fit, nd; type = :link)[1:9] ≈ _R1121_FIXED_VALUE atol = 1e-8
        @test predict(fit, nd)[10:18] == predict(fit, nd)[1:9]   # g is not in the model

        p = predict(fit, nd; type = :link, se = true)
        @test p.prediction ≈ η
        @test p.se[1:9] ≈ _R1121_FIXED_SE atol = 1e-5
        @test predict(fit, nd; se = true).se ≈ p.se     # identity link: same SE

        pp = predict_parameters(fit, nd)
        @test collect(keys(pp)) == [:mu]                 # cutpoints are not a formula block
        @test pp[:mu] ≈ η
        @test predict_parameters(fit, nd; type = :link)[:mu] ≈ η
        pps = predict_parameters(fit, nd; se = true)
        @test pps[:mu].value ≈ η
        @test pps[:mu].se ≈ p.se
    end

    @testset "no factor predictor" begin
        fx = drm(bf(@formula(y ~ x)), CumulativeLogit(); data = d)
        @test Dict(fx.coefnames)[:mu] == ["x"]
        xs = [-1.0, 0.0, 1.5]
        @test predict(fx, (; x = xs); type = :link) ≈ xs .* coef(fx, :mu)[1]
        @test predict_parameters(fx, (; x = xs))[:mu] ≈ xs .* coef(fx, :mu)[1]
    end

    @testset "(1 | g) route: population-level X·β" begin
        ri = _read_1121("cumulative_logit_ri.csv")
        fr = drm(bf(@formula(y ~ x + f + (1 | g))), CumulativeLogit(); data = ri)
        η = _X1121(nd) * coef(fr, :mu)
        @test predict(fr, nd; type = :link) ≈ η
        @test predict(fr, nd; se = true).prediction ≈ η
        @test predict_parameters(fr, nd; se = true)[:mu].value ≈ η
    end

    @testset "#1000 guard still refuses a factor-level subset" begin
        # Without a cached schema the contrasts are rebuilt from `newdata`; a
        # subset of levels gives [x, f: b], not the fitted [x, f: b, f: c].
        sub = (; x = [0.0, 1.0], f = ["a", "b"])
        @test_throws ArgumentError predict(fit, sub)
        err = try
            predict(fit, sub); nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test !occursin("(Intercept)", sprint(showerror, err))
    end
end
