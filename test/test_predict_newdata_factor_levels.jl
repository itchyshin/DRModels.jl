# predict(fit, newdata) with a factor, on the generic (non-Gaussian) routes.
#
# Gap map v2 (area M): for Poisson, Binomial, Gamma, NB2, Beta, Student, LogNormal
# and others, `predict` threw unless `newdata` held every level of every factor,
# and a single-row `newdata` failed with "only one level found". The fitters of
# those families build their designs without the schema cache, so `predict`
# rebuilt the contrasts from the rows of `newdata`. `drm` now records the
# training factor levels, and `predict` codes the factor with them. An unseen
# level is still refused, as drmTMB does ("unknown factor level").
#
# No RNG: the data come from a closed-form pseudo-noise, so the test does not
# depend on a Julia version. Expected values are computed by hand from the fitted
# coefficients (dummy coding on the first level, as at fit time).
using DRModels
using Test

const _PN_N = 72
_u(i) = mod(sin(12.9898i) * 43758.5453, 1.0)         # deterministic in (0, 1)
const _PN_LEVELS = ["a", "b", "c", "d"]
const _PN_X = [round(2.4 * _u(i) - 1.2; digits = 4) for i in 1:_PN_N]
const _PN_F = [_PN_LEVELS[mod1(i, 4)] for i in 1:_PN_N]
const _PN_G = ["g" * string(mod1(3i + div(i, 4), 6)) for i in 1:_PN_N]
const _PN_B = Dict("a" => 0.0, "b" => 0.5, "c" => -0.4, "d" => 0.9)
_eta(i) = 0.4 + 0.6 * _PN_X[i] + _PN_B[_PN_F[i]]
_logistic(z) = 1 / (1 + exp(-z))

const _PN_Y = Dict(
    :poisson => [round(Int, exp(_eta(i)) * (0.5 + _u(i + 100))) for i in 1:_PN_N],
    :binomial => [_u(i + 200) < _logistic(_eta(i) - 0.4) ? 1 : 0 for i in 1:_PN_N],
    :gamma => [exp(_eta(i)) * (0.6 + 0.8 * _u(i + 300)) for i in 1:_PN_N],
    :nb2 => [round(Int, exp(_eta(i)) * (0.1 + 2.2 * _u(i + 400)^2)) for i in 1:_PN_N],
    :beta => [clamp(_logistic(_eta(i) - 0.5 + 0.8 * (_u(i + 500) - 0.5)), 0.02, 0.98) for i in 1:_PN_N],
    :student => [_eta(i) + 0.4 * (_u(i + 600) - 0.5) * (i % 9 == 0 ? 6.0 : 1.0) for i in 1:_PN_N],
    :lognormal => [exp(_eta(i) + 0.5 * (_u(i + 700) - 0.5)) for i in 1:_PN_N],
)

_data(key) = (; y = _PN_Y[key], x = _PN_X, f = _PN_F, g = _PN_G)

# (key, family, extra formulas, inverse link)
const _PN_CASES = [
    (:poisson, Poisson(), (), exp),
    (:binomial, Binomial(), (), _logistic),
    (:gamma, Gamma(), (@formula(sigma ~ 1),), exp),
    (:nb2, NegBinomial2(), (@formula(sigma ~ 1),), exp),
    (:beta, Beta(), (@formula(sigma ~ 1),), _logistic),
    (:student, Student(), (@formula(sigma ~ 1), @formula(nu ~ 1)), identity),
    # drmTMB's response-scale predict for LogNormal is the log-scale mu (#1282), so no inverse link
    (:lognormal, LogNormal(), (@formula(sigma ~ 1),), identity),
]

# η from the fitted `mu` coefficients; names are "(Intercept)", "x", "f: b", ...
function _hand_eta(fit, x, f)
    β = coef(fit, :mu)
    names = last(only(filter(p -> first(p) === :mu, fit.coefnames)))
    return [sum(β[k] * (n == "(Intercept)" ? 1.0 : n == "x" ? xi :
                        startswith(n, "f: ") ? Float64(fi == n[4:end]) : 0.0)
                for (k, n) in enumerate(names)) for (xi, fi) in zip(x, f)]
end

@testset "predict(newdata) with a factor, generic routes" begin
    for (key, fam, extra, ilink) in _PN_CASES
        @testset "$key" begin
            data = _data(key)
            fit = drm(bf(@formula(y ~ x + f), extra...), fam; data)
            @test fit.converged
            all_rows = (; x = _PN_X, f = _PN_F)

            # all levels present: the case that always worked
            full_link = predict(fit, all_rows; type = :link)
            @test full_link ≈ _hand_eta(fit, _PN_X, _PN_F)
            @test predict(fit, all_rows) ≈ ilink.(full_link)

            for (label, rows) in (("single row", [6]), ("single row, level d", [4]),
                                  ("subset of levels", [2, 3, 6, 11]),
                                  ("reordered levels", [4, 3, 2, 1, 8, 7]))
                nd = (; x = _PN_X[rows], f = _PN_F[rows])
                @testset "$label" begin
                    @test predict(fit, nd; type = :link) ≈ full_link[rows]
                    @test predict(fit, nd; type = :link) ≈ _hand_eta(fit, nd.x, nd.f)
                    @test predict(fit, nd) ≈ ilink.(full_link[rows])
                    @test predict(fit, nd; type = :response) ≈ predict(fit, all_rows)[rows]
                end
            end

            # a response column in newdata is allowed and ignored
            @test predict(fit, (; y = data.y[[5]], x = _PN_X[[5]], f = _PN_F[[5]]);
                          type = :link) ≈ full_link[[5]]

            # an unseen level is refused, naming the column, the level and the fitted levels
            err = try
                predict(fit, (; x = [0.1], f = ["z"])); nothing
            catch e
                e
            end
            @test err isa ArgumentError
            msg = err === nothing ? "" : sprint(showerror, err)
            @test occursin("`f`", msg) && occursin("z", msg) && occursin("a, b, c, d", msg)
            @test_throws ArgumentError predict(fit, (; x = [0.1, 0.2], f = ["a", "z"]))
        end
    end

    @testset "other refusals keep their own message" begin
        fit = drm(bf(@formula(y ~ x + f)), Poisson(); data = _data(:poisson))
        # factor fitted on strings: a Symbol column is not the same level
        @test_throws ArgumentError predict(fit, (; x = [0.1], f = [:b]))
        @test_throws ArgumentError predict(fit, (; x = [0.1]))                  # column missing
        @test_throws ArgumentError predict(fit, (; x = [0.1], f = [missing]))   # value missing
    end

    @testset "scale block with a factor: predict_parameters" begin
        data = _data(:gamma)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ f)), Gamma(); data)
        all_rows = (; x = _PN_X, f = _PN_F)
        full = predict_parameters(fit, all_rows; type = :link)
        for rows in ([3], [2, 4], [4, 1, 3])
            nd = (; x = _PN_X[rows], f = _PN_F[rows])
            one = predict_parameters(fit, nd; type = :link)
            @test one[:mu] ≈ full[:mu][rows]
            @test one[:sigma] ≈ full[:sigma][rows]
        end
    end

    @testset "random-intercept model" begin
        for (key, fam, ilink) in ((:poisson, Poisson(), exp), (:gamma, Gamma(), exp))
            data = _data(key)
            fit = drm(bf(@formula(y ~ x + f + (1 | g))), fam; data)
            all_rows = (; x = _PN_X, f = _PN_F)
            full = predict(fit, all_rows; type = :link)
            @test full ≈ _hand_eta(fit, _PN_X, _PN_F)
            for rows in ([6], [2, 3], [4, 1, 3])
                nd = (; x = _PN_X[rows], f = _PN_F[rows])
                @test predict(fit, nd; type = :link) ≈ full[rows]
                @test predict(fit, nd) ≈ ilink.(full[rows])
            end
            @test_throws ArgumentError predict(fit, (; x = [0.1], f = ["z"]))
        end
    end

    @testset "a bf object reused for a second fit follows the new data" begin
        # The level catalogue lives on the formula; refitting on data with a
        # different factor must replace it.
        form = bf(@formula(y ~ x + f))
        d1 = _data(:poisson)
        fit1 = drm(form, Poisson(); data = d1)
        keep = findall(!=("d"), _PN_F)
        d2 = (; y = d1.y[keep], x = d1.x[keep], f = d1.f[keep], g = d1.g[keep])
        fit2 = drm(form, Poisson(); data = d2)
        @test length(coef(fit2, :mu)) == 4                    # intercept, x, f: b, f: c
        @test isfinite(only(predict(fit2, (; x = [0.2], f = ["c"]))))
        @test_throws ArgumentError predict(fit2, (; x = [0.2], f = ["d"]))
    end
end
