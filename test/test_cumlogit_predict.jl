# CumulativeLogit predict / predict_parameters (#1121). The fitter drops the
# location intercept (the cutpoints absorb it), so the prediction design must be
# built without `(Intercept)` too; it used to be rebuilt with one and the #1000
# column guard then threw on every newdata, the training data included.
using DRModels
using Test, Random

@testset "CumulativeLogit predict / predict_parameters (#1121)" begin
    Random.seed!(1121)
    n = 300
    x = randn(n)
    f = rand(["a", "b", "c"], n)
    eta = 0.5 .* x .+ [fi == "b" ? 0.4 : fi == "c" ? 0.9 : 0.0 for fi in f]
    cuts = (-1.0, 0.5, 1.8)
    y = [1.0 + (e + log(u / (1 - u)) > cuts[1]) + (e + log(u / (1 - u)) > cuts[2]) +
         (e + log(u / (1 - u)) > cuts[3]) for (e, u) in zip(eta, rand(n))]
    d = (; y, x, f)
    fit = drm(bf(@formula(y ~ x + f)), CumulativeLogit(); data = d)
    @test Dict(fit.coefnames)[:mu] == ["x", "f: b", "f: c"]
    β = DRModels.coef(fit, :mu)

    nd = (; x = [-1.0, 0.0, 1.5, 2.0], f = ["a", "b", "c", "a"])
    X = [-1.0 0 0; 0.0 1 0; 1.5 0 1; 2.0 0 0]
    expected = X * β

    @test DRModels.predict(fit, nd; type = :link) ≈ expected atol = 1e-12
    @test DRModels.predict(fit, nd) ≈ expected atol = 1e-12
    pp = DRModels.predict_parameters(fit, nd)
    @test pp[:mu] ≈ expected atol = 1e-12

    # the training data itself
    Xtrain = hcat(x, f .== "b", f .== "c")
    @test DRModels.predict(fit, d) ≈ Xtrain * β atol = 1e-12

    # se = true variants
    s = DRModels.predict(fit, nd; se = true)
    @test s.prediction ≈ expected atol = 1e-12
    V = DRModels.vcov(fit)
    r = Dict(fit.blocks)[:mu]
    @test s.se ≈ [sqrt(X[i, :]' * V[r, r] * X[i, :]) for i in 1:4] atol = 1e-10
    ps = DRModels.predict_parameters(fit, nd; se = true)
    @test ps[:mu].value ≈ expected atol = 1e-12
    @test ps[:mu].se ≈ s.se atol = 1e-12

    # no-factor model
    fit2 = drm(bf(@formula(y ~ x)), CumulativeLogit(); data = d)
    @test DRModels.predict(fit2, (; x = [1.0, 2.0])) ≈ [1.0, 2.0] .* DRModels.coef(fit2, :mu)[1] atol = 1e-12
end
