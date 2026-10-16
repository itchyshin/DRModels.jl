# #1000: bivariate predict / predict_parameters must reuse the training schema
# (factor contrasts + level order) cached at fit time, not rebuild it from newdata.
using DRModels
using Test, Random

@testset "#1000 bivariate predict reuses the training factor schema" begin
    Random.seed!(1000)
    n = 150; x = randn(n); grp = repeat(["a", "b", "c"], outer = 50)
    y1 = 0.5 .+ x .+ (grp .== "b") .+ 0.3 .* randn(n)
    y2 = -0.5 .+ 0.5x .- (grp .== "c") .+ 0.3 .* randn(n)
    data = (; y1, y2, x, grp)
    fit = drm(bf(mu1 = @formula(y1 ~ x + grp), mu2 = @formula(y2 ~ x + grp),
                 sigma1 = @formula(sigma1 ~ grp), rho12 = @formula(rho12 ~ 1)),
              Gaussian(); data = data)
    ref = predict(fit, data; type = :link)
    refp = predict_parameters(fit, data; type = :link)

    # Subset of levels ("a" is the fitted reference level and is absent here).
    sub = findall(g -> g != "a", grp)[1:6]
    nd_sub = (x = x[sub], grp = grp[sub])
    ps = predict(fit, nd_sub; type = :link)
    @test ps[:mu1] ≈ ref[:mu1][sub]
    @test ps[:mu2] ≈ ref[:mu2][sub]
    pp = predict_parameters(fit, nd_sub; type = :link)
    @test pp[:mu1] ≈ refp[:mu1][sub]
    @test pp[:sigma1] ≈ refp[:sigma1][sub]          # factor in the scale formula too

    # Reordered rows with the same number of levels: the reference level is
    # still the fitted one, not the first level that happens to appear.
    ord = [3, 2, 1, 6, 5, 4]                        # grp = c, b, a, c, b, a
    nd_ord = (x = x[ord], grp = grp[ord])
    @test predict(fit, nd_ord; type = :link)[:mu1] ≈ ref[:mu1][ord]
    @test predict_parameters(fit, nd_ord; type = :link)[:sigma1] ≈ refp[:sigma1][ord]

    # SE path builds the same design.
    pse = predict(fit, nd_sub; type = :link, se = true)
    @test pse.prediction[:mu1] ≈ ref[:mu1][sub]

    # An unseen level is still refused, loudly.
    @test_throws ArgumentError predict(fit, (x = [0.0], grp = ["d"]))

    # Refitting the same formula object on data with different levels re-learns
    # the schema rather than reusing the stale one.
    f = bf(mu1 = @formula(y1 ~ x + grp), mu2 = @formula(y2 ~ x + grp))
    drm(f, Gaussian(); data = data)
    keep = grp .!= "a"
    data2 = (y1 = y1[keep], y2 = y2[keep], x = x[keep], grp = grp[keep])
    fit2 = drm(f, Gaussian(); data = data2)
    @test length(coef(fit2, :mu1)) == 3
    @test length(predict(fit2, data2; type = :link)[:mu1]) == length(data2.x)
    @test_throws ArgumentError predict(fit2, (x = [0.0], grp = ["a"]))   # "a" not in fit2's data
end
