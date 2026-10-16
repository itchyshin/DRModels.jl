# #1000: the training schema that `predict` / `predict_parameters` reuse (factor
# contrasts, reference level, level order) belongs to the FIT, not to the formula.
#
# It used to be cached on the formula object, so fitting ONE formula object to two
# data sets coupled the fits:
#   - univariate: the cache was first-write-wins, so the second fit silently
#     reused the first fit's schema at FIT time (and failed on a level the first
#     data set lacked);
#   - bivariate: the cache was rewritten by every fit, so the second fit changed
#     the FIRST fit's predictions (here: an `ArgumentError` from the column check).
# Each fit now records its own `training_schema`.
using DRModels
using Test, Random

@testset "#1000 univariate: refitting the formula leaves the first fit's predictions alone" begin
    Random.seed!(1000)
    n = 150; x = randn(n); grp = repeat(["a", "b", "c"], outer = 50)
    y = 0.5 .+ x .+ (grp .== "b") .- 0.5 .* (grp .== "c") .+
        exp.(-1.0 .+ 0.3 .* (grp .== "c")) .* randn(n)
    data = (; y, x, grp)
    f = bf(@formula(y ~ x + grp), @formula(sigma ~ grp))
    fit1 = drm(f, Gaussian(); data = data)

    # A subset of fit1's levels ("a", its reference level, absent): only fit1's
    # training schema gives these rows the fitted contrasts.
    sub = findall(!=("a"), grp)[1:6]
    nd = (x = x[sub], grp = grp[sub])
    pred1 = predict(fit1, nd; type = :link)
    par1 = predict_parameters(fit1, nd; type = :link)
    @test pred1 ≈ predict(fit1, data; type = :link)[sub]

    # Refit the SAME formula object to data with other levels: no "a", a new "d".
    m = 120; x2 = randn(m); grp2 = repeat(["b", "c", "d"], outer = 40)
    y2 = -0.3 .+ 0.8 .* x2 .+ (grp2 .== "d") .+ exp(-1.2) .* randn(m)
    data2 = (y = y2, x = x2, grp = grp2)
    fit2 = drm(f, Gaussian(); data = data2)

    # fit2 learned its own levels (reference "b") rather than reusing fit1's schema.
    @test Dict(fit2.coefnames)[:mu] == ["(Intercept)", "x", "grp: c", "grp: d"]
    @test Dict(fit2.coefnames)[:sigma] == ["(Intercept)", "grp: c", "grp: d"]

    # fit1's predictions survive the refit exactly.
    @test predict(fit1, nd; type = :link) == pred1
    par1_after = predict_parameters(fit1, nd; type = :link)
    @test keys(par1_after) == keys(par1)
    for k in keys(par1)
        @test par1_after[k] == par1[k]
    end
    @test predict(fit1, nd; type = :link, se = true).prediction == pred1
    # "d" is a level only fit2 has seen.
    @test_throws ArgumentError predict(fit1, (x = [0.0], grp = ["d"]))

    # fit2 predicts a subset of ITS levels with its own schema.
    sub2 = findall(!=("b"), grp2)[1:6]
    @test predict(fit2, (x = x2[sub2], grp = grp2[sub2]); type = :link) ≈
          predict(fit2, data2; type = :link)[sub2]
    @test_throws ArgumentError predict(fit2, (x = [0.0], grp = ["a"]))
end

@testset "#1000 bivariate: refitting the formula leaves the first fit's predictions alone" begin
    Random.seed!(1001)
    n = 150; x = randn(n); grp = repeat(["a", "b", "c"], outer = 50)
    y1 = 0.5 .+ x .+ (grp .== "b") .+ 0.3 .* randn(n)
    y2 = -0.5 .+ 0.5 .* x .- (grp .== "c") .+ 0.3 .* randn(n)
    data = (; y1, y2, x, grp)
    f = bf(mu1 = @formula(y1 ~ x + grp), mu2 = @formula(y2 ~ x + grp),
           sigma1 = @formula(sigma1 ~ grp), rho12 = @formula(rho12 ~ 1))
    fit1 = drm(f, Gaussian(); data = data)

    sub = findall(!=("a"), grp)[1:6]
    nd = (x = x[sub], grp = grp[sub])
    pred1 = predict(fit1, nd; type = :link)
    par1 = predict_parameters(fit1, nd; type = :link)
    @test pred1[:mu1] ≈ predict(fit1, data; type = :link)[:mu1][sub]

    m = 120; x2 = randn(m); grp2 = repeat(["b", "c", "d"], outer = 40)
    z1 = 0.2 .+ x2 .+ (grp2 .== "d") .+ 0.3 .* randn(m)
    z2 = -0.2 .+ 0.5 .* x2 .- (grp2 .== "c") .+ 0.3 .* randn(m)
    data2 = (y1 = z1, y2 = z2, x = x2, grp = grp2)
    fit2 = drm(f, Gaussian(); data = data2)

    @test Dict(fit2.coefnames)[:mu1] == ["(Intercept)", "x", "grp: c", "grp: d"]
    @test Dict(fit2.coefnames)[:sigma1] == ["(Intercept)", "grp: c", "grp: d"]

    # fit1's predictions survive the refit exactly (they used to come from fit2's
    # schema and fail the column check).
    pred1_after = predict(fit1, nd; type = :link)
    @test pred1_after[:mu1] == pred1[:mu1]
    @test pred1_after[:mu2] == pred1[:mu2]
    par1_after = predict_parameters(fit1, nd; type = :link)
    @test keys(par1_after) == keys(par1)
    for k in keys(par1)
        @test par1_after[k] == par1[k]
    end
    @test_throws ArgumentError predict(fit1, (x = [0.0], grp = ["d"]))

    sub2 = findall(!=("b"), grp2)[1:6]
    @test predict(fit2, (x = x2[sub2], grp = grp2[sub2]); type = :link)[:mu1] ≈
          predict(fit2, data2; type = :link)[:mu1][sub2]
    @test_throws ArgumentError predict(fit2, (x = [0.0], grp = ["a"]))
end

# LogNormal fits that delegate to the Gaussian fit on log(y) rebuild it with the
# Jacobian-shifted likelihood; the rebuild must keep the Gaussian fit's schema.
@testset "#1000 univariate LogNormal (crossed) keeps the delegated fit's training schema" begin
    Random.seed!(1004)
    G = 12; H = 10; n = 220
    g = rand(1:G, n); h = rand(1:H, n); x = randn(n)
    grp = repeat(["a", "b", "c", "d"], outer = 55)
    bg = 0.3 .* randn(G); bh = 0.25 .* randn(H)
    y = exp.(0.1 .+ 0.35 .* x .+ 0.4 .* (grp .== "b") .- 0.3 .* (grp .== "d") .+
             bg[g] .+ bh[h] .+ 0.4 .* randn(n))
    data = (; y, x, grp, g, h)
    fit = drm(bf(@formula(y ~ x + grp + (1 | g) + (1 | h))), LogNormal(); data = data)
    sub = findall(!=("a"), grp)[1:6]
    @test predict(fit, (x = x[sub], grp = grp[sub]); type = :link) ≈
          predict(fit, data; type = :link)[sub]
end

@testset "#1000 bivariate LogNormal keeps the delegated fit's training schema" begin
    Random.seed!(1002)
    n = 150; x = randn(n); grp = repeat(["a", "b", "c"], outer = 50)
    y1 = exp.(0.2 .+ 0.5 .* x .+ 0.4 .* (grp .== "b") .+ 0.3 .* randn(n))
    y2 = exp.(-0.1 .+ 0.3 .* x .- 0.4 .* (grp .== "c") .+ 0.3 .* randn(n))
    data = (; y1, y2, x, grp)
    fit = drm(bf(mu1 = @formula(y1 ~ x + grp), mu2 = @formula(y2 ~ x + grp)), LogNormal();
              data = data)
    sub = findall(!=("a"), grp)[1:6]
    ps = predict(fit, (x = x[sub], grp = grp[sub]); type = :link)
    ref = predict(fit, data; type = :link)
    @test ps[:mu1] ≈ ref[:mu1][sub]
    @test ps[:mu2] ≈ ref[:mu2][sub]
end

@testset "#1000 bivariate Student records its training schema" begin
    rng = MersenneTwister(1003)
    n = 300; x = randn(rng, n); grp = repeat(["a", "b", "c"], outer = 100)
    sh = sqrt.(6 ./ [sum(randn(rng, 6) .^ 2) for _ in 1:n])   # t_6 mixing
    y1 = 0.5 .+ 0.8 .* x .+ (grp .== "b") .+ 0.7 .* randn(rng, n) .* sh
    y2 = -0.3 .+ 0.4 .* x .- (grp .== "c") .+ 1.1 .* randn(rng, n) .* sh
    data = (; y1, y2, x, grp)
    fit = drm(bf(mu1 = @formula(y1 ~ x + grp), mu2 = @formula(y2 ~ x + grp)), Student();
              data = data)
    sub = findall(!=("a"), grp)[1:6]
    ps = predict(fit, (x = x[sub], grp = grp[sub]); type = :link)
    ref = predict(fit, data; type = :link)
    @test ps[:mu1] ≈ ref[:mu1][sub]
    @test ps[:mu2] ≈ ref[:mu2][sub]
end
