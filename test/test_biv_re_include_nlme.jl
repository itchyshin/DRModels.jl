# Bivariate Gaussian `(1 | p | g)` with `method = :REML` and `miss_control(response = "include")`.
# drmTMB 0.8.1 refuses this combination, so there is no R reference from drmTMB. The reference here is
# an independent REML fit of the same model by nlme::lme in long format (one row per observed entry;
# random effects with a 2 x 2 covariance per group, a residual SD per response, and a residual
# correlation between the two entries of one data row), made by fixtures/biv_re_include/reml_nlme.R
# (R 4.6.1) on three generated data sets (reml_data1.csv to reml_data3.csv; 9 to 25 percent of the cells of
# each response missing, some rows missing both, reml_data3 with three mu1 covariates).
# Bounds: log-likelihood 1e-5 (the nlme optimiser stops about 6e-7 below the optimum on data set 3),
# fixed effects 1e-4, variance parameters 1e-3.
using DRModels
using Test
using DelimitedFiles

function _nl_data(k)
    m, h = readdlm(joinpath(@__DIR__, "fixtures", "biv_re_include", "reml_data$k.csv"), ',', Any; header = true)
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(String.(vec(h)))
        c = m[:, j]
        v = n == "g" ? string.(c) :
            any(x -> x == "NA", c) ? Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c] : Float64.(c)
        push!(cols, (n == "y1" ? :y1m : n == "y2" ? :y2m : Symbol(n)) => v)
    end
    return (; cols...)
end

function _nl_reference()
    m = readdlm(joinpath(@__DIR__, "fixtures", "biv_re_include", "reml_nlme_reference.csv"), ',', Any; skipstart = 1)
    ref = Dict{Int,Dict{String,Float64}}()
    for r in 1:size(m, 1)
        get!(ref, Int(m[r, 1]), Dict{String,Float64}())[String(m[r, 2])] = Float64(m[r, 3])
    end
    return ref
end

@testset "bivariate (1 | p | g): REML with include equals an independent nlme::lme REML fit" begin
    ref = _nl_reference()
    inc = miss_control(response = "include")
    for k in 1:3
        d = _nl_data(k); r = ref[k]
        mu1 = k == 3 ? @formula(y1m ~ x1 + xf2 + xf3 + (1 | p | g)) : @formula(y1m ~ x1 + (1 | p | g))
        f = bf(mu1 = mu1, mu2 = @formula(y2m ~ x2 + (1 | p | g)),
               sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
        fit = drm(f, Gaussian(); data = d, missing = inc, method = :REML)
        @testset "data set $k" begin
            @test fit.converged
            @test count(!ismissing, d.y1m) + count(!ismissing, d.y2m) == Int(r["N"])      # observed entries used
            @test abs(loglikelihood(fit) - r["ll"]) < 1e-5
            b = vcat(coef(fit, :mu1), coef(fit, :mu2))
            bn = k == 3 ? [r["r1"], r["x1"], r["xf2"], r["xf3"], r["r2"], r["x2"]] : [r["r1"], r["x1"], r["r2"], r["x2"]]
            @test maximum(abs.(b .- bn)) < 1e-4
            Lam = Matrix(fit.ranef.Sigma_g)
            @test abs(Lam[1, 1] - r["Sigma11"]) < 1e-3 && abs(Lam[1, 2] - r["Sigma12"]) < 1e-3 && abs(Lam[2, 2] - r["Sigma22"]) < 1e-3
            @test abs(exp(only(coef(fit, :sigma1))) - r["sigma1"]) < 1e-3
            @test abs(exp(only(coef(fit, :sigma2))) - r["sigma2"]) < 1e-3
            @test abs(DRModels.RHO_GUARD * tanh(only(coef(fit, :rho12))) - r["rho"]) < 1e-3
        end
    end
end
