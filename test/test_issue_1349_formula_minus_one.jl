# refs #1349 — `@formula(y ~ x - 1)` must not silently keep an intercept.
module TestIssue1349FormulaMinusOne

using DRModels
using Test

function _refusal(thunk)
    try
        thunk()
        return nothing
    catch e
        return e
    end
end

@testset "#1349 refuse R-style x - 1" begin
    x = [2sin(i) for i in 1:60]
    z = [cos(3i) for i in 1:60]
    y = [3 + 0.5 * x[i] + 0.3cos(7i) for i in 1:60]
    d = (; y, x, z)

    @testset "mu ~ x - 1" begin
        err = _refusal(() -> drm(bf(@formula(y ~ x - 1), @formula(sigma ~ 1)), Gaussian(); data = d))
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("0 + x", msg)
        @test !occursin("-1 + x", msg)
        fit0 = drm(bf(@formula(y ~ 0 + x), @formula(sigma ~ 1)), Gaussian(); data = d)
        @test !any(==("(Intercept)"), fit0.coefnames[1].second)
    end

    @testset "sigma ~ z - 1" begin
        err = _refusal(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ z - 1)), Gaussian(); data = d))
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("`sigma`", msg)
        @test occursin("0 + x", msg)
        @test !occursin("-1 + x", msg)
    end

    @testset "(x - 1 | g) inside a random-effect bar" begin
        g = repeat(1:6, inner = 10)
        dg = (; y, x, z, g)
        err = _refusal(() -> drm(bf(@formula(y ~ x + (x - 1 | g)), @formula(sigma ~ 1)),
                                 Gaussian(); data = dg))
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("(0 + x | g)", msg)
        @test !occursin("-1 + x", msg)
        errn = _refusal(() -> drm(bf(@formula(y ~ x + (x - 1 | g)), @formula(sigma ~ 1)),
                                  NegBinomial2(); data = dg))
        @test errn isa ArgumentError
        @test occursin("(0 + x | g)", sprint(showerror, errn))
    end

    @testset "bivariate and cross-family routes" begin
        n = 60
        y1 = [1 + 0.4 * x[i] + 0.3cos(5i) for i in 1:n]
        y2 = [2 - 0.3 * x[i] + 0.3sin(11i) for i in 1:n]
        yc = [round(Int, 3 + sin(i) + cos(2i)) + 2 for i in 1:n]
        g = repeat(1:6, inner = 10)
        db = (; y1, y2, x, z, g)
        fbad = bf(mu1 = @formula(y1 ~ x - 1), mu2 = @formula(y2 ~ x),
                  sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                  rho12 = @formula(rho12 ~ 1))
        fbad_sigma = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                        sigma1 = @formula(sigma1 ~ z - 1), sigma2 = @formula(sigma2 ~ 1),
                        rho12 = @formula(rho12 ~ 1))
        fbad_bar = bf(mu1 = @formula(y1 ~ x + (x - 1 | g)), mu2 = @formula(y2 ~ x),
                      sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                      rho12 = @formula(rho12 ~ 1))
        fmix = bf(mu1 = @formula(y1 ~ x - 1), mu2 = @formula(yc ~ x),
                  sigma1 = @formula(sigma1 ~ 1))
        dmix = (; y1, yc, x)

        for fam in (Gaussian(), LogNormal(), Student())
            d_fam = fam isa LogNormal ? merge(db, (; y1 = exp.(y1), y2 = exp.(y2))) : db
            err = _refusal(() -> drm(fbad, fam; data = d_fam))
            @test err isa ArgumentError
            @test occursin("`mu1`", sprint(showerror, err))
            @test occursin("0 + x", sprint(showerror, err))
            errs = _refusal(() -> drm(fbad_sigma, fam; data = d_fam))
            @test errs isa ArgumentError
            @test occursin("`sigma1`", sprint(showerror, errs))
        end

        errb = _refusal(() -> drm(fbad_bar, Gaussian(); data = db))
        @test errb isa ArgumentError
        @test occursin("(0 + x | g)", sprint(showerror, errb))

        errm = _refusal(() -> drm(fmix, (Gaussian(), Poisson()); data = dmix))
        @test errm isa ArgumentError
        @test occursin("`mu1`", sprint(showerror, errm))
        @test occursin("0 + x", sprint(showerror, errm))
    end
end

end
