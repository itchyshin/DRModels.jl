using DRModels, Test, Random, Logging, DelimitedFiles

# The printed fit report names the final gradient norm when the optimiser did not converge: how far from
# stationary the reported point is. It is the same number `check_drm(fit).max_abs_grad` returns. A fit
# whose optimiser converged prints exactly what it printed before.
const _FRG_DIR = joinpath(@__DIR__, "fixtures", "corr_slope_retry")

function _frg_data(ds)
    raw, _ = readdlm(joinpath(_FRG_DIR, "data.csv"), ','; header = true)
    rows = String.(raw[:, 1]) .== ds
    return (; y = Float64.(raw[rows, 2]), x1 = Float64.(raw[rows, 3]), g = String.(raw[rows, 4]))
end

_frg_show(fit) = sprint(show, MIME"text/plain"(), fit)
_frg_value(txt) = parse(Float64, match(r"final gradient: max\|grad\| = ([0-9.e+-]+)", txt).captures[1])

@testset "a converged fit prints no gradient line" begin
    Random.seed!(7)
    x = randn(100); y = 0.3 .+ 0.5 .* x .+ exp.(0.2 .* x) .* randn(100)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + x)), Gaussian(); data = (; y, x))
    @test fit.converged
    @test !occursin("final gradient", _frg_show(fit))
end

@testset "an unconverged fit prints the final gradient norm, equal to check_drm's max_abs_grad" begin
    Random.seed!(7)
    x = randn(100); y = 0.3 .+ 0.5 .* x .+ exp.(0.2 .* x) .* randn(100)
    fit = with_logger(NullLogger()) do
        drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + x)), Gaussian(); data = (; y, x), control = DRMControl(maxit = 2))
    end
    @test !fit.converged
    txt = _frg_show(fit)
    @test occursin("converged = false", txt)           # the existing status line is still there
    @test occursin("Wald SEs withheld", txt)           # and so is the existing withheld-SE note
    @test occursin("final gradient: max|grad| = ", txt)
    rep = with_logger(NullLogger()) do; check_drm(fit); end
    @test _frg_value(txt) ≈ rep.max_abs_grad rtol = 1e-3
    @test rep.max_abs_grad > 1e-3                      # a stopped-early fit is visibly away from stationary
end

@testset "correlated-slope retry routes: check_drm reports a finite final gradient" begin
    f = bf(@formula(y ~ x1 + (1 + x1 | g)), @formula(sigma ~ 1))
    for (ds, fam) in (("gamma3", Gamma()), ("student8", Student()))
        fit = with_logger(NullLogger()) do; drm(f, fam; data = _frg_data(ds)); end
        rep = with_logger(NullLogger()) do; check_drm(fit); end
        @test isfinite(rep.max_abs_grad) && rep.grad_source in (:stored, :forward, :finite)
        txt = _frg_show(fit)
        # printed exactly when the optimiser flag is false, whatever the Julia version
        @test occursin("final gradient:", txt) == !fit.converged
        fit.converged || @test _frg_value(txt) ≈ rep.max_abs_grad rtol = 1e-3
    end
end
