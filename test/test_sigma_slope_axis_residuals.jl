# A random slope on `sigma` is stored under `<group>_<slope>_logsigma`. Everything that decides from the
# SD name which axis a random effect lives on must read that name as a sigma-axis effect: the quantile
# residuals (which were integrating a phantom MEAN random intercept out of the fit), `repeatability` /
# `heritability`, and the variance-boundary diagnostic.
using DRModels, Test, DelimitedFiles, Statistics
import Distributions: Normal, quantile, cdf

const _SSA_M, _SSA_H = readdlm(joinpath(@__DIR__, "fixtures", "sigma_re", "sigma_re_data.csv"), ','; header = true)
const SSA = NamedTuple(Symbol(h) => (h in ("g", "g2", "sb", "flb") ? Int.(_SSA_M[:, i]) : Float64.(_SSA_M[:, i]))
                       for (i, h) in enumerate(vec(_SSA_H)))

const SSA_FORMS = Dict(
    :slope0 => bf(@formula(y_g ~ x1), @formula(sigma ~ x1 + (0 + x1 | g))),
    :slope0_const => bf(@formula(y_g ~ x1), @formula(sigma ~ 1 + (0 + x1 | g))),
    :corr => bf(@formula(y_g ~ x1), @formula(sigma ~ x1 + (1 + x1 | g))),
    :corr_const => bf(@formula(y_g ~ x1), @formula(sigma ~ 1 + (1 + x1 | g))))
const SSA_FITS = Dict(k => drm(f, Gaussian(); data = SSA) for (k, f) in SSA_FORMS)
msg(f) = try; f(); ""; catch e; sprint(showerror, e); end

@testset "sigma slope: quantile residuals are the population residuals" begin
    for k in keys(SSA_FITS)
        fit = SSA_FITS[k]
        # the residual clamps its probability to [eps, 1 - eps], i.e. |z| <= 8.2; clamp the reference alike
        zmax = abs(quantile(Normal(), eps()))
        pop = clamp.((SSA.y_g .- fit.means[:mu]) ./ fit.scales[:sigma], -zmax, zmax)
        rq = residuals(fit; type = :quantile)
        # compare on the probability scale: Phi^-1 loses digits in the far tail
        @test maximum(abs.(cdf.(Normal(), rq) .- cdf.(Normal(), pop))) < 1e-9
        @test abs(std(rq) - std(pop)) < 1e-2
    end
end

@testset "sigma slope: the SD name is read as a sigma-axis effect" begin
    fit = SSA_FITS[:slope0]
    @test DRModels._is_logsigma_re(fit, "g_x1_logsigma")
    @test DRModels._is_logsigma_re(SSA_FITS[:corr], "g_logsigma")
    @test DRModels._is_logsigma_re(SSA_FITS[:corr], "g_x1_logsigma")
    # a group that is not in the sigma formula stays a mean group, whatever its name ends in
    @test !DRModels._is_logsigma_re(fit, "g2_x1_logsigma")
    @test !DRModels._is_logsigma_re(fit, "g2_logsigma")
    @test !DRModels._is_logsigma_re(fit, "g_x1")
end

@testset "sigma slope: repeatability and the boundary diagnostic do not count it as a mean component" begin
    for k in (:slope0_const, :corr_const)
        fit = SSA_FITS[k]
        @test occursin("only on the scale", msg(() -> repeatability(fit)))
        @test DRModels._variance_boundary(fit) === nothing
    end
end
