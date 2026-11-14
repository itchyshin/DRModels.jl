using DRModels, Test
import DelimitedFiles

# G7 (2026-11-10): the Tweedie crossed-intercept route `(1 | g) + (1 | g2)` fitted by sparse
# Laplace returned `stderror(fit) = Inf` for every coefficient on a converged fit. The inner
# fits run with `se = false` and hold the power fixed, so the assembled vcov was all NaN and
# `stderror` maps a non-identified variance to Inf. The route now builds one joint
# finite-difference Hessian of the same Laplace objective over (mu, sigma, both log SDs, the
# power). Point estimates are unchanged.
#
# R reference: drmTMB 0.8.1 `engine = "tmb"`, ML, on this exact data set (gap-map rerun cell
# `tweedie|crossed|lab0|q1|sd-none|cm0|ML`, `summary(fit$sdr, "fixed")`; pdHess TRUE,
# is_converged TRUE). R order: mu (4), sigma, nu, log SD g, log SD g2. Julia order: mu (4),
# sigma, log SD g, log SD g2, nu, so R values are listed in Julia order here.
# Measured on Totoro, Julia 1.13.3: largest relative SE difference from R 2.5e-7; the bound
# below is 1e-3, four orders of slack for BLAS and Brent noise, far below the 6.7 percent
# discrepancy the SE comparison was built to catch.
const _TWSE_LL_R = -724.7952281237
const _TWSE_EST_R = [0.259617294, 0.3363374143, 0.3101762133, -0.2623660826, 0.1761610162,
                     -0.4901962427, -1.062614477, 0.187657096]
const _TWSE_SE_R = [0.1313717016, 0.06088331537, 0.06087826348, 0.05738492507, 0.03402445053,
                    0.1643175372, 0.2705816916, 0.09864284159]

@testset "G7 Tweedie crossed intercepts: finite SEs that match drmTMB" begin
    m, h = DelimitedFiles.readdlm(joinpath(@__DIR__, "fixtures", "tweedie_crossed_se", "data.csv"),
                                  ',', Any; header = true)
    h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
    dat = (; y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")),
           x3 = Float64.(col("x3")), g = String.(col("g")), g2 = String.(col("g2")))
    fit = drm(bf(@formula(y ~ x1 + x2 + x3 + (1 | g) + (1 | g2))), Tweedie(); data = dat)
    @test fit.converged
    @test abs(loglikelihood(fit) - _TWSE_LL_R) < 1e-4
    est, se = coef(fit), stderror(fit)
    @test length(se) == 8
    @test all(isfinite, se)
    @test all(abs.(est .- _TWSE_EST_R) .<= 1e-3 .* _TWSE_SE_R)
    @test all(abs.(se ./ _TWSE_SE_R .- 1) .< 1e-3)
end

# A log SD that collapses to the floor (#1190 rule, as on the NB2 crossed route): the objective
# is flat in that coordinate, so it is held fixed at the boundary, the Hessian of the remaining
# coordinates is inverted, and the clamped coordinate gets a non-finite SE plus a boundary
# warning. Data: `clamped_sd.csv`, simulated crossed Tweedie data (seed 8; mu = 0.3 + 0.4 x +
# group-g effect, sd 0.35; NO effect of h; phi = 0.6, p = 1.6; n = 220, 12 g levels, 10 h
# levels), a data set on which the h log SD runs below the -8 clamp. No R reference: this
# pins the rule, not a number (R's SDs are unclamped).
@testset "G7 Tweedie crossed intercepts: a clamped log SD is held fixed, the rest keep SEs" begin
    m, h = DelimitedFiles.readdlm(joinpath(@__DIR__, "fixtures", "tweedie_crossed_se", "clamped_sd.csv"),
                                  ',', Any; header = true)
    h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
    dat = (; y = Float64.(col("y")), x = Float64.(col("x")), g = Int.(col("g")), h = Int.(col("h")))
    logger = Test.TestLogger(min_level = Base.CoreLogging.Warn)
    fit = Base.CoreLogging.with_logger(logger) do
        drm(bf(@formula(y ~ x + (1 | g) + (1 | h))), Tweedie(); data = dat)
    end
    est, se = coef(fit), stderror(fit)
    @test est[5] <= -8                                  # the h log SD sits at its floor
    @test any(r -> occursin("treated as fixed", r.message), logger.logs)
    free = [1, 2, 3, 4, 6]                              # mu (2), sigma, log SD g, power
    @test all(isfinite, se[free]) && all(>(0), se[free])
    @test !isfinite(se[5])
    # The free-coordinate SEs are the usual size (no blow-up from the flat direction).
    @test all(se[free] .< 1.0)
end
