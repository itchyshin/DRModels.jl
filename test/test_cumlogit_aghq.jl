# test_cumlogit_aghq.jl — CumulativeLogit `(1 | g)` / `(0 + x | g)` random-effect
# integrals by per-group ADAPTIVE Gauss–Hermite quadrature (src/adaptive_ghq.jl).
#
# Before this change both routes integrated b_g on a fixed 32-node PRIOR-scale grid
# (b = √2 σ_b z). With a large σ_b that grid is badly under-resolved: on the seed-24
# dataset below (n = 48, G = 6, `y ~ x + (1 | id)`) the old objective sat 1.36 nat
# BELOW the true marginal at its own optimum (18.95 vs 20.31), and adding more
# non-adaptive nodes did not converge (64 → 20.655, 128 → 21.032, 256 → 20.229).
#
# Reference values:
# * The true ML optimum, from an exact per-group QuadGK integral (rtol 1e-12) of the
#   marginal, maximised by Nelder–Mead + BFGS from four starts: nll = 19.83614244,
#   β = 7.997, cutpoints = (−4.152, 12.306), σ_b = 7.646. `ordinal::clmm` with
#   adaptive GHQ at nAGQ ≥ 30 reproduces it to 1e-8 (generated output only; no
#   ordinal/drmTMB source is used here).
# * The fixed-θ objective values in `_EXACT_*` below: the same QuadGK integral
#   (u = b/σ_b scale, integrand shifted by its grid maximum), agreeing to 12
#   digits between rtol = 1e-12 and 1e-14.
module TestCumlogitAGHQ

using DRModels
using Test
using DelimitedFiles: readdlm

# seed-24 fuzzer dataset (test_cumlogit_nan.jl DGP), embedded as literals.
const _Y = [1, 3, 3, 1, 3, 2, 2, 3, 2, 3, 2, 1, 1, 2, 2, 1, 2, 2, 1, 3, 2, 2, 2, 3,
            2, 2, 2, 2, 1, 1, 2, 3, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1]
const _X = [-1.5357326913335736, 1.378698870585084, 1.0412697158647648, -1.4457887904030167,
            1.0301562523297891, -0.19231229983888343, 0.46854445955942303, 0.8208240819360563,
            0.3218644510600864, 1.5958970670285182, -0.6347424024611212, -0.6944376142703856,
            -0.9289076488847438, 0.26570980339830835, 0.5380312176278805, -1.2307031599983758,
            0.3128256314901107, 0.06537413034469267, -2.0563956613411505, 0.6842887247778661,
            -0.5875911977687214, 0.11663019141094479, -0.16920238471317184, 1.181509571186613,
            -0.0727183112271787, 0.9256349595561054, -0.9401084857514107, 0.5482950918851354,
            -1.5552157217792446, -2.1066877154947847, 0.3058907811130988, 1.2942796748833747,
            -0.5609630864975566, -1.9339879401472766, 0.0394187537988978, 1.7081934945882222,
            -2.331860935079038, 1.3334081156948483, 2.236680510541948, 1.0750367414109951,
            0.24460452808777028, -0.8320547881488922, -2.329283365934041, 1.1841651576596224,
            -0.5231531828684736, 1.5303030654006202, -0.8505821743358403, -1.0365530376837637]
const _ID = repeat(1:6, inner = 8)

# θ = [β(x), δ1, δ2 (log cutpoint increment), log σ_b].
const _θ_MAIN   = [3.855077419302269, -0.8384295206269546, 1.8967583900573493, 1.214621917300004]   # origin/main θ̂
const _θ_GHQ32  = [11.208104226016133, -3.6482766241508, 3.1828555788124167, 2.6261729830755853]   # spurious GHQ-32 θ̂ (#873)
const _θ_EXACT  = [7.996679801630745, -4.152096389787965, 2.8008391027585384, 2.034150852403324]   # exact ML optimum

# Exact QuadGK marginal nll at those θ, `(1 | id)`.
const _EXACT_INT = [21.300119400926, 20.310904492802, 19.836142443365]
# Exact QuadGK marginal nll, `(0 + x | id)`, at two large-σ_b θ.
const _θ_SLOPE = ([0.5, -0.5, 0.0, log(3.0)], [2.0, -1.0, 0.5, log(8.0)])
const _EXACT_SLOPE = [52.350734837251, 50.507020810055]

_cuts(δ) = (c = similar(δ); c[1] = δ[1]; for k in 2:length(δ); c[k] = c[k-1] + exp(δ[k]); end; c)

@testset "CumulativeLogit random effects — adaptive GHQ" begin
    data = (; y = Float64.(_Y), x = _X, id = _ID)

    @testset "(a) (1 | id) fit reaches the exact / clmm ML optimum" begin
        fit = drm(bf(@formula(y ~ x + (1 | id))), CumulativeLogit(); data = data, se = false)
        @test fit.converged
        @test fit.nll(fit.theta) ≈ 19.83614244 atol = 1e-4
        @test -loglik(fit) ≈ 19.83614244 atol = 1e-4
        @test coef(fit, :mu)[1] ≈ 7.997 atol = 1e-2
        @test _cuts(coef(fit, :cutpoints)) ≈ [-4.152, 12.306] atol = 1e-2
        @test exp(coef(fit, :resd)[1]) ≈ 7.646 atol = 1e-2
    end

    @testset "(b) fixed-θ objective matches the exact per-group integral" begin
        fit = drm(bf(@formula(y ~ x + (1 | id))), CumulativeLogit(); data = data, se = false)
        for (θ, ref) in zip((_θ_MAIN, _θ_GHQ32, _θ_EXACT), _EXACT_INT)
            @test fit.nll(θ) ≈ ref atol = 1e-6
        end
        fs = drm(bf(@formula(y ~ x + (0 + x | id))), CumulativeLogit(); data = data, se = false)
        for (θ, ref) in zip(_θ_SLOPE, _EXACT_SLOPE)
            @test fs.nll(θ) ≈ ref atol = 1e-6
        end
    end

    # (c) Well-identified fixtures (18 and 15 obs/group, σ_b ≈ 0.69 / 0.34) were
    # already accurate under the old 32-node prior-scale grid, so switching the
    # integrator must move their fitted logLik only at integration-error scale.
    # Pre-change values: fitted logLik on the merged #846 + #873 base (GHQ-32).
    @testset "(c) well-identified fixtures: logLik moves only at integration-error scale" begin
        fx = joinpath(@__DIR__, "parity", "fixtures")
        load(dir) = begin
            raw, header = readdlm(joinpath(fx, dir, "data.csv"), ','; header = true)
            j = Dict(Symbol(strip(string(c))) => i for (i, c) in enumerate(vec(header)))
            (; id = string.(raw[:, j[:id]]),
               x = Float64[parse(Float64, string(v)) for v in raw[:, j[:x]]],
               y = Float64[parse(Float64, string(v)) for v in raw[:, j[:y_int]]])
        end
        for (dir, f, ll_ghq32) in (
                ("cumlogit-mu-ranef", @formula(y ~ x + (1 | id)), -1025.9199589398),
                ("cumlogit-mu-slope-ranef", @formula(y ~ x + (0 + x | id)), -775.2972690826))
            fit = drm(bf(f), CumulativeLogit(); data = load(dir), se = false)
            @test fit.converged
            @test abs(loglik(fit) - ll_ghq32) < 1e-3
        end
    end
end

end # module
