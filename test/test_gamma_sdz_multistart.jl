using DRModels, Test
import DelimitedFiles

# Gamma coordinate-SD (`sd-z`) two-field routes (q2: intercept + slope field, one SD
# formula `~ z` per field) have a second interior mode in the SD coefficients. From
# the single default start L-BFGS stopped at the lower one in wave-9 shapes B5 seeds 9
# and 10 (spatial), where drmTMB 0.8.1 (ML) sits at the higher one. The route now runs
# the shared `_q2_boundary_restarts` (the 6 alternative starts, accepted only on a
# clear objective gain) for every such fit and records the outcome in
# `fit.scales[:q2_restarts]` = [ran, winner (0 = default start), gain, converged flipped].
# R reference values: drmTMB 0.8.1 ML, wave9-gap-close `results/r_all.tsv`; thetaR is
# R's estimate in the Julia parameter order (mu, sigma, sd-intercept field, sd-slope field).

const _GSZ_DIR = joinpath(@__DIR__, "fixtures", "gamma_sdz")

function _gsz_data(name)
    m, h = DelimitedFiles.readdlm(joinpath(_GSZ_DIR, name), ',', Any; header = true)
    h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
    return (; y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")),
            x3 = Float64.(col("x3")), z = Float64.(col("z")), g = String.(col("g")))
end

function _gsz_coords()
    m = DelimitedFiles.readdlm(joinpath(_GSZ_DIR, "coords.csv"), ',', Any; skipstart = 1)
    return Dict(String(m[i, 1]) => [Float64(m[i, 2]), Float64(m[i, 3])] for i in 1:size(m, 1))
end

_gsz_spatial_formula() = bf(@formula(y ~ x1 + x2 + x3 + spatial(1 + x1 | g)), @formula(sigma ~ 1),
    @formula(sd(g, spatial, coef(intercept)) ~ z), @formula(sd(g, spatial, coef(x1)) ~ z))
_gsz_phylo_formula() = bf(@formula(y ~ x1 + x2 + x3 + phylo(1 + x1 | g)), @formula(sigma ~ 1),
    @formula(sd(g, phylogenetic, coef(intercept)) ~ z), @formula(sd(g, phylogenetic, coef(x1)) ~ z))

@testset "Gamma sd-z q2 spatial: restarts reach the higher mode (R's)" begin
    coords = _gsz_coords()
    fx = [(file = "gamma_spatial_s9.csv", llR = -386.7076467753,
           thetaR = [0.15393835, 0.22199977, 0.24365678, -0.15132059, -0.57665574,
                     -0.93143483, 0.28182764, -1.6300658, 0.62274397],
           ll_default = -387.16514912, theta_default = [0.14682, 0.0043009, 0.23792, -0.15139,
                     -0.5673, -0.94615, 0.28236, -1.8893, -0.50013]),
          (file = "gamma_spatial_s10.csv", llR = -777.2656163691,
           thetaR = [0.58764042, 0.34593523, 0.2183432, 0.083394883, -0.49595968,
                     -0.68821913, 0.36948332, -0.80996231, 0.29967219],
           ll_default = -778.67546729, theta_default = [1.2684, 0.35591, 0.22594, 0.085698,
                     -0.49481, -0.65795, -0.2515, -0.78178, 0.30594])]
    for f in fx
        d = _gsz_data(f.file)
        fit = drm(_gsz_spatial_formula(), DRModels.Gamma(); data = d, coords)
        @test fit.converged
        # the objective is the same one R evaluates: R's estimate gives R's log-likelihood
        @test -fit.nll(f.thetaR) ≈ f.llR atol = 1e-4
        # the default-start stop is a stationary point of this objective, 0.4 to 1.4 lower
        @test -fit.nll(f.theta_default) ≈ f.ll_default atol = 5e-3
        # the fit now sits at R's mode
        @test loglikelihood(fit) ≈ f.llR atol = 1e-4
        @test loglikelihood(fit) > f.ll_default + 0.4
        rs = fit.scales[:q2_restarts]
        @test rs[1] == 1.0 && rs[2] >= 1.0
        @test rs[3] ≈ loglikelihood(fit) - f.ll_default atol = 5e-3
        @test coef(fit, :mu) ≈ f.thetaR[1:4] atol = 5e-3
    end
end

@testset "Gamma sd-z q2 phylo: the higher mode is kept (R stops lower)" begin
    # wave-9 B2 seed 2: Julia's default start is already at the higher mode
    # (-479.0120) and drmTMB stops at -479.4689. Restarts must not move it.
    d = _gsz_data("gamma_phylo_s2.csv")
    tree = strip(read(joinpath(_GSZ_DIR, "tree.nwk"), String))
    fit = drm(_gsz_phylo_formula(), DRModels.Gamma(); data = d, tree)
    @test fit.converged
    @test loglikelihood(fit) ≈ -479.01204833 atol = 1e-4
    rs = fit.scales[:q2_restarts]
    @test rs[1] == 1.0 && rs[2] == 0.0 && rs[3] == 0.0 && rs[4] == 0.0
end

@testset "Gamma sd-one two-field fit pays no restart cost" begin
    d = _gsz_data("gamma_spatial_s9.csv")
    coords = _gsz_coords()
    f = bf(@formula(y ~ x1 + x2 + x3 + spatial(1 + x1 | g)), @formula(sigma ~ 1),
           @formula(sd(g, spatial, coef(intercept)) ~ 1), @formula(sd(g, spatial, coef(x1)) ~ 1))
    fit = drm(f, DRModels.Gamma(); data = d, coords, se = false)
    @test !haskey(fit.scales, :q2_restarts)
end
