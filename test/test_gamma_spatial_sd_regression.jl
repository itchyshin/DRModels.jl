using DRModels, Test, Random, LinearAlgebra, Statistics
import Distributions
import DelimitedFiles

function _gamma_spatial_sd_fixture(; G = 48, m = 12, seed = 20261022)
    rng = MersenneTwister(seed)
    coords = hcat(mod.(0:(G - 1), 5), div.(0:(G - 1), 5))
    distances = [sqrt(sum(abs2, coords[i, :] .- coords[j, :]))
                 for i in 1:G, j in 1:G]
    range = sum(distances) / (G^2 - G)
    covariance = exp.(-distances ./ range) + 1e-8I
    group = repeat(1:G, inner = m)
    zgroup = [zeros(G ÷ 2); ones(G ÷ 2)][randperm(rng, G)]
    z = zgroup[group]
    x = randn(rng, G * m)
    alpha = [-0.7, 0.45]
    latent = cholesky(Symmetric(covariance)).L * randn(rng, G)
    eta = 0.25 .+ 0.2 .* x .+
          exp.(alpha[1] .+ alpha[2] .* zgroup)[group] .* latent[group]
    sigma = 0.32
    shape = 1 / sigma^2
    y = [rand(rng, Distributions.Gamma(shape, exp(v) / shape)) for v in eta]
    return (; data = (; y = Float64.(y), x, z, g = group), coords,
            covariance, group, zgroup, alpha, sigma)
end

@testset "Gamma spatial random-effect SD regression" begin
    fixture = _gamma_spatial_sd_fixture()
    data, coords = fixture.data, fixture.coords
    formula = bf(@formula(y ~ x + spatial(1 | g)),
                 @formula(sigma ~ 1),
                 @formula(sd(g, spatial) ~ z))
    fit = drm(formula, Gamma(); data, coords)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - fixture.alpha[2]) < 0.7
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(length(fixture.zgroup)),
                                             fixture.zgroup) * coef(fit, :sd))
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    @test all(isfinite, predict_parameters(fit, data; se = true)[:sd].se)

    constant_fit = drm(bf(@formula(y ~ x + spatial(1 | g)),
                          @formula(sigma ~ 1),
                          @formula(sd(g, spatial) ~ 1)),
                       Gamma(); data, coords, se = false)
    base_fit = drm(bf(@formula(y ~ x + spatial(1 | g)),
                      @formula(sigma ~ 1)), Gamma(); data, coords, se = false)
    @test constant_fit.loglik ≈ base_fit.loglik atol = 3e-5
    @test coef(constant_fit, :mu) ≈ coef(base_fit, :mu) atol = 1e-4
    @test coef(constant_fit, :sigma) ≈ coef(base_fit, :sigma) atol = 1e-4
end

# Kernel contract: the Gamma coordinate-spatial SD route must use the shared fixed
# native kernel exp(-d / median positive distance) + 1e-6 I, entered RAW, exactly as
# R/drmTMB does (`drm_spatial_coords_precision`). drmTMB 0.7.1 has no Gamma
# `spatial()` or `sd(site, spatial)` route, so the R reference is the equivalent
# `relmat(1 | site, K = K)` fit with that same K:
#   drmTMB(bf(y ~ x + relmat(1 | site, K = K), sigma ~ 1), family = Gamma(link = "log"))
# on test/data/gamma_spatial_sd_r_parity (committed data: set.seed(33), 30 sites x 15,
# jittered 6-wide grid, gamma shape 1/0.25^2). Reference numbers are R's, to 12 digits.
@testset "Gamma spatial SD route uses the shared R/drmTMB kernel" begin
    dir = joinpath(@__DIR__, "data", "gamma_spatial_sd_r_parity")
    d, _ = DelimitedFiles.readdlm(joinpath(dir, "data.csv"), ','; header = true)
    c, _ = DelimitedFiles.readdlm(joinpath(dir, "coords.csv"), ','; header = true)
    site = String.(d[:, 3])
    @test unique(site) == String.(c[:, 1])           # coords rows = first-seen site order
    coords = Matrix{Float64}(c[:, 2:3])
    data = (; y = Float64.(d[:, 1]), x = Float64.(d[:, 2]), g = site)

    fit = drm(bf(@formula(y ~ x + spatial(1 | g)), @formula(sigma ~ 1),
                 @formula(sd(g, spatial) ~ 1)), Gamma(); data, coords, se = false)
    @test fit.converged
    # R/drmTMB estimates at the optimum. The previous mean-distance / 1e-8 kernel
    # missed log-SD by 4.7e-3 on this data, so atol = 1e-4 separates the two kernels.
    @test coef(fit, :mu)[1]    ≈  0.637212904204 atol = 1e-4
    @test coef(fit, :mu)[2]    ≈  0.269266420157 atol = 1e-4
    @test coef(fit, :sigma)[1] ≈ -1.402274459947 atol = 1e-4
    @test coef(fit, :sd)[1]    ≈ log(0.445063656138) atol = 1e-4
    @test fit.loglik           ≈ -332.148261236 atol = 1e-4

    # Independent kernel identity: build the native K in the test and fit it as a
    # user-supplied relmat; the two routes must be the same model.
    D = [hypot(coords[i, 1] - coords[j, 1], coords[i, 2] - coords[j, 2])
         for i in axes(coords, 1), j in axes(coords, 1)]
    pos = [D[i, j] for i in axes(D, 1) for j in (i + 1):size(D, 2)]
    K = exp.(-D ./ Statistics.median(pos)) + 1e-6 * LinearAlgebra.I
    rfit = drm(bf(@formula(y ~ x + relmat(1 | g)), @formula(sigma ~ 1),
                  @formula(sd(g, relmat) ~ 1)), Gamma(); data, K, se = false)
    @test fit.loglik ≈ rfit.loglik atol = 1e-6
    @test coef(fit, :mu) ≈ coef(rfit, :mu) atol = 1e-5
    @test coef(fit, :sd) ≈ coef(rfit, :sd) atol = 1e-5
end
