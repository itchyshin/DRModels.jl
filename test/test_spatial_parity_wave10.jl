# Wave-10 spatial parity with drmTMB 0.8.1.
#
# (B) Poisson `spatial(1 | g)` with `coords` uses drmTMB's fixed-range kernel,
#     exp(-d / r) + 1e-6 I with r the median positive pairwise distance. It used to estimate
#     the range (a sixth parameter), which reached a logLik 1.23 above drmTMB's. Reference:
#     drmTMB 0.8.1 ML on the same data and coordinates (design note
#     docs/dev-log/2026-11-09-spatial-parity.md, rule written before the run):
#     |dLL| <= 1e-4, fixed effects within 1e-3 R standard errors, log SD within 5e-3.
# (A) NB2 `spatial(1 + x1 | p | g)` with `sd(g, spatial, coef(.)) ~ z` for both coefficients:
#     the default start stopped at a lower interior mode (-554.2127); the route now also
#     starts from SD-slope tilts of both blocks and keeps the best mode (-554.0242, drmTMB).
using DRModels, Test, DelimitedFiles, Random

const _SP_DIR = joinpath(@__DIR__, "fixtures", "spatial_parity")
const _SP_COORDS = let m = DelimitedFiles.readdlm(joinpath(@__DIR__, "fixtures", "gamma_sdz", "coords.csv"),
                                                  ',', Any; skipstart = 1)
    Dict(String(m[i, 1]) => [Float64(m[i, 2]), Float64(m[i, 3])] for i in 1:size(m, 1))
end

function _sp_data(dir, name)
    m, h = DelimitedFiles.readdlm(joinpath(dir, name), ',', Any; header = true)
    h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
    return (; y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")),
            x3 = Float64.(col("x3")), z = Float64.(col("z")), g = String.(col("g")))
end

@testset "Poisson coordinate spatial reproduces drmTMB's fixed-range fit" begin
    d = _sp_data(joinpath(@__DIR__, "fixtures", "nonconv_triage"), "data_poisson_spatial.csv")
    fit = drm(bf(@formula(y ~ x1 + x2 + x3 + spatial(1 | g))), Poisson(); data = d, coords = _SP_COORDS)
    @test fit.converged
    # drmTMB 0.8.1: 4 fixed effects and log SD (5 parameters; no range), SEs of the fixed effects
    ref = [0.31525651, 0.27482578, 0.02939719, -0.56410442, -0.58101362]
    se = [0.33134966, 0.03930448, 0.03784573, 0.03641343]
    @test length(fit.theta) == 5
    @test !any(p -> first(p) === :range, fit.blocks)
    @test loglikelihood(fit) ≈ -675.6946875075 atol = 1e-4
    @test all(abs.(fit.theta[1:4] .- ref[1:4]) .<= 1e-3 .* se)
    @test abs(fit.theta[5] - ref[5]) <= 5e-3
    # the bootstrap simulator draws from the kernel the fit used (no `:range` block to read)
    Mxy = reduce(vcat, [permutedims(_SP_COORDS[l]) for l in unique(d.g)])   # rows in first-seen order
    sim = DRModels._marginal_simulator(fit, d; coords = Mxy)
    ysim = sim(MersenneTwister(1))
    @test length(ysim) == length(d.y) && all(>=(0), ysim)
end

@testset "NB2 spatial q2 sd-z reaches the better interior mode" begin
    d = _sp_data(_SP_DIR, "data_nb2_spatial.csv")
    f = bf(@formula(y ~ x1 + x2 + x3 + spatial(1 + x1 | p | g)), @formula(sigma ~ 1),
           @formula(sd(g, spatial, coef(intercept)) ~ z), @formula(sd(g, spatial, coef(x1)) ~ z))
    fit = drm(f, NegBinomial2(); data = d, coords = _SP_COORDS, se = false)
    @test fit.converged
    @test loglikelihood(fit) ≈ -554.0242 atol = 1e-4        # drmTMB 0.8.1 with multi_start = 8
    # the lower mode the single default start used to stop at
    @test loglikelihood(fit) > -554.2127 + 0.1
end

@testset "Poisson spatial q2 sd-z (labelled): the sign-flipped mode is found" begin
    # simulated cell (G = 40 sites, 400 rows, seed 91015 of the wave-10 spatial-parity study):
    # the default and second starts stopped at -651.5954; a tilted start of the two SD blocks
    # reaches -651.4527
    d = _sp_data(_SP_DIR, "data_poisson_spatial_q2_sdz_seed91015.csv")
    f = bf(@formula(y ~ x1 + x2 + x3 + spatial(1 + x1 | p | g)),
           @formula(sd(g, spatial, coef(intercept)) ~ z), @formula(sd(g, spatial, coef(x1)) ~ z))
    fit = drm(f, Poisson(); data = d, coords = _SP_COORDS, se = false)
    @test loglikelihood(fit) > -651.5954 + 0.1
    # (the converged flag of this better mode is not asserted: its correlation sits on a ridge,
    # atanh rho about 8.5, and the flag differs between Julia 1.10 and 1.12)
end
