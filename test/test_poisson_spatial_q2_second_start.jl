# Regression: Poisson `spatial(1 + x | g)` SD regression, scale-300 campaign seed 147.
# The default start (SD coefficients log 0.4 / log 0.25) converges to a stationary point with
# logLik -1574.1401, 1.02 below the best optimum; native drmTMB 0.8.1 reaches -1573.12098.
# The spatial route now also optimises from a second start (SD coefficients log 0.7) and keeps
# the better optimum if it improves the objective by more than 1e-6. The other 299 campaign
# datasets are unchanged bit for bit (checked outside this file).
using DRModels, Test, DelimitedFiles

function _ps147_data()
    dir = joinpath(@__DIR__, "fixtures", "poisson_spatial_q2_seed147")
    m, h = readdlm(joinpath(dir, "data.csv"), ',', Any; header = true)
    c = Dict(String(h[j]) => m[:, j] for j in eachindex(vec(h)))
    cm, _ = readdlm(joinpath(dir, "coords.csv"), ',', Any; header = true)
    coords = Dict(String(cm[k, 1]) => [Float64(cm[k, 2]), Float64(cm[k, 3])] for k in 1:size(cm, 1))
    d = (; y = Float64.(c["y"]), x = Float64.(c["x"]), z0 = Float64.(c["z0"]),
         z1 = Float64.(c["z1"]), g = String.(c["g"]))
    return d, coords
end

@testset "Poisson spatial q2 SD regression reaches the best optimum (seed 147)" begin
    d, coords = _ps147_data()
    f = bf(@formula(y ~ x + spatial(1 + x | g)),
           @formula(sd(g, spatial, coef(intercept)) ~ z0),
           @formula(sd(g, spatial, coef(x)) ~ z1))
    fit = drm(f, Poisson(); data = d, coords = coords)
    @test fit.converged
    # native drmTMB 0.8.1 optimum and coefficients (mu: 2, sd(intercept): 2, sd(x): 2)
    @test isapprox(loglikelihood(fit), -1573.120983211433; atol = 1e-6)
    ref = [0.329545514985337, 0.247556672360009, -1.91069007932451, 0.976855333091886,
           -0.96595870615427, -0.505103638589578]
    se = [0.0459594960465928, 0.144785330131505, 0.596860962792898, 0.567237269956709,
          0.215217839080549, 0.219515678047814]
    @test maximum(abs.(fit.theta .- ref) ./ se) < 1e-5
end
