# Offset cells that drmTMB 0.8.1 fits and Julia used to refuse (gap map v2 group "Offset limits"):
#   - two offsets, `y ~ x1 + offset(off) + offset(log(w))` (Poisson); R adds them;
#   - ZIP with an offset on the mean, `y ~ x1 + offset(off), zi ~ x2`;
#   - Gaussian `meta_V(vi)` with an offset on the mean.
# Plus one pin that an offset on an already supported route (Poisson, one offset) is applied on the
# linear-predictor scale exactly as drmTMB does (same coefficients and log-likelihood, fitted mean
# exp(Xb + off), predict(newdata) reads the offset from newdata).
# Reference: drmTMB 0.8.1, ML (the R version is also checked in ref_offset_routes.R; test/fixtures/offset_routes/reference_drmtmb.csv; script
# ref_offset_routes.R, same folder). Rule fixed before the Julia run:
#   |logLik diff| <= 1e-4; fixed effects within 1e-3 R standard errors; SEs within 1e-3 relative;
#   R reports is_converged() = TRUE and a positive-definite Hessian.
using DRModels, Test, DelimitedFiles, LinearAlgebra, StableRNGs, Statistics

const _OR_DIR = joinpath(@__DIR__, "fixtures", "offset_routes")

function _or_load()
    m, h = readdlm(joinpath(_OR_DIR, "data_offset_routes.csv"), ',', Any; header = true)
    h = String.(replace.(string.(vec(h)), "\"" => "")); col(n) = Float64.(m[:, findfirst(==(n), h)])
    d = (; (Symbol(n) => col(n) for n in ["x1", "x2", "off", "w", "y_pois", "y_zip", "vi", "yi"])...)
    r, rh = readdlm(joinpath(_OR_DIR, "reference_drmtmb.csv"), ',', Any; header = true)
    r = r[:, :]
    ref(cell) = (idx = findall(==(cell), String.(r[:, 1]));
                 (est = Float64.(r[idx, 3]), se = Float64.(r[idx, 4]), ll = Float64(r[idx[1], 5]),
                  ok = all(lowercase.(string.(r[idx, 6])) .== "true") && all(lowercase.(string.(r[idx, 7])) .== "true")))
    return d, ref
end

const _OR_D, _OR_REF = _or_load()

function _or_check(fit, ref)
    @test ref.ok                                        # R: is_converged() and pdHess
    @test fit.converged
    @test length(fit.theta) == length(ref.est)
    @test loglikelihood(fit) ≈ ref.ll atol = 1e-4
    @test all(abs.(fit.theta .- ref.est) .<= 1e-3 .* ref.se)
    se = sqrt.(diag(vcov(fit)))
    @test all(abs.(se .- ref.se) .<= 1e-3 .* ref.se)
end

@testset "Poisson, one offset: linear-predictor scale as drmTMB" begin
    ref = _OR_REF("pois_one")
    fit = drm(bf(@formula(y_pois ~ x1 + offset(off))), Poisson(); data = _OR_D)
    _or_check(fit, ref)
    X = hcat(ones(length(_OR_D.x1)), _OR_D.x1)
    @test fitted(fit) ≈ exp.(X * fit.theta .+ _OR_D.off) rtol = 1e-10
    nd = (; x1 = [0.3, -1.0], off = [0.5, -0.2])
    @test predict(fit, nd) ≈ exp.(fit.theta[1] .+ fit.theta[2] .* nd.x1 .+ nd.off) rtol = 1e-8
end

@testset "Poisson, two offsets are summed as in drmTMB" begin
    ref = _OR_REF("pois_two")
    fit = drm(bf(@formula(y_pois ~ x1 + offset(off) + offset(log(w)))), Poisson(); data = _OR_D)
    _or_check(fit, ref)
    one = drm(bf(@formula(y_pois ~ x1 + offset(s))), Poisson();
              data = merge(_OR_D, (; s = _OR_D.off .+ log.(_OR_D.w))))
    @test loglikelihood(fit) ≈ loglikelihood(one) atol = 1e-8
    # predict(newdata) reads both offsets and adds them
    nd = (; x1 = [0.3, -1.0], off = [0.5, -0.2], w = [2.0, 0.5])
    @test predict(fit, nd) ≈ exp.(fit.theta[1] .+ fit.theta[2] .* nd.x1 .+ nd.off .+ log.(nd.w)) rtol = 1e-8
    # a missing offset column is a clear error
    err = try; predict(fit, (; x1 = nd.x1, off = nd.off)); nothing; catch e; e; end
    @test err isa ArgumentError && occursin("offset column `w`", sprint(showerror, err))
end

@testset "ZIP with an offset on the mean" begin
    ref = _OR_REF("zip_off")
    fit = drm(bf(@formula(y_zip ~ x1 + offset(off)), @formula(zi ~ x2)), Poisson(); data = _OR_D)
    _or_check(fit, ref)
    # an offset is not an estimated column
    @test !any(occursin("offset", n) for (_, v) in fit.coefnames for n in v)
    # hurdle + offset stays refused; ZIP + offset + a random effect is refused below
    @test_throws ErrorException drm(bf(@formula(y_zip ~ x1 + offset(off)), @formula(hu ~ x2)), Poisson(); data = _OR_D)
    g = repeat(string.("g", 1:10), inner = 16)
    dg = merge(_OR_D, (; g))
    @test_throws ErrorException drm(bf(@formula(y_zip ~ x1 + offset(off) + (1 | g)), @formula(zi ~ x2)), Poisson(); data = dg)
    # predict gives the Poisson mean and moves with the offset
    @test predict(fit, _OR_D) ≈ exp.(fit.theta[1] .+ fit.theta[2] .* _OR_D.x1 .+ _OR_D.off) rtol = 1e-8   # the Poisson mean lambda
    nd = (; x1 = _OR_D.x1[1:5], x2 = _OR_D.x2[1:5], off = _OR_D.off[1:5])
    @test predict(fit, merge(nd, (; off = nd.off .+ 1.0))) ≈ exp(1.0) .* predict(fit, nd) rtol = 1e-8
    # simulate with newdata uses the offset: a large offset raises the simulated counts
    lo = simulate(fit; nsim = 20, rng = StableRNG(1), newdata = nd)
    hi = simulate(fit; nsim = 20, rng = StableRNG(1), newdata = merge(nd, (; off = nd.off .+ 3.0)))
    @test mean(hi) > 3 * mean(lo)
end

@testset "Gaussian meta_V with an offset on the mean" begin
    ref = _OR_REF("meta_off")
    fit = drm(bf(@formula(yi ~ x1 + offset(off) + meta_V(vi)), @formula(sigma ~ 1)), Gaussian(); data = _OR_D)
    _or_check(fit, ref)
    X = hcat(ones(length(_OR_D.x1)), _OR_D.x1)
    @test fitted(fit) ≈ X * fit.theta[1:2] .+ _OR_D.off rtol = 1e-10   # offset is in the fitted mean
    # predict(newdata) adds the offset from newdata to the fixed-effects mean
    nd = (; x1 = [0.3, -1.0], off = [0.5, -0.2])
    @test predict(fit, nd) ≈ fit.theta[1] .+ fit.theta[2] .* nd.x1 .+ nd.off rtol = 1e-8
    # REML with an offset and meta_V is refused (not verified against drmTMB; plain meta_V REML is refused too)
    e = try; drm(bf(@formula(yi ~ x1 + offset(off) + meta_V(vi)), @formula(sigma ~ 1)), Gaussian();
                 data = _OR_D, method = :REML); nothing; catch e; e; end
    @test e isa ArgumentError && occursin("REML", sprint(showerror, e))
end
