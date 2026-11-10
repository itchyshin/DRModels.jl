# Bivariate Gaussian with an ordinary correlated random intercept `(1 | p | g)` in
# mu1 and mu2, pinned to drmTMB 0.8.1 (`biv_gaussian()`, 2x2 group covariance kept
# separate from the residual rho12).
#
# Reference: test/fixtures/biv_ordinary_re/reference.tsv, written by make_reference.R
# in the same folder from data_biv_re.csv (gap-map v2 data, seed 20261109).
# Agreement bounds, fixed before the Julia numbers were read: |dLL| <= 1e-4; fixed
# effects within 1e-3 of the R SE; SEs within 1e-3 relative; log SD of the mu1
# group effect within 5e-3 (SE within 1e-3 relative); log SD of the mu2 group
# effect within 5e-3.
#
# Boundary note: on this data drmTMB's fit sits on the boundary (group correlation
# 0.99999994, mu2 group SD 0.018, pdHess TRUE only because TMB reports a huge SE
# for eta_cor). The mu2 log SD (R SE 3.5) and the correlation are therefore checked
# for the point estimate only; their SEs are not comparable across parametrisations.
using DRModels
using Test, DelimitedFiles, LinearAlgebra, Logging

const _FIX = joinpath(@__DIR__, "fixtures", "biv_ordinary_re")

function _re_data()
    m, h = readdlm(joinpath(_FIX, "data_biv_re.csv"), ',', Any; header = true)
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(String.(vec(h)))
        c = m[:, j]
        push!(cols, Symbol(n) => n == "g" ? String.(c) :
            any(x -> x == "NA", c) ? Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c] :
            Float64.(c))
    end
    return (; cols...)
end

function _re_ref()
    m, _ = readdlm(joinpath(_FIX, "reference.tsv"), '\t', Any; header = true)
    ref = Dict{String,Vector{Tuple{String,Float64,Float64}}}()
    for r in eachrow(m)
        push!(get!(ref, String(r[1]), Tuple{String,Float64,Float64}[]),
              (String(r[3]), Float64(r[4]), r[5] == "NA" ? NaN : Float64(r[5])))
    end
    return ref
end

_rows(ref, cell, prefixes...) = [r for r in ref[cell] if any(p -> startswith(r[1], p), prefixes)]
_one(ref, cell, name) = only(r for r in ref[cell] if r[1] == name)

const _D = _re_data()
const _REF = _re_ref()
const _F1 = bf(mu1 = @formula(y1 ~ x1 + (1 | p | g)), mu2 = @formula(y2 ~ x2 + (1 | p | g)),
               sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))

function _check(cell, fit, ncoef)
    ref = _REF[cell]
    @test abs(fit.loglik - _one(_REF, cell, "logLik")[2]) <= 1e-4
    @test fit.converged
    @test fit.nobs == Int(_one(_REF, cell, "nobs")[2])
    V = fit.vcov
    coefs = vcat([_rows(_REF, cell, p) for p in ("beta_mu1", "beta_mu2", "beta_sigma1", "beta_sigma2", "beta_rho12")]...)   # R lists REML rows in another order
    @test length(coefs) == ncoef
    for (k, (nm, est, se)) in enumerate(coefs)
        @test abs(fit.theta[k] - est) <= 1e-3 * se
        @test isapprox(sqrt(V[k, k]), se; rtol = 1e-3)
    end
    L = fit.ranef.Sigma_g
    sd = sqrt.(diag(L))
    sdref = _rows(_REF, cell, "log_sd_mu")
    @test abs(log(sd[1]) - sdref[1][2]) <= 5e-3
    @test abs(log(sd[2]) - sdref[2][2]) <= 5e-3
    k1 = ncoef + 1                       # log L11 = log SD1
    @test isapprox(sqrt(V[k1, k1]), sdref[1][3]; rtol = 1e-3)
    @test L[1, 2] / (sd[1] * sd[2]) > 0.9999
    @test _one(_REF, cell, "rho_mu_re")[2] > 0.9999
end

@testset "bivariate ordinary (1 | p | g), pinned to drmTMB" begin
    with_logger(NullLogger()) do
        @testset "ML, rho12 ~ 1" begin
            _check("ml_rho1", drm(_F1, Gaussian(); data = _D), 7)
        end
        @testset "ML, rho12 ~ x1" begin
            f = bf(mu1 = @formula(y1 ~ x1 + (1 | p | g)), mu2 = @formula(y2 ~ x2 + (1 | p | g)),
                   sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ x1))
            _check("ml_rhox1", drm(f, Gaussian(); data = _D), 8)
        end
        @testset "ML, sigma1 ~ x1" begin
            f = bf(mu1 = @formula(y1 ~ x1 + (1 | p | g)), mu2 = @formula(y2 ~ x2 + (1 | p | g)),
                   sigma1 = @formula(sigma1 ~ x1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
            _check("ml_sig1x1", drm(f, Gaussian(); data = _D), 8)
        end
        @testset "REML" begin
            fit = drm(_F1, Gaussian(); data = _D, method = :REML)
            _check("reml_rho1", fit, 7)
        end
        @testset "missing responses: default drops incomplete rows, as drmTMB does" begin
            f = bf(mu1 = @formula(y1m ~ x1 + (1 | p | g)), mu2 = @formula(y2m ~ x2 + (1 | p | g)),
                   sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
            _check("miss_drop", drm(f, Gaussian(); data = _D), 7)
        end
    end

    @testset "unsupported variants stay refusals" begin
        mk(m1, m2, s1 = @formula(sigma1 ~ 1), s2 = @formula(sigma2 ~ 1)) =
            bf(mu1 = m1, mu2 = m2, sigma1 = s1, sigma2 = s2, rho12 = @formula(rho12 ~ 1))
        # random effect on sigma1 and sigma2
        @test_throws ArgumentError drm(mk(@formula(y1 ~ x1), @formula(y2 ~ x2),
            @formula(sigma1 ~ 1 + (1 | p | g)), @formula(sigma2 ~ 1 + (1 | p | g))), Gaussian(); data = _D)
        # one-sided and unlabelled terms
        @test_throws ArgumentError drm(mk(@formula(y1 ~ x1 + (1 | p | g)), @formula(y2 ~ x2)), Gaussian(); data = _D)
        @test_throws ArgumentError drm(mk(@formula(y1 ~ x1 + (1 | g)), @formula(y2 ~ x2 + (1 | g))), Gaussian(); data = _D)
        # different labels or groups on the two axes
        @test_throws ArgumentError drm(mk(@formula(y1 ~ x1 + (1 | p | g)), @formula(y2 ~ x2 + (1 | q | g))), Gaussian(); data = _D)
        # partly observed rows (missing = include) are not supported on this route
        @test_throws ArgumentError drm(mk(@formula(y1m ~ x1 + (1 | p | g)), @formula(y2m ~ x2 + (1 | p | g))),
                                       Gaussian(); data = _D, missing = miss_control(response = "include"))
        # row weights (grouped likelihoods do not take them yet)
        @test_throws ArgumentError drm(_F1, Gaussian(); data = _D, weights = fill(1.0, length(_D.y1)))
    end
end
