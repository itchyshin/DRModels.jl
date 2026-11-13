# Shared helpers for the bivariate ordinary (1 | p | g) review tests
# (test_bivariate_ordinary_re_review.jl, test_bivariate_ordinary_re_large.jl).
# Reference: test/fixtures/biv_ordinary_re/reference_review.tsv, written by
# make_reference_review.R from the csv files gen_review_data.R writes (drmTMB 0.8.1).
using DRModels, ForwardDiff
using Test, DelimitedFiles, LinearAlgebra, Logging

if !isdefined(@__MODULE__, :_BRE_FIX)
const _BRE_FIX = joinpath(@__DIR__, "fixtures", "biv_ordinary_re")

function _bre_data(file)
    m, h = readdlm(joinpath(_BRE_FIX, file), ',', Any; header = true)
    return (; [Symbol(n) => (n == "g" ? String.(m[:, j]) : Float64.(m[:, j])) for (j, n) in enumerate(String.(vec(h)))]...)
end

function _bre_ref()
    m, _ = readdlm(joinpath(_BRE_FIX, "reference_review.tsv"), '\t', Any; header = true)
    ref = Dict{String,Vector{Tuple{String,Float64,Float64}}}()
    for r in eachrow(m)
        push!(get!(ref, String(r[1]), Tuple{String,Float64,Float64}[]),
              (String(r[3]), Float64(r[4]), r[5] == "NA" ? NaN : Float64(r[5])))
    end
    return ref
end

const _BRE_REF = _bre_ref()
_bre_rows(cell, prefixes...) = [r for r in _BRE_REF[cell] if any(p -> startswith(r[1], p), prefixes)]
_bre_one(cell, name) = only(r for r in _BRE_REF[cell] if r[1] == name)

const _BRE_F0 = bf(mu1 = @formula(y1 ~ x1 + (1 | p | g)), mu2 = @formula(y2 ~ x2 + (1 | p | g)),
                   sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
const _BRE_FC = bf(mu1 = @formula(y1 ~ x1 + (1 | p | g)), mu2 = @formula(y2 ~ x2 + (1 | p | g)),
                   sigma1 = @formula(sigma1 ~ x1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ x1))

# (log SD1, log SD2, group correlation) from the log-Cholesky entries.
function _bre_summ(lc)
    L = DRModels.lc_to_chol(lc, 2).L
    S = Matrix(L * L')
    return [0.5 * log(S[1, 1]), 0.5 * log(S[2, 2]), S[1, 2] / sqrt(S[1, 1] * S[2, 2])]
end

# Fixed bounds (never loosened): |dLL| <= 1e-4, fixed effects within 1e-3 of the R SE,
# SEs within 1e-3 relative, log SDs within 5e-3 (SE 1e-3 relative). The group
# correlation (interior here, so its SE is comparable too) is held to the same
# 5e-3 / 1e-3 pair; Julia's SEs for log SD and correlation are delta-method from the
# log-Cholesky block, R's are its own ADREPORT.
function _bre_check(cell, fit, ncoef)
    @test abs(fit.loglik - _bre_one(cell, "logLik")[2]) <= 1e-4
    @test fit.converged
    @test fit.nobs == Int(_bre_one(cell, "nobs")[2])
    @test _bre_one(cell, "pdHess")[2] == 1
    V = fit.vcov
    coefs = vcat([_bre_rows(cell, p) for p in ("beta_mu1", "beta_mu2", "beta_sigma1", "beta_sigma2", "beta_rho12")]...)
    @test length(coefs) == ncoef
    for (k, (nm, est, se)) in enumerate(coefs)
        @test abs(fit.theta[k] - est) <= 1e-3 * se
        @test isapprox(sqrt(V[k, k]), se; rtol = 1e-3)
    end
    idx = (ncoef + 1):(ncoef + 3)
    lc = fit.theta[idx]
    sm = _bre_summ(lc)
    J = ForwardDiff.jacobian(_bre_summ, lc)
    Vs = J * V[idx, idx] * J'
    sdref = _bre_rows(cell, "log_sd_mu")
    corr = _bre_one(cell, "rho_mu_re")
    for (i, r) in enumerate((sdref[1], sdref[2], corr))
        @test abs(sm[i] - r[2]) <= 5e-3
        @test isapprox(sqrt(Vs[i, i]), r[3]; rtol = 1e-3)
    end
end
end
