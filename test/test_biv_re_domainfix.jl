# Regression test: the bivariate Gaussian with an ordinary correlated random intercept pair
# `(1 | p | g)` on mu1 and mu2 must not abort on data that drmTMB fits.
#
# Before the fix, about 1 in 8 ordinary simulated data sets threw `DomainError: log(-1.0)` (or a
# `SingularException`): a line-search trial with |rho| at its guard and a tiny sigma makes the row
# precision huge and nearly rank one, so `M2 = I + L' A L` is finite but numerically indefinite and
# `logdet` throws. Such a trial is now rejected (flat 1e18, as the sibling routes do) and the line
# search backs off. A second trial type, where `yVy - 2 b'c + b'H b` cancels to a large negative
# number, is rejected the same way (d42, REML: it used to give a non-converged "fit" with logLik
# 1.4e16 once the first rejection was in place).
#
# The value of a rejected trial matters too. It must be a finite constant (1e18): the Hager-Zhang line
# search asserts `isfinite` while it brackets, so a rejected trial that returns `Inf` or `NaN` makes
# it throw `AssertionError: isfinite(phi_c) ...` on some data. With every rejection value in this route
# replaced by `Inf` (or by `NaN`), r7080 (REML) and r7083 (REML) crash on Julia 1.13.3 and r7084 (ML and
# REML) crash on Julia 1.10.12 (the line-search version differs); with 1e18 all of them fit. r7080 (ML)
# crashed in the version of the fix where only the newly added rejections returned 1e18 and the older
# non-finite exits still returned `Inf`; it fits now that every rejection returns 1e18. The last
# testset below pins that.
#
# Data: test/fixtures/biv_re_domainfix/d{168,68,183,42}.csv (make_data.jl) and r{7080,7083,7084}.csv
# (make_data_rev.jl). Reference: reference.tsv (make_reference.R) and reference_rev.tsv
# (make_reference_rev.R) in the same folder, from real drmTMB 0.8.1 runs. Agreement bounds,
# fixed beforehand: |dLL| <= 1e-4; fixed effects within 1e-3 of the R SE; log SDs of the two group
# effects and atanh of the group correlation within 5e-3. d42 (REML) has a group correlation of 1 in
# drmTMB too, so there the correlation is only checked to exceed 0.9999 (its atanh is flat).
using DRModels
using Test, DelimitedFiles, LinearAlgebra, Logging

const _FIXD = joinpath(@__DIR__, "fixtures", "biv_re_domainfix")

function _df_data(name)
    m, h = readdlm(joinpath(_FIXD, name * ".csv"), ',', Any; header = true)
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(String.(vec(h)))
        c = m[:, j]
        push!(cols, Symbol(n) => n == "g" ? String.(c) : Float64.(c))
    end
    return (; cols...)
end

function _df_ref(file, ref = Dict{String,Vector{Tuple{String,Float64,Float64}}}())
    m, _ = readdlm(joinpath(_FIXD, file), '\t', Any; header = true)
    for r in eachrow(m)
        push!(get!(ref, String(r[1]), Tuple{String,Float64,Float64}[]),
              (String(r[3]), Float64(r[4]), r[5] == "NA" ? NaN : Float64(r[5])))
    end
    return ref
end

const _DF_REF = _df_ref("reference_rev.tsv", _df_ref("reference.tsv"))
_df_one(cell, name) = only(r for r in _DF_REF[cell] if r[1] == name)
_df_rows(cell, prefixes...) = [r for r in _DF_REF[cell] if any(p -> startswith(r[1], p), prefixes)]

const _DF_F = bf(mu1 = @formula(y1 ~ x1 + (1 | p | g)), mu2 = @formula(y2 ~ x2 + (1 | p | g)),
                 sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))

function _df_check(cell, data, method; corr_pinned = true)
    # no DomainError, no SingularException, no assertion from the line search
    fit = with_logger(NullLogger()) do
        drm(_DF_F, Gaussian(); data = data, method = method)
    end
    @test fit.converged
    @test isfinite(fit.loglik)
    @test abs(fit.loglik - _df_one(cell, "logLik")[2]) <= 1e-4
    @test fit.nobs == Int(_df_one(cell, "nobs")[2])
    coefs = vcat([_df_rows(cell, p) for p in ("beta_mu1", "beta_mu2", "beta_sigma1", "beta_sigma2", "beta_rho12")]...)
    @test length(coefs) == 7
    for (k, (nm, est, se)) in enumerate(coefs)
        @test abs(fit.theta[k] - est) <= 1e-3 * se
    end
    S = fit.ranef.Sigma_g
    sd = sqrt.(diag(S))
    sdref = _df_rows(cell, "log_sd_mu")
    @test abs(log(sd[1]) - sdref[1][2]) <= 5e-3
    @test abs(log(sd[2]) - sdref[2][2]) <= 5e-3
    rho = S[1, 2] / (sd[1] * sd[2])
    if corr_pinned
        @test abs(atanh(rho) - _df_one(cell, "eta_cor_mu")[2]) <= 5e-3
    else
        @test rho > 0.9999
    end
    return fit
end

@testset "bivariate ordinary (1 | p | g): no DomainError on data drmTMB fits" begin
    for s in (168, 68, 183)
        d = _df_data("d$s")
        @testset "d$s ML" begin _df_check("d$(s)_ml", d, :ML) end
        @testset "d$s REML" begin _df_check("d$(s)_reml", d, :REML) end
    end
    @testset "d42 REML (negative-sum-of-squares trial; correlation 1 in drmTMB)" begin
        _df_check("d42_reml", _df_data("d42"), :REML; corr_pinned = false)
    end
end

@testset "bivariate ordinary (1 | p | g): a rejected trial returns a finite value (not Inf, not NaN)" begin
    # With `Inf` or `NaN` as the rejection value r7080 REML and r7083 REML stop with an assertion from
    # the line search on Julia 1.13.3, and r7084 (ML and REML) on Julia 1.10.12. r7080 ML stops the same
    # way if some rejections return `Inf` and the others 1e18 (the state before the owner accepted
    # one value for every rejection). ML on r7083 is the ordinary neighbour.
    @testset "r7080 REML" begin _df_check("r7080_reml", _df_data("r7080"), :REML) end
    @testset "r7080 ML" begin _df_check("r7080_ml", _df_data("r7080"), :ML) end
    @testset "r7083 REML" begin _df_check("r7083_reml", _df_data("r7083"), :REML) end
    @testset "r7083 ML" begin _df_check("r7083_ml", _df_data("r7083"), :ML) end
    @testset "r7084 REML" begin _df_check("r7084_reml", _df_data("r7084"), :REML) end
    @testset "r7084 ML" begin _df_check("r7084_ml", _df_data("r7084"), :ML) end
end
