# Bivariate Gaussian with an ordinary correlated random intercept pair `(1 | p | g)` on sigma1 and
# sigma2 (log-sigma scale), Laplace, pinned to drmTMB 0.8.1.
#
# Reference: test/fixtures/biv_sigma_re/{reference,sigma_reference}.tsv, written by make_reference.R in
# the same folder from harness.csv (the gap-map v2 cell) and from sre_*.csv (gen_data.R, interior cells
# with a group SD of 0.35 and 0.25 on the log scale).
# Agreement bounds, fixed before the Julia numbers were read and never loosened: |dLL| <= 1e-4; fixed
# effects within 1e-3 of the R SE; log SDs and the transformed group correlation (atanh) within 5e-3.
# Standard errors are compared too (relative gap reported, flagged above 1e-2).
# Boundary note: on the harness cell drmTMB itself sits on a boundary (group correlation 1, SDs 0.048
# and 0.030, SE of the log SDs 3.5 and 4.5), so its log SDs and correlation are reported, not pinned.
using DRModels
using Test, DelimitedFiles, LinearAlgebra, Logging, ForwardDiff, StableRNGs, Statistics

const _BSR_FIX = joinpath(@__DIR__, "fixtures", "biv_sigma_re")

function _bsr_data(file)
    m, h = readdlm(joinpath(_BSR_FIX, file), ',', Any; header = true)
    return (; [Symbol(n) => (n == "g" ? String.(m[:, j]) : Float64.(m[:, j])) for (j, n) in enumerate(String.(vec(h)))]...)
end

function _bsr_ref(file = "reference.tsv")
    m, _ = readdlm(joinpath(_BSR_FIX, file), '\t', Any; header = true)
    ref = Dict{String,Vector{Tuple{String,Float64,Float64}}}()
    for r in eachrow(m)
        push!(get!(ref, String(r[1]), Tuple{String,Float64,Float64}[]),
              (String(r[3]), Float64(r[4]), r[5] == "NA" ? NaN : Float64(r[5])))
    end
    return ref
end

function _bsr_sigma_ref()
    m, _ = readdlm(joinpath(_BSR_FIX, "sigma_reference.tsv"), '\t', Any; header = true)
    out = Dict{String,Matrix{Float64}}()
    for cell in unique(String.(m[:, 1]))
        rows = m[String.(m[:, 1]) .== cell, :]
        out[cell] = Float64.(rows[:, 3:4])                   # conditional sigma1, sigma2 per row
    end
    return out
end

const _BSR_REF = _bsr_ref()
const _BSR_REF_STARTS = _bsr_ref("reference_starts.tsv")     # ss16: written by make_reference_starts.R
const _BSR_SIG = _bsr_sigma_ref()
_bsr_rows(cell, prefixes...) = [r for r in _BSR_REF[cell] if any(p -> startswith(r[1], p), prefixes)]
_bsr_one(cell, name) = only(r for r in _BSR_REF[cell] if r[1] == name)

const _BSR_F0 = bf(mu1 = @formula(y1 ~ x1), mu2 = @formula(y2 ~ x2),
                   sigma1 = @formula(sigma1 ~ 1 + (1 | p | g)), sigma2 = @formula(sigma2 ~ 1 + (1 | p | g)),
                   rho12 = @formula(rho12 ~ 1))
const _BSR_FC = bf(mu1 = @formula(y1 ~ x1), mu2 = @formula(y2 ~ x2),
                   sigma1 = @formula(sigma1 ~ x1 + (1 | p | g)), sigma2 = @formula(sigma2 ~ 1 + (1 | p | g)),
                   rho12 = @formula(rho12 ~ x1))

const _BSR_SE_GAPS = Float64[]

# (log SD1, log SD2, atanh group correlation) from the log-Cholesky entries.
function _bsr_summ(lc)
    L = DRModels.lc_to_chol(lc, 2).L
    S = Matrix(L * L')
    return [0.5 * log(S[1, 1]), 0.5 * log(S[2, 2]), atanh(S[1, 2] / sqrt(S[1, 1] * S[2, 2]))]
end

# Conditional sigma of every row from the fitted modes: exp(Xs c) times exp(u_g).
function _bsr_conditional_sigma(fit)
    U = fit.ranef.effects[Symbol(fit.ranef.group)]
    g = fit.ranef.group_index
    return hcat(fit.scales[:sigma1] .* exp.(U[1, g]), fit.scales[:sigma2] .* exp.(U[2, g]))
end

function _bsr_check(cell, fit, ncoef; boundary = false)
    @test abs(fit.loglik - _bsr_one(cell, "logLik")[2]) <= 1e-4
    @test fit.converged
    @test fit.nobs == Int(_bsr_one(cell, "nobs")[2])
    @test _bsr_one(cell, "pdHess")[2] == 1
    V = fit.vcov
    coefs = vcat([_bsr_rows(cell, p) for p in ("beta_mu1", "beta_mu2", "beta_sigma1", "beta_sigma2", "beta_rho12")]...)
    @test length(coefs) == ncoef
    for (k, (nm, est, se)) in enumerate(coefs)
        @test abs(fit.theta[k] - est) <= 1e-3 * se
        push!(_BSR_SE_GAPS, abs(sqrt(V[k, k]) - se) / se)
    end
    idx = (ncoef + 1):(ncoef + 3)
    lc = fit.theta[idx]
    sm = _bsr_summ(lc)
    J = ForwardDiff.jacobian(_bsr_summ, lc)
    Vs = J * V[idx, idx] * J'
    sdref = _bsr_rows(cell, "log_sd_sigma")
    eta = _bsr_one(cell, "eta_cor_sigma")
    for (i, r) in enumerate((sdref[1], sdref[2], eta))
        boundary && break                                   # reported, not pinned (see header)
        @test abs(sm[i] - r[2]) <= 5e-3
        push!(_BSR_SE_GAPS, abs(sqrt(Vs[i, i]) - r[3]) / r[3])
    end
    cs = _bsr_conditional_sigma(fit)
    @test maximum(abs.(cs .- _BSR_SIG[cell][:, 1:2])) <= (boundary ? 1e-3 : 5e-3)
    return sm
end

@testset "bivariate (1|p|g) on sigma1 and sigma2: pinned to drmTMB 0.8.1" begin
    empty!(_BSR_SE_GAPS)
    for (cell, f, file, nc) in (("bal", _BSR_F0, "sre_bal.csv", 7), ("unbal", _BSR_F0, "sre_unbal.csv", 7),
                                ("cov", _BSR_FC, "sre_cov.csv", 9))
        @testset "$cell" begin
            d = _bsr_data(file)
            fit = drm(f, Gaussian(); data = d)
            _bsr_check(cell, fit, nc)
            @test fit.ranef.kind === :ordinary_sigma
            @test Dict(fit.coefnames)[:groupcov] == ["Sigma_g:L11", "Sigma_g:L21", "Sigma_g:L22"]
            # the means carry no group effect
            @test fit.means[:mu1] ≈ hcat(ones(length(d.y1)), d.x1) * fit.theta[1:2]
        end
    end
    @testset "harness cell (boundary in drmTMB too)" begin
        d = _bsr_data("harness.csv")
        fit = with_logger(NullLogger()) do
            drm(_BSR_F0, Gaussian(); data = d)
        end
        sm = _bsr_check("harness", fit, 7; boundary = true)
        # the Laplace objective is the same function as drmTMB's: value at R's estimate
        # (checked by hand: 377.33867595589 against R's 377.3386760), and the same boundary mode is reached
        @test abs(exp(sm[1]) - _bsr_one("harness", "sd_sigma_re")[2]) < 2e-3
        @test abs(exp(sm[2]) - _bsr_rows("harness", "sd_sigma_re")[2][2]) < 2e-3
        @test tanh(sm[3]) > 0.9999
    end
    gaps = sort(_BSR_SE_GAPS)
    @test maximum(gaps) <= 1e-2                              # SE gaps above 1e-2 would be flagged, not tolerated
    @info "biv sigma RE: standard error relative gaps" max = maximum(gaps) n_over_1e2 = count(>(1e-2), gaps) n = length(gaps)
end

@testset "a failing start does not lose the fit" begin
    # ss16.csv (60 rows, 12 unbalanced groups): one or two of the three starts stop with an assertion
    # from the line search (two on Julia 1.13.3, one on 1.10.12). Before, the first failure aborted the whole fit although drmTMB fits the data
    # (logLik -150.7103) and the third start reaches the same optimum.
    d = _bsr_data("ss16.csv")
    ref = _BSR_REF_STARTS["ss16"]
    one(name) = only(r for r in ref if r[1] == name)
    rows(prefixes...) = [r for r in ref if any(p -> startswith(r[1], p), prefixes)]
    fit = @test_logs (:warn, r"[12] of 3 starts of the bivariate sigma random-effect fit stopped with a numerical error \(AssertionError\)") match_mode = :any drm(_BSR_F0, Gaussian(); data = d)
    @test fit.converged
    @test abs(fit.loglik - one("logLik")[2]) <= 1e-4
    @test fit.nobs == Int(one("nobs")[2])
    coefs = vcat([rows(p) for p in ("beta_mu1", "beta_mu2", "beta_sigma1", "beta_sigma2", "beta_rho12")]...)
    @test length(coefs) == 7
    for (k, (nm, est, se)) in enumerate(coefs)
        @test abs(fit.theta[k] - est) <= 1e-3 * se
    end
    sm = _bsr_summ(fit.theta[8:10])
    sdref = rows("log_sd_sigma")
    @test abs(sm[1] - sdref[1][2]) <= 5e-3
    @test abs(sm[2] - sdref[2][2]) <= 5e-3
    @test tanh(sm[3]) < -0.9999              # drmTMB is on the correlation = -1 boundary here too
    # only numerical failures cost a start; a programming error still surfaces
    @test DRModels._sre_numerical_failure(AssertionError("x"))
    @test DRModels._sre_numerical_failure(ArgumentError("matrix contains Infs or NaNs"))
    @test DRModels._sre_numerical_failure(DomainError(-1.0, "log"))
    @test !DRModels._sre_numerical_failure(MethodError(sin, ()))
    @test !DRModels._sre_numerical_failure(BoundsError([1], 2))
end

@testset "refusal text no longer calls the supported model unsupported" begin
    d = _bsr_data("sre_bal.csv")
    y1, y2 = @formula(y1 ~ x1), @formula(y2 ~ x2)
    s1p, s2p = @formula(sigma1 ~ 1 + (1 | p | g)), @formula(sigma2 ~ 1 + (1 | p | g))
    s1, s2 = @formula(sigma1 ~ 1), @formula(sigma2 ~ 1)
    r = @formula(rho12 ~ 1)
    msg(f, fam = Gaussian()) = try
        drm(f, fam; data = d); ""
    catch e
        e isa ArgumentError || rethrow()
        sprint(showerror, e)
    end
    cases = (
        "slope on the means" => bf(mu1 = @formula(y1 ~ x1 + (1 + x1 | p | g)), mu2 = @formula(y2 ~ x2 + (1 + x2 | p | g)), sigma1 = s1, sigma2 = s2, rho12 = r),
        "slope on the scales" => bf(mu1 = y1, mu2 = y2, sigma1 = @formula(sigma1 ~ 1 + (1 + x1 | p | g)), sigma2 = @formula(sigma2 ~ 1 + (1 + x2 | p | g)), rho12 = r),
        "unlabelled on the scales" => bf(mu1 = y1, mu2 = y2, sigma1 = @formula(sigma1 ~ 1 + (1 | g)), sigma2 = @formula(sigma2 ~ 1 + (1 | g)), rho12 = r),
    )
    for (label, f) in cases
        @testset "$label" begin
            m = msg(f)
            @test !isempty(m)
            @test occursin("bivariate q=4", m) && occursin("drmTMB", m)          # wording kept for #1094
            # the old text said this model is not supported and that drmTMB fits it
            @test !occursin("with ordinary correlated random effects is not supported", m)
            @test !occursin("drmTMB fits it", m)
            # the new text names the one supported shape and what else is refused
            @test occursin("intercept pair `(1 | p | g)`", m) && occursin("Gaussian family", m)
            @test occursin("slope terms", m) && occursin("Student-t", m)
        end
    end
    @testset "Student with the pair on the scales" begin
        f = bf(mu1 = y1, mu2 = y2, sigma1 = s1p, sigma2 = s2p, rho12 = r, nu = @formula(nu ~ 1))
        m = msg(f, Student())
        @test !isempty(m) && occursin("Student-t", m) && occursin("Gaussian family", m)
        @test !occursin("drmTMB fits it", m)
    end
end

@testset "everything outside the supported shape stays a clear refusal" begin
    d = _bsr_data("sre_bal.csv")
    f(m1, m2, s1, s2; rho = @formula(rho12 ~ 1)) = bf(mu1 = m1, mu2 = m2, sigma1 = s1, sigma2 = s2, rho12 = rho)
    y1, y2 = @formula(y1 ~ x1), @formula(y2 ~ x2)
    s1p, s2p = @formula(sigma1 ~ 1 + (1 | p | g)), @formula(sigma2 ~ 1 + (1 | p | g))
    s1, s2 = @formula(sigma1 ~ 1), @formula(sigma2 ~ 1)
    msg(e) = sprint(showerror, e)
    # means and scales together
    e = try drm(f(@formula(y1 ~ x1 + (1 | p | g)), @formula(y2 ~ x2 + (1 | p | g)), s1p, s2p), Gaussian(); data = d); nothing catch err; err end
    @test e isa ArgumentError && occursin("AND one on", msg(e))
    # REML, weights, known V
    e = try drm(f(y1, y2, s1p, s2p), Gaussian(); data = d, method = :REML); nothing catch err; err end
    @test e isa ArgumentError && occursin("REML", msg(e))
    @test_throws ArgumentError drm(f(y1, y2, s1p, s2p), Gaussian(); data = d, weights = fill(1.0, length(d.y1)))
    @test_throws ArgumentError drm(f(y1, y2, s1p, s2p), Gaussian(); data = d, V = zeros(length(d.y1), 2, 2))
    # one-sided, different labels, unlabelled, slope
    @test_throws ArgumentError drm(f(y1, y2, s1p, s2), Gaussian(); data = d)
    @test_throws ArgumentError drm(f(y1, y2, s1p, @formula(sigma2 ~ 1 + (1 | q | g))), Gaussian(); data = d)
    @test_throws ArgumentError drm(f(y1, y2, @formula(sigma1 ~ 1 + (1 | g)), @formula(sigma2 ~ 1 + (1 | g))), Gaussian(); data = d)
    @test_throws ArgumentError drm(f(y1, y2, @formula(sigma1 ~ 1 + (1 + x1 | p | g)), @formula(sigma2 ~ 1 + (1 + x1 | p | g))), Gaussian(); data = d)
    # a random effect on rho12 is not a thing
    @test_throws Exception drm(f(y1, y2, s1, s2; rho = @formula(rho12 ~ 1 + (1 | p | g))), Gaussian(); data = d)
    # keywords of other routes are refused, not ignored
    @test_throws ArgumentError drm(f(y1, y2, s1p, s2p), Gaussian(); data = d, K = Matrix(1.0I, 30, 30))
    # incomplete response rows with the include policy
    dm = merge(d, (; y1 = Union{Missing,Float64}[missing; d.y1[2:end]]))
    @test_throws ArgumentError drm(f(y1, y2, s1p, s2p), Gaussian(); data = dm, missing = miss_control(response = "include"))
end

@testset "simulate redraws the group effects on log sigma; plain simulate needs the data" begin
    d = _bsr_data("sre_bal.csv")
    fit = drm(_BSR_F0, Gaussian(); data = d)
    @test_throws ArgumentError simulate(fit)
    short = (; (k => getproperty(d, k)[1:end-1] for k in keys(d))...)
    @test_throws ArgumentError simulate(fit; data = short)
    gi = fit.ranef.group_index
    G = maximum(gi)
    # Balanced, 10 rows per group. The variance over groups of the log residual SD of mu1 is
    # Sigma_g[1,1] (the group effect) plus a sampling part that `re = :zero` draws also have, so the
    # difference of the two variances estimates Sigma_g[1,1]. 300 draws x 30 groups.
    function varlog(sim)
        v = Float64[]
        for s in sim
            r = s[:mu1] .- fit.means[:mu1]
            append!(v, [0.5 * log(mean(r[gi .== k] .^ 2)) for k in 1:G])
        end
        return var(v)
    end
    v1 = varlog(simulate(fit; data = d, nsim = 300, rng = StableRNG(7)))
    v0 = varlog(simulate(fit; re = :zero, nsim = 300, rng = StableRNG(8)))
    S11 = fit.ranef.Sigma_g[1, 1]
    @test abs((v1 - v0) - S11) <= 0.3 * S11
    # the bootstrap refits the sigma random-effect model on each redraw (a group-free refit would
    # give a log SD far below the fitted one)
    br = DRModels.bootstrap_result(fit; data = d, B = 6, rng = StableRNG(5))
    @test br.used >= 4
    lrow = first(r for r in br.summary if r.param === DRModels._BIV_RE_GROUP_BLOCK && endswith(r.coef, "L11"))
    @test lrow.std_error > 0 && lrow.upper - lrow.lower > 0.01
    # a row dropped for a missing response no longer lines up with the fit: refuse, never draw group-free
    dm = merge(d, (; y1 = Union{Missing,Float64}[missing; d.y1[2:end]]))
    fm = with_logger(NullLogger()) do
        drm(_BSR_F0, Gaussian(); data = dm)
    end
    @test_throws ArgumentError DRModels.bootstrap_result(fm; data = dm, B = 2, rng = StableRNG(1))
end
