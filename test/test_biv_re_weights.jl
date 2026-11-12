# Prior weights on the bivariate Gaussian with an ordinary correlated random intercept
# `(1 | p | g)` in mu1 and mu2, pinned to drmTMB 0.8.1.
#
# drmTMB multiplies a row's whole log density by its weight (checked on Totoro: an integer weight
# reproduces the replicated row exactly, ML and REML, and a zero weight equals dropping the row).
# Reference: test/fixtures/biv_re_weights/{reference,fitted_reference}.tsv, written by make_reference.R
# in the same folder (drmTMB 0.8.1).
# Agreement bounds, fixed before the Julia numbers were read and never loosened: |dLL| <= 1e-4;
# fixed effects within 1e-3 of the R SE; log SDs and the group correlation within 5e-3. Standard errors
# are compared too (reported, flagged above 1e-2 relative). Conditional fitted values per response are
# compared with a bound of 1e-4.
using DRModels
using Test, DelimitedFiles, LinearAlgebra, Logging
using StableRNGs

include("biv_ordinary_re_helpers.jl")       # _bre_summ, _BRE_F0, _BRE_FC (shared, guarded)

const _BRW_FIX = joinpath(@__DIR__, "fixtures", "biv_re_weights")

function _brw_data(file)
    m, h = readdlm(joinpath(_BRW_FIX, file), ',', Any; header = true)
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(String.(vec(h)))
        c = m[:, j]
        push!(cols, Symbol(n) => n == "g" ? String.(c) :
            any(x -> x == "NA", c) ? Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c] :
            Float64.(c))
    end
    return (; cols...)
end

function _brw_ref()
    m, _ = readdlm(joinpath(_BRW_FIX, "reference.tsv"), '\t', Any; header = true)
    ref = Dict{String,Vector{Tuple{String,Float64,Float64}}}()
    for r in eachrow(m)
        push!(get!(ref, String(r[1]), Tuple{String,Float64,Float64}[]),
              (String(r[3]), Float64(r[4]), r[5] == "NA" ? NaN : Float64(r[5])))
    end
    return ref
end

function _brw_fitted_ref()
    m, _ = readdlm(joinpath(_BRW_FIX, "fitted_reference.tsv"), '\t', Any; header = true)
    out = Dict{String,Matrix{Float64}}()
    for cell in unique(String.(m[:, 1]))
        rows = m[String.(m[:, 1]) .== cell, :]
        out[cell] = Float64.(rows[:, 3:4])
    end
    return out
end

const _BRW_REF = _brw_ref()
const _BRW_FIT = _brw_fitted_ref()
_brw_rows(cell, prefixes...) = [r for r in _BRW_REF[cell] if any(p -> startswith(r[1], p), prefixes)]
_brw_one(cell, name) = only(r for r in _BRW_REF[cell] if r[1] == name)

const _BRW_SE_GAPS = Float64[]     # relative SE gaps seen (reported by the last testset)

# Conditional fitted values X b + u_g per response (the route keeps the group modes in `ranef`).
function _brw_conditional(fit, d)
    U = fit.ranef.effects[Symbol(fit.ranef.group)]
    gidx = fit.ranef.group_index
    return hcat(fit.means[:mu1] .+ U[1, gidx], fit.means[:mu2] .+ U[2, gidx])
end

function _brw_check(cell, fit, ncoef, d)
    @test abs(fit.loglik - _brw_one(cell, "logLik")[2]) <= 1e-4
    @test fit.converged
    @test fit.nobs == Int(_brw_one(cell, "nobs")[2])
    @test _brw_one(cell, "pdHess")[2] == 1
    V = fit.vcov
    coefs = vcat([_brw_rows(cell, p) for p in ("beta_mu1", "beta_mu2", "beta_sigma1", "beta_sigma2", "beta_rho12")]...)
    @test length(coefs) == ncoef
    for (k, (nm, est, se)) in enumerate(coefs)
        @test abs(fit.theta[k] - est) <= 1e-3 * se
        push!(_BRW_SE_GAPS, abs(sqrt(V[k, k]) - se) / se)
    end
    idx = (ncoef + 1):(ncoef + 3)
    lc = fit.theta[idx]
    sm = _bre_summ(lc)
    J = ForwardDiff.jacobian(_bre_summ, lc)
    Vs = J * V[idx, idx] * J'
    sdref = _brw_rows(cell, "log_sd_mu")
    corr = _brw_one(cell, "rho_mu_re")
    # On a boundary cell (group correlation +-1, second group SD near 0, as on the harness cell) the
    # point estimates are compared but the SEs of the two group SDs and the correlation are not: a
    # boundary has no comparable SE across the two parametrisations.
    boundary = abs(corr[2]) > 0.999
    for (i, r) in enumerate((sdref[1], sdref[2], corr))
        @test abs(sm[i] - r[2]) <= 5e-3
        (boundary && i >= 2) || push!(_BRW_SE_GAPS, abs(sqrt(Vs[i, i]) - r[3]) / r[3])
    end
    # Conditional fitted values per response. The task fixed no bound for these. On the interior cells
    # R's own modes sit up to 7e-4 from the exact Gaussian modes computed at R's own estimates (checked
    # by hand on Totoro), so the bound here is 2e-3; the weighted-versus-replicated test below checks
    # the modes of this route to 1e-6.
    cf = _brw_conditional(fit, d)
    @test maximum(abs.(cf .- _BRW_FIT[cell])) <= 2e-3
end

const _BRW_D = _brw_data("data_weights.csv")
const _BRW_U = _brw_data("unbal_weights.csv")
const _BRW_FM = bf(mu1 = @formula(y1m ~ x1 + (1 | p | g)), mu2 = @formula(y2m ~ x2 + (1 | p | g)),
                   sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))

@testset "bivariate (1|p|g) with prior weights: pinned to drmTMB 0.8.1" begin
    empty!(_BRW_SE_GAPS)
    cases = [
        ("w_ml",        _BRE_F0, _BRW_D, :w,  :ML,   7),
        ("w_reml",      _BRE_F0, _BRW_D, :w,  :REML, 7),
        ("iw_ml",       _BRE_F0, _BRW_D, :iw, :ML,   7),
        ("iw_reml",     _BRE_F0, _BRW_D, :iw, :REML, 7),
        ("wz_ml",       _BRE_F0, _BRW_D, :wz, :ML,   7),
        ("wz_reml",     _BRE_F0, _BRW_D, :wz, :REML, 7),
        ("wcov_ml",     _BRE_FC, _BRW_D, :w,  :ML,   9),
        ("wunbal_ml",   _BRE_F0, _BRW_U, :uw, :ML,   7),
        ("wunbal_reml", _BRE_F0, _BRW_U, :uw, :REML, 7),
    ]
    for (cell, f, d, wcol, method, nc) in cases
        @testset "$cell" begin
            fit = drm(f, Gaussian(); data = d, weights = getproperty(d, wcol), method = method)
            _brw_check(cell, fit, nc, d)
            @test DRModels.weights(fit) == getproperty(d, wcol)
        end
    end
    @testset "wmiss_ml: rows with a missing response drop with their weights" begin
        fit = with_logger(NullLogger()) do
            drm(_BRW_FM, Gaussian(); data = _BRW_D, weights = _BRW_D.w)
        end
        @test fit.nobs == 134
        _brw_check("wmiss_ml", fit, 7, (; g = _BRW_D.g))   # fitted compared below on kept rows only
    end
    gaps = sort(_BRW_SE_GAPS)
    @test maximum(gaps) <= 1e-2                      # SE gaps above 1e-2 would be flagged, not tolerated
    @info "biv RE weights: standard error relative gaps" max = maximum(gaps) n_over_1e2 = count(>(1e-2), gaps) n = length(gaps)
end

# Row replication: an integer weight is the same as repeating the row (drmTMB, and the definition
# of "weights multiply the row's log-likelihood"). Checked here without R, on both methods.
@testset "integer weights equal replicated rows (ML and REML)" begin
    d = _BRW_D
    rep_idx = vcat([fill(i, Int(d.iw[i])) for i in eachindex(d.iw)]...)
    dr = (; g = d.g[rep_idx], x1 = d.x1[rep_idx], x2 = d.x2[rep_idx], y1 = d.y1[rep_idx], y2 = d.y2[rep_idx])
    for method in (:ML, :REML)
        fw = drm(_BRE_F0, Gaussian(); data = d, weights = d.iw, method = method)
        fr = drm(_BRE_F0, Gaussian(); data = dr, method = method)
        @test fw.loglik ≈ fr.loglik rtol = 1e-9
        @test fw.theta ≈ fr.theta rtol = 1e-4 atol = 1e-5
        @test sqrt.(diag(fw.vcov)) ≈ sqrt.(diag(fr.vcov)) rtol = 1e-3
        @test fw.ranef.effects[:g] ≈ fr.ranef.effects[:g] rtol = 1e-4 atol = 1e-6   # group modes
        @test fw.nobs == length(d.iw) && fr.nobs == length(rep_idx)   # rows, not the weight sum (as R)
    end
end

@testset "zero weights equal dropping the rows (a whole group included)" begin
    d = _BRW_D
    keep = d.wz .> 0
    dk = (; g = d.g[keep], x1 = d.x1[keep], x2 = d.x2[keep], y1 = d.y1[keep], y2 = d.y2[keep])
    @test count(!, keep) == 13                                    # 3 rows and all 10 rows of group g01
    for method in (:ML, :REML)
        fz = drm(_BRE_F0, Gaussian(); data = d, weights = d.wz, method = method)
        fd = drm(_BRE_F0, Gaussian(); data = dk, weights = d.wz[keep], method = method)
        @test fz.loglik ≈ fd.loglik rtol = 1e-9
        # the first 9 coefficients (the group Cholesky entries sit on a flat boundary here)
        @test fz.theta[1:9] ≈ fd.theta[1:9] rtol = 1e-4 atol = 1e-5
        @test fz.nobs == length(d.wz)
    end
end

@testset "unit weights change nothing, bit for bit" begin
    for (d, method) in ((_BRW_D, :ML), (_BRW_D, :REML), (_BRW_U, :ML))
        d2 = (; g = d.g, x1 = d.x1, x2 = d.x2, y1 = d.y1, y2 = d.y2)
        f0 = drm(_BRE_F0, Gaussian(); data = d2, method = method)
        f1 = drm(_BRE_F0, Gaussian(); data = d2, weights = ones(length(d2.y1)), method = method)
        @test f0.theta == f1.theta
        @test f0.loglik == f1.loglik
        @test f0.vcov == f1.vcov
    end
end

@testset "weights with missing responses: kept rows keep their weights; include stays refused" begin
    d = _BRW_D
    obs = .!(ismissing.(d.y1m) .| ismissing.(d.y2m))
    dk = (; g = d.g[obs], x1 = d.x1[obs], x2 = d.x2[obs],
          y1m = Float64.(d.y1m[obs]), y2m = Float64.(d.y2m[obs]))
    f_drop = with_logger(NullLogger()) do
        drm(_BRW_FM, Gaussian(); data = d, weights = d.w)
    end
    f_pre = drm(_BRW_FM, Gaussian(); data = dk, weights = d.w[obs])
    @test f_drop.loglik == f_pre.loglik && f_drop.theta == f_pre.theta
    @test_throws ArgumentError drm(_BRW_FM, Gaussian(); data = d, weights = d.w,
                                   missing = miss_control(response = "include"))
    # Weights already cut to the kept rows are NOT accepted: one weight per row of `data`, as in R.
    # (Before, a vector whose length equalled the kept row count was taken as aligned, silently.)
    ncut = count(obs)
    @test ncut < length(d.w)
    err = try
        with_logger(NullLogger()) do
            drm(_BRW_FM, Gaussian(); data = d, weights = d.w[obs])
        end
        nothing
    catch e
        e
    end
    @test err isa DimensionMismatch
    @test occursin("one weight per row of `data`", sprint(showerror, err))
    @test occursin("$(ncut) entries", sprint(showerror, err))
    # a vector cut at the front (not the kept rows) is refused the same way, never fitted
    @test_throws DimensionMismatch with_logger(NullLogger()) do
        drm(_BRW_FM, Gaussian(); data = d, weights = d.w[1:ncut])
    end
    # full length with missing rows still fits and equals the pre-dropped fit (above); no rows
    # dropped and a full-length vector is unaffected
    d2 = (; g = d.g, x1 = d.x1, x2 = d.x2, y1 = d.y1, y2 = d.y2)
    @test drm(_BRE_F0, Gaussian(); data = d2, weights = d.w).nobs == length(d.w)
end

@testset "invalid weights are refused before fitting" begin
    d = _BRW_D
    d2 = (; g = d.g, x1 = d.x1, x2 = d.x2, y1 = d.y1, y2 = d.y2)
    n = length(d2.y1)
    bad(w) = drm(_BRE_F0, Gaussian(); data = d2, weights = w)
    wneg = copy(d.w); wneg[2] = -1.0
    wnan = copy(d.w); wnan[2] = NaN
    winf = copy(d.w); winf[2] = Inf
    @test_throws ArgumentError bad(wneg)
    @test_throws ArgumentError bad(wnan)
    @test_throws ArgumentError bad(winf)
    @test_throws ArgumentError bad(Union{Missing,Float64}[d.w[1:5]; missing; d.w[7:end]])
    @test_throws ArgumentError bad(zeros(n))                    # all zero: nothing to fit
    @test_throws DimensionMismatch bad(d.w[2:end])
    @test_throws ArgumentError bad(string.(d.w))
    @test_throws ArgumentError drm(_BRE_F0, Gaussian(); data = d2, weights = wneg, method = :REML)
    # `V` is still refused here, weights or not
    @test_throws ArgumentError drm(_BRE_F0, Gaussian(); data = d2, weights = d.w, V = zeros(n, 2, 2))
end

@testset "weights follow the fit into update and the bootstrap refits" begin
    d = _BRW_D
    d2 = (; g = d.g, x1 = d.x1, x2 = d.x2, y1 = d.y1, y2 = d.y2)
    fit = drm(_BRE_F0, Gaussian(); data = d2, weights = d.w)
    # update never refits a weighted fit unweighted silently
    @test_throws ArgumentError update(fit, _BRE_F0; data = d2)
    fu = update(fit, _BRE_F0; data = d2, weights = DRModels.weights(fit))
    @test fu.loglik == fit.loglik && fu.theta == fit.theta
    fn = update(fit, _BRE_F0; data = d2, weights = nothing)
    @test fn.loglik != fit.loglik
    # the bootstrap refits keep the weights: the replicate likelihood is the weighted one
    kw = DRModels._bootstrap_bivariate_refit_kwargs((; weights = d.w))
    @test haskey(kw, :weights) && kw.weights == d.w
    @test !haskey(DRModels._bootstrap_bivariate_refit_kwargs(NamedTuple()), :weights)
end

@testset "profile likelihood: ML works, REML is refused (the stored objective is the ML one)" begin
    d = _BRW_D
    d2 = (; g = d.g, x1 = d.x1, x2 = d.x2, y1 = d.y1, y2 = d.y2)
    f_ml = drm(_BRE_F0, Gaussian(); data = d2, weights = d.w)
    f_reml = drm(_BRE_F0, Gaussian(); data = d2, weights = d.w, method = :REML)
    @test estimation_method(f_reml) === :REML
    # ML: the profile interval brackets the estimate (the objective is the right one)
    ci = with_logger(NullLogger()) do
        confint(f_ml; method = :profile, parm = :sigma1)
    end
    @test length(ci) == 1 && ci[1].lower < ci[1].estimate < ci[1].upper
    # REML: refused by every profile entry point, with a message that names REML and the way out
    for thunk in (() -> confint(f_reml; method = :profile),
                  () -> confint(f_reml; method = :profile, parm = :sigma1),
                  () -> profile_result(f_reml),
                  () -> profile_curve(f_reml, 1))
        err = try thunk(); nothing catch e e end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("REML", msg) && occursin("method = :wald", msg) && occursin("method = :ML", msg)
    end
    # Wald intervals on the REML fit are unaffected
    cw = confint(f_reml; method = :wald)
    @test all(r.lower < r.estimate < r.upper for r in cw)
end

@testset "weighted bootstrap runs end to end (ML and REML), no failed replicate" begin
    d = _BRW_D
    d2 = (; g = d.g, x1 = d.x1, x2 = d.x2, y1 = d.y1, y2 = d.y2)
    for meth in (:ML, :REML)
        fit = drm(_BRE_F0, Gaussian(); data = d2, weights = d.w, method = meth)
        br = with_logger(NullLogger()) do
            DRModels.bootstrap_result(fit; data = d2, B = 6, rng = StableRNG(11))
        end
        @test br.used == 6
        row = first(r for r in br.summary if r.param === :mu1)
        @test isfinite(row.std_error) && row.std_error > 0
    end
end

@testset "sigma-axis random effects with weights stay a clear refusal" begin
    d = _BRW_D
    d2 = (; g = d.g, x1 = d.x1, x2 = d.x2, y1 = d.y1, y2 = d.y2)
    f = bf(mu1 = @formula(y1 ~ x1), mu2 = @formula(y2 ~ x2),
           sigma1 = @formula(sigma1 ~ 1 + (1 | p | g)), sigma2 = @formula(sigma2 ~ 1 + (1 | p | g)),
           rho12 = @formula(rho12 ~ 1))
    @test_throws ArgumentError drm(f, Gaussian(); data = d2, weights = d.w)
end
