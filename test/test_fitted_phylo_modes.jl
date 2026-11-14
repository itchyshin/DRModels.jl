# fitted() keeps the random-effect modes on the two phylo groups, and quantile residuals / sigma return used rows
# (maintainer decisions of 11 Nov, on PR #1324).
#
#   group 1: Gaussian mean with ONE structured intercept: relmat(1 | g), animal(1 | g), dense phylo(1 | g)
#   group 2: phylo(1 | g) Laplace fits for Poisson, NB2, Binomial, BetaBinomial, Gamma and Beta
#
# What is checked, per route:
#   1. fitted(fit) against drmTMB 0.8.1 fitted() at 1e-6 (both engines reach the same logLik, checked first),
#      fixtures: test/fixtures/p17_fitted/p17_fitted_phylo_reference.csv from p17_fitted_phylo_r.R (a Linux server).
#      drmTMB 0.8.1 refuses phylo() for beta_binomial, so BetaBinomial has no R value; it is pinned to an
#      independent recomputation only.
#   2. fitted(fit) against an independent recomputation from the fitted parameters alone: the BLUP from the dense
#      marginal covariance (group 1), the mode equations solved by Newton on the log-density of Distributions.jl,
#      differentiated with ForwardDiff (group 2). Nothing there calls the fitters' own mode solvers.
#   3. fitted(fit; re = :zero) is the population value; residuals(fit) = y - fitted(fit) on the used rows.
#   4. the stored fit (coefficients, logLik, vcov) is bit-identical to the commit before the change
#      (p20_prechange_fit_values_*.txt, from p20_capture.jl run at a6b12b0be).
#   5. a missing response (group 2 on the full data; group 1 refuses a missing response at fit, so its case is
#      the rows dropped, as drmTMB does by default).
using DRModels
using Test
using DelimitedFiles
using LinearAlgebra
using ForwardDiff
using StableRNGs
using Logging
import Distributions as Dist

# a quiet logger for the fits that warn about the dropped rows
with_logger_quiet(f) = Logging.with_logger(f, Logging.NullLogger())

include(joinpath(@__DIR__, "fixtures", "p17_fitted", "p20_cells.jl"))

function _pm_reference()
    m = readdlm(joinpath(_PM_DIR, "p17_fitted_phylo_reference.csv"), ',', Any; skipstart = 1)
    ref = Dict{String,NamedTuple{(:ll, :b1, :cond, :pop),Tuple{Float64,Float64,Vector{Float64},Vector{Float64}}}}()
    for id in unique(String.(m[:, 1]))
        r = m[String.(m[:, 1]) .== id, :]
        ref[id] = (ll = Float64(r[1, 3]), b1 = Float64(r[1, 4]), cond = Float64.(r[2:end, 3]),
                   pop = Float64[x isa Number ? Float64(x) : NaN for x in r[2:end, 4]])
    end
    return ref
end

# the rows with an observed response, as plain Float64 columns (what drmTMB drops by default)
function _pm_complete(d)
    keep = .!ismissing.(d.y_gm)
    dc = map(v -> v[keep], d)
    return merge(dc, (y_g_m = Float64.(dc.y_g_m), y_pois_m = Float64.(dc.y_pois_m), y_nb_m = Float64.(dc.y_nb_m),
                      y_gam_m = Float64.(dc.y_gam_m), y_beta_m = Float64.(dc.y_beta_m),
                      s_m = Float64.(dc.s_m), fl_m = Float64.(dc.fl_m)))
end

_pm_sigma_rows(fit) = DRModels._on_used_rows(fit, fit.scales[:sigma])   # the fitted scale, one value per used row

# ---------------------------------------------------------------------------------------------------
# independent recomputation, group 1: the BLUP from the dense marginal covariance
# ---------------------------------------------------------------------------------------------------
function _pm_gidx(labels)           # "g07" -> 7 (the rows of K, A and the tree are g01 ... g16)
    return [parse(Int, l[2:end]) for l in labels]
end

# tip-to-tip correlation of tree16.newick: a balanced binary tree of height 1, branches 0.25
function _pm_tree_correlation()
    C = zeros(16, 16)
    for a in 0:15, b in 0:15
        shared = count(l -> (a >> (4 - l)) == (b >> (4 - l)), 1:3)
        C[a+1, b+1] = a == b ? 1.0 : 0.25 * shared
    end
    return C
end

function _pm_blup_fitted(fit, d, y, Cg)
    X = hcat(ones(length(y)), d.x1)
    βμ = DRModels.coef(fit, :mu)
    σe = sigma(fit)
    σs = exp(fit.theta[first(DRModels._block_range(fit, :resd))])
    gi = _pm_gidx(d.g)
    Cz = Cg[gi, gi]
    r = y .- X * βμ
    V = σs^2 .* Cz + Diagonal(σe .^ 2)
    return X * βμ .+ σs^2 .* (Cz * (V \ r)), X * βμ
end

# ---------------------------------------------------------------------------------------------------
# independent recomputation, group 2: solve the mode equations directly
# ---------------------------------------------------------------------------------------------------
# row log-density at link-scale predictor η, from Distributions.jl, for each family
function _pm_logpdf(fam::Symbol, d, cols, σ)
    y = getproperty(d, cols[1])
    if fam === :poisson
        return (i, η) -> Dist.logpdf(Dist.Poisson(exp(η)), round(Int, y[i]))
    elseif fam === :nb2
        return (i, η) -> (r = 1 / σ[i]^2; μ = exp(η); Dist.logpdf(Dist.NegativeBinomial(r, r / (r + μ)), round(Int, y[i])))
    elseif fam === :gamma
        return (i, η) -> (α = 1 / σ[i]^2; μ = exp(η); Dist.logpdf(Dist.Gamma(α, μ / α), y[i]))
    elseif fam === :beta
        return (i, η) -> (φ = 1 / σ[i]^2; p = 1 / (1 + exp(-η)); Dist.logpdf(Dist.Beta(p * φ, (1 - p) * φ), y[i]))
    elseif fam === :binomial
        n = round.(Int, getproperty(d, cols[1]) .+ getproperty(d, cols[2]))
        return (i, η) -> Dist.logpdf(Dist.Binomial(n[i], 1 / (1 + exp(-η))), round(Int, y[i]))
    else # :betabinomial
        n = round.(Int, getproperty(d, cols[1]) .+ getproperty(d, cols[2]))
        return (i, η) -> (φ = 1 / σ[i]^2; p = 1 / (1 + exp(-η));
                          Dist.logpdf(Dist.BetaBinomial(n[i], p * φ, (1 - p) * φ), round(Int, y[i])))
    end
end
_pm_linkinv(fam::Symbol, η) = fam in (:poisson, :nb2, :gamma) ? exp(η) : 1 / (1 + exp(-η))

# Newton with step halving on f(b) = -sum_i log p(y_i | η_i + b[leaf(i)]) + 1/(2 σb²) b'Qb, from b = 0
function _pm_mode(logpdf, η0, leaf, Q, σb)
    Qd = Matrix(Q)
    f(b) = -sum(logpdf(i, η0[i] + b[leaf[i]]) for i in eachindex(η0)) + dot(b, Qd * b) / (2 * σb^2)
    b = zeros(size(Qd, 1))
    gnorm = Inf
    for _ in 1:100
        g = ForwardDiff.gradient(f, b)
        gnorm = maximum(abs, g)
        gnorm < 1e-11 && break
        H = ForwardDiff.hessian(f, b)
        step = H \ g
        f0 = f(b); α = 1.0
        while f(b .- α .* step) > f0 + 1e-13 && α > 1e-10
            α /= 2
        end
        b = b .- α .* step
    end
    H = ForwardDiff.hessian(f, b)
    return b, gnorm, eigmin(Symmetric(H))
end

function _pm_laplace_fitted(fit, fam, d, cols; tree = _pm_tree())
    X = hcat(ones(length(getproperty(d, cols[1]))), d.x1)
    βμ = DRModels.coef(fit, :mu)
    η0 = X * βμ
    σ = fam in (:nb2, :gamma, :beta, :betabinomial) ? _pm_sigma_rows(fit) : nothing
    σb = exp(fit.theta[first(DRModels._block_range(fit, :resd))])
    Q, leaf, _ = DRModels._poisson_phylo_setup(tree, d.g)
    b, gnorm, hmin = _pm_mode(_pm_logpdf(fam, d, cols, σ), η0, leaf, Q, σb)
    return [_pm_linkinv(fam, η0[i] + b[leaf[i]]) for i in eachindex(η0)],
           [_pm_linkinv(fam, η0[i]) for i in eachindex(η0)], gnorm, hmin
end

# id => (family symbol, response columns used by the independent log-density, complete-case data of the cell)
const _PM_FAM = Dict(
    "G2|poisson" => (:poisson, (:y_pois,)), "G2|nb2" => (:nb2, (:y_nb,)), "G2|binomial" => (:binomial, (:s, :fl)),
    "G2|betabinomial" => (:betabinomial, (:s, :fl)), "G2|gamma" => (:gamma, (:y_gam,)), "G2|beta" => (:beta, (:y_beta,)),
    "G2M|poisson" => (:poisson, (:y_pois_m,)), "G2M|nb2" => (:nb2, (:y_nb_m,)), "G2M|binomial" => (:binomial, (:s_m, :fl_m)),
    "G2M|betabinomial" => (:betabinomial, (:s_m, :fl_m)), "G2M|gamma" => (:gamma, (:y_gam_m,)), "G2M|beta" => (:beta, (:y_beta_m,)),
    "X|poisson|REML" => (:poisson, (:y_pois,)), "X|nb2|sigma=x1" => (:nb2, (:y_nb,)),
    "X|gamma|sigma=x1" => (:gamma, (:y_gam,)), "X|beta|sigma=x1" => (:beta, (:y_beta,)))
# the families whose response is a proportion or a count: the response column that `residuals` subtracts from
_pm_resp(fam, d, cols) = fam in (:binomial, :betabinomial) ? getproperty(d, cols[1]) ./ (getproperty(d, cols[1]) .+ getproperty(d, cols[2])) :
                                                          getproperty(d, cols[1])

const _PM_BOUND = 1e-6          # the bound used for every drmTMB comparison in test_fitted_conditional.jl

@testset "fitted(): the two phylo groups keep the random-effect modes (P20a)" begin
    d = _pm_mr(_pm_data())
    dc = _pm_complete(d)
    ref = _pm_reference()
    cells = Dict(id => f for (id, f) in _PM_CELLS)

    @testset "group 1 against drmTMB 0.8.1 and the dense BLUP" begin
        for id in ("G1|relmat", "G1|animal", "G1|phylo-dense")
            @testset "$id" begin
                fit = cells[id](d)
                r = ref[id]
                cond = fitted(fit); zero = fitted(fit; re = :zero)
                @test length(cond) == nobs(fit) == length(r.cond) == 160
                @test abs(fit.loglik - r.ll) < _PM_BOUND                          # the same fit as drmTMB's
                @test abs(DRModels.coef(fit, :mu)[1] - r.b1) < _PM_BOUND
                @test maximum(abs.(cond .- r.cond)) < _PM_BOUND                   # drmTMB fitted(fit)
                @test maximum(abs.(zero .- r.pop)) < _PM_BOUND                    # population value
                @test maximum(abs.(cond .- zero)) > 0.1                           # the modes are really in
                @test zero == fit.means[:mu]
                @test fitted(fit; re = :conditional) == cond
                # independent: BLUP from the dense marginal covariance at the fitted parameters
                Cg = id == "G1|relmat" ? _pm_K() : id == "G1|animal" ? _pm_A() : _pm_tree_correlation()
                bl, pop = _pm_blup_fitted(fit, d, d.y_g, Cg)
                @test maximum(abs.(cond .- bl)) < 1e-9
                @test maximum(abs.(zero .- pop)) < 1e-12
                # residuals are y - fitted on the used rows
                @test residuals(fit) == d.y_g .- cond
                @test residuals(fit; re = :zero) == d.y_g .- zero
            end
        end
    end

    @testset "group 1 with a missing response: the fit refuses, the dropped rows are the case (drmTMB default)" begin
        for id in ("G1M|relmat", "G1M|animal", "G1M|phylo-dense")
            @testset "$id" begin
                @test_throws ArgumentError cells[id](d)           # unchanged: the dense routes do not take a missing response
                fit = cells[id](dc)
                r = ref[id]
                cond = fitted(fit); zero = fitted(fit; re = :zero)
                @test length(cond) == nobs(fit) == length(r.cond) == 144
                @test abs(fit.loglik - r.ll) < _PM_BOUND
                @test maximum(abs.(cond .- r.cond)) < _PM_BOUND
                @test maximum(abs.(zero .- r.pop)) < _PM_BOUND
                # the dense phylo fit has its group SD at the boundary on these rows: no gap, in both engines
                id == "G1M|phylo-dense" || @test maximum(abs.(cond .- zero)) > 0.1
                Cg = id == "G1M|relmat" ? _pm_K() : id == "G1M|animal" ? _pm_A() : _pm_tree_correlation()
                bl, _ = _pm_blup_fitted(fit, dc, dc.y_g_m, Cg)
                @test maximum(abs.(cond .- bl)) < 1e-9
            end
        end
    end

    @testset "group 1 variants through the same fitter: REML and sigma ~ x1 (dense BLUP only)" begin
        for id in ("X|relmat|REML", "X|relmat|sigma=x1")
            fit = cells[id](d)
            cond = fitted(fit)
            bl, pop = _pm_blup_fitted(fit, d, d.y_g, _pm_K())
            @test maximum(abs.(cond .- bl)) < 1e-9
            @test maximum(abs.(fitted(fit; re = :zero) .- pop)) < 1e-12
            @test maximum(abs.(cond .- fitted(fit; re = :zero))) > 0.1
        end
    end

    @testset "group 2 against drmTMB 0.8.1 and the directly solved mode equations" begin
        for id in ("G2|poisson", "G2|nb2", "G2|binomial", "G2|betabinomial", "G2|gamma", "G2|beta",
                   "G2M|poisson", "G2M|nb2", "G2M|binomial", "G2M|betabinomial", "G2M|gamma", "G2M|beta")
            @testset "$id" begin
                fit = id in ("G2M|poisson", "G2M|nb2", "G2M|binomial", "G2M|betabinomial", "G2M|gamma", "G2M|beta") ?
                      with_logger_quiet(() -> cells[id](d)) : cells[id](d)
                fam, cols = _PM_FAM[id]
                du = startswith(id, "G2M") ? dc : d                 # the rows the fit used
                n = startswith(id, "G2M") ? 144 : 160
                cond = fitted(fit); zero = fitted(fit; re = :zero)
                @test length(cond) == length(zero) == nobs(fit) == n
                @test all(isfinite, cond)
                @test maximum(abs.(cond .- zero)) > 0.1                       # the modes are really in
                @test zero == fit.means[:mu][.!isnan.(fit.obs[:mu])]
                @test fitted(fit; re = :conditional) == cond
                # independent recomputation from the fitted parameters alone
                ind, indpop, gnorm, hmin = _pm_laplace_fitted(fit, fam, du, cols)
                @test gnorm < 1e-9 && hmin > 0                                 # the Newton solution is a mode
                @test maximum(abs.(cond .- ind)) < 1e-7
                @test maximum(abs.(zero .- indpop)) < 1e-9
                if haskey(ref, id)
                    r = ref[id]
                    @test length(r.cond) == n
                    @test abs(fit.loglik - r.ll) < _PM_BOUND                  # the same fit as drmTMB's
                    @test maximum(abs.(cond .- r.cond)) < _PM_BOUND           # drmTMB fitted(fit)
                    @test maximum(abs.(zero .- r.pop)) < _PM_BOUND
                end
                # residuals on the used rows
                @test residuals(fit) ≈ _pm_resp(fam, du, cols) .- cond atol = 1e-12
            end
        end
    end

    @testset "group 2 variants through the same fitters: REML, sigma ~ x1 (directly solved modes only)" begin
        for id in ("X|poisson|REML", "X|nb2|sigma=x1", "X|gamma|sigma=x1", "X|beta|sigma=x1")
            @testset "$id" begin
                fit = cells[id](d)
                fam, cols = _PM_FAM[id]
                cond = fitted(fit)
                ind, indpop, gnorm, hmin = _pm_laplace_fitted(fit, fam, d, cols)
                @test gnorm < 1e-9 && hmin > 0
                @test maximum(abs.(cond .- ind)) < 1e-7
                @test maximum(abs.(fitted(fit; re = :zero) .- indpop)) < 1e-9
                @test maximum(abs.(cond .- fitted(fit; re = :zero))) > 0.1
            end
        end
    end

    @testset "a missing response: the fit with the NA rows is the fit on the complete rows (group 2)" begin
        for (idm, id) in (("G2M|poisson", "G2|poisson"), ("G2M|nb2", "G2|nb2"), ("G2M|binomial", "G2|binomial"),
                          ("G2M|gamma", "G2|gamma"), ("G2M|beta", "G2|beta"), ("G2M|betabinomial", "G2|betabinomial"))
            fit = with_logger_quiet(() -> cells[idm](d))
            fc = cells[idm](dc)
            @test maximum(abs.(fitted(fit) .- fitted(fc))) < 1e-8
            @test maximum(abs.(fitted(fit; re = :zero) .- fitted(fc; re = :zero))) < 1e-8
            @test residuals(fit) ≈ residuals(fc) atol = 1e-8
        end
    end
end

# ---------------------------------------------------------------------------------------------------
# re = :zero is the population value; the routes of items 3 to 8 still refuse the conditional default
# ---------------------------------------------------------------------------------------------------
@testset "fitted(): other routes still refuse, with the same error" begin
    d = _pm_mr(_pm_data())
    nr(f) = (err = try fitted(f); nothing catch e; e end; err isa ArgumentError && occursin("re = :zero", sprint(showerror, err)))
    # a Poisson relmat (item 3) is not one of the two groups: it keeps no modes and refuses
    G, m = 15, 4
    rng = StableRNG(20261111)
    K = [0.5^abs(i - j) for i in 1:G, j in 1:G]
    dat = (; id = repeat(1:G, inner = m), x = randn(rng, G * m), y = rand(rng, 0:6, G * m))
    fit = drm(bf(@formula(y ~ x + relmat(1 | id))), Poisson(); data = dat, K = K)
    @test nr(fit)
    @test length(fitted(fit; re = :zero)) == G * m
    @test_throws ArgumentError residuals(fit)
    # ZeroOneBeta with a random intercept (item 4) and a crossed AGHQ Poisson (item 8)
    zf = drm(bf(@formula(y_zoib ~ x1 + (1 | g)), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)),
             ZeroOneBeta(); data = d)
    @test nr(zf)
    cf = drm(bf(@formula(y_pois ~ x1 + (1 | g2) + (1 | f2))), Poisson(); data = d, se = false, marginal = :AGHQ)
    @test nr(cf)
    # the relmat / animal / phylo routes of the other counts families (not in the two groups) are not touched:
    # a Poisson relmat above, and the Gaussian ones now answer
    @test !nr(drm(bf(@formula(y_g ~ x1 + relmat(1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, K = _pm_K()))
end

# ---------------------------------------------------------------------------------------------------
# the stored fit is untouched: bit-identical to the commit before the change (a6b12b0be)
# ---------------------------------------------------------------------------------------------------
@testset "stored fit values equal those of the pre-change commit (P20a: no fitted parameter changed)" begin
    d = _pm_mr(_pm_data())
    cells = Dict(id => f for (id, f) in _PM_CELLS)
    # The values were captured with Julia 1.13.3 and 1.10.12. Other Julia versions can differ in the
    # last bits, so they are compared with the 1.10.12 values at the loose bound below.
    tag = VERSION >= v"1.13" ? "113" : "110"
    begin
        lines = filter(!isempty, readlines(joinpath(_PM_DIR, "p20_prechange_fit_values_$tag.txt")))
        checked = String[]
        # The values were captured on one machine. Another CPU or BLAS build moves the last
        # bits of an optimised fit (seen on the GitHub runners: theta by up to 3e-8, logLik by 1e-10),
        # so the exact comparison is only meaningful on the capturing machine. Set
        # DRM_BITIDENTICAL_CHECK=1 there to compare with `==`; elsewhere the same values are checked
        # to 1e-6 (theta, vcov) and 1e-8 (logLik).
        exact = get(ENV, "DRM_BITIDENTICAL_CHECK", "") == "1"
        for ln in lines
            parts = split(ln, '\t')
            occursin("FIT FAILED", ln) && continue
            id = String(parts[1])
            val(k) = eval(Meta.parse(split(first(filter(p -> startswith(p, k * "="), parts)), '='; limit = 2)[2]))
            fit = with_logger_quiet(() -> cells[id](d))
            if exact
                @test fit.theta == val("theta")
                @test fit.loglik == val("loglik")
                @test Matrix(fit.vcov) == val("vcov")
            else
                @test maximum(abs.(fit.theta .- val("theta"))) < 1e-6
                @test abs(fit.loglik - val("loglik")) < 1e-8
                V0 = val("vcov")
                @test maximum(abs.(Matrix(fit.vcov) .- V0)) < 1e-6 * max(1.0, maximum(abs.(V0)))
            end
            push!(checked, id)
        end
        @test length(checked) == 21
        # every route of the two groups is among them
        @test all(id -> id in checked, ["G1|relmat", "G1|animal", "G1|phylo-dense", "G2|poisson", "G2|nb2", "G2|binomial",
                                        "G2|betabinomial", "G2|gamma", "G2|beta"])
    end
end

# ---------------------------------------------------------------------------------------------------
# residuals(type = :quantile) and sigma(fit) return the used rows
# ---------------------------------------------------------------------------------------------------
@testset "residuals(type = :quantile) and sigma(fit) return used rows (P20c)" begin
    d = _pm_mr(_pm_data())
    dc = _pm_complete(d)
    cells = Dict(id => f for (id, f) in _PM_CELLS)
    @testset "phylo Laplace fits with a missing response, against the fit on the complete rows" begin
        for (id, continuous) in (("G2M|poisson", false), ("G2M|nb2", false), ("G2M|binomial", false),
                                 ("G2M|betabinomial", false), ("G2M|gamma", true), ("G2M|beta", true))
            @testset "$id" begin
                fit = with_logger_quiet(() -> cells[id](d))
                fc = cells[id](dc)
                q = residuals(fit; type = :quantile, rng = StableRNG(1))
                @test length(q) == nobs(fit) == length(fitted(fit)) == 144
                @test all(isfinite, q)
                continuous && @test maximum(abs.(q .- residuals(fc; type = :quantile, rng = StableRNG(1)))) < 1e-6
                s = sigma(fit)
                if s isa AbstractVector
                    @test length(s) == 144
                    @test maximum(abs.(s .- sigma(fc))) < 1e-8
                end
            end
        end
    end
    @testset "fixed-effect Gaussian and ordinary random-effect fits" begin
        fx = drm(bf(@formula(y_gm ~ x1), @formula(sigma ~ x1)), Gaussian(); data = d)
        fxc = drm(bf(@formula(y_g_m ~ x1), @formula(sigma ~ x1)), Gaussian(); data = dc)
        @test length(sigma(fx)) == nobs(fx) == 144
        @test sigma(fx) ≈ sigma(fxc) atol = 1e-8
        @test residuals(fx; type = :quantile) ≈ residuals(fxc; type = :quantile) atol = 1e-8
        fr = drm(bf(@formula(y_g_m ~ x1 + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d)
        @test length(residuals(fr; type = :quantile)) == length(sigma(fr)) == length(fitted(fr)) == nobs(fr) == 144
        # no missing response: nothing changes (all rows, the very object stored)
        fa = drm(bf(@formula(y_g ~ x1), @formula(sigma ~ x1)), Gaussian(); data = d)
        @test sigma(fa) === fa.scales[:sigma]
        @test length(residuals(fa; type = :quantile)) == 160
    end
    @testset "every family: length equals nobs and fitted, rows line up with the complete-case fit" begin
        fams = [
            ("LogNormal", (dd) -> drm(bf(@formula(y_ln_m ~ x1), @formula(sigma ~ 1)), LogNormal(); data = dd), :y_ln),
            ("Student", (dd) -> drm(bf(@formula(y_st_m ~ x1), @formula(sigma ~ 1), @formula(nu ~ 1)), Student(); data = dd), :y_st),
            ("Poisson", (dd) -> drm(bf(@formula(y_pois_m ~ x1)), Poisson(); data = dd), :y_pois),
            ("NegBinomial2", (dd) -> drm(bf(@formula(y_nb_m ~ x1), @formula(sigma ~ 1)), NegBinomial2(); data = dd), :y_nb),
            ("Gamma", (dd) -> drm(bf(@formula(y_gam_m ~ x1), @formula(sigma ~ 1)), DRModels.Gamma(); data = dd), :y_gam),
            ("Beta", (dd) -> drm(bf(@formula(y_beta_m ~ x1), @formula(sigma ~ 1)), DRModels.Beta(); data = dd), :y_beta),
            ("Binomial", (dd) -> drm(bf(@formula(cbind(s_m, fl_m) ~ x1)), Binomial(); data = dd), :s),
            ("BetaBinomial", (dd) -> drm(bf(@formula(cbind(s_m, fl_m) ~ x1), @formula(sigma ~ 1)), BetaBinomial(); data = dd), :s),
            ("ZeroOneBeta", (dd) -> drm(bf(@formula(y_zoib_m ~ x1), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)), ZeroOneBeta(); data = dd), :y_zoib),
            ("CumulativeLogit", (dd) -> drm(bf(@formula(y_ord_m ~ x1)), CumulativeLogit(); data = dd), :y_ord),
        ]
        dd = merge(d, (y_ln_m = _pm_mask(d, :y_ln), y_st_m = _pm_mask(d, :y_st), y_zoib_m = _pm_mask(d, :y_zoib),
                       y_ord_m = _pm_mask(d, :y_ord)))
        keep = .!ismissing.(d.y_gm)
        ddc = map(v -> v[keep], dd)
        ddc = merge(ddc, (y_ln_m = Float64.(ddc.y_ln_m), y_st_m = Float64.(ddc.y_st_m), y_zoib_m = Float64.(ddc.y_zoib_m),
                          y_ord_m = Int.(ddc.y_ord_m), y_pois_m = Float64.(ddc.y_pois_m), y_nb_m = Float64.(ddc.y_nb_m),
                          y_gam_m = Float64.(ddc.y_gam_m), y_beta_m = Float64.(ddc.y_beta_m),
                          s_m = Float64.(ddc.s_m), fl_m = Float64.(ddc.fl_m)))
        for (name, fit_it, _) in fams
            @testset "$name" begin
                fit = with_logger_quiet(() -> fit_it(dd))
                fc = fit_it(ddc)
                @test nobs(fit) == 144
                @test length(fitted(fit)) == 144
                q = residuals(fit; type = :quantile, rng = StableRNG(7))
                @test length(q) == 144
                @test all(isfinite, q)
                s = sigma(fit)
                s isa AbstractVector && @test length(s) == 144
                # the same fit on the same rows gives the same sigma; a per-row entry has 144 values
                # (an entry that is not per row, such as the ordinal cutpoints, is the same short vector)
                sc = sigma(fc)
                s isa AbstractVector && @test maximum(abs.(s .- sc)) < 1e-6
                if s isa AbstractDict
                    @test Set(keys(s)) == Set(keys(sc))
                    for (k, v) in s
                        v isa AbstractVector || continue
                        @test length(v) == length(sc[k])
                        @test maximum(abs.(v .- sc[k])) < 1e-6
                    end
                end
            end
        end
    end
    @testset "the R bridge payload keeps all input rows (it reads the stored per-row values)" begin
        rng = StableRNG(646)
        n = 60
        x = randn(rng, n)
        y = 0.3 .+ 0.5 .* x .+ randn(rng, n)
        y[1:6] .= NaN
        out = drm_bridge(; formula = "y ~ x", family = "gaussian", data = (y = y, x = x))
        native = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (y = y, x = x))
        @test length(sigma(native)) == nobs(native) == 54
        @test length(out["sigma"]) == 60                  # unchanged: one value per input row
        @test length(out["fitted"]) == 60
        @test length(out["residuals"]) == 60
        @test out["sigma"] ≈ DRModels._sigma_all_rows(native)
    end
end
