# fitted() default: random-effect modes included, used rows only, as drmTMB 0.8.1 (P17a, P17b).
#
# `fitted(fit)` is conditional (`re = :conditional`): it adds the fitted random-effect modes, as
# drmTMB's `fitted()` does. `fitted(fit; re = :zero)` is the population-level value `fitted(fit)`
# returned before the default changed. With a missing response only the used rows are returned.
#
# Reference values: test/fixtures/p17_fitted/p17_fitted_r_reference.csv, written by
# test/fixtures/p17_fitted/p17_fitted_r.R (drmTMB 0.8.1, R 4.6.1, run on a Linux server) from the data in
# test/fixtures/p17_fitted/data.csv. Nothing here draws random numbers except the synthetic
# checks at the end, which use StableRNGs.
using DRModels
using Test
using DelimitedFiles
using StableRNGs
using Statistics
using LinearAlgebra

const _FC_DIR = joinpath(@__DIR__, "fixtures", "p17_fitted")
const _FC_FACS = Set(["g", "g2", "h", "f", "f2", "gna", "gu", "y_cat", "y_ordf"])

function _fc_data()
    m, h = readdlm(joinpath(_FC_DIR, "data.csv"), ',', Any; header = true)
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(String.(vec(h)))
        c = m[:, j]
        v = if n in _FC_FACS
            any(x -> x == "NA", c) ? Union{Missing,String}[x == "NA" ? missing : String(x) for x in c] : String.(c)
        elseif any(x -> x == "NA", c)
            Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c]
        elseif all(x -> x isa Integer, c)
            Int.(c)
        else
            Float64.(c)
        end
        push!(cols, Symbol(n) => v)
    end
    return (; cols...)
end

function _fc_reference(file = "p17_fitted_r_reference.csv")
    m = readdlm(joinpath(_FC_DIR, file), ',', Any; skipstart = 1)
    ref = Dict{String,NamedTuple{(:cond, :pop),Tuple{Vector{Float64},Vector{Float64}}}}()
    for id in unique(String.(m[:, 1]))
        keep = String.(m[:, 1]) .== id
        ref[id] = (cond = Float64.(m[keep, 3]),
                   pop = Float64[x isa Number ? Float64(x) : NaN for x in m[keep, 4]])
    end
    return ref
end

# (cell id, fitting closure). The ids are the gap-map cell keys (the 17 cells of P17a, the 4 of P17b).
const _FC_CELLS = [
    ("C|gaussian|(1|g)+weights", d -> drm(bf(@formula(y_g ~ x1 + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, weights = d.w)),
    ("C|gaussian|(1|g)+weights+REML", d -> drm(bf(@formula(y_g ~ x1 + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, weights = d.w, method = :REML)),
    ("H|meta|(1|g)", d -> drm(bf(@formula(yi ~ x1 + (1 | g) + meta_V(vi)), @formula(sigma ~ 1)), Gaussian(); data = d)),
    ("H|meta|(1|g)+(1|g2)", d -> drm(bf(@formula(yi ~ x1 + (1 | g) + (1 | g2) + meta_V(vi)), @formula(sigma ~ 1)), Gaussian(); data = d)),
    ("H|meta|(1|g)|REML", d -> drm(bf(@formula(yi ~ x1 + (1 | g) + meta_V(vi)), @formula(sigma ~ 1)), Gaussian(); data = d, method = :REML)),
    ("I|cumul|mu=x1+(1|g)+(1|g2)", d -> drm(bf(@formula(y_ord ~ x1 + (1 | g) + (1 | g2))), CumulativeLogit(); data = d)),
    ("J|poisson|(1|g)+(1|g2)+(1|h)", d -> drm(bf(@formula(y_pois ~ x1 + (1 | g) + (1 | g2) + (1 | h))), Poisson(); data = d)),
    ("J|gaussian|mu=(1|g)|sigma=x1", d -> drm(bf(@formula(y_g ~ x1 + (1 | g)), @formula(sigma ~ x1)), Gaussian(); data = d)),
    ("J|lognormal|mu=(1|g)|sigma=x1", d -> drm(bf(@formula(y_ln ~ x1 + (1 | g)), @formula(sigma ~ x1)), LogNormal(); data = d)),
    ("J|gaussian|group=numeric code gn", d -> drm(bf(@formula(y_g ~ x1 + (1 | gn)), @formula(sigma ~ 1)), Gaussian(); data = d)),
    ("J|gaussian|group=unbalanced with singletons gu", d -> drm(bf(@formula(y_g ~ x1 + (1 | gu)), @formula(sigma ~ 1)), Gaussian(); data = d)),
    ("J|gaussian|group has 2 levels (1|f2)", d -> drm(bf(@formula(y_g ~ x1 + (1 | f2)), @formula(sigma ~ 1)), Gaussian(); data = d)),
    ("J|gaussian|slopes on two different groups", d -> drm(bf(@formula(y_g ~ x1 + (1 | g) + (0 + x1 | g) + (0 + x2 | g2)), @formula(sigma ~ 1)), Gaussian(); data = d)),
    ("K|gaussian|NA in both + (1|g)", d -> drm(bf(@formula(y_gm ~ x1m + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d)),
    ("M|gaussian|mu=x1+(1|g)|entry-points", d -> drm(bf(@formula(y_g ~ x1 + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d)),
    # P17b: a missing response
    ("K|gaussian|NA in response", d -> drm(bf(@formula(y_gm ~ x1), @formula(sigma ~ 1)), Gaussian(); data = d)),
    ("K|gaussian|NA in response|REML", d -> drm(bf(@formula(y_gm ~ x1), @formula(sigma ~ 1)), Gaussian(); data = d, method = :REML)),
    ("K|gaussian|NA in response + weights", d -> drm(bf(@formula(y_gm ~ x1), @formula(sigma ~ 1)), Gaussian(); data = d, weights = d.w)),
    ("K|poisson|NA in response", d -> drm(bf(@formula(y_pm ~ x1)), Poisson(); data = d)),
]

# Two more cells of the P17a list now fit (a LogNormal `sigma` random effect; a mean and a `sigma` random
# effect on two different groups), but those routes keep no random-effect modes, so the conditional default
# refuses by name and points to `re = :zero`. Their `re = :zero` values are pinned to drmTMB below. When a
# route starts to keep its modes, move the cell into `_FC_CELLS`.
const _FC_FITS_WITHOUT_MODES = [
    ("J|lognormal|sigma=x1+(1|g)", d -> drm(bf(@formula(y_ln ~ x1), @formula(sigma ~ x1 + (1 | g))), LogNormal(); data = d)),
    ("J|gaussian|mu=(1|g)|sigma=(1|g2)", d -> drm(bf(@formula(y_g ~ x1 + (1 | g)), @formula(sigma ~ x1 + (1 | g2))), Gaussian(); data = d)),
]

# Julia against R. The Laplace and exact Gaussian routes agree to well below 1e-6; the ordinal crossed
# route estimates its cutpoints by Nelder-Mead and its logLik is 9e-7 from drmTMB's, which moves the
# fitted scores by up to 7e-6.
_fc_tol(id) = occursin("cumul", id) ? 2e-5 : 1e-6

@testset "fitted(): conditional default, used rows (P17a, P17b)" begin
    d = _fc_data()
    ref = _fc_reference()
    @testset "the two P17a cells that fit but keep no modes: default refuses by name, re = :zero is drmTMB's population value" begin
        for (id, fit_it) in _FC_FITS_WITHOUT_MODES
            r = ref[id]
            fit = fit_it(d)
            ferr = try; fitted(fit); nothing; catch e; e; end
            @test ferr isa ArgumentError
            @test occursin("re = :zero", sprint(showerror, ferr))
            zero = fitted(fit; re = :zero)
            @test length(zero) == nobs(fit) == length(r.pop)
            @test !any(isnan, r.pop)
            @test maximum(abs.(zero .- r.pop)) < _fc_tol(id)                   # drmTMB population value
        end
    end
    for (id, fit_it) in _FC_CELLS
        @testset "$id" begin
            fit = fit_it(d)
            r = ref[id]
            cond = fitted(fit)
            @test length(cond) == nobs(fit) == length(r.cond)
            @test !any(isnan, cond)
            @test maximum(abs.(cond .- r.cond)) < _fc_tol(id)                  # drmTMB fitted(fit)
            @test fitted(fit; re = :conditional) == cond
            zero = fitted(fit; re = :zero)
            @test length(zero) == length(cond)
            if !any(isnan, r.pop) && length(r.pop) == length(zero)
                @test maximum(abs.(zero .- r.pop)) < _fc_tol(id)               # population value, as before
            end
            if startswith(id, "K|") && !occursin("NA in both", id)
                # P17b: the 16 rows with a missing response are not returned
                @test nobs(fit) == 144 && length(cond) == 144 && length(zero) == 144
            end
            # residuals are y - fitted on the same rows and at the same random-effect level
            res = residuals(fit)
            @test length(res) == length(cond)
            @test !any(isnan, res)
            ysum = first(values(fit.obs))
            used = any(isnan, ysum) ? .!isnan.(ysum) : trues(length(ysum))
            @test res ≈ ysum[used] .- cond
            @test residuals(fit; re = :zero) ≈ ysum[used] .- zero
        end
    end
end

@testset "fitted(): semantics" begin
    d = _fc_data()
    @testset "no random effect: both levels agree, `re` is validated" begin
        fit = drm(bf(@formula(y_g ~ x1), @formula(sigma ~ x2)), Gaussian(); data = d)
        @test fitted(fit) == fitted(fit; re = :zero) == fit.means[:mu]
        @test residuals(fit) == residuals(fit; re = :zero)
        @test_throws ArgumentError fitted(fit; re = :marginal)
        @test_throws ArgumentError residuals(fit; re = :nope)
        fp = drm(bf(@formula(y_pois ~ x1)), Poisson(); data = d)
        @test fitted(fp) == fitted(fp; re = :zero) == fp.means[:mu]
    end
    @testset "ordinary (1 | g): conditional = population + the mode of the row's group" begin
        fit = drm(bf(@formula(y_g ~ x1 + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d)
        gl = unique(d.g)
        b = ranef(fit)[:g]
        @test fitted(fit; re = :zero) == fit.means[:mu]                       # the old value
        @test fitted(fit) ≈ fit.means[:mu] .+ b[indexin(d.g, gl)]
        @test residuals(fit; re = :zero) ≈ d.y_g .- fit.means[:mu]
        @test maximum(abs.(fitted(fit) .- fitted(fit; re = :zero))) > 0.1       # not the same thing
        # log link: conditional = exp(Xβ + b)
        fl = drm(bf(@formula(y_ln ~ x1 + (1 | g)), @formula(sigma ~ x1)), LogNormal(); data = d)
        @test all(fitted(fl) .> 0)
        @test fitted(fl) != fitted(fl; re = :zero)
    end
    @testset "a sigma-only random effect leaves the Gaussian mean alone" begin
        fit = drm(bf(@formula(y_g ~ x1), @formula(sigma ~ x2 + (1 | g))), Gaussian(); data = d)
        @test fitted(fit) == fitted(fit; re = :zero)
    end
    @testset "bivariate fits: unchanged, used rows" begin
        bfit = drm(bf(mu1 = @formula(y1 ~ x1), mu2 = @formula(y2 ~ x2)), Gaussian(); data = d)
        fv = fitted(bfit)
        @test fv isa AbstractDict && Set(keys(fv)) == Set([:mu1, :mu2])
        @test fv == fitted(bfit; re = :zero)
        rv = residuals(bfit)
        @test length(rv[:mu1]) == length(fv[:mu1])
    end
    @testset "a route that keeps no modes refuses the conditional default and never fakes it" begin
        # A Poisson relmat (not one of the two phylo groups that keep their modes, P20a) keeps no modes.
        rng = StableRNG(20261110)
        G, m = 15, 4
        n = G * m
        K = [0.5^abs(i - j) for i in 1:G, j in 1:G]
        dat = (; id = repeat(1:G, inner = m), x = randn(rng, n), y = rand(rng, 0:6, n))
        fit = drm(bf(@formula(y ~ x + relmat(1 | id))), Poisson(); data = dat, K = K)
        err = try fitted(fit); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("re = :zero", sprint(showerror, err))
        z = fitted(fit; re = :zero)
        @test z == fit.means[:mu]
        @test length(residuals(fit; re = :zero)) == n
        @test_throws ArgumentError residuals(fit)
    end
end

@testset "fitted(): route-level checks against independent computations" begin
    rng = StableRNG(20261110)
    # one group factor with both a random intercept and a correlated slope
    G, m = 12, 8
    g = repeat(1:G, inner = m)
    x = randn(rng, G * m)
    b0 = 0.8 .* randn(rng, G); b1 = 0.5 .* randn(rng, G)
    y = 1.0 .+ 0.5 .* x .+ b0[g] .+ b1[g] .* x .+ 0.6 .* randn(rng, G * m)
    dat = (; g = string.("g", g), x, y)
    @testset "(1 + x | g) Gaussian: b0 + x b1" begin
        fit = drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1)), Gaussian(); data = dat)
        B = ranef(fit)[:g]
        gi = indexin(dat.g, unique(dat.g))
        @test fitted(fit) ≈ fit.means[:mu] .+ B[gi, 1] .+ x .* B[gi, 2]
        @test fitted(fit; re = :zero) == fit.means[:mu]
    end
    @testset "Poisson (1 | g): exp(Xβ + b), and the modes are the conditional modes" begin
        yp = [rand(rng, DRModels.Distributions.Poisson(exp(0.3 + 0.2 * x[i] + b0[g[i]] * 0.5))) for i in eachindex(x)]
        dp = (; g = dat.g, x, y = yp)
        fit = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = dp)
        cond = fitted(fit); pop = fitted(fit; re = :zero)
        @test pop == fit.means[:mu]
        # exp(η + b̂_g) / exp(η) is constant within a group, and the group modes solve the score equation
        ratio = cond ./ pop
        for k in unique(dp.g)
            rows = dp.g .== k
            @test all(isapprox.(ratio[rows], ratio[findfirst(rows)]; rtol = 1e-10))
            sd = exp(coef(fit)[end])
            bk = log(ratio[findfirst(rows)])
            score = sum(dp.y[rows] .- cond[rows]) - bk / sd^2
            @test abs(score) < 1e-6
        end
    end
    @testset "Binomial (1 | g) with probit link: link-scale shift" begin
        yb = [rand(rng) < DRModels.Distributions.cdf(DRModels.Distributions.Normal(), 0.2 + 0.3 * x[i] + 0.6 * b0[g[i]]) ? 1.0 : 0.0 for i in eachindex(x)]
        db = (; g = dat.g, x, y = yb, n = ones(length(x)))
        fit = drm(bf(@formula(y ~ x + (1 | g))), Binomial(:probit); data = db)
        @test all(0 .< fitted(fit) .< 1)
        @test fitted(fit; re = :zero) == fit.means[:mu]
        @test fitted(fit) != fitted(fit; re = :zero)
    end
    @testset "temporal AR1: conditional adds the fitted temporal path" begin
        S, T = 6, 10
        ts = repeat(1:T, outer = S)
        sid = repeat(string.("s", 1:S), inner = T)
        xt = randn(rng, S * T)
        yt = zeros(S * T)
        for s in 1:S
            u = zeros(T); u[1] = 0.7 * randn(rng)
            for t in 2:T
                u[t] = 0.6 * u[t-1] + 0.7 * randn(rng) * sqrt(1 - 0.36)
            end
            yt[(s-1)*T+1:s*T] .= 1.0 .+ 0.4 .* xt[(s-1)*T+1:s*T] .+ u .+ 0.3 .* randn(rng, T)
        end
        dt = (; id = sid, t = Float64.(ts), x = xt, y = yt)
        fit = drm(bf(@formula(y ~ x + temporal(1 | id, t, ar1)), @formula(sigma ~ 1)), Gaussian(); data = dt)
        @test fitted(fit; re = :zero) == fit.means[:mu]
        @test fitted(fit) ≈ fit.means[:mu] .+ ranef(fit)[:id]
        # conditional residuals / sigma are the quantile residuals on this route
        @test residuals(fit) ./ exp(only(coef(fit, :sigma))) ≈ residuals(fit; type = :quantile)
    end
end

# ---------------------------------------------------------------------------------------------
# Review fixes for #1324. R values: p17_fitted_review_reference.csv, from p17_fitted_review_r.R
# (drmTMB 0.8.1, a Linux server). The MR cells put a missing response on the rows where `y_gm` is NA.
# ---------------------------------------------------------------------------------------------
function _fc_mask(d, name)
    na = ismissing.(d.y_gm)
    y = getproperty(d, name)
    return Union{Missing,Float64}[na[i] ? missing : Float64(y[i]) for i in eachindex(y)]
end

const _FC_TREE = strip(read(joinpath(_FC_DIR, "tree16.newick"), String))
_fc_tree() = DRModels.augmented_phy(_FC_TREE)

# (id, formula + family + keywords as a function of the data). Same models as the R script.
const _FC_WIDE = [
    ("W|beta|(1|g)", d -> drm(bf(@formula(y_beta ~ x1 + (1 | g)), @formula(sigma ~ 1)), DRModels.Beta(); data = d)),
    ("W|gamma|(1|g)", d -> drm(bf(@formula(y_gam ~ x1 + (1 | g)), @formula(sigma ~ 1)), DRModels.Gamma(); data = d)),
    ("W|nb2|(1|g)", d -> drm(bf(@formula(y_nb ~ x1 + (1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = d)),
    ("W|student|(1|g)", d -> drm(bf(@formula(y_st ~ x1 + (1 | g)), @formula(sigma ~ 1), @formula(nu ~ 1)), Student(); data = d)),
    ("W|tweedie|(1|g)", d -> drm(bf(@formula(y_tw ~ x1 + (1 | g)), @formula(sigma ~ 1), @formula(nu ~ 1)), Tweedie(); data = d)),
    ("W|betabinomial|(1|g)", d -> drm(bf(@formula(cbind(s, fl) ~ x1 + (1 | g)), @formula(sigma ~ 1)), BetaBinomial(); data = d)),
    ("W|binomial|(1|g)", d -> drm(bf(@formula(y_bin ~ x1 + (1 | g))), Binomial(); data = d)),
    ("W|cumul|(1|g)", d -> drm(bf(@formula(y_ord ~ x1 + (1 | g))), CumulativeLogit(); data = d)),
    ("W|poisson|offset+(1|g)", d -> drm(bf(@formula(y_pois ~ x1 + offset(off) + (1 | g))), Poisson(); data = d)),
    ("W|nb2|offset+(1|g)", d -> drm(bf(@formula(y_nb ~ x1 + offset(off) + (1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = d)),
]

_fc_mr(d) = merge(d, (y_g_m = _fc_mask(d, :y_g), y_ln_m = _fc_mask(d, :y_ln),
                      y_pois_m = _fc_mask(d, :y_pois), y_gam_m = _fc_mask(d, :y_gam)))
const _FC_MR = [
    ("MR|gaussian|(1|g)", (d, kw...) -> drm(bf(@formula(y_g_m ~ x1 + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, kw...)),
    ("MR|lognormal|(1|g)", (d, kw...) -> drm(bf(@formula(y_ln_m ~ x1 + (1 | g)), @formula(sigma ~ 1)), LogNormal(); data = d, kw...)),
    ("MR|poisson|(1|g)", (d, kw...) -> drm(bf(@formula(y_pois_m ~ x1 + (1 | g))), Poisson(); data = d, kw...)),
    ("MR|gamma|(1|g)", (d, kw...) -> drm(bf(@formula(y_gam_m ~ x1 + (1 | g)), @formula(sigma ~ 1)), DRModels.Gamma(); data = d, kw...)),
    ("MR|poisson|(1|g)+(1|g2)+(1|h)", (d, kw...) -> drm(bf(@formula(y_pois_m ~ x1 + (1 | g) + (1 | g2) + (1 | h))), Poisson(); data = d, kw...)),
    ("MR|gaussian|phylo(1|g)", (d, kw...) -> drm(bf(@formula(y_g_m ~ x1 + phylo(1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, tree = _fc_tree(), kw...)),
    ("MR|lognormal|phylo(1|g)", (d, kw...) -> drm(bf(@formula(y_ln_m ~ x1 + phylo(1 | g)), @formula(sigma ~ 1)), LogNormal(); data = d, tree = _fc_tree(), kw...)),
]

# Where the two engines fit the SAME model to the same precision, the conditional values agree at the
# bound used everywhere above (1e-6). The AGHQ routes here integrate with adaptive quadrature and
# drmTMB uses the Laplace approximation, so the two FITS differ and so do the modes. Measured
# (max |Julia - drmTMB| on `fitted(fit)`, and on the population value `fitted(fit; re = :zero)`):
#
#   cell              conditional    population (fit gap alone)
#   Student           1.1e-9         2.4e-10       <- pinned at 1e-6 below (its group SD is at the boundary)
#   BetaBinomial      1.5e-4         4.1e-5
#   Beta              3.2e-4         3.0e-5
#   Gamma             1.2e-3         3.8e-5
#   NB2               3.0e-3         6.8e-4
#   NB2 + offset      2.8e-3         8.0e-4
#   Tweedie           9.8e-4         1.3e-4
#   Binomial (0/1)   5.2e-3         3.7e-4
#   ordinal (1 | g)   5.1e-3         4.3e-4
#   Poisson + offset  1.4e-4         1.5e-4
#
# The population gap shows the fits themselves differ; no bound is loosened for them. For those cells
# the test checks what must hold on the Julia side alone: the conditional value is the population
# value moved on the link scale by one number per group (the fitted mode), and the shift is finite.
@testset "fitted(): more families, an offset and an ordinal fit against drmTMB" begin
    d = _fc_data()
    # the Binomial cell has its own R script and file (p17_fitted_binomial_r.R)
    ref = merge(_fc_reference("p17_fitted_review_reference.csv"),
                _fc_reference("p17_fitted_binomial_reference.csv"))
    gi = indexin(d.g, unique(d.g))
    for (id, fit_it) in _FC_WIDE
        @testset "$id" begin
            fit = fit_it(d)
            cond = fitted(fit); pop = fitted(fit; re = :zero)
            @test length(cond) == nobs(fit) == length(ref[id].cond)
            @test all(isfinite, cond)
            if id == "W|student|(1|g)"
                @test maximum(abs.(cond .- ref[id].cond)) < _fc_tol(id)
                @test maximum(abs.(pop .- ref[id].pop)) < _fc_tol(id)
            else
                # the recorded shift is one value per group, and it is what `fitted` adds on the link scale
                eta = fit.scales[:cond_fit_eta_re]
                for k in unique(gi)
                    rows = gi .== k
                    @test all(isapprox.(eta[rows], eta[findfirst(rows)]; atol = 1e-12))
                end
                @test maximum(abs.(cond .- pop)) > 1e-3
                @test sign.(cond .- pop) == sign.(eta) || id == "W|cumul|(1|g)" || all(==(0), eta)
            end
        end
    end
end

@testset "fitted(): a missing response on a random-effect fit (review of #1324)" begin
    d = _fc_mr(_fc_data())
    ref = _fc_reference("p17_fitted_review_reference.csv")
    keep = .!ismissing.(d.y_g_m)
    dc = map(v -> v[keep], d)                     # the same used rows, complete cases
    dc = merge(dc, (y_g_m = Float64.(dc.y_g_m), y_ln_m = Float64.(dc.y_ln_m),
                    y_pois_m = Float64.(dc.y_pois_m), y_gam_m = Float64.(dc.y_gam_m)))
    for (id, fit_it) in _FC_MR
        @testset "$id" begin
            fit = fit_it(d)
            fc = fit_it(dc)
            cond = fitted(fit); zero = fitted(fit; re = :zero)
            @test length(cond) == length(zero) == nobs(fit) == 144
            # drmTMB. The two AGHQ cells below differ from drmTMB's Laplace fit by 2.1e-3 (Poisson) and
            # 1.3e-3 (Gamma) for the reason given above the wide cells; they are checked against the fit
            # on the complete cases instead (identical rows, identical model), not against R.
            id in ("MR|poisson|(1|g)", "MR|gamma|(1|g)") ||
                @test maximum(abs.(cond .- ref[id].cond)) < _fc_tol(id)
            # the fit with the missing rows and the fit on the complete cases are one fit
            @test maximum(abs.(cond .- fitted(fc))) < 1e-8
            @test maximum(abs.(zero .- fitted(fc; re = :zero))) < 1e-8
            # the modes really are in (the Gaussian phylogenetic SD is at its boundary on this data: no gap)
            id == "MR|gaussian|phylo(1|g)" || @test maximum(abs.(cond .- zero)) > 1e-3
            @test residuals(fit) ≈ residuals(fc) atol = 1e-8
        end
    end
end

@testset "fitted(): internal keys do not leak into user output (review of #1324)" begin
    d = _fc_data()
    for fit in (drm(bf(@formula(y_st ~ x1 + (1 | g)), @formula(sigma ~ 1), @formula(nu ~ 1)), Student(); data = d),
                drm(bf(@formula(y_tw ~ x1 + (1 | g)), @formula(sigma ~ 1), @formula(nu ~ 1)), Tweedie(); data = d),
                drm(bf(@formula(y_gam ~ x1 + (1 | g)), @formula(sigma ~ 1)), DRModels.Gamma(); data = d))
        @test haskey(fit.scales, :cond_fit_eta_re)                  # the record exists ...
        s = sigma(fit)
        ks = s isa AbstractDict ? Set(keys(s)) : Set([:sigma])
        @test !any(k -> startswith(String(k), "cond_"), ks)         # ... and is not in sigma(fit)
        # the table is what the fit's own scales were before the record existed
        old = Dict(k => v for (k, v) in fit.scales if !startswith(String(k), "cond_"))
        s isa AbstractDict ? (@test s == old) : (@test s == old[:sigma])
        @test !any(k -> startswith(String(k), "cond_"), keys(DRModels._bridge_dpars(fit)))
    end
end

@testset "fitted(): a group mode that was reset is detected, not read as a fit (review of #1324)" begin
    sc = Dict{Symbol,Vector{Float64}}()
    bcache = zeros(1, 3)
    # a clean objective evaluation records the modes
    DRModels._record_cond_fit_aghq!(sc, θ -> 1.0, nothing, bcache, [1, 2, 3], ones(3, 1))
    @test haskey(sc, :cond_fit_eta_re) && !haskey(sc, :cond_fit_failed)
    # an objective that had to reset one group's mode to 0 marks the record failed
    sc = Dict{Symbol,Vector{Float64}}()
    DRModels._record_cond_fit_aghq!(sc, θ -> (Threads.atomic_add!(DRModels._AGHQ_MODE_RESETS, 1); 1.0),
                                    nothing, bcache, [1, 2, 3], ones(3, 1))
    @test haskey(sc, :cond_fit_failed) && !haskey(sc, :cond_fit_eta_re)
    # an infinite objective (a failed mode solve) does too
    sc = Dict{Symbol,Vector{Float64}}()
    DRModels._record_cond_fit_aghq!(sc, θ -> Inf, nothing, bcache, [1, 2, 3], ones(3, 1))
    @test haskey(sc, :cond_fit_failed)
    # the objective really is infinite when a group's mode solve fails (what the record step reads)
    members = [[1, 2], [3, 4]]
    bc = zeros(1, 2)
    v = DRModels._aghq_marginal_loglik((i, η) -> -Inf, members, zeros(4), ones(4, 1), fill(1.0, 1, 1),
                                       DRModels._AGHQRule(1, 5), bc)
    @test !isfinite(v)
    # a fit whose record is failed refuses the conditional default and still gives re = :zero
    d = _fc_data()
    fit = drm(bf(@formula(y_pois ~ x1 + (1 | g))), Poisson(); data = d)
    fit.scales[:cond_fit_failed] = [1.0]
    err = try fitted(fit); nothing catch e; e end
    @test err isa ArgumentError && occursin("mode solve failed", sprint(showerror, err))
    @test fitted(fit; re = :zero) == fit.means[:mu]
end

@testset "residuals(type = :quantile) takes no `re` (review of #1324)" begin
    d = _fc_data()
    fit = drm(bf(@formula(y_g ~ x1 + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d)
    q = residuals(fit; type = :quantile, rng = StableRNG(1))
    @test_throws ArgumentError residuals(fit; type = :quantile, re = :zero)
    @test_throws ArgumentError residuals(fit; type = :quantile, re = :nope)
    @test residuals(fit; type = :quantile, re = :conditional, rng = StableRNG(1)) == q   # unchanged values
end

@testset "fitted(): routes with no recorded modes refuse the conditional default" begin
    d = _fc_data()
    nr(f) = (err = try fitted(f); nothing catch e; e end; err isa ArgumentError && occursin("re = :zero", sprint(showerror, err)))
    # 1. and 2. (Gaussian relmat / animal / dense phylo, and phylo Laplace for the six count and continuous
    # families) keep their modes since P20a: see test_fitted_phylo_modes.jl.
    # 4. ZeroOneBeta with a random intercept
    zf = drm(bf(@formula(y_zoib ~ x1 + (1 | g)), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)),
             ZeroOneBeta(); data = d)
    @test nr(zf)
    # 8. crossed random intercepts solved by adaptive quadrature (opt-in)
    cf = drm(bf(@formula(y_pois ~ x1 + (1 | g2) + (1 | f2))), Poisson(); data = d, se = false, marginal = :AGHQ)
    @test nr(cf)
end
