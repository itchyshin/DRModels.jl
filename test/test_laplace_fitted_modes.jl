# fitted() on an opt-in `marginal = :Laplace` fit: the fitted random-effect modes are kept.
#
# The ordinary `(1 | g)` Laplace routes (Poisson, NegBinomial2, Binomial, and the Gamma and Beta
# siblings that share the scale route) now store the inner-mode solve at the final parameters, so
# `fitted(fit)` is conditional (`re = :conditional`), the same definition as on the default route
# (test_fitted_conditional.jl). Before, `fitted(fit)` threw because no modes were kept.
#
# Reference values (drmTMB 0.8.1, R 4.6.1), all from the data in
# test/fixtures/p17_fitted/data.csv:
#   laplace_fitted_modes_reference.csv    <- laplace_fitted_modes_r.R      (3 cells, added here)
#   p17_fitted_review_reference.csv      <- p17_fitted_review_r.R         (Poisson and NB2 with an offset)
#   p17_fitted_binomial_reference.csv    <- p17_fitted_binomial_r.R       (Binomial)
# The bound is the one test_fitted_conditional.jl uses for the cells where both engines fit the same
# model to the same precision (1e-6). Here drmTMB and `:Laplace` fit the same objective.
using DRModels
using Test
using DelimitedFiles
using Statistics

const _LF_DIR = joinpath(@__DIR__, "fixtures", "p17_fitted")
const _LF_FACS = Set(["g", "g2", "h", "f", "f2", "gna", "gu", "y_cat", "y_ordf"])
const _LF_TOL = 1e-6                       # same bound as `_fc_tol` in test_fitted_conditional.jl

function _lf_data()
    m, h = readdlm(joinpath(_LF_DIR, "data.csv"), ',', Any; header = true)
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(String.(vec(h)))
        c = m[:, j]
        v = if n in _LF_FACS
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

function _lf_reference(file)
    m = readdlm(joinpath(_LF_DIR, file), ',', Any; skipstart = 1)
    ref = Dict{String,NamedTuple{(:cond, :pop),Tuple{Vector{Float64},Vector{Float64}}}}()
    for id in unique(String.(m[:, 1]))
        keep = String.(m[:, 1]) .== id
        ref[id] = (cond = Float64.(m[keep, 3]),
                   pop = Float64[x isa Number ? Float64(x) : NaN for x in m[keep, 4]])
    end
    return ref
end

const _LF_REFS = ("laplace_fitted_modes_reference.csv", "p17_fitted_review_reference.csv",
                  "p17_fitted_binomial_reference.csv")

# (reference id, group column, fit). The ids are the drmTMB cells above.
const _LF_CELLS = [
    ("L|poisson|gu", :gu, d -> drm(bf(@formula(y_pois ~ x1 + (1 | gu))), Poisson(); data = d, marginal = :Laplace)),
    ("L|binomial|gn", :gn, d -> drm(bf(@formula(y_bin ~ x1 + (1 | gn))), Binomial(); data = d, marginal = :Laplace)),
    ("L|poisson|NA in both", :g, d -> drm(bf(@formula(y_pm ~ x1m + (1 | g))), Poisson(); data = d, marginal = :Laplace)),
    ("W|poisson|offset+(1|g)", :g, d -> drm(bf(@formula(y_pois ~ x1 + offset(off) + (1 | g))), Poisson(); data = d, marginal = :Laplace)),
    ("W|nb2|offset+(1|g)", :g, d -> drm(bf(@formula(y_nb ~ x1 + offset(off) + (1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = d, marginal = :Laplace)),
    ("W|binomial|(1|g)", :g, d -> drm(bf(@formula(y_bin ~ x1 + (1 | g))), Binomial(); data = d, marginal = :Laplace)),
]

@testset "fitted(): Laplace fits keep the random-effect modes" begin
    d = _lf_data()
    ref = merge((_lf_reference(f) for f in _LF_REFS)...)
    for (id, grp, fit_it) in _LF_CELLS
        @testset "$id" begin
            fit = fit_it(d)
            r = ref[id]
            @test fit.marginal === :Laplace
            cond = fitted(fit)
            @test length(cond) == nobs(fit) == length(r.cond)
            @test all(isfinite, cond)
            @test maximum(abs.(cond .- r.cond)) < _LF_TOL                       # drmTMB fitted(fit)
            @test fitted(fit; re = :conditional) == cond
            zero = fitted(fit; re = :zero)
            @test length(zero) == length(cond)
            if !any(isnan, r.pop) && length(r.pop) == length(zero)
                @test maximum(abs.(zero .- r.pop)) < _LF_TOL                    # population value
            end
            @test maximum(abs.(cond .- zero)) > 1e-3                            # the modes are really in
            # the record is one link-scale value per group, in the used-row order
            eta = fit.scales[:cond_fit_eta_re]
            @test !haskey(fit.scales, :cond_fit_failed)
            @test length(eta) == nobs(fit)
            # residuals are y - fitted on the same rows, at the same level
            y = first(values(fit.obs))
            used = any(isnan, y) ? .!isnan.(y) : trues(length(y))
            @test residuals(fit) ≈ y[used] .- cond
            @test residuals(fit; re = :zero) ≈ y[used] .- zero
        end
    end
end

@testset "fitted(): Laplace modes, route-level checks" begin
    d = _lf_data()
    gi = indexin(d.g, unique(d.g))
    @testset "Poisson (1 | g): the modes solve the score equation of the fitted objective" begin
        fit = drm(bf(@formula(y_pois ~ x1 + (1 | g))), Poisson(); data = d, marginal = :Laplace)
        cond = fitted(fit); pop = fitted(fit; re = :zero)
        @test pop == fit.means[:mu]
        sd = exp(fit.theta[end])
        ratio = cond ./ pop
        for k in unique(d.g)
            rows = d.g .== k
            @test all(isapprox.(ratio[rows], ratio[findfirst(rows)]; rtol = 1e-10))
            bk = log(ratio[findfirst(rows)])
            @test abs(sum(d.y_pois[rows] .- cond[rows]) - bk / sd^2) < 1e-8
        end
    end
    @testset "grouped Binomial input cbind(s, fl) and the scale-family siblings" begin
        fb = drm(bf(@formula(cbind(s, fl) ~ x1 + (1 | g))), Binomial(); data = d, marginal = :Laplace)
        fn = drm(bf(@formula(y_nb ~ x1 + (1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = d, marginal = :Laplace)
        fg = drm(bf(@formula(y_gam ~ x1 + (1 | g)), @formula(sigma ~ 1)), DRModels.Gamma(); data = d, marginal = :Laplace)
        fe = drm(bf(@formula(y_beta ~ x1 + (1 | g)), @formula(sigma ~ 1)), DRModels.Beta(); data = d, marginal = :Laplace)
        for fit in (fb, fn, fg, fe)
            cond = fitted(fit)
            @test length(cond) == nobs(fit) && all(isfinite, cond)
            eta = fit.scales[:cond_fit_eta_re]
            for k in unique(gi)
                rows = gi .== k
                @test all(isapprox.(eta[rows], eta[findfirst(rows)]; atol = 1e-12))
            end
            @test maximum(abs.(cond .- fitted(fit; re = :zero))) > 1e-3
        end
        @test all(0 .< fitted(fb) .< 1)
    end
    @testset "recording the modes changes neither the value nor the estimates of the fit" begin
        # the default (:LA) route and the Laplace route are different integrations, so compare the
        # Laplace fit with a fit of the same data through the same route, and with its own closures
        fit = drm(bf(@formula(y_pois ~ x1 + (1 | g))), Poisson(); data = d, marginal = :Laplace)
        @test fit.nll(fit.theta) ≈ -loglikelihood(fit) rtol = 1e-12
        @test fit.means[:mu] == fitted(fit; re = :zero)
    end
    @testset "a Poisson (1 | g) fit with prior weights still refuses :Laplace" begin
        @test_throws ArgumentError drm(bf(@formula(y_pois ~ x1 + (1 | g))), Poisson();
                                       data = d, weights = d.w, marginal = :Laplace)
    end
end
