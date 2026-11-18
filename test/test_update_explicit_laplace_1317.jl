# test_update_explicit_laplace_1317.jl -- `update()` and an explicit `marginal = :Laplace`
# on fits that main already accepts (independent intercept + slope, Gamma / NegBinomial2 /
# Binomial / Poisson) and on the routes this branch adds (Binomial with three crossed
# intercepts, zi / hurdle / truncated / skew-normal with one `(1 | g)`, ordinal independent
# terms).
#
# Why: a fit tagged `marginal = :Laplace` is refitted by `update()` with that tag, so the
# `:Laplace` keyword has to reach the route that produced the fit. The first version of this
# branch sat on a base without that redirect, so `update()` and the explicit keyword were
# refused on fits that main accepts (independent review, 12 Nov 2026). Every refit and every
# explicit call below must equal the default fit exactly (`==`, not `≈`).
using DRModels, Test, Random

const _UE_DIR = joinpath(@__DIR__, "fixtures", "zi_re_parity")

function _ue_data(route)
    L = readlines(joinpath(_UE_DIR, route, "data.csv"))
    hdr = [strip(c, '"') for c in split(L[1], ',')]
    cols = Dict{Symbol,Any}()
    for (j, h) in enumerate(hdr)
        raw = [strip(split(l, ',')[j], '"') for l in L[2:end]]
        cols[Symbol(h)] = all(s -> tryparse(Float64, s) !== nothing, raw) ? parse.(Float64, raw) : String.(raw)
    end
    return NamedTuple{Tuple(keys(cols))}(Tuple(values(cols)))
end

same_fit(a, b) = loglik(a) == loglik(b) && coef(a) == coef(b) && a.marginal === b.marginal

# One data set with every response type the slope shapes need.
r = Random.Xoshiro(20261112); n = 180; ng = 12
g = string.("g", rand(r, 1:ng, n)); gi = parse.(Int, SubString.(g, 2))
x = randn(r, n); z = randn(r, n)
eta = 0.3 .+ 0.4 .* x .+ (0.5 .* randn(r, ng))[gi] .+ (0.3 .* randn(r, ng))[gi] .* z
ypos = [exp(e) * (0.6 + 0.8 * rand(r)) for e in eta]
yc = Float64[sum(rand(r) < 0.35 for _ in 1:Int(ceil(2 * exp(e)))) for e in eta]
sb = Float64[sum(rand(r) < 1 / (1 + exp(-e)) for _ in 1:10) for e in eta]
dslope = (; ypos, yc, x, z, g, s = sb, f = 10.0 .- sb)

@testset "slope fits that main accepts: update and explicit :Laplace give the default fit" begin
    cases = (("Gamma", bf(@formula(ypos ~ x + (1 | g) + (0 + z | g))), DRModels.Gamma()),
             ("NegBinomial2", bf(@formula(yc ~ x + (1 | g) + (0 + z | g))), NegBinomial2()),
             ("Binomial", bf(@formula(cbind(s, f) ~ x + (1 | g) + (0 + z | g))), Binomial()),
             ("Poisson", bf(@formula(yc ~ x + (1 | g) + (0 + z | g))), Poisson()))
    for (nm, form, fam) in cases
        @testset "$nm" begin
            fit = drm(form, fam; data = dslope)
            @test fit.marginal === :Laplace
            @test same_fit(update(fit, form; data = dslope), fit)
            @test same_fit(update(fit, form; data = dslope, marginal = :Laplace), fit)
            @test same_fit(drm(form, fam; data = dslope, marginal = :Laplace), fit)
            @test same_fit(drm(form, fam; data = dslope, marginal = :LA), fit)
        end
    end
end

@testset "new routes of this branch: update and explicit :Laplace reach the same fit" begin
    # Binomial, three crossed intercepts (dense Laplace spine, tag :Laplace)
    d = _ue_data("binom3")
    form = bf(@formula(y ~ x + (1 | g) + (1 | g2) + (1 | h)))
    fit = drm(form, Binomial(); data = d)
    @test fit.marginal === :Laplace
    @test same_fit(update(fit, form; data = d), fit)
    @test same_fit(update(fit, form; data = d, marginal = :Laplace), fit)
    @test same_fit(drm(form, Binomial(); data = d, marginal = :Laplace), fit)
    # AGHQ on three components stays refused by name
    @test_throws ArgumentError drm(form, Binomial(); data = d, marginal = :AGHQ)
    @test_throws ArgumentError drm(form, Binomial(); data = d, marginal = :VA)

    # zi / hurdle / truncated / skew-normal with one (1 | g)
    cases = (("zip_g", bf(@formula(y ~ x + (1 | g)), @formula(zi ~ 1)), Poisson()),
             ("zinb_g", bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(zi ~ 1)), NegBinomial2()),
             ("tnb_g", bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), TruncatedNegBinomial2()),
             ("skew_g", bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(nu ~ 1)), SkewNormal()),
             ("cumul_gx", bf(@formula(y ~ x + (1 | g) + (0 + x | g))), CumulativeLogit()))
    for (route, fm, fam) in cases
        @testset "$route" begin
            d = _ue_data(route)
            fit = drm(fm, fam; data = d)
            @test fit.marginal === :Laplace
            @test same_fit(update(fit, fm; data = d), fit)
            @test same_fit(update(fit, fm; data = d, marginal = :Laplace), fit)
            @test same_fit(drm(fm, fam; data = d, marginal = :Laplace), fit)
        end
    end
end

@testset "unchanged: exactly two crossed intercepts keep their tag and their refusal" begin
    d = _ue_data("binom3")
    form = bf(@formula(y ~ x + (1 | g) + (1 | h)))
    fit = drm(form, Binomial(); data = d)
    @test fit.marginal === :LA
    @test same_fit(update(fit, form; data = d), fit)
    err = try drm(form, Binomial(); data = d, marginal = :Laplace); nothing catch e; e end
    @test err isa ArgumentError && occursin("crossed/multiple random effects", sprint(showerror, err))
end
