# Animal / relmat intercept-plus-slope routes added for the wave-10 gap map (ranks 5 and 9):
#   Beta bare `animal|relmat(1 + x | g)`, Gaussian REML bare `animal|relmat(1 + x | g)`,
#   Binomial `animal(1 + x | g)` (independent fields) and `animal(1 + x | p | g)` (correlated)
#   with sd(g, animal, coef(.)) ~ 1 or ~ z.
# Reference: test/fixtures/animal_slope/reference.tsv (drmTMB 0.8.1; made by reference.R on the
# committed datasets). Thresholds fixed before the coefficient and SE comparison
# (docs/dev-log/2026-11-09-animal-slope-scope.md):
#   |loglik - R| <= 1e-3;  |coef - R| <= 5e-3;  SE ratio in [0.95, 1.05].
using DRModels, Test, LinearAlgebra, Random
using StableRNGs: StableRNG

const _ASL_DIR = joinpath(@__DIR__, "fixtures", "animal_slope")

function _asl_data(name)
    lines = readlines(joinpath(_ASL_DIR, "data_$name.csv"))
    hdr = split(lines[1], ',')
    rows = [split(l, ',') for l in lines[2:end]]
    num(c) = parse.(Float64, [r[findfirst(==(c), hdr)] for r in rows])
    return (; y = num("y"), x1 = num("x1"), x2 = num("x2"), x3 = num("x3"), z = num("z"),
              g = String[r[findfirst(==("g"), hdr)] for r in rows])
end

function _asl_A()
    ids = String.(strip.(readlines(joinpath(_ASL_DIR, "A_ids.csv"))[2:end]))
    A = reduce(vcat, [permutedims(parse.(Float64, split(l, ','))) for l in readlines(joinpath(_ASL_DIR, "A.csv"))])
    return A, ids
end

function _asl_reference()
    ref = Dict{String,Any}()
    for l in readlines(joinpath(_ASL_DIR, "reference.tsv"))[2:end]
        f = split(l, '\t')
        r = get!(ref, String(f[1]), (; est = Float64[], se = Float64[], ll = parse(Float64, f[6]), conv = f[7] == "TRUE"))
        push!(r.est, parse(Float64, f[4])); push!(r.se, parse(Float64, f[5]))
    end
    return ref
end

# Fitted coordinates in drmTMB's order: mu, then sigma, then the SD blocks, then the correlation.
# `reml = true`: drmTMB's REML fit reports only the variance components, so drop the mu block.
function _asl_in_r_order(fit; reml::Bool = false)
    b = fit.blocks
    pick(pred) = collect(Iterators.flatten([last(p) for p in b if pred(first(p))]))
    idx = vcat(reml ? Int[] : pick(==(:mu)), pick(==(:sigma)),
               pick(n -> n !== :mu && n !== :sigma && n !== :recov), pick(==(:recov)))
    return fit.theta[idx], stderror(fit)[idx]
end

function _asl_check(fit, ref; reml::Bool = false)
    @test fit.converged
    @test all(isfinite, stderror(fit))
    th, se = _asl_in_r_order(fit; reml)
    @test length(th) == length(ref.est)
    @test abs(loglikelihood(fit) - ref.ll) <= 1e-3
    @test maximum(abs, th .- ref.est) <= 5e-3
    @test all(0.95 .<= se ./ ref.se .<= 1.05)
end

@testset "animal / relmat intercept-plus-slope routes (wave-10 ranks 5 and 9)" begin
    A, ids = _asl_A()
    REF = _asl_reference()
    @test all(r -> r.conv, values(REF))

    @testset "Beta bare animal and relmat (ML)" begin
        d = _asl_data("beta_animal")
        fa = drm(bf(@formula(y ~ x1 + x2 + x3 + animal(1 + x1 | g)), @formula(sigma ~ 1)), Beta();
                 data = d, A = A, ids = ids)
        _asl_check(fa, REF["beta_animal_bare"])
        # The bare form IS the explicit constant-SD form.
        fe = drm(bf(@formula(y ~ x1 + x2 + x3 + animal(1 + x1 | g)), @formula(sigma ~ 1),
                    @formula(sd(g, animal, coef(intercept)) ~ 1), @formula(sd(g, animal, coef(x1)) ~ 1)),
                 Beta(); data = d, A = A, ids = ids)
        @test loglikelihood(fa) ≈ loglikelihood(fe) atol = 1e-8
        dr = _asl_data("beta_relmat")
        fr = drm(bf(@formula(y ~ x1 + x2 + x3 + relmat(1 + x1 | g)), @formula(sigma ~ 1)), Beta();
                 data = dr, K = A)
        _asl_check(fr, REF["beta_relmat_bare"])
        # R refuses the bare labelled Beta form; so does Julia.
        @test_throws r"is not implemented on the univariate routes; only the intercept form is generally available" drm(bf(@formula(y ~ x1 + x2 + x3 + animal(1 + x1 | p | g)), @formula(sigma ~ 1)),
                                       Beta(); data = d, A = A, ids = ids)
    end

    @testset "Gaussian bare animal and relmat (REML)" begin
        d = _asl_data("gaussian_animal")
        fa = drm(bf(@formula(y ~ x1 + x2 + x3 + animal(1 + x1 | g))), Gaussian(); data = d, A = A, ids = ids, method = :REML)
        _asl_check(fa, REF["gauss_animal_reml"]; reml = true)
        dr = _asl_data("gaussian_relmat")
        fr = drm(bf(@formula(y ~ x1 + x2 + x3 + relmat(1 + x1 | g))), Gaussian(); data = dr, K = A, method = :REML)
        _asl_check(fr, REF["gauss_relmat_reml"]; reml = true)
        # ML bare form unchanged: still the ML route, different objective from REML.
        fm = drm(bf(@formula(y ~ x1 + x2 + x3 + animal(1 + x1 | g))), Gaussian(); data = d, A = A, ids = ids)
        @test abs(loglikelihood(fm) - loglikelihood(fa)) > 1
    end

    @testset "Binomial animal: independent vs correlated" begin
        d = _asl_data("binomial_animal")
        sdz = (@formula(sd(g, animal, coef(intercept)) ~ z), @formula(sd(g, animal, coef(x1)) ~ z))
        sd1 = (@formula(sd(g, animal, coef(intercept)) ~ 1), @formula(sd(g, animal, coef(x1)) ~ 1))
        for (nm, sdf) in (("one", sd1), ("z", sdz))
            fi = drm(bf(@formula(y ~ x1 + x2 + x3 + animal(1 + x1 | g)), sdf...), Binomial(:logit); data = d, A = A, ids = ids)
            fl = drm(bf(@formula(y ~ x1 + x2 + x3 + animal(1 + x1 | p | g)), sdf...), Binomial(:logit); data = d, A = A, ids = ids)
            @testset "sd ~ $nm" begin
                _asl_check(fi, REF["binom_animal_indep_$nm"])
                _asl_check(fl, REF["binom_animal_lab_$nm"])
                @test !any(p -> first(p) === :recov, fi.blocks)
                @test any(p -> first(p) === :recov, fl.blocks)
                @test loglikelihood(fl) > loglikelihood(fi)
            end
        end
        fl = drm(bf(@formula(y ~ x1 + x2 + x3 + animal(1 + x1 | p | g)), sd1...), Binomial(:logit); data = d, A = A, ids = ids)
        fi = drm(bf(@formula(y ~ x1 + x2 + x3 + animal(1 + x1 | g)), sd1...), Binomial(:logit); data = d, A = A, ids = ids)
        # Post-fit paths accept both forms.
        for fit in (fi, fl)
            @test length(predict(fit, d)) == length(d.y)
            ys = simulate(fit; data = d, A = A, rng = StableRNG(1))
            @test length(ys) == length(d.y) && all(y -> y in (0, 1), ys)
        end
        # R refuses the bare Binomial animal forms (no SD formula); so does Julia.
        @test_throws r"is not implemented on the univariate routes; only the intercept form is generally available" drm(bf(@formula(y ~ x1 + x2 + x3 + animal(1 + x1 | g))), Binomial(:logit); data = d, A = A, ids = ids)
        @test_throws r"is not implemented on the univariate routes; only the intercept form is generally available" drm(bf(@formula(y ~ x1 + x2 + x3 + animal(1 + x1 | p | g))), Binomial(:logit); data = d, A = A, ids = ids)
        # Binomial relmat q2 has no route (R fits it but without convergence): refused, not mis-routed.
        @test_throws r"relmat\(1 \+ x1 \| g\)` is not implemented on the univariate routes" drm(bf(@formula(y ~ x1 + x2 + x3 + relmat(1 + x1 | g))), Binomial(:logit); data = d, K = A)
    end
end
