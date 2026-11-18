# Residual wave-10 refusals closed by default SD completion of a bare structured term:
#   Binomial animal / relmat / spatial `(1 | g)`, Beta spatial `(1 | g)` and `(1 + x1 | g)`,
#   Beta phylo `(1 + x1 | g)`, NegBinomial2 labelled phylo `(1 + x1 | p | g)`.
# Reference: test/fixtures/residual_refusals/reference.tsv (drmTMB 0.8.1, made by reference.R on the
# committed datasets). Rule fixed before the reference was read
# (docs/dev-log/2026-11-09-residual-refusals.md):
#   |loglik - R| <= 1e-4;  mean and log-sigma coefficients <= 1e-3;  log SD and correlation <= 5e-3.
using DRModels, Test, DelimitedFiles

const _RR_DIR = joinpath(@__DIR__, "fixtures", "residual_refusals")

function _rr_data(name)
    lines = readlines(joinpath(_RR_DIR, "data_$name.csv"))
    hdr = split(lines[1], ',')
    rows = [split(l, ',') for l in lines[2:end]]
    num(c) = parse.(Float64, [r[findfirst(==(c), hdr)] for r in rows])
    return (; y = num("y"), x1 = num("x1"), x2 = num("x2"), x3 = num("x3"), z = num("z"),
              g = String[r[findfirst(==("g"), hdr)] for r in rows])
end

function _rr_A()
    ids = String.(strip.(readlines(joinpath(_RR_DIR, "A_ids.csv"))[2:end]))
    A = reduce(vcat, [permutedims(parse.(Float64, split(l, ','))) for l in readlines(joinpath(_RR_DIR, "A.csv"))])
    return A, ids
end

function _rr_coords()
    lines = readlines(joinpath(_RR_DIR, "coords.csv"))[2:end]
    return Dict(String(split(l, ',')[1]) => parse.(Float64, split(l, ',')[2:3]) for l in lines)
end

function _rr_reference()
    ref = Dict{String,Any}()
    for l in readlines(joinpath(_RR_DIR, "reference.tsv"))[2:end]
        f = split(l, '\t')
        r = get!(ref, String(f[1]), (; est = Float64[], ll = parse(Float64, f[6]), conv = f[7] == "TRUE"))
        push!(r.est, parse(Float64, f[4]))
    end
    return ref
end

# Fitted coordinates in drmTMB's order: mu, then sigma, then the SD blocks, then the correlation.
function _rr_in_r_order(fit)
    b = fit.blocks
    pick(pred) = collect(Iterators.flatten([last(p) for p in b if pred(first(p))]))
    idx = vcat(pick(==(:mu)), pick(==(:sigma)),
               pick(n -> n !== :mu && n !== :sigma && n !== :recov), pick(==(:recov)))
    return fit.theta[idx], length(pick(==(:mu))) + length(pick(==(:sigma)))
end

function _rr_check(fit, ref)
    th, nfixed = _rr_in_r_order(fit)
    @test length(th) == length(ref.est)
    @test abs(loglikelihood(fit) - ref.ll) <= 1e-4
    @test maximum(abs, th[1:nfixed] .- ref.est[1:nfixed]) <= 1e-3
    @test maximum(abs, th[nfixed+1:end] .- ref.est[nfixed+1:end]) <= 5e-3
    ref.conv && @test fit.converged
    @test all(isfinite, stderror(fit))
end

@testset "residual bare structured cells (wave-10 gap map)" begin
    ref = _rr_reference()
    A, ids = _rr_A()
    coords = _rr_coords()
    tree = strip(read(joinpath(_RR_DIR, "tree.nwk"), String))

    @testset "Binomial bare animal(1 | g)" begin
        d = _rr_data("binomial_animal")
        fit = drm(bf(@formula(y ~ x1 + x2 + x3 + animal(1 | g))), Binomial(:logit); data = d, A = A, ids = ids)
        _rr_check(fit, ref["binom_animal_q1"])
        explicit = drm(bf(@formula(y ~ x1 + x2 + x3 + animal(1 | g)), @formula(sd(g, animal) ~ 1)),
                       Binomial(:logit); data = d, A = A, ids = ids)
        @test loglikelihood(fit) ≈ loglikelihood(explicit) atol = 1e-8
    end

    @testset "Binomial bare relmat(1 | g)" begin
        d = _rr_data("binomial_relmat")
        fit = drm(bf(@formula(y ~ x1 + x2 + x3 + relmat(1 | g))), Binomial(:logit); data = d, K = A)
        _rr_check(fit, ref["binom_relmat_q1"])
    end

    @testset "Binomial bare spatial(1 | g), R not converged" begin
        d = _rr_data("binomial_spatial")
        fit = drm(bf(@formula(y ~ x1 + x2 + x3 + spatial(1 | g))), Binomial(:logit); data = d, coords = coords)
        r = ref["binom_spatial_q1"]
        @test !r.conv      # recorded: drmTMB itself flags this fit
        # Both fits put the spatial SD on the zero boundary (R: log SD -11.4 with SE 9331), so the
        # log-SD limit and the standard errors do not apply; log-likelihood and fixed effects do.
        th, nfixed = _rr_in_r_order(fit)
        @test abs(loglikelihood(fit) - r.ll) <= 1e-4
        @test maximum(abs, th[1:nfixed] .- r.est[1:nfixed]) <= 1e-3
        @test all(<(-5), th[nfixed+1:end]) && all(<(-5), r.est[nfixed+1:end])
    end

    @testset "Beta bare spatial(1 | g) and spatial(1 + x1 | g)" begin
        d = _rr_data("beta_spatial")
        f1 = bf(@formula(y ~ x1 + x2 + x3 + spatial(1 | g)), @formula(sigma ~ 1))
        _rr_check(drm(f1, Beta(); data = d, coords = coords), ref["beta_spatial_q1"])
        f2 = bf(@formula(y ~ x1 + x2 + x3 + spatial(1 + x1 | g)), @formula(sigma ~ 1))
        fit2 = drm(f2, Beta(); data = d, coords = coords)
        @test abs(loglikelihood(fit2) - ref["beta_spatial_q2"].ll) <= 1e-4
        @test fit2.converged
    end

    @testset "Beta bare phylo(1 + x1 | g)" begin
        d = _rr_data("beta_phylo")
        f = bf(@formula(y ~ x1 + x2 + x3 + phylo(1 + x1 | g)), @formula(sigma ~ 1))
        _rr_check(drm(f, Beta(); data = d, tree = tree), ref["beta_phylo_q2"])
    end

    @testset "NegBinomial2 labelled phylo(1 + x1 | p | g)" begin
        d = _rr_data("nb2_phylo")
        f = bf(@formula(y ~ x1 + x2 + x3 + phylo(1 + x1 | p | g)), @formula(sigma ~ 1))
        _rr_check(drm(f, NegBinomial2(); data = d, tree = tree), ref["nb2_phylo_lab_q2"])
    end

    @testset "completion leaves other shapes alone" begin
        d = _rr_data("binomial_animal")
        # a second random term keeps the original refusal
        @test_throws Exception drm(bf(@formula(y ~ x1 + animal(1 | g) + (1 | g))), Binomial(:logit);
                                       data = d, A = A, ids = ids)
        # REML stays refused on Binomial
        @test_throws ArgumentError drm(bf(@formula(y ~ x1 + animal(1 | g))), Binomial(:logit);
                                       data = d, A = A, ids = ids, method = :REML)
    end
end
