# Structured and positive-family SD cells added after the wave-10 gap map, checked against drmTMB 0.8.1:
#   Poisson, NB2   spatial(1 | g) + sd(g, spatial) ~ 1 | z      (coords; kernel exp(-d / median d) + 1e-6 I)
#   NB2            phylo(1 | g)   + sd(g, phylogenetic) ~ 1 | z (tip correlation of an ultrametric tree)
#   Gamma          (1 | g) + (0 + x1 | g) with ONE sd formula (the other coefficient keeps a constant SD)
#   LogNormal      (1 | g) + (1 | g2) with sd(g) and sd(g2)     (log y is Gaussian: Gaussian route + Jacobian)
#   Student        (1 | g) + (1 | g2) with sd(g) and sd(g2)     (crossed Laplace, log SD regression per factor)
# Reference: test/fixtures/struct_sd_counts/reference_drmtmb.csv (script ref_struct_sd_counts.R, same folder,
# says how to regenerate it); datasets are the committed wave-8 gap-map data (400 rows).
# Thresholds, fixed before the coefficient comparison (docs/dev-log/2026-11-09-struct-sd-counts.md):
#   |loglik_julia - loglik_R| <= 1e-4;  every mu / sigma / nu / SD coefficient |diff| <= 1e-3 and <= 0.02 R-SE;
#   |se_julia / se_R - 1| <= 1e-2 (finite R SEs).
# Binomial relmat(1 + x1 | g) + SD is NOT here: R does not converge on those two cells (is_converged() is FALSE,
# an SD coefficient at -10.7) so they stay refused; the refusal is pinned at the end.
using DRModels, Test, DelimitedFiles

const _SSC_DIR = joinpath(@__DIR__, "fixtures", "struct_sd_counts")

function _ssc_data(name)
    m, h = readdlm(joinpath(_SSC_DIR, "data_$(name).csv"), ',', Any; header = true)
    h = String.(vec(h))
    col(n) = m[:, findfirst(==(n), h)]
    return (; y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")), x3 = Float64.(col("x3")),
            z = Float64.(col("z")), z2 = Float64.(col("z2")), g = String.(col("g")), g2 = String.(col("g2")))
end

const _SSC_COORDS = let m = readdlm(joinpath(_SSC_DIR, "coords.csv"), ',', Any; skipstart = 1)
    Dict(String(m[i, 1]) => [Float64(m[i, 2]), Float64(m[i, 3])] for i in 1:size(m, 1))
end
const _SSC_TREE = strip(read(joinpath(_SSC_DIR, "tree.nwk"), String))

# Reference table: cell id => (loglik, [(block, name, estimate, se), ...]) in R's coefficient order.
const _SSC_REF = let (m, h) = readdlm(joinpath(_SSC_DIR, "reference_drmtmb.csv"), ',', Any; header = true)
    h = String.(vec(h)); ix(n) = findfirst(==(n), h)
    out = Dict{String,Any}()
    for r in 1:size(m, 1)
        id = String(m[r, ix("cell_id")])
        e = get!(out, id, (; loglik = Float64(m[r, ix("loglik")]), rows = Any[]))
        push!(e.rows, (String(m[r, ix("block")]), String(m[r, ix("name")]),
                       Float64(m[r, ix("estimate")]), Float64(m[r, ix("se")])))
    end
    out
end

# (cell id, family, dataset, mean-and-SD formulas, keywords)
_ssc_cases() = [
    ("poisson|spatial|lab0|q1|sd-one|cm0|ML", Poisson(), "poisson_spatial",
     [@formula(y ~ x1 + x2 + x3 + spatial(1 | g)), @formula(sd(g, spatial) ~ 1)], (; coords = _SSC_COORDS)),
    ("poisson|spatial|lab0|q1|sd-z|cm0|ML", Poisson(), "poisson_spatial",
     [@formula(y ~ x1 + x2 + x3 + spatial(1 | g)), @formula(sd(g, spatial) ~ z)], (; coords = _SSC_COORDS)),
    ("nb2|spatial|lab0|q1|sd-one|cm0|ML", NegBinomial2(), "nb2_spatial",
     [@formula(y ~ x1 + x2 + x3 + spatial(1 | g)), @formula(sigma ~ 1), @formula(sd(g, spatial) ~ 1)],
     (; coords = _SSC_COORDS)),
    ("nb2|spatial|lab0|q1|sd-z|cm0|ML", NegBinomial2(), "nb2_spatial",
     [@formula(y ~ x1 + x2 + x3 + spatial(1 | g)), @formula(sigma ~ 1), @formula(sd(g, spatial) ~ z)],
     (; coords = _SSC_COORDS)),
    ("nb2|phylo|lab0|q1|sd-one|cm0|ML", NegBinomial2(), "nb2_phylo",
     [@formula(y ~ x1 + x2 + x3 + phylo(1 | g)), @formula(sigma ~ 1), @formula(sd(g, phylogenetic) ~ 1)],
     (; tree = _SSC_TREE)),
    ("nb2|phylo|lab0|q1|sd-z|cm0|ML", NegBinomial2(), "nb2_phylo",
     [@formula(y ~ x1 + x2 + x3 + phylo(1 | g)), @formula(sigma ~ 1), @formula(sd(g, phylogenetic) ~ z)],
     (; tree = _SSC_TREE)),
    ("gamma|indep|lab0|q2|sd-first|cm0|ML", DRModels.Gamma(), "gamma_ordinary",
     [@formula(y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g)), @formula(sigma ~ 1),
      @formula(sd(g, coef(intercept)) ~ z)], (;)),
    ("gamma|indep|lab0|q2|sd-last|cm0|ML", DRModels.Gamma(), "gamma_ordinary",
     [@formula(y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g)), @formula(sigma ~ 1),
      @formula(sd(g, coef(x1)) ~ z)], (;)),
    ("lognormal|crossed|lab0|q1|sd-one|cm0|ML", LogNormal(), "lognormal_crossed",
     [@formula(y ~ x1 + x2 + x3 + (1 | g) + (1 | g2)), @formula(sigma ~ 1),
      @formula(sd(g) ~ 1), @formula(sd(g2) ~ 1)], (;)),
    ("lognormal|crossed|lab0|q1|sd-z|cm0|ML", LogNormal(), "lognormal_crossed",
     [@formula(y ~ x1 + x2 + x3 + (1 | g) + (1 | g2)), @formula(sigma ~ 1),
      @formula(sd(g) ~ z), @formula(sd(g2) ~ z2)], (;)),
    ("student|crossed|lab0|q1|sd-one|cm0|ML", Student(), "student_crossed",
     [@formula(y ~ x1 + x2 + x3 + (1 | g) + (1 | g2)), @formula(sigma ~ 1),
      @formula(sd(g) ~ 1), @formula(sd(g2) ~ 1)], (;)),
    ("student|crossed|lab0|q1|sd-z|cm0|ML", Student(), "student_crossed",
     [@formula(y ~ x1 + x2 + x3 + (1 | g) + (1 | g2)), @formula(sigma ~ 1),
      @formula(sd(g) ~ z), @formula(sd(g2) ~ z2)], (;)),
]

# Julia blocks matched to R blocks: mu / sigma / nu by name; each R SD block takes the first unused
# Julia SD block with the same coefficient names (R omits the constant SD of the unselected term of a
# one-target independent pair, so Julia may carry one more SD block than R).
function _ssc_match(fit, rows)
    jb = [first(p) => last(p) for p in fit.blocks]
    jn = Dict(first(p) => last(p) for p in fit.coefnames)
    used = Set{Symbol}()
    out = Any[]
    rblocks = unique(first.(rows))
    for rb in rblocks
        rr = filter(r -> r[1] == rb, rows)
        jsym = if rb in ("mu", "sigma", "nu")
            Symbol(rb)
        else
            cand = [first(p) for p in jb if !(first(p) in (:mu, :sigma, :nu)) && !(first(p) in used) &&
                    jn[first(p)] == [r[2] for r in rr]]
            isempty(cand) ? nothing : first(cand)
        end
        if jsym === nothing
            # The Gaussian crossed SD route reports every SD coefficient in one `:sd` block, named
            # "group: coefficient"; match the SD rows by position when the counts agree.
            rsd = filter(r -> !(r[1] in ("mu", "sigma", "nu")), rows)
            jsd = reduce(vcat, [collect(last(p)) for p in jb if !(first(p) in (:mu, :sigma, :nu))]; init = Int[])
            length(rsd) == length(jsd) || return nothing
            return vcat([(k, r) for (k, r) in out if first(r) in ("mu", "sigma", "nu")],
                        collect(zip(jsd, rsd)))
        end
        push!(used, jsym)
        rng = only(last(p) for p in jb if first(p) == jsym)
        length(rng) == length(rr) || return nothing
        for (k, r) in zip(rng, rr)
            push!(out, (k, r))
        end
    end
    return out
end

@testset "structured / positive-family SD cells agree with drmTMB 0.8.1" begin
    for (id, fam, dsname, forms, kw) in _ssc_cases()
        @testset "$id" begin
            ref = _SSC_REF[id]
            fit = drm(bf(forms...), fam; data = _ssc_data(dsname), kw...)
            @test fit.converged === true
            @test isfinite(loglikelihood(fit))
            @test abs(loglikelihood(fit) - ref.loglik) <= 1e-4
            _report = !isempty(get(ENV, "DRM_SSC_REPORT", ""))
            matched = _ssc_match(fit, ref.rows)
            @test matched !== nothing
            if matched !== nothing
                θ = fit.theta
                se = stderror(fit)
                @test all(isfinite, se)
                _report && println("REPORT $id dLL=", loglikelihood(fit) - ref.loglik,
                    " max|dcoef|=", maximum(abs(θ[k] - r[3]) for (k, r) in matched),
                    " max|dcoef|/SE=", maximum(abs(θ[k] - r[3]) / r[4] for (k, r) in matched),
                    " se_ratio=", extrema(se[k] / r[4] for (k, r) in matched))
                for (k, r) in matched
                    @test abs(θ[k] - r[3]) <= 1e-3
                    @test abs(θ[k] - r[3]) <= 0.02 * r[4]
                    @test abs(se[k] / r[4] - 1) <= 1e-2
                end
            end
        end
    end
end

@testset "constant-SD Student crossed route unchanged by the per-level SD generalisation" begin
    d = _ssc_data("student_crossed")
    fit = drm(bf(@formula(y ~ x1 + x2 + x3 + (1 | g) + (1 | g2))), Student(); data = d)
    @test fit.converged === true
    @test :resd in first.(fit.blocks)
    @test isfinite(loglikelihood(fit))
end

@testset "records: refused cells stay refused with a clear message" begin
    d = _ssc_data("binomial_relmat")
    A = readdlm(joinpath(_SSC_DIR, "A.csv"), ',', Float64)
    @test_throws ArgumentError drm(bf(@formula(y ~ x1 + x2 + x3 + relmat(1 + x1 | g)),
        @formula(sd(g, relmat, coef(intercept)) ~ 1), @formula(sd(g, relmat, coef(x1)) ~ 1)),
        Binomial(:logit); data = d, K = A)
    # Gamma: a one-target pair must name a coefficient of the pair.
    dg = _ssc_data("gamma_ordinary")
    @test_throws ArgumentError drm(bf(@formula(y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g)), @formula(sigma ~ 1),
        @formula(sd(g, coef(x2)) ~ z)), DRModels.Gamma(); data = dg)
end
