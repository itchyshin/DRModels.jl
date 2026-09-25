# Arc 2 receipt: DRModels.jl fits of the same fixtures `native-fit.R` fitted
# with drmTMB (engine = "tmb"), compared row by row with native.tsv.
#
# Run from the DRModels.jl repository root, after native-fit.R:
#   JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
#     julia --project=. docs/dev-log/evidence/arc2-metav-random-effect/julia-fit.jl
# Writes comparison.tsv next to this script and exits non-zero if any row
# misses the bar: |ΔlogLik| ≤ 1e-6, equal df, estimates ≤ 1e-5 relative.
using DRModels, LinearAlgebra, Printf

const HERE = joinpath("docs", "dev-log", "evidence", "arc2-metav-random-effect")
LinearAlgebra.BLAS.set_num_threads(1)

function read_tsv(path)
    lines = filter(!isempty, readlines(path))
    hdr = split(lines[1], '\t')
    cols = Dict(Symbol(h) => String[] for h in hdr)
    for l in lines[2:end]
        for (h, v) in zip(hdr, split(l, '\t'))
            push!(cols[Symbol(h)], v)
        end
    end
    return cols, Symbol.(hdr)
end

function fixture_data(name)
    cols, _ = read_tsv(joinpath(HERE, "fixtures", name * ".tsv"))
    num(k) = parse.(Float64, cols[k])
    nt = (; y = num(:y), x = num(:x), v = num(:v))
    for k in (:study, :sp, :id)
        haskey(cols, k) && (nt = merge(nt, NamedTuple{(k,)}((cols[k],))))
    end
    return nt
end

fixtures = [
    ("study-a", "y ~ x + meta_V(V = v) + (1 | study)",
     bf(@formula(y ~ x + meta_V(v) + (1 | study)), @formula(sigma ~ 1)), Dict{Symbol,Any}()),
    ("study-b", "y ~ x + meta_V(V = v) + (1 | study)",
     bf(@formula(y ~ x + meta_V(v) + (1 | study)), @formula(sigma ~ 1)), Dict{Symbol,Any}()),
    ("study-b", "y ~ x + meta_V(V = v) + (1 | study); sigma ~ x",
     bf(@formula(y ~ x + meta_V(v) + (1 | study)), @formula(sigma ~ x)), Dict{Symbol,Any}()),
    ("phylo-a", "y ~ x + meta_V(V = v) + phylo(1 | sp)",
     bf(@formula(y ~ x + meta_V(v) + phylo(1 | sp)), @formula(sigma ~ 1)),
     Dict{Symbol,Any}(:tree => read(joinpath(HERE, "fixtures", "phylo-a.nwk"), String))),
    ("phylo-b", "y ~ x + meta_V(V = v) + phylo(1 | sp)",
     bf(@formula(y ~ x + meta_V(v) + phylo(1 | sp)), @formula(sigma ~ 1)),
     Dict{Symbol,Any}(:tree => read(joinpath(HERE, "fixtures", "phylo-b.nwk"), String))),
    ("relmat-a", "y ~ x + meta_V(V = v) + relmat(1 | id)",
     bf(@formula(y ~ x + meta_V(v) + relmat(1 | id)), @formula(sigma ~ 1)),
     Dict{Symbol,Any}(:K => let rows = filter(!isempty, readlines(joinpath(HERE, "fixtures", "relmat-a-K.tsv")))
         reduce(vcat, [permutedims(parse.(Float64, split(r, '\t'))) for r in rows])
     end)),
    ("phylo-study", "y ~ x + meta_V(V = v) + phylo(1 | sp) + (1 | study)",
     bf(@formula(y ~ x + meta_V(v) + phylo(1 | sp) + (1 | study)), @formula(sigma ~ 1)),
     Dict{Symbol,Any}(:tree => read(joinpath(HERE, "fixtures", "phylo-a.nwk"), String))),
]

# Julia estimate on native's scale, by native quantity name. A phylo SD is
# reported by DRModels on the RAW branch-length scale (the default phylo-mean
# route's convention); drmTMB reports it on the tip-correlation scale. On an
# ultrametric tree they differ by sqrt(mean root-to-tip depth), the same factor
# drmTMB's R bridge applies (drm_julia_phylo_sd_scale), so it is applied here.
function julia_estimate(fit, q, kw)
    startswith(q, "mu:") && return coef(fit, :mu)[findfirst(==(q[4:end]), fit.coefnames[1][2])]
    if startswith(q, "sigma:")
        nm = Dict(fit.coefnames)[:sigma]
        return coef(fit, :sigma)[findfirst(==(q[7:end]), nm)]
    end
    startswith(q, "sd:") || error("unknown quantity $q")
    grp = match(r"\|\s*(\w+)\)$", q)[1]
    nm = Dict(fit.coefnames)[:resd]
    sd = exp(coef(fit, :resd)[findfirst(==(grp), nm)])
    if startswith(q, "sd:phylo(")
        phy = augmented_phy(kw[:tree])
        sd *= sqrt(sum(diag(sigma_phy_dense(phy))) / phy.n_leaves)
    end
    return sd
end

native, _ = read_tsv(joinpath(HERE, "native.tsv"))
out = IOBuffer()
println(out, join(["fixture", "formula", "quantity", "tmb_df", "julia_df", "tmb_logLik",
                   "julia_logLik", "abs_dlogLik", "tmb_estimate", "julia_estimate", "rel_diff",
                   "julia_converged", "pass"], '\t'))
allpass = true
for (name, ftxt, form, kw) in fixtures
    data = fixture_data(name)
    fit = drm(form, Gaussian(); data = data, kw...)
    jdf = length(coef(fit)); jll = loglik(fit)
    idx = findall(i -> native[:fixture][i] == name && native[:formula][i] == ftxt,
                  eachindex(native[:fixture]))
    isempty(idx) && error("no native rows for $name / $ftxt")
    for i in idx
        q = native[:quantity][i]
        tdf = parse(Int, native[:df][i]); tll = parse(Float64, native[:logLik][i])
        te = parse(Float64, native[:estimate][i]); je = julia_estimate(fit, q, kw)
        rel = abs(je - te) / max(abs(te), 1e-8)
        ok = tdf == jdf && abs(jll - tll) <= 1e-6 && rel <= 1e-5 && fit.converged
        global allpass &= ok
        println(out, join([name, ftxt, q, tdf, jdf, @sprintf("%.9f", tll), @sprintf("%.9f", jll),
                           @sprintf("%.3e", abs(jll - tll)), @sprintf("%.9f", te),
                           @sprintf("%.9f", je), @sprintf("%.3e", rel), fit.converged,
                           ok ? "PASS" : "FAIL"], '\t'))
    end
end
tsv = String(take!(out))
write(joinpath(HERE, "comparison.tsv"), tsv)
print(tsv)
println(allpass ? "ALL PASS" : "SOME ROWS FAIL")
allpass || exit(1)
