# Julia side of the wave-9 agreement study. Usage: julia --project=<env> agree.jl <jobs.tsv> <k> <N> <out.tsv>
# Runs jobs k, k+N, k+2N, ... (1-based). Output: one tab-separated line per job.
using DRModels, DelimitedFiles, Logging, Printf
const DD = get(ENV, "W9_DATA", "data")
jobs_file, kk, NN, outfile = ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]), ARGS[4]
jobs = [split(l, '\t') for l in readlines(jobs_file) if !isempty(l)]
const NWK = strip(read(joinpath(DD, "tree.nwk"), String))
const COORDS = let m = readdlm(joinpath(DD, "coords.csv"), ',', Any; skipstart = 1)
    Dict(String(m[i, 1]) => [Float64(m[i, 2]), Float64(m[i, 3])] for i in 1:size(m, 1)) end
const IDS = String.(vec(readdlm(joinpath(DD, "A_ids.csv"), ',', Any; skipstart = 1)))
const AMAT = Matrix{Float64}(readdlm(joinpath(DD, "A.csv"), ',', Float64))
fam_of(f) = f == "gaussian" ? Gaussian() : f == "poisson" ? Poisson() : f == "nb2" ? NegBinomial2() :
    f == "gamma" ? DRModels.Gamma() : f == "binomial" ? Binomial(:logit) : error("family $f")
has_sigma(f) = !(f in ("poisson", "binomial"))
function load_data(fam, kind, seed)
    file = seed == 0 ? "data_$(fam)_$(kind).csv" : "data_$(fam)_$(kind)_s$(seed).csv"
    m, h = readdlm(joinpath(DD, file), ',', Any; header = true); h = String.(vec(h))
    col(n) = m[:, findfirst(==(n), h)]
    (; y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")), x3 = Float64.(col("x3")),
       z = Float64.(col("z")), g = String.(col("g")), g2 = String.(col("g2")))
end
const CNAME = ["intercept", "x1", "x2", "x3"]
const LEVEL = Dict("animal" => "animal", "relmat" => "relmat", "phylo" => "phylogenetic", "spatial" => "spatial")
mkf(s) = eval(Meta.parse("@formula(" * s * ")"))
function build(fam, kind, lab, q, pat)
    bar = join(["1", "x1", "x2", "x3"][1:q], " + "); labs = lab == 1 ? " | p | " : " | "
    inner = bar * labs * "g"
    re = "$kind($inner)"
    mu = mkf("y ~ x1 + x2 + x3 + $re")
    sel = pat == "none" ? Int[] : collect(1:q); rhs = pat == "one" ? "1" : "z"
    sds = [mkf("sd(g, $(LEVEL[kind]), coef($(CNAME[j]))) ~ $rhs") for j in sel]
    parts = Any[mu]; has_sigma(fam) && push!(parts, mkf("sigma ~ 1"))
    return bf(parts..., sds...)
end
kwargs_for(kind) = kind == "phylo" ? (; tree = NWK) : kind == "spatial" ? (; coords = COORDS) :
    kind == "animal" ? (; A = AMAT, ids = IDS) : (; K = AMAT)
clean(x) = replace(String(x), '\t' => ' ', '\n' => ' ')
open(outfile, "a") do io
    for i in kk:NN:length(jobs)
        sid, cell, seed = jobs[i][1], jobs[i][2], parse(Int, jobs[i][3])
        fam, kind, lab, q, pat = split(cell, '|')[1], split(cell, '|')[2], parse(Int, split(cell, '|')[3][4:end]),
            parse(Int, split(cell, '|')[4][2:end]), split(cell, '|')[5][4:end]
        t0 = time(); msgs = String[]
        local fit = nothing; local err = nothing
        lg = Logging.SimpleLogger(devnull)
        try
            d = load_data(fam, kind, seed)
            Logging.with_logger(lg) do
                fit = drm(build(fam, kind, lab, q, pat), fam_of(fam); data = d, kwargs_for(kind)...)
            end
        catch e
            err = first(replace(sprint(showerror, e), r"\s+" => " "), 300)
        end
        secs = time() - t0
        if fit === nothing
            println(io, join((sid, seed, "ERROR", "", "", "", "", "", "", @sprintf("%.1f", secs), clean(something(err, ""))), "\t")); flush(io); continue
        end
        ll = Float64(loglikelihood(fit)); conv = fit.converged
        se = try Float64.(stderror(fit)) catch; Float64[] end
        th = Float64.(fit.theta)
        blocks = join(("$(first(p)):$(first(last(p)))-$(last(last(p)))" for p in fit.blocks), ";")
        println(io, join((sid, seed, "OK", @sprintf("%.10f", ll), string(conv), join((@sprintf("%.8g", v) for v in th), ","),
            join((@sprintf("%.8g", v) for v in se), ","), clean(blocks), string(try join(string.(fit.scales[:q2_restarts]), ",") catch; "" end), @sprintf("%.1f", secs), ""), "\t")); flush(io)
    end
end
