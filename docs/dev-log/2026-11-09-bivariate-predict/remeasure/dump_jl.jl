# Dump full fitted/predict vectors from DRModels for a list of gap-map cell ids. Usage: julia dump_jl.jl cells.tsv ids.txt outdir
using DRModels, DelimitedFiles, Logging, Printf, Statistics, LinearAlgebra
const DD = get(ENV, "FUZZ_DATA", "data")
cells_file, idfile, outdir = ARGS[1], ARGS[2], ARGS[3]
idlist = readlines(idfile); mkpath(outdir)
const CELLS = Dict{String,NamedTuple}()
for (k, l) in enumerate(readlines(cells_file))
    k == 1 && continue
    p = split(l, '\t'; keepempty = true)
    CELLS[String(p[1])] = (; id = String(p[1]), jl = String(p[6]))
end
const FACS = Set(["g", "g2", "h", "f", "f2", "gna", "gu", "y_cat", "y_ordf"])
const d = let (m, h) = readdlm(joinpath(DD, "data_v2.csv"), ',', Any; header = true)
    h = String.(vec(h)); cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(h)
        c = m[:, j]
        v = if n in FACS
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
    (; cols...)
end
# R spellings that Julia's @formula spells differently: interaction `a:b` is `a & b`, and `I(x^2)` has no
# counterpart, so it is rewritten to the function term `sq(x)`. Applied to every formula string.
sq(x) = x .^ 2
jl_spell(s) = replace(replace(s, "I(x1^2)" => "sq(x1)"), r"(\w+):(\w+)" => s"\1 & \2")
F(s) = eval(Meta.parse("@formula(" * jl_spell(s) * ")"))


function tomat(x)
    x isa NamedTuple && (x = x.prediction)
    if x isa AbstractDict
        ks = collect(keys(x)); return reduce(hcat, [Float64.([ismissing(v) ? NaN : v for v in x[k]]) for k in ks]), string.(ks)
    end
    x isa AbstractMatrix && return Float64.(x), String[]
    return reshape(Float64.([ismissing(v) ? NaN : v for v in x]), :, 1), String[]
end
for (k, id) in enumerate(idlist)
    cl = CELLS[id]; cl.jl == "" && continue
    fit = try Logging.with_logger(Logging.NullLogger()) do; Base.invokelatest(() -> eval(Meta.parse(cl.jl))); end catch e; nothing end
    fit === nothing && continue
    fv = try tomat(fitted(fit)) catch; nothing end
    pv = try tomat(predict(fit, d)) catch; nothing end
    fv !== nothing && writedlm(joinpath(outdir, @sprintf("%03d_fitted.csv", k)), fv[1], ',')
    pv !== nothing && writedlm(joinpath(outdir, @sprintf("%03d_predict.csv", k)), pv[1], ',')
    println(@sprintf("%03d\t%s\tnobs=%s\tfitted=%s\tpredict=%s", k, id, try nobs(fit) catch; "NA" end, fv === nothing ? "NULL" : join(size(fv[1]), "x") * (isempty(fv[2]) ? "" : ":" * join(fv[2], ",")), pv === nothing ? "NULL" : join(size(pv[1]), "x") * (isempty(pv[2]) ? "" : ":" * join(pv[2], ","))))
end
