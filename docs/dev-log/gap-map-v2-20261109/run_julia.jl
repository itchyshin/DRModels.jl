# Native Julia DRModels runner for the gap-map v2 grid. One process, many cells, one result line per cell.
# Usage: julia --project=<env> run_julia.jl <cells.tsv> <idlist.txt> <start_index> <outfile>
# Env: FUZZ_DATA (data dir).
using DRModels, DelimitedFiles, Logging, Printf, Statistics, LinearAlgebra
const DD = get(ENV, "FUZZ_DATA", "data")
cells_file, idfile, start, outfile = ARGS[1], ARGS[2], parse(Int, ARGS[3]), ARGS[4]
idlist = readlines(idfile)

const CELLS = Dict{String,NamedTuple}()
for (k, l) in enumerate(readlines(cells_file))
    k == 1 && continue
    p = split(l, '\t'; keepempty = true)
    CELLS[String(p[1])] = (; id = String(p[1]), probes = String(p[4]), r = String(p[5]), jl = String(p[6]))
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

struct CaptureLogger <: Logging.AbstractLogger
    msgs::Vector{String}
end
Logging.min_enabled_level(::CaptureLogger) = Logging.Warn
Logging.shouldlog(::CaptureLogger, args...) = true
Logging.catch_exceptions(::CaptureLogger) = true
function Logging.handle_message(l::CaptureLogger, level, message, args...; kwargs...)
    level >= Logging.Warn && push!(l.msgs, string(message))
    return nothing
end
function first_line(s)
    ls = filter(!isempty, [strip(replace(l, '\t' => ' ')) for l in split(String(s), '\n')])
    isempty(ls) ? "" : first(ls[1], 400)
end
full_msg(s) = first(strip(replace(String(s), r"\s+" => " ")), 600)
clean(x) = replace(String(x), '\t' => ' ', '\n' => ' ')
function mm_msg(e::MethodError)
    a = e.args
    !isempty(a) && a[1] isa NamedTuple && return "MethodError: no method matching keyword call " * string(a[2]) * " with keywords " * string(keys(a[1]))
    return "MethodError: no method matching " * string(e.f) * "(" * join(string.(typeof.(a)), ", ") * ")"
end

fnum(x) = @sprintf("%.10g", x)
fin(v) = (v = collect(skipmissing(v)); Float64[x for x in v if x isa Real && isfinite(x)])
function probe(f)
    try
        return (true, f())
    catch e
        return (false, first(clean(sprint(showerror, e)), 120))
    end
end
pstat(p) = p[1] ? "ok" : "ERR:" * p[2]

function subset_nt(nt, idx)
    NamedTuple{keys(nt)}(map(v -> v[idx], values(nt)))
end
numvec(x) = x isa NamedTuple ? numvec(x.prediction) : x isa AbstractDict ? numvec(first(values(x))) : x isa AbstractMatrix ? vec(Float64.(x)) : x isa AbstractVector ? Float64.([ismissing(v) ? NaN : v for v in x]) : [Float64(x)]

function run_one(id)
    cl = CELLS[id]
    cl.jl == "" && return ("NOSYNTAX", "", "Julia has no spelling for this request", "", "", "", "", "0", "", "", "")
    lg = CaptureLogger(String[])
    t0 = time()
    local fit, err = nothing, nothing
    Logging.with_logger(lg) do
        try
            fit = Base.invokelatest(() -> eval(Meta.parse(cl.jl)))
        catch e
            err = (e, catch_backtrace())
        end
    end
    secs = time() - t0
    w1 = isempty(lg.msgs) ? "" : first_line(lg.msgs[1])
    if err !== nothing
        e = err[1]
        msg = e isa MethodError ? mm_msg(e) : first_line(sprint(showerror, e))
        fullm = e isa MethodError ? msg : full_msg(sprint(showerror, e))
        loc = ""
        if !(e isa ArgumentError)
            for fr in Base.stacktrace(err[2])
                s = string(fr.file)
                if occursin("DRModels", s) || occursin("/src/", s)
                    loc = basename(s) * ":" * string(fr.line); break
                end
            end
        end
        return ("ERROR", string(typeof(e)), msg, fullm, "", "", loc == "" ? "" : "at=" * loc, @sprintf("%.2f", secs), w1, "", "")
    end
    if fit isa NamedTuple   # cross-family route returns a NamedTuple: no generic accessors apply, only loglik and converged
        okm = isfinite(fit.loglik) && fit.converged === true
        return (okm ? "FITS" : "NONCONVERGED", "", "", "", @sprintf("%.10f", fit.loglik), string(fit.converged), "", @sprintf("%.2f", secs), w1, "mixed_family_namedtuple=true", "")
    end
    ll = try Float64(loglikelihood(fit)) catch; NaN end
    conv = try fit.converged catch; missing end
    ok = isfinite(ll) && conv === true
    m = String[]; pr = String[]
    pa(k, v) = push!(m, string(k, "=", v))
    pp(k, p) = push!(pr, string(k, "=", pstat(p)))
    pa("aic", try fnum(aic(fit)) catch; "NA" end); pa("bic", try fnum(bic(fit)) catch; "NA" end)
    pa("nobs", try string(nobs(fit)) catch; "NA" end)
    p = probe(() -> fitted(fit)); pp("fitted", p)
    if p[1]; v = fin(numvec(p[2])); pa("fit_n", length(v)); pa("fit_mean", fnum(mean(v))); pa("fit_sd", fnum(std(v))); end
    p = probe(() -> predict(fit, d)); pp("predict", p)
    if p[1]; v = fin(numvec(p[2])); pa("pred_mean", fnum(mean(v))); end
    idx = [5, 1, 9]
    nd = subset_nt(d, idx)
    pf = probe(() -> predict(fit, nd)); pp("predict_new", pf)
    if pf[1]; pa("pred_new", join(fnum.(numvec(pf[2])), ",")); end
    if p[1] && pf[1]
        full = numvec(p[2]); sm = numvec(pf[2])
        pa("pred_new_consistent", length(full) >= 9 && length(sm) == 3 && isapprox(sm, full[idx]; rtol = 1e-6, atol = 1e-9))
    end
    p = probe(() -> residuals(fit)); pp("resid", p)
    if p[1]; v = fin(numvec(p[2])); pa("resid_sd", fnum(std(v))); end
    p = probe(() -> simulate(fit)); pp("simulate", p)
    if p[1]; pa("sim_n", length(fin(numvec(p[2])))); end
    p = probe(() -> vcov(fit)); pp("vcov", p)
    if p[1]; V = Matrix(p[2]); pa("npar", size(V, 1)); pa("se_finite", all(isfinite, sqrt.(abs.(diag(V))))); end
    p = probe(() -> coef(fit)); pp("coef", p)
    if p[1]; pa("ncoef", length(p[2])); end
    if cl.probes == "ext"
        pp("predict_link", probe(() -> predict(fit, nd; type = :link)))
        pp("predict_parameters", probe(() -> predict_parameters(fit, nd)))
        ndn = haskey(d, :f) ? merge(nd, (; f = ["a", "b", "z"])) : nd
        pp("predict_unseen_level", probe(() -> predict(fit, ndn)))
        p = probe(() -> simulate(fit; nsim = 3)); pp("simulate3", p); if p[1]; pa("sim3_ncol", p[2] isa AbstractMatrix ? size(p[2], 2) : 1); end
        p = probe(() -> confint(fit)); pp("confint", p)
        if p[1]; M = p[2]; try pa("ci_rows", size(M, 1)); pa("ci_finite", mean(isfinite.(Float64.(Matrix(M[:, end-1:end]))))) catch; end; end
        pp("summary", probe(() -> sprint(show, MIME("text/plain"), summary(fit))))
        
    end
    return (ok ? "FITS" : "NONCONVERGED", "", "", "", @sprintf("%.10f", ll), string(conv), "", @sprintf("%.2f", secs), w1, join(m, ";"), join(pr, ";"))
end

open(outfile, "a") do io
    for i in start:length(idlist)
        id = idlist[i]
        println(io, "S\t", i, "\t", id); flush(io)
        r = try run_one(id) catch e
            ("RUNNER_ERROR", string(typeof(e)), first_line(sprint(showerror, e)), full_msg(sprint(showerror, e)), "", "", "", "", "", "", "")
        end
        println(io, "R\t", i, "\t", id, "\t", join(clean.(collect(r)), "\t")); flush(io)
    end
end
