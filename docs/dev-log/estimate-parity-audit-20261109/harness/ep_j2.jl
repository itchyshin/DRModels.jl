# Second Julia pass: warnings, convergence status, re_sd accessor with delta SEs, marginal simulate without the harness-only `ids` kwarg.
include("ep_cell_j.jl")
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
ids_all = readlines(ARGS[1])[2:end]; outfile = ARGS[2]
const NSIM = 1000
io = open(outfile, "w")
emit(id, sec, key, v1 = "", v2 = "", v3 = "") = (println(io, join([id, sec, key, v1, v2, v3], '\t')); flush(io))
num(x) = @sprintf("%.12g", x)
msg(e) = replace(first(sprint(showerror, e), 300), r"[\t\n]+" => " ")
function delta_se(fit, f::Function)
    th = fit.theta; V = fit.vcov; x0 = f(); n = length(x0); k = length(th); J = zeros(n, k)
    for j in 1:k
        h = 1e-6 * max(1.0, abs(th[j])); old = th[j]
        th[j] = old + h; xp = f(); th[j] = old - h; xm = f(); th[j] = old
        J[:, j] = (xp .- xm) ./ (2h)
    end
    return x0, sqrt.(max.(diag(J * V * J'), 0.0))
end
for id in ids_all
    cl = CELLS[id]
    d = load_data(cl.family, cl.kind in ("indep", "slope") ? "ordinary" : cl.kind)
    lg = CaptureLogger(String[])
    fit = nothing
    Logging.with_logger(lg) do
        try
            f = build(cl); kw = kwargs_for(cl.kind)
            fit = cl.est == "REML" ? drm(f, fam_of(cl.family); data = d, method = :REML, kw...) : drm(f, fam_of(cl.family); data = d, kw...)
        catch e
            emit(id, "meta", "FIT_ERROR", msg(e))
        end
    end
    fit === nothing && continue
    for (i, m) in enumerate(unique(lg.msgs)); i <= 4 && emit(id, "warn", string(i), replace(first(m, 220), r"[\t\n]+" => " ")); end
    try emit(id, "conv", "is_converged", string(is_converged(fit)), string(convergence_status(fit))[1:min(end, 200)]) catch e; emit(id, "conv", "ERROR", msg(e)) end
    try emit(id, "conv", "vcov_finite", string(all(isfinite, vcov(fit))), string(count(!isfinite, stderror(fit)))) catch e; emit(id, "conv", "vcov_ERROR", msg(e)) end
    for sc in (:native, :drmtmb)
        try
            kws = sc === :drmtmb && cl.kind == "phylo" ? (; scale = sc, tree = NWK) : (; scale = sc)
            sd0 = re_sd(fit; kws...); ks = sort(collect(keys(sd0)); by = string)
            vals() = (s = re_sd(fit; kws...); [s[k] for k in ks])
            x0, se = delta_se(fit, vals)
            for (k, a, b) in zip(ks, x0, se); emit(id, "resd:" * string(sc), string(k), num(a), num(b)); end
        catch e
            emit(id, "resd:" * string(sc), "ERROR", msg(e))
        end
    end
    idx = unique(round.(Int, range(1, length(d.y); length = 8)))
    kwm = (; data = d, (k => v for (k, v) in pairs(kwargs_for(cl.kind)) if k != :ids)...)
    try
        sm = Matrix{Float64}(simulate(fit; nsim = NSIM, rng = MersenneTwister(1), kwm...))
        size(sm, 1) != length(d.y) && (sm = permutedims(sm))
        emit(id, "sim2", "grand_mean", num(mean(sm))); emit(id, "sim2", "grand_var", num(var(vec(sm))))
        emit(id, "sim2", "mean_perobs_var", num(mean(var(sm; dims = 2)))); emit(id, "sim2", "p_zero", num(mean(sm .== 0)))
    catch e
        emit(id, "sim2", "ERROR", msg(e))
    end
end
close(io)
