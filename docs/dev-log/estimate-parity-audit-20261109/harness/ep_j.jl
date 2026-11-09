# Estimate-parity extractor, Julia side. Same cells (the wave-8/10 gap-map builder), same output layout as ep_r.R.
# Usage: julia --project=<env> ep_j.jl <ids.tsv> <outfile>   (cwd holds ep_cell_j.jl, cells_all.tsv, data/)
include("ep_cell_j.jl")
ids_all = readlines(ARGS[1])[2:end]; outfile = ARGS[2]
const NSIM = 1000
io = open(outfile, "w")
emit(id, sec, key, v1 = "", v2 = "", v3 = "") = (println(io, join([id, sec, key, v1, v2, v3], '\t')); flush(io))
num(x) = @sprintf("%.12g", x)
vec_emit(id, sec, v) = for (i, x) in enumerate(v); emit(id, sec, string(i), num(x)); end
msg(e) = replace(first(sprint(showerror, e), 300), r"[\t\n]+" => " ")
macro tryf(id, sec, ex)
    quote
        try
            $(esc(ex))
        catch e
            emit($(esc(id)), $(esc(sec)), "ERROR", msg(e))
        end
    end
end
# numeric delta-method SEs for any function of theta (sd, corr, vc entries): J V J'
function delta_se(fit, f::Function)
    th = fit.theta; V = fit.vcov
    x0 = f(); n = length(x0); k = length(th)
    J = zeros(n, k)
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
    t0 = time()
    fit = try
        f = build(cl); kw = kwargs_for(cl.kind)
        cl.est == "REML" ? drm(f, fam_of(cl.family); data = d, method = :REML, kw...) : drm(f, fam_of(cl.family); data = d, kw...)
    catch e
        emit(id, "meta", "FIT_ERROR", msg(e)); nothing
    end
    fit === nothing && continue
    emit(id, "meta", "secs", num(time() - t0))
    @tryf id "meta" begin
        emit(id, "meta", "logLik", num(loglikelihood(fit))); emit(id, "meta", "df", num(dof(fit)))
        emit(id, "meta", "AIC", num(aic(fit))); emit(id, "meta", "BIC", num(bic(fit))); emit(id, "meta", "nobs", num(nobs(fit)))
        emit(id, "meta", "converged", string(fit.converged)); emit(id, "meta", "estim_method", string(fit.estim_method))
        emit(id, "meta", "nblocks", join(string.(first.(fit.blocks)), ","))
    end
    @tryf id "coef" begin
        se = stderror(fit)
        for ((p, r), (_, nms)) in zip(fit.blocks, fit.coefnames), (j, ix) in enumerate(r)
            emit(id, "coef", string(p, ":", nms[j]), num(fit.theta[ix]), num(se[ix]))
        end
    end
    @tryf id "par" begin
        # response-scale random-effect SDs and correlations from vc(), with numeric delta SEs
        keys_ref = Ref(String[])
        function vals()
            v = vc(fit); out = Float64[]; ks = String[]
            for k in sort(collect(keys(v)); by = string)
                M = v[k]
                for i in 1:size(M, 1); push!(out, sqrt(M[i, i])); push!(ks, "sd|$(k)|$i"); end
                for i in 1:size(M, 1), j in 1:(i-1); push!(out, M[i, j] / sqrt(M[i, i] * M[j, j])); push!(ks, "cor|$(k)|$i,$j"); end
            end
            keys_ref[] = ks; return out
        end
        x0, sd = delta_se(fit, vals)
        for (k, a, b) in zip(keys_ref[], x0, sd); emit(id, "par", k, num(a), num(b), "response"); end
    end
    @tryf id "ci" begin
        for r in confint(fit); emit(id, "ci", string(r.param, ":", r.coef), num(r.lower), num(r.upper), "wald"); end
    end
    idx = unique(round.(Int, range(1, length(d.y); length = 8)))
    nd = (; x1 = d.x1[idx], x2 = d.x2[idx], x3 = d.x3[idx], g = d.g[idx], z = d.z[idx], z2 = d.z2[idx], g2 = d.g2[idx])
    @tryf id "pred_resp" vec_emit(id, "pred_resp", collect(predict(fit, nd; type = :response)))
    @tryf id "pred_link" vec_emit(id, "pred_link", collect(predict(fit, nd; type = :link)))
    @tryf id "fitted" vec_emit(id, "fitted", collect(fitted(fit)))
    @tryf id "resid_q" vec_emit(id, "resid_q", collect(residuals(fit; type = :quantile, rng = MersenneTwister(1))))
    @tryf id "resid_r" vec_emit(id, "resid_r", collect(residuals(fit; type = :response)))
    for (mode, kw) in (("marginal", (; data = d, kwargs_for(cl.kind)...)), ("zero", (; re = :zero)))
        @tryf id "simerr:$mode" begin
            sm = simulate(fit; nsim = NSIM, rng = MersenneTwister(1), kw...); n = length(d.y)
            sm = sm isa AbstractMatrix ? Matrix{Float64}(sm) : error("simulate returned $(typeof(sm))")
            size(sm, 1) != n && (sm = permutedims(sm))
            emit(id, "sim", mode * ":grand_mean", num(mean(sm))); emit(id, "sim", mode * ":grand_var", num(var(vec(sm))))
            emit(id, "sim", mode * ":mean_perobs_var", num(mean(var(sm; dims = 2)))); emit(id, "sim", mode * ":p_zero", num(mean(sm .== 0)))
            emit(id, "sim", mode * ":mean_perobs_mean_first8", num(mean(vec(mean(sm; dims = 2))[idx]))); emit(id, "sim", mode * ":dim", join(size(sm), "x"))
        end
    end
end
close(io)
