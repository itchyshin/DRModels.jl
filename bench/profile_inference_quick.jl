# profile_inference_quick.jl -- quick timing for profile-likelihood and bootstrap.
#
# This isolates the inference pipeline cost from the single-fit crossed-Laplace
# benchmark. It intentionally uses small deterministic Gaussian fixtures so the
# script stays cheap enough to run during engine work.
#
# Run:
#   julia --project=bench bench/profile_inference_quick.jl

import Pkg
Pkg.activate(dirname(@__DIR__))

using DRModels
using LinearAlgebra, Printf, Random, Statistics
using Distributions: Chisq, quantile
import Distributions
import Optim

BLAS.set_num_threads(1)

const OUT = joinpath(@__DIR__, "..", "report", "inference-profile-quick.md")

function fixed_gaussian_fixture()
    rng = MersenneTwister(8101)
    n = 600
    x = randn(rng, n)
    z = randn(rng, n)
    y = 0.4 .+ 0.75 .* x .- 0.35 .* z .+ exp(-0.45) .* randn(rng, n)
    form = bf(@formula(y ~ x + z), @formula(sigma ~ 1))
    return form, (; y, x, z)
end

function crossed_gaussian_fixture()
    rng = MersenneTwister(8102)
    n = 900
    G = 20
    H = 20
    x = randn(rng, n)
    g = [Symbol("g", rand(rng, 1:G)) for _ in 1:n]
    h = [Symbol("h", rand(rng, 1:H)) for _ in 1:n]
    bg = 0.45 .* randn(rng, G)
    bh = 0.35 .* randn(rng, H)
    y = [
        0.25 +
        0.55 * x[i] +
        bg[parse(Int, String(g[i])[2:end])] +
        bh[parse(Int, String(h[i])[2:end])] +
        0.45 * randn(rng) for i in 1:n
    ]
    form = bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1))
    return form, (; y=Float64.(y), x, g, h)
end

function poisson_re_fixture()
    rng = MersenneTwister(8103)
    G = 30
    m = 20
    n = G * m
    x = randn(rng, n)
    g = [Symbol("g", fld(i - 1, m) + 1) for i in 1:n]
    bg = 0.45 .* randn(rng, G)
    η = [0.25 + 0.55 * x[i] + bg[fld(i - 1, m) + 1] for i in 1:n]
    y = Float64[rand(rng, Distributions.Poisson(exp(η[i]))) for i in 1:n]
    form = bf(@formula(y ~ x + (1 | g)))
    return form, (; y, x, g)
end

function timecall(f)
    out = nothing
    t = @elapsed out = f()
    return t, out
end

function profiled_nll_warm(nll, θhat, k, v, ustart)
    p = length(θhat)
    idx = [i for i in 1:p if i != k]
    function obj(u)
        θ = Vector{eltype(u)}(undef, p)
        θ[k] = convert(eltype(u), v)
        @inbounds for (t, i) in enumerate(idx)
            θ[i] = u[t]
        end
        return nll(θ)
    end
    res = Optim.optimize(obj, ustart, Optim.LBFGS(); autodiff=:forward)
    return Optim.minimum(res), Optim.minimizer(res)
end

function profile_endpoint_warm(nll, θhat, k, nllhat, half, s, dir)
    idx = [i for i in 1:length(θhat) if i != k]
    ustart = copy(θhat[idx])
    target = nllhat + half
    function h(t)
        val, u = profiled_nll_warm(nll, θhat, k, θhat[k] + dir * t, ustart)
        ustart .= u
        return val - target
    end
    tlo = 0.0
    thi = s
    hval = h(thi)
    iters = 0
    while hval < 0 && iters < 40
        tlo = thi
        thi *= 1.6
        hval = h(thi)
        iters += 1
    end
    hval < 0 && return dir < 0 ? -Inf : Inf
    for _ in 1:60
        tm = (tlo + thi) / 2
        h(tm) < 0 ? (tlo = tm) : (thi = tm)
        thi - tlo < 1e-7 && break
    end
    return θhat[k] + dir * (tlo + thi) / 2
end

function profile_ci_warm(fit; level=0.95)
    nll = fit.nll
    θhat = copy(fit.theta)
    nllhat = nll(θhat)
    half = quantile(Chisq(1), level) / 2
    se = sqrt.(diag(fit.vcov))
    out = NamedTuple[]
    for ((pp, r), (_, nms)) in zip(fit.blocks, fit.coefnames)
        for (j, k) in enumerate(r)
            est = θhat[k]
            s = (isfinite(se[k]) && se[k] > 0) ? se[k] : max(abs(est), 1.0)
            lo = profile_endpoint_warm(nll, θhat, k, nllhat, half, s, -1)
            hi = profile_endpoint_warm(nll, θhat, k, nllhat, half, s, +1)
            push!(out, (param=pp, coef=nms[j], estimate=est, lower=lo, upper=hi))
        end
    end
    return out
end

function profile_ci_threaded_warm(fit; level=0.95)
    nll = fit.nll
    θhat = copy(fit.theta)
    nllhat = nll(θhat)
    half = quantile(Chisq(1), level) / 2
    se = sqrt.(diag(fit.vcov))
    jobs = NamedTuple[]
    for ((pp, r), (_, nms)) in zip(fit.blocks, fit.coefnames)
        for (j, k) in enumerate(r)
            push!(jobs, (param=pp, coef=nms[j], k=k))
        end
    end
    out = Vector{NamedTuple}(undef, length(jobs))
    Threads.@threads for i in eachindex(jobs)
        job = jobs[i]
        k = job.k
        est = θhat[k]
        s = (isfinite(se[k]) && se[k] > 0) ? se[k] : max(abs(est), 1.0)
        lo = profile_endpoint_warm(nll, θhat, k, nllhat, half, s, -1)
        hi = profile_endpoint_warm(nll, θhat, k, nllhat, half, s, +1)
        out[i] = (param=job.param, coef=job.coef, estimate=est, lower=lo, upper=hi)
    end
    return out
end

function parameter_surface_cold(fit, k1, k2; npoints=11, span=3.0)
    nll = fit.nll
    θhat = copy(fit.theta)
    nllhat = nll(θhat)
    se = sqrt.(diag(fit.vcov))
    s1 = (isfinite(se[k1]) && se[k1] > 0) ? se[k1] : max(abs(θhat[k1]), 1.0)
    s2 = (isfinite(se[k2]) && se[k2] > 0) ? se[k2] : max(abs(θhat[k2]), 1.0)
    x = range(θhat[k1] - span * s1, θhat[k1] + span * s1; length=npoints)
    y = range(θhat[k2] - span * s2, θhat[k2] + span * s2; length=npoints)
    rest = [i for i in 1:length(θhat) if i != k1 && i != k2]
    z = Matrix{Float64}(undef, npoints, npoints)
    for i in 1:npoints, j in 1:npoints
        function obj(u)
            θ = Vector{eltype(u)}(undef, length(θhat))
            θ[k1] = convert(eltype(u), x[i])
            θ[k2] = convert(eltype(u), y[j])
            @inbounds for (t, r) in enumerate(rest)
                θ[r] = u[t]
            end
            return nll(θ)
        end
        res = Optim.optimize(obj, θhat[rest], Optim.LBFGS(); autodiff=:forward)
        z[i, j] = max(0.0, 2 * (Optim.minimum(res) - nllhat))
    end
    return (x=collect(x), y=collect(y), z=z)
end

form_fixed, data_fixed = fixed_gaussian_fixture()
drm(form_fixed, Gaussian(); data=data_fixed)
GC.gc()
t_fit_fixed, fit_fixed = timecall(() -> drm(form_fixed, Gaussian(); data=data_fixed))
confint(fit_fixed; method=:wald)
confint(fit_fixed; method=:profile)
bootstrap_ci(form_fixed, Gaussian(); data=data_fixed, B=2, rng=MersenneTwister(90))
bootstrap_ci(
    form_fixed, Gaussian(); data=data_fixed, B=2, rng=MersenneTwister(90), threads=true
)
bootstrap_summary(form_fixed, Gaussian(); data=data_fixed, B=2, rng=MersenneTwister(90))
profile_curve(fit_fixed, 2; npoints=21)
parameter_surface(fit_fixed, 1, 2; npoints=7)
GC.gc()
t_wald, wald = timecall(() -> confint(fit_fixed; method=:wald))
t_profile, prof = timecall(() -> confint(fit_fixed; method=:profile))
t_boot20, boot20 = timecall(
    () -> bootstrap_ci(
        form_fixed, Gaussian(); data=data_fixed, B=20, rng=MersenneTwister(91)
    ),
)
t_boot20_threads, boot20_threads = timecall(
    () -> bootstrap_ci(
        form_fixed,
        Gaussian();
        data=data_fixed,
        B=20,
        rng=MersenneTwister(91),
        threads=true,
    ),
)
t_boot_summary20, boot_summary20 = timecall(
    () -> bootstrap_summary(
        form_fixed, Gaussian(); data=data_fixed, B=20, rng=MersenneTwister(91)
    ),
)
t_curve_fixed, curve_fixed = timecall(() -> profile_curve(fit_fixed, 2; npoints=21))
t_surface_cold, surf_cold = timecall(
    () -> parameter_surface_cold(fit_fixed, 1, 2; npoints=11)
)
t_surface_warm, surf_warm = timecall(() -> parameter_surface(fit_fixed, 1, 2; npoints=11))
surface_delta = maximum(abs.(surf_warm.z .- surf_cold.z))
surface_speedup = t_surface_cold / t_surface_warm

form_crossed, data_crossed = crossed_gaussian_fixture()
drm(form_crossed, Gaussian(); data=data_crossed)
GC.gc()
t_fit_crossed, fit_crossed = timecall(
    () -> drm(form_crossed, Gaussian(); data=data_crossed)
)
confint(fit_crossed; method=:profile)
confint(fit_crossed; method=:profile, threads=true)
profile_curve(fit_crossed, 2; npoints=21)
GC.gc()
t_profile_crossed, prof_crossed = timecall(() -> confint(fit_crossed; method=:profile))
t_profile_crossed_threaded, prof_crossed_threaded = timecall(
    () -> confint(fit_crossed; method=:profile, threads=true)
)
t_curve_crossed, curve_crossed = timecall(() -> profile_curve(fit_crossed, 2; npoints=21))
threaded_delta = maximum(
    max(
        abs(prof_crossed[i].lower - prof_crossed_threaded[i].lower),
        abs(prof_crossed[i].upper - prof_crossed_threaded[i].upper),
    ) for i in eachindex(prof_crossed)
)

form_pois_re, data_pois_re = poisson_re_fixture()
fit_pois_re = drm(form_pois_re, Poisson(); data=data_pois_re)
profile_result(fit_pois_re; parm=:resd)
profile_result(fit_pois_re; parm=:resd, threads=true)
bootstrap_result(
    form_pois_re, Poisson(); data=data_pois_re, B=2, rng=MersenneTwister(92),
    failures=:skip, check_converged=true
)
GC.gc()
t_profile_pois_re_serial, prof_pois_re_serial = timecall(
    () -> profile_result(fit_pois_re; parm=:resd)
)
t_profile_pois_re_threaded, prof_pois_re_threaded = timecall(
    () -> profile_result(fit_pois_re; parm=:resd, threads=true)
)
t_boot_pois_re_serial, boot_pois_re_serial = timecall(
    () -> bootstrap_result(
        form_pois_re,
        Poisson();
        data=data_pois_re,
        B=12,
        rng=MersenneTwister(93),
        failures=:skip,
        check_converged=true,
    ),
)
profile_pois_re_delta = maximum(
    max(
        abs(prof_pois_re_serial.ci[i].lower - prof_pois_re_threaded.ci[i].lower),
        abs(prof_pois_re_serial.ci[i].upper - prof_pois_re_threaded.ci[i].upper),
    ) for i in eachindex(prof_pois_re_serial.ci)
)
t_boot_pois_re_threaded, boot_pois_re_threaded = timecall(
    () -> bootstrap_result(
        form_pois_re,
        Poisson();
        data=data_pois_re,
        B=12,
        rng=MersenneTwister(93),
        threads=true,
        failures=:skip,
        check_converged=true,
    ),
)

mkpath(dirname(OUT))
open(OUT, "w") do io
    println(io, "# Quick inference profile")
    println(io)
    println(
        io,
        "CPU-aware run: Julia threads = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads()).",
    )
    println(io)
    println(io, "| task | fixture | n | params | elapsed/s |")
    println(io, "|:-----|:--------|--:|-------:|----------:|")
    @printf(
        io,
        "| fit | fixed Gaussian | %d | %d | %.4f |\n",
        length(data_fixed.y),
        length(coef(fit_fixed)),
        t_fit_fixed
    )
    @printf(
        io,
        "| Wald CI | fixed Gaussian | %d | %d | %.4f |\n",
        length(data_fixed.y),
        length(wald),
        t_wald
    )
    @printf(
        io,
        "| profile CI warm | fixed Gaussian | %d | %d | %.4f |\n",
        length(data_fixed.y),
        length(prof),
        t_profile
    )
    @printf(
        io,
        "| bootstrap CI B=20 serial | fixed Gaussian | %d | %d | %.4f |\n",
        length(data_fixed.y),
        length(boot20),
        t_boot20
    )
    @printf(
        io,
        "| bootstrap CI B=20 threaded | fixed Gaussian | %d | %d | %.4f |\n",
        length(data_fixed.y),
        length(boot20_threads),
        t_boot20_threads
    )
    @printf(
        io,
        "| bootstrap summary B=20 serial | fixed Gaussian | %d | %d | %.4f |\n",
        length(data_fixed.y),
        length(boot_summary20),
        t_boot_summary20
    )
    @printf(
        io,
        "| profile curve n=21 | fixed Gaussian | %d | %d | %.4f |\n",
        length(data_fixed.y),
        length(curve_fixed.x),
        t_curve_fixed
    )
    @printf(
        io,
        "| parameter surface n=11 cold | fixed Gaussian | %d | %d | %.4f |\n",
        length(data_fixed.y),
        length(surf_cold.z),
        t_surface_cold
    )
    @printf(
        io,
        "| parameter surface n=11 warm | fixed Gaussian | %d | %d | %.4f |\n",
        length(data_fixed.y),
        length(surf_warm.z),
        t_surface_warm
    )
    @printf(
        io,
        "| fit | crossed Gaussian | %d | %d | %.4f |\n",
        length(data_crossed.y),
        length(coef(fit_crossed)),
        t_fit_crossed
    )
    @printf(
        io,
        "| profile CI warm | crossed Gaussian | %d | %d | %.4f |\n",
        length(data_crossed.y),
        length(prof_crossed),
        t_profile_crossed
    )
    @printf(
        io,
        "| profile CI threaded warm | crossed Gaussian | %d | %d | %.4f |\n",
        length(data_crossed.y),
        length(prof_crossed_threaded),
        t_profile_crossed_threaded
    )
    @printf(
        io,
        "| profile curve n=21 | crossed Gaussian | %d | %d | %.4f |\n",
        length(data_crossed.y),
        length(curve_crossed.x),
        t_curve_crossed
    )
    @printf(
        io,
        "| profile result resd serial | Poisson (1|g) | %d | %d | %.4f |\n",
        length(data_pois_re.y),
        length(prof_pois_re_serial.ci),
        t_profile_pois_re_serial
    )
    @printf(
        io,
        "| profile result resd threaded endpoints | Poisson (1|g) | %d | %d | %.4f |\n",
        length(data_pois_re.y),
        length(prof_pois_re_threaded.ci),
        t_profile_pois_re_threaded
    )
    @printf(
        io,
        "| bootstrap result B=12 serial | Poisson (1|g) | %d | %d | %.4f |\n",
        length(data_pois_re.y),
        length(boot_pois_re_serial.summary),
        t_boot_pois_re_serial
    )
    @printf(
        io,
        "| bootstrap result B=12 threaded | Poisson (1|g) | %d | %d | %.4f |\n",
        length(data_pois_re.y),
        length(boot_pois_re_threaded.summary),
        t_boot_pois_re_threaded
    )
    println(io)
    println(io, "Interpretation guardrails:")
    println(
        io, "- This measures DRModels.jl local costs only; it is not an R-vs-Julia comparison."
    )
    println(
        io,
        "- Profile CI now warm-starts nuisance fits in the production `confint(..., method=:profile)` path.",
    )
    @printf(
        io,
        "- Threaded profile max endpoint delta versus serial warm profile CI: %.3e.\n",
        threaded_delta
    )
    @printf(
        io,
        "- Warm parameter-surface max deviance delta versus cold grid: %.3e; measured speedup %.2fx.\n",
        surface_delta,
        surface_speedup
    )
    println(
        io,
        "- Threaded bootstrap uses independent per-replicate RNG seeds; timings are only comparable at the explicit thread count above.",
    )
    @printf(
        io,
        "- Poisson RE profile endpoint-thread max CI delta versus serial: %.3e; serial endpoint evaluations %d, threaded worker threads %d.\n",
        profile_pois_re_delta,
        prof_pois_re_serial.stats[1].evaluations,
        prof_pois_re_threaded.worker_threads
    )
    @printf(
        io,
        "- Poisson RE bootstrap accounting: serial used %d/%d (failed %d), threaded used %d/%d (failed %d).\n",
        boot_pois_re_serial.used,
        boot_pois_re_serial.attempted,
        boot_pois_re_serial.failed,
        boot_pois_re_threaded.used,
        boot_pois_re_threaded.attempted,
        boot_pois_re_threaded.failed
    )
end

@printf(
    "fixed fit %.4fs, profile %.4fs, bootstrap B=20 %.4fs threaded %.4fs\n",
    t_fit_fixed,
    t_profile,
    t_boot20,
    t_boot20_threads
)
@printf(
    "crossed fit %.4fs, crossed profile %.4fs, threaded warm %.4fs, delta %.3e\n",
    t_fit_crossed,
    t_profile_crossed,
    t_profile_crossed_threaded,
    threaded_delta
)
@printf(
    "profile surface warm %.4fs vs cold %.4fs, speedup %.2fx, delta %.3e\n",
    t_surface_warm,
    t_surface_cold,
    surface_speedup,
    surface_delta
)
@printf(
    "poisson RE bootstrap B=12 serial %.4fs threaded %.4fs, used %d/%d vs %d/%d\n",
    t_boot_pois_re_serial,
    t_boot_pois_re_threaded,
    boot_pois_re_serial.used,
    boot_pois_re_serial.attempted,
    boot_pois_re_threaded.used,
    boot_pois_re_threaded.attempted
)
@printf(
    "poisson RE profile resd serial %.4fs threaded endpoints %.4fs, delta %.3e\n",
    t_profile_pois_re_serial,
    t_profile_pois_re_threaded,
    profile_pois_re_delta
)
println("wrote ", OUT)
