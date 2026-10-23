# Profile-likelihood confidence intervals for the q=4 phylogenetic among-axis SDs.
#
# The parametric bootstrap (bootstrap_sigma_a) undercovers the SCALE-axis SDs at a
# boundary (~0.52 at nominal 0.90 — a known boundary-bootstrap effect). The profile
# likelihood is the principled fix: it needs NO Hessian (so it sidesteps the
# pdHess=FALSE that defeats Wald), it respects the SD ≥ 0 boundary, and it has good
# coverage for variance components.
#
# Construction: the Fisher-z parameterization writes Σ_a = D·R·D with D=diag(exp.(φ[1:4])),
# so the SD of axis a IS exp(φ[a]). To profile sd_a = s we FIX φ[a]=log(s) and
# re-optimise the other 9 φ + the β block (a constrained marginal-likelihood fit — no
# Hessian). The profile deviance is the (non-negative) LR statistic
#   D(s) = 2(ℓ̂ − ℓ_p(s)) = 2(nll_p(s) − nll_hat) ≥ 0,
# and the (1−α) CI is {s : D(s) ≤ thr}, thr from the χ²₁ reference. At the boundary the
# lower endpoint is 0 (the deviance floor stays below thr down to s≈0) — the honest "no
# detectable scale-phylo signal" interval.
#
# Numerical care (panel review): the inner Laplace mode is warm-started but the OUTER
# ψ-optimum is re-checked for convergence (with a cold restart on failure) so warm-start
# continuation cannot mode-lock and bias endpoints inward; the root-finder refuses to
# bisect a non-straddling bracket (returns an unbounded endpoint instead of fabricating
# one); and nll_hat is computed on the SAME code path as the profile (a full
# unconstrained re-optimisation), so D(sd̂)≈0 by construction.

"""
    profile_sigma_a(fit; level=0.95, axes=:all, chibar=false, n_newton=nothing,
                    g_tol=1e-3, max_bisect=14) -> NamedTuple

Profile-likelihood CIs for the among-axis SDs `sqrt.(diag(Σ_a))` of a bivariate q=4
phylogenetic location–scale fit. Returns a `NamedTuple` with `summary` rows
`(param, coef, estimate, lower, upper, deviance_floor, bounded)` for
`param ∈ (:sd_mu1, :sd_mu2, :sd_sigma1, :sd_sigma2)`. `lower=0` ⇒ the axis is at /
near the boundary; `upper=Inf` with `bounded=false` ⇒ no upper crossing was found
(the profile is too flat — report it honestly, do not invent a bound). Also returns
`level`, `chibar`, `threshold`, `nll_hat`, `axes`.

Threshold: `chibar=false` (default) uses the standard profile-LR cutoff `χ²₁(level)`.
For an **identified (interior)** SD this gives ~nominal coverage; for a **collapsed
(boundary)** axis it is **conservative** (over-covers — the LR statistic is then
≈0.5·χ²₀+0.5·χ²₁, so the exclusion rate is ≈0.5·(1−level), e.g. a 90% naive CI covers
~95% on the boundary) and the lower bound is 0. This conservatism is the safe failure
mode for an axis of unknown status, so naive is the publication default. `chibar=true`
uses the boundary mixture cutoff `χ²₁(2·level−1)` — nominal **only** when the axis is
known to be at the boundary AND the other variance/correlation nuisances are interior
(a single boundary component); with several axes near their boundaries the 0.5:0.5
mixture is itself approximate. Needs no Hessian; the fit must carry `fit.ranef.prob`.
The calibrated complement to [`bootstrap_sigma_a`].
"""
function profile_sigma_a(fit::DrmFit; level::Real = 0.95, axes = :all, chibar::Bool = false,
                         n_newton::Union{Nothing,Int} = nothing, g_tol::Real = 1e-3,
                         max_bisect::Int = 14)
    level = _validate_ci_level(level; what="profile_sigma_a")
    re = fit.ranef
    (re isa NamedTuple && haskey(re, :prob) && haskey(re, :Sigma_a) && haskey(re, :Q_cond)) ||
        throw(ArgumentError("profile_sigma_a requires a bivariate q=4 phylogenetic fit whose " *
            "ranef carries `prob` (re-fit on this DRModels version)"))
    is_converged(fit) ||
        @warn "profile_sigma_a: the supplied fit did not converge; profile CIs may be unreliable"
    prob = re.prob
    Q_cond = re.Q_cond
    Σa = Matrix{Float64}(re.Sigma_a)
    nn = n_newton === nothing ? Int(get(re, :n_newton, 40)) : n_newton
    g_tol = Float64(g_tol)
    nβ = theta_len(prob) - 10
    φ̂0 = fz_init_from_Sigma(Σa)
    ψ̂0 = vcat(Vector{Float64}(fit.theta[1:nβ]), Vector{Float64}(φ̂0))

    # nll_hat on the SAME path as the profile: one unconstrained re-optimisation from the
    # fitted point (it should barely move — the fit is already the MLE — but this puts the
    # deviance reference on exactly the scale dev() uses, so D(sd̂) ≈ 0).
    ψ̂, nll_hat = _q4_full_refit(prob, Q_cond, ψ̂0, nn, g_tol)
    sd_hat = sqrt.(max.(diag(Σa), 0.0))         # = vc(fit); the reported point estimate
    # keep nll_hat ≤ the fit's own nll so dev(sd̂) ≥ 0 (the refit only refines convergence)
    nll_hat = min(nll_hat, fz_marginal_nll(prob, Q_cond, ψ̂0; n_newton = nn))

    thr = chibar ? quantile(Chisq(1), max(2 * level - 1, 1e-6)) : quantile(Chisq(1), level)
    axisnames = (:sd_mu1, :sd_mu2, :sd_sigma1, :sd_sigma2)
    want = axes === :all ? (1:4) : collect(axes)

    RowT = NamedTuple{(:param, :coef, :estimate, :lower, :upper, :deviance_floor, :bounded),
                      Tuple{Symbol,String,Float64,Float64,Float64,Float64,Bool}}
    rows = RowT[]
    for a in want
        lo, hi, dfloor, bnd = _q4_profile_axis(prob, Q_cond, ψ̂, nβ, a, nll_hat, thr,
                                               sd_hat[a], nn, g_tol, max_bisect)
        lo = min(lo, sd_hat[a]); hi = max(hi, sd_hat[a])     # containment guard
        push!(rows, (param = axisnames[a], coef = String(axisnames[a]), estimate = sd_hat[a],
                     lower = lo, upper = hi, deviance_floor = dfloor, bounded = bnd))
    end
    return (summary = rows, level = level, chibar = chibar, threshold = thr,
            nll_hat = nll_hat, axes = axisnames)
end

# Unconstrained re-optimisation of the full Fisher-z ψ from a start, returning (ψ̂, nll̂).
# Shares the exact fg! contract used inside the profile so the deviance reference matches.
function _q4_full_refit(prob, Q_cond, ψ0::Vector{Float64}, nn::Int, g_tol::Float64)
    u_cache = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    fg! = function (F, G, ψ)
        nll, g = _q4_nll_grad(prob, Q_cond, ψ, nn, u_cache)
        G !== nothing && copyto!(G, g)
        return nll
    end
    res = Optim.optimize(Optim.only_fg!(fg!), copy(ψ0),
        Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
        Optim.Options(g_tol = g_tol, iterations = 250, time_limit = 30.0))
    return Optim.minimizer(res), _objective_at_minimizer_fg(fg!, res)
end

# Shared marginal NLL + FULL ψ-gradient with the Inf robustness barrier. Returns
# (nll, g_full); callers fill Optim's G (sliced to whatever block they optimise).
# Caches the inner Laplace mode (warm start). On a caught barrier error returns
# (Inf, zeros) so a line search can't read a stale gradient.
function _q4_nll_grad(prob, Q_cond, ψ::AbstractVector, nn::Int,
                      u_cache::Ref{Union{Nothing,Vector{Float64}}})
    try
        nll, g, û, _ = fz_marginal_and_grad(prob, Q_cond, Vector{Float64}(ψ);
                                            u0 = u_cache[], n_newton = nn)
        any(!isfinite, g) && return (Inf, zeros(length(ψ)))
        u_cache[] = û
        return (nll, g)
    catch e
        (e isa DomainError || e isa LinearAlgebra.PosDefException ||
         e isa LinearAlgebra.SingularException || e isa ArgumentError) || rethrow(e)
        return (Inf, zeros(length(ψ)))
    end
end

# Profile one axis. Returns (lower, upper, deviance_floor, bounded). `bounded=false`
# ⇒ no upper crossing within the search budget (upper=Inf), reported honestly.
function _q4_profile_axis(prob, Q_cond, ψ̂::Vector{Float64}, nβ::Int, a::Int,
                          nll_hat::Float64, thr::Float64, sd_a::Float64,
                          nn::Int, g_tol::Float64, max_bisect::Int)
    fix = nβ + a
    free = setdiff(1:length(ψ̂), [fix])
    ψ_free0 = ψ̂[free]                            # the global-MLE free block (cold reference)
    u_cache = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    ψ_warm = Ref(copy(ψ_free0))

    # constrained profile NLL at sd=s, with a convergence gate + cold restart so warm-start
    # continuation cannot mode-lock to a sub-optimal (NLL-too-high) branch and bias the
    # interval inward.
    function dev(s)
        ψw = copy(ψ̂); ψw[fix] = log(max(s, 1e-8))
        best = Inf; bestψ = ψ_warm[]
        for (start, warmu) in ((ψ_warm[], true), (ψ_free0, false))
            uc = Ref{Union{Nothing,Vector{Float64}}}(warmu ? u_cache[] : nothing)
            fg! = function (F, G, ψf)
                ψw[free] .= ψf
                nll, g = _q4_nll_grad(prob, Q_cond, ψw, nn, uc)
                G !== nothing && copyto!(G, @view g[free])   # slice full grad → free block
                return nll
            end
            res = Optim.optimize(Optim.only_fg!(fg!), copy(start),
                Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
                Optim.Options(g_tol = g_tol, iterations = 150, time_limit = 15.0))
            m = _objective_at_minimizer_fg(fg!, res)
            if isfinite(m) && m < best
                best = m; bestψ = Optim.minimizer(res); u_cache[] = uc[]
            end
            # only pay for the cold restart when the warm solve didn't converge cleanly
            (warmu && Optim.converged(res) && isfinite(m)) && break
        end
        ψ_warm[] = bestψ
        return 2 * (best - nll_hat)
    end

    # lower endpoint: deviance floor near s≈0. If it stays below thr, the bound IS 0.
    sfloor = max(sd_a * 1e-3, 1e-4)
    dfloor = dev(sfloor)
    lower = (isfinite(dfloor) && dfloor < thr) ? 0.0 :
            _bisect_dev(dev, sfloor, dfloor, sd_a, thr, max_bisect)

    # upper endpoint: expand until the deviance exceeds thr, then bisect a STRADDLING
    # bracket. If the budget runs out without a crossing, report upper=Inf (unbounded) —
    # never bisect a non-straddling bracket.
    ψ_warm[] = copy(ψ_free0)                     # reset warm start before the upper scan
    u_cache[] = nothing
    s_hi = sd_a * 1.6 + 0.05
    d_hi = dev(s_hi); its = 0
    while isfinite(d_hi) && d_hi < thr && its < max_bisect
        s_hi *= 1.6; d_hi = dev(s_hi); its += 1
    end
    if isfinite(d_hi) && d_hi >= thr
        upper = _bisect_dev(dev, sd_a, 0.0, s_hi, thr, max_bisect)   # dev(sd_a)≈0 < thr
        bounded = true
    else
        upper = Inf; bounded = false
    end
    return lower, upper, dfloor, bounded
end

# Bisection for dev(s)=thr on [a,b] whose endpoint deviances (da,db) STRADDLE thr. Refuses
# to proceed (returns the inner endpoint) if they do not straddle — no fabricated root.
function _bisect_dev(dev, a::Float64, da::Float64, b::Float64, thr::Float64, iters::Int)
    db = dev(b)
    fa = da - thr; fb = db - thr
    sign(fa) == sign(fb) && return (abs(fa) <= abs(fb) ? a : b)   # no straddle → inner end
    for _ in 1:iters
        (b - a) < 1e-3 && break
        m = 0.5 * (a + b)
        fm = dev(m) - thr
        abs(fm) < 1e-2 && return m
        if sign(fm) == sign(fa)
            a = m; fa = fm
        else
            b = m
        end
    end
    return 0.5 * (a + b)
end
