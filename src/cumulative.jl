# cumulative.jl — Cumulative-logit ordinal regression. Ordered categorical
# response y ∈ {1,…,K}; Pr(y ≤ k) = logistic(θ_k − η) with ordered cutpoints
# θ_1 < … < θ_{K-1} and a single linear predictor η. The location intercept is
# dropped (a free intercept and free cutpoints are not jointly identifiable), so
# the `mu` formula contributes slopes only. No `sigma`. The cutpoints are kept
# ordered by the increment parameterisation θ_1 = δ_1, θ_k = θ_{k-1} + exp(δ_k).
# Mirrors drmTMB's `cumulative_logit`. Fixed effects, ML, plus an ordinary
# random intercept `(1 | g)` or independent random slope `(0 + x | g)` on `mu`
# (#563 S8) via per-group adaptive Gauss–Hermite quadrature (src/adaptive_ghq.jl),
# plus an unlabelled, intercept-only `phylo(1 | species)`
# structured `mu` effect (#563 S8 follow-on) via the sparse augmented-state
# Laplace GLMM route (src/sparse_laplace_glmm.jl), matching drmTMB 0.7.0's own
# `validate_ordinal_phylo_mu_structured_term()`. Crossed/multiple random
# intercepts, `(1 | g) + (1 | h)` (#738), reuse the PLAIN (no-nuisance)
# sparse-Laplace crossed engine (`_fit_crossed_mean_laplace`) that
# Poisson/Binomial's crossed routes already share; the `nc = K - 1` ordered
# cutpoints (which that engine's θ layout has no slot for) are profiled out
# by an outer Nelder–Mead search, refitting the (β, σ_g, σ_h) crossed model
# via the existing engine at each candidate cutpoint vector. Correlated
# slopes `(1 + x | g)` and `relmat`/`animal`/`spatial` structured markers are
# refused, matching drmTMB 0.7.0's own
# `validate_cumulative_logit_mu_random_terms()` /
# `validate_ordinal_phylo_mu_structured_term()` scope.

"""
    CumulativeLogit()

Cumulative-logit (proportional-odds) ordinal family. The response is an ordered
category coded `1, 2, …, K`. `Pr(y ≤ k) = logistic(θ_k − η)` with ordered
cutpoints `θ_1 < … < θ_{K-1}`; the single linear predictor `η` comes from the
`mu` formula **with its intercept dropped** (cutpoints absorb it). No scale
parameter. `coef(fit, :mu)` are the slopes; `coef(fit, :cutpoints)` are the raw
increment parameters (`θ_1 = δ_1`, `θ_k = θ_{k-1} + exp(δ_k)`). `fitted` returns
the expected ordered-category score `Σ_k k·Pr(y=k)`. Mirrors `drmTMB`'s
`cumulative_logit`. An ordinary random intercept `(1 | g)` or an independent
random slope `(0 + x | g)` on `mu` integrates the group-level term out by
per-group adaptive Gauss–Hermite quadrature (41 nodes centred on each group's
conditional mode); `coef(fit, :resd)` is the log random-effect
SD. An unlabelled, intercept-only phylogenetic random intercept
`phylo(1 | species)` on `mu` is fit via the sparse-Laplace GLMM route instead
(needs `tree = …`); it cannot be combined with an ordinary random effect. For
the `phylo` route, `re_sd(fit)[:group]` is on the RAW branch-length scale (tip
variance equals the tree height `h`, not 1) — this differs from drmTMB's
`sdpars\$mu[["phylo(1 | species)"]]`, which is on the CORRELATION scale
(`ape::vcv(tree, corr = TRUE)`, tip variance 1 regardless of `h`). Convert with
`re_sd(fit)[:group] * sqrt(phylo_tree_height(augmented_phy(tree)))` to compare
against drmTMB's number (the same convention used throughout the Gaussian
phylo-mean route, e.g. `test_parity_gaussian_phylo_mean.jl`). Crossed/multiple
random intercepts, `(1 | g) + (1 | h)`, are also supported, via the same
sparse-Laplace engine Poisson/Binomial's crossed routes reuse; the cutpoints
are found by an outer profile search since that engine has no cutpoint slot.
Correlated slopes `(1 + x | g)` and `relmat`/`animal`/`spatial` structured
markers are not implemented.

```julia
fit = drm(bf(y ~ x), CumulativeLogit(); data = dat)          # y coded 1..K
fit_re = drm(bf(y ~ x + (1 | g)), CumulativeLogit(); data = dat)
fit_slope = drm(bf(y ~ x + (0 + x | g)), CumulativeLogit(); data = dat)
fit_phylo = drm(bf(y ~ x + phylo(1 | species)), CumulativeLogit(); data = dat, tree = tr)
fit_crossed = drm(bf(y ~ x + (1 | g) + (1 | h)), CumulativeLogit(); data = dat)
re_sd(fit_crossed)                                            # Dict(:g => ..., :h => ...)
```
"""
struct CumulativeLogit end

# Ordered cutpoints from the free increment parameters: θ_1 = δ_1,
# θ_k = θ_{k-1} + exp(δ_k) for k ≥ 2. Shared by the fixed-effect and
# random-effect fitters below.
function _cumulative_cuts(δ)
    nc = length(δ)
    cuts = similar(δ)
    cuts[1] = δ[1]
    for k in 2:nc
        cuts[k] = cuts[k-1] + exp(δ[k])
    end
    return cuts
end

# Per-observation cumulative-logit log-likelihood log P(y=k | η, cuts).
@inline function _cumulative_loglik(k::Int, η, cuts, K::Int, nc::Int)
    if k == 1
        return _log_logistic(cuts[1] - η)                    # P(y=1) = F(θ_1−η)
    elseif k == K
        return _log_logistic(η - cuts[nc])                   # P(y=K) = 1−F(θ_{K-1}−η)
    else
        # P(y=k) = F(a) - F(b), a = cuts[k]-η > b = cuts[k-1]-η. Write
        # F(a)-F(b) = F(a)·(1 - F(b)/F(a)) = F(a)·(1 - exp(logF(b)-logF(a))),
        # so log P(y=k) = logF(a) + log1mexp(logF(b)-logF(a)) — only ever
        # combines two already-stable log-probabilities (`_log_logistic`),
        # never subtracts raw probabilities, and its `log1mexp` argument is
        # ≤ 0 by construction (logF is monotone, b < a), so it never throws
        # or NaNs, only saturating to the mathematically-correct -Inf when
        # a and b coincide (or round to bit-identical floats).
        la = _log_logistic(cuts[k] - η)
        lb = _log_logistic(cuts[k-1] - η)
        # `lb - la` is ≤ 0 analytically (monotone `_log_logistic`, `cuts[k-1] <
        # cuts[k]`), but two independently-rounded evaluations at very close
        # arguments can round to a tiny POSITIVE float; clamp before
        # `_log1mexp` so that never throws/NaNs (only ever saturates to the
        # correct -Inf when `la`/`lb` coincide).
        return la + _log1mexp(min(lb - la, zero(lb - la)))
    end
end

# Expected ordered-category score Σ_k k·P(y=k|η,cuts), the `fitted()` value.
function _cumulative_score(η̂, cuts_hat, K)
    n = length(η̂); nc = K - 1
    score = Vector{Float64}(undef, n)
    for i in 1:n
        sc = 0.0
        for k in 1:K
            Pk = k == 1 ? _logistic(cuts_hat[1] - η̂[i]) :
                 k == K ? 1 - _logistic(cuts_hat[nc] - η̂[i]) :
                 _logistic(cuts_hat[k] - η̂[i]) - _logistic(cuts_hat[k-1] - η̂[i])
            sc += k * Pk
        end
        score[i] = sc
    end
    return score
end

# Initial cutpoint increments from empirical cumulative category proportions.
function _cumulative_cut_init(y, K, n)
    nc = K - 1
    cnt = [count(==(k), y) for k in 1:K]
    cum = clamp.(cumsum(cnt)[1:nc] ./ n, 1e-3, 1 - 1e-3)
    θc0 = log.(cum ./ (1 .- cum))
    δ0 = zeros(nc)
    δ0[1] = θc0[1]
    for k in 2:nc
        δ0[k] = log(max(θc0[k] - θc0[k-1], 1e-2))
    end
    return δ0
end

function drm(f::DrmFormula, fam::CumulativeLogit; data, tree = nothing, g_tol::Real = 1e-8,
             se::Bool = true)
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, tree = tree, g_tol = g_tol, se = se)
    end
    missing_fit !== nothing && return missing_fit

    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing ||
        error("CumulativeLogit() does not support meta_V markers")
    if st !== nothing
        kind, grp = st
        kind === :phylo ||
            error("CumulativeLogit() structured `mu` random effects support only an " *
                  "unlabelled, intercept-only phylo(1 | group) term (matches drmTMB " *
                  "0.7.0's validate_ordinal_phylo_mu_structured_term()); relmat/animal/" *
                  "spatial are not implemented in this slice")
        tree === nothing && error("phylo(1 | $grp) needs `tree = …`")
        isempty(re) ||
            error("CumulativeLogit() phylo(1 | $grp) cannot be combined with an ordinary " *
                  "random effect on `mu` yet")
    end
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    ic = findfirst(==("(Intercept)"), nmμ)               # drop the location intercept
    if ic !== nothing
        keep = setdiff(1:length(nmμ), ic)
        Xμ = Xμ[:, keep]; nmμ = nmμ[keep]
    end
    (all(yi -> yi >= 1 && isinteger(yi), y) && maximum(y) >= 2) ||
        error("CumulativeLogit() requires ordered integer categories coded 1, 2, …, K (K ≥ 2)")
    K = round(Int, maximum(y))
    yi = round.(Int, y)
    if st !== nothing                  # phylo(1 | g) random intercept → sparse Laplace (#563 S8 follow-on)
        _, grp = st
        labels = getproperty(data, grp)
        return _withformula(
            _fit_cumulative_phylo_laplace(fam, yi, Xμ, K, labels, tree, nmμ, grp, g_tol; se = se), f)
    end
    if !isempty(re)                    # random intercept/slope on mu → GHQ marginal (#563 S8)
        if length(re) > 1               # crossed/multiple intercepts (#738)
            all(_re_kind(r[1])[1] === :intercept for r in re) ||
                error("CumulativeLogit() supports multiple random effects only as " *
                      "crossed/nested intercepts, e.g. `(1 | g) + (1 | h)`")
            comps = map(re) do r
                grp = r[2]; gidx, G = _group_index(getproperty(data, grp))
                (ones(length(yi)), gidx, G, String(grp))
            end
            return _withformula(_fit_cumulative_crossed_laplace(fam, yi, Xμ, K, comps, nmμ, g_tol), f)
        end
        length(re) == 1 ||
            error("CumulativeLogit() supports only a single `(1 | g)` random intercept or " *
                  "`(0 + x | g)` random slope on `mu`; crossed/multiple random effects are " *
                  "not implemented in this slice")
        (rk, var) = _re_kind(re[1][1]); grp = re[1][2]
        gidx, G = _group_index(getproperty(data, grp))
        if rk === :intercept
            return _withformula(_fit_cumulative_ranef(fam, yi, Xμ, K, gidx, G, nmμ, grp, g_tol), f)
        elseif rk === :slope
            xs = Float64.(getproperty(data, var))
            return _withformula(
                _fit_cumulative_slope_ranef(fam, yi, Xμ, K, xs, gidx, G, nmμ, grp, g_tol), f)
        else
            error("CumulativeLogit() supports only an ordinary `(1 | g)` random intercept " *
                  "or an independent `(0 + x | g)` random slope on `mu`; correlated random " *
                  "slopes `(1 + x | g)` are not implemented (matches drmTMB 0.7.0: " *
                  "\"Only independent cumulative_logit() mu random intercepts and slopes " *
                  "are implemented in this slice\")")
        end
    end
    return _withformula(_fit_cumulative(fam, yi, Xμ, K, nmμ, g_tol), f)
end

function _fit_cumulative(fam::CumulativeLogit, y::Vector{Int}, Xμ, K, nmμ, g_tol)
    n = length(y); pμ = size(Xμ, 2); nc = K - 1
    function nll(θ)
        β = θ[1:pμ]; δ = θ[pμ+1:pμ+nc]
        cuts = _cumulative_cuts(δ)
        η = pμ == 0 ? zeros(eltype(θ), n) : Xμ * β
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            s -= _cumulative_loglik(y[i], η[i], cuts, K, nc)
        end
        return s
    end
    θ0 = zeros(pμ + nc)
    θ0[(pμ+1):(pμ+nc)] = _cumulative_cut_init(y, K, n)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :cutpoints => (pμ+1):(pμ+nc)]
    names = [:mu => nmμ, :cutpoints => ["theta$k" for k in 1:nc]]
    # fitted = expected category score Σ_k k·P(y=k)
    β̂ = θ̂[1:pμ]; δ̂ = θ̂[pμ+1:pμ+nc]
    # `cuts_hat`, NOT `cuts` (#549 class): the nll closure above also assigns
    # `cuts`, and sharing the name here would make it one boxed variable shared
    # with a closure that threaded profile CIs call concurrently.
    cuts_hat = _cumulative_cuts(δ̂)
    η̂ = pμ == 0 ? zeros(n) : Xμ * β̂
    score = _cumulative_score(η̂, cuts_hat, K)
    means = Dict(:mu => score); obs = Dict(:mu => Float64.(y))
    scales = Dict(:ordinal_eta => η̂, :ordinal_cuts => Float64.(cuts_hat))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Default AGHQ nodes for the CumulativeLogit `(1 | g)` / `(0 + x | g)` routes. The
# ordinal group posterior is far from Gaussian when σ_b is large (a group with all
# observations in an end category has a flat, logistic-tailed posterior), so these
# routes need more nodes than the `_RANEF1D_AGHQ_K = 5` used by the other families.
# Swept against an exact per-group QuadGK integral on the seed-24 fuzzer dataset
# (test/test_cumlogit_aghq.jl; n = 48, G = 6, σ_b up to 13.8), worst |error| over
# three θ: K=5 1.4e-2, K=9 4.7e-4, K=15 2.0e-5, K=21 2.6e-6, K=31 4.7e-7,
# K=41 2.9e-9 nat. K = 41 is the first with a >100× margin on a 1e-6 check; the
# per-group cost stays close to the old 32-node prior-scale grid it replaces.
const _CUMLOGIT_AGHQ_K = 41

# Cumulative-logit ordinal GLMM with a random intercept (1|g) on the latent
# linear predictor η. b_g ~ N(0,σ_b²) integrated out per group by ADAPTIVE
# Gauss–Hermite quadrature (`_aghq_marginal_loglik`, src/adaptive_ghq.jl: nodes
# b̂_g + √2 C z at each group's conditional mode), `nq` nodes (`nq = 1` is
# Laplace). The old fixed 32-node prior-scale grid (b = √2 σ_b z) was badly
# under-resolved for large σ_b: on the seed-24 fuzzer dataset it sat 1.36 nat
# below the true marginal at its own optimum. Cutpoints stay ordinary fixed
# effects (shared across groups). #563 S8.
function _fit_cumulative_ranef(fam::CumulativeLogit, y::Vector{Int}, Xμ, K, gidx, G, nmμ, grp, g_tol;
                               nq::Int = _CUMLOGIT_AGHQ_K)
    n = length(y); pμ = size(Xμ, 2); nc = K - 1
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, nq); Zre = ones(n, 1); bcache = zeros(1, G)
    function nll(θ)
        β = θ[1:pμ]; δ = θ[pμ+1:pμ+nc]; σb = exp(θ[pμ+nc+1])
        cuts = _cumulative_cuts(δ)
        η0 = pμ == 0 ? zeros(eltype(θ), n) : Xμ * β
        ll = (i, η) -> _cumulative_loglik(y[i], η, cuts, K, nc)
        return -_aghq_marginal_loglik(ll, members, η0, Zre, reshape([σb], 1, 1), rule, bcache)
    end
    θ0 = zeros(pμ + nc + 1)
    θ0[(pμ+1):(pμ+nc)] = _cumulative_cut_init(y, K, n)
    θ0[pμ+nc+1] = log(0.5)             # σ_b
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :cutpoints => (pμ+1):(pμ+nc), :resd => (pμ+nc+1):(pμ+nc+1)]
    names = [:mu => nmμ, :cutpoints => ["theta$k" for k in 1:nc], :resd => [String(grp)]]
    β̂ = θ̂[1:pμ]; cuts_hat = _cumulative_cuts(θ̂[pμ+1:pμ+nc])
    η̂ = pμ == 0 ? zeros(n) : Xμ * β̂     # population (b=0) linear predictor
    score = _cumulative_score(η̂, cuts_hat, K)
    means = Dict(:mu => score); obs = Dict(:mu => Float64.(y))
    scales = Dict(:ordinal_eta => η̂, :ordinal_cuts => Float64.(cuts_hat))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Cumulative-logit ordinal GLMM with an INDEPENDENT random slope (0+x|g) on
# the latent linear predictor η. b_g ~ N(0,σ_b²) integrated out per group by
# ADAPTIVE Gauss–Hermite quadrature (as `_fit_cumulative_ranef` above); the group
# term enters as b_g·x_i rather than b_g (random-effect design column `xs`). #563 S8.
function _fit_cumulative_slope_ranef(fam::CumulativeLogit, y::Vector{Int}, Xμ, K, xs, gidx, G, nmμ, grp, g_tol;
                                     nq::Int = _CUMLOGIT_AGHQ_K)
    n = length(y); pμ = size(Xμ, 2); nc = K - 1
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, nq); Zre = reshape(Float64.(xs), n, 1); bcache = zeros(1, G)
    function nll(θ)
        β = θ[1:pμ]; δ = θ[pμ+1:pμ+nc]; σb = exp(θ[pμ+nc+1])
        cuts = _cumulative_cuts(δ)
        η0 = pμ == 0 ? zeros(eltype(θ), n) : Xμ * β
        ll = (i, η) -> _cumulative_loglik(y[i], η, cuts, K, nc)
        return -_aghq_marginal_loglik(ll, members, η0, Zre, reshape([σb], 1, 1), rule, bcache)
    end
    θ0 = zeros(pμ + nc + 1)
    θ0[(pμ+1):(pμ+nc)] = _cumulative_cut_init(y, K, n)
    θ0[pμ+nc+1] = log(0.5)             # σ_b
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :cutpoints => (pμ+1):(pμ+nc), :resd => (pμ+nc+1):(pμ+nc+1)]
    names = [:mu => nmμ, :cutpoints => ["theta$k" for k in 1:nc], :resd => [String(grp)]]
    β̂ = θ̂[1:pμ]; cuts_hat = _cumulative_cuts(θ̂[pμ+1:pμ+nc])
    η̂ = pμ == 0 ? zeros(n) : Xμ * β̂     # population (b=0) linear predictor
    score = _cumulative_score(η̂, cuts_hat, K)
    means = Dict(:mu => score); obs = Dict(:mu => Float64.(y))
    scales = Dict(:ordinal_eta => η̂, :ordinal_cuts => Float64.(cuts_hat))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# ---- phylogenetic random intercept on the mean, `phylo(1 | g)` (#563 S8 follow-on) ----
#
# drmTMB 0.7.0 admits an unlabelled, intercept-only `phylo(1 | species)` term on
# `cumulative_logit()`'s `mu` (`validate_ordinal_phylo_mu_structured_term()`,
# R/drmTMB.R:10500). This route reuses the sparse augmented-state Laplace GLMM
# spine already proven for Poisson/Gamma/Binomial/Beta `phylo(1 | g)`
# (src/sparse_laplace_glmm.jl): the random-effect mode is found by the SAME
# generic `_phylo_mean_mode` Newton solver (dispatched on `Val(:cumlogit)` via
# `_laplace_value`/`_laplace_d12` below), and the exact O(p) implicit-log-det
# outer gradient generalises `_phylo_mean_laplace_hetero_fg`'s scalar/Xσ-chained
# nuisance-parameter pattern (#164) from a scalar dispersion to the ordinal
# cutpoints — the one genuinely new piece this family needs. Cutpoints are NOT
# a per-observation linear predictor (unlike `Xσ·βσ`): each observation's
# likelihood touches at most two of the `nc = K-1` cutpoints directly
# (`cuts[y-1]`, `cuts[y]`), so their partials are accumulated straight into a
# length-`nc` vector per group rather than chained through a design matrix,
# and the resulting ∂/∂cuts gradient is mapped to ∂/∂δ (the ordered increment
# parameterisation) by the reparameterisation's own Jacobian at the end.

# Sigmoid and its first three derivatives (shared by all three cumulative-logit
# categories below via the "phantom boundary" trick: k=1 has no lower
# cutpoint, k=K has no upper one — represented by g=0/g=1 with all derivatives
# 0, so ONE formula covers k=1, interior k, and k=K; verified the interior
# formula's boundary limit reduces exactly to the direct k=1/k=K derivatives).
@inline function _sig_derivs(x)
    g = _logistic(x)
    gp = g * (1 - g)
    gpp = gp * (1 - 2g)
    gppp = gpp * (1 - 2g) - 2 * gp^2
    return g, gp, gpp, gppp
end

# Per-observation cumulative-logit NEGATIVE-loglik kernel for category k=y[i]:
# value + derivatives in η (order 1-3, for mode-finding and the IFT log-det
# sensitivity) AND in the (at most two) active cutpoints cuts[k-1]/cuts[k]
# (direct partial `nval`, the η-cross partial `nr` needed for the IFT implicit
# correction, and the log-det-sensitivity partial `nw`). `has_lo`/`has_hi`
# flag which of the two are active (k=1: hi only; k=K: lo only; interior:
# both). Derived by hand from D = σ(cuts[k]-η) - σ(cuts[k-1]-η),
# value = log D (cross-checked against the direct k=1/k=K single-cutpoint
# forms in the phantom limit, and against finite differences by
# test/test_cumlogit_phylo.jl's gradient-sanity test).
function _cumulative_phylo_kernel(k::Int, η, cuts, K::Int, nc::Int)
    has_lo = k >= 2
    has_hi = k <= nc
    a = has_hi ? cuts[k] - η : zero(η)
    b = has_lo ? cuts[k-1] - η : zero(η)
    ga, gpa, gppa, gpppa = has_hi ? _sig_derivs(a) : (one(η), zero(η), zero(η), zero(η))
    gb, gpb, gppb, gpppb = has_lo ? _sig_derivs(b) : (zero(η), zero(η), zero(η), zero(η))
    # D = P(y=k). The general `ga - gb` cancels for the TOP category (has_hi is
    # false, so ga is the phantom 1 and D = 1 - σ(b)): use the logistic identity
    # σ(-b) = 1 - σ(b) instead, computed exactly rather than by subtraction from
    # 1 (Opus review D-2 — matches the fixed-effect route's `_cumulative_loglik`
    # and drmTMB's `drm_log1m_inv_logit`, both exact to |η-c| ≈ 700 instead of
    # cancelling from |η-c| ≈ 20). The k=1 branch (`!has_lo`) already avoids
    # cancellation since D = ga there; interior categories keep the plain
    # subtraction (not the reported issue).
    D = has_hi ? (has_lo ? ga - gb : ga) : _logistic(-b)
    v = -log(D)                                     # NLL contribution

    N1 = gpb - gpa                                   # D'  (w.r.t. η)
    d1 = -N1 / D
    N1p = gppa - gppb                                # N1' (w.r.t. η)
    N2 = N1p * D - N1^2                               # D²·(∂²logD/∂η²)
    d2 = -N2 / D^2
    N1pp = gpppb - gpppa                              # N1''(w.r.t. η)
    N2p = N1pp * D - N1 * N1p
    d3 = -(N2p * D - 2 * N2 * N1) / D^3

    nval_hi = has_hi ? -gpa / D : zero(η)
    nval_lo = has_lo ? gpb / D : zero(η)
    nr_hi = has_hi ? (gppa * D + gpa * N1) / D^2 : zero(η)
    nr_lo = has_lo ? -(gppb * D + gpb * N1) / D^2 : zero(η)
    dN2_hi = gpppa * D + N1p * gpa + 2 * N1 * gppa
    dN2_lo = -gpppb * D - N1p * gpb - 2 * N1 * gppb
    nw_hi = has_hi ? -(dN2_hi * D - 2 * N2 * gpa) / D^3 : zero(η)
    nw_lo = has_lo ? -(dN2_lo * D + 2 * N2 * gpb) / D^3 : zero(η)

    return v, d1, d2, d3, has_lo, has_hi, nval_lo, nval_hi, nr_lo, nr_hi, nw_lo, nw_hi
end

# `Val(:cumlogit)` dispatch for the generic `_phylo_mean_mode` Newton solver
# (src/sparse_laplace_glmm.jl): `aux = (y, cuts, K, nc)`, rebuilt from the
# current outer θ's cutpoint increments at the top of
# `_cumulative_phylo_laplace_fg` below (cutpoints are fixed constants during
# the inner Newton solve, exactly like a nuisance-family's `aux_from(θσ)`).
_laplace_value(::Val{:cumlogit}, aux, i, η) =
    _cumulative_phylo_kernel(aux.y[i], η, aux.cuts, aux.K, aux.nc)[1]

function _laplace_d12(::Val{:cumlogit}, aux, i, η)
    _, d1, d2, = _cumulative_phylo_kernel(aux.y[i], η, aux.cuts, aux.K, aux.nc)
    return d1, d2
end

function _cumulative_phylo_laplace_fg(y::Vector{Int}, K::Int, nc::Int, n::Int, Xμ,
                                      leaf_node, Q, logdetQ, θ; grad::Bool = false,
                                      b0 = nothing, newton_tol::Real = 1e-10,
                                      newton_maxiter::Int = 60)
    pμ = length(θ) - nc - 1
    βμ = θ[1:pμ]
    δ = θ[pμ+1:pμ+nc]
    logσ = clamp(θ[pμ+nc+1], -8.0, 3.0)
    cuts = _cumulative_cuts(δ)
    aux = (y = y, cuts = cuts, K = K, nc = nc)
    η0 = pμ == 0 ? zeros(eltype(θ), n) : Xμ * βμ
    b, ch, _, ok = _phylo_mean_mode(Val(:cumlogit), aux, η0, leaf_node, Q, logσ; b0 = b0,
                                    tol = newton_tol, maxiter = newton_maxiter)
    if !ok
        return grad ? (1e18, zeros(length(θ)), b, false) : (1e18, b, false)
    end

    q = size(Q, 1)
    invσ2 = exp(-2 * logσ)
    data = zero(eltype(θ))
    prior = 0.5 * invσ2 * dot(b, Q * b)
    gradβ_raw = zeros(eltype(θ), pμ)
    gcuts_raw = zeros(eltype(θ), nc)
    wstore = Vector{eltype(θ)}(undef, n)
    tstore = Vector{eltype(θ)}(undef, n)
    hlostore = Vector{Bool}(undef, n)
    hhistore = Vector{Bool}(undef, n)
    nrlostore = Vector{eltype(θ)}(undef, n)
    nrhistore = Vector{eltype(θ)}(undef, n)
    nwlostore = Vector{eltype(θ)}(undef, n)
    nwhistore = Vector{eltype(θ)}(undef, n)
    @inbounds for i in 1:n
        η = η0[i] + b[leaf_node[i]]
        v, d1, d2, d3, has_lo, has_hi, nval_lo, nval_hi, nr_lo, nr_hi, nw_lo, nw_hi =
            _cumulative_phylo_kernel(y[i], η, cuts, K, nc)
        data += v
        wstore[i] = d2
        tstore[i] = d3
        hlostore[i] = has_lo
        hhistore[i] = has_hi
        nrlostore[i] = nr_lo
        nrhistore[i] = nr_hi
        nwlostore[i] = nw_lo
        nwhistore[i] = nw_hi
        for k in 1:pμ
            gradβ_raw[k] += Xμ[i, k] * d1
        end
        yi = y[i]
        has_hi && (gcuts_raw[yi] += nval_hi)
        has_lo && (gcuts_raw[yi-1] += nval_lo)
    end
    val = data + prior + q * logσ - 0.5 * logdetQ + 0.5 * logdet(ch)
    grad || return val, b, true

    S = takahashi_selinv(ch)
    hd = diag(S)
    traceSQ = _sparse_trace_product(S, Q)
    tlogdet = zeros(eltype(θ), q)
    crossβ = zeros(eltype(θ), q, pμ)
    crossδcuts = zeros(eltype(θ), q, nc)
    gβ = gradβ_raw
    gcuts = gcuts_raw
    @inbounds for i in 1:n
        li = leaf_node[i]
        lever = hd[li]
        tlever = tstore[i] * lever
        tlogdet[li] += tlever
        adj = 0.5 * tlever
        for k in 1:pμ
            xik = Xμ[i, k]
            gβ[k] += xik * adj
            crossβ[li, k] += wstore[i] * xik
        end
        yi = y[i]
        if hhistore[i]
            gcuts[yi] += 0.5 * nwhistore[i] * lever
            crossδcuts[li, yi] += nrhistore[i]
        end
        if hlostore[i]
            gcuts[yi-1] += 0.5 * nwlostore[i] * lever
            crossδcuts[li, yi-1] += nrlostore[i]
        end
    end
    implicit = ch \ tlogdet
    @inbounds for k in 1:pμ
        gβ[k] -= 0.5 * dot(@view(crossβ[:, k]), implicit)
    end
    @inbounds for m in 1:nc
        gcuts[m] -= 0.5 * dot(@view(crossδcuts[:, m]), implicit)
    end

    # Map ∂(val)/∂cuts → ∂(val)/∂δ via the increment reparameterisation's own
    # Jacobian (cuts[1]=δ[1]; cuts[j]=cuts[j-1]+exp(δ[j]) for j≥2):
    # ∂cuts[j]/∂δ[1]=1 for every j (δ[1] shifts every cutpoint equally);
    # ∂cuts[j]/∂δ[m]=exp(δ[m]) for j≥m≥2, else 0. So g_δ[m] is the TAIL SUM
    # Σ_{j=m}^{nc} g_cuts[j], scaled by exp(δ[m]) for m≥2 (unscaled at m=1).
    gδ = zeros(eltype(θ), nc)
    tail = zero(eltype(θ))
    for m in nc:-1:1
        tail += gcuts[m]
        gδ[m] = m == 1 ? tail : tail * exp(δ[m])
    end

    Qb = Q * b
    Pu = invσ2 .* Qb
    gσ = q - invσ2 * (dot(b, Qb) + traceSQ) + dot(implicit, Pu)
    return val, vcat(gβ, gδ, [gσ]), b, true
end

"""
    _fit_cumulative_phylo_laplace(fam, y, Xμ, K, labels, tree, nmμ, grp, g_tol; se)

Sparse-Laplace `CumulativeLogit()` fit with a phylogenetic random intercept
`phylo(1 | grp)` on the mean linear predictor `η` (#563 S8 follow-on). Cutpoints
stay ordinary fixed effects (shared across the tree, exactly as in
`_fit_cumulative_ranef`'s GHQ route); only the `mu` intercept varies by
tip. See the file-level comment above for the exact-gradient design.
"""
function _fit_cumulative_phylo_laplace(fam::CumulativeLogit, y::Vector{Int}, Xμ, K, labels,
                                       tree, nmμ, grp, g_tol; se::Bool = false,
                                       polish_iterations::Int = 0)
    n = length(y); pμ = size(Xμ, 2); nc = K - 1
    Q, leaf_node, _ = _poisson_phylo_setup(tree, labels)
    q = size(Q, 1)
    qchol = cholesky(Symmetric(Q); check = false)
    issuccess(qchol) ||
        error("phylo(1 | $grp) tree precision is not positive definite after root conditioning")
    logdetQ = logdet(qchol)
    last_b = zeros(q)

    function eval_laplace(θ; grad::Bool = false)
        if grad
            val, g, b, ok = _cumulative_phylo_laplace_fg(
                y, K, nc, n, Xμ, leaf_node, Q, logdetQ, θ; grad = true, b0 = last_b
            )
            if !ok
                val, g, b, ok = _cumulative_phylo_laplace_fg(
                    y, K, nc, n, Xμ, leaf_node, Q, logdetQ, θ; grad = true, b0 = zeros(q)
                )
            end
            ok || return 1e18, zeros(length(θ))
            last_b .= b
            return val, g
        else
            val, b, ok = _cumulative_phylo_laplace_fg(
                y, K, nc, n, Xμ, leaf_node, Q, logdetQ, θ; grad = false, b0 = last_b
            )
            if !ok
                val, b, ok = _cumulative_phylo_laplace_fg(
                    y, K, nc, n, Xμ, leaf_node, Q, logdetQ, θ; grad = false, b0 = zeros(q)
                )
            end
            ok || return 1e18
            last_b .= b
            return val
        end
    end

    nll(θ) = eval_laplace(θ; grad = false)
    function grad!(Gout, θ)
        _, g = eval_laplace(θ; grad = true)
        Gout .= g
        return Gout
    end

    θ0 = zeros(pμ + nc + 1)
    θ0[(pμ+1):(pμ+nc)] = _cumulative_cut_init(y, K, n)
    θ0[pμ+nc+1] = log(0.4)             # σ_b
    od = Optim.OnceDifferentiable(nll, grad!, θ0)
    method = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking())
    res_fast = Optim.optimize(od, θ0, method, Optim.Options(g_tol = g_tol, iterations = 250))
    res = if polish_iterations > 0
        try
            θp = Optim.minimizer(res_fast)
            odp = Optim.OnceDifferentiable(nll, grad!, θp)
            Optim.optimize(odp, θp, Optim.LBFGS(),
                           Optim.Options(g_tol = g_tol, iterations = polish_iterations))
        catch
            res_fast
        end
    else
        res_fast
    end
    θ̂ = Optim.minimizer(res)
    nllhat = nll(θ̂)
    θ̂, nllhat = _laplace_boundary_polish(nll, grad!, θ̂, nllhat)   # issue #422
    gfinal = zeros(length(θ̂))
    grad!(gfinal, θ̂)
    converged = _laplace_outer_converged(res, nllhat, gfinal, θ̂, n, g_tol)
    V = if se
        Hθ = _finite_hessian(nll, θ̂; h = _fd_hessian_step(n))
        _vcov_from_hessian(Hθ; context = "sparse-Laplace CumulativeLogit (phylo-mean)")
    else
        fill(NaN, length(θ̂), length(θ̂))
    end
    blocks = [:mu => 1:pμ, :cutpoints => (pμ+1):(pμ+nc), :resd => (pμ+nc+1):(pμ+nc+1)]
    names = [:mu => nmμ, :cutpoints => ["theta$k" for k in 1:nc], :resd => [String(grp)]]
    β̂ = θ̂[1:pμ]; cuts_hat = _cumulative_cuts(θ̂[pμ+1:pμ+nc])
    η̂ = pμ == 0 ? zeros(n) : Xμ * β̂     # population (b=0) linear predictor
    score = _cumulative_score(η̂, cuts_hat, K)
    means = Dict(:mu => score); obs = Dict(:mu => Float64.(y))
    scales = Dict(:ordinal_eta => η̂, :ordinal_cuts => Float64.(cuts_hat))
    fit = DrmFit(fam, blocks, names, θ̂, Matrix(V), -nllhat, n, converged, means, obs, scales)
    return _withnll(fit, nll, grad!)
end

# ---------------------------------------------------------------------------
# Crossed/multiple random intercepts (1|g)+(1|h) on the latent linear
# predictor η (#738), reusing the PLAIN (no-nuisance) sparse augmented-state
# Laplace GLMM engine already shared by Poisson/Binomial's crossed routes
# (`_fit_crossed_mean_laplace` in src/sparse_laplace_glmm.jl) — NOT a new
# integrator for the random effects.
#
# That shared engine's outer θ layout is `[βμ; logσ_g; logσ_h]` — no slot for
# the `nc = K - 1` ordered-cutpoint parameters, which (unlike a dispersion
# nuisance) enter the likelihood as extra fixed structure, not a single
# scalar. Rather than widen the shared engine's layout (a shared-file change
# out of this family file's ownership), the cutpoints are PROFILED OUT: an
# outer derivative-free (Nelder–Mead, since ForwardDiff cannot see through
# the inner engine's own nested `Optim.optimize` calls) search over the
# unconstrained increment parameterisation `_cumulative_cuts` refits the full
# (β, σ_g, σ_h) crossed model at each candidate cutpoint vector via the
# existing engine, and finds the cutpoints minimising that profile deviance
# — exact profile-likelihood optimisation, not an approximation.
#
# Per-observation value/derivatives reuse `_cumulative_loglik` (already
# exact and smooth in η for FIXED cutpoints and category) via nested
# `ForwardDiff.derivative` on η, verified against central finite differences
# (interactive check, not part of the test suite; the same nested-autodiff
# pattern `src/tweedie.jl` documents and checks for `_logpdf_tweedie`).
#
# `_laplace_mean(kind, η)` (the shared engine's per-fit convenience "mean" for
# `fitted()`/reporting) receives only `(kind, η)` — no `aux` — so it cannot
# reconstruct the expected category score, which needs the cutpoints. It
# returns `NaN` here rather than a plausible-looking wrong number; the
# outer `_fit_cumulative_crossed_laplace` overwrites `means`/`scales` with the
# correct expected-category-score convention (matching `_fit_cumulative`)
# once the profiled cutpoints are known.
_laplace_value(::Val{:cumlogit_fixed}, aux, i, η) =
    -_cumulative_loglik(aux.y[i], η, aux.cuts, aux.K, aux.nc)

_laplace_d1(::Val{:cumlogit_fixed}, aux, i, η) =
    ForwardDiff.derivative(x -> _laplace_value(Val(:cumlogit_fixed), aux, i, x), η)

_laplace_d2(::Val{:cumlogit_fixed}, aux, i, η) =
    ForwardDiff.derivative(x -> _laplace_d1(Val(:cumlogit_fixed), aux, i, x), η)

_laplace_d3(::Val{:cumlogit_fixed}, aux, i, η) =
    ForwardDiff.derivative(x -> _laplace_d2(Val(:cumlogit_fixed), aux, i, x), η)

_laplace_mean(::Val{:cumlogit_fixed}, η) = NaN
_laplace_obs(::Val{:cumlogit_fixed}, aux, i) = Float64(aux.y[i])

function _fit_cumulative_crossed_laplace(fam::CumulativeLogit, y::Vector{Int}, Xμ, K, comps, nmμ, g_tol)
    length(comps) == 2 || error("_fit_cumulative_crossed_laplace requires two random-intercept components")
    n = length(y); pμ = size(Xμ, 2); nc = K - 1
    θβ0 = zeros(pμ)

    function inner_fit(δ::Vector{<:Real}; polish_iterations::Int = 0)
        cuts = _cumulative_cuts(δ)
        aux = (y = y, cuts = cuts, K = K, nc = nc)
        return _fit_crossed_mean_laplace(
            fam, Val(:cumlogit_fixed), aux, n, Xμ,
            comps[1][2], comps[1][3], comps[2][2], comps[2][3], nmμ,
            [comps[1][4], comps[2][4]], g_tol;
            θβ0 = θβ0, se = false, polish_iterations = polish_iterations)
    end

    δ0 = _cumulative_cut_init(y, K, n)
    obj(δ) = -inner_fit(δ).loglik
    res = Optim.optimize(obj, δ0, Optim.NelderMead(),
                         Optim.Options(iterations = 500, g_tol = g_tol))
    δ̂ = Optim.minimizer(res)
    fit = inner_fit(δ̂; polish_iterations = 25)

    pμ2 = pμ    # (== pμ; named for readability against the block-index arithmetic below)
    nblk = pμ2 + 2   # mu + 2 crossed RE sds, the engine's own blocks
    blocks = vcat(fit.blocks, [:cutpoints => (nblk + 1):(nblk + nc)])
    names = vcat(fit.coefnames, [:cutpoints => ["theta$k" for k in 1:nc]])
    theta = vcat(fit.theta, δ̂)
    V = fill(NaN, length(theta), length(theta))
    V[1:nblk, 1:nblk] .= fit.vcov

    β̂ = fit.theta[1:pμ2]
    cuts_hat = _cumulative_cuts(δ̂)
    η̂ = pμ2 == 0 ? zeros(n) : Xμ * β̂       # population (b=0) linear predictor
    score = _cumulative_score(η̂, cuts_hat, K)
    means = Dict(:mu => score)
    obs = Dict(:mu => Float64.(y))
    scales = Dict(:ordinal_eta => η̂, :ordinal_cuts => Float64.(cuts_hat))

    return DrmFit(fam, blocks, names, theta, V, fit.loglik, fit.nobs, fit.converged,
                 means, obs, scales)
end
