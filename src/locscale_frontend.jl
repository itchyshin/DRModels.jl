# locscale_frontend.jl — public `drm()` routing for the non-Gaussian location–
# scale model (#202, slice 3b). Mirrors drmTMB/brms: a correlated random-effect
# tag `(1 | <tag> | group)` shared by the `mu` and `sigma` formulas couples the
# mean-axis and dispersion-axis intercept REs into ONE 2×2 group-level covariance
# Λ — exactly what `_fit_locscale` estimates. No new family handle: the shared
# tag across the two formulas is itself the trigger.
#
#   drm(bf(@formula(y ~ x + (1|p|sp)), @formula(sigma ~ x + (1|p|sp))),
#       NegBinomial2(); data) → _fit_locscale(Val(:nb2), …; se=true)
#
# Intercept-only coupled term `(1 | tag | group)`. Bare `group` → i.i.d. (Q = I);
# `phylo(group)` / `relmat` / `animal` / `spatial` → structured Q (#209 closed;
# family `drm()` must forward `tree`/`K`/`A`/`coords` into this frontend).

using SparseArrays: sparse
using LinearAlgebra: I

# Group slot of a coupled `(1 | tag | <group>)` term: either a bare grouping
# variable `g` (→ `(:iid, g)`) or a structured marker `phylo(g)` / `relmat(g)` /
# `animal(g)` / `spatial(g)` (→ `(:phylo, g)` etc.). The structured form routes
# the group-level covariance through `_locscale_relmat_setup` / the phylo path
# instead of the i.i.d. `Q = I`.
function _ls_group_slot(slot)
    if slot isa FunctionTerm && slot.f === relmat
        return (:relmat, slot.args[1].sym)
    elseif slot isa FunctionTerm && slot.f === animal
        return (:animal, slot.args[1].sym)
    elseif slot isa FunctionTerm && slot.f === phylo
        return (:phylo, slot.args[1].sym)
    elseif slot isa FunctionTerm && slot.f === spatial
        return (:spatial, slot.args[1].sym)
    elseif slot isa Term
        return (:iid, slot.sym)
    else
        error("location–scale coupled group must be a grouping variable or a " *
              "phylo/relmat/animal/spatial marker (got `$slot`)")
    end
end

# Split a formula rhs into its fixed part and a single coupled
# `(1 | tag | group)` term, if present. Returns `(fixed_rhs, coupled)` where
# `coupled` is `(tag, group, struct)` or `nothing`: `tag`/`group` are `Symbol`s
# and `struct` is the structure kind (`:iid` for a bare group, else
# `:phylo`/`:relmat`/`:animal`/`:spatial`). Julia parses `1 | p | g` left-
# associatively as `|(|(1, p), g)`, so the coupled term is a `FunctionTerm{|}`
# whose first argument is itself a `FunctionTerm{|}`; a structured group wraps the
# third slot, e.g. `(1 | p | relmat(g))` → `|(|(1,p), relmat(g))`.
function _ls_parse_coupled(rhs)
    terms = rhs isa Tuple ? collect(rhs) : Any[rhs]
    fixed = Any[]
    coupled = nothing
    for t in terms
        if t isa FunctionTerm && t.f === (|) &&
           t.args[1] isa FunctionTerm && t.args[1].f === (|)
            coef = t.args[1].args[1]
            (coef isa ConstantTerm && coef.n == 1) ||
                error("location–scale coupled random effect must be intercept-only: " *
                      "`(1 | tag | group)` (got `($coef | … | …)`)")
            coupled === nothing ||
                error("only one coupled `(1 | tag | group)` term per formula is supported")
            struct_kind, grp = _ls_group_slot(t.args[2])
            coupled = (t.args[1].args[2].sym, grp, struct_kind)   # (tag, group, struct)
        else
            push!(fixed, t)
        end
    end
    fixed_rhs = isempty(fixed) ? ConstantTerm(1) :
                length(fixed) == 1 ? fixed[1] : Tuple(fixed)
    return fixed_rhs, coupled
end

# Detect a coupled location–scale random effect shared by the mean and sigma
# formulas. Returns `(group, struct, fixed_mu, fixed_sigma)` or `nothing`. Errors
# if a coupled term appears on only one axis or the tag/group/structure disagree.
function _ls_coupled_re(rhs_mu, rhs_sigma)
    fμ, cμ = _ls_parse_coupled(rhs_mu)
    fσ, cσ = _ls_parse_coupled(rhs_sigma)
    if cμ === nothing && cσ === nothing
        return nothing
    elseif cμ === nothing || cσ === nothing
        error("a coupled random effect `(1 | tag | group)` must appear in BOTH the mean " *
              "and the `sigma` formula to form a location–scale covariance")
    elseif cμ != cσ
        error("the coupled random-effect tag/group/structure must match across mu and sigma " *
              "(got `$(cμ)` vs `$(cσ)`)")
    end
    return (group = cμ[2], structkind = cμ[3], fixed_mu = fμ, fixed_sigma = fσ)
end

# Build a `DrmFit` from a `_fit_locscale` result. The engine packs
# λ = [log L11, L21, log L22]; the package's `:recov` block (reused for summary /
# `vc`) expects [log L11, log L22, L21], so we permute the last three entries of
# θ and of the covariance. The mean/obs/scale slots are family-aware: NB2/Gamma
# use the log link (means = exp Xμβ) and report the raw response; Beta/Binomial
# use the logit link (means = logistic Xμβ) and report the observed proportion
# (`obs_prop`/`trials` are passed in for the count-with-trials BetaBinomial).
function _build_locscale_drmfit(kind, fam, fitres, y, Xμ, Xψ, nmμ, nmσ, grp::String;
                                obs_prop = nothing, trials = nothing)
    pμ = size(Xμ, 2); pψ = size(Xψ, 2); n = size(Xμ, 1)
    βμ = fitres.beta_mu; βψ = fitres.beta_psi
    λ = fitres.θ[pμ+pψ+1:pμ+pψ+3]                         # [logL11, L21, logL22]
    theta = vcat(βμ, βψ, λ[1], λ[3], λ[2])                # → recov order
    perm = vcat(collect(1:(pμ+pψ)), [pμ+pψ+1, pμ+pψ+3, pμ+pψ+2])
    V = fitres.vcov === nothing ? fill(NaN, length(theta), length(theta)) :
        fitres.vcov[perm, perm]
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pψ),
              :recov => (pμ+pψ+1):(pμ+pψ+3)]
    names = [:mu => nmμ, :sigma => nmσ,
             :recov => ["$grp:L11", "$grp:L22", "$grp:L21"]]
    logit_link = kind isa Val{:beta} || kind isa Val{:betabinomial}
    means = Dict(:mu => logit_link ? _logistic.(Xμ * βμ) : exp.(Xμ * βμ))
    obs = Dict(:mu => obs_prop === nothing ? Float64.(y) : Float64.(obs_prop))
    scales = Dict(:sigma => exp.(Xψ * βψ))
    trials === nothing || (scales[:trials] = Float64.(trials))
    return DrmFit(fam, blocks, names, theta, V, -fitres.nll, n, fitres.converged,
                  means, obs, scales)
end

# Build the engine response and the family-specific design / validation for a
# coupled location–scale fit. Returns `(y, Xμ, Xψ, nmμ, nmσ, obs_prop, trials)`:
# `y` is the per-observation response the kernels consume (a scalar for
# NB2/Gamma/Beta, a `(successes, trials)` tuple for BetaBinomial); `obs_prop` /
# `trials` populate the DrmFit response slots (BetaBinomial only). The mean-axis
# start is left to the engine (`_ls_default_betastart`), single-sourced there.
function _ls_frontend_design(kind, f, lc, data)
    if kind isa Val{:betabinomial}
        f.response2 === nothing &&
            error("BetaBinomial() location–scale needs a two-column response: " *
                  "bf(cbind(successes, failures) ~ … (1|tag|g), sigma ~ … (1|tag|g))")
        _require_finite_inputs(; response=(_table_column(data, f.response), string(f.response)),
                               response_allow_missing=false)
        _require_finite_inputs(; response=(_table_column(data, f.response2), string(f.response2)),
                               response_allow_missing=false)
        s = Float64.(getproperty(data, f.response))
        fl = Float64.(getproperty(data, f.response2))
        (all(si -> si ≥ 0 && isinteger(si), s) && all(fi -> fi ≥ 0 && isinteger(fi), fl)) ||
            error("BetaBinomial() requires non-negative integer successes and failures")
        ntr = s .+ fl
        _, Xμ, nmμ = _design(f.response, lc.fixed_mu, data)
        _, Xψ, nmσ = _design(f.response, lc.fixed_sigma, data)
        yeng = [(s[i], ntr[i]) for i in eachindex(s)]
        return yeng, Xμ, Xψ, nmμ, nmσ, s ./ ntr, ntr
    end
    y, Xμ, nmμ = _design(f.response, lc.fixed_mu, data)
    _, Xψ, nmσ = _design(f.response, lc.fixed_sigma, data)
    if kind isa Val{:nb2}
        all(yi -> yi ≥ 0 && isinteger(yi), y) ||
            error("NegBinomial2() requires non-negative integer counts as the response")
    elseif kind isa Val{:gamma}
        all(yi -> yi > 0, y) ||
            error("Gamma() requires strictly positive responses")
    elseif kind isa Val{:beta}
        all(yi -> 0 < yi < 1, y) ||
            error("Beta() requires responses strictly in the open interval (0, 1)")
    else
        error("unsupported location–scale family kind: $kind")
    end
    return y, Xμ, Xψ, nmμ, nmσ, nothing, nothing
end

# Resolve the group-level precision `(Q, gidx, G)` for the coupled term. A bare
# group is i.i.d. (`Q = I`); a `phylo`/`relmat`/`animal`/`spatial` group routes
# through the same structured-covariance setup the mean-only Laplace routes use
# (tree precision, or `C⁻¹` for a user-supplied PD covariance).
function _ls_frontend_grouping(lc, data, tree, K, A, coords)
    labels = getproperty(data, lc.group)
    if lc.structkind === :iid
        gidx, G = _group_index(labels)
        return sparse(1.0 * I, G, G), gidx, G
    elseif lc.structkind === :phylo
        tree === nothing &&
            error("(1 | tag | phylo($(lc.group))) needs `tree = …`")
        return _locscale_phylo_setup(tree, labels)
    else  # :relmat / :animal / :spatial
        C = _poisson_structured_cov(lc.structkind, lc.group, K, A, coords)
        return _locscale_relmat_setup(C, labels)
    end
end

# Route a coupled location–scale formula to the engine and wrap the result.
# Structured-covariance kwargs (`tree`/`K`/`A`/`coords`) are forwarded from the
# family `drm()` and consumed only when the coupled group carries a structured
# marker; an i.i.d. coupled group ignores them.
function _fit_locscale_frontend(kind, fam, f, rhs, lc, data; g_tol, se,
                                tree = nothing, K = nothing, A = nothing, coords = nothing)
    y, Xμ, Xψ, nmμ, nmσ, obs_prop, trials = _ls_frontend_design(kind, f, lc, data)
    Q, gidx, G = _ls_frontend_grouping(lc, data, tree, K, A, coords)
    # A public LSS fit is the baseline for profile and bootstrap inference.  The
    # default 1,000 outer iterations can leave a small Gamma fixture
    # uncertified on some BLAS/runner combinations. A 2,000-iteration budget
    # preserves the existing tolerance and stopping rule while giving that
    # public fit a bounded additional chance to reach its gradient criterion.
    fitres = _fit_locscale(
        kind, y, Xμ, Xψ, gidx, G, Q;
        g_tol=g_tol, iterations=2_000, se=se, whitened=true,
    )
    fit = _build_locscale_drmfit(kind, fam, fitres, y, Xμ, Xψ, nmμ, nmσ, String(lc.group);
                                 obs_prop = obs_prop, trials = trials)
    # Carry the structured design so `confint(:profile)` can route to the robust
    # location–scale profiler. The objective uses the engine packing; the DrmFit
    # `theta` is in `:recov` order, so the profile router permutes between them.
    return _withnll(fit, LocScaleObjective(kind, y, Xμ, Xψ, gidx, G, Q; whitened=true))
end
