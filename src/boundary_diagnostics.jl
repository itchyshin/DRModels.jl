# boundary_diagnostics.jl — variance-component boundary diagnostic for Gaussian
# fits with a structured / grouped random effect and a homoscedastic residual.
#
# Why this exists (#724, #697). With one observation per tip,
#     y ~ N(Xβ, σ_a² A + σ_e² I),
# the MLE very often sits ON a variance boundary: σ_e² = 0 (the structured term
# absorbs everything; geiger::carnivores, n = 16) or σ_a² = 0 (a near-star tree,
# where A ≈ h·I makes σ_a² and σ_e² indistinguishable). At such a point the
# likelihood is still *increasing* in the boundary variance (profile Δnll ≈ c·σ²
# with c ≈ 5.9 on the #724 repro), so the working-scale gradient d nll / d log σ
# = 2σ²·c vanishes quadratically as σ → 0 and an L-BFGS run on log σ stops wherever
# the gradient first drops under `g_tol`. The reported σ̂ (1e-5 … 1e-4) is therefore an
# optimiser-stopping artefact that differs between engines (drmTMB 4.3e-5,
# DRModels.jl 3.1e-4 on the issue's data) although μ̂, loglik and the structured SD
# agree to 1e-8. The right behaviour is to SAY the component is at its boundary,
# not to chase twin equality on it.
#
# CALIBRATION (2026-09-30, Julia 1.10.12 on Totoro; 48 simulated phylogenetic fits
# with n ∈ {16, 64}, true residual share 0 … 0.9, plus the full test suite — see
# docs/dev-log/2026-09-30-class3-boundary.md): boundary fits form a tight cluster
# σ̂_e/rms(BLUP) ≤ 4.4e-5 (σ̂_e itself 3e-5 … 6e-5, the optimiser's stopping level), and
# every interior fit has σ̂_e/rms(BLUP) ≥ 2.2e-2, a 2.7-order empty gap. The threshold
# 1e-3 is the geometric middle of that gap, and it is the same 1e-3 relative cut used
# for the σ_b → 0 restart in `gaussian_ranef.jl`.

using Printf: @sprintf

"""
Relative scale below which a variance component counts as "at the boundary":
its SD, as a fraction of the SD of everything else in the model (see
`_variance_boundary`).
"""
const _VARIANCE_BOUNDARY_RATIO = 1e-3

# Task-local switch so bootstrap / profile REFITS (which routinely land on the same
# boundary as the base fit) do not repeat the warning B times.
function _without_boundary_warnings(f)
    tls = task_local_storage()
    old = get(tls, :drm_quiet_boundary, false)
    tls[:drm_quiet_boundary] = true
    try
        return f()
    finally
        tls[:drm_quiet_boundary] = old
    end
end

# rms of a BLUP vector, or `nothing` when it is not a plain finite numeric vector.
function _blup_rms(b)
    b isa AbstractVector{<:Real} || return nothing
    isempty(b) && return nothing
    all(isfinite, b) || return nothing
    return sqrt(sum(abs2, b) / length(b))
end

"""
    _variance_boundary(fit; ratio = 1e-3) -> Union{Nothing,NamedTuple}

Boundary diagnostic for a Gaussian fit with at least one grouped / structured
random effect (`phylo`, `relmat`, `animal`, `spatial`, `(1 | g)`) and a
HOMOSCEDASTIC residual (`sigma ~ 1`). Returns `nothing` when the fit has no such
structure (covariate-dependent `sigma`, `sd(group) ~ …`, no random effect, no stored
BLUPs): the diagnostic is then not defined, not "clean".

Scale reference. The structured scale of component `k` is `rms(BLUP_k)` — the
root-mean-square of its conditional modes, which are stored in response units on
every route (so the raw-branch-length versus correlation-scale convention of
`sd_phylo` does not matter). `rms(BLUP_k) ≤ ` the marginal SD of the component, so the
ratios below can only be too LARGE, i.e. the diagnostic errs towards NOT flagging.

Fields of the result:

- `residual_at_boundary` — `σ̂_e < ratio · sqrt(Σ_k rms_k²)`: the residual variance
  share is below `ratio² ≈ 1e-6`. σ̂_e is then an optimiser-stopping artefact.
- `residual_ratio` — `σ̂_e / sqrt(Σ_k rms_k²)`.
- `structured_at_boundary` — component names with
  `rms_k < ratio · sqrt(σ̂_e² + Σ_{j≠k} rms_j²)` (the structured SD has collapsed to 0).
- `structured_ratios` — those ratios, one per component.
- `one_obs_per_group` — some component has exactly one BLUP per observation, so
  σ_a and σ_e are separated only by the covariance structure `A`, not by replicates.
"""
function _variance_boundary(fit::DrmFit; ratio::Real = _VARIANCE_BOUNDARY_RATIO)
    fit.family isa Gaussian || return nothing
    have = Dict(p => r for (p, r) in fit.blocks)
    # Residual log σ: homoscedastic only (`sigma ~ 1`, or the dedicated :resid block).
    rblock = haskey(have, :resid) ? have[:resid] : get(have, :sigma, nothing)
    (rblock !== nothing && length(rblock) == 1) || return nothing
    haskey(have, :resd) || return nothing                      # structured SDs live here
    any(p -> first(p) in (:sd, :sd_phylo), fit.blocks) && return nothing
    fit.ranef isa AbstractDict || return nothing
    ci = findfirst(cn -> cn[1] === :resd, fit.coefnames)
    ci === nothing && return nothing
    nms = fit.coefnames[ci][2]
    σe = exp(fit.theta[first(rblock)])
    isfinite(σe) || return nothing

    comps = Symbol[]
    rms = Float64[]
    ngroups = Int[]
    for nm in nms
        b = get(fit.ranef, Symbol(nm), nothing)
        r = b === nothing ? nothing : _blup_rms(b)
        r === nothing && continue
        push!(comps, Symbol(nm)); push!(rms, r); push!(ngroups, length(b))
    end
    isempty(comps) && return nothing

    tot_struct = sqrt(sum(abs2, rms))
    resid_ratio = tot_struct > 0 ? σe / tot_struct : Inf
    sratios = Dict{Symbol,Float64}()
    for (k, c) in enumerate(comps)
        other = sqrt(σe^2 + sum(abs2, rms) - rms[k]^2)
        sratios[c] = other > 0 ? rms[k] / other : Inf
    end
    return (
        residual_at_boundary = isfinite(resid_ratio) && resid_ratio < ratio,
        residual_ratio = resid_ratio,
        structured_at_boundary = [c for c in comps if sratios[c] < ratio],
        structured_ratios = sratios,
        one_obs_per_group = any(==(fit.nobs), ngroups),
    )
end

# Fit-time advisory. Silent on a clean fit, on fits the diagnostic does not cover,
# and inside bootstrap refits (`_without_boundary_warnings`).
function _warn_variance_boundary(fit)
    fit isa DrmFit || return fit
    get(task_local_storage(), :drm_quiet_boundary, false) === true && return fit
    _is_temporal_fit(fit) && return _warn_temporal_boundary(fit)   # temporal(): its own rules
    vb = try
        _variance_boundary(fit)
    catch
        nothing        # a diagnostic must never break a fit
    end
    vb === nothing && return fit
    msg = _variance_boundary_message(vb)
    msg === nothing || @warn msg
    return fit
end

function _variance_boundary_message(vb)
    (vb.residual_at_boundary || !isempty(vb.structured_at_boundary)) || return nothing
    io = IOBuffer()
    if vb.residual_at_boundary
        println(io, "Residual σ is at its lower boundary (σ̂ is ", @sprintf("%.1e", vb.residual_ratio),
                " × the SD of the random-effect BLUPs; boundary cut 1e-3): the structured term absorbs ",
                "essentially all the variance (variance share ≈ 1).")
        println(io, "The MLE is σ² = 0. The likelihood is still increasing in σ² there, so the gradient on ",
                "log σ vanishes like σ² and the optimiser stops wherever it first falls under `g_tol`: ",
                "the reported σ̂ is an optimiser-stopping artefact, NOT an estimate, and it will differ ",
                "between engines (e.g. drmTMB vs DRModels.jl) while μ̂, loglik and the structured SD agree. ",
                "Its Wald SE is not meaningful. Compare μ, loglik and the structured SD, not σ.")
    end
    if !isempty(vb.structured_at_boundary)
        println(io, "Structured SD at its lower boundary for ", join(string.(vb.structured_at_boundary), ", "),
                ": the grouped / phylogenetic variance is estimated as 0 relative to the residual, so its ",
                "SE (and any heritability / repeatability built on it) is not meaningful.")
    end
    if vb.one_obs_per_group
        println(io, "This design has ONE observation per group, so the structured variance σ_a² and the ",
                "residual σ_e² are separated only by the off-diagonal structure of the covariance matrix ",
                "A (relatedness), not by replicates; when A is nearly diagonal (a near-star tree, weak ",
                "signal) they are not separately identified and one collapses to the boundary. Add ",
                "within-group replicates, or report σ_a² + σ_e² and use `confint(fit; method = :profile)`.")
    end
    return rstrip(String(take!(io)))
end
