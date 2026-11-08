# Structured sigma random effects for NB2 and Gamma (dense Laplace).
#
# Two shapes, kept apart because native drmTMB 0.8.1 treats them differently
# (probed 2026-11-08, record in DESIGN-structured-sigma.md):
#
#   A. NB2 `sigma ~ 1 + animal|relmat|phylo|spatial(1 + x | g)`, NO SD formula.
#      Native drmTMB FITS this (independent intercept and slope fields with one
#      constant SD each). Gamma: native refuses every structured sigma term.
#   B. NB2 / Gamma `sigma ~ 1 + animal|relmat|phylo|spatial(1 | g)` with an SD
#      formula `sd(g, level, sigma) ~ z`. Native drmTMB REFUSES this for both
#      families. It is implemented here as a JULIA-ONLY extension, checked
#      against a dense ForwardDiff Laplace oracle only (no R twin exists).
#
# Model (mean fixed; sigma-side latent fields u_k ~ N(0, K), K raw animal/relmat,
# tree correlation for phylo, the native fixed kernel for spatial):
#
#   eta_i = (X_sigma beta_sigma)_i + sum_k load_k[i] * s_k[g_k[i]] * v_k[g_k[i]],
#   v_k ~ N(0, K_k)            (SD s_k = exp(Z alpha_k): constant in A, formula in B)
#
# Laplace value as `_fit_gaussian_sigma_sd_laplace` (same kernels, same inner
# solver, same outer optimiser): nll = F(v^) + 1/2 log det H + 1/2 sum log det K_k.

# Joint negative log density in v (concatenated fields) with gradient / dense Hessian.
function _sf_joint(fam, y, mu0, eta0, gs, loads, ds, Ps, off, m, v; deriv::Bool)
    T = promote_type(eltype(mu0), eltype(eta0), eltype(v), eltype(first(ds)))
    nf = length(gs)
    n = length(y)
    F = zero(T)
    g = deriv ? zeros(T, m) : nothing
    H = deriv ? zeros(T, m, m) : nothing
    idx = Vector{Int}(undef, nf)
    dd = Vector{T}(undef, nf)
    @inbounds for i in 1:n
        eta = eta0[i]
        for k in 1:nf
            idx[k] = off[k] + gs[k][i]
            dd[k] = ds[k][i] * loads[k][i]
            eta += dd[k] * v[idx[k]]
        end
        f, _, fe, _, _, fee = _gsl_kernel(fam, y[i], mu0[i], eta)
        F += f
        if deriv
            for k in 1:nf
                g[idx[k]] += dd[k] * fe
                for l in 1:nf
                    H[idx[k], idx[l]] += dd[k] * dd[l] * fee
                end
            end
        end
    end
    for k in 1:nf
        r = (off[k] + 1):(off[k] + size(Ps[k], 1))
        Pv = Ps[k] * view(v, r)
        F += 0.5 * dot(view(v, r), Pv)
        if deriv
            g[r] .+= Pv
            H[r, r] .+= Ps[k]
        end
    end
    return F, g, H
end

# fields: vector of (gidx, G, Z, nm, grp, loading, Kmat, block, member).
function _fit_sigma_fields_laplace(fam::Union{NegBinomial2,Gamma}, y, Xμ, Xσ, fields, nmμ, nmσ,
                                   g_tol; se::Bool = true)
    n = length(y)
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    nf = length(fields)
    Gs = [fl[2] for fl in fields]
    off = cumsum([0; Gs[1:(end - 1)]])
    m = sum(Gs)
    m <= _GSL_MAX_LATENT || throw(ArgumentError(
        "drm: structured sigma random effects use a dense Laplace engine limited to " *
        "$(_GSL_MAX_LATENT) latent coordinates (got $m)."))
    pz = [size(fl[3], 2) for fl in fields]
    ioff = pμ + pσ .+ cumsum([0; pz[1:(end - 1)]])
    iz = [(ioff[k] + 1):(ioff[k] + pz[k]) for k in 1:nf]
    ntheta = pμ + pσ + sum(pz)
    Ps = Matrix{Float64}[]
    logdetK = 0.0
    for fl in fields
        Ck = cholesky(Symmetric(Matrix{Float64}(fl[7])); check = false)
        issuccess(Ck) || throw(ArgumentError(
            "drm: the structured sigma matrix must be symmetric positive definite."))
        P = Matrix(inv(Ck)); P = (P .+ P') ./ 2
        push!(Ps, P)
        logdetK += logdet(Ck)
    end
    gs = [fl[1] for fl in fields]
    loads = [fl[6] for fl in fields]

    function pieces(θ)
        mu0 = Xμ * θ[1:pμ]
        eta0 = Xσ * θ[(pμ + 1):(pμ + pσ)]
        ds = [exp.(fields[k][3] * θ[iz[k]])[gs[k]] for k in 1:nf]
        return mu0, eta0, ds
    end
    joint_at(θ) = begin
        mu0, eta0, ds = pieces(θ)
        (vv, deriv) -> _sf_joint(fam, y, mu0, eta0, gs, loads, ds, Ps, off, m, vv; deriv = deriv)
    end
    function laplace(θ::AbstractVector{T}, vhat::Vector{Float64}) where {T}
        joint = joint_at(θ)
        v = convert(Vector{T}, vhat)
        if T <: ForwardDiff.Dual
            _, g, H = joint(vhat, true)
            # A Hessian that is not positive definite here used to raise PosDefException
            # out of `drm()`; it is the same non-PD penalty the line below applies.
            Cn = cholesky(Symmetric(H); check = false)
            issuccess(Cn) || return T(REML_NONPD_PENALTY)
            v = vhat .- (Cn \ g)
        end
        F, _, H = joint(v, true)
        C = cholesky(Symmetric(H); check = false)
        issuccess(C) || return T(REML_NONPD_PENALTY)
        return F + 0.5 * logdet(C) + 0.5 * logdetK
    end
    fg = function (θ, b0, grad)
        θf = Vector{Float64}(θ)
        joint = joint_at(θf)
        v, ok = _gsl_mode(joint, Vector{Float64}(b0))
        ok || ((v, ok) = _gsl_mode(joint, zeros(m)))
        ok || return grad ? (1e18, fill(NaN, ntheta), zeros(m), false) : (1e18, zeros(m), false)
        _, _, H = joint(v, true)
        issuccess(cholesky(Symmetric(H); check = false)) ||
            return grad ? (1e18, fill(NaN, ntheta), v, false) : (1e18, v, false)
        val = laplace(θf, v)
        grad || return (val, v, true)
        gθ = ForwardDiff.gradient(t -> laplace(t, v), θf)
        return (val, gθ, v, all(isfinite, gθ))
    end

    θ0 = zeros(ntheta)
    θ0[1:pμ] .= Xμ \ log.(y .+ 0.5)
    θ0[pμ + 1] = log(0.5)
    for k in 1:nf
        θ0[iz[k]] .= fields[k][3] \ fill(log(0.3), Gs[k])
    end
    what = "$(nameof(typeof(fam))) structured sigma random effect"
    θ̂, nllhat, conv, V, nll, grad! = _ordinary_laplace_optimize(
        fg, θ0, n, m, g_tol; se = false, context = what)
    # Multimodality check (as the Gaussian engine): a zero start must reach the
    # same Laplace value, otherwise the fit is not reported as converged.
    let joint = joint_at(θ̂)
        v0, ok0 = _gsl_mode(joint, zeros(m))
        if ok0
            _, _, H0 = joint(v0, true)
            if issuccess(cholesky(Symmetric(H0); check = false))
                nll0 = laplace(θ̂, v0)
                if isfinite(nllhat) && abs(nll0 - nllhat) > 1e-6 * max(1.0, abs(nllhat))
                    @warn "drm: $what: the random-effect mode is not unique at the estimate " *
                          "(Laplace log-likelihood $(round(-nllhat; digits = 4)) from the optimiser's " *
                          "path, $(round(-nll0; digits = 4)) from a zero start); the fit is marked " *
                          "not converged."
                    conv = false
                end
            end
        end
    end
    if se
        Hθ = _finite_hessian_from_gradient(grad!, θ̂; context = what)
        V = all(isfinite, Hθ) ? Matrix(_vcov_from_hessian(Hθ)) : fill(NaN, ntheta, ntheta)
    end

    vhat, _ = _gsl_mode(joint_at(θ̂), zeros(m))
    βμ = θ̂[1:pμ]; βσ = θ̂[(pμ + 1):(pμ + pσ)]
    eta_fixed = Xσ * βσ
    sd_g = [exp.(fields[k][3] * θ̂[iz[k]]) for k in 1:nf]
    bb = [sd_g[k] .* vhat[(off[k] + 1):(off[k] + Gs[k])] for k in 1:nf]   # raw log-sigma effects
    eta_cond = copy(eta_fixed)
    for k in 1:nf
        eta_cond .+= loads[k] .* bb[k][gs[k]]
    end
    means = Dict(:mu => exp.(Xμ * βμ))
    obs = Dict(:mu => Vector{Float64}(y))
    grp1 = fields[1][5]
    scales = Dict{Symbol,Vector{Float64}}(
        :sigma => exp.(eta_cond),
        :sigma_fixed => exp.(eta_fixed),
        :sigma_random_effect => bb[1],
        :sigma_group_index => Float64.(gs[1]),
        :sd_group => sd_g[1],
        :sd => sd_g[1][gs[1]])
    nf > 1 && (scales[:sigma_random_effect_slope] = bb[2])
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:pμ, :sigma => (pμ + 1):(pμ + pσ)]
    names = Pair{Symbol,Vector{String}}[:mu => nmμ, :sigma => nmσ]
    ranef = Dict{Symbol,Any}()
    for k in 1:nf
        push!(blocks, fields[k][8] => iz[k]); push!(names, fields[k][8] => fields[k][4])
        ranef[Symbol(fields[k][5], "_logsigma_", fields[k][9])] = bb[k]
    end
    fit = DrmFit(fam, blocks, names, θ̂, V, -nllhat, n, conv, means, obs, scales)
    return _withranef(_withmarginal(_withnll(fit, nll, grad!), :Laplace), ranef)
end

# Kernel / row map of one structured sigma term, the SAME objects the fit uses
# (also rebuilt by the marginal simulator). Returns (C, gidx, G).
function _ss_kernel(kind::Symbol, grp::Symbol, data, tree, K, A, coords)
    hasproperty(data, grp) || throw(ArgumentError("drm: data are missing the grouping column `$grp`"))
    if kind === :phylo
        (K === nothing && A === nothing && coords === nothing) || throw(ArgumentError(
            "drm: phylo sigma structured effects use `tree = …` only."))
        tree === nothing && throw(ArgumentError("phylo(… | $grp) on sigma needs `tree = …`"))
        phy = tree isa AbstractString ? augmented_phy(tree) : tree
        gidx = _phylo_mean_leaf_index(phy, getproperty(data, grp))
        return _phylo_correlation(phy), gidx, phy.n_leaves
    elseif kind === :spatial
        (K === nothing && A === nothing && tree === nothing) || throw(ArgumentError(
            "drm: spatial sigma structured effects use `coords = …` only."))
        coords === nothing && throw(ArgumentError("spatial(… | $grp) on sigma needs `coords = …`"))
        gidx, G = _group_index(getproperty(data, grp); col = grp)
        return _gaussian_spatial_sd_kernel(coords, G, grp), gidx, G
    end
    tree === nothing || throw(ArgumentError("drm: $kind sigma structured effects do not use `tree`."))
    gidx, G = _group_index(getproperty(data, grp); col = grp)
    if kind === :relmat
        (A === nothing && coords === nothing) || throw(ArgumentError(
            "drm: relmat sigma structured effects use `K = …` only."))
        K === nothing && throw(ArgumentError("relmat(… | $grp) on sigma needs the covariance matrix `K = …`"))
        C = Matrix{Float64}(K)
    else
        (K === nothing && coords === nothing) || throw(ArgumentError(
            "drm: animal sigma structured effects use `A = …` only."))
        A === nothing && throw(ArgumentError("animal(… | $grp) on sigma needs the relatedness matrix `A = …`"))
        C = Matrix{Float64}(A)
    end
    size(C) == (G, G) || throw(ArgumentError(
        "structured matrix must be $(G)×$(G) (the number of `$grp` levels)"))
    all(isfinite, C) || throw(ArgumentError("structured matrix must be finite"))
    return C, gidx, G
end

const _SS_LEVEL = Dict(:animal => :animal, :relmat => :relmat, :phylo => :phylogenetic,
                       :spatial => :spatial)

function _ss_check_response(fam, y)
    if fam isa Gamma
        all(yi -> isfinite(yi) && yi > 0, y) || throw(ArgumentError(
            "Gamma() requires finite, strictly positive responses"))
    else
        all(yi -> isfinite(yi) && yi >= 0 && isinteger(yi), y) || throw(ArgumentError(
            "NegBinomial2() requires finite, non-negative integer counts as the response"))
    end
end

# ---- Shape B: structured sigma INTERCEPT with an SD formula (Julia-only) ------

_ss_has_sigma_structured_sd(f::DrmFormula) = any(
    item -> item.target.dpar === :sigma && item.target.level in (:animal, :relmat, :phylogenetic, :spatial),
    _sd_target_parts(f))

function _drm_structured_sigma_sd_laplace(f::DrmFormula, fam::Union{NegBinomial2,Gamma};
                                          data, tree = nothing, K = nothing, A = nothing,
                                          coords = nothing, g_tol::Real = 1e-8, se::Bool = true)
    famname = fam isa Gamma ? "Gamma" : "NegBinomial2"
    what = "$famname structured sigma SD regression (Julia-only; native drmTMB 0.8.1 refuses it)"
    isempty(_sdphylo_parts(f)) || throw(ArgumentError(
        "drm: $what cannot be combined with `sd(g, phylogenetic)` on the mean."))
    targets = _sd_target_parts(f)
    length(targets) == 1 || throw(ArgumentError(
        "drm: $what takes exactly one `sd(g, level, sigma) ~ …` formula (got $(length(targets)))."))
    sig_t = only(targets)
    allowed = Set{Symbol}((:mu, :sigma, sig_t.key))
    all(p -> first(p) in allowed, f.forms) || throw(ArgumentError(
        "drm: $what supports only `mu`, `sigma` and one `sd(...)` formula (no `zi`/`hu`)."))
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    (mv === nothing && st === nothing && isempty(re)) || throw(ArgumentError(
        "drm: $what takes a fixed-effect mean (no mean random or structured terms)."))
    _has_offset_term(rhs[:mu]) && throw(ArgumentError("drm: $what does not support `offset(...)`."))
    fixed_sigma, sigma_re, mvs, sts = _split_ranef(rhs[:sigma])
    (sts !== nothing && isempty(sigma_re) && mvs === nothing) || throw(ArgumentError(
        "drm: $what needs `sigma ~ … + animal|relmat|phylo|spatial(1 | g)` with the matching `sd(g, level, sigma)`."))
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "drm: $what requires observed responses; remove incomplete rows first (`drm_listwise`)."))
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    _, Xσ, nmσ = _design(f.response, fixed_sigma, data)
    _ss_check_response(fam, y)
    sigma_spec = _gsl_structured_sigma_spec(f, what, sig_t.target, sig_t, sts, data, length(y),
                                            K, A, coords, tree, item -> _sd_target_block(item.target))
    # `_gsl_structured_sigma_spec` builds (gidx, G, Z, nm, grp, loading, K, block, kind).
    fit = _fit_sigma_fields_laplace(fam, y, Xμ, Xσ, [(sigma_spec[1:8]..., :intercept)],
                                    nmμ, nmσ, g_tol; se = se)
    return _withformula(fit, f)
end

# ---- Shape A: NB2 structured sigma intercept + one slope, fixed SDs (R fits) --

# Detect `sigma ~ … + provider(1 + x | g)` (unlabelled, one marker) with NO SD
# formula. Returns (kind, block) or nothing.
function _sigma_struct_q2_block(f::DrmFormula)
    sr = get(Dict(f.forms), :sigma, nothing)
    sr === nothing && return nothing
    isempty(_sd_target_parts(f)) && isempty(_sdphylo_parts(f)) || return nothing
    for (kind, marker) in ((:animal, animal), (:relmat, relmat), (:phylo, phylo), (:spatial, spatial))
        block = _animal_intercept_slope(sr; kind = marker)
        block === nothing || return (kind, block)
    end
    return nothing
end

function _drm_sigma_struct_q2_laplace(f::DrmFormula, fam::NegBinomial2, kind::Symbol, block;
                                      data, tree = nothing, K = nothing, A = nothing,
                                      coords = nothing, g_tol::Real = 1e-8, se::Bool = true,
                                      method = nothing)
    grp, slope = block.group, block.slope
    what = "NegBinomial2 `sigma ~ … + $kind(1 + $slope | $grp)` (fixed SDs)"
    (method === nothing || method === :ML) || throw(ArgumentError(
        "drm: $what supports maximum likelihood only (the marginal is a Laplace approximation)."))
    all(p -> first(p) in (:mu, :sigma), f.forms) || throw(ArgumentError(
        "drm: $what supports only `mu` and `sigma` formulas (no `zi`/`hu`/`sd`)."))
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    (mv === nothing && st === nothing && isempty(re)) || throw(ArgumentError(
        "drm: $what takes a fixed-effect mean (no mean random or structured terms)."))
    _has_offset_term(rhs[:mu]) && throw(ArgumentError("drm: $what does not support `offset(...)`."))
    fixed_sigma, sigma_re, mvs, sts, sslope, _, aslope = _split_ranef(rhs[:sigma];
        allow_animal_intercept_slope = true, allow_relmat_intercept_slope = true,
        allow_spatial_intercept_slope = true, allow_phylo_slope = true)
    (sts !== nothing && isempty(sigma_re) && mvs === nothing && sts[1] === kind && sts[2] === grp) ||
        throw(ArgumentError("drm: $what takes one structured term and no other sigma random effects."))
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "drm: $what requires observed responses; remove incomplete rows first (`drm_listwise`)."))
    hasproperty(data, slope) || throw(ArgumentError("drm: sigma slope predictor `$slope` is absent from data."))
    n = length(_table_column(data, f.response))
    raw = getproperty(data, slope)
    raw isa AbstractVector{<:Real} && length(raw) == n && all(isfinite, raw) || throw(ArgumentError(
        "drm: sigma slope predictor `$slope` must be finite numeric values, one per response."))
    any(!iszero, raw) || throw(ArgumentError(
        "drm: sigma slope predictor `$slope` must contain at least one nonzero value."))
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    _, Xσ, nmσ = _design(f.response, fixed_sigma, data)
    _ss_check_response(fam, y)
    C, gidx, G = _ss_kernel(kind, grp, data, tree, K, A, coords)
    size(C) == (G, G) || throw(ArgumentError("structured matrix must be $(G)×$(G)"))
    # Block names avoid the `sdtarget_` prefix: there is no `sd(...)` formula to resolve.
    blk(member) = Symbol("sigma_sd_", member)
    one_z = ones(G, 1)
    fields = [(gidx, G, one_z, ["log SD intercept"], grp, ones(length(y)), C, blk(:intercept), :intercept),
              (gidx, G, one_z, ["log SD $slope"], grp, Float64.(raw), C, blk(slope), slope)]
    fit = _fit_sigma_fields_laplace(fam, y, Xμ, Xσ, fields, nmμ, nmσ, g_tol; se = se)
    return _withformula(fit, f)
end

# ---- Post-fit helpers ----------------------------------------------------------

# Population design for predict_parameters(:sigma): drop a fitted unlabelled
# `provider(1 + x | g)` term (shape A); the remaining fixed part is the design.
function _strip_sigma_struct_q2_terms(fit, rhs)
    (fit.family isa NegBinomial2 && haskey(fit.scales, :sigma_random_effect_slope)) || return rhs
    terms = rhs isa Tuple ? collect(rhs) : Any[rhs]
    keep = Any[]
    for t in terms
        drop = false
        for marker in (animal, relmat, phylo, spatial)
            t isa FunctionTerm && t.f === marker || continue
            _animal_intercept_slope(t; kind = marker) === nothing || (drop = true)
        end
        drop || push!(keep, t)
    end
    length(keep) == length(terms) && return rhs
    isempty(keep) && return ConstantTerm(1)
    return length(keep) == 1 ? keep[1] : Tuple(keep)
end

# Marginal (fresh sigma-side random effects) simulator for shapes A and B:
# sigma_i* = exp(eta_fixed_i + sum_k load_k[i] s_k[g] (L_k z_k)[g]), then the
# family draw at the fitted mean.
function _sigma_struct_marginal_simulator(fit::DrmFit, data; K = nothing, A = nothing,
                                          tree = nothing, coords = nothing)
    f = fit.formula
    f isa DrmFormula && fit.family isa Union{NegBinomial2,Gamma} || return nothing
    haskey(fit.scales, :sigma_fixed) || return nothing
    sr = get(Dict(f.forms), :sigma, nothing)
    sr === nothing && return nothing
    targets = _sd_target_parts(f)
    fields = NamedTuple[]
    n = length(fit.scales[:sigma_fixed])
    if haskey(fit.scales, :sigma_random_effect_slope)
        q2 = _sigma_struct_q2_block(f)
        q2 === nothing && return nothing
        kind, block = q2
        C, gidx, G = _ss_kernel(kind, block.group, data, tree, K, A, coords)
        raw = Float64.(getproperty(data, block.slope))
        for (member, load) in ((:intercept, ones(n)), (block.slope, raw))
            key = Symbol("sigma_sd_", member)
            push!(fields, (; C, gidx, G, load, scale = fill(exp(only(coef(fit, key))), G)))
        end
    else
        length(targets) == 1 || return nothing
        t = only(targets)
        (_, _, _, sts) = _split_ranef(sr)
        sts === nothing && return nothing
        kind, grp = sts
        C, gidx, G = _ss_kernel(kind, grp, data, tree, K, A, coords)
        Zg, _ = _sd_group_design(f.response, t.rhs, data, gidx, G, grp; schema_cache = f.schema_cache,
                                 schema_key = t.key, store_schema = false)
        push!(fields, (; C, gidx, G, load = ones(n), scale = exp.(Zg * coef(fit, :sd))))
    end
    length(first(fields).gidx) == n || return nothing
    facs = [cholesky(Symmetric(fl.C)).L for fl in fields]
    eta_fixed = log.(fit.scales[:sigma_fixed])
    return function (rng)
        eta = copy(eta_fixed)
        for (fl, L) in zip(fields, facs)
            u = fl.scale .* (L * randn(rng, fl.G))
            eta .+= fl.load .* u[fl.gidx]
        end
        return _simulate_once(fit, rng; sigma = exp.(clamp.(eta, -15.0, 15.0)))
    end
end
