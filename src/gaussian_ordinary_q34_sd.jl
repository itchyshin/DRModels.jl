# Gaussian ordinary correlated block of three or four coefficients with
# coefficient-specific SD formulas (design: DESIGN-gaussian-q3q4.md):
#   y ~ x + z + (1 + x + z | g),  sd(g, coef(intercept)) ~ w, sd(g, coef(x)) ~ w, …
# b_g ~ N(0, D(w_g) R D(w_g)), exact marginal y_g ~ N(X_g β, S_g + U_g Σ_g U_g'),
# R from TMB's UNSTRUCTURED_CORR_t map (unit-diagonal lower L, row-major free
# entries, R = scale(L L')), so `theta` matches native drmTMB's `theta_re_cov`.
# Evaluated per group by the q×q Woodbury form: no n×n matrix.

# `(1 + x + z | g)` lhs → [:x, :z]; nothing when the lhs is not an
# intercept-plus-one-or-more-numeric-variable block (q = 2 is the ordinary
# `(1 + x | g)` case: the same machinery with q = 2). `_re_kind` stays q ≤ 2.
function _re_multi_corr_vars(lhs)
    (lhs isa FunctionTerm && lhs.f === (+)) || return nothing
    consts = filter(t -> t isa ConstantTerm, lhs.args)
    vars = filter(t -> t isa Term, lhs.args)
    (length(consts) == 1 && only(consts).n == 1 && length(vars) ≥ 1 &&
     length(vars) + 1 == length(lhs.args)) || return nothing
    return Symbol[v.sym for v in vars]
end

# Unit-diagonal lower factor L(θ) (TMB `UNSTRUCTURED_CORR_t`) → Cholesky factor
# of the correlation matrix, `Lc` with `Lc Lc' = R`.
function _q34_corr_chol(θ::AbstractVector{T}, q::Int) where {T}
    L = zeros(T, q, q)
    k = 0
    for i in 1:q
        L[i, i] = one(T)
        for j in 1:(i-1)
            k += 1
            L[i, j] = θ[k]
        end
    end
    Σ0 = L * L'
    d = [inv(sqrt(Σ0[i, i])) for i in 1:q]
    # Rows of L scaled by 1/√Σ0_ii give a factor of R (R = S^-1/2 L L' S^-1/2).
    return d .* L
end
function _q34_corr_from_theta(θ::AbstractVector, q::Int = Int((1 + sqrt(1 + 8 * length(θ))) / 2))
    Lc = _q34_corr_chol(Vector{Float64}(θ), q)
    return Lc * Lc'
end

function _drm_gaussian_ordinary_q34_sd(f::DrmFormula, fam::Gaussian, sdpp, re, multi,
                                       structured, structured_sigma, sigma_re, metav,
                                       has_missing_response, y, Xμ, Xσ, nmμ, nmσ, data,
                                       g_tol, method, algorithm)
    isempty(sdpp) && structured === nothing && structured_sigma === nothing &&
        isempty(sigma_re) && metav === nothing && length(re) == 1 || throw(ArgumentError(
        "drm: a Gaussian ordinary `(1 + x + z | g)` block with `sd()` formulas must be the only " *
        "random-effect term (no other ordinary, structured, `sigma` or `meta_V` terms)."))
    (_, grp) = re[1]
    vars = multi[1]
    q = 1 + length(vars)
    q in (2, 3, 4) || throw(ArgumentError(
        "drm: Gaussian `sd()` regression on a correlated block supports 2 to 4 coefficients " *
        "(got $q); use independent terms or a smaller block."))
    length(unique(vars)) == length(vars) || throw(ArgumentError(
        "drm: repeated coefficient in `(1 + $(join(vars, " + ")) | $grp)`."))
    members = Symbol[:intercept; vars]
    has_missing_response && throw(ArgumentError(
        "drm: Gaussian correlated-block SD regression requires observed responses; remove incomplete rows first."))
    algorithm === :auto || throw(ArgumentError(
        "drm: `algorithm = :$algorithm` is not implemented for Gaussian correlated-block SD regression."))
    targets = _sd_target_parts(f)
    reml = method === :REML
    reml && _gaussian_block_reml_gate(targets, Xμ, Xσ, "a Gaussian correlated block `(1 + … | $grp)`")
    all(item -> item.target.level === :iid && item.target.group === grp &&
        item.target.dpar === :mu && item.target.block === nothing, targets) ||
        throw(ArgumentError("drm: Gaussian correlated-block SD regression requires ordinary " *
            "mean targets `sd($grp, coef(…)) ~ …` on the block's grouping."))
    chosen = [item.target.member for item in targets]
    all(m -> m isa Symbol && m in members, chosen) || throw(ArgumentError(
        "drm: every `sd($grp, coef(…))` target in a correlated block must name a block " *
        "coefficient exactly (one of $(join(members, ", ")))."))
    length(unique(chosen)) == length(chosen) || throw(ArgumentError(
        "drm: each coefficient of the block takes at most one `sd($grp, coef(…))` formula."))
    allowed = Set(vcat([:mu, :sigma], [item.key for item in targets]))
    all(first(pair) in allowed for pair in f.forms) || throw(ArgumentError(
        "drm: Gaussian correlated-block SD regression supports only `mu`, `sigma` and the SD formulas."))
    gidx, G = _group_index(getproperty(data, grp); col = grp)
    n = length(y)
    U = ones(Float64, n, q)
    for (k, v) in enumerate(vars)
        _require_finite_data_column(data, v)
        U[:, k+1] = Float64.(getproperty(data, v))
    end
    Zgs = Matrix{Float64}[]; nmsds = Vector{String}[]; blocks = Symbol[]
    for m in members
        idx = findfirst(==(m), chosen)
        if idx === nothing
            push!(Zgs, ones(G, 1)); push!(nmsds, ["(Intercept)"])
            push!(blocks, _sd_target_block((group = grp, level = :iid, member = m,
                                            dpar = :mu, block = nothing); disambiguate = true))
        else
            item = targets[idx]
            Zg, nm = _sd_group_design(f.response, item.rhs, data, gidx, G, grp;
                                      schema_cache = f.schema_cache, schema_key = item.key)
            size(Zg, 2) ≤ G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
                "drm: Gaussian SD target `coef($m)` has a rank-deficient group-level design."))
            push!(Zgs, Matrix{Float64}(Zg)); push!(nmsds, nm)
            push!(blocks, _sd_target_block(item.target; disambiguate = true))
        end
    end
    return _fit_gaussian_ordinary_q34_sd(fam, y, Xμ, Xσ, U, Zgs, gidx, G, nmμ, nmσ,
                                         nmsds, blocks, members, grp, g_tol; reml)
end

# Shared REML admission rule for the Gaussian correlated-block routes (single block q = 2..4,
# and a block beside crossed intercepts). The restriction term is exact for every shape here
# (the marginal is Gaussian-linear in `beta_mu`), but drmTMB 0.8.1 admits REML on these blocks
# only without `sd()` formulas and with a constant residual scale (measured on the 3008-cell
# gap map, 2026-11-09: every `sd-one` / `sd-z` / `sd-first` REML cell of these shapes is
# refused by R), so the same two cases stay refused here rather than ship an untested route.
function _gaussian_block_reml_gate(targets, Xμ, Xσ, what::AbstractString)
    isempty(targets) || throw(ArgumentError(
        "drm: method = :REML is not implemented for $what together with `sd()` formulas: " *
        "drmTMB refuses this combination too, and no restricted objective is validated for it. " *
        "Use method = :ML, or drop the `sd()` formulas for REML."))
    size(Xσ, 2) == 1 || throw(ArgumentError(
        "drm: method = :REML for $what requires a constant residual scale (`sigma ~ 1`); " *
        "a `sigma` predictor is not validated against drmTMB there. Use `sigma ~ 1` or method = :ML."))
    rank(Xμ) == size(Xμ, 2) || throw(ArgumentError(
        "drm: method = :REML requires a full-rank `mu` fixed-effect design."))
    return nothing
end

function _fit_gaussian_ordinary_q34_sd(fam::Gaussian, y, Xμ, Xσ, U, Zgs, gidx, G,
                                       nmμ, nmσ, nmsds, blocks, members, grp, g_tol;
                                       reml::Bool = false)
    n = length(y); q = size(U, 2)
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    ps = size.(Zgs, 2)
    rg = UnitRange{Int}[]
    off = pμ + pσ
    for p in ps
        push!(rg, (off+1):(off+p)); off += p
    end
    npair = q * (q - 1) ÷ 2
    rθ = (off+1):(off+npair)
    ntot = off + npair
    n ≥ ntot || throw(ArgumentError("drm: only $n observations for $ntot parameters in the " *
        "Gaussian correlated-block SD model."))
    rows = [Int[] for _ in 1:G]
    for i in 1:n
        push!(rows[gidx[i]], i)
    end
    const_2pi = 0.5 * n * log(2π)
    function nll(θ)
        T = eltype(θ)
        ημ = Xμ * θ[1:pμ]; ησ = Xσ * θ[pμ+1:pμ+pσ]
        Lc = _q34_corr_chol(θ[rθ], q)
        sdg = [exp.(Zgs[k] * θ[rg[k]]) for k in 1:q]
        tot = zero(T)
        for g in 1:G
            rg_ = rows[g]
            A = zeros(T, q, q); b = zeros(T, q)
            ldS = zero(T); rr = zero(T)
            for i in rg_
                si = exp(-2 * ησ[i]); ri = y[i] - ημ[i]
                ldS += 2 * ησ[i]; rr += si * ri * ri
                for a in 1:q
                    ua = U[i, a] * si
                    b[a] += ua * ri
                    for c in 1:q
                        A[a, c] += ua * U[i, c]
                    end
                end
            end
            L = Matrix{T}(undef, q, q)
            for a in 1:q, c in 1:q
                L[a, c] = sdg[a][g] * Lc[a, c]
            end
            M = Matrix{T}(I, q, q) + L' * A * L
            chM = cholesky(Symmetric(M); check = false)
            issuccess(chM) || return convert(T, 1e18)
            v = L' * b
            tot += 0.5 * (ldS + 2 * sum(log, diag(chM.L)) + rr - dot(v, chM \ v))
        end
        return tot + const_2pi
    end
    # REML: nll_ML + ½ logdet(Xμ′V⁻¹Xμ) − ½ pμ log(2π), V = ⊕_g (S_g + (U_g L_g)(U_g L_g)′).
    # X′V⁻¹X is the sum over groups of the stacked-QR restriction of Z̃_g = U_g L_g
    # (`_reml_xtvinvx_qr`): PSD by construction, no Woodbury subtraction.
    function nll_reml(θ)
        val = nll(θ)
        val >= 1e18 && return val
        T = eltype(θ)
        ησ = Xσ * θ[pμ+1:pμ+pσ]
        Lc = _q34_corr_chol(θ[rθ], q)
        sdg = [exp.(Zgs[k] * θ[rg[k]]) for k in 1:q]
        XtVinvX = zeros(T, pμ, pμ)
        for g in 1:G
            r = rows[g]
            isempty(r) && continue
            L = Matrix{T}(undef, q, q)
            for a in 1:q, c in 1:q
                L[a, c] = sdg[a][g] * Lc[a, c]
            end
            XtVinvX .+= _reml_xtvinvx_qr(U[r, :] * L, exp.(-ησ[r]), Xμ[r, :])
        end
        return _reml_add_restriction(val, Symmetric(XtVinvX), pμ)
    end
    nll_opt = reml ? nll_reml : nll
    βμ0 = Xμ \ y; res0 = y - Xμ * βμ0
    θ0 = zeros(ntot)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(res0) / sqrt(q) + eps())
    s0 = std(res0) / sqrt(q)
    for k in 1:q
        scale_k = s0 / max(sqrt(sum(abs2, @view U[:, k]) / n), eps())
        θ0[rg[k]] .= Zgs[k] \ fill(log(scale_k + eps()), G)
    end
    # The first L-BFGS step is the raw gradient (`InitialStatic`, length ~ 1e2-1e3 here,
    # because the objective is a sum over n rows), so the line search probes points with
    # `sigma` or the SD coefficients of order 1e2: `exp(-2 eta_sigma)` and the group
    # information matrix overflow, the Cholesky of an Inf/NaN matrix "succeeds", and the
    # objective or its AD gradient is NaN. HagerZhang then asserts
    # `isfinite(phi_c) && isfinite(dphi_c)` and the fit is lost (36/300 seeds of the
    # scale-300 campaign on Linux/x86; the same data fit on macOS/aarch64 only because the
    # probe happened to hit the non-PD sentinel instead). Map non-finite probes to the 1e18
    # sentinel inside the optimiser only (probes with a finite value and gradient are
    # untouched, so fits that already succeeded follow the identical path), and retry with
    # BackTracking if the default line search still asserts. The reported value, Hessian
    # and gradient check below use the raw `nll`.
    res = _optimize_with_fallback(nll_opt, θ0, g_tol;
        opts = Optim.Options(g_tol = g_tol, x_abstol = NaN, x_reltol = NaN,
                             f_abstol = NaN, f_reltol = NaN))
    θ̂ = Optim.minimizer(res)
    H = ForwardDiff.hessian(nll_opt, θ̂)
    V = _vcov_from_hessian(H)
    converged = !Optim.iteration_limit_reached(res) &&
        _unitfree_converged(H, ForwardDiff.gradient(nll_opt, θ̂))
    block_list = Pair{Symbol,UnitRange{Int}}[:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ)]
    name_list = Pair{Symbol,Vector{String}}[:mu => nmμ, :sigma => nmσ]
    scales = Dict{Symbol,Vector{Float64}}(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    for k in 1:q
        push!(block_list, blocks[k] => rg[k]); push!(name_list, blocks[k] => nmsds[k])
        sdk = exp.(Zgs[k] * θ̂[rg[k]])
        scales[blocks[k]] = sdk[gidx]
        scales[Symbol(string(blocks[k]), "_group")] = sdk
    end
    # Conditional modes at θ̂ (closed form: the model is Gaussian). Group g has
    # b_g = L z, z ~ N(0, I), L = diag(sd_g) Lc, so E[z | y_g] = M⁻¹ L'(U' diag(1/σ²) r)
    # with M = I + L' A L, and the row contribution is U_i' L ẑ_g. Recorded as the
    # per-coefficient unit-scale effects (Lc ẑ_g)_k (effect = sd_k[g] · mode_k[g], mode ~ N(0, R)).
    let μhat = Xμ * θ̂[1:pμ], σhat = exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]),
        Lc = _q34_corr_chol(θ̂[rθ], q), sdg = [exp.(Zgs[k] * θ̂[rg[k]]) for k in 1:q]
        eta_re = zeros(n); modes = [zeros(G) for _ in 1:q]
        for g in 1:G
            A = zeros(q, q); b = zeros(q)
            for i in rows[g]
                si = 1 / σhat[i]^2; ri = y[i] - μhat[i]
                for a in 1:q
                    b[a] += U[i, a] * si * ri
                    for c in 1:q
                        A[a, c] += U[i, a] * si * U[i, c]
                    end
                end
            end
            L = [sdg[a][g] * Lc[a, c] for a in 1:q, c in 1:q]
            zhat = (I + L' * A * L) \ (L' * b)
            bg = L * zhat
            for a in 1:q
                modes[a][g] = (Lc * zhat)[a]
            end
            for i in rows[g]
                eta_re[i] = dot(view(U, i, :), bg)
            end
        end
        _attach_conditional_structured!(scales, eta_re, modes...)
    end
    names_θ = String[]
    for i in 1:q, j in 1:(i-1)
        push!(names_θ, "theta($(members[i]), $(members[j]))")
    end
    push!(block_list, :recov => rθ); push!(name_list, :recov => names_θ)
    fit = DrmFit(fam, block_list, name_list, θ̂, V, -nll_opt(θ̂), n, converged,
                 Dict(:mu => Xμ * θ̂[1:pμ]), Dict(:mu => Vector{Float64}(y)), scales)
    fit = _withiterations(_withnll(fit, nll_opt), Optim.iterations(res))
    reml && return _withreml(fit, -nll_reml(θ̂), -nll(θ̂))
    return fit
end
