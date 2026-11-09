# Gaussian ordinary random effects that the single-block and single-component
# routes refuse (design: docs/dev-log/2026-11-09-gaussian-sd-gaps.md):
#
#   1. a correlated block `(1 + x + .. | g)` (q = 2, 3, 4) beside one or more
#      random intercepts on OTHER groupings, `(1 + x | g) + (1 | g2)`, with an
#      optional `sd(g, coef(·)) ~ …` per block coefficient and `sd(g2) ~ …` per
#      extra intercept;
#   2. independent intercept and slope terms on one grouping,
#      `(1 | g) + (0 + x | g)`, with `sd(g, coef(·)) ~ …` per term.
#
# Both are exact dense maximum likelihood of the Gaussian marginal
#   y ~ N(X β, diag(σ²) + Σ_blocks U_g Σ_g U_g' + Σ_extra Z S Z'),
# the same engine family as `_fit_gaussian_lss_multi` (case 2 reuses it with a
# per-row slope loading) and `_fit_gaussian_ordinary_q34_sd` (case 1 reuses its
# correlation map, so `theta` matches native drmTMB's `theta_re_cov`). No new
# estimation algorithm: a dense n×n Cholesky and ForwardDiff, capped at 5000 rows.

# Position of the one correlated block and of the extra random intercepts, or
# `nothing` when `re` is not `(block) + (1 | h) + …` on distinct groupings.
function _corr_crossed_split(re)
    length(re) ≥ 2 || return nothing
    bi = findall(r -> _re_multi_corr_vars(r[1]) !== nothing, re)
    length(bi) == 1 || return nothing
    others = setdiff(collect(1:length(re)), bi)
    all(i -> re[i][1] isa ConstantTerm && re[i][1].n == 1, others) || return nothing
    allunique(Symbol[g for (_, g) in re]) || return nothing
    return bi[1], others
end

function _drm_gaussian_corr_crossed(f::DrmFormula, fam::Gaussian, split, re, sdpp, sdp,
                                    structured, structured_sigma, sigma_re, metav,
                                    has_missing_response, y, Xμ, Xσ, nmμ, nmσ, data,
                                    g_tol, method, algorithm)
    (isempty(sdpp) && structured === nothing && structured_sigma === nothing &&
     isempty(sigma_re) && metav === nothing) || throw(ArgumentError(
        "drm: a Gaussian correlated block beside other random intercepts supports ordinary " *
        "mean effects only (no structured, `sigma` or `meta_V` terms)."))
    has_missing_response && throw(ArgumentError(
        "drm: a Gaussian correlated block beside other random intercepts requires observed " *
        "responses; remove incomplete rows first."))
    algorithm === :auto || throw(ArgumentError(
        "drm: `algorithm = :$algorithm` is not implemented for a Gaussian correlated block " *
        "beside other random intercepts (dense exact marginal only)."))
    reml = method === :REML
    reml && _gaussian_block_reml_gate(_sd_target_parts(f), Xμ, Xσ,
        "a Gaussian correlated block beside other random intercepts")
    bi, others = split
    (_, grp) = re[bi]
    vars = _re_multi_corr_vars(re[bi][1])
    q = 1 + length(vars)
    q in (2, 3, 4) || throw(ArgumentError(
        "drm: a correlated block supports 2 to 4 coefficients (got $q)."))
    length(unique(vars)) == length(vars) || throw(ArgumentError(
        "drm: repeated coefficient in `(1 + $(join(vars, " + ")) | $grp)`."))
    members = Symbol[:intercept; vars]
    other_grps = Symbol[re[i][2] for i in others]
    targets = _sd_target_parts(f)
    all(item -> item.target.level === :iid && item.target.dpar === :mu &&
        item.target.block === nothing, targets) || throw(ArgumentError(
        "drm: with a correlated block beside other random intercepts, `sd()` targets must be " *
        "ordinary mean targets."))
    allowed = Set(vcat([:mu, :sigma], [item.key for item in targets]))
    all(first(pair) in allowed for pair in f.forms) || throw(ArgumentError(
        "drm: this route supports only `mu`, `sigma` and the SD formulas."))
    block_targets = [item for item in targets if item.target.group === grp]
    all(item -> item.target.member isa Symbol && item.target.member in members, block_targets) ||
        throw(ArgumentError("drm: every `sd($grp, coef(…))` target on the correlated block must " *
            "name one of its coefficients ($(join(members, ", ")))."))
    chosen = [item.target.member for item in block_targets]
    length(unique(chosen)) == length(chosen) || throw(ArgumentError(
        "drm: each coefficient of the block takes at most one `sd($grp, coef(…))` formula."))
    for item in targets
        item.target.group === grp || item.target.group in other_grps || throw(ArgumentError(
            "drm: `sd($(item.target.group))` names no random effect in the mean formula."))
        (item.target.group in other_grps &&
         !(item.target.member === nothing || item.target.member === :intercept)) &&
            throw(ArgumentError("drm: `sd($(item.target.group), coef(…))` on a random intercept " *
                "must be plain `sd($(item.target.group))` or `coef(intercept)`."))
    end
    n = length(y)
    gidx, G = _group_index(getproperty(data, grp); col = grp)
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
            item = block_targets[idx]
            Zg, nm = _sd_group_design(f.response, item.rhs, data, gidx, G, grp;
                                      schema_cache = f.schema_cache, schema_key = item.key)
            size(Zg, 2) ≤ G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
                "drm: Gaussian SD target `coef($m)` has a rank-deficient group-level design."))
            push!(Zgs, Matrix{Float64}(Zg)); push!(nmsds, nm)
            push!(blocks, _sd_target_block(item.target; disambiguate = true))
        end
    end
    extras = NamedTuple[]
    for h in other_grps
        gi2, G2 = _group_index(getproperty(data, h); col = h)
        hit = [item for item in targets if item.target.group === h]
        length(hit) ≤ 1 || throw(ArgumentError("drm: one `sd($h) ~ …` formula per random intercept."))
        if isempty(hit)
            Zg = ones(G2, 1); nm = ["(Intercept)"]
            blk = _sd_target_block((group = h, level = :iid, member = nothing,
                                    dpar = :mu, block = nothing); disambiguate = true)
        else
            Zg, nm = _sd_group_design(f.response, hit[1].rhs, data, gi2, G2, h;
                                      schema_cache = f.schema_cache, schema_key = hit[1].key)
            size(Zg, 2) ≤ G2 && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
                "drm: SD target `sd($h)` has a rank-deficient group-level design."))
            blk = _sd_target_block(hit[1].target; disambiguate = true)
        end
        push!(extras, (; grp = h, gidx = gi2, G = G2, Zg = Matrix{Float64}(Zg),
                       nm = Vector{String}(nm), block = blk))
    end
    return _fit_gaussian_corr_crossed(fam, y, Xμ, Xσ, U, Zgs, gidx, G, extras, nmμ, nmσ,
                                      nmsds, blocks, members, grp, g_tol; reml)
end

function _fit_gaussian_corr_crossed(fam::Gaussian, y, Xμ, Xσ, U, Zgs, gidx, G, extras,
                                    nmμ, nmσ, nmsds, blocks, members, grp, g_tol;
                                    reml::Bool = false)
    n = length(y); q = size(U, 2)
    n ≤ 5000 || throw(ArgumentError("drm: a correlated block beside other random intercepts " *
        "assembles a dense $(n)×$(n) marginal covariance and is limited to 5000 rows."))
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    off = pμ + pσ
    rg = UnitRange{Int}[]
    for Zg in Zgs
        push!(rg, (off+1):(off+size(Zg, 2))); off += size(Zg, 2)
    end
    npair = q * (q - 1) ÷ 2
    rθ = (off+1):(off+npair); off += npair
    re_ = UnitRange{Int}[]
    for e in extras
        push!(re_, (off+1):(off+size(e.Zg, 2))); off += size(e.Zg, 2)
    end
    ntot = off
    n ≥ ntot || throw(ArgumentError("drm: only $n observations for $ntot parameters."))
    rows = [Int[] for _ in 1:G]
    for i in 1:n
        push!(rows[gidx[i]], i)
    end
    erows = [begin
                 r = [Int[] for _ in 1:e.G]
                 for i in 1:n
                     push!(r[e.gidx[i]], i)
                 end
                 r
             end for e in extras]
    const_2pi = 0.5 * n * log(2π)
    const_pμ = 0.5 * pμ * log(2π)
    # One dense objective for ML and REML. REML (Patterson-Thompson; exact because the marginal
    # is Gaussian-linear in beta_mu) adds ½ logdet(Xμ′V⁻¹Xμ) − ½ pμ log(2π); with the Cholesky
    # V = L L′, X′V⁻¹X = (L⁻¹X)′(L⁻¹X) is PSD by construction.
    function nll_full(θ, restricted::Bool)
        T = eltype(θ)
        ημ = Xμ * θ[1:pμ]; ησ = Xσ * θ[pμ+1:pμ+pσ]
        Lc = _q34_corr_chol(θ[rθ], q)
        sdg = [exp.(Zgs[k] * θ[rg[k]]) for k in 1:q]
        Vm = zeros(T, n, n)
        for g in 1:G
            r = rows[g]
            isempty(r) && continue
            L = Matrix{T}(undef, q, q)
            for a in 1:q, c in 1:q
                L[a, c] = sdg[a][g] * Lc[a, c]
            end
            UL = U[r, :] * L
            Vm[r, r] .+= UL * UL'
        end
        for (ei, e) in enumerate(extras)
            s2 = exp.(2 .* (e.Zg * θ[re_[ei]]))
            for h in 1:e.G
                r = erows[ei][h]
                isempty(r) && continue
                Vm[r, r] .+= s2[h]
            end
        end
        @inbounds for i in 1:n
            Vm[i, i] += exp(2 * ησ[i])
        end
        Vfac = cholesky(Symmetric(Vm); check = false)
        issuccess(Vfac) || return convert(T, 1e18)
        rr = y .- ημ
        val = 0.5 * (logdet(Vfac) + dot(rr, Vfac \ rr)) + const_2pi
        restricted || return val
        W = Vfac.L \ Xμ
        return _reml_add_restriction(val, Symmetric(W' * W), pμ)
    end
    nll(θ) = nll_full(θ, reml)
    nll_ml(θ) = nll_full(θ, false)
    βμ0 = Xμ \ y; res0 = y - Xμ * βμ0
    ncomp = q + length(extras)
    θ0 = zeros(ntot)
    θ0[1:pμ] .= βμ0
    s0 = std(res0) / sqrt(ncomp + 1)
    θ0[pμ+1] = log(s0 + eps())
    for k in 1:q
        scale_k = s0 / max(sqrt(sum(abs2, @view U[:, k]) / n), eps())
        θ0[rg[k]] .= Zgs[k] \ fill(log(scale_k + eps()), G)
    end
    for (ei, e) in enumerate(extras)
        θ0[re_[ei]] .= e.Zg \ fill(log(s0 + eps()), e.G)
    end
    opt = _optimize_with_fallback(nll, θ0, g_tol;
        opts = Optim.Options(g_tol = g_tol, x_abstol = NaN, x_reltol = NaN,
                             f_abstol = NaN, f_reltol = NaN))
    θ̂ = Optim.minimizer(opt)
    H = ForwardDiff.hessian(nll, θ̂)
    V = _vcov_from_hessian(H)
    converged = !Optim.iteration_limit_reached(opt) &&
        _unitfree_converged(H, ForwardDiff.gradient(nll, θ̂))
    block_list = Pair{Symbol,UnitRange{Int}}[:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ)]
    name_list = Pair{Symbol,Vector{String}}[:mu => nmμ, :sigma => nmσ]
    scales = Dict{Symbol,Vector{Float64}}(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    for k in 1:q
        push!(block_list, blocks[k] => rg[k]); push!(name_list, blocks[k] => nmsds[k])
    end
    names_θ = String[]
    for i in 1:q, j in 1:(i-1)
        push!(names_θ, "theta($(members[i]), $(members[j]))")
    end
    push!(block_list, :recov => rθ); push!(name_list, :recov => names_θ)
    for (ei, e) in enumerate(extras)
        push!(block_list, e.block => re_[ei]); push!(name_list, e.block => e.nm)
    end
    fit = DrmFit(fam, block_list, name_list, θ̂, V, -nll(θ̂), n, converged,
                 Dict(:mu => Xμ * θ̂[1:pμ]), Dict(:mu => Vector{Float64}(y)), scales)
    fit = _withiterations(_withnll(fit, nll), Optim.iterations(opt))
    reml && return _withreml(fit, -nll(θ̂), -nll_ml(θ̂))
    return fit
end

# Independent intercept and slope terms, `(1 | g) + (0 + x | g) [+ (0 + z | g)]`, each with
# at most one `sd(g, coef(·)) ~ …` formula. Components are `_LssComp`s of the dense
# multi-component engine; a slope term carries its row loading in `w`.
function _drm_gaussian_indep_sd(f::DrmFormula, fam::Gaussian, re, re_kinds, sdpp,
                                structured, structured_sigma, sigma_re, metav,
                                has_missing_response, y, Xμ, Xσ, nmμ, nmσ, data,
                                g_tol, method, algorithm)
    (isempty(sdpp) && structured === nothing && structured_sigma === nothing &&
     isempty(sigma_re) && metav === nothing) || throw(ArgumentError(
        "drm: independent intercept and slope terms with `sd()` formulas support ordinary " *
        "mean effects only (no structured, `sigma` or `meta_V` terms)."))
    has_missing_response && throw(ArgumentError(
        "drm: independent intercept and slope terms with `sd()` formulas require observed " *
        "responses; remove incomplete rows first."))
    method === :REML && throw(ArgumentError(
        "drm: independent intercept and slope terms with `sd()` formulas support maximum " *
        "likelihood only (`method = :REML` is not implemented for this shape)."))
    algorithm === :auto || throw(ArgumentError(
        "drm: `algorithm = :$algorithm` is not implemented for independent intercept and " *
        "slope terms with `sd()` formulas (dense exact marginal only)."))
    targets = _sd_target_parts(f)
    all(item -> item.target.level === :iid && item.target.dpar === :mu &&
        item.target.block === nothing, targets) || throw(ArgumentError(
        "drm: independent intercept and slope terms take ordinary mean `sd()` targets only."))
    allowed = Set(vcat([:mu, :sigma], [item.key for item in targets]))
    all(first(pair) in allowed for pair in f.forms) || throw(ArgumentError(
        "drm: this route supports only `mu`, `sigma` and the SD formulas."))
    # member of each term: :intercept or the slope variable
    mem = Symbol[k === :intercept ? :intercept : v for (k, v) in re_kinds]
    ids = [(re[i][2], mem[i]) for i in eachindex(re)]
    allunique(ids) || throw(ArgumentError(
        "drm: repeated random-effect term on one grouping in the mean formula."))
    n = length(y)
    used = Set{Int}()
    comps = _LssComp[]
    for i in eachindex(re)
        grp = re[i][2]; m = mem[i]
        gidx, G = _group_index(getproperty(data, grp); col = grp)
        hit = findall(targets) do item
            t = item.target
            t.group === grp && (t.member === m ||
                (t.member === nothing && m === :intercept))
        end
        length(hit) ≤ 1 || throw(ArgumentError(
            "drm: one `sd($grp, coef($m)) ~ …` formula per term."))
        if isempty(hit)
            Zg = ones(G, 1); nm = ["(Intercept)"]
        else
            item = targets[hit[1]]
            push!(used, hit[1])
            Zg, nm = _sd_group_design(f.response, item.rhs, data, gidx, G, grp;
                                      schema_cache = f.schema_cache, schema_key = item.key)
            size(Zg, 2) ≤ G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
                "drm: SD target for `$m` has a rank-deficient group-level design."))
        end
        w = nothing
        if m !== :intercept
            _require_finite_data_column(data, m)
            w = Float64.(getproperty(data, m))
        end
        label = string(grp, m === :intercept ? "" : string(" ", m))
        push!(comps, _LssComp(gidx, G, Matrix{Float64}(Zg), Vector{String}(nm), nothing, label, w))
    end
    length(used) == length(targets) || throw(ArgumentError(
        "drm: an `sd()` target names a coefficient that is not a random-effect term in the " *
        "mean formula (terms: " * join(string.(ids), ", ") * ")."))
    return _fit_gaussian_lss_multi(fam, y, Xμ, Xσ, comps, nmμ, nmσ, g_tol; reml = false)
end
