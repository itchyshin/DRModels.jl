# Gaussian ordinary random effects that the single-block and single-component
# routes refuse (design: docs/dev-log/2026-11-09-gaussian-sd-gaps.md and
# docs/dev-log/2026-11-09-gaussian-sd-groupf.md):
#
#   1. one or more correlated blocks `(1 + x | g)` (q = 2, 3, 4), each on its own
#      grouping, beside independent random terms on any grouping: intercepts
#      `(1 | h)` and slopes `(0 + z | h)`, e.g. `(1 + x | g) + (1 | h) + (0 + z | h)`,
#      with an optional `sd(g, coef(·)) ~ …` per block coefficient and
#      `sd(h[, coef(·)]) ~ …` per independent term;
#   2. independent intercept and slope terms on one grouping,
#      `(1 | g) + (0 + x | g)`, with `sd(g, coef(·)) ~ …` per term.
#
# Both are exact dense maximum likelihood of the Gaussian marginal
#   y ~ N(X β, diag(σ²) + Σ_blocks U_g Σ_g U_g' + Σ_extra w w' ∘ (Z S Z')),
# the same engine family as `_fit_gaussian_lss_multi` (case 2 reuses it with a
# per-row slope loading) and `_fit_gaussian_ordinary_q34_sd` (case 1 reuses its
# correlation map, so `theta` matches native drmTMB's `theta_re_cov`). No new
# estimation algorithm: a dense n×n Cholesky and ForwardDiff, capped at 5000 rows.

# Kind of one independent term's lhs: `(:intercept, nothing)` for `1`, `(:slope, x)`
# for `0 + x`, `nothing` for anything else (a block, a label, an interaction).
function _cc_term_kind(lhs)
    lhs isa ConstantTerm && lhs.n == 1 && return (:intercept, nothing)
    if lhs isa FunctionTerm && lhs.f === (+) && length(lhs.args) == 2
        consts = filter(t -> t isa ConstantTerm, lhs.args)
        vars = filter(t -> t isa Term, lhs.args)
        length(consts) == 1 && length(vars) == 1 && only(consts).n == 0 &&
            return (:slope, only(vars).sym)
    end
    return nothing
end

# Positions of the correlated blocks and of the independent terms, or `nothing`
# when `re` is not "at least one block plus independent intercepts / slopes".
function _corr_crossed_split(re)
    length(re) ≥ 2 || return nothing
    bi = findall(r -> _re_multi_corr_vars(r[1]) !== nothing, re)
    isempty(bi) && return nothing
    others = setdiff(collect(1:length(re)), bi)
    (isempty(others) && length(bi) < 2) && return nothing     # a lone block keeps its own route
    all(i -> _cc_term_kind(re[i][1]) !== nothing, others) || return nothing
    return bi, others
end

function _drm_gaussian_corr_crossed(f::DrmFormula, fam::Gaussian, split, re, sdpp, sdp,
                                    structured, structured_sigma, sigma_re, metav,
                                    has_missing_response, y, Xμ, Xσ, nmμ, nmσ, data,
                                    g_tol, method, algorithm)
    (isempty(sdpp) && structured === nothing && structured_sigma === nothing &&
     isempty(sigma_re) && metav === nothing) || throw(ArgumentError(
        "drm: a Gaussian correlated block beside other random terms supports ordinary " *
        "mean effects only (no structured, `sigma` or `meta_V` terms)."))
    has_missing_response && throw(ArgumentError(
        "drm: a Gaussian correlated block beside other random terms requires observed " *
        "responses; remove incomplete rows first."))
    algorithm === :auto || throw(ArgumentError(
        "drm: `algorithm = :$algorithm` is not implemented for a Gaussian correlated block " *
        "beside other random terms (dense exact marginal only)."))
    reml = method === :REML
    # REML was checked against drmTMB for one block beside random INTERCEPTS only; every wider
    # shape (two blocks, a block beside a slope term) keeps the maximum-likelihood-only refusal.
    reml && (length(split[1]) == 1 && all(i -> _cc_term_kind(re[i][1])[1] === :intercept, split[2]) ||
        throw(ArgumentError(
            "drm: a Gaussian correlated block beside other random terms supports maximum " *
            "likelihood only: method = :REML is not implemented for this shape (REML is available " *
            "for one correlated block beside random INTERCEPTS on other groupings).")))
    reml && _gaussian_block_reml_gate(_sd_target_parts(f), Xμ, Xσ,
        "a Gaussian correlated block beside other random intercepts")
    bis, others = split
    block_grps = Symbol[re[b][2] for b in bis]
    dup = [g for g in unique(block_grps) if count(==(g), block_grps) > 1]
    isempty(dup) || throw(ArgumentError(
        "drm: two correlated blocks on the grouping `$(first(dup))` are not supported; put every " *
        "coefficient of a grouping in one `(1 + x + … | $(first(dup)))` block."))
    targets = _sd_target_parts(f)
    all(item -> item.target.level === :iid && item.target.dpar === :mu &&
        item.target.block === nothing, targets) || throw(ArgumentError(
        "drm: with a correlated block beside other random terms, `sd()` targets must be " *
        "ordinary mean targets."))
    allowed = Set(vcat([:mu, :sigma], [item.key for item in targets]))
    all(first(pair) in allowed for pair in f.forms) || throw(ArgumentError(
        "drm: this route supports only `mu`, `sigma` and the SD formulas."))
    n = length(y)
    # independent terms: (grouping, :intercept | slope variable)
    ext = [(grp = re[i][2], mem = let k = _cc_term_kind(re[i][1]); k[1] === :intercept ? :intercept : k[2] end)
           for i in others]
    allunique([(e.grp, e.mem) for e in ext]) || throw(ArgumentError(
        "drm: repeated random-effect term on one grouping in the mean formula."))
    # block membership, to refuse overlaps and to match SD targets
    blkinfo = map(bis) do b
        vars = _re_multi_corr_vars(re[b][1])
        q = 1 + length(vars)
        q in (2, 3, 4) || throw(ArgumentError(
            "drm: a correlated block supports 2 to 4 coefficients (got $q)."))
        length(unique(vars)) == length(vars) || throw(ArgumentError(
            "drm: repeated coefficient in `(1 + $(join(vars, " + ")) | $(re[b][2]))`."))
        (grp = re[b][2], members = Symbol[:intercept; vars], q = q)
    end
    # Verified against drmTMB 0.8.1 for 2-coefficient blocks only (docs/dev-log/2026-11-09-gaussian-sd-groupf.md):
    # two blocks, or a block beside a slope term, with a larger block are refused by name.
    length(blkinfo) ≥ 2 && any(bk -> bk.q != 2, blkinfo) && throw(ArgumentError(
        "drm: more than one correlated block is supported for 2-coefficient blocks `(1 + x | g)` only " *
        "(checked against drmTMB for q = 2; larger blocks beside another block are not)."))
    any(e -> e.mem !== :intercept, ext) && any(bk -> bk.q ≥ 3, blkinfo) && throw(ArgumentError(
        "drm: a correlated block of 3 or 4 coefficients beside a random slope term is not " *
        "implemented: drmTMB's own fit of this shape did not converge on the check data " *
        "(docs/dev-log/2026-11-09-gaussian-sd-groupf.md). Keep it beside random intercepts, or use " *
        "a 2-coefficient block."))
    for e in ext, bk in blkinfo
        e.grp === bk.grp && e.mem in bk.members && throw(ArgumentError(
            "drm: overlapping random-effect terms on `$(e.grp)`: the coefficient `$(e.mem)` is in the " *
            "correlated block and also a separate term. Use either the block or separate " *
            "independent terms for it (drmTMB refuses the same overlap)."))
    end
    # R drmTMB refuses an sd() formula on a grouping that carries a block AND a separate term
    for bk in blkinfo
        shared = any(e -> e.grp === bk.grp, ext)
        shared && any(item -> item.target.group === bk.grp, targets) && throw(ArgumentError(
            "drm: an `sd($(bk.grp), …)` formula on a grouping that carries a correlated block and " *
            "another random-effect term is not implemented (drmTMB refuses it: \"cannot yet " *
            "combine another random-effect coefficient on the same group\")."))
    end
    used = Set{Int}()
    blocks_ = NamedTuple[]
    for bk in blkinfo
        grp = bk.grp; q = bk.q; members = bk.members
        hit_idx = [k for (k, item) in enumerate(targets) if item.target.group === grp]
        block_targets = [targets[k] for k in hit_idx]
        all(item -> item.target.member isa Symbol && item.target.member in members, block_targets) ||
            throw(ArgumentError("drm: every `sd($grp, coef(…))` target on the correlated block must " *
                "name one of its coefficients ($(join(members, ", ")))."))
        union!(used, hit_idx)
        chosen = [item.target.member for item in block_targets]
        length(unique(chosen)) == length(chosen) || throw(ArgumentError(
            "drm: each coefficient of the block takes at most one `sd($grp, coef(…))` formula."))
        gidx, G = _group_index(getproperty(data, grp); col = grp)
        U = ones(Float64, n, q)
        for (k, v) in enumerate(members[2:end])
            _require_finite_data_column(data, v)
            U[:, k+1] = Float64.(getproperty(data, v))
        end
        Zgs = Matrix{Float64}[]; nmsds = Vector{String}[]; keys_ = Symbol[]
        for m in members
            idx = findfirst(==(m), chosen)
            if idx === nothing
                push!(Zgs, ones(G, 1)); push!(nmsds, ["(Intercept)"])
                push!(keys_, _sd_target_block((group = grp, level = :iid, member = m,
                                               dpar = :mu, block = nothing); disambiguate = true))
            else
                item = block_targets[idx]
                Zg, nm = _sd_group_design(f.response, item.rhs, data, gidx, G, grp;
                                          schema_cache = f.schema_cache, schema_key = item.key)
                size(Zg, 2) ≤ G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
                    "drm: Gaussian SD target `coef($m)` has a rank-deficient group-level design."))
                push!(Zgs, Matrix{Float64}(Zg)); push!(nmsds, nm)
                push!(keys_, _sd_target_block(item.target; disambiguate = true))
            end
        end
        push!(blocks_, (; grp, q, members, U, Zgs, gidx, G, nmsds, keys = keys_))
    end
    extras = NamedTuple[]
    for e in ext
        gi2, G2 = _group_index(getproperty(data, e.grp); col = e.grp)
        hit_idx = [k for (k, item) in enumerate(targets)
                   if item.target.group === e.grp &&
                      (item.target.member === e.mem ||
                       (item.target.member === nothing && e.mem === :intercept))]
        length(hit_idx) ≤ 1 || throw(ArgumentError(
            "drm: one `sd($(e.grp)$(e.mem === :intercept ? "" : ", coef($(e.mem))")) ~ …` formula per term."))
        if isempty(hit_idx)
            Zg = ones(G2, 1); nm = ["(Intercept)"]
            blk = _sd_target_block((group = e.grp, level = :iid,
                                    member = e.mem === :intercept ? nothing : e.mem,
                                    dpar = :mu, block = nothing); disambiguate = true)
        else
            item = targets[hit_idx[1]]
            union!(used, hit_idx)
            Zg, nm = _sd_group_design(f.response, item.rhs, data, gi2, G2, e.grp;
                                      schema_cache = f.schema_cache, schema_key = item.key)
            size(Zg, 2) ≤ G2 && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
                "drm: SD target `sd($(e.grp)$(e.mem === :intercept ? "" : ", coef($(e.mem))"))` has a rank-deficient group-level design."))
            blk = _sd_target_block(item.target; disambiguate = true)
        end
        w = nothing
        if e.mem !== :intercept
            _require_finite_data_column(data, e.mem)
            w = Float64.(getproperty(data, e.mem))
        end
        push!(extras, (; grp = e.grp, mem = e.mem, gidx = gi2, G = G2, Zg = Matrix{Float64}(Zg),
                       nm = Vector{String}(nm), block = blk, w))
    end
    for (k, item) in enumerate(targets)
        k in used && continue
        tgt = item.target
        what = tgt.member === nothing ? "sd($(tgt.group))" : "sd($(tgt.group), coef($(tgt.member)))"
        terms = join(vcat(["(1 + … | $(b.grp))" for b in blocks_],
                          ["($(e.mem === :intercept ? "1" : "0 + $(e.mem)") | $(e.grp))" for e in extras]), ", ")
        throw(ArgumentError("drm: `$what` names no random-effect term in the mean formula (terms: $terms)."))
    end
    return _fit_gaussian_corr_crossed(fam, y, Xμ, Xσ, blocks_, extras, nmμ, nmσ, g_tol; reml)
end

function _fit_gaussian_corr_crossed(fam::Gaussian, y, Xμ, Xσ, blocks_, extras,
                                    nmμ, nmσ, g_tol; reml::Bool = false)
    n = length(y)
    n ≤ 5000 || throw(ArgumentError("drm: a correlated block beside other random terms " *
        "assembles a dense $(n)×$(n) marginal covariance and is limited to 5000 rows."))
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    off = pμ + pσ
    # parameter layout: [βμ; βσ; (SD coefficients of each coefficient, correlation θ) per block;
    # SD coefficients of each independent term]
    rg = Vector{UnitRange{Int}}[]; rθ = UnitRange{Int}[]
    for bk in blocks_
        r = UnitRange{Int}[]
        for Zg in bk.Zgs
            push!(r, (off+1):(off+size(Zg, 2))); off += size(Zg, 2)
        end
        push!(rg, r)
        npair = bk.q * (bk.q - 1) ÷ 2
        push!(rθ, (off+1):(off+npair)); off += npair
    end
    re_ = UnitRange{Int}[]
    for e in extras
        push!(re_, (off+1):(off+size(e.Zg, 2))); off += size(e.Zg, 2)
    end
    ntot = off
    n ≥ ntot || throw(ArgumentError("drm: only $n observations for $ntot parameters."))
    rowsof(gidx, G) = begin
        r = [Int[] for _ in 1:G]
        for i in 1:n
            push!(r[gidx[i]], i)
        end
        r
    end
    brows = [rowsof(bk.gidx, bk.G) for bk in blocks_]
    erows = [rowsof(e.gidx, e.G) for e in extras]
    const_2pi = 0.5 * n * log(2π)
    # One dense objective for ML and REML. REML (Patterson-Thompson; exact because the marginal
    # is Gaussian-linear in beta_mu) adds ½ logdet(Xμ′V⁻¹Xμ) − ½ pμ log(2π); with the Cholesky
    # V = L L′, X′V⁻¹X = (L⁻¹X)′(L⁻¹X) is PSD by construction.
    function nll_full(θ, restricted::Bool)
        T = eltype(θ)
        ημ = Xμ * θ[1:pμ]; ησ = Xσ * θ[pμ+1:pμ+pσ]
        Vm = zeros(T, n, n)
        for (bi, bk) in enumerate(blocks_)
            q = bk.q
            Lc = _q34_corr_chol(θ[rθ[bi]], q)
            sdg = [exp.(bk.Zgs[k] * θ[rg[bi][k]]) for k in 1:q]
            for g in 1:bk.G
                r = brows[bi][g]
                isempty(r) && continue
                L = Matrix{T}(undef, q, q)
                for a in 1:q, c in 1:q
                    L[a, c] = sdg[a][g] * Lc[a, c]
                end
                UL = bk.U[r, :] * L
                Vm[r, r] .+= UL * UL'
            end
        end
        for (ei, e) in enumerate(extras)
            s2 = exp.(2 .* (e.Zg * θ[re_[ei]]))
            for h in 1:e.G
                r = erows[ei][h]
                isempty(r) && continue
                if e.w === nothing
                    Vm[r, r] .+= s2[h]
                else
                    wr = e.w[r]
                    Vm[r, r] .+= s2[h] .* (wr * wr')
                end
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
    ncomp = sum(bk.q for bk in blocks_) + length(extras)
    θ0 = zeros(ntot)
    θ0[1:pμ] .= βμ0
    s0 = std(res0) / sqrt(ncomp + 1)
    θ0[pμ+1] = log(s0 + eps())
    for (bi, bk) in enumerate(blocks_), k in 1:bk.q
        scale_k = s0 / max(sqrt(sum(abs2, @view bk.U[:, k]) / n), eps())
        θ0[rg[bi][k]] .= bk.Zgs[k] \ fill(log(scale_k + eps()), bk.G)
    end
    for (ei, e) in enumerate(extras)
        scale_e = e.w === nothing ? s0 : s0 / max(sqrt(sum(abs2, e.w) / n), eps())
        θ0[re_[ei]] .= e.Zg \ fill(log(scale_e + eps()), e.G)
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
    single = length(blocks_) == 1
    for (bi, bk) in enumerate(blocks_)
        for k in 1:bk.q
            push!(block_list, bk.keys[k] => rg[bi][k]); push!(name_list, bk.keys[k] => bk.nmsds[k])
        end
        names_θ = String[]
        for i in 1:bk.q, j in 1:(i-1)
            push!(names_θ, single ? "theta($(bk.members[i]), $(bk.members[j]))" :
                           "theta($(bk.members[i]), $(bk.members[j]) | $(bk.grp))")
        end
        key = single ? :recov : Symbol("recov_", bk.grp)
        push!(block_list, key => rθ[bi]); push!(name_list, key => names_θ)
    end
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
