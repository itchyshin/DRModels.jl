# Two latent mean fields, each with its own predictor-dependent SD formula, for
# the log-link Gamma and NB2 families (TMB-convention Laplace, `sigma ~ 1`):
#
#   eta_i = X_i beta + sum_k w_k[i] * exp(Z_k[g_k(i), :] alpha_k) * u_k[g_k(i)],
#   u_k ~ N(0, Q_k^{-1})  (independent fields),
#
# where Q_k = I for an ordinary `(1 | g)` / `(0 + x | g)` term and Q_k = C^{-1}
# for a raw (never rescaled) `animal`/`relmat` covariance C, as in native
# drmTMB. One engine covers the shapes native drmTMB fits with two SD formulas:
#
#   same-group pair     (1 | g) + (0 + x | g)        sd(g, coef(intercept)), sd(g, coef(x))
#   crossed intercepts  (1 | g) + (1 | h)            sd(g), sd(h)
#   ordinary+structured (1 | g) + relmat/animal(1 | p)  sd(g), sd(p, level)
#   structured q2       relmat(1 + x | p)            sd(p, relmat, coef(intercept)), sd(p, relmat, coef(x))
#
# and, with one cross-field correlation rho = tanh(eta_cor) (native drmTMB's
# `eta_cor_*` is the same atanh scale), the correlated same-group shapes
#
#   labelled structured q2   animal/relmat(1 + x | p | g)   (Gamma, NB2)
#   ordinary correlated q2   (1 + x | g)                    (NB2; native drmTMB refuses Gamma)
#
# whose prior precision is kron(R(rho)^{-1}, Q): logdet P = 2 logdet Q - G log(1 - rho^2).
#
# The Laplace marginal is  sum_i v_i(eta*) + 0.5 u*' P u* - 0.5 logdet P + 0.5 logdet H,
# P = blockdiag(Q_1, Q_2), H = P + L' diag(d2) L, L the n x (G_1 + G_2) loading
# matrix. The 2*pi constants of the prior and of the Laplace normaliser cancel.

struct _TwoField
    Q::SparseMatrixCSC{Float64,Int}
    logdetQ::Float64
    gidx::Vector{Int}
    w::Vector{Float64}        # per-row base loading: 1 (intercept) or x (slope)
    Zg::Matrix{Float64}       # group-level SD design
end

_twofield_kind(::Gamma) = Val(:gamma_fixed)
_twofield_kind(::NegBinomial2) = Val(:nb2_fixed)

function _twofield_loadings(fields, theta, pmu)
    off = pmu + 1
    loads = Vector{Vector{Float64}}(undef, length(fields))
    for (k, fk) in enumerate(fields)
        p = size(fk.Zg, 2)
        a = @view theta[(off + 1):(off + p)]
        sdg = exp.(fk.Zg * a)
        loads[k] = fk.w .* sdg[fk.gidx]
        off += p
    end
    return loads
end

function _twofield_mode(kind, aux, y, eta0, loads, fields, P; b0 = nothing,
                        maxiter::Int = 100, tol::Real = 1e-10)
    n = length(y)
    Gs = [size(fk.Q, 1) for fk in fields]
    offs = cumsum([0; Gs[1:end-1]])
    Gt = sum(Gs)
    # Sparse loading matrix L (n x Gt), fixed pattern.
    I_ = Int[]; J_ = Int[]; V_ = Float64[]
    for (k, fk) in enumerate(fields), i in 1:n
        push!(I_, i); push!(J_, offs[k] + fk.gidx[i]); push!(V_, loads[k][i])
    end
    L = sparse(I_, J_, V_, n, Gt)
    Lt = sparse(L')
    u = b0 === nothing ? zeros(Gt) : copy(b0)
    function joint(uu)
        eta = eta0 .+ L * uu
        value = 0.5 * dot(uu, P * uu)
        @inbounds for i in 1:n
            value += first(_laplace_v123(kind, aux, i, eta[i]))
        end
        value
    end
    d1 = zeros(n); d2 = zeros(n)
    function derivs!(uu)
        eta = eta0 .+ L * uu
        @inbounds for i in 1:n
            _, a1, a2, _ = _laplace_v123(kind, aux, i, eta[i])
            d1[i] = a1; d2[i] = a2
        end
    end
    for _ in 1:maxiter
        derivs!(u)
        grad = P * u .+ Lt * d1
        H = P + Lt * spdiagm(0 => d2) * L
        ch = cholesky(Symmetric(H); check = false)
        issuccess(ch) || return u, ch, false
        step = ch \ grad
        if norm(step) <= tol * (1 + norm(u))
            u .-= step
            derivs!(u)
            H = P + Lt * spdiagm(0 => d2) * L
            ch = cholesky(Symmetric(H); check = false)
            return u, ch, issuccess(ch)
        end
        if norm(step) <= 1e-3 * (1 + norm(u))
            u .-= step
            continue
        end
        f0 = joint(u)
        amount, accepted = 1.0, false
        while amount >= 1e-6
            trial = u .- amount .* step
            if joint(trial) <= f0
                u = trial
                accepted = true
                break
            end
            amount *= 0.5
        end
        accepted || return u, ch, false
    end
    return u, nothing, false
end

function _twofield_laplace(kind, y, Xmu, fields, P, logdetP, theta, aux_from;
                           b0 = nothing, clamp_logsigma::Bool = false)
    pmu = size(Xmu, 2)
    ls = theta[pmu + 1]
    aux = aux_from(clamp_logsigma ? clamp(ls, -4.0, 4.0) : ls)
    eta0 = Xmu * theta[1:pmu]
    loads = _twofield_loadings(fields, theta, pmu)
    all(l -> all(isfinite, l), loads) || return 1e18, nothing, false
    u, ch, ok = _twofield_mode(kind, aux, y, eta0, loads, fields, P; b0)
    ok || return 1e18, u, false
    n = length(y)
    eta = copy(eta0)
    Gs = [size(fk.Q, 1) for fk in fields]
    offs = cumsum([0; Gs[1:end-1]])
    for (k, fk) in enumerate(fields), i in 1:n
        eta[i] += loads[k][i] * u[offs[k] + fk.gidx[i]]
    end
    value = 0.5 * dot(u, P * u)
    @inbounds for i in 1:n
        value += first(_laplace_v123(kind, aux, i, eta[i]))
    end
    return value - 0.5 * logdetP + 0.5 * logdet(ch), u, true
end

"""
    _fit_twofield_sd_laplace(fam, y, Xmu, Xsigma, fields, nmmu, nmsigma, nmsds, blocks, g_tol; se)

Fit two independent latent mean fields (see the file header) with one SD
formula each. `fields` is a 2-vector of `_TwoField`; `blocks` / `nmsds` name
the two SD-coefficient blocks. `sigma ~ 1` only.
"""
function _fit_twofield_sd_laplace(fam::Union{Gamma,NegBinomial2}, y, Xmu, Xsigma,
                                  fields::Vector{_TwoField}, nmmu, nmsigma, nmsds,
                                  blocks, g_tol; se::Bool = true, maxiter::Int = 300,
                                  correlated::Bool = false,
                                  rho_name::AbstractString = "atanh_rho",
                                  context::AbstractString = "two-field SD regression")
    length(fields) == 2 || throw(ArgumentError("expected two latent SD fields"))
    size(Xsigma, 2) == 1 || throw(ArgumentError("$context requires `sigma ~ 1`."))
    kind = _twofield_kind(fam)
    isgamma = fam isa Gamma
    if correlated
        (fields[1].gidx == fields[2].gidx && fields[1].Q == fields[2].Q) || throw(ArgumentError(
            "$context: a correlated pair needs one grouping factor and one kernel."))
    end
    P0 = blockdiag(fields[1].Q, fields[2].Q)
    logdetP0 = fields[1].logdetQ + fields[2].logdetQ
    Gc = size(fields[1].Q, 1)
    function prior(theta)
        correlated || return P0, logdetP0
        rho = tanh(theta[end])
        Rinv = sparse([1.0 -rho; -rho 1.0] ./ (1 - rho^2))
        return kron(Rinv, fields[1].Q), 2 * fields[1].logdetQ - Gc * log1p(-rho^2)
    end
    aux_from, beta0, logsigma0 = isgamma ? _gamma_laplace_setup(y, Xmu) :
                                           _nb2_laplace_setup(y, Xmu)
    pmu = size(Xmu, 2)
    psd = [size(fk.Zg, 2) for fk in fields]
    ranges = (pmu + 2):(pmu + 1 + psd[1]), (pmu + 2 + psd[1]):(pmu + 1 + sum(psd))
    theta0 = zeros(pmu + 1 + sum(psd) + correlated)
    theta0[1:pmu] .= beta0
    theta0[pmu + 1] = isgamma ? clamp(logsigma0, -3.9, 3.9) : logsigma0
    theta0[ranges[1]] .= fields[1].Zg \ fill(log(0.4), size(fields[1].Zg, 1))
    theta0[ranges[2]] .= fields[2].Zg \ fill(log(0.3), size(fields[2].Zg, 1))
    Gt = size(P0, 1)
    last_u = zeros(Gt)
    function nll(theta)
        correlated && abs(theta[end]) > 15 && return 1e18
        P, logdetP = prior(theta)
        value, u, ok = _twofield_laplace(kind, y, Xmu, fields, P, logdetP, theta,
                                         aux_from; b0 = last_u, clamp_logsigma = isgamma)
        if !ok
            value, u, ok = _twofield_laplace(kind, y, Xmu, fields, P, logdetP, theta,
                                             aux_from; clamp_logsigma = isgamma)
        end
        ok || return 1e18
        last_u .= u
        value
    end
    function grad!(g, theta)
        for j in eachindex(theta)
            h = 2e-4 * (1 + abs(theta[j]))
            e = zeros(length(theta)); e[j] = h
            g[j] = (nll(theta .+ e) - nll(theta .- e)) / (2h)
        end
        return g
    end
    function run(start)
        od = Optim.OnceDifferentiable(nll, grad!, start)
        res = if isgamma
            lower, upper = fill(-Inf, length(start)), fill(Inf, length(start))
            lower[pmu + 1], upper[pmu + 1] = -4.0, 4.0
            Optim.optimize(od, lower, upper, start,
                Optim.Fminbox(Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking())),
                Optim.Options(g_tol = g_tol, iterations = maxiter))
        else
            Optim.optimize(od, start,
                Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
                Optim.Options(g_tol = g_tol, iterations = maxiter))
        end
        return res, Optim.minimizer(res), Optim.minimum(res)
    end
    result, theta, fmin = run(theta0)
    if !isgamma && theta[pmu + 1] < _NB2_SIGMA_PLATEAU_LOGSIGMA
        restart = copy(theta0); restart[pmu + 1] = log(0.5); last_u .= 0
        result2, theta2, fmin2 = run(restart)
        fmin2 < fmin && ((result, theta, fmin) = (result2, theta2, fmin2))
    end
    nll_hat = nll(theta)
    gf = zeros(length(theta)); grad!(gf, theta)
    near_bound = isgamma && abs(theta[pmu + 1]) >= 4 -
        _fd_hessian_step(length(y)) * (1 + abs(theta[pmu + 1]))
    V = se && !near_bound ? _vcov_from_fd_hessian(
        _finite_hessian(nll, theta; h = _fd_hessian_step(length(y))); context) :
        fill(NaN, length(theta), length(theta))
    block_list = Pair{Symbol,UnitRange{Int}}[
        :mu => 1:pmu, :sigma => (pmu + 1):(pmu + 1),
        blocks[1] => ranges[1], blocks[2] => ranges[2]]
    name_list = Pair{Symbol,Vector{String}}[
        :mu => nmmu, :sigma => nmsigma,
        blocks[1] => nmsds[1], blocks[2] => nmsds[2]]
    if correlated
        push!(block_list, :recov => length(theta):length(theta))
        push!(name_list, :recov => [String(rho_name)])
    end
    scales = Dict{Symbol,Vector{Float64}}(:sigma => fill(exp(theta[pmu + 1]), length(y)))
    correlated && (scales[:correlation] = [tanh(theta[end])])
    for k in 1:2
        sdk = exp.(fields[k].Zg * theta[ranges[k]])
        scales[blocks[k]] = sdk[fields[k].gidx]
        scales[Symbol(string(blocks[k]), "_group")] = sdk
    end
    # Conditional modes at theta_hat (fresh inner solve), recorded so
    # `residuals(type = :quantile)` conditions on them as drmTMB's does.
    let (Ph, logdetPh) = prior(theta)
        _, uhat, okhat = _twofield_laplace(kind, y, Xmu, fields, Ph, logdetPh, theta,
                                           aux_from; clamp_logsigma = isgamma)
        if okhat
            loads = _twofield_loadings(fields, theta, pmu)
            Gs = [size(fk.Q, 1) for fk in fields]
            offs = cumsum([0; Gs[1:end-1]])
            eta_re = zeros(length(y))
            for (k, fk) in enumerate(fields), i in eachindex(y)
                eta_re[i] += loads[k][i] * uhat[offs[k] + fk.gidx[i]]
            end
            _attach_conditional_structured!(scales, eta_re,
                [uhat[(offs[k] + 1):(offs[k] + Gs[k])] for k in 1:2]...)
        else
            _mark_cond_failed!(scales)
        end
    end
    fit = DrmFit(fam, block_list, name_list, theta, Matrix(V), -nll_hat,
        length(y), _laplace_outer_converged(result, nll_hat, gf, theta,
        length(y), g_tol), Dict(:mu => exp.(Xmu * theta[1:pmu])),
        Dict(:mu => Vector{Float64}(y)), scales)
    return _withiterations(_withnll(fit, _nll_pinned(nll, theta, nll_hat)),
                           Optim.iterations(result))
end

# ---------------------------------------------------------------------------
# Router: recognise the two-SD shapes on the Gamma / NB2 mean and build fields.
# Returns `nothing` when the formula is not one of these shapes, so the
# caller's existing routes (and refusals) stay in charge.
# ---------------------------------------------------------------------------

_twofield_famname(::Gamma) = "Gamma"
_twofield_famname(::NegBinomial2) = "NegBinomial2"

function _twofield_numeric(data, v::Symbol, n::Int, famname)
    hasproperty(data, v) || throw(ArgumentError(
        "$famname random-slope predictor `$v` is absent from data."))
    raw = getproperty(data, v)
    raw isa AbstractVector{<:Real} && length(raw) == n && all(isfinite, raw) ||
        throw(ArgumentError("$famname random-slope predictor `$v` must be numeric, finite, and have one value per response."))
    any(!iszero, raw) || throw(ArgumentError(
        "$famname random-slope predictor `$v` is all zero, so its SD is unidentified."))
    return Float64.(raw)
end

function _drm_twofield_sd_route(f::DrmFormula, fam::Union{Gamma,NegBinomial2};
                                data, tree = nothing, K = nothing, A = nothing,
                                Ainv = nothing, coords = nothing,
                                g_tol::Real = 1e-8, se::Bool = true)
    b = _twofield_build(f, fam; data, tree, K, A, Ainv, coords)
    b === nothing && return nothing
    famname = _twofield_famname(fam)
    fit = _fit_twofield_sd_laplace(fam, b.y, b.Xmu, b.Xsigma, b.fields, b.nmmu, b.nmsigma,
        b.nmsds, b.blocks, g_tol; se, correlated = b.correlated,
        rho_name = "$(b.group):atanh_rho",   # as the other labelled q2 routes
        context = "$famname two-field SD regression")
    return _withformula(fit, f)
end

# Recognise the shape and build the response, designs and the two `_TwoField`s.
# Shared by the fitter and by the post-fit marginal simulator, so both see the
# same fields (row-to-level maps, kernels, loadings, SD designs). Returns
# `nothing` for formulas that are not two-field shapes. `store_schema = false`
# (post-fit) reads the fitted SD-design schema without overwriting it.
function _twofield_build(f::DrmFormula, fam::Union{Gamma,NegBinomial2};
                         data, tree = nothing, K = nothing, A = nothing,
                         Ainv = nothing, coords = nothing, store_schema::Bool = true)
    famname = _twofield_famname(fam)
    targets = _sd_target_parts(f)
    length(targets) == 2 || return nothing
    all(t -> t.target.dpar === :mu && t.target.block === nothing &&
        t.target.level in (:iid, :animal, :relmat), targets) || return nothing
    rhs = Dict(f.forms)
    haskey(rhs, :mu) || return nothing
    correlated = false
    labelled = _twofield_labelled_q2(rhs[:mu])
    if labelled !== nothing
        # Rebuild the mean without the labelled term so _split_ranef sees only the
        # fixed part and any other (refused) random terms.
        terms = rhs[:mu] isa Tuple ? collect(rhs[:mu]) : Any[rhs[:mu]]
        rest = [t for t in terms if t !== labelled.term]
        rest_rhs = isempty(rest) ? ConstantTerm(1) : length(rest) == 1 ? rest[1] : Tuple(rest)
        fixed_mu, re, metav, st = _split_ranef(rest_rhs)
        (isempty(re) && metav === nothing && st === nothing) || throw(ArgumentError(
            "$famname labelled `$(labelled.provider)(1 + $(labelled.slope) | $(labelled.label) | $(labelled.group))` " *
            "must be the only random-effect term."))
        structured = [(labelled.provider, labelled.group)]
        sslope = labelled.slope
        correlated = true
    else
        fixed_mu, re, metav, st, _, _, sslope = _split_ranef(rhs[:mu];
            allow_relmat_intercept_slope = true, allow_animal_intercept_slope = true)
    end
    metav === nothing || return nothing
    labelled === nothing && (structured = _collect_structured(rhs[:mu]))
    length(structured) <= 1 || return nothing
    # Identify the shape and the two (term, target) pairs.
    specs = nothing   # Vector of (; kind, group, slope, provider, target)
    levels = Set(t.target.level for t in targets)
    if labelled !== nothing
        prov, pgrp = only(structured)
        levels == Set([prov]) || throw(ArgumentError(
            "$famname labelled $prov q2 needs two `sd($pgrp, $prov, coef(…))` formulas."))
        all(t -> t.target.group === pgrp, targets) || return nothing
        Set(t.target.member for t in targets) == Set([:intercept, sslope]) || throw(ArgumentError(
            "$famname $prov SD formulas must select `coef(intercept)` and `coef($sslope)` exactly once."))
        specs = [(; group = pgrp, slope = m === :intercept ? nothing : m, provider = prov,
                   target = only(filter(t -> t.target.member === m, targets)))
                 for m in (:intercept, sslope)]
    elseif isempty(structured) && levels == Set([:iid]) && length(re) == 1 &&
           first(_re_kind(re[1][1])) === :corr && fam isa NegBinomial2
        # Ordinary correlated (1 + x | g) with coef-selected SDs (NB2 only; native
        # drmTMB refuses this shape for Gamma).
        g = re[1][2]; slope = _re_kind(re[1][1])[2]
        all(t -> t.target.group === g, targets) || return nothing
        Set(t.target.member for t in targets) == Set([:intercept, slope]) || throw(ArgumentError(
            "$famname `(1 + $slope | $g)` SD regression needs `sd($g, coef(intercept))` and `sd($g, coef($slope))` exactly once each."))
        specs = [(; group = g, slope = m === :intercept ? nothing : m, provider = :iid,
                   target = only(filter(t -> t.target.member === m, targets)))
                 for m in (:intercept, slope)]
        correlated = true
    elseif isempty(structured) && levels == Set([:iid]) && length(re) == 2
        kinds = [_re_kind(r[1]) for r in re]
        grps = [r[2] for r in re]
        if grps[1] == grps[2]
            # Same-group independent pair (1 | g) + (0 + x | g).
            sort_kinds = Set(first.(kinds))
            sort_kinds == Set([:intercept, :slope]) || return nothing
            slope = only(k[2] for k in kinds if k[1] === :slope)
            members = Set(t.target.member for t in targets)
            members == Set([:intercept, slope]) || throw(ArgumentError(
                "$famname same-group `(1 | $(grps[1])) + (0 + $slope | $(grps[1]))` SD regression " *
                "needs `sd($(grps[1]), coef(intercept))` and `sd($(grps[1]), coef($slope))` exactly once each."))
            all(t -> t.target.group === grps[1], targets) || return nothing
            specs = [(; group = grps[1], slope = m === :intercept ? nothing : m,
                       provider = :iid,
                       target = only(filter(t -> t.target.member === m, targets)))
                     for m in (:intercept, slope)]
        else
            # Crossed intercepts (1 | g) + (1 | h).
            all(k -> k[1] === :intercept, kinds) || return nothing
            specs = map(grps) do g
                ms = filter(t -> t.target.group === g, targets)
                length(ms) == 1 || throw(ArgumentError(
                    "$famname crossed `(1 | g) + (1 | h)` SD regression needs one `sd(group) ~ …` formula per grouping factor."))
                t = only(ms)
                t.target.member in (nothing, :intercept) || throw(ArgumentError(
                    "$famname crossed SD target `coef($(t.target.member))` does not match the intercept-only term."))
                (; group = g, slope = nothing, provider = :iid, target = t)
            end
        end
    elseif length(structured) == 1 && length(re) == 1 && sslope === nothing
        # Ordinary intercept beside a structured relmat/animal intercept.
        prov, pgrp = only(structured)
        prov in (:relmat, :animal) || return nothing
        k, _ = _re_kind(re[1][1])
        k === :intercept || return nothing
        g = re[1][2]
        g === pgrp && throw(ArgumentError(
            "$famname ordinary and $prov intercepts must use different grouping factors."))
        tg = filter(t -> t.target.group === g && t.target.level === :iid, targets)
        tp = filter(t -> t.target.group === pgrp && t.target.level === prov, targets)
        (length(tg) == 1 && length(tp) == 1) || throw(ArgumentError(
            "$famname `(1 | $g) + $prov(1 | $pgrp)` SD regression needs `sd($g) ~ …` and `sd($pgrp, $prov) ~ …`."))
        for t in (only(tg), only(tp))
            t.target.member in (nothing, :intercept) || throw(ArgumentError(
                "$famname SD target `coef($(t.target.member))` does not match an intercept-only term."))
        end
        specs = [(; group = g, slope = nothing, provider = :iid, target = only(tg)),
                 (; group = pgrp, slope = nothing, provider = prov, target = only(tp))]
    elseif length(structured) == 1 && isempty(re) && sslope !== nothing &&
           first(only(structured)) === :relmat && fam isa Gamma
        # Independent relmat(1 + x | p) q2 (Gamma; the NB2 relmat q2 route exists).
        _, pgrp = only(structured)
        levels == Set([:relmat]) || return nothing
        all(t -> t.target.group === pgrp, targets) || return nothing
        Set(t.target.member for t in targets) == Set([:intercept, sslope]) || throw(ArgumentError(
            "$famname relmat SD formulas must select `coef(intercept)` and `coef($sslope)` exactly once."))
        specs = [(; group = pgrp, slope = m === :intercept ? nothing : m, provider = :relmat,
                   target = only(filter(t -> t.target.member === m, targets)))
                 for m in (:intercept, sslope)]
    else
        return nothing
    end
    # Common gates.
    sigma_rhs = get(rhs, :sigma, nothing)
    sigma_rhs isa ConstantTerm && sigma_rhs == ConstantTerm(1) || throw(ArgumentError(
        "$famname two-field SD regression requires a fixed `sigma ~ 1` submodel."))
    (haskey(rhs, :zi) || haskey(rhs, :hu)) && throw(ArgumentError(
        "$famname two-field SD regression cannot be combined with `zi` or `hu`."))
    allowed = Set(vcat([:mu, :sigma], [t.key for t in targets]))
    all(first(p) in allowed for p in f.forms) || throw(ArgumentError(
        "$famname two-field SD regression supports only `mu`, fixed `sigma ~ 1`, and its two SD formulas."))
    fixed_mu, offset_term = _extract_offset(fixed_mu)
    offset_term === nothing || throw(ArgumentError(
        "$famname two-field SD regression does not support `offset(...)`."))
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "$famname two-field SD regression requires observed responses; remove incomplete rows before fitting."))
    providers = Set(s.provider for s in specs)
    tree === nothing && coords === nothing || throw(ArgumentError(
        "$famname two-field SD regression does not use `tree` or `coords`."))
    if :relmat in providers
        (A === nothing && Ainv === nothing) || throw(ArgumentError(
            "$famname relmat SD regression uses `K = …` only."))
        K === nothing && throw(ArgumentError("relmat(…) needs the covariance matrix `K = …`."))
    elseif :animal in providers
        (K === nothing && Ainv === nothing) || throw(ArgumentError(
            "$famname two-field animal SD regression uses `A = …` only."))
        A === nothing && throw(ArgumentError("animal(…) needs the covariance matrix `A = …`."))
    else
        (K === nothing && A === nothing && Ainv === nothing) || throw(ArgumentError(
            "$famname ordinary two-field SD regression does not use `K`, `A` or `Ainv`."))
    end
    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    _, Xsigma, nmsigma = _design(f.response, sigma_rhs, data)
    if fam isa Gamma
        all(yi -> isfinite(yi) && yi > 0, y) || throw(ArgumentError(
            "Gamma() requires strictly positive responses"))
    else
        all(yi -> isfinite(yi) && yi >= 0 && isinteger(yi), y) || throw(ArgumentError(
            "NegBinomial2() requires non-negative integer counts as the response"))
    end
    n = length(y)
    fields = _TwoField[]
    nmsds = Vector{String}[]
    blocks = Symbol[]
    for s in specs
        labels = getproperty(data, s.group)
        if s.provider === :iid
            gidx, G = _group_index(labels; col = s.group)
            Q = sparse(1.0I, G, G); ldq = 0.0
        else
            C = s.provider === :relmat ? K : A
            Q, gidx = _general_cov_setup(C, labels; normalize = false)
            G = size(Q, 1)
            qc = cholesky(Symmetric(Matrix(Q)); check = false)
            issuccess(qc) || throw(ArgumentError("$famname $(s.provider) covariance must be positive definite."))
            ldq = logdet(qc)
        end
        Zg, nm = _sd_group_design(f.response, s.target.rhs, data, gidx, G, s.group;
            schema_cache = f.schema_cache, schema_key = s.target.key, store_schema)
        size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
            "$famname SD formula for `$(s.group)` has a rank-deficient group-level design."))
        w = s.slope === nothing ? ones(n) : _twofield_numeric(data, s.slope, n, famname)
        push!(fields, _TwoField(Q, ldq, gidx, w, Matrix{Float64}(Zg)))
        push!(nmsds, nm)
        push!(blocks, _sd_target_block(s.target.target; disambiguate = true))
    end
    length(unique(blocks)) == 2 || throw(ArgumentError(
        "$famname two-field SD regression needs two distinct SD targets."))
    if specs[1].group === specs[2].group && all(==(first(fields[2].w)), fields[2].w) &&
       isequal(fields[1].Zg, fields[2].Zg)
        throw(ArgumentError("$famname cannot separate the intercept and slope SDs when the slope predictor is constant and both SD formulas are identical."))
    end
    return (; y, Xmu, Xsigma, nmmu, nmsigma, fields, nmsds, blocks, correlated,
            group = specs[1].group)
end

# Post-fit marginal simulator for the two-field shapes (`simulate(fit; data, …)`
# with the default `re = :marginal`, and `bootstrap_result`). Rebuilds the fields
# from `data` (and `K` / `A`), draws fresh latent fields u_k ~ N(0, Q_k^{-1})
# (correlated pair: cov kron(R(rho), C), rho = tanh(atanh_rho) as in the fit),
# eta = X beta + sum_k w_k * sd_k[g_k] * u_k[g_k], and y from the family at the
# fitted sigma. Returns `nothing` when the formula is not a two-field shape.
function _twofield_sd_marginal_simulator(fit::DrmFit, data; K = nothing, A = nothing)
    fam = fit.family
    b = _twofield_build(fit.formula, fam; data, K, A, store_schema = false)
    b === nothing && return nothing
    b.nmmu == _block_coefnames(fit, :mu) || throw(ArgumentError(
        "bootstrap: two-field mean design does not match the fitted `mu` block"))
    for k in 1:2
        b.nmsds[k] == _block_coefnames(fit, b.blocks[k]) || throw(ArgumentError(
            "bootstrap: two-field SD design does not match fitted block `$(b.blocks[k])`"))
    end
    eta_fixed = b.Xmu * coef(fit, :mu)
    loads = [b.fields[k].w .* exp.(b.fields[k].Zg * coef(fit, b.blocks[k]))[b.fields[k].gidx]
             for k in 1:2]
    # Q_k is the prior precision (I or C^{-1}); draw u_k = U_k \ z with Q_k = U_k'U_k.
    facs = map(b.fields) do fk
        ch = cholesky(Symmetric(Matrix(fk.Q)); check = false)
        issuccess(ch) || throw(ArgumentError(
            "bootstrap: two-field prior precision is not positive definite"))
        ch
    end
    rho = b.correlated ? tanh(only(coef(fit, :recov))) : 0.0
    sigma = fill(exp(only(coef(fit, :sigma))), length(b.y))
    G = [size(fk.Q, 1) for fk in b.fields]
    return function (rng)
        z1 = facs[1].U \ randn(rng, G[1])
        z2 = facs[2].U \ randn(rng, G[2])
        u2 = b.correlated ? rho .* z1 .+ sqrt(1 - rho^2) .* z2 : z2
        eta = eta_fixed .+ loads[1] .* z1[b.fields[1].gidx] .+
              loads[2] .* u2[b.fields[2].gidx]
        return _simulate_once(fit, rng; mu = exp.(eta), sigma)
    end
end

# A labelled structured q2 term `animal(1 + x | p | g)` / `relmat(1 + x | p | g)`:
# StatsModels parses the bar left-associatively, `((1 + x) | p) | g`.
function _twofield_labelled_q2(rhs)
    terms = rhs isa Tuple ? collect(rhs) : Any[rhs]
    found = nothing
    for t in terms
        (t isa FunctionTerm && (t.f === animal || t.f === relmat)) || continue
        bar = t.args[1]
        (bar isa FunctionTerm && bar.f === (|)) || continue
        inner = bar.args[1]
        (inner isa FunctionTerm && inner.f === (|)) || continue
        lhs, lab = inner.args[1], inner.args[2]
        lab isa Term && bar.args[2] isa Term || continue
        lhs isa FunctionTerm && lhs.f === (+) || continue
        consts = filter(x -> x isa ConstantTerm, lhs.args)
        vars = filter(x -> x isa Term, lhs.args)
        (length(lhs.args) == 2 && length(consts) == 1 && consts[1].n == 1 &&
         length(vars) == 1) || throw(ArgumentError(
            "drm: a labelled `$(nameof(t.f))(… | $(lab.sym) | $(bar.args[2].sym))` term is implemented " *
            "for the intercept-plus-one-slope form `1 + x` only."))
        found === nothing || throw(ArgumentError("drm: only one labelled structured term is implemented."))
        found = (; term = t, provider = t.f === animal ? :animal : :relmat,
                 group = bar.args[2].sym, label = lab.sym, slope = vars[1].sym)
    end
    return found
end
