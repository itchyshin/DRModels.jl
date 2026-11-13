# Binomial/Bernoulli SD regression for ordinary group effects. Each independent
# coefficient can have its own group-level log-SD regression; one ordinary
# intercept-first correlated block can also use predictor-dependent marginal
# SDs and constant partial-correlation parameters. Bernoulli responses have no
# residual scale.

"""
    phylo_interaction(1 | host & pollinator)
    phylo_interaction(0 + x | host & pollinator)

Mark a Bernoulli/Binomial phylogenetic interaction random intercept or one
numeric random slope in a `bf()` location formula. Supply `tree1` and `tree2`
to `drm()` and a pair ID column for the matching
`sd(pair, phylogenetic_interaction[, coef(x)])` formula. The fitted
exponential SD is a multiplier of the two supplied raw tree covariances; the
marginal pair SD also reflects both kernel diagonals.
"""
phylo_interaction(x) = x

function _binomial_phylo_interaction_parts(rhs)
    terms = rhs isa Tuple ? collect(rhs) : Any[rhs]
    positions = findall(t -> t isa StatsModels.FunctionTerm &&
                              t.f === phylo_interaction, terms)
    length(positions) <= 1 || throw(ArgumentError(
        "Binomial formulas support one `phylo_interaction(1 | host & pollinator)` term."))
    isempty(positions) && return nothing
    term = terms[only(positions)]
    length(term.args) == 1 && term.args[1] isa StatsModels.FunctionTerm &&
        term.args[1].f === (|) || throw(ArgumentError(
        "`phylo_interaction()` must contain `1 | host & pollinator`."))
    ranef = only(term.args)
    length(ranef.args) == 2 || throw(ArgumentError(
        "`phylo_interaction()` requires one coefficient design and two partners."))
    lhs = ranef.args[1]
    slope = if lhs isa StatsModels.ConstantTerm && lhs.n == 1
        nothing
    elseif lhs isa StatsModels.FunctionTerm && lhs.f === (+) &&
           length(lhs.args) == 2 && lhs.args[1] isa StatsModels.ConstantTerm &&
           lhs.args[1].n == 0 && lhs.args[2] isa StatsModels.Term
        lhs.args[2].sym
    else
        throw(ArgumentError(
            "Binomial phylogenetic interaction supports `1` or one numeric slope `0 + x` in `phylo_interaction(0 + x | host & pollinator)`."))
    end
    partners = ranef.args[2]
    length(partners.args) == 2 && partners.f === (&) &&
        all(x -> x isa StatsModels.Term, partners.args) || throw(ArgumentError(
        "`phylo_interaction()` requires two grouping columns joined by `&`."))
    first_group, second_group = (x.sym for x in partners.args)
    first_group !== second_group || throw(ArgumentError(
        "The two phylogenetic-interaction grouping columns must be different."))
    deleteat!(terms, only(positions))
    fixed_rhs = isempty(terms) ? StatsModels.ConstantTerm(1) :
        length(terms) == 1 ? only(terms) : Tuple(terms)
    return (; fixed_rhs, group1 = first_group, group2 = second_group,
            slope)
end

# Stable sech(eta): unlike sqrt(1 - tanh(eta)^2), this remains positive for
# large finite eta after tanh(eta) rounds to +/-1.
function _binomial_stable_sech(eta)
    a = abs(eta)
    return exp(log(2) - a - log1p(exp(-2a)))
end

# Match drmTMB's default smooth guard on latent log-SD predictors. The fitted
# coefficients remain on the raw linear-predictor scale.
_binomial_sd_scale(eta) = exp.(_softclamp.(eta, -12.0, 12.0, 3.0))
_binomial_structured_sd_group_scale(Zg, alpha) = _binomial_sd_scale(Zg * alpha)

# Partial correlation and innovation scale of one correlation parameter `eta` (atanh scale).
# `eta` is soft-clamped with `_softclamp` at +-atanh(0.999999) = +-7.2543 and margin 3: the
# identity (bit for bit) for |eta| <= 7.2543, then a smooth, monotone saturation at
# +-10.254 (|rho| up to about 1 - 2.5e-9). The only purpose is to cut off the region where
# the Laplace / AGHQ objective is rounding noise: once |eta| exceeds about 12 the group prior
# precision is ~1/(1 - rho^2) ~ 1e10 and the objective carries noise of order 1e-4, which the
# optimiser can accept as progress (Julia 1.13.3, cell `binomial|corr|lab0|q2|sd-last`).
# It is NOT a 0.999999 bound: the q2 Poisson / NB2 / Gaussian routes scale, rho = 0.999999 *
# tanh(eta), and drmTMB's Binomial q2 block uses plain tanh(eta) with no bound.
# Reporting: the coefficient table and `:recov` keep the raw `eta` the optimiser holds, while
# `:corpars`, the correlation in the likelihood and the bootstrap simulators use the clamped
# value. They agree for |eta| <= 7.2543 and differ beyond it (raw eta = -8.29 is effective
# -8.25; raw -13 is -10.13), where the objective is flat in `eta` anyway. Reported on purpose
# rather than rewriting the stored coefficient, which would make `coef` disagree with `theta`.
function _binomial_bounded_partial(eta)
    e = _softclamp(eta, -atanh(_Q2_RHO_BOUND), atanh(_Q2_RHO_BOUND), 3.0)
    return tanh(e), _binomial_stable_sech(e)
end

# Lower-triangular Cholesky factor for a correlation matrix from row-major
# partial correlations. For row i and column j<i, the partial parameter is
# conditional on the preceding j-1 coordinates. This preserves the established
# q=2 and q=3 ordering while extending it to arbitrary block dimension.
function _binomial_partial_correlation_cholesky(correlation_eta, q::Integer)
    q >= 2 || throw(ArgumentError("a correlated Binomial block needs at least two coefficients"))
    ncor = q * (q - 1) ÷ 2
    length(correlation_eta) == ncor || throw(ArgumentError(
        "a q=$q correlated Binomial block needs $ncor partial-correlation parameters"))
    T = eltype(correlation_eta)
    L = zeros(T, q, q)
    L[1, 1] = one(T)
    position = 1
    for i in 2:q
        residual = one(T)
        for j in 1:(i - 1)
            eta = correlation_eta[position]
            partial, innovation = _binomial_bounded_partial(eta)
            L[i, j] = partial * residual
            residual *= innovation
            position += 1
        end
        L[i, i] = residual
    end
    return L
end

function _binomial_pairwise_correlations(correlation_eta, q::Integer)
    L = _binomial_partial_correlation_cholesky(correlation_eta, q)
    R = L * transpose(L)
    return [R[i, j] for i in 1:(q - 1) for j in (i + 1):q]
end

function _binomial_partial_correlation_names(group, q::Integer)
    q == 2 && return ["$(group):atanh_rho"]
    names = String[]
    for i in 2:q, j in 1:(i - 1)
        suffix = j == 1 ? "$(i)$(j)" :
            "$(i)$(j)_given_$(join(1:(j - 1), "_"))"
        push!(names, "$(group):atanh_partial_rho_$(suffix)")
    end
    return names
end

# Observed derivatives of the probit Binomial negative log likelihood.
function _binomial_probit_v123(aux, i, eta)
    s, n = aux.s[i], aux.ntr[i]
    normal = Distributions.Normal()
    logphi = Distributions.logpdf(normal, eta)
    logcdf = Distributions.logcdf(normal, eta)
    logccdf = Distributions.logccdf(normal, eta)
    mills_lower = exp(logphi - logcdf)
    mills_upper = exp(logphi - logccdf)
    d_mills_lower = -eta * mills_lower - mills_lower^2
    d_mills_upper = -eta * mills_upper + mills_upper^2
    d2_mills_lower = -mills_lower - (eta + 2mills_lower) * d_mills_lower
    d2_mills_upper = -mills_upper + (-eta + 2mills_upper) * d_mills_upper
    value = -aux.logchoose[i] - s * logcdf - (n - s) * logccdf
    d1 = -s * mills_lower + (n - s) * mills_upper
    d2 = s * (eta * mills_lower + mills_lower^2) +
         (n - s) * (-eta * mills_upper + mills_upper^2)
    d3 = -s * d2_mills_lower + (n - s) * d2_mills_upper
    return value, d1, d2, d3
end

# For the complementary-log-log link, let x = exp(eta) and
# A = x / expm1(x). The observed negative log likelihood is
# -s*log(1-exp(-x)) + (n-s)*x - logchoose, with derivatives
# -s*A + (n-s)*x, -s*A' + (n-s)*x, and -s*A'' + (n-s)*x.
# Series near x=0 avoid cancellation in A' and A''; the large-x branch avoids
# overflow in expm1(x) and its polynomial derivative factors.
function _binomial_cloglog_v123(aux, i, eta)
    s, n = aux.s[i], aux.ntr[i]
    failures = n - s
    x = exp(eta)
    if iszero(x)
        logp = eta
        A, A1, A2 = 1.0, 0.0, 0.0
    elseif x < 1e-4
        x2 = x * x
        A = 1 - x / 2 + x2 / 12 - x2^2 / 720 + x2^3 / 30240
        A1 = -x / 2 + x2 / 6 - x2^2 / 180 + x2^3 / 5040
        A2 = -x / 2 + x2 / 3 - x2^2 / 45 + x2^3 / 840
        logp = log(-expm1(-x))
    elseif x > 1e150
        # A and its derivatives have underflowed to zero at this scale. Avoid
        # forming x^2, which would overflow, while retaining the finite limits.
        A, A1, A2 = 0.0, 0.0, 0.0
        logp = 0.0
    elseif x > 40
        A = x * exp(-x)
        A1 = A * (1 - x)
        A2 = A * (1 - 3x + x^2)
        logp = log1p(-exp(-x))
    else
        A = x / expm1(x)
        A1 = A * (1 - x - A)
        A2 = A1 * (1 - x - 2A) - A * x
        logp = log(-expm1(-x))
    end
    failure_term = iszero(failures) ? 0.0 : failures * x
    value = -s * logp + failure_term - aux.logchoose[i]
    d1 = -s * A + failure_term
    d2 = -s * A1 + failure_term
    d3 = -s * A2 + failure_term
    return value, d1, d2, d3
end

const _BinomialLinkLaplaceAux = NamedTuple{(:s, :ntr, :logchoose, :link)}

function _binomial_link_v123(aux, i, eta)
    aux.link === :probit && return _binomial_probit_v123(aux, i, eta)
    aux.link === :cloglog && return _binomial_cloglog_v123(aux, i, eta)
    throw(ArgumentError("link-aware Binomial Laplace auxiliaries require `link = :probit` or `:cloglog`."))
end

_laplace_v123(::Val{:binomial}, aux::_BinomialLinkLaplaceAux, i, eta) =
    _binomial_link_v123(aux, i, eta)

_laplace_value(::Val{:binomial}, aux::_BinomialLinkLaplaceAux, i, eta) =
    first(_binomial_link_v123(aux, i, eta))
_laplace_d1(::Val{:binomial}, aux::_BinomialLinkLaplaceAux, i, eta) =
    _binomial_link_v123(aux, i, eta)[2]
_laplace_d2(::Val{:binomial}, aux::_BinomialLinkLaplaceAux, i, eta) =
    _binomial_link_v123(aux, i, eta)[3]
_laplace_d3(::Val{:binomial}, aux::_BinomialLinkLaplaceAux, i, eta) =
    _binomial_link_v123(aux, i, eta)[4]
function _laplace_d12(::Val{:binomial}, aux::_BinomialLinkLaplaceAux, i, eta)
    _, d1, d2, _ = _binomial_link_v123(aux, i, eta)
    return d1, d2
end

function _binomial_sd_scale_derivative(eta)
    upper_slope = 1 .- tanh.((eta .- 12.0) ./ 3.0).^2
    lower_slope = 1 .- tanh.((-12.0 .- eta) ./ 3.0).^2
    log_slope = ifelse.(eta .> 12.0, upper_slope,
        ifelse.(eta .< -12.0, lower_slope, 1.0))
    return _binomial_sd_scale(eta) .* log_slope
end

# Two independent ordinary effects on distinct grouping factors, each with a
# separate group-level SD regression. A factor may contribute an intercept or
# one numeric slope. The standardized latent coefficients have prior N(0, I);
# each observation loading is its coefficient design value times the
# group-specific SD. The outer score differentiates the Laplace determinant
# analytically, including the latent-mode response; ForwardDiff is used only
# for the final observed Hessian, never through the latent mode.
function _fit_binomial_crossed_sd_laplace(fam::Binomial, s, ntr, Xmu, Zg, nmsg,
                                          gidx, G, hidx, Hh, loading_g, loading_h,
                                          nmmu, groups, g_tol; se::Bool = true,
                                          target_blocks = [:sd_g, :sd_h])
    n = length(s)
    pμ = size(Xmu, 2)
    pgs, phs = size(Zg[1], 2), size(Zg[2], 2)
    length(s) == length(ntr) == length(gidx) == length(hidx) == n ||
        throw(ArgumentError("crossed Binomial SD dimensions do not match the response"))
    length(loading_g) == length(loading_h) == n || throw(ArgumentError(
        "crossed Binomial coefficient loadings must match the response length"))
    size(Zg[1], 1) == G && size(Zg[2], 1) == Hh ||
        throw(ArgumentError("crossed Binomial SD designs must have one row per level"))
    length(target_blocks) == 2 || throw(ArgumentError("crossed Binomial SD fitting needs two target blocks"))
    sint, nint = round.(Int, s), round.(Int, ntr)
    lc = [_logchoose(nint[i], sint[i]) for i in 1:n]
    aux = fam.link in (:probit, :cloglog) ?
        (s = sint, ntr = nint, logchoose = lc, link = fam.link) :
        (s = sint, ntr = nint, logchoose = lc)
    q = G + Hh
    last_u = zeros(q)
    group_ranges = (pμ + 1:pμ + pgs, pμ + pgs + 1:pμ + pgs + phs)

    function scales_at(theta)
        eta_g = Zg[1] * @view(theta[group_ranges[1]])
        eta_h = Zg[2] * @view(theta[group_ranges[2]])
        logtau_g = _softclamp.(eta_g, -12.0, 12.0, 3.0)
        logtau_h = _softclamp.(eta_h, -12.0, 12.0, 3.0)
        return exp.(logtau_g), exp.(logtau_h), logtau_g, logtau_h
    end

    function mode(theta; start = last_u)
        tau_g, tau_h, _, _ = scales_at(theta)
        eta0 = Xmu * @view(theta[1:pμ])
        u = copy(start)
        function joint(uu)
            val = 0.5 * dot(uu, uu)
            @inbounds for i in 1:n
                ag = tau_g[gidx[i]] * loading_g[i]
                ah = tau_h[hidx[i]] * loading_h[i]
                eta = eta0[i] + ag * uu[gidx[i]] + ah * uu[G + hidx[i]]
                val += _laplace_value(Val(:binomial), aux, i, eta)
            end
            return val
        end
        ch = nothing
        for _ in 1:60
            grad = copy(u)
            diagH = ones(Float64, q)
            crossw = Vector{Float64}(undef, n)
            all_w_nonneg = true
            @inbounds for i in 1:n
                gi, hi = gidx[i], G + hidx[i]
                ag = tau_g[gidx[i]] * loading_g[i]
                ah = tau_h[hidx[i]] * loading_h[i]
                eta = eta0[i] + ag * u[gi] + ah * u[hi]
                _, r, w, _ = _laplace_v123(Val(:binomial), aux, i, eta)
                grad[gi] += ag * r
                grad[hi] += ah * r
                diagH[gi] += ag^2 * w
                diagH[hi] += ah^2 * w
                crossw[i] = ag * ah * w
                w < 0 && (all_w_nonneg = false)
            end
            Hmat = _crossed_hessian(diagH, crossw, gidx, G, hidx, Hh)
            ch = cholesky(Symmetric(Hmat); check = false)
            issuccess(ch) || return u, ch, false
            step = ch \ grad
            sn = norm(step)
            sn <= 1e-9 * (1 + norm(u)) && return u, ch, true
            if all_w_nonneg && sn <= 1e-3 * (1 + norm(u))
                u .-= step
            else
                old = joint(u)
                rate = 1.0
                accepted = false
                while rate >= 1e-5
                    trial = u .- rate .* step
                    if joint(trial) <= old
                        u .= trial
                        accepted = true
                        break
                    end
                    rate *= 0.5
                end
                accepted || return u, ch, false
            end
        end
        return u, ch, false
    end

    function eval_laplace(theta; gradient::Bool = false)
        tau_g, tau_h, logtau_g, logtau_h = scales_at(theta)
        u, ch, ok = mode(theta)
        if !ok
            u, ch, ok = mode(theta; start = zeros(q))
        end
        ok || return gradient ? (1e18, zeros(length(theta))) : 1e18
        last_u .= u
        eta0 = Xmu * @view(theta[1:pμ])
        vstore, rstore, wstore, tstore = (Vector{Float64}(undef, n) for _ in 1:4)
        eta = Vector{Float64}(undef, n)
        value = 0.5 * dot(u, u)
        @inbounds for i in 1:n
            gi, hi = gidx[i], hidx[i]
            ag = tau_g[gi] * loading_g[i]
            ah = tau_h[hi] * loading_h[i]
            eta[i] = eta0[i] + ag * u[gi] + ah * u[G + hi]
            v, r, w, t = _laplace_v123(Val(:binomial), aux, i, eta[i])
            value += v
            vstore[i], rstore[i], wstore[i], tstore[i] = v, r, w, t
        end
        value += 0.5 * logdet(ch)
        gradient || return value

        hd, crossinv = _crossed_selected_inverse_entries(ch, gidx, G, hidx, Hh)
        qform = Vector{Float64}(undef, n)
        @inbounds for i in 1:n
            gi, hi = gidx[i], hidx[i]
            ag, ah = tau_g[gi] * loading_g[i], tau_h[hi] * loading_h[i]
            qform[i] = ag^2 * hd[gi] + ah^2 * hd[G + hi] +
                       2 * ag * ah * crossinv[i]
        end

        grad = zeros(Float64, length(theta))
        # Fixed-effect derivative: the mode response cancels from the joint
        # term. It remains in dH through the third derivative of the likelihood.
        for k in 1:pμ
            direct = @view Xmu[:, k]
            dmode_score = zeros(Float64, q)
            @inbounds for i in 1:n
                ag = tau_g[gidx[i]] * loading_g[i]
                ah = tau_h[hidx[i]] * loading_h[i]
                dmode_score[gidx[i]] += ag * wstore[i] * direct[i]
                dmode_score[G + hidx[i]] += ah * wstore[i] * direct[i]
            end
            du = -(ch \ dmode_score)
            score = 0.0
            @inbounds for i in 1:n
                ag = tau_g[gidx[i]] * loading_g[i]
                ah = tau_h[hidx[i]] * loading_h[i]
                total_eta = direct[i] + ag * du[gidx[i]] + ah * du[G + hidx[i]]
                score += rstore[i] * direct[i] + 0.5 * tstore[i] * total_eta * qform[i]
            end
            grad[k] = score
        end

        # A group SD coefficient changes both its observation loading and its
        # latent mode. Differentiate the mode equation, then the selected
        # inverse entries in log|H| without differentiating through the solve.
        for (axis, (idx, nlev, Z, logtau, tau, range)) in enumerate(
                ((gidx, G, Zg[1], logtau_g, tau_g, group_ranges[1]),
                 (hidx, Hh, Zg[2], logtau_h, tau_h, group_ranges[2])))
            raw_sd_predictor = Z * @view(theta[range])
            dclamp = [ForwardDiff.derivative(x -> _softclamp(x, -12.0, 12.0, 3.0),
                                              raw_sd_predictor[j]) for j in 1:nlev]
            for k in 1:size(Z, 2)
                direct_loading = Vector{Float64}(undef, n)
                direct_eta = Vector{Float64}(undef, n)
                dmode_score = zeros(Float64, q)
                @inbounds for i in 1:n
                    gi, hi = gidx[i], hidx[i]
                    level = idx[i]
                    loading = axis == 1 ? loading_g[i] : loading_h[i]
                    direct_loading[i] = loading * tau[level] * dclamp[level] * Z[level, k]
                    direct_eta[i] = direct_loading[i] * u[axis == 1 ? gi : G + hi]
                    dmode_score[gi] += (axis == 1 ? direct_loading[i] * rstore[i] : 0.0)
                    dmode_score[G + hi] += (axis == 2 ? direct_loading[i] * rstore[i] : 0.0)
                    dmode_score[gi] += tau_g[gi] * loading_g[i] * wstore[i] * direct_eta[i]
                    dmode_score[G + hi] += tau_h[hi] * loading_h[i] * wstore[i] * direct_eta[i]
                end
                du = -(ch \ dmode_score)
                score = 0.0
                @inbounds for i in 1:n
                    gi, hi = gidx[i], hidx[i]
                    ag, ah = tau_g[gi] * loading_g[i], tau_h[hi] * loading_h[i]
                    total_eta = direct_eta[i] + ag * du[gi] + ah * du[G + hi]
                    if axis == 1
                        score += rstore[i] * direct_eta[i] +
                            wstore[i] * direct_loading[i] * (ag * hd[gi] + ah * crossinv[i])
                    else
                        score += rstore[i] * direct_eta[i] +
                            wstore[i] * direct_loading[i] * (ah * hd[G + hi] + ag * crossinv[i])
                    end
                    score += 0.5 * tstore[i] * total_eta * qform[i]
                end
                grad[first(range) + k - 1] = score
            end
        end
        return value, grad
    end

    nll(theta) = eval_laplace(theta; gradient = false)
    function grad!(out, theta)
        _, g = eval_laplace(theta; gradient = true)
        out .= g
        return out
    end
    theta0 = zeros(pμ + pgs + phs)
    pbar = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)
    theta0[1] = _binomial_start(fam, pbar)
    theta0[group_ranges[1]] .= Zg[1] \ fill(log(0.45), G)
    theta0[group_ranges[2]] .= Zg[2] \ fill(log(0.45), Hh)
    od = Optim.OnceDifferentiable(nll, grad!, theta0)
    res = Optim.optimize(od, theta0, Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
                         Optim.Options(g_tol = g_tol, iterations = 350))
    theta_hat = Optim.minimizer(res)
    final_grad = zeros(length(theta_hat))
    grad!(final_grad, theta_hat)
    nll_hat = nll(theta_hat)
    converged = _laplace_outer_converged(res, nll_hat, final_grad, theta_hat, n, g_tol)
    V = if se
        hstep = _fd_hessian_step(n)
        Htheta = _finite_hessian(nll, theta_hat; h = hstep)
        _vcov_from_fd_hessian(Htheta; context = "crossed Binomial SD regression")
    else
        fill(NaN, length(theta_hat), length(theta_hat))
    end
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => (1:pμ),
        target_blocks[1] => group_ranges[1], target_blocks[2] => group_ranges[2]]
    names = Pair{Symbol,Vector{String}}[:mu => nmmu,
        target_blocks[1] => nmsg[1], target_blocks[2] => nmsg[2]]
    mu_eta = Xmu * theta_hat[1:pμ]
    means = Dict(:mu => _logistic.(mu_eta))
    obs = Dict(:mu => s ./ ntr)
    tau_g, tau_h, _, _ = scales_at(theta_hat)
    sd_g = _binomial_sd_scale(Zg[1] * theta_hat[group_ranges[1]])
    sd_h = _binomial_sd_scale(Zg[2] * theta_hat[group_ranges[2]])
    scales = Dict{Symbol,Vector{Float64}}(
        :trials => Float64.(nint),
        Symbol("sd_group_index_", groups[1]) => Float64.(gidx),
        Symbol("sd_group_index_", groups[2]) => Float64.(hidx),
        target_blocks[1] => sd_g[gidx],
        target_blocks[2] => sd_h[hidx],
    )
    fit = DrmFit(fam, blocks, names, theta_hat, Matrix(V), -nll_hat, n,
        converged, means, obs, scales)
    for (label, eta_sd) in ((groups[1], Zg[1] * theta_hat[group_ranges[1]]),
                            (groups[2], Zg[2] * theta_hat[group_ranges[2]]))
        _warn_at_clamp("Binomial SD regression",
            ["$label random-effect log SD" => (collect(Float64, eta_sd), -12.0, 12.0)])
    end
    return _withiterations(_withnll(fit, _nll_pinned(nll, theta_hat, nll_hat), grad!), Optim.iterations(res))
end

function _binomial_phylo_sd_slope_name(lhs, group::Symbol)
    lhs isa ConstantTerm && lhs.n == 1 && return nothing
    if lhs isa FunctionTerm && lhs.f === (+)
        constants = filter(t -> t isa ConstantTerm, lhs.args)
        variables = filter(t -> t isa Term, lhs.args)
        is_slope_only = length(lhs.args) == 2 && length(constants) == 1 &&
            constants[1].n == 0 && length(variables) == 1
        is_slope_only && return variables[1].sym
    end
    throw(ArgumentError(
        "Binomial phylogenetic SD regression supports `phylo(1 | group)` or one numeric slope in `phylo(0 + x | group)`."))
end

function _drm_binomial_phylo_sd_regression(f::DrmFormula, fam::Binomial; data,
                                          tree, g_tol::Real = 1e-8,
                                          se::Bool = true)
    sdpp = _sdphylo_parts(f)
    length(sdpp) == 1 || throw(ArgumentError(
        "Binomial phylogenetic SD regression requires exactly one `sd(group, phylogenetic) ~ …` formula."))
    isempty(_sd_parts(f)) || throw(ArgumentError(
        "Binomial phylogenetic SD regression cannot be combined with an ordinary `sd(group)` formula yet."))
    sgrp, srhs = only(sdpp)
    rhs = Dict(f.forms)
    mu_terms = rhs[:mu] isa Tuple ? collect(rhs[:mu]) : Any[rhs[:mu]]
    phylo_terms = filter(t -> t isa FunctionTerm && t.f === phylo, mu_terms)
    length(phylo_terms) == 1 || throw(ArgumentError(
        "Binomial phylogenetic SD regression requires exactly one phylo() term."))
    phylo_term = only(phylo_terms)
    phylo_re = phylo_term.args[1]
    species = phylo_re.args[2].sym
    slope_name = _binomial_phylo_sd_slope_name(phylo_re.args[1], species)
    fixed_terms = filter(t -> !(t isa FunctionTerm && t.f === phylo), mu_terms)
    fixed_rhs = isempty(fixed_terms) ? ConstantTerm(1) :
        length(fixed_terms) == 1 ? only(fixed_terms) : Tuple(fixed_terms)
    fixed_mu, re, mv, st, _, _, _ = _split_ranef(fixed_rhs)
    mv === nothing || throw(ArgumentError("Binomial phylogenetic SD regression cannot use meta_V()."))
    st === nothing || throw(ArgumentError(
        "Binomial phylogenetic SD regression currently supports one phylo() term and no other structured effect."))
    tree === nothing && throw(ArgumentError("phylo(1 | $sgrp) needs `tree = …`"))
    isempty(re) || throw(ArgumentError(
        "Binomial phylogenetic random-slope SD regression currently requires no additional random effects."))
    species === sgrp || throw(ArgumentError(
        "Binomial `sd($sgrp, phylogenetic)` must match the group in its phylo() term."))
    for (pname, _) in f.forms
        (pname === :mu || pname === Symbol("sdphy_", sgrp)) && continue
        pname === :sigma && Dict(f.forms)[pname] == ConstantTerm(1) && continue
        throw(ArgumentError("Binomial phylogenetic SD regression supports only `mu` and one phylogenetic SD formula."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Binomial phylogenetic SD regression requires observed responses; remove incomplete rows before fitting."))

    s, ntr = _binomial_response(f, data)
    _, Xmu, nmmu = _design(f.response, fixed_mu, data)
    labels = getproperty(data, sgrp)
    _, tip_index, G = _lss_phylo_group_index(tree, labels, sgrp)
    group_index, Gdata = _group_index(labels)
    Gdata == G || error("phylogenetic group indexing changed unexpectedly")
    Zg, nmsd = _sd_group_design(f.response, srhs, data, group_index, Gdata, sgrp;
                                schema_cache = f.schema_cache, schema_key = :sd_phylo)
    tip_group = zeros(Int, G)
    @inbounds for i in eachindex(tip_index)
        ti, gi = tip_index[i], group_index[i]
        tip_group[ti] == 0 || tip_group[ti] == gi || throw(ArgumentError(
            "Binomial phylogenetic SD predictors must be constant within each tree tip."))
        tip_group[ti] = gi
    end
    all(!iszero, tip_group) || throw(ArgumentError(
        "Binomial phylogenetic SD regression requires SD predictors for every tree tip."))
    Ztip = Matrix{Float64}(Zg[tip_group, :])
    Zobs = Ztip[tip_index, :]
    re_design = if slope_name === nothing
        ones(Float64, length(s))
    else
        hasproperty(data, slope_name) || throw(ArgumentError(
            "Binomial phylogenetic random-slope predictor `$slope_name` is missing from data."))
        raw_slope = getproperty(data, slope_name)
        raw_slope isa AbstractVector{<:Real} && length(raw_slope) == length(s) &&
            all(isfinite, raw_slope) || throw(ArgumentError(
            "Binomial phylogenetic random-slope predictor `$slope_name` must be numeric, finite, and have one value per response."))
        any(!iszero, raw_slope) || throw(ArgumentError(
            "Binomial phylogenetic random-slope predictor `$slope_name` is zero for every observation, so its SD is not identifiable."))
        Float64.(raw_slope)
    end
    return _withformula(_fit_binomial_phylo_sd_laplace(
        fam, s, ntr, Xmu, Zobs, labels, tree, nmmu, nmsd, sgrp, g_tol;
        se, re_design,
    ), f)
end

function _fit_binomial_phylo_interaction_sd_laplace(fam, s, ntr, Xμ, Zobs,
        leaf_node, Q, nmμ, nmsd, block, g_tol; se::Bool = true,
        re_design = ones(Float64, length(s)), maxiter::Int = 250)
    n = length(s)
    sint, nint = round.(Int, s), round.(Int, ntr)
    logchoose = [_logfactorial(nint[i]) - _logfactorial(sint[i]) -
                 _logfactorial(nint[i] - sint[i]) for i in eachindex(sint)]
    aux = fam.link in (:probit, :cloglog) ?
        (s = sint, ntr = nint, logchoose = logchoose, link = fam.link) :
        (s = sint, ntr = nint, logchoose = logchoose)
    length(leaf_node) == n || throw(ArgumentError(
        "phylogenetic-interaction latent indices must have one entry per response"))
    size(Zobs, 1) == n || throw(ArgumentError(
        "phylogenetic-interaction SD design must have one row per response"))
    size(Zobs, 2) == length(nmsd) || throw(ArgumentError(
        "phylogenetic-interaction SD names do not match its design"))
    qchol = cholesky(Symmetric(Q); check = false)
    issuccess(qchol) || throw(ArgumentError(
        "the Kronecker phylogenetic-interaction precision is not positive definite"))
    logdetQ = logdet(qchol)
    pμ, psd = size(Xμ, 2), size(Zobs, 2)
    last_b = zeros(size(Q, 1))
    function nll(θ)
        value, b, ok = _binomial_phylo_sd_laplace(aux, n, Xμ, Zobs,
            leaf_node, Q, logdetQ, θ; b0 = last_b, re_design)
        if !ok
            value, b, ok = _binomial_phylo_sd_laplace(aux, n, Xμ, Zobs,
                leaf_node, Q, logdetQ, θ; b0 = zeros(length(last_b)),
                re_design)
        end
        ok || return 1e18
        last_b .= b
        return value
    end
    function grad!(g, θ)
        for j in eachindex(θ)
            h = 2e-4 * (1 + abs(θ[j]))
            e = zeros(length(θ)); e[j] = h
            g[j] = (nll(θ .+ e) - nll(θ .- e)) / (2h)
        end
        return g
    end
    pbar = clamp(sum(s) / max(sum(ntr), 1), 1e-4, 1 - 1e-4)
    θ0 = zeros(pμ + psd)
    θ0[1] = _binomial_start(fam, pbar)
    θ0[pμ + 1] = log(0.4)
    od = Optim.OnceDifferentiable(nll, grad!, θ0)
    res = Optim.optimize(od, θ0,
        Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
        Optim.Options(g_tol = g_tol, iterations = maxiter))
    θhat = Optim.minimizer(res)
    nll_hat = nll(θhat)
    gfinal = zeros(length(θhat)); grad!(gfinal, θhat)
    converged = _laplace_outer_converged(res, nll_hat, gfinal, θhat, n, g_tol)
    V = se ? _vcov_from_fd_hessian(
        _finite_hessian(nll, θhat; h = _fd_hessian_step(n));
        context = "sparse-Laplace Binomial phylogenetic-interaction SD regression") :
        fill(NaN, length(θhat), length(θhat))
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => (1:pμ), block => (pμ+1:pμ+psd)]
    names = Pair{Symbol,Vector{String}}[:mu => nmμ, block => nmsd]
    means = Dict(:mu => _binomial_mean.(Ref(fam), Xμ * θhat[1:pμ]))
    obs = Dict(:mu => s ./ ntr)
    sdobs = _binomial_sd_scale(Zobs * θhat[pμ+1:end])
    scales = Dict(:trials => Float64.(nint), :sd => sdobs, block => sdobs)
    fit = DrmFit(fam, blocks, names, θhat, Matrix(V), -nll_hat, n,
                 converged, means, obs, scales)
    _warn_at_clamp("Binomial phylogenetic-interaction SD regression",
        ["$(block) random-effect log SD" =>
            (collect(Float64, Zobs * θhat[pμ+1:end]), -12.0, 12.0)])
    return _withiterations(_withnll(fit, _nll_pinned(nll, θhat, nll_hat)), Optim.iterations(res))
end

function _drm_binomial_phylo_interaction_sd_regression(f::DrmFormula,
        fam::Binomial; data, tree1, tree2, tree = nothing, K = nothing,
        A = nothing, Ainv = nothing, coords = nothing,
        g_tol::Real = 1e-8, se::Bool = true)
    targets = filter(item -> item.target.level === :phylogenetic_interaction,
                     _sd_target_parts(f))
    length(targets) == 1 || throw(ArgumentError(
        "Binomial phylogenetic-interaction SD regression requires exactly one " *
        "`sd(pair, phylogenetic_interaction) ~ …` formula."))
    target = only(targets)
    marker = _binomial_phylo_interaction_parts(Dict(f.forms)[:mu])
    marker === nothing && throw(ArgumentError(
        "`sd($(target.target.group), phylogenetic_interaction)` requires " *
        "a matching `phylo_interaction()` term in the location formula."))
    (target.target.dpar === :mu && target.target.member === marker.slope &&
     target.target.block === nothing) || throw(ArgumentError(
        "Binomial phylogenetic-interaction SD must target the matching location coefficient."))
    tree1 === nothing && throw(ArgumentError(
        "`phylo_interaction(1 | $(marker.group1) & $(marker.group2))` needs `tree1 = …`."))
    tree2 === nothing && throw(ArgumentError(
        "`phylo_interaction(1 | $(marker.group1) & $(marker.group2))` needs `tree2 = …`."))
    tree === nothing && K === nothing && A === nothing && Ainv === nothing &&
        coords === nothing || throw(ArgumentError(
        "Binomial phylogenetic-interaction SD uses `tree1` and `tree2` only."))
    length(_sd_target_parts(f)) == 1 && isempty(_sdphylo_parts(f)) ||
        throw(ArgumentError(
        "Binomial phylogenetic-interaction SD currently supports one SD target only."))
    for (pname, _) in f.forms
        (pname === :mu || pname === target.key) && continue
        pname === :sigma && Dict(f.forms)[pname] == ConstantTerm(1) && continue
        throw(ArgumentError(
            "Binomial phylogenetic-interaction SD supports only `mu` and one SD formula."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Binomial phylogenetic-interaction SD requires observed responses; remove incomplete rows before fitting."))
    host = getproperty(data, marker.group1)
    pollinator = getproperty(data, marker.group2)
    pair = getproperty(data, target.target.group)
    (any(ismissing, host) || any(ismissing, pollinator) || any(ismissing, pair)) &&
        throw(ArgumentError("phylogenetic-interaction grouping and pair columns cannot contain missing values."))
    length(host) == length(pollinator) == length(pair) || throw(ArgumentError(
        "phylogenetic-interaction grouping and pair columns must have the same number of rows."))
    pair_index, Gpair = _group_index(pair)
    pair_to_combination = Dict{Int,Tuple{String,String}}()
    combination_to_pair = Dict{Tuple{String,String},Int}()
    for i in eachindex(pair_index)
        combo = (string(host[i]), string(pollinator[i]))
        gi = pair_index[i]
        if haskey(pair_to_combination, gi) && pair_to_combination[gi] != combo
            throw(ArgumentError("pair ID `$(target.target.group)` must identify one host-pollinator combination."))
        end
        if haskey(combination_to_pair, combo) && combination_to_pair[combo] != gi
            throw(ArgumentError("each host-pollinator combination must have one unique pair ID."))
        end
        pair_to_combination[gi] = combo
        combination_to_pair[combo] = gi
    end
    s, ntr = _binomial_response(f, data)
    fixed_mu, re, mv, structured, _, _, _ = _split_ranef(marker.fixed_rhs)
    isempty(re) && mv === nothing && structured === nothing || throw(ArgumentError(
        "Binomial phylogenetic-interaction SD currently supports no additional random or structured effects."))
    _, Xμ, nmμ = _design(f.response, fixed_mu, data)
    Q1, node1, _ = _poisson_phylo_setup(tree1, host)
    Q2, node2, _ = _poisson_phylo_setup(tree2, pollinator)
    length(node1) == length(s) == length(node2) || throw(ArgumentError(
        "the two tree indices must align with the retained Binomial response rows."))
    Q = kron(Q2, Q1)
    leaf_node = (node2 .- 1) .* size(Q1, 1) .+ node1
    re_design = if marker.slope === nothing
        ones(Float64, length(s))
    else
        hasproperty(data, marker.slope) || throw(ArgumentError(
            "Binomial phylogenetic-interaction slope predictor `$(marker.slope)` is missing from data."))
        raw_slope = getproperty(data, marker.slope)
        raw_slope isa AbstractVector{<:Real} && length(raw_slope) == length(s) &&
            all(isfinite, raw_slope) || throw(ArgumentError(
            "Binomial phylogenetic-interaction slope predictor `$(marker.slope)` must be numeric, finite, and have one value per response."))
        any(!iszero, raw_slope) || throw(ArgumentError(
            "Binomial phylogenetic-interaction slope predictor `$(marker.slope)` is zero for every observation, so its SD is not identifiable."))
        Float64.(raw_slope)
    end
    Zpair, nmsd = _sd_group_design(f.response, target.rhs, data,
        pair_index, Gpair, target.target.group; schema_cache = f.schema_cache,
        schema_key = target.key)
    Zobs = Zpair[pair_index, :]
    block = _sd_target_block(target.target; disambiguate = false)
    fit = _fit_binomial_phylo_interaction_sd_laplace(fam, s, ntr, Xμ,
        Zobs, leaf_node, Q, nmμ, nmsd, block, g_tol; se, re_design)
    return _withformula(fit, f)
end

const _BINOMIAL_Q2_GAMMA_CAP = 10.0

function _binomial_animal_q2_mode(aux, s, ntr, Xmu, x, gidx, Qjoint,
        load0, load1, theta; b0 = nothing, maxiter::Int = 80,
        tol::Real = 1e-10)
    G = length(load0)
    beta = @view theta[1:size(Xmu, 2)]
    b = b0 === nothing ? zeros(2G) : copy(b0)
    eta0 = Xmu * beta
    function joint(bb)
        value = 0.5 * dot(bb, Qjoint * bb)
        @inbounds for i in eachindex(s)
            gi = gidx[i]
            value += _laplace_value(Val(:binomial), aux, i,
                eta0[i] + load0[gi] * bb[2gi - 1] +
                    load1[i] * bb[2gi])
        end
        value
    end
    # Gradient and Hessian of the joint objective at `bb`.
    function newton_parts(bb)
        grad = Qjoint * bb
        I = Vector{Int}(undef, 4length(s))
        J = similar(I)
        V = Vector{Float64}(undef, length(I))
        @inbounds for i in eachindex(s)
            gi = gidx[i]
            j0, j1 = 2gi - 1, 2gi
            l0, l1 = load0[gi], load1[i]
            eta = eta0[i] + l0 * bb[j0] + l1 * bb[j1]
            _, d1, d2, _ = _laplace_v123(Val(:binomial), aux, i, eta)
            grad[j0] += l0 * d1
            grad[j1] += l1 * d1
            k = 4(i - 1)
            I[k+1], J[k+1], V[k+1] = j0, j0, d2 * l0^2
            I[k+2], J[k+2], V[k+2] = j0, j1, d2 * l0 * l1
            I[k+3], J[k+3], V[k+3] = j1, j0, d2 * l0 * l1
            I[k+4], J[k+4], V[k+4] = j1, j1, d2 * l1^2
        end
        return grad, Qjoint + sparse(I, J, V, 2G, 2G)
    end
    for _ in 1:maxiter
        grad, H = newton_parts(b)
        ch = cholesky(Symmetric(H); check = false)
        issuccess(ch) || return b, ch, false
        step = ch \ grad
        norm(step) <= tol * (1 + norm(b)) && return b, ch, true
        if norm(step) <= 1e-7 * (1 + norm(b))
            # The Laplace value is second order in the mode error but logdet(H)
            # is first order, so returning `ch` at the un-stepped `b` left an
            # error of about 1e-7 in the objective. Take the Newton step (its
            # error is quadratic) and refactor at the stepped mode.
            b = b .- step
            _, H = newton_parts(b)
            ch = cholesky(Symmetric(H); check = false)
            return b, ch, issuccess(ch)
        end
        f0 = joint(b)
        amount, accepted = 1.0, false
        while amount >= 1e-5
            trial = b .- amount .* step
            if joint(trial) <= f0
                b = trial
                accepted = true
                break
            end
            amount *= 0.5
        end
        if !accepted
            # Noise floor: with a near-singular joint precision (correlation near
            # +-1 once the slope SD has collapsed) the Newton step is only accurate
            # to roughly cond * eps, so no trial can lower `joint` although the
            # mode is already pinned down. The predicted gain of the full step,
            # 0.5 * grad' * step, bounds the remaining objective error; when it is
            # at the rounding level the mode is converged. Before this the solve
            # returned failure here and the 1e18 sentinel leaked into the
            # finite-difference gradient (weak-optimum case c). Changes the inner
            # solve itself (not only restarts): Binomial "weak" coefficients moved
            # by up to 0.28 on fits that never restart.
            0.5 * dot(grad, step) <= 1e-9 * (1 + abs(f0)) && return b, ch, true
            return b, ch, false
        end
    end
    return b, nothing, false
end

function _binomial_animal_q2_laplace(aux, s, ntr, Xmu, Xsd0, Xsd1,
        gidx, Q, logdetQ, x, theta; b0 = nothing, correlated::Bool = true)
    pmu = size(Xmu, 2)
    p0, p1 = size(Xsd0, 2), size(Xsd1, 2)
    r0 = (pmu + 1):(pmu + p0)
    r1 = (last(r0) + 1):(last(r0) + p1)
    rc = last(r1) + 1
    # `correlated = false`: two independent fields (rho fixed 0, no atanh-rho parameter).
    # The objective is held constant beyond |gamma| = `_BINOMIAL_Q2_GAMMA_CAP`
    # (rho = 1 - 4e-9): there the joint precision has condition number ~ e^(2|gamma|),
    # the Laplace value is rounding noise of order 1e-3 that depends on the BLAS/LAPACK
    # build (Julia 1.10 vs 1.12), and a finite-difference L-BFGS run exploits that noise
    # and reports a spuriously lower objective (weak-optimum case c on Julia 1.12).
    gamma = correlated ? clamp(theta[rc], -_BINOMIAL_Q2_GAMMA_CAP, _BINOMIAL_Q2_GAMMA_CAP) : 0.0
    rho = tanh(gamma)
    sech = _binomial_stable_sech(gamma)
    den = sech^2
    isfinite(den) && den > 0 || return 1e18, zeros(2size(Q, 1)), false
    Rinv = [1.0 -rho; -rho 1.0] ./ den
    Qjoint = kron(Q, sparse(Rinv))
    logdetQjoint = 2logdetQ - 2size(Q, 1) * log(sech)
    alpha0, alpha1 = @view(theta[r0]), @view(theta[r1])
    scale0 = _binomial_sd_scale(Xsd0 * alpha0)
    scale1 = _binomial_sd_scale(Xsd1 * alpha1)
    load1 = x .* scale1[gidx]
    all(isfinite, scale0) && all(isfinite, load1) && isfinite(logdetQjoint) ||
        return 1e18, zeros(2size(Q, 1)), false
    b, ch, ok = _binomial_animal_q2_mode(aux, s, ntr, Xmu, x, gidx,
        Qjoint, scale0, load1, theta; b0)
    ok || return 1e18, b, false
    beta = @view theta[1:pmu]
    eta0 = Xmu * beta
    value = 0.5 * dot(b, Qjoint * b)
    @inbounds for i in eachindex(s)
        gi = gidx[i]
        value += _laplace_value(Val(:binomial), aux, i,
            eta0[i] + scale0[gi] * b[2gi - 1] + load1[i] * b[2gi])
    end
    return value - 0.5 * logdetQjoint + 0.5 * logdet(ch), b, true
end

function _fit_binomial_animal_q2_laplace(fam, s, ntr, Xmu, Xsd0,
        Xsd1, gidx, Q, x, nmμ, nmsd0, nmsd1, blocks, group, g_tol;
        se::Bool = true, maxiter::Int = 250, correlated::Bool = true,
        second_start::Bool = false, hessian_step::Union{Nothing,Real} = nothing)
    G = size(Q, 1)
    qchol = cholesky(Symmetric(Q); check = false)
    issuccess(qchol) || throw(ArgumentError(
        "Binomial animal covariance precision must be positive definite."))
    logdetQ = logdet(qchol)
    sint, nint = round.(Int, s), round.(Int, ntr)
    logchoose = [_logfactorial(nint[i]) - _logfactorial(sint[i]) -
                 _logfactorial(nint[i] - sint[i]) for i in eachindex(sint)]
    aux = fam.link in (:probit, :cloglog) ?
        (s = sint, ntr = nint, logchoose = logchoose, link = fam.link) :
        (s = sint, ntr = nint, logchoose = logchoose)
    pmu, p0, p1 = size(Xmu, 2), size(Xsd0, 2), size(Xsd1, 2)
    ranges = (pmu + 1):(pmu + p0), (pmu + p0 + 1):(pmu + p0 + p1)
    rc = pmu + p0 + p1 + 1
    theta0 = zeros(correlated ? rc : rc - 1)
    pbar = clamp(sum(s) / max(sum(ntr), 1), 1e-4, 1 - 1e-4)
    theta0[1] = _binomial_start(fam, pbar)
    theta0[ranges[1]] .= Xsd0 \ fill(log(0.4), G)
    theta0[ranges[2]] .= Xsd1 \ fill(log(0.3), G)
    last_b = zeros(2G)
    function nll(theta)
        value, b, ok = _binomial_animal_q2_laplace(aux, s, ntr, Xmu,
            Xsd0, Xsd1, gidx, Q, logdetQ, x, theta; b0 = last_b, correlated)
        if !ok
            value, b, ok = _binomial_animal_q2_laplace(aux, s, ntr, Xmu,
                Xsd0, Xsd1, gidx, Q, logdetQ, x, theta; b0 = zeros(2G), correlated)
        end
        ok || return 1e18
        last_b .= b
        value
    end
    function grad!(g, theta)
        for j in eachindex(theta)
            h = 2e-4 * (1 + abs(theta[j]))
            e = zeros(length(theta)); e[j] = h
            g[j] = (nll(theta .+ e) - nll(theta .- e)) / (2h)
        end
        g
    end
    run_from(th0) = Optim.optimize(Optim.OnceDifferentiable(nll, grad!, th0), th0,
        Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
        Optim.Options(g_tol = g_tol, iterations = maxiter))
    result = run_from(theta0)
    if second_start
        # Opt-in second start (SD coefficients log 0.7), kept only if it improves the
        # objective by more than 1e-6; same device as the Poisson spatial q2 route.
        nll_first = nll(Optim.minimizer(result))
        b_first = copy(last_b)
        theta1 = copy(theta0)
        theta1[ranges[1]] .= Xsd0 \ fill(log(0.7), G)
        theta1[ranges[2]] .= Xsd1 \ fill(log(0.7), G)
        result2 = run_from(theta1)
        if nll(Optim.minimizer(result2)) < nll_first - 1e-6
            result = result2
        else
            last_b .= b_first
        end
    end
    # Boundary restarts (see `_q2_boundary_restarts`): a fit that ends with a
    # collapsed slope SD (and rho running to +-1) sits on a flat, noisy ridge, and
    # the default start can stop there unconverged or short of the interior optimum.
    restart_info = Ref{Any}((ran = false, winner = 0, gain = 0.0, flipped = false))
    result = _q2_boundary_restarts(run_from, last_b, result,
        (sd0, sd1, het) -> _q2_alt_theta(theta0, ranges, (Xsd0, Xsd1), sd0, sd1, het),
        th -> min(minimum(_binomial_sd_scale(Xsd0 * th[ranges[1]])),
                  minimum(_binomial_sd_scale(Xsd1 * th[ranges[2]]))),
        r -> begin
            th = Optim.minimizer(r); g = zeros(length(th)); grad!(g, th)
            _laplace_outer_converged(r, Optim.minimum(r), g, th, length(s), g_tol)
        end; info = restart_info)
    ri = restart_info[]
    (ri.gain > _Q2_RESTART_GAIN || ri.flipped) &&
        @info "drm: Binomial animal q2 boundary restart changed the accepted optimum " *
              "(objective gain $(round(ri.gain; sigdigits = 3)), converged flipped = $(ri.flipped), " *
              "winning start $(ri.winner) of $(length(_Q2_ALT_STARTS)); 0 = default start). " *
              "See `fit.scales[:q2_restarts]`."
    theta = Optim.minimizer(result)
    nll_hat = nll(theta)
    gf = zeros(length(theta)); grad!(gf, theta)
    converged = _laplace_outer_converged(result, nll_hat, gf, theta,
        length(s), g_tol)
    V = se ? _vcov_from_fd_hessian(
        _finite_hessian(nll, theta;
            h = hessian_step === nothing ? _fd_hessian_step(length(s)) : hessian_step);
        context = "sparse-Laplace Binomial animal q2 SD regression") :
        fill(NaN, length(theta), length(theta))
    block_list = Pair{Symbol,UnitRange{Int}}[
        :mu => 1:pmu, blocks[1] => ranges[1], blocks[2] => ranges[2]]
    name_list = Pair{Symbol,Vector{String}}[
        :mu => nmμ, blocks[1] => nmsd0, blocks[2] => nmsd1]
    if correlated
        push!(block_list, :recov => rc:rc)
        push!(name_list, :recov => ["$(group):atanh_rho"])
    end
    sd0 = _binomial_sd_scale(Xsd0 * theta[ranges[1]])
    sd1 = _binomial_sd_scale(Xsd1 * theta[ranges[2]])
    scales = Dict{Symbol,Vector{Float64}}(
        :trials => Float64.(nint),
        blocks[1] => sd0[gidx], blocks[2] => sd1[gidx],
        Symbol(string(blocks[1]), "_group") => sd0,
        Symbol(string(blocks[2]), "_group") => sd1,
    )
    correlated && (scales[:correlation] = [tanh(theta[rc])])
    # Restart bookkeeping (skipped by simulate/predict/the bridge, like the other
    # bookkeeping keys): [ran restarts (0/1), winning start (0 = default start, k = k-th
    # alternative of `_Q2_ALT_STARTS`), objective gain over the first run, converged flipped (0/1)].
    scales[:q2_restarts] = [ri.ran ? 1.0 : 0.0, Float64(ri.winner), Float64(ri.gain), ri.flipped ? 1.0 : 0.0]
    # b34: record the conditional correlated field at theta-hat (fresh inner
    # solve; `last_b` was moved by the finite-difference stencils above). The
    # joint mode interleaves the intercept and slope blocks per group.
    _, b_hat, ok_hat = _binomial_animal_q2_laplace(aux, s, ntr, Xmu, Xsd0, Xsd1,
        gidx, Q, logdetQ, x, theta; b0 = zeros(2G), correlated)
    if ok_hat
        _attach_conditional_structured!(scales,
            sd0[gidx] .* b_hat[2 .* gidx .- 1] .+ x .* sd1[gidx] .* b_hat[2 .* gidx],
            b_hat[1:2:end], b_hat[2:2:end])
    else
        _mark_cond_failed!(scales)
    end
    fit = DrmFit(fam, block_list, name_list, theta, Matrix(V), -nll_hat,
        length(s), converged,
        Dict(:mu => _binomial_mean.(Ref(fam), Xmu * theta[1:pmu])),
        Dict(:mu => s ./ ntr), scales)
    return _withiterations(_withnll(fit, _nll_pinned(nll, theta, nll_hat), grad!), Optim.iterations(result))
end

function _drm_binomial_animal_q2_sd_regression(f::DrmFormula, fam::Binomial;
        data, A, Ainv, tree, K, coords, g_tol::Real, se::Bool)
    targets = filter(item -> item.target.level === :animal,
                     _sd_target_parts(f))
    length(targets) == 2 || throw(ArgumentError(
        "Binomial animal q2 SD regression requires one intercept and one slope SD formula."))
    rhs = Dict(f.forms)
    # Unlabelled `animal(1 + x | g)`: two independent fields (rho fixed 0). Labelled
    # `animal(1 + x | p | g)`: one free cross-coefficient correlation. Native drmTMB
    # 0.8.1 fits the two as different models (logLik differs by 1.4 on the wave-8 data).
    correlated = something(_animal_intercept_slope(rhs[:mu]; allow_label = true),
                           (; label = nothing)).label !== nothing
    fixed_mu, re, mv, structured, _, _, slope = _split_ranef(
        rhs[:mu]; allow_animal_intercept_slope = true,
        allow_labelled_intercept_slope = correlated)
    slope === nothing && throw(ArgumentError(
        "Binomial animal q2 SD regression requires `animal(1 + x | group)`."))
    group = first(targets).target.group
    animal_targets = filter(item -> item.target.dpar === :mu &&
        item.target.group === group &&
        item.target.block === nothing, targets)
    Set(item.target.member for item in animal_targets) == Set((:intercept, slope)) ||
        throw(ArgumentError("Binomial animal q2 SD formulas must select `coef(intercept)` and `coef($slope)` exactly once."))
    structured == (:animal, group) && isempty(re) && mv === nothing ||
        throw(ArgumentError("Binomial animal q2 SD regression requires one matching animal block and no additional random effects."))
    tree === nothing && K === nothing && coords === nothing || throw(ArgumentError(
        "Binomial animal q2 SD regression uses `A`, `Ainv`, or pedigree input only."))
    (A === nothing) != (Ainv === nothing) || throw(ArgumentError(
        "Binomial animal q2 SD regression requires exactly one of `A` or `Ainv`."))
    allowed = Set([:mu; [item.key for item in targets]])
    for (pname, form) in f.forms
        pname in allowed && continue
        pname === :sigma && form == ConstantTerm(1) && continue
        throw(ArgumentError("Binomial animal q2 SD regression supports only `mu`, fixed `sigma ~ 1`, and the two animal SD formulas."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Binomial animal q2 SD regression requires observed responses; remove incomplete rows before fitting."))
    s, ntr = _binomial_response(f, data)
    _, Xmu, nmμ = _design(f.response, fixed_mu, data)
    labels = getproperty(data, group)
    gidx, G = _group_index(labels)
    Q = if Ainv === nothing
        C = Matrix{Float64}(A)
        size(C) == (G, G) || throw(ArgumentError(
            "animal covariance matrix must be $(G)×$(G) in first-seen level order."))
        ch = _checked_relmat_chol(C, group)
        sparse(inv(ch))
    else
        size(Ainv) == (G, G) || throw(ArgumentError(
            "animal Ainv must be $(G)×$(G) in first-seen level order."))
        sparse(Float64.(Ainv))
    end
    ordered = map((:intercept, slope)) do member
        only(filter(item -> item.target.member === member, animal_targets))
    end
    Zgs = Matrix{Float64}[]
    nmsds = Vector{String}[]
    for item in ordered
        Zg, names = _sd_group_design(f.response, item.rhs, data, gidx, G, group;
            schema_cache = f.schema_cache, schema_key = item.key)
        size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
            "Binomial animal SD target `coef($(item.target.member))` has a rank-deficient group-level design."))
        push!(Zgs, Matrix{Float64}(Zg)); push!(nmsds, names)
    end
    hasproperty(data, slope) || throw(ArgumentError(
        "Binomial animal random-slope predictor `$slope` is absent from data."))
    x = getproperty(data, slope)
    x isa AbstractVector{<:Real} && length(x) == length(s) && all(isfinite, x) ||
        throw(ArgumentError("Binomial animal random-slope predictor `$slope` must be numeric, finite, and have one value per response."))
    any(!iszero, x) || throw(ArgumentError(
        "Binomial animal random-slope predictor `$slope` is all zero, so its SD is not identifiable."))
    if all(==(first(x)), x) && isequal(Zgs[1], Zgs[2])
        throw(ArgumentError("Binomial animal q2 SD regression cannot separate intercept and slope variance when `$slope` is constant and both SD designs are identical."))
    end
    blocks = [_sd_target_block(item.target; disambiguate = true)
              for item in ordered]
    fit = _fit_binomial_animal_q2_laplace(fam, s, ntr, Xmu, Zgs[1],
        Zgs[2], gidx, Q, Float64.(x), nmμ, nmsds[1], nmsds[2],
        blocks, group, g_tol; se, correlated,
        # Same finite-difference step as the phylo/spatial q2 route (see the comment there):
        # the inner-solve noise is amplified by the default step in the weak correlation SE.
        hessian_step = 3e-3)
    return _withformula(fit, f)
end

# Binomial phylo / coordinate-spatial intercept-plus-slope q2:
#   y ~ x + phylo(1 + x | g)   with sd(g, phylogenetic, coef(intercept)) ~ z0,
#                                   sd(g, phylogenetic, coef(x)) ~ z1
# (and the `spatial(1 + x | g)` analogue). Unlabelled: two independent fields, rho fixed 0.
# Labelled `phylo(1 + x | p | g)`: one free cross-coefficient correlation (stored as the
# `:recov` block, `atanh_rho`), as the Poisson/NB2 labelled route. Same kernels as the
# Poisson/NB2 route (`_structured_q2_kernel`) and the animal q2 Laplace code above.
# Scope matches native drmTMB 0.8.1 (R/binomial-q2-sd.R): logit link and an unweighted
# 0/1 response; probit, cloglog, cbind() trials, the labelled SPATIAL form and `mesh` are
# refused (native admits labelled Binomial for animal and for logit phylo only).
function _drm_binomial_phylo_spatial_q2_sd(f::DrmFormula, fam::Binomial,
        kind::Symbol, block; data, tree, K, A, Ainv, coords, mesh, g_tol, se)
    level = _q2_target_level(kind)
    group, slope = block.group, block.slope
    correlated = block.label !== nothing
    term = correlated ? "$kind(1 + $slope | $(block.label) | $group)" : "$kind(1 + $slope | $group)"
    who = "Binomial `$term` SD regression"
    correlated && kind === :spatial && throw(ArgumentError(
        "$who: the labelled correlated spatial form is not supported " *
        "(native drmTMB 0.8.1 admits labelled Binomial intercept-and-slope blocks only for " *
        "`animal` and for logit `phylo`); use the unlabelled independent-field " *
        "`spatial(1 + $slope | $group)` form."))
    fam.link === :logit || throw(ArgumentError(
        "$who supports the logit link only (native drmTMB 0.8.1 refuses `probit` and " *
        "`cloglog` for structured intercept-and-slope Binomial models)."))
    f.response2 === nothing || throw(ArgumentError(
        "$who requires an unweighted 0/1 Bernoulli response, as native drmTMB 0.8.1 does; " *
        "`cbind(successes, failures)` trial counts are not supported on this route."))
    mesh === nothing || throw(ArgumentError(
        "$who uses `tree = …` / `coords = …`; the mesh/SPDE route is not covered."))
    targets = _sd_target_parts(f)
    length(targets) == 2 || throw(ArgumentError(
        "$who needs one SD formula for the intercept and one for the slope " *
        "(`sd($group, $level, coef(intercept)) ~ …` and `sd($group, $level, coef($slope)) ~ …`)."))
    all(item -> item.target.level === level && item.target.group === group &&
        item.target.dpar === :mu && item.target.block === nothing, targets) ||
        throw(ArgumentError("$who requires two matching `$level` mean targets."))
    Set(item.target.member for item in targets) == Set((:intercept, slope)) ||
        throw(ArgumentError("$who: SD formulas must select `coef(intercept)` and `coef($slope)` exactly once."))
    isempty(_sdphylo_parts(f)) || throw(ArgumentError(
        "$who cannot be combined with an intercept-only `sd($group, phylogenetic) ~ …` formula."))
    Ainv === nothing || throw(ArgumentError("$who does not use `Ainv`."))
    rhs = Dict(f.forms)
    fixed_mu, re, metav, structured, phylo_slope, _, other_slope = _split_ranef(rhs[:mu];
        allow_phylo_slope = kind === :phylo,
        allow_spatial_intercept_slope = kind === :spatial,
        allow_labelled_intercept_slope = correlated)
    parsed_slope = kind === :phylo ? phylo_slope : other_slope
    metav === nothing && isempty(re) && structured == (kind, group) &&
        parsed_slope === slope || throw(ArgumentError(
        "$who requires one matching `$term` term and no ordinary random effects or `meta_V`."))
    _collect_structured(rhs[:mu]) == [(kind, group)] || throw(ArgumentError(
        "$who requires exactly one structured term, with no additional structured effects."))
    allowed = Set(vcat([:mu], [item.key for item in targets]))
    for (pname, form) in f.forms
        pname in allowed && continue
        pname === :sigma && form == ConstantTerm(1) && continue
        throw(ArgumentError("$who supports only `mu` and its two SD formulas."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "$who requires observed responses; remove incomplete rows before fitting."))
    fixed_mu, offset_term = _extract_offset(fixed_mu)
    offset_term === nothing || throw(ArgumentError(
        "$who does not yet support `offset(...)`."))
    s, ntr = _binomial_response(f, data)
    _, Xmu, nmmu = _design(f.response, fixed_mu, data)
    kind === :spatial && (coords = _resolve_coords_labels(f, data, coords, nothing))   # #957
    C, gidx, G = _structured_q2_kernel(kind, group, getproperty(data, group);
                                       tree, K, A, coords, who)
    Cchol = cholesky(Symmetric(C); check = false)
    issuccess(Cchol) || throw(ArgumentError("$who: the $kind kernel is not positive definite."))
    Q = sparse(Symmetric(inv(Cchol)))
    ordered = map((:intercept, slope)) do member
        only(filter(item -> item.target.member === member, targets))
    end
    Zgs = Matrix{Float64}[]
    nmsds = Vector{String}[]
    for item in ordered
        Zg, names = _sd_group_design(f.response, item.rhs, data, gidx, G, group;
            schema_cache = f.schema_cache, schema_key = item.key)
        size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
            "$who: SD target `coef($(item.target.member))` has a rank-deficient group-level design."))
        push!(Zgs, Matrix{Float64}(Zg))
        push!(nmsds, names)
    end
    hasproperty(data, slope) || throw(ArgumentError(
        "$who: random-slope predictor `$slope` is absent from data."))
    x = getproperty(data, slope)
    x isa AbstractVector{<:Real} && length(x) == length(s) && all(isfinite, x) ||
        throw(ArgumentError("$who: random-slope predictor `$slope` must be numeric, finite, and have one value per response."))
    any(!iszero, x) || throw(ArgumentError(
        "$who: random-slope predictor `$slope` is all zero, so its SD is not identifiable."))
    if all(==(first(x)), x) && isequal(Zgs[1], Zgs[2])
        throw(ArgumentError("$who cannot separate the intercept and slope variances when " *
            "`$slope` is constant and both SD formulas have identical group-level designs."))
    end
    blocks = Symbol[_sd_target_block(item.target; disambiguate = true) for item in ordered]
    fit = _fit_binomial_animal_q2_laplace(fam, s, ntr, Xmu, Zgs[1], Zgs[2], gidx, Q,
        Float64.(x), nmmu, nmsds[1], nmsds[2], blocks, group, g_tol; se,
        correlated, second_start = kind === :spatial,
        # The inner Newton solve stops at a relative step of 1e-7 (it cannot go lower: the
        # line search then stalls at the floating-point floor), so the Laplace objective
        # carries noise that the default finite-difference step (~5e-4) amplifies in the
        # weakly identified slope-SD coefficients (SEs up to 19% below native drmTMB in
        # the wave-6 study). Steps of 3e-3 to 3e-2 reproduce R's SEs to 3 digits.
        hessian_step = 3e-3)
    return _withformula(fit, f)
end

function _drm_binomial_related_sd_regression(f::DrmFormula, fam::Binomial;
                                             data, covariance, level::Symbol,
                                             Ainv = nothing,
                                             tree, K, A, coords,
                                             g_tol::Real, se::Bool)
    targets = _sd_target_parts(f)
    length(targets) == 1 || throw(ArgumentError(
        "Binomial $level SD regression currently requires exactly one `sd(group, $level) ~ …` formula."))
    target = only(targets)
    descriptor = target.target
    (descriptor.level === level && descriptor.dpar === :mu &&
     descriptor.block === nothing) ||
        throw(ArgumentError("Binomial $level SD regression targets the sole structured mean coefficient."))
    if level === :animal
        tree === nothing && K === nothing && coords === nothing ||
            throw(ArgumentError("Binomial animal SD regression uses `A = …` or `Ainv = …` only; do not also supply `tree`, `K`, or `coords`."))
        (covariance === nothing) == (Ainv === nothing) && throw(ArgumentError(
            "animal(1 | $(descriptor.group)) with predictor-dependent SD requires exactly one of `A` or `Ainv`."))
    elseif level === :spatial
        tree === nothing && K === nothing && A === nothing && Ainv === nothing ||
            throw(ArgumentError("Binomial spatial SD regression uses `coords = …` only; do not also supply `tree`, `K`, `A`, or `Ainv`."))
        covariance === nothing || throw(ArgumentError(
            "Binomial spatial SD regression builds its covariance from `coords`; do not supply a separate covariance matrix."))
    else
        Ainv === nothing || throw(ArgumentError(
            "Binomial `Ainv` is supported for animal SD regression only."))
        tree === nothing && A === nothing && coords === nothing ||
            throw(ArgumentError("Binomial relmat SD regression uses `K = …` only; do not also supply `tree`, `A`, or `coords`."))
        covariance === nothing && throw(ArgumentError(
            "relmat(1 | $(descriptor.group)) needs the covariance matrix `K = …`"))
    end

    rhs = Dict(f.forms)
    fixed_mu, re, mv, st, _, _, animal_slope =
        _split_ranef(rhs[:mu]; allow_animal_slope = level === :animal,
                     allow_relmat_slope = level === :relmat,
                     allow_spatial_slope = level === :spatial)
    descriptor.member in (nothing, animal_slope) || throw(ArgumentError(
        "Binomial $level SD target must select the matching structured coefficient."))
    mv === nothing || throw(ArgumentError("Binomial $level SD regression cannot use meta_V()."))
    st !== nothing && first(st) === level || throw(ArgumentError(
        "Binomial `sd($(descriptor.group), $level)` requires one matching structured term in the location formula."))
    structured = _collect_structured(rhs[:mu])
    length(structured) == 1 && only(structured) == (level, descriptor.group) ||
        throw(ArgumentError("Binomial $level SD regression currently requires exactly one structured term and no other structured effects."))
    isempty(re) || throw(ArgumentError(
        "Binomial $level SD regression currently requires one structured coefficient without additional random effects."))
    last(st) === descriptor.group || throw(ArgumentError(
        "The $level SD formula must name the group in the structured term `$level(… | $(last(st)))`."))
    for (pname, form) in f.forms
        (pname === :mu || pname === target.key) && continue
        pname === :sigma && form === ConstantTerm(1) && continue
        throw(ArgumentError("Binomial $level SD regression supports only `mu`, fixed `sigma ~ 1`, and one structured SD formula."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Binomial $level SD regression requires observed responses; remove incomplete rows before fitting."))

    s, ntr = _binomial_response(f, data)
    _, Xmu, nmmu = _design(f.response, fixed_mu, data)
    labels = getproperty(data, descriptor.group)
    gidx, G = _group_index(labels)
    C = if level === :spatial
        coords isa AbstractDict && (coords = _resolve_coords_labels(f, data, coords, nothing))   # red team: label => coordinate Dict, as on the other families
        _spatial_covariance_from_coords(descriptor.group, G, coords, nothing;
                                        legacy_mean_range = true)   # no drmTMB binomial spatial: see docs/dev-log/spatial-kernel-audit
    else
        covariance === nothing ? nothing : Matrix{Float64}(covariance)
    end
    C === nothing || size(C) == (G, G) || throw(ArgumentError(
        "$level covariance matrix must be $(G)×$(G) (the number of `$(descriptor.group)` levels, in first-seen order)"))
    if Ainv !== nothing
        size(Ainv) == (G, G) || throw(ArgumentError(
            "animal `Ainv` must be a $(G)×$(G) precision matrix ordered by first-seen levels of `$(descriptor.group)`."))
    end
    Zg, nmsd = _sd_group_design(f.response, target.rhs, data, gidx, G,
                                descriptor.group; schema_cache = f.schema_cache,
                                schema_key = target.key)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "Binomial $level SD target has a rank-deficient group-level design."))
    Zobs = Zg[gidx, :]
    re_design = if animal_slope === nothing
        ones(Float64, length(gidx))
    else
        hasproperty(data, animal_slope) || throw(ArgumentError(
            "Binomial animal slope term is missing predictor `$animal_slope` from data."))
        raw = getproperty(data, animal_slope)
        raw isa AbstractVector{<:Real} && length(raw) == length(gidx) && all(isfinite, raw) ||
            throw(ArgumentError("Binomial structured slope predictor `$animal_slope` must be numeric, finite, and have one value per response."))
        any(!iszero, raw) || throw(ArgumentError(
            "Binomial structured slope predictor `$animal_slope` is zero for every observation, so its SD is not identifiable."))
        if level === :spatial
            has_within_group_variation = any(g ->
                length(unique(raw[gidx .== g])) > 1, 1:G)
            has_within_group_variation || throw(ArgumentError(
                "Binomial spatial slope predictor `$animal_slope` must vary within at least one site."))
        end
        Float64.(raw)
    end
    fit = _fit_binomial_related_sd_laplace(
        fam, s, ntr, Xmu, Zobs, Zg, labels, C, gidx, nmmu, nmsd,
        descriptor.group, level, g_tol; se, re_design, Ainv,
    )
    return _withformula(fit, f)
end

function _drm_binomial_animal_sd_regression(f::DrmFormula, fam::Binomial;
        data, A, Ainv = nothing, tree, K, coords, g_tol::Real, se::Bool)
    targets = filter(item -> item.target.level === :animal,
                     _sd_target_parts(f))
    if length(targets) == 2
        return _drm_binomial_animal_q2_sd_regression(f, fam; data, A, Ainv,
            tree, K, coords, g_tol, se)
    end
    return _drm_binomial_related_sd_regression(f, fam; data, covariance = A,
        level = :animal, Ainv, tree, K, A, coords, g_tol, se)
end

function _binomial_animal_q2_sd_marginal_simulator(fit::DrmFit{Binomial}, data,
                                                    A, Ainv)
    f = fit.formula
    f isa DrmFormula || throw(ArgumentError(
        "bootstrap: Binomial animal q2 simulation requires the fitted formula."))
    targets = filter(item -> item.target.level === :animal,
                     _sd_target_parts(f))
    length(targets) == 2 || throw(ArgumentError(
        "bootstrap: Binomial animal q2 simulation requires separate intercept and slope SD targets"))
    all(item -> item.target.dpar === :mu &&
                item.target.group === first(targets).target.group &&
                item.target.block === nothing,
        targets) || throw(ArgumentError(
        "bootstrap: Binomial animal q2 simulation requires two mean SD targets for one animal group"))
    by_member = Dict(item.target.member => item for item in targets)
    haskey(by_member, :intercept) || throw(ArgumentError(
        "bootstrap: Binomial animal q2 simulation is missing its intercept SD target"))
    rhs = Dict(f.forms)[:mu]
    fixed_mu, re, mv, structured, _, _, slope = _split_ranef(
        rhs; allow_animal_intercept_slope = true, allow_labelled_intercept_slope = true)
    group = first(targets).target.group
    slope !== nothing && isempty(re) && mv === nothing &&
        structured == (:animal, group) || throw(ArgumentError(
        "bootstrap: Binomial animal q2 simulation requires one matching animal intercept-and-slope effect"))
    haskey(by_member, slope) || throw(ArgumentError(
        "bootstrap: Binomial animal q2 simulation is missing the `$slope` SD target"))
    hasproperty(data, group) || throw(ArgumentError(
        "bootstrap: data are missing animal grouping column `$group`"))
    gidx, G = _group_index(getproperty(data, group))
    length(gidx) == fit.nobs || throw(ArgumentError(
        "bootstrap: animal group indexes do not match fitted response rows"))
    (A === nothing) == (Ainv === nothing) && throw(ArgumentError(
        "bootstrap: supply exactly one of `A` and `Ainv` for Binomial animal q2 simulation"))
    matrix = Ainv === nothing ? A : Ainv
    C = Matrix{Float64}(matrix)
    size(C) == (G, G) || throw(ArgumentError(
        "bootstrap: animal matrix must match first-seen group order for `$group`"))
    factor = if Ainv === nothing
        _checked_relmat_chol(C, group)
    else
        all(isfinite, C) || throw(ArgumentError(
            "bootstrap: animal `Ainv` must contain only finite values"))
        isapprox(C, C'; rtol = sqrt(eps(Float64))) || throw(ArgumentError(
            "bootstrap: animal `Ainv` must be symmetric"))
        qchol = cholesky(Symmetric(sparse(C)); check = false)
        issuccess(qchol) || throw(ArgumentError(
            "bootstrap: animal `Ainv` must be positive definite"))
        qchol
    end
    group_sd = Dict{Symbol,Vector{Float64}}()
    for member in (:intercept, slope)
        target = by_member[member]
        block = _sd_target_block(target.target; disambiguate = true)
        Zg, names = _sd_group_design(f.response, target.rhs, data, gidx, G,
            group; schema_cache = f.schema_cache, schema_key = target.key,
            store_schema = false)
        names == _block_coefnames(fit, block) || throw(ArgumentError(
            "bootstrap: animal q2 SD design columns do not match fitted block `$block`"))
        group_sd[member] = exp.(Zg * coef(fit, block))
    end
    raw_loading = getproperty(data, slope)
    raw_loading isa AbstractVector{<:Real} &&
        length(raw_loading) == fit.nobs && all(isfinite, raw_loading) ||
        throw(ArgumentError(
            "bootstrap: animal slope predictor `$slope` must be finite and have one value per response"))
    loading = Float64.(raw_loading)
    rho = haskey(fit.scales, :correlation) ? only(fit.scales[:correlation]) : 0.0   # unlabelled: independent fields
    eta_fixed = Vector{Float64}(predict(fit, data; type = :link))
    length(eta_fixed) == fit.nobs || throw(ArgumentError(
        "bootstrap: Binomial fixed predictions do not match fitted response rows"))
    return function (rng)
        z0, z1 = randn(rng, G), randn(rng, G)
        field0, field1 = _binomial_animal_q2_correlated_fields(
            factor, Ainv !== nothing, z0, z1, rho)
        eta = _binomial_animal_q2_eta(eta_fixed, group_sd[:intercept],
            group_sd[slope], gidx, loading, field0, field1)
        return _simulate_once(fit, rng;
            mu = _binomial_mean.(Ref(fit.family), eta))
    end
end

function _binomial_animal_q2_eta(eta_fixed, sd_intercept, sd_slope, gidx,
                                 loading, field_intercept, field_slope)
    length(eta_fixed) == length(gidx) == length(loading) ||
        throw(DimensionMismatch(
            "Binomial animal q2 fixed predictor, group index, and slope loading must have equal lengths"))
    length(sd_intercept) == length(sd_slope) ==
        length(field_intercept) == length(field_slope) ||
        throw(DimensionMismatch(
            "Binomial animal q2 SDs and latent fields must have equal group counts"))
    return eta_fixed .+ (sd_intercept .* field_intercept)[gidx] .+
           loading .* (sd_slope .* field_slope)[gidx]
end

function _binomial_animal_q2_correlated_fields(factor, is_precision::Bool,
                                               z0, z1, rho::Real)
    correlated = rho .* z0 .+ sqrt(1 - rho^2) .* z1
    if is_precision
        return factor.UP \ z0, factor.UP \ correlated
    end
    return factor.L * z0, factor.L * correlated
end

function _marginal_simulator_build(fit::DrmFit{Binomial}, data; K = nothing,
        A = nothing, Ainv = nothing, tree = nothing, tree1 = nothing,
        tree2 = nothing, coords = nothing)
    targets = fit.formula isa DrmFormula ? _sd_target_parts(fit.formula) : Any[]
    animal_targets = filter(item -> item.target.level === :animal, targets)
    if length(targets) == 2 && length(animal_targets) == 2
        return _binomial_animal_q2_sd_marginal_simulator(fit, data, A, Ainv)
    end
    # phylo / coordinate-spatial q2: rebuild the fit's kernel and row map (as the
    # Poisson/NB2 bootstrap does) and draw the two independent fields.
    q2ps = fit.formula isa DrmFormula ? _phylo_spatial_q2_block(fit.formula) : nothing
    if q2ps !== nothing
        kind, block = q2ps
        tree1 === nothing && tree2 === nothing && K === nothing && A === nothing &&
            Ainv === nothing || throw(ArgumentError(
            "bootstrap/simulate: the Binomial `$(kind)(1 + $(block.slope) | $(block.group))` " *
            "route uses `tree = …` (phylo) or `coords = …` (spatial) only; " *
            "`tree1`, `tree2`, `K`, `A` and `Ainv` are not used and are refused rather than ignored."))
        hasproperty(data, block.group) || throw(ArgumentError(
            "bootstrap: data are missing the $(kind) grouping column `$(block.group)`"))
        length(getproperty(data, block.group)) == fit.nobs || throw(ArgumentError(
            "bootstrap/simulate: the Binomial `$(kind)(1 + $(block.slope) | $(block.group))` " *
            "marginal simulator redraws the fitted rows (its 0/1 trials vector is tied to the " *
            "$(fit.nobs) training rows); `data` has $(length(getproperty(data, block.group))) " *
            "rows. Pass the fitting data, or use `newdata = …` with `re = :zero`."))
        kcoords = kind === :spatial ?
            _resolve_coords_labels(fit.formula, data, coords, nothing) : nothing
        kernel = _structured_q2_kernel(kind, block.group, getproperty(data, block.group);
            tree, K = nothing, A = nothing, coords = kcoords,
            who = "bootstrap: `$(kind)(1 + $(block.slope) | $(block.group))` simulation")
        return _structured_q2_sd_marginal_simulator(
            fit, data, nothing; provider = kind, block, kernel)
    end
    return invoke(_marginal_simulator_build, Tuple{DrmFit,Any}, fit, data;
        K, A, Ainv, tree, tree1, tree2, coords)
end

# Population-level predictions need the SD target metadata and fixed mu design,
# but generic formula parsing cannot yet consume Binomial `animal(1 + x | g)`.
# Use the fitted fixed part for that prediction pass while preserving the SD
# target formulas, schemas, coefficients, and original fitted object.
function _binomial_animal_q2_prediction_fit(fit::DrmFit{Binomial})
    f = fit.formula
    f isa DrmFormula || return fit
    targets = filter(item -> item.target.level === :animal,
                     _sd_target_parts(f))
    length(targets) == 2 || return fit
    rhs = Dict(f.forms)[:mu]
    fixed_mu, re, mv, structured, _, _, slope = _split_ranef(
        rhs; allow_animal_intercept_slope = true, allow_labelled_intercept_slope = true)
    slope === nothing && return fit
    group = first(targets).target.group
    structured == (:animal, group) && isempty(re) && mv === nothing ||
        return fit
    forms = copy(f.forms)
    mu_position = findfirst(pair -> first(pair) === :mu, forms)
    mu_position === nothing && return fit
    forms[mu_position] = :mu => fixed_mu
    prediction_formula = DrmFormula(f.response, forms, f.response2,
        f.schema_cache, f.sd_targets)
    return _withformula(fit, prediction_formula)
end

function predict(fit::DrmFit{Binomial}, newdata;
                 type::Symbol = :response, se::Bool = false)
    prediction_fit = _binomial_animal_q2_prediction_fit(fit)
    return invoke(predict, Tuple{DrmFit,Any}, prediction_fit, newdata;
                  type, se)
end

function predict_parameters(fit::DrmFit{Binomial}, newdata;
                            type::Symbol = :response, se::Bool = false)
    prediction_fit = _binomial_animal_q2_prediction_fit(fit)
    return invoke(predict_parameters, Tuple{DrmFit,Any}, prediction_fit,
                  newdata; type, se)
end

_drm_binomial_relmat_sd_regression(f::DrmFormula, fam::Binomial; data, K,
                                   tree, A, coords, g_tol::Real, se::Bool) =
    _drm_binomial_related_sd_regression(f, fam; data, covariance = K,
        level = :relmat, tree, K, A, coords, g_tol, se)

_drm_binomial_spatial_sd_regression(f::DrmFormula, fam::Binomial; data, coords,
                                    tree, K, A, g_tol::Real, se::Bool) =
    _drm_binomial_related_sd_regression(f, fam; data, covariance = nothing,
        level = :spatial, tree, K, A, coords, g_tol, se)

function _drm_binomial_mesh_sd_regression(f::DrmFormula, fam::Binomial;
        data, mesh, tree, tree1, tree2, K, A, Ainv,
        g_tol::Real, se::Bool)
    targets = _sd_target_parts(f)
    length(targets) == 1 || throw(ArgumentError(
        "Binomial mesh spatial SD regression currently requires exactly one spatial SD formula."))
    target = only(targets)
    descriptor = target.target
    (descriptor.level === :spatial && descriptor.dpar === :mu &&
     descriptor.block === nothing && descriptor.member === nothing) ||
        throw(ArgumentError("Binomial mesh spatial SD regression targets one spatial mean intercept."))
    tree === nothing && tree1 === nothing && tree2 === nothing && K === nothing &&
        A === nothing && Ainv === nothing || throw(ArgumentError(
        "Binomial mesh spatial SD regression accepts only `mesh = …`; do not also supply tree, covariance, precision, or coordinate inputs."))
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st, _, _, _ = _split_ranef(rhs[:mu])
    mv === nothing || throw(ArgumentError("Binomial mesh spatial SD regression cannot use meta_V()."))
    st !== nothing && first(st) === :spatial && last(st) === descriptor.group &&
        _collect_structured(rhs[:mu]) == [(:spatial, descriptor.group)] || throw(ArgumentError(
        "Binomial mesh SD regression requires exactly one matching `spatial(1 | $(descriptor.group))` term."))
    isempty(re) || throw(ArgumentError(
        "Binomial mesh spatial SD regression does not yet support additional IID random effects."))
    for (pname, form) in f.forms
        (pname === :mu || pname === target.key) && continue
        pname === :sigma && form === ConstantTerm(1) && continue
        throw(ArgumentError("Binomial mesh spatial SD regression supports only `mu`, fixed `sigma ~ 1`, and one spatial SD formula."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Binomial mesh spatial SD regression requires observed responses; remove incomplete rows before fitting."))

    projection, precision, alignment_ids = _binomial_mesh_inputs(mesh, data, f.response)
    s, ntr = _binomial_response(f, data)
    _, Xmu, nmmu = _design(f.response, fixed_mu, data)
    labels = getproperty(data, descriptor.group)
    gidx, G = _group_index(labels)
    size(projection, 1) == length(s) || throw(ArgumentError(
        "Binomial mesh projection must have one row per retained response."))
    Zg, nmsd = _sd_group_design(f.response, target.rhs, data, gidx, G,
        descriptor.group; schema_cache = f.schema_cache, schema_key = target.key)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "Binomial spatial SD target has a rank-deficient group-level design."))
    Zobs = Zg[gidx, :]
    fit = _fit_binomial_mesh_sd_laplace(fam, s, ntr, Xmu, Zobs, Zg,
        projection, precision, gidx, nmmu, nmsd, descriptor.group,
        alignment_ids, g_tol; se)
    return _withformula(fit, f)
end

function _drm_binomial_animal_iid_sd_regression(f::DrmFormula, fam::Binomial;
                                                data, A, Ainv = nothing,
                                                tree, K, coords,
                                                g_tol::Real, se::Bool)
    targets = _sd_target_parts(f)
    animal_targets = filter(item -> item.target.level === :animal, targets)
    iid_targets = filter(item -> item.target.level === :iid, targets)
    length(animal_targets) == length(iid_targets) == 1 || throw(ArgumentError(
        "Binomial animal-plus-IID SD regression requires one animal and one ordinary SD formula."))
    animal_target, iid_target = only(animal_targets), only(iid_targets)
    for target in (animal_target.target, iid_target.target)
        (target.dpar === :mu && target.member === nothing && target.block === nothing) ||
            throw(ArgumentError("Binomial animal-plus-IID SD regression targets intercepts in the mean model."))
    end
    tree === nothing && K === nothing && coords === nothing || throw(ArgumentError(
        "Binomial animal-plus-IID SD regression uses `A = …` or `Ainv = …` only."))
    (A === nothing) != (Ainv === nothing) || throw(ArgumentError(
        "animal SD regression requires exactly one of `A` or `Ainv`."))

    rhs = Dict(f.forms)
    fixed_mu, re, mv, _, _, _, animal_slope =
        _split_ranef(rhs[:mu]; allow_animal_slope = true)
    mv === nothing || throw(ArgumentError("Binomial animal-plus-IID SD regression cannot use meta_V()."))
    structured = _collect_structured(rhs[:mu])
    structured == [(:animal, animal_target.target.group)] || throw(ArgumentError(
        "Binomial animal-plus-IID SD regression requires exactly one matching animal term."))
    animal_terms = filter(term -> _collect_structured(term) == [
        (:animal, animal_target.target.group)], rhs[:mu] isa Tuple ? collect(rhs[:mu]) : Any[rhs[:mu]])
    length(animal_terms) == 1 || throw(ArgumentError(
        "Binomial animal-plus-IID SD regression requires one animal random intercept or slope."))
    animal_term = only(animal_terms)
    parsed_animal_slope = _check_animal_re_lhs(
        animal_term.args[1].args[1], animal_target.target.group; allow_slope = true)
    parsed_animal_slope === animal_slope || throw(ArgumentError(
        "Binomial animal-plus-IID SD regression requires a single animal intercept or slope."))
    fixed_terms = filter(term -> isempty(_collect_structured(term)),
        fixed_mu isa Tuple ? collect(fixed_mu) : Any[fixed_mu])
    fixed_mu = isempty(fixed_terms) ? ConstantTerm(1) :
               length(fixed_terms) == 1 ? only(fixed_terms) : Tuple(fixed_terms)
    length(re) == 1 && first(_re_kind(only(re)[1])) === :intercept || throw(ArgumentError(
        "Binomial animal-plus-IID SD regression currently requires one ordinary random intercept."))
    iid_group = last(only(re))
    iid_group === iid_target.target.group || throw(ArgumentError(
        "The IID SD formula must target the ordinary grouping factor `$iid_group`."))
    iid_group === animal_target.target.group && throw(ArgumentError(
        "The animal and IID effects must use different grouping factors."))
    allowed = Set([:mu, animal_target.key, iid_target.key])
    for (pname, form) in f.forms
        pname in allowed && continue
        pname === :sigma && form === ConstantTerm(1) && continue
        throw(ArgumentError("Binomial animal-plus-IID SD regression supports only two SD formulas and fixed `sigma ~ 1`."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Binomial animal-plus-IID SD regression requires observed responses; remove incomplete rows before fitting."))

    s, ntr = _binomial_response(f, data)
    _, Xmu, nmmu = _design(f.response, fixed_mu, data)
    animal_labels = getproperty(data, animal_target.target.group)
    iid_labels = getproperty(data, iid_group)
    animal_index, G = _group_index(animal_labels)
    iid_index, H = _group_index(iid_labels)
    length(s) == length(animal_index) == length(iid_index) || throw(ArgumentError(
        "Binomial animal and IID group indexes must match the response length."))
    animal_design = if animal_slope === nothing
        ones(Float64, length(s))
    else
        hasproperty(data, animal_slope) || throw(ArgumentError(
            "Binomial animal slope term is missing predictor `$animal_slope` from data."))
        raw = getproperty(data, animal_slope)
        raw isa AbstractVector{<:Real} && length(raw) == length(s) && all(isfinite, raw) ||
            throw(ArgumentError(
                "Binomial animal slope predictor `$animal_slope` must be numeric, finite, and have one value per response."))
        any(!iszero, raw) || throw(ArgumentError(
            "Binomial animal slope predictor `$animal_slope` is zero for every observation, so its SD is not identifiable."))
        Float64.(raw)
    end

    Qanimal, logdetQanimal = if Ainv === nothing
        C = Matrix{Float64}(A)
        size(C) == (G, G) || throw(ArgumentError(
            "animal covariance matrix must be $(G)×$(G) in first-seen animal order."))
        chol = _checked_relmat_chol(C, animal_target.target.group)
        sparse(Symmetric(inv(chol))), -logdet(chol)
    else
        size(Ainv) == (G, G) || throw(ArgumentError(
            "animal `Ainv` must be $(G)×$(G) in first-seen animal order."))
        Q = sparse(Float64.(Ainv))
        all(isfinite, nonzeros(Q)) || throw(ArgumentError("animal `Ainv` must be finite."))
        norm(Q - transpose(Q), Inf) <= 1e-10 * max(norm(Q, Inf), 1.0) ||
            throw(ArgumentError("animal `Ainv` must be symmetric."))
        Q = sparse(Symmetric((Q + transpose(Q)) / 2))
        chol = cholesky(Symmetric(Q); check = false)
        issuccess(chol) || throw(ArgumentError("animal `Ainv` must be positive definite."))
        Q, logdet(chol)
    end
    Wanimal, nm_animal = _sd_group_design(f.response, animal_target.rhs, data,
        animal_index, G, animal_target.target.group;
        schema_cache = f.schema_cache, schema_key = animal_target.key)
    Wiid, nm_iid = _sd_group_design(f.response, iid_target.rhs, data,
        iid_index, H, iid_group; schema_cache = f.schema_cache,
        schema_key = iid_target.key)
    for (W, nlevels, name) in ((Wanimal, G, animal_target.target.group), (Wiid, H, iid_group))
        size(W, 1) == nlevels && size(W, 2) <= nlevels && rank(W) == size(W, 2) ||
            throw(ArgumentError("Binomial SD design for `$name` is rank deficient across groups."))
    end

    nint, sint = round.(Int, ntr), round.(Int, s)
    logchoose = [_logfactorial(nint[i]) - _logfactorial(sint[i]) -
                 _logfactorial(nint[i] - sint[i]) for i in eachindex(sint)]
    aux = fam.link in (:probit, :cloglog) ?
        (s = sint, ntr = nint, logchoose = logchoose, link = fam.link) :
        (s = sint, ntr = nint, logchoose = logchoose)
    pμ, pa, pi = size(Xmu, 2), size(Wanimal, 2), size(Wiid, 2)
    animal_range = (pμ + 1):(pμ + pa)
    iid_range = (pμ + pa + 1):(pμ + pa + pi)
    Q = [Qanimal spzeros(G, H); spzeros(H, G) spdiagm(0 => ones(H))]
    logdetQ = logdetQanimal
    last_b = zeros(G + H)

    function nll(theta)
        tau_animal = _binomial_sd_scale(Wanimal * @view(theta[animal_range]))
        tau_iid = _binomial_sd_scale(Wiid * @view(theta[iid_range]))
        all(isfinite, tau_animal) && all(isfinite, tau_iid) || return 1e18
        rows = vcat(collect(1:length(s)), collect(1:length(s)))
        cols = vcat(animal_index, G .+ iid_index)
        vals = vcat(tau_animal[animal_index] .* animal_design, tau_iid[iid_index])
        loading = sparse(rows, cols, vals, length(s), G + H)
        eta0 = Xmu * @view(theta[1:pμ])
        b, ch, ok = _binomial_phylo_ordinary_mode(aux, eta0, loading, Q, last_b)
        if !ok
            b, ch, ok = _binomial_phylo_ordinary_mode(aux, eta0, loading, Q,
                                                       zeros(length(last_b)))
        end
        ok || return 1e18
        last_b .= b
        eta = eta0 + loading * b
        value = 0.5 * dot(b, Q * b) - 0.5 * logdetQ + 0.5 * logdet(ch)
        @inbounds for i in eachindex(eta)
            value += _laplace_value(Val(:binomial), aux, i, eta[i])
        end
        return value
    end
    grad! = function (gradient, theta)
        for j in eachindex(theta)
            h = 2e-4 * (1 + abs(theta[j]))
            offset = zeros(length(theta)); offset[j] = h
            gradient[j] = (nll(theta .+ offset) - nll(theta .- offset)) / (2h)
        end
        gradient
    end
    pbar = clamp(sum(s) / max(sum(ntr), 1), 1e-4, 1 - 1e-4)
    theta0 = zeros(pμ + pa + pi)
    theta0[1] = _binomial_start(fam, pbar)
    theta0[animal_range] .= Wanimal \ fill(log(0.4), G)
    theta0[iid_range] .= Wiid \ fill(log(0.4), H)
    od = Optim.OnceDifferentiable(nll, grad!, theta0)
    res = Optim.optimize(od, theta0, Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
                         Optim.Options(g_tol = g_tol, iterations = 300))
    theta_hat = Optim.minimizer(res)
    nll_hat = nll(theta_hat)
    gfinal = zeros(length(theta_hat)); grad!(gfinal, theta_hat)
    converged = _laplace_outer_converged(res, nll_hat, gfinal, theta_hat, length(s), g_tol)
    V = se ? _vcov_from_fd_hessian(
        _finite_hessian(nll, theta_hat; h = _fd_hessian_step(length(s)));
        context = "joint sparse-Laplace Binomial animal and IID SD regression") :
        fill(NaN, length(theta_hat), length(theta_hat))

    blocks_sd = _sd_target_blocks(targets)
    animal_block = only(blocks_sd[findall(item -> item.target.level === :animal, targets)])
    iid_block = only(blocks_sd[findall(item -> item.target.level === :iid, targets)])
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => (1:pμ), animal_block => animal_range,
                                         iid_block => iid_range]
    names = Pair{Symbol,Vector{String}}[:mu => nmmu, animal_block => nm_animal,
                                        iid_block => nm_iid]
    means = Dict(:mu => _binomial_mean.(Ref(fam), Xmu * theta_hat[1:pμ]))
    obs = Dict(:mu => s ./ ntr)
    sd_animal = _binomial_sd_scale(Wanimal * theta_hat[animal_range])
    sd_iid = _binomial_sd_scale(Wiid * theta_hat[iid_range])
    scales = Dict{Symbol,Vector{Float64}}(
        :trials => Float64.(nint),
        Symbol("sd_group_index_", animal_target.target.group) => Float64.(animal_index),
        Symbol("sd_group_index_", iid_group) => Float64.(iid_index),
        animal_block => sd_animal[animal_index],
        iid_block => sd_iid[iid_index],
    )
    fit = DrmFit(fam, blocks, names, theta_hat, Matrix(V), -nll_hat, length(s),
                 converged, means, obs, scales)
    _warn_at_clamp("Binomial SD regression", [
        "$(animal_target.target.group) random-effect log SD" =>
            (collect(Float64, Wanimal * theta_hat[animal_range]), -12.0, 12.0),
        "$iid_group random-effect log SD" =>
            (collect(Float64, Wiid * theta_hat[iid_range]), -12.0, 12.0),
    ])
    return _withformula(_withiterations(_withnll(fit, _nll_pinned(nll, theta_hat, nll_hat), grad!),
                                        Optim.iterations(res)), f)
end

function _drm_binomial_relmat_iid_sd_regression(f::DrmFormula, fam::Binomial;
        data, K, A, Ainv = nothing, tree, coords, g_tol::Real, se::Bool)
    targets = _sd_target_parts(f)
    rel_targets = filter(item -> item.target.level === :relmat, targets)
    iid_targets = filter(item -> item.target.level === :iid, targets)
    length(rel_targets) == length(iid_targets) == 1 || throw(ArgumentError(
        "Binomial related-matrix-plus-IID SD regression requires one relmat and one ordinary SD formula."))
    rel_target, iid_target = only(rel_targets), only(iid_targets)
    for target in (rel_target.target, iid_target.target)
        (target.dpar === :mu && target.member === nothing && target.block === nothing) ||
            throw(ArgumentError("Binomial related-matrix-plus-IID SD regression targets intercepts in the mean model."))
    end
    A === nothing && Ainv === nothing && tree === nothing && coords === nothing ||
        throw(ArgumentError("Binomial related-matrix-plus-IID SD regression uses `K = …` only."))
    K === nothing && throw(ArgumentError(
        "relmat(1 | $(rel_target.target.group)) with predictor-dependent SD requires `K = …`."))

    rhs = Dict(f.forms)
    fixed_mu, re, mv, structured, _, _, animal_slope =
        _split_ranef(rhs[:mu]; allow_relmat_slope = false)
    mv === nothing || throw(ArgumentError("Binomial related-matrix-plus-IID SD regression cannot use meta_V()."))
    structured == (:relmat, rel_target.target.group) || throw(ArgumentError(
        "Binomial related-matrix-plus-IID SD regression requires one matching relmat intercept."))
    animal_slope === nothing || throw(ArgumentError(
        "Binomial related-matrix-plus-IID SD regression supports intercepts only."))
    length(re) == 1 && first(_re_kind(only(re)[1])) === :intercept || throw(ArgumentError(
        "Binomial related-matrix-plus-IID SD regression currently requires one ordinary random intercept."))
    iid_group = last(only(re))
    iid_group === iid_target.target.group || throw(ArgumentError(
        "The IID SD formula must target the ordinary grouping factor `$iid_group`."))
    iid_group === rel_target.target.group && throw(ArgumentError(
        "The related-matrix and IID effects must use different grouping factors."))
    allowed = Set([:mu, rel_target.key, iid_target.key])
    for (pname, form) in f.forms
        pname in allowed && continue
        pname === :sigma && form === ConstantTerm(1) && continue
        throw(ArgumentError("Binomial related-matrix-plus-IID SD regression supports only two SD formulas and fixed `sigma ~ 1`."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Binomial related-matrix-plus-IID SD regression requires observed responses; remove incomplete rows before fitting."))

    s, ntr = _binomial_response(f, data)
    _, Xmu, nmmu = _design(f.response, fixed_mu, data)
    rel_index, G = _group_index(getproperty(data, rel_target.target.group))
    iid_index, H = _group_index(getproperty(data, iid_group))
    length(s) == length(rel_index) == length(iid_index) || throw(ArgumentError(
        "Binomial related-matrix and IID group indexes must match the response length."))
    C = Matrix{Float64}(K)
    size(C) == (G, G) || throw(ArgumentError(
        "related-matrix covariance must be $(G)×$(G) in first-seen group order."))
    chol = _checked_relmat_chol(C, rel_target.target.group)
    Qrel = sparse(Symmetric(inv(chol)))
    logdetQrel = -logdet(chol)
    Wrel, nm_rel = _sd_group_design(f.response, rel_target.rhs, data,
        rel_index, G, rel_target.target.group; schema_cache = f.schema_cache,
        schema_key = rel_target.key)
    Wiid, nm_iid = _sd_group_design(f.response, iid_target.rhs, data,
        iid_index, H, iid_group; schema_cache = f.schema_cache,
        schema_key = iid_target.key)
    for (W, nlevels, name) in ((Wrel, G, rel_target.target.group), (Wiid, H, iid_group))
        size(W, 1) == nlevels && size(W, 2) <= nlevels && rank(W) == size(W, 2) ||
            throw(ArgumentError("Binomial SD design for `$name` is rank deficient across groups."))
    end

    nint, sint = round.(Int, ntr), round.(Int, s)
    logchoose = [_logfactorial(nint[i]) - _logfactorial(sint[i]) -
                 _logfactorial(nint[i] - sint[i]) for i in eachindex(sint)]
    aux = fam.link in (:probit, :cloglog) ?
        (s = sint, ntr = nint, logchoose, link = fam.link) :
        (s = sint, ntr = nint, logchoose)
    pμ, pr, pi = size(Xmu, 2), size(Wrel, 2), size(Wiid, 2)
    rel_range = (pμ + 1):(pμ + pr)
    iid_range = (pμ + pr + 1):(pμ + pr + pi)
    Q = [Qrel spzeros(G, H); spzeros(H, G) spdiagm(0 => ones(H))]
    last_b = zeros(G + H)
    function nll(theta)
        tau_rel = _binomial_sd_scale(Wrel * @view(theta[rel_range]))
        tau_iid = _binomial_sd_scale(Wiid * @view(theta[iid_range]))
        all(isfinite, tau_rel) && all(isfinite, tau_iid) || return 1e18
        rows = vcat(collect(1:length(s)), collect(1:length(s)))
        cols = vcat(rel_index, G .+ iid_index)
        vals = vcat(tau_rel[rel_index], tau_iid[iid_index])
        loading = sparse(rows, cols, vals, length(s), G + H)
        eta0 = Xmu * @view(theta[1:pμ])
        b, ch, ok = _binomial_phylo_ordinary_mode(aux, eta0, loading, Q, last_b)
        if !ok
            b, ch, ok = _binomial_phylo_ordinary_mode(aux, eta0, loading, Q,
                                                       zeros(length(last_b)))
        end
        ok || return 1e18
        last_b .= b
        eta = eta0 + loading * b
        value = 0.5 * dot(b, Q * b) - 0.5 * logdetQrel + 0.5 * logdet(ch)
        @inbounds for i in eachindex(eta)
            value += _laplace_value(Val(:binomial), aux, i, eta[i])
        end
        return value
    end
    grad! = function (gradient, theta)
        for j in eachindex(theta)
            h = 2e-4 * (1 + abs(theta[j]))
            offset = zeros(length(theta)); offset[j] = h
            gradient[j] = (nll(theta .+ offset) - nll(theta .- offset)) / (2h)
        end
        gradient
    end
    pbar = clamp(sum(s) / max(sum(ntr), 1), 1e-4, 1 - 1e-4)
    theta0 = zeros(pμ + pr + pi)
    theta0[1] = _binomial_start(fam, pbar)
    theta0[rel_range] .= Wrel \ fill(log(0.4), G)
    theta0[iid_range] .= Wiid \ fill(log(0.4), H)
    od = Optim.OnceDifferentiable(nll, grad!, theta0)
    res = Optim.optimize(od, theta0, Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
                         Optim.Options(g_tol = g_tol, iterations = 300))
    theta_hat = Optim.minimizer(res)
    nll_hat = nll(theta_hat)
    gfinal = zeros(length(theta_hat)); grad!(gfinal, theta_hat)
    converged = _laplace_outer_converged(res, nll_hat, gfinal, theta_hat, length(s), g_tol)
    V = se ? _vcov_from_fd_hessian(
        _finite_hessian(nll, theta_hat; h = _fd_hessian_step(length(s)));
        context = "joint sparse-Laplace Binomial related-matrix and IID SD regression") :
        fill(NaN, length(theta_hat), length(theta_hat))
    blocks_sd = _sd_target_blocks(targets)
    rel_block = only(blocks_sd[findall(item -> item.target.level === :relmat, targets)])
    iid_block = only(blocks_sd[findall(item -> item.target.level === :iid, targets)])
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => (1:pμ), rel_block => rel_range,
                                         iid_block => iid_range]
    names = Pair{Symbol,Vector{String}}[:mu => nmmu, rel_block => nm_rel,
                                        iid_block => nm_iid]
    means = Dict(:mu => _binomial_mean.(Ref(fam), Xmu * theta_hat[1:pμ]))
    obs = Dict(:mu => s ./ ntr)
    scales = Dict{Symbol,Vector{Float64}}(
        :trials => Float64.(nint),
        Symbol("sd_group_index_", rel_target.target.group) => Float64.(rel_index),
        Symbol("sd_group_index_", iid_group) => Float64.(iid_index),
        rel_block => _binomial_sd_scale(Wrel * theta_hat[rel_range])[rel_index],
        iid_block => _binomial_sd_scale(Wiid * theta_hat[iid_range])[iid_index],
    )
    fit = DrmFit(fam, blocks, names, theta_hat, Matrix(V), -nll_hat, length(s),
                 converged, means, obs, scales)
    return _withformula(_withiterations(_withnll(fit, _nll_pinned(nll, theta_hat, nll_hat), grad!),
                                        Optim.iterations(res)), f)
end

function _binomial_phylo_ordinary_mode(aux, eta0, A, Q, b0;
                                      maxiter::Int = 60, tol::Real = 1e-10)
    b = copy(b0)
    function joint(bb)
        eta = eta0 + A * bb
        value = 0.5 * dot(bb, Q * bb)
        @inbounds for i in eachindex(eta)
            value += _laplace_value(Val(:binomial), aux, i, eta[i])
        end
        value
    end
    for _ in 1:maxiter
        eta = eta0 + A * b
        grad = Q * b
        weights = zeros(Float64, length(eta))
        all_w_nonneg = true
        @inbounds for i in eachindex(eta)
            _, r, w, _ = _laplace_v123(Val(:binomial), aux, i, eta[i])
            grad .+= vec(Array(A[i, :])) .* r
            weights[i] = w
            w < 0 && (all_w_nonneg = false)
        end
        H = Q + A' * spdiagm(0 => weights) * A
        ch = cholesky(Symmetric(H); check = false)
        issuccess(ch) || return b, ch, false
        step = ch \ grad
        norm(step) <= tol * (1 + norm(b)) && return b, ch, true
        if all_w_nonneg && norm(step) <= 1e-3 * (1 + norm(b))
            b .-= step
            continue
        end
        f0 = joint(b)
        a = 1.0
        accepted = false
        while a >= 1e-5
            trial = b .- a .* step
            if joint(trial) <= f0
                b = trial
                accepted = true
                break
            end
            a *= 0.5
        end
        accepted || return b, ch, false
    end
    return b, nothing, false
end

function _drm_binomial_phylo_ordinary_sd_regression(f::DrmFormula, fam::Binomial;
                                                    data, tree, g_tol::Real,
                                                    se::Bool)
    phy_targets = _sdphylo_parts(f)
    ordinary_targets = _sd_target_parts(f)
    length(phy_targets) == 1 || throw(ArgumentError(
        "Binomial mixed phylogenetic SD regression requires exactly one phylogenetic SD formula."))
    length(ordinary_targets) == 1 || throw(ArgumentError(
        "Binomial mixed phylogenetic SD regression currently requires exactly one ordinary SD formula."))
    species, phy_rhs = only(phy_targets)
    ordinary = only(ordinary_targets)
    target = ordinary.target
    (target.level === :iid && target.dpar === :mu && target.block === nothing) ||
        throw(ArgumentError("The ordinary SD formula must target one IID mean coefficient."))
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing || throw(ArgumentError("Binomial mixed phylogenetic SD regression cannot use meta_V()."))
    st !== nothing && first(st) === :phylo || throw(ArgumentError(
        "Binomial `sd($species, phylogenetic)` requires `phylo(1 | $species)` in the location formula."))
    tree === nothing && throw(ArgumentError("phylo(1 | $species) needs `tree = …`"))
    last(st) === species || throw(ArgumentError(
        "The phylogenetic SD formula must name the group in `phylo(1 | $(last(st)))`."))
    length(re) == 1 || throw(ArgumentError(
        "Binomial mixed phylogenetic SD regression currently requires one ordinary random intercept."))
    term, group = only(re)
    first(_re_kind(term)) === :intercept || throw(ArgumentError(
        "This Binomial mixed phylogenetic route currently requires an ordinary random intercept."))
    target.group === group || throw(ArgumentError(
        "The ordinary SD formula must name the ordinary effect group `$group`."))
    for (pname, form) in f.forms
        (pname === :mu || pname === ordinary.key || pname === Symbol("sdphy_", species)) && continue
        pname === :sigma && form == ConstantTerm(1) && continue
        throw(ArgumentError(
            "Binomial mixed phylogenetic SD regression supports `mu`, one ordinary SD formula, and one phylogenetic SD formula."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Binomial mixed phylogenetic SD regression requires observed responses; remove incomplete rows before fitting."))

    s, ntr = _binomial_response(f, data)
    _, Xmu, nmmu = _design(f.response, fixed_mu, data)
    labels = getproperty(data, species)
    _, tip_index, Gtip = _lss_phylo_group_index(tree, labels, species)
    group_index, Ggroup = _group_index(getproperty(data, group))
    Wgroup, nmgroup = _sd_group_design(f.response, ordinary.rhs, data,
        group_index, Ggroup, group; schema_cache = f.schema_cache,
        schema_key = ordinary.key)
    size(Wgroup, 2) <= Ggroup && rank(Wgroup) == size(Wgroup, 2) ||
        throw(ArgumentError("Binomial ordinary SD design is rank deficient across groups."))
    species_group, Gspecies = _group_index(labels)
    Gspecies == Gtip || error("phylogenetic group indexing changed unexpectedly")
    Wspecies, nmphy = _sd_group_design(f.response, phy_rhs, data, species_group,
        Gspecies, species; schema_cache = f.schema_cache, schema_key = :sd_phylo)
    tip_group = zeros(Int, Gtip)
    for i in eachindex(tip_index)
        ti, gi = tip_index[i], species_group[i]
        tip_group[ti] == 0 || tip_group[ti] == gi || throw(ArgumentError(
            "Phylogenetic SD predictors must be constant within each tree tip."))
        tip_group[ti] = gi
    end
    all(!iszero, tip_group) || throw(ArgumentError(
        "Every tree tip needs a phylogenetic SD predictor."))
    Wtip = Matrix{Float64}(Wspecies[tip_group, :])
    Zphy = Wtip[tip_index, :]
    nint, sint = round.(Int, ntr), round.(Int, s)
    logchoose = [_logfactorial(nint[i]) - _logfactorial(sint[i]) -
                 _logfactorial(nint[i] - sint[i]) for i in eachindex(sint)]
    aux = fam.link === :logit ?
        (s = sint, ntr = nint, logchoose = logchoose) :
        (s = sint, ntr = nint, logchoose = logchoose, link = fam.link)
    Qphy, leaf_node, _ = _poisson_phylo_setup(tree, labels)
    qphy, pμ, pord, pphy = size(Qphy, 1), size(Xmu, 2),
                            size(Wgroup, 2), size(Wtip, 2)
    cholphy = cholesky(Symmetric(Qphy); check = false)
    issuccess(cholphy) || error("phylo(1 | $species) tree precision is not positive definite after root conditioning")
    logdetQphy = logdet(cholphy)
    last_b = zeros(qphy + Ggroup)
    function nll(theta)
        beta = @view theta[1:pμ]
        alpha_ord = @view theta[(pμ + 1):(pμ + pord)]
        alpha_phy = @view theta[(pμ + pord + 1):(pμ + pord + pphy)]
        tau_ord = _binomial_sd_scale(Wgroup * alpha_ord)
        tau_phy = _binomial_sd_scale(Wtip * alpha_phy)
        all(isfinite, tau_ord) && all(isfinite, tau_phy) || return 1e18
        n = length(s)
        A = sparse(vcat(collect(1:n), collect(1:n)),
                   vcat(leaf_node, qphy .+ group_index),
                   vcat(tau_phy[tip_index], tau_ord[group_index]), n, qphy + Ggroup)
        Qord = spdiagm(0 => ones(Ggroup))
        Q = [Qphy spzeros(qphy, Ggroup); spzeros(Ggroup, qphy) Qord]
        logdetQ = logdetQphy
        eta0 = Xmu * beta
        b, ch, ok = _binomial_phylo_ordinary_mode(aux, eta0, A, Q, last_b)
        if !ok
            b, ch, ok = _binomial_phylo_ordinary_mode(aux, eta0, A, Q,
                                                       zeros(length(last_b)))
        end
        ok || return 1e18
        last_b .= b
        eta = eta0 + A * b
        joint = 0.5 * dot(b, Q * b)
        @inbounds for i in eachindex(eta)
            joint += _laplace_value(Val(:binomial), aux, i, eta[i])
        end
        return joint - 0.5 * logdetQ + 0.5 * logdet(ch)
    end
    grad! = function (gradient, theta)
        for j in eachindex(theta)
            h = 2e-4 * (1 + abs(theta[j]))
            offset = zeros(length(theta)); offset[j] = h
            gradient[j] = (nll(theta .+ offset) - nll(theta .- offset)) / (2h)
        end
        gradient
    end
    pbar = clamp(sum(s) / max(sum(ntr), 1), 1e-4, 1 - 1e-4)
    theta0 = zeros(pμ + pord + pphy)
    theta0[1] = _binomial_start(fam, pbar)
    theta0[(pμ + 1):(pμ + pord)] .= Wgroup \ fill(log(0.4), Ggroup)
    theta0[pμ + pord + 1] = log(0.4)
    od = Optim.OnceDifferentiable(nll, grad!, theta0)
    res = Optim.optimize(od, theta0,
        Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
        Optim.Options(g_tol = g_tol, iterations = 250))
    theta_hat = Optim.minimizer(res)
    value = nll(theta_hat)
    gfinal = zeros(length(theta_hat)); grad!(gfinal, theta_hat)
    converged = _laplace_outer_converged(res, value, gfinal, theta_hat, length(s), g_tol)
    V = se ? _vcov_from_fd_hessian(
        _finite_hessian(nll, theta_hat; h = _fd_hessian_step(length(s)));
        context = "joint sparse-Laplace Binomial ordinary and phylogenetic SD regression") :
        fill(NaN, length(theta_hat), length(theta_hat))
    ord_block = _sd_target_block(target)
    ord_range, phy_range = (pμ + 1):(pμ + pord), (pμ + pord + 1):(pμ + pord + pphy)
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => (1:pμ), ord_block => ord_range,
                                         :sd_phylo => phy_range]
    names = Pair{Symbol,Vector{String}}[:mu => nmmu, ord_block => nmgroup,
                                        :sd_phylo => nmphy]
    means = Dict(:mu => _binomial_mean.(Ref(fam), Xmu * theta_hat[1:pμ]))
    obs = Dict(:mu => s ./ ntr)
    tau_ord = _binomial_sd_scale(Wgroup * theta_hat[ord_range])
    tau_phy = _binomial_sd_scale(Wtip * theta_hat[phy_range])
    tau_ord_obs, tau_phy_obs = tau_ord[group_index], tau_phy[tip_index]
    scales = Dict(:trials => Float64.(nint), :sd_group_index => Float64.(group_index),
                  :sd_group => tau_ord, :sd => tau_ord_obs, ord_block => tau_ord_obs,
                  :sd_phylo => tau_phy_obs, :tree_tip_index => Float64.(tip_index))
    fit = DrmFit(fam, blocks, names, theta_hat, Matrix(V), -value, length(s),
                 converged, means, obs, scales)
    return _withformula(_withiterations(_withnll(fit, nll), Optim.iterations(res)), f)
end

# One ordinary correlated block `(1 + x | g)` on the mean (no structured term) whose `sd()` targets
# are all ordinary mean coefficients: the shape `marginal = :Laplace` takes as a one-node group integral.
function _binomial_ordinary_corr_sd(f::DrmFormula)
    rhs = Dict(f.forms)
    haskey(rhs, :mu) || return false
    split = try
        _split_ranef(rhs[:mu])
    catch err
        err isa ArgumentError || rethrow()
        return false
    end
    _, re, mv, st = split
    (mv === nothing && st === nothing && length(re) == 1) || return false
    _laplace_lhs_kind(re[1][1]) === :corr || return false
    isempty(_sdphylo_parts(f)) || return false
    targets = _sd_target_parts(f)
    return !isempty(targets) && all(item -> item.target.level === :iid && item.target.dpar === :mu &&
                                            item.target.block === nothing, targets)
end

function _drm_binomial_sd_regression(f::DrmFormula, fam::Binomial; data,
                                     tree = nothing, tree1 = nothing,
                                     tree2 = nothing, K = nothing, A = nothing,
                                     Ainv = nothing,
                                     coords = nothing, mesh = nothing,
                                     g_tol::Real = 1e-8,
                                     se::Bool = true, marginal::Symbol = :LA,
                                     method = nothing, nq::Int = _BINOMIAL_RANEF_AGHQ_K)
    _reject_method_as_marginal(fam, method)
    if _scalar_laplace_requested(marginal)
        # Independent terms: the dense TMB-convention Laplace. One ordinary correlated block
        # `(1 + x | g)` with SD formulas: the same group integral with a single adaptive node,
        # which is the Laplace approximation (the node count is the only difference to the default).
        laplace_fit = _laplace_sd_route(f, fam, Val(:binomial), marginal; data, g_tol,
                                        fallback = _binomial_ordinary_corr_sd(f))
        laplace_fit === nothing || return laplace_fit
        return _withmarginal(_drm_binomial_sd_regression(f, fam; data, tree, tree1, tree2, K, A, Ainv,
                                                         coords, mesh, g_tol, se, marginal = :LA,
                                                         method, nq = 1), :Laplace)
    end
    marginal === :LA || throw(ArgumentError(
        "Binomial `sd(group) ~ …` currently supports `marginal = :LA` " *
            "only: one ordinary grouping variable uses adaptive Gauss-Hermite quadrature, " *
            "while crossed independent intercepts or numeric slopes use a Laplace approximation."))
    q2ps = _phylo_spatial_q2_block(f)
    q2ps === nothing || return _drm_binomial_phylo_spatial_q2_sd(f, fam, q2ps...;
        data, tree, K, A, Ainv, coords, mesh, g_tol, se)
    if fam.link !== :logit
        has_phylogenetic_sd = !isempty(_sdphylo_parts(f))
        targets = _sd_target_parts(f)
        has_structured_sd = any(item -> item.target.level !== :iid, targets)
        nonlogit_phylogenetic_only = fam.link in (:probit, :cloglog) &&
                                     has_phylogenetic_sd && isempty(_sd_parts(f))
        nonlogit_mixed_phylo_iid = fam.link in (:probit, :cloglog) &&
            has_phylogenetic_sd && length(_sd_parts(f)) == 1
        one_related_sd = length(targets) == 1 &&
                         only(targets).target.level in (:animal, :relmat, :spatial) &&
                         length(_sd_parts(f)) == 1 &&
                         first(only(_sd_parts(f))) === only(targets).target.group &&
                         !has_phylogenetic_sd
        animal_iid_related_sd = fam.link in (:probit, :cloglog) &&
            length(targets) == 2 &&
            count(item -> item.target.level === :animal, targets) == 1 &&
            count(item -> item.target.level === :iid, targets) == 1 &&
            length(_sd_parts(f)) == 2 && !has_phylogenetic_sd
        relmat_iid_related_sd = fam.link in (:probit, :cloglog) &&
            length(targets) == 2 &&
            count(item -> item.target.level === :relmat, targets) == 1 &&
            count(item -> item.target.level === :iid, targets) == 1 &&
            length(_sd_parts(f)) == 2 && !has_phylogenetic_sd
        animal_q2_sd = fam.link in (:probit, :cloglog) &&
            length(targets) == 2 &&
            count(item -> item.target.level === :animal, targets) == 2 &&
            length(_sd_parts(f)) == 2 && !has_phylogenetic_sd
        nonlogit_related_sd = fam.link in (:probit, :cloglog) &&
                              (one_related_sd || animal_iid_related_sd || relmat_iid_related_sd ||
                               animal_q2_sd)
        nonlogit_phylo_interaction = fam.link in (:probit, :cloglog) &&
            length(targets) == 1 &&
            only(targets).target.level === :phylogenetic_interaction
        (has_structured_sd || has_phylogenetic_sd) &&
            !(nonlogit_phylogenetic_only || nonlogit_mixed_phylo_iid ||
              nonlogit_related_sd || nonlogit_phylo_interaction) &&
            throw(ArgumentError(
                "Binomial links other than :logit are currently supported for ordinary IID SD models, " *
                "and :probit and :cloglog are additionally supported for one phylogenetic SD target or one mixed ordinary-intercept and phylogenetic-intercept target; :probit and " *
                ":cloglog also support one animal, related-matrix, or spatial SD target and the animal q2 SD block."))
    end
    targets = _sd_target_parts(f)
    if any(item -> item.target.level === :phylogenetic_interaction, targets)
        return _drm_binomial_phylo_interaction_sd_regression(
            f, fam; data, tree1, tree2, tree, K, A, Ainv, coords, g_tol, se,
        )
    end
    tree1 === nothing && tree2 === nothing || throw(ArgumentError(
        "Binomial `tree1` and `tree2` are used only by `phylo_interaction()` SD models."))
    if Ainv !== nothing
        any(item -> item.target.level === :animal, targets) || throw(ArgumentError(
            "Binomial `Ainv` is supported for animal SD regression only."))
        isempty(_sdphylo_parts(f)) || throw(ArgumentError(
            "Binomial animal `Ainv` cannot be combined with a phylogenetic SD target."))
    end
    if !isempty(_sdphylo_parts(f))
        isempty(_sd_parts(f)) && return _drm_binomial_phylo_sd_regression(
            f, fam; data, tree, g_tol, se)
        return _drm_binomial_phylo_ordinary_sd_regression(
            f, fam; data, tree, g_tol, se)
    end
    if any(item -> item.target.level === :relmat, targets)
        if any(item -> item.target.level === :iid, targets)
            return _drm_binomial_relmat_iid_sd_regression(
                f, fam; data, K, A, Ainv, tree, coords, g_tol, se,
            )
        end
        return _drm_binomial_relmat_sd_regression(
            f, fam; data, K, tree, A, coords, g_tol, se,
        )
    end
    if any(item -> item.target.level === :spatial, targets)
        if mesh !== nothing
            coords === nothing || throw(ArgumentError(
                "Binomial mesh spatial SD regression accepts `mesh = …` or `coords = …`, not both."))
            return _drm_binomial_mesh_sd_regression(
                f, fam; data, mesh, tree, tree1, tree2, K, A, Ainv, g_tol, se,
            )
        end
        return _drm_binomial_spatial_sd_regression(
            f, fam; data, coords, tree, K, A, g_tol, se,
        )
    end
    mesh === nothing || throw(ArgumentError(
        "Binomial `mesh` is currently used only for spatial random-effect SD regression."))
    if any(item -> item.target.level === :animal, targets)
        if any(item -> item.target.level === :iid, targets)
            return _drm_binomial_animal_iid_sd_regression(
                f, fam; data, A, Ainv, tree, K, coords, g_tol, se,
            )
        end
        return _drm_binomial_animal_sd_regression(
            f, fam; data, A, Ainv, tree, K, coords, g_tol, se,
        )
    end
    Ainv === nothing || throw(ArgumentError(
        "Binomial `Ainv` is supported for animal SD regression only."))
    (tree === nothing && K === nothing && A === nothing && coords === nothing) ||
        throw(ArgumentError("Binomial SD regression currently supports ordinary " *
            "random intercepts, numeric slopes, and correlated intercept-slope blocks only; " *
            "structured covariance inputs are not used."))
    isempty(_sdphylo_parts(f)) || throw(ArgumentError(
        "Binomial `sd(group, phylogenetic) ~ …` is not implemented by the ordinary random-effect route."))
    bare_corr = isempty(targets) && _binomial_bare_corr_block(f, fam)
    isempty(targets) && !bare_corr && throw(ArgumentError("Binomial SD regression needs at least one ordinary SD formula."))
    all(item -> item.target.level === :iid && item.target.dpar === :mu &&
                item.target.block === nothing, targets) || throw(ArgumentError(
        "Binomial ordinary SD regression targets IID mean random-effect coefficients only."))
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing || throw(ArgumentError("Binomial SD regression cannot use meta_V()."))
    st === nothing || throw(ArgumentError("Binomial SD regression does not yet support structured random-effect terms."))
    corr_members(term) = if term isa FunctionTerm && term.f === (+) &&
                            any(t -> t isa ConstantTerm && t.n == 1, term.args)
        vars = filter(t -> t isa Term, term.args)
        [:intercept; getproperty.(vars, :sym)]
    else
        nothing
    end
    correlated = length(re) == 1 && corr_members(re[1][1]) !== nothing
    if correlated
        (bare_corr ? 0 : 1) <= length(targets) <= length(corr_members(re[1][1])) || throw(ArgumentError(
            "Binomial correlated-block SD regression needs one SD formula per selected coefficient."))
    else
        if length(re) != length(targets)
            # Some terms without a formula (constant SD): the shared independent-term Laplace route.
            indep_shape = _indep_sd_shape(f)
            indep_shape === nothing ||
                return _drm_indep_sd_laplace(f, fam, Val(:binomial), indep_shape; data, g_tol)
            throw(ArgumentError(
                "Binomial SD regression requires one SD formula for each independent random-effect coefficient."))
        end
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Binomial random-effect SD regression does not yet support missing responses; " *
        "drop incomplete rows before fitting so the fitted group SD design and bootstrap rows remain aligned."))

    # Resolve each scale formula to one coefficient. The explicit
    # `coef(term)` selector is required for a correlated block.
    re_meta = if correlated
        term, group = only(re)
        members = corr_members(term)
        map(members) do member
            kind = member === :intercept ? :intercept : :slope
            slope = kind === :slope ? member : nothing
            (; term, group, member, kind, slope)
        end
    else
        map(re) do (term, group)
            kind, slope = _re_kind(term)
            kind in (:intercept, :slope) || throw(ArgumentError(
                "Binomial SD regression supports independent `(1 | group)`, `(0 + x | group)` terms, or one correlated `(1 + x | group)` block."))
            member = kind === :intercept ? :intercept : slope
            (; term, group, member, kind, slope)
        end
    end
    group_names = unique(x.group for x in re_meta)
    used = Set{Symbol}()
    ordered_targets = Vector{Union{Nothing,eltype(targets)}}(undef, length(re_meta))
    for (j, effect) in enumerate(re_meta)
        matches = filter(targets) do item
            t = item.target
            t.group === effect.group && (t.member === nothing || t.member === effect.member)
        end
        length(matches) <= 1 || throw(ArgumentError(
            "Binomial SD target is ambiguous for correlated coefficients; select `coef(intercept)` or `coef($(effect.member))`."))
        if isempty(matches)
            correlated || throw(ArgumentError(
                "Binomial SD formulas must identify each independent random-effect coefficient exactly once."))
            ordered_targets[j] = nothing
            continue
        end
        target = only(matches)
        target.key in used && throw(ArgumentError("Binomial SD formulas target one random-effect coefficient more than once."))
        push!(used, target.key)
        ordered_targets[j] = target
    end
    length(used) == length(targets) || throw(ArgumentError(
        "Every Binomial SD formula must match a random-effect coefficient in the mean formula."))

    for (pname, _) in f.forms
        (pname === :mu || haskey(f.sd_targets, pname)) && continue
        pname === :sigma && Dict(f.forms)[pname] === ConstantTerm(1) && continue
        throw(ArgumentError("Binomial SD regression is a mean-only model with no residual sigma formula."))
    end

    if length(group_names) > 1
        (!correlated && length(re_meta) == 2 && length(group_names) == 2) ||
            throw(ArgumentError("Binomial SD regression supports two distinct groups only as independent intercept or numeric-slope terms."))
        all(effect -> effect.kind in (:intercept, :slope), re_meta) || throw(ArgumentError(
            "Binomial crossed SD regression supports ordinary intercepts or independent numeric slopes only."))
        s, ntr = _binomial_response(f, data)
        _, Xmu, nmmu = _design(f.response, fixed_mu, data)
        group_indices = [_group_index(getproperty(data, effect.group)) for effect in re_meta]
        loadings = map(re_meta) do effect
            if effect.kind === :intercept
                ones(Float64, length(s))
            else
                hasproperty(data, effect.slope) || throw(ArgumentError(
                    "Binomial random-slope predictor `$(effect.slope)` is absent from data."))
                values = getproperty(data, effect.slope)
                values isa AbstractVector{<:Real} || throw(ArgumentError(
                    "Binomial random-slope predictor `$(effect.slope)` must be numeric."))
                length(values) == length(s) && all(isfinite, values) || throw(ArgumentError(
                    "Binomial random-slope predictor `$(effect.slope)` must contain one finite value per retained response."))
                Float64.(values)
            end
        end
        for (j, effect) in enumerate(re_meta)
            gidx_j, G_j = group_indices[j]
            informative = falses(G_j)
            @inbounds for i in eachindex(loadings[j])
                !iszero(loadings[j][i]) && (informative[gidx_j[i]] = true)
            end
            any(informative) || throw(ArgumentError(
                "Binomial random-slope predictor for `$(effect.group)` must have at least one nonzero retained value."))
        end
        Zgs, nmsds = Matrix{Float64}[], Vector{String}[]
        group_labels = String[]
        for (j, effect) in enumerate(re_meta)
            gidx_j, G_j = group_indices[j]
            target = ordered_targets[j]
            Zg, nmsd = _sd_group_design(f.response, target.rhs, data, gidx_j,
                                        G_j, effect.group;
                                        schema_cache = f.schema_cache,
                                        schema_key = target.key)
            informative = falses(G_j)
            @inbounds for i in eachindex(loadings[j])
                !iszero(loadings[j][i]) && (informative[gidx_j[i]] = true)
            end
            active_design = Zg[findall(informative), :]
            size(Zg, 2) <= size(active_design, 1) &&
                rank(active_design) == size(Zg, 2) || throw(ArgumentError(
                "Binomial SD target for `$(effect.group)` has a rank-deficient design over group levels contributing nonzero random-effect loadings."))
            push!(Zgs, Matrix{Float64}(Zg))
            push!(nmsds, nmsd)
            push!(group_labels, String(effect.group))
        end
        fit = _fit_binomial_crossed_sd_laplace(
            fam, s, ntr, Xmu, Zgs, nmsds,
            group_indices[1][1], group_indices[1][2],
            group_indices[2][1], group_indices[2][2],
            loadings[1], loadings[2],
            nmmu, group_labels, g_tol; se,
            target_blocks = Symbol[_sd_target_block(
                item.target; disambiguate = count(item -> item !== nothing, ordered_targets) > 1,
            ) for item in ordered_targets],
        )
        return _withformula(fit, f)
    end

    s, ntr = _binomial_response(f, data)
    _, Xmu, nmmu = _design(f.response, fixed_mu, data)
    group = first(re_meta).group
    grpcol = getproperty(data, group)
    gidx, G = _group_index(grpcol)
    n = length(s)
    Zre = Matrix{Float64}(undef, n, length(re_meta))
    Zgs = Matrix{Float64}[]
    nmsds = Vector{String}[]
    sd_ranges = UnitRange{Int}[]
    target_blocks = Symbol[]
    for (j, effect) in enumerate(re_meta)
        raw = if effect.kind === :intercept
            ones(n)
        else
            hasproperty(data, effect.slope) || throw(ArgumentError(
                "Binomial random-slope predictor `$(effect.slope)` is absent from data."))
            values = getproperty(data, effect.slope)
            values isa AbstractVector{<:Real} || throw(ArgumentError(
                "Binomial random-slope predictor `$(effect.slope)` must be numeric."))
            all(isfinite, values) || throw(ArgumentError(
                "Binomial random-slope predictor `$(effect.slope)` must contain finite values."))
            Float64.(values)
        end
        Zre[:, j] = raw
        target = ordered_targets[j]
        if target === nothing
            Zg = ones(G, 1)
            nmsd = ["$(group):log_sd_$(effect.member)"]
            push!(target_blocks, Symbol("sdconst_", effect.member))
        else
            Zg, nmsd = _sd_group_design(f.response, target.rhs, data, gidx, G, group;
                                        schema_cache = f.schema_cache,
                                        schema_key = target.key)
            push!(target_blocks, _sd_target_block(
                target.target; disambiguate = length(targets) > 1,
            ))
        end
        size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
            "Binomial SD target for `$(effect.member)` has a rank-deficient group-level design."))
        push!(Zgs, Zg)
        push!(nmsds, nmsd)
    end
    if correlated
        for j in 2:size(Zre, 2)
            _binomial_check_slope_identifiable(Float64.(Zre[:, j]), gidx, G, group)
        end
    end
    return _withformula(_fit_binomial_sd_regression(
        fam, s, ntr, Xmu, Zgs, nmsds, gidx, G, nmmu, group, g_tol,
        target_blocks; nq = nq, Zre = Zre, correlated = correlated,
    ), f)
end

# Nats an unconverged restart must gain over a converged first fit to replace it (independent terms).
const _BINOMIAL_INDEP_BIG_GAIN = 1.0

function _fit_binomial_sd_regression(fam::Binomial, s, ntr, Xmu, Zgs, nmsds,
                                      gidx, G, nmmu, group, g_tol, target_blocks;
                                      nq::Int = _BINOMIAL_RANEF_AGHQ_K,
                                      Zre = ones(length(s), 1), correlated::Bool = false)
    n = length(s)
    pmu = size(Xmu, 2)
    q = length(Zgs)
    length(Zre) == n * q || throw(ArgumentError("random-effect design dimensions do not match the response"))
    sint, nint = round.(Int, s), round.(Int, ntr)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    lc = [_logchoose(nint[i], sint[i]) for i in 1:n]
    rule = _AGHQRule(q, nq)
    psd = size.(Zgs, 2)
    is_constant = startswith.(String.(target_blocks), "sdconst_")
    offset = pmu
    sd_ranges = Vector{UnitRange{Int}}(undef, q)
    for j in 1:q
        if !is_constant[j]
            sd_ranges[j] = offset + 1:offset + psd[j]
            offset += psd[j]
        end
    end
    constant_ranges = UnitRange{Int}[]
    constant_names = String[]
    for j in 1:q
        if is_constant[j]
            sd_ranges[j] = offset + 1:offset + psd[j]
            offset += psd[j]
            push!(constant_ranges, sd_ranges[j])
            push!(constant_names, only(nmsds[j]))
        end
    end
    ncor = correlated ? q * (q - 1) ÷ 2 : 0
    corr_range = offset + 1:offset + ncor
    offset += ncor
    function nll(theta)
        betamu = @view theta[1:pmu]
        eta0 = Xmu * betamu
        sdgroup = hcat((_binomial_sd_scale(Zgs[j] * @view(theta[sd_ranges[j]])) for j in 1:q)...)
        ll = (i, eta) -> _binomial_link_ll(fam, nint[i], sint[i], eta, lc[i])
        bcache = zeros(eltype(theta), q, G)
        total = zero(eltype(theta))
        correlation_eta = correlated ? @view(theta[corr_range]) : eltype(theta)[]
        Lcorr = correlated ? _binomial_partial_correlation_cholesky(correlation_eta, q) : nothing
        for gi in 1:G
            Lg = if correlated
                Matrix(Diagonal(@view(sdgroup[gi, :])) * Lcorr)
            else
                Matrix(Diagonal(@view(sdgroup[gi, :])))
            end
            value, bhat = _aghq_group_logint(ll, members[gi], eta0, Zre, Lg,
                                               rule, @view(bcache[:, gi]))
            bcache[:, gi] .= ifelse.(isfinite.(bhat), bhat, zero(eltype(theta)))
            total += value
        end
        return -total
    end
    theta0 = zeros(offset)
    pbar = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)
    theta0[1] = _binomial_start(fam, pbar)
    for j in 1:q
        # An intercept-only SD design (`sd(g, coef(x)) ~ 1`) starts at exactly the constant-SD
        # value: `ones \ fill(log(0.5), G)` is one ulp off `log(0.5)`, and that ulp alone sent the
        # q = 3 correlated fit down a different path from the bare `(1 + x1 + x2 | g)` spelling.
        if is_constant[j] || (size(Zgs[j], 2) == 1 && all(isone, Zgs[j]))
            theta0[sd_ranges[j]] .= log(0.5)
        else
            theta0[sd_ranges[j]] .= Zgs[j] \ fill(log(0.5), G)
        end
    end
    correlated && (theta0[corr_range] .= 0.0)
    run_from(th0) = _optimize_with_fallback(nll, th0, g_tol)
    res = run_from(theta0)
    # Restarts for independent random effects (see `_q2_boundary_restarts`). With a
    # covariate in an SD formula (`sd(g, coef(.)) ~ z`) the objective is a ridge in the
    # SD coefficients: the single default start can stop at a smooth interior point,
    # converged and with every SD well above the boundary, that lies 1 to 6 nats below
    # a point the same objective scores higher (G12 of the wave-8 disagreement map;
    # drmTMB reports a higher value, on an objective that differs by about 0.3 nats there). The boundary trigger never fires there, so such
    # fits restart unconditionally, as the Gamma sd-z two-field route does; intercept-only
    # SD formulas keep the usual trigger (fit near the boundary or not converged).
    # An alternative is accepted only on a clear objective gain, so a fit whose first
    # optimum stands is unchanged. An UNCONVERGED alternative replaces a CONVERGED first
    # fit only when it gains more than `_BINOMIAL_INDEP_BIG_GAIN` nats: with the generic
    # 1e-3 rule a restart turned converged fits into unconverged ones (q = 2 seed 11, gain
    # 0.18; q = 3 seeds 6 and 27, gains 0.29 and 0.98), leaving a non-stationary point for
    # the Hessian. Correlated blocks are not touched.
    restart_info = Ref{Any}((ran = false, winner = 0, gain = 0.0, flipped = false))
    if !correlated && q >= 2
        res = _q2_boundary_restarts(run_from, Float64[], res,
            (sd0, sd1, het) -> begin
                th = copy(theta0)
                for j in 2:q
                    th = _q2_alt_theta(th, sd_ranges[[1, j]], (Zgs[1], Zgs[j]), sd0, sd1, het)
                end
                th
            end,
            th -> minimum(minimum(_binomial_sd_scale(Zgs[j] * th[sd_ranges[j]])) for j in 1:q),
            drm_optim_converged; info = restart_info,
            force = any(>(1), psd), big_gain = _BINOMIAL_INDEP_BIG_GAIN)
    end
    ri = restart_info[]
    thetahat = Optim.minimizer(res)
    V = _vcov_or_nan(ForwardDiff.hessian(nll, thetahat))
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:pmu]
    names = Pair{Symbol,Vector{String}}[:mu => nmmu]
    for j in 1:q
        is_constant[j] && continue
        push!(blocks, target_blocks[j] => sd_ranges[j])
        push!(names, target_blocks[j] => nmsds[j])
    end
    if correlated
        recov_start = isempty(constant_ranges) ? first(corr_range) : first(first(constant_ranges))
        push!(blocks, :recov => (recov_start:last(corr_range)))
        corr_names = _binomial_partial_correlation_names(group, q)
        push!(names, :recov => [constant_names; corr_names])
    end
    means = Dict(:mu => _binomial_mean.(Ref(fam), Xmu * thetahat[1:pmu]))
    obs = Dict(:mu => s ./ ntr)
    sdgroups = [_binomial_sd_scale(Zgs[j] * thetahat[sd_ranges[j]]) for j in 1:q]
    sdobs = [sdgroups[j][gidx] for j in 1:q]
    scales = Dict(:trials => Float64.(nint),
                  :sd_group_index => Float64.(gidx),
                  :aghq_nodes => [Float64(nq)])
    (correlated || q < 2) || (scales[:q2_restarts] = [ri.ran ? 1.0 : 0.0, Float64(ri.winner), Float64(ri.gain), ri.flipped ? 1.0 : 0.0])
    if correlated
        eta_corr = @view thetahat[corr_range]
        pairwise_corr = _binomial_pairwise_correlations(eta_corr, q)
        # `:recov` retains the unconstrained partial-correlation parameters;
        # `:corpars` exposes marginal pairwise correlations in coefficient
        # lexicographic order (1,2), (1,3), …, (q-1,q).
        scales[:corpars] = Float64.(pairwise_corr)
    end
    # Preserve historical aliases for a single target. With multiple targets,
    # generic concatenated vectors would have no single-target meaning.
    if q == 1
        scales[:sd_group] = sdgroups[1]
        scales[:sd] = sdobs[1]
    end
    # `marginal_parameters()` follows fitted parameter blocks, so each SD block
    # needs its own row-level values under that exact fitted block name.
    for j in 1:q
        is_constant[j] || (scales[target_blocks[j]] = sdobs[j])
    end
    fit = DrmFit(fam, blocks, names, thetahat, V, -nll(thetahat), n,
        drm_optim_converged(res), means, obs, scales)
    return _withiterations(_withnll(fit, nll), Optim.iterations(res))
end
