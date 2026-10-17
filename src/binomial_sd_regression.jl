# Binomial/Bernoulli random-intercept SD regression.  For group g,
# logit(p_i) = X_i β + b_g, b_g ~ N(0, exp(2 Z_g α)).  The SD predictor is
# evaluated once per group, not per observation; there is no Bernoulli residual
# scale parameter.  Each independent scalar b_g is integrated by the existing
# adaptive Gauss-Hermite rule, with a group-specific 1×1 Cholesky factor.

function _drm_binomial_sd_regression(f::DrmFormula, fam::Binomial; data,
                                     tree = nothing, K = nothing, A = nothing,
                                     coords = nothing, g_tol::Real = 1e-8,
                                     se::Bool = true, marginal::Symbol = :LA,
                                     method = nothing, nq::Int = _BINOMIAL_RANEF_AGHQ_K)
    _reject_method_as_marginal(fam, method)
    marginal === :LA || throw(ArgumentError(
        "Binomial `sd(group) ~ …` currently supports `marginal = :LA` " *
        "(the ordinary q=1 adaptive-Gauss-Hermite route) only."))
    (tree === nothing && K === nothing && A === nothing && coords === nothing) ||
        throw(ArgumentError("Binomial `sd(group) ~ …` currently supports an ordinary " *
            "random intercept only; structured covariance inputs are not used."))
    isempty(_sdphylo_parts(f)) || throw(ArgumentError(
        "Binomial `sd(group, phylogenetic) ~ …` is not implemented by the ordinary random-intercept route."))
    sdp = _sd_parts(f)
    length(sdp) == 1 || throw(ArgumentError(
        "Binomial random-effect SD regression requires exactly one `sd(group) ~ …` formula."))
    sgrp, srhs = only(sdp)
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing || throw(ArgumentError("Binomial `sd($sgrp)` cannot use meta_V()."))
    st === nothing || throw(ArgumentError("Binomial `sd($sgrp)` currently requires an ordinary `(1 | $sgrp)` effect."))
    length(re) == 1 || throw(ArgumentError(
        "Binomial `sd($sgrp)` currently requires exactly one ordinary `(1 | $sgrp)` random intercept."))
    kind, _ = _re_kind(re[1][1])
    kind === :intercept || throw(ArgumentError(
        "Binomial `sd($sgrp)` currently supports `(1 | $sgrp)` only; slope SD regression is pending."))
    grp = re[1][2]
    grp === sgrp || throw(ArgumentError(
        "Binomial `sd($sgrp)` must name the random-effect grouping variable `$grp`."))
    for (pname, r) in f.forms
        pname in (:mu, Symbol("sd_", sgrp)) && continue
        pname === :sigma && r === ConstantTerm(1) && continue
        throw(ArgumentError("Binomial `sd($sgrp)` is a mean-only model with no residual sigma formula."))
    end
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        _drm_binomial_sd_regression(f, fam; data = data_observed, g_tol = g_tol,
                                    se = se, marginal = marginal, method = method, nq = nq)
    end
    missing_fit !== nothing && return missing_fit

    s, ntr = _binomial_response(f, data)
    _, Xmu, nmmu = _design(f.response, fixed_mu, data)
    gidx, G = _group_index(getproperty(data, grp))
    Zg, nmsd = _sd_group_design(f.response, srhs, data, gidx, G, grp)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "Binomial `sd($grp)` group-level design is rank deficient; use predictors that vary independently across groups."))
    return _withformula(_fit_binomial_sd_regression(fam, s, ntr, Xmu, Zg, gidx, G,
        nmmu, nmsd, grp, g_tol; nq = nq), f)
end

function _fit_binomial_sd_regression(fam::Binomial, s, ntr, Xmu, Zg, gidx, G,
                                      nmmu, nmsd, grp, g_tol;
                                      nq::Int = _BINOMIAL_RANEF_AGHQ_K)
    n = length(s); pmu = size(Xmu, 2); psd = size(Zg, 2)
    sint = round.(Int, s); nint = round.(Int, ntr)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    lc = [_logchoose(nint[i], sint[i]) for i in 1:n]
    rule = _AGHQRule(1, nq)
    Zre = ones(n, 1)
    bcache = zeros(1, G)
    function nll(theta)
        betamu = @view theta[1:pmu]
        alpha = @view theta[(pmu+1):(pmu+psd)]
        eta0 = Xmu * betamu
        sdgroup = exp.(Zg * alpha)
        ll = (i, eta) -> _binomial_logit_ll(nint[i], sint[i], eta, lc[i])
        total = zero(eltype(theta))
        for gi in 1:G
            Lg = reshape([sdgroup[gi]], 1, 1)
            value, bhat = _aghq_group_logint(ll, members[gi], eta0, Zre, Lg,
                                               rule, @view(bcache[:, gi]))
            bcache[1, gi] = isfinite(bhat[1]) ? bhat[1] : 0.0
            total += value
        end
        return -total
    end
    pbar = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)
    theta0 = zeros(pmu + psd)
    theta0[1] = log(pbar / (1 - pbar))
    theta0[(pmu+1):end] .= Zg \ fill(log(0.5), G)
    res = _optimize_with_fallback(nll, theta0, g_tol)
    thetahat = Optim.minimizer(res)
    V = _vcov_or_nan(ForwardDiff.hessian(nll, thetahat))
    blocks = [:mu => 1:pmu, :sd => (pmu+1):(pmu+psd)]
    names = [:mu => nmmu, :sd => nmsd]
    means = Dict(:mu => _logistic.(Xmu * thetahat[1:pmu]))
    obs = Dict(:mu => s ./ ntr)
    scales = Dict(:trials => Float64.(nint),
                  :sd_group => exp.(Zg * thetahat[(pmu+1):end]),
                  :sd_group_index => Float64.(gidx),
                  :aghq_nodes => [Float64(nq)])
    return _withiterations(_withnll(DrmFit(fam, blocks, names, thetahat, V,
        -nll(thetahat), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end
