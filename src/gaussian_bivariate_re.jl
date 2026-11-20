# gaussian_bivariate_re.jl -- bivariate Gaussian with an ORDINARY correlated
# random intercept `(1 | p | g)` on mu1 and mu2 (drmTMB's `biv_gaussian()` with a
# 2x2 group-level covariance, separate from the residual `rho12`).
#
# Model.  Row i of group g: y_i = X_i b + u_g + e_i, u_g ~ N(0, Lam),
# e_i ~ N(0, S_i), S_i = [s1^2, rho s1 s2; rho s1 s2, s2^2] with log s1, log s2 and
# atanh(rho / RHO_GUARD) linear in their own designs (the residual route's links).
# mu1 and mu2 may have DIFFERENT fixed-effect designs. Groups are independent, and
# within a group the 2n_g-vector has covariance D + Z Lam Z', D = blockdiag(S_i),
# Z = 1_n (x) I_2. Woodbury with Lam = L L' (never Lam^-1, so a boundary Lam is
# fine): with A = sum_i S_i^-1, s = sum_i S_i^-1 r_i, W = (I + L'AL)^-1,
#   logdet V_g = sum_i logdet S_i + logdet(I + L'AL)
#   r'V_g^-1 r = sum_i r_i'S_i^-1 r_i - (L's)' W (L's).
# Everything is a sum of 2x2 pieces, so the objective is ForwardDiff-friendly.
#
# Not the structured q=2 route: that one shares one design and an intercept-only
# sigma/rho, and builds a sparse Laplace problem. This route is dense per group.
#
# Post-fit conventions. `fit.means` holds X b only (no BLUPs), so `residuals(fit)` are MARGINAL
# (y - X b), the same convention as the other bivariate routes; drmTMB's `residuals()` are
# conditional on the fitted group effects. The BLUPs are in `ranef(fit)`. `simulate(fit; data)`
# and the parametric bootstrap draw fresh group effects from `fit.ranef.Sigma_g`
# (`_biv_ordinary_re_simulator`).
#
# REML integrates out the mu1/mu2 fixed effects: -loglik_REML = -loglik_ML(beta_hat)
# + 0.5 logdet(X'V^-1 X) - 0.5 p log(2 pi), the same convention (and the same
# vcov assembly) as `_fit_bivariate_residual_reml`.

# Coefficient names of the 2x2 group covariance (log-Cholesky). `Sigma_g`, not `Sigma_a`:
# `Sigma_a` belongs to the structured (phylo / relmat / animal / spatial) routes, and the
# bridge and bootstrap code keys on it; keeping the names apart stops this fit being read as one.
# The block is named by `_BIV_RE_GROUP_BLOCK` (below), not `:phylocov`: nothing here is
# phylogenetic, and `:phylocov` stays the name on the truly phylogenetic bivariate routes.

# THE ONE PLACE that names the group-covariance coefficient block of this route (owner answer P15 = b,
# 10 Nov 2026: rename on this route only; the name was left to the lane). To rename, edit this line.
# `:recov` was not reused: on the univariate routes it holds [logL11, logL22, L21] named "g:L11" and
# the shared extractors (vc, ranef, profile, simulate) parse it in that layout, not this one.
const _BIV_RE_GROUP_BLOCK = :groupcov
_biv_re_group_names() = ["Sigma_g:L11", "Sigma_g:L21", "Sigma_g:L22"]

# Pull one `(1 | tag | group)` term out of mu1 and mu2. Returns `(rhs', spec)` with
# `spec = (tag, group)` or `nothing` (no ordinary term on either side). Anything
# not exactly this shape is left in place so the existing refusal still fires.
function _extract_ordinary_biv_re(rhs)
    spec = Dict{Symbol,Any}()
    rhs2 = Dict{Symbol,Any}(rhs)
    for p in (:mu1, :mu2)
        terms = rhs[p] isa Tuple ? collect(rhs[p]) : Any[rhs[p]]
        keep = Any[]
        for t in terms
            inner = t isa FunctionTerm && t.f === (|) ? t.args[1] : nothing
            if !haskey(spec, p) && inner isa FunctionTerm && inner.f === (|) &&
               inner.args[1] isa ConstantTerm && inner.args[1].n == 1 &&
               inner.args[2] isa Term && t.args[2] isa Term
                spec[p] = (inner.args[2].sym, t.args[2].sym)
            else
                push!(keep, t)
            end
        end
        haskey(spec, p) &&
            (rhs2[p] = isempty(keep) ? ConstantTerm(1) : length(keep) == 1 ? keep[1] : Tuple(keep))
    end
    isempty(spec) && return rhs, nothing
    length(spec) == 2 && spec[:mu1] == spec[:mu2] || throw(ArgumentError(
        "drm: the bivariate ordinary random effect must be written `(1 | p | g)` in BOTH mu1 and mu2 " *
        "with the same label `p` and grouping factor `g` (a 2x2 group covariance). " *
        "One-sided or unlabelled `(1 | g)` terms are not supported in DRModels.jl yet."))
    return rhs2, spec[:mu1]
end

# Starting points for the group covariance. The default start (group SD 0.5, covariance 0.02) can slide to
# Sigma_g = 0, a local optimum where the gradient in the log SD vanishes, while drmTMB finds a higher mode
# with the group SDs away from 0 (about 3 percent of small data sets; reviewed on 240 + 100 data sets). Two
# more starts with group SD 0.2 and a strongly correlated pair (+0.99, -0.99) reach that mode. This is not a
# global search: a better optimum can exist that none of the three starts finds.
const _BIV_RE_EXTRA_STARTS = ((0.2, 0.99), (0.2, -0.99))
# An extra start replaces the default result only when it converged and its objective is lower by more than
# this. The package has no multi-start threshold of its own (`DRMControl(multi_start)` keeps any lower
# objective), so 1e-6 is used: far above optimiser noise, far below the gaps seen between modes (1.6e-4 and up).
const _BIV_RE_START_GAIN = 1e-6

# Errors that a start may raise from a wild line-search trial: they cost that start, not the fit.
_biv_re_start_failure(err) =
    err isa Union{AssertionError,ArgumentError,DomainError,LinearAlgebra.SingularException,
                  LinearAlgebra.PosDefException,LinearAlgebra.LAPACKException}

# Run the default start, then the extra starts (deterministic, in this order), and return the Optim result
# to keep. `run(x)` optimises from `x`; the last three entries of `x0` are the log-Cholesky of Sigma_g.
# Starts are compared on `obj` evaluated in Float64 at the returned minimiser, the value that is reported as
# the log-likelihood, not on Optim's own minimum (computed with dual numbers, which can differ from the
# Float64 value at a nearly singular point).
# The default result is kept unless an extra start converged and is lower by more than `_BIV_RE_START_GAIN`,
# so a fit already at its best optimum is returned unchanged. A start that fails is skipped. If the default
# start fails, the best converged extra start is used (any finite one if none converged); if every start
# fails, the default start's error is rethrown, as before.
function _biv_re_best_start(run, obj, x0::AbstractVector)
    value(r) = obj(Optim.minimizer(r))
    res0 = nothing; err0 = nothing; v0 = NaN
    try
        res0 = run(x0); v0 = value(res0)
    catch err
        _biv_re_start_failure(err) || rethrow()
        res0 = nothing; err0 = err
    end
    conv = nothing; vconv = NaN; anyfinite = nothing; vany = NaN
    for (sd0, cr0) in _BIV_RE_EXTRA_STARTS
        xs = copy(x0)
        xs[(end-2):end] .= cov_to_lc(Matrix(Symmetric(sd0^2 .* [1.0 cr0; cr0 1.0])))
        r = nothing; v = NaN
        try
            r = run(xs); v = value(r)
        catch err
            _biv_re_start_failure(err) || rethrow()
            continue
        end
        isfinite(v) || continue
        (anyfinite === nothing || v < vany) && (anyfinite = r; vany = v)
        Optim.converged(r) || continue
        (conv === nothing || v < vconv) && (conv = r; vconv = v)
    end
    if res0 === nothing
        best = conv !== nothing ? conv : anyfinite
        best === nothing && throw(err0)
        return best
    end
    conv !== nothing && (!isfinite(v0) || vconv < v0 - _BIV_RE_START_GAIN) && return conv
    return res0
end

function _fit_bivariate_ordinary_re(f::BivariateDrmFormula, fam::Gaussian, data, fixed, spec;
                                    g_tol::Real, method::Symbol = :ML)
    tag, grp = spec
    d = _biv_residual_designs(f, fixed, data)
    all(d.obs1) && all(d.obs2) || throw(ArgumentError(
        "drm: the bivariate ordinary random-effect route needs both responses observed on every row; " *
        "drop incomplete rows (the default `missing` policy does) or use the residual route."))
    y1, X1, nm1, y2, X2, nm2 = d.y1, d.X1, d.nm1, d.y2, d.X2, d.nm2
    Xs1, nms1, Xs2, nms2, Xr, nmr = d.Xs1, d.nms1, d.Xs2, d.nms2, d.Xr, d.nmr
    n, offs = d.n, d.offs
    p1, p2 = size(X1, 2), size(X2, 2)
    pβ = p1 + p2
    pφ0 = offs[end] - pβ                 # sigma1, sigma2, rho12 coefficients
    rng(k) = (offs[k]+1):offs[k+1]
    gidx, G = _group_index(getproperty(data, grp); col = grp)
    G >= 2 || throw(ArgumentError("drm: grouping factor `$grp` needs at least 2 levels"))
    rows_of = [findall(==(k), gidx) for k in 1:G]
    Y = hcat(Vector{Float64}(y1), Vector{Float64}(y2))
    Xi = [begin
              M = zeros(2, pβ)
              M[1, 1:p1] .= X1[i, :]
              M[2, (p1+1):pβ] .= X2[i, :]
              M
          end for i in 1:n]
    const_2pi = n * log(2π)

    # Group-summed GLS pieces at variance parameters `φ` = (sigma1, sigma2, rho12 coefs, lc[3]).
    function pieces(φ)
        T = eltype(φ)
        ls1 = Xs1 * φ[1:size(Xs1, 2)]
        ls2 = Xs2 * φ[(size(Xs1, 2)+1):(size(Xs1, 2)+size(Xs2, 2))]
        ηr = Xr * φ[(size(Xs1, 2)+size(Xs2, 2)+1):pφ0]
        L = Matrix(lc_to_chol(φ[(pφ0+1):(pφ0+3)], 2).L)
        logdetS = zero(T); yVy = zero(T)
        c = zeros(T, pβ); H = zeros(T, pβ, pβ)
        for rows in rows_of
            A = zeros(T, 2, 2); sy = zeros(T, 2); B = zeros(T, 2, pβ)
            for i in rows
                ρ = RHO_GUARD * tanh(ηr[i]); om = 1 - ρ * ρ
                e1 = exp(-ls1[i]); e2 = exp(-ls2[i])
                Si = [e1*e1 -ρ*e1*e2; -ρ*e1*e2 e2*e2] ./ om      # S_i^-1
                logdetS += 2 * ls1[i] + 2 * ls2[i] + log(om)
                yi = @view Y[i, :]
                Sy = Si * yi
                yVy += dot(yi, Sy)
                A += Si; sy += Sy; B += Si * Xi[i]
                c += Xi[i]' * Sy
                H += Xi[i]' * Si * Xi[i]
            end
            M2 = Matrix(I, 2, 2) + L' * A * L
            # A wild line-search trial (log scale near -370) overflows exp(-ls) and makes M2
            # non-finite. Report that trial as non-finite so the optimiser backs off, rather than
            # letting `inv` throw and abort the fit (n of about 1200 and up).
            all(isfinite, M2) || return (T(NaN), T(NaN), fill(T(NaN), pβ), Symmetric(fill(T(NaN), pβ, pβ)), L)
            # A trial with |rho| at its guard and a tiny sigma makes `A` huge and nearly rank one, so
            # M2 = I + L' A L is finite but numerically indefinite or singular (the `I` is lost to
            # rounding). `logdet` then throws a `DomainError` (negative determinant) or `inv` throws
            # a `SingularException`, and the fit aborts. M2 is positive definite in exact arithmetic,
            # so reject such a trial instead. The trial is reported as non-finite here (a NaN tuple) and
            # the callers `nll_full` and `nll_reml` turn it into the one rejection value, a large finite
            # constant 1e18 (flat, zero gradient), as the sibling routes do. Not NaN or Inf: the
            # Hager-Zhang line search asserts a finite value and gradient while it brackets.
            local W
            try
                ldM, sgM = logabsdet(Symmetric(M2))
                sgM < 0 && return (T(NaN), T(NaN), fill(T(NaN), pβ), Symmetric(fill(T(NaN), pβ, pβ)), L)
                W = inv(Symmetric(M2))
                logdetS += ldM + log(sgM)                          # = logdet(Symmetric(M2))
            catch err
                err isa SingularException || rethrow()
                return (T(NaN), T(NaN), fill(T(NaN), pβ), Symmetric(fill(T(NaN), pβ, pβ)), L)
            end
            Ls = L' * sy; LB = L' * B
            yVy -= dot(Ls, W * Ls)
            c -= LB' * (W * Ls)
            H -= LB' * W * LB
        end
        return logdetS, yVy, c, Symmetric(H), L
    end
    ml_from(logdetS, yVy, c, H, β) = const_2pi + 0.5 * (logdetS + yVy - 2 * dot(β, c) + dot(β, H * β))
    function nll_full(θ)
        pf = pieces(θ[(pβ+1):end])
        isfinite(pf[1]) || return convert(eltype(θ), 1e18)   # rejected trial: the one rejection value
        # The weighted residual sum of squares can not be negative. At a trial with a huge, nearly
        # singular row precision the cancellation in `yVy - 2 b'c + b'H b` loses every digit and can
        # return a large negative number, which would look like a far better fit. Reject it (flat 1e18).
        pf[2] - 2 * dot(θ[1:pβ], pf[3]) + dot(θ[1:pβ], pf[4] * θ[1:pβ]) < 0 && return convert(eltype(θ), 1e18)
        v = ml_from(pf[1], pf[2], pf[3], pf[4], θ[1:pβ])
        # Every rejected trial returns the same finite 1e18. Mixing 1e18 with `Inf` made the Hager-Zhang
        # line search throw `AssertionError: isfinite(phi_c) ...` (seed 7080, ML: trial 2 was 1e18, trial 3 `Inf`).
        return isfinite(v) ? v : oftype(v, 1e18)     # non-finite trial: reject, do not throw
    end

    β1 = X1 \ Vector{Float64}(y1); β2 = X2 \ Vector{Float64}(y2)
    φ0 = zeros(pφ0 + 3)
    φ0[offs[3]-pβ+1] = _seed_ls(y1 - X1 * β1, y1)
    φ0[offs[4]-pβ+1] = _seed_ls(y2 - X2 * β2, y2)
    φ0[(pφ0+1):(pφ0+3)] .= cov_to_lc(Matrix(Symmetric([0.25 0.02; 0.02 0.25])))
    function nll_reml(φ)
        # (local names differ from the enclosing scope's: a shared name would be captured, not local)
        pr = pieces(φ)
        isfinite(pr[1]) || return oftype(pr[1], 1e18)   # non-finite trial: reject, do not throw
        chH = cholesky(pr[4]; check = false)
        issuccess(chH) || return oftype(pr[1], 1e18)   # X'V^-1 X numerically indefinite: reject the step
        bh = chH \ pr[3]
        pr[2] - 2 * dot(bh, pr[3]) + dot(bh, pr[4] * bh) < 0 && return oftype(pr[1], 1e18)   # same: negative sum of squares
        return ml_from(pr[1], pr[2], pr[3], pr[4], bh) + logdet(chH) / 2 - 0.5 * pβ * log(2π)
    end
    θ0 = vcat(β1, β2, φ0)

    if method === :REML
        res = _biv_re_best_start(nll_reml, φ0) do x
            Optim.optimize(nll_reml, x, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
        end
        φ̂ = Optim.minimizer(res)
        pc = pieces(φ̂); β̂ = pc[4] \ pc[3]
        Vφ = _vcov_from_hessian(ForwardDiff.hessian(nll_reml, φ̂);
                                context = "bivariate Gaussian ordinary random-effect REML (variance parameters)")
        Gm = ForwardDiff.jacobian(φ -> (q = pieces(φ); q[4] \ q[3]), φ̂)
        Vβ = _vcov_from_gram(Matrix(pc[4]); context = "bivariate Gaussian ordinary random-effect REML (mean GLS)") .+
             Gm * Vφ * transpose(Gm)
        θ̂ = vcat(β̂, φ̂)
        npar = length(θ̂)
        V = zeros(npar, npar)
        V[1:pβ, 1:pβ] .= Vβ
        V[1:pβ, (pβ+1):npar] .= Gm * Vφ
        V[(pβ+1):npar, 1:pβ] .= transpose(Gm * Vφ)
        V[(pβ+1):npar, (pβ+1):npar] .= Vφ
        ll = -nll_reml(φ̂); ml_ll = -nll_full(θ̂)
    else
        res = _biv_re_best_start(nll_full, θ0) do x
            Optim.optimize(nll_full, x, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
        end
        θ̂ = Optim.minimizer(res)
        V = _vcov_from_hessian(ForwardDiff.hessian(nll_full, θ̂);
                               context = "bivariate Gaussian ordinary random-effect")
        ll = -nll_full(θ̂); ml_ll = ll
    end

    θb = θ̂[1:pβ]; φ̂ = θ̂[(pβ+1):end]
    logdetS, yVy, c, H, L = pieces(φ̂)
    # BLUPs u_g = L W L' sum_i S_i^-1 (y_i - X_i b), 2 x G.
    ls1 = Xs1 * φ̂[1:size(Xs1, 2)]; ls2 = Xs2 * φ̂[(size(Xs1, 2)+1):(size(Xs1, 2)+size(Xs2, 2))]
    ρv = RHO_GUARD .* tanh.(Xr * φ̂[(size(Xs1, 2)+size(Xs2, 2)+1):pφ0])
    U = zeros(2, G)
    for (k, rows) in enumerate(rows_of)
        A = zeros(2, 2); sy = zeros(2)
        for i in rows
            e1 = exp(-ls1[i]); e2 = exp(-ls2[i]); om = 1 - ρv[i]^2
            Si = [e1*e1 -ρv[i]*e1*e2; -ρv[i]*e1*e2 e2*e2] ./ om
            A += Si; sy += Si * (Y[i, :] - Xi[i] * θb)
        end
        U[:, k] = L * (inv(Symmetric(I + L' * A * L)) * (L' * sy))
    end

    blocks = [:mu1 => rng(1), :mu2 => rng(2), :sigma1 => rng(3), :sigma2 => rng(4),
              :rho12 => rng(5), _BIV_RE_GROUP_BLOCK => (offs[end]+1):(offs[end]+3)]
    names = [:mu1 => nm1, :mu2 => nm2, :sigma1 => nms1, :sigma2 => nms2, :rho12 => nmr,
             _BIV_RE_GROUP_BLOCK => _biv_re_group_names()]
    means = Dict(:mu1 => X1 * θb[1:p1], :mu2 => X2 * θb[(p1+1):pβ])
    obs = Dict(:mu1 => Vector{Float64}(y1), :mu2 => Vector{Float64}(y2))
    scales = Dict(:sigma1 => exp.(ls1), :sigma2 => exp.(ls2), :rho12 => ρv)
    # Reorder to the public layout [b_mu1, b_mu2, sigma1, sigma2, rho12, lc] (already so).
    fit = DrmFit(fam, blocks, names, θ̂, V, ll, n, Optim.converged(res), means, obs, scales)
    fit = method === :REML ? _withreml(fit, ll, ml_ll) : fit
    re = (; effects = Dict(Symbol(grp) => U), Sigma_g = L * L', axes = (:mu1, :mu2),
          kind = :ordinary, tag = tag, group = grp, group_index = gidx)
    return _withiterations(_withranef(_withformula(_withnll(fit, nll_full), f), re), Optim.iterations(res))
end

# Fresh-draw simulator for `simulate(fit; data)` and the parametric bootstrap: per group a new
# 2-vector u_g ~ N(0, Sigma_g) is added to mu1 and mu2, then the bivariate residual pair
# (sigma1, sigma2, rho12 at each row). The draw has one row per fitted row, so `data` must be the
# table the fit used with no row dropped (rows lost to missing values would make the draw
# disagree with `data`); anything else is refused, never silently drawn without the group effect.
_is_biv_ordinary_re_fit(fit) =
    fit.formula isa BivariateDrmFormula && fit.ranef isa NamedTuple &&
    get(fit.ranef, :kind, nothing) === :ordinary && haskey(fit.ranef, :Sigma_g)

function _biv_ordinary_re_simulator(fit::DrmFit, data)
    re = fit.ranef
    grp = Symbol(re.group)
    (data !== nothing && hasproperty(data, grp)) || throw(ArgumentError(
        "simulate/bootstrap: this bivariate ordinary random-effect fit needs the fitting `data` " *
        "(with the grouping column `$grp`) to redraw the group effects."))
    col = getproperty(data, grp)
    mu1, mu2 = fit.means[:mu1], fit.means[:mu2]
    n = length(mu1)
    length(col) == n || throw(ArgumentError(
        "simulate/bootstrap: `data` has $(length(col)) rows but the fit used $n (rows with missing " *
        "values are dropped at fit time). Pass the complete table the model was fitted to, or drop " *
        "the incomplete rows first, so the group effects can be redrawn row for row."))
    gidx, G = _group_index(col; col = grp)
    gidx == re.group_index || throw(ArgumentError(
        "simulate/bootstrap: the grouping column `$grp` of `data` does not match the fitted groups."))
    σ1, σ2, ρ = fit.scales[:sigma1], fit.scales[:sigma2], fit.scales[:rho12]
    F = eigen(Symmetric(Matrix{Float64}(re.Sigma_g)))
    Lg = F.vectors * Diagonal(sqrt.(max.(F.values, 0.0)))     # Lg Lg' = Sigma_g, also on the boundary
    return function (rng)
        u = Lg * randn(rng, 2, G)
        z1 = randn(rng, n); z2 = randn(rng, n)
        return Dict(:mu1 => mu1 .+ u[1, gidx] .+ σ1 .* z1,
                    :mu2 => mu2 .+ u[2, gidx] .+ σ2 .* (ρ .* z1 .+ sqrt.(1 .- ρ .^ 2) .* z2))
    end
end
