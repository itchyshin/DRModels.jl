# gaussian_bivariate.jl — bivariate Gaussian location–scale with a
# predictor-dependent residual correlation ρ12 (fixed effects, ML). Mirrors
# drmTMB's bivariate Gaussian:
#
#   bf(mu1 = y1 ~ x, mu2 = y2 ~ x, sigma1 = …, sigma2 = …, rho12 = …)
#
# σ1, σ2 use a log link; ρ12 uses a tanh link (so its coefficients act on
# atanh ρ12, keeping ρ12 ∈ (-1, 1)).

"""
    BivariateDrmFormula

Two-response formula bundle (μ1, μ2, σ1, σ2, ρ12), built by the keyword form of
[`bf`](@ref).
"""
struct BivariateDrmFormula
    response1::Symbol
    response2::Symbol
    forms::Vector{Pair{Symbol,Any}}
end

_rhs_or_intercept(f) = f === nothing ? ConstantTerm(1) : f.rhs

# Validate a bivariate placeholder formula's LHS. The σ1/σ2/ρ12 formulas carry a
# *placeholder* left-hand side, so — mirroring the univariate rejection
# discipline (and drmTMB) — that placeholder must be the parameter's own name.
# `nothing` (the parameter omitted ⇒ `~ 1`) is always allowed.
function _check_bivariate_lhs(f, expected::Symbol)
    f === nothing && return f
    f isa FormulaTerm || throw(ArgumentError("bf: `$expected` must be a formula " *
        "`$expected ~ …` (got `$(repr(f))`)."))
    lhs = f.lhs
    lhs isa Term || throw(ArgumentError("bf: the `$expected` formula must read " *
        "`$expected ~ …` with the parameter name on the left (got `$lhs`)."))
    name = lhs.sym
    if name === :tau && (expected === :sigma1 || expected === :sigma2)
        throw(ArgumentError("bf: the scale parameter is named `sigma1`/`sigma2`, never " *
            "`tau` — write `$expected ~ …`."))
    elseif name !== expected
        throw(ArgumentError("bf: the `$expected` formula must read `$expected ~ …` " *
            "(got `$name ~ …`); in the bivariate keyword form each placeholder LHS " *
            "must be its own parameter name."))
    end
    return f
end

# The two mean formulas each name a single response column on the left. Reject a
# two-column `cbind(…)` (univariate-only) with a clear message rather than a
# cryptic `getproperty` error on a FunctionTerm.
function _bivariate_response_sym(f::FormulaTerm, kw::Symbol)
    f.lhs isa Term || throw(ArgumentError("bf: `$kw = response ~ …` needs a single " *
        "response column on the left (got `$(f.lhs)`); the bivariate form takes one " *
        "response per mean, not `cbind(…)`."))
    return f.lhs.sym
end

"""
    bf(; mu1, mu2, sigma1=…, sigma2=…, nu=…, rho12=…)

Bivariate formula bundle, mirroring drmTMB. `mu1 = y1 ~ …` and
`mu2 = y2 ~ …` set the two responses and their mean predictors; `sigma1`,
`sigma2` (log σ) and `rho12` (atanh ρ) default to `~ 1`. For one-sided
predictors give the parameter name as a placeholder LHS, e.g.
`sigma1 = @formula(sigma1 ~ x)`.

`nu` is the shared degrees-of-freedom formula for the bivariate Student-t
family (`drm(…, Student())`, drmTMB's `biv_student()`), on the `logm2` scale
`ν = 2 + exp(η)`. It is **omitted entirely** unless supplied, so the Gaussian
and lognormal bundles are unchanged. Under the exact bivariate-t density a
single scalar mixing variable governs both margins, so `ν` is *structurally*
shared — there is no per-margin `nu1`/`nu2`.

Like the univariate form, `bf` rejects reserved / mis-typed syntax: a placeholder
LHS that is not its own parameter name (e.g. `sigma1 = @formula(tau ~ x)` or a
swapped `sigma1`/`sigma2`), and a two-column `cbind(…)` response on `mu1`/`mu2`.
"""
function bf(; mu1::FormulaTerm, mu2::FormulaTerm, sigma1 = nothing, sigma2 = nothing,
            nu = nothing, rho12 = nothing)
    _check_bivariate_lhs(sigma1, :sigma1)
    _check_bivariate_lhs(sigma2, :sigma2)
    _check_bivariate_lhs(nu, :nu)
    _check_bivariate_lhs(rho12, :rho12)
    forms = Pair{Symbol,Any}[
        :mu1 => mu1.rhs,
        :mu2 => mu2.rhs,
        :sigma1 => _rhs_or_intercept(sigma1),
        :sigma2 => _rhs_or_intercept(sigma2),
    ]
    # `nu` is Student-only: keep it out of the bundle unless asked for, so the
    # Gaussian/lognormal `forms` (and everything that reads them, e.g.
    # `_bivariate_q4_marker`) are byte-identical to before.
    nu === nothing || push!(forms, :nu => _rhs_or_intercept(nu))
    push!(forms, :rho12 => _rhs_or_intercept(rho12))
    return BivariateDrmFormula(_bivariate_response_sym(mu1, :mu1),
                               _bivariate_response_sym(mu2, :mu2), forms)
end

"""
    drm(formula::BivariateDrmFormula, Gaussian(); data, tree = nothing,
        K = nothing, A = nothing, coords = nothing, g_tol = 1e-8,
        q4_g_tol = 1e-3, q4_iterations = 300, q4_n_newton = 40,
        q4_vcov = true, method = :ML) -> DrmFit

Fit a bivariate Gaussian distributional regression model.

With no structured-effect marker, this is the residual-correlation model:
`mu1`, `mu2`, `sigma1`, `sigma2`, and residual `rho12` each have their own fixed
effect formula.

With matching structured markers on `mu1` and `mu2` only, this routes to the
q=2 exact-Gaussian point-fit cell, `method = :ML` (default) or `method =
:REML` (Patterson–Thompson; marginalises `beta_mu1`/`beta_mu2` only — see
`reml_q2.jl`). The q=2 route currently accepts `phylo(1 | group)` with
`tree`, `relmat(1 | group)` with `K`, `animal(1 | group)` with `A`, or
`spatial(1 | group)` with `coords`; it requires complete responses, identical
mean fixed-effect designs, and intercept-only `sigma1`, `sigma2`, and `rho12`.
`spatial(1 | group)` supplies a covariance exactly as `relmat`/`animal` do: the
exponential kernel `exp(-d / rho)` at a FIXED range `rho` (keyword
`spatial_range`, else the mean off-diagonal pairwise distance), the same rule
the q=4 structured route uses. It does NOT estimate the range jointly the way
the univariate spatial route (`_fit_spatial_gaussian`) does, so the two are not
the same model and must not be compared as one. The range in force is recorded
as `fit.ranef.spatial_range`.

Row `k` of `coords` is the `k`-th level of `group` **in the order the levels
first appear in `data`**, not sorted label order, which is the same contract
`K` and `A` carry on this route. Only the row COUNT is checked, so coordinates
supplied in the wrong order fit a different spatial structure without error.

With the same `phylo(1 | group)` marker on all four location/scale predictors
(`mu1`, `mu2`, `sigma1`, and `sigma2`) and a supplied `tree`, this routes to the
verified q=4 phylogenetic location-scale engine. The residual `rho12` formula
remains the residual correlation; the group-level 4×4 covariance `Σ_a` is stored
as `fit.ranef.Sigma_a`, with axes `(:mu1, :mu2, :sigma1, :sigma2)`. Population
parameter prediction skips the internal `:phylocov` coefficient block.

```julia
fit = drm(
    bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
       mu2 = @formula(y2 ~ x + phylo(1 | species)),
       sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
       sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
       rho12 = @formula(rho12 ~ 1)),
    Gaussian();
    data = dat,
    tree = phy,
)
```
"""
function drm(f::BivariateDrmFormula, fam::Gaussian; data, tree = nothing,
             K = nothing, A = nothing, coords = nothing,
             spatial_range = nothing,
             g_tol::Real = 1e-8, q4_g_tol::Real = 1e-3,
             q4_iterations::Int = 300, q4_n_newton::Int = 40,
             q4_vcov::Bool = true, method::Symbol = :ML, V = nothing)
    method in (:ML, :REML) ||
        throw(ArgumentError("drm: `method` must be :ML (default) or :REML (got :$method)"))
    rhs = Dict(f.forms)
    fixed, structured_marker = _bivariate_q4_marker(rhs)
    # A8: known bivariate sampling covariance (drmTMB's meta_vcov_bivariate) is
    # consumed by the residual route only in this slice.
    V !== nothing && structured_marker !== nothing &&
        throw(ArgumentError("drm: known sampling covariance `V` is implemented for the " *
            "residual bivariate route only; combining it with structured " *
            "phylo/relmat/animal/spatial markers is a later slice."))
    # #470: `V` is accepted only on the residual-only route (checked just above),
    # and that route computes each row's Gaussian density directly from
    # V_i + Sigma_het with no random effect marginalised — so there is no
    # Schur-complement axis for this package's (single) REML formulation to
    # correct, the same reason method = :REML is rejected for the residual-only
    # model without `V` (below). A meta-analytic REML that profiles the fixed
    # effects via GLS under V_i + Sigma_het (as e.g. metafor's `rma`) would be a
    # legitimate but DIFFERENT derivation on a route this task does not touch;
    # this is a scope boundary, not a temporary gap, hence the permanent wording.
    V !== nothing && method === :REML &&
        throw(ArgumentError("drm: `V` (known sampling covariance) has no REML target in this " *
            "package: it is accepted only on the bivariate residual-correlation route, which " *
            "marginalises no random effect and so has nothing for a restricted likelihood to " *
            "correct — use method = :ML."))
    if structured_marker !== nothing && structured_marker[1] === :phylo_q4
        return _fit_bivariate_q4_phylo(
            f, fam, data, fixed, structured_marker, tree;
            q4_g_tol = q4_g_tol,
            q4_iterations = q4_iterations,
            q4_n_newton = q4_n_newton,
            q4_vcov = q4_vcov,
            method = method,
        )
    elseif structured_marker !== nothing && structured_marker[1] === :structured_q4
        return _fit_bivariate_q4_structured(
            f, fam, data, fixed, structured_marker, tree, K, A, coords;
            spatial_range = spatial_range,
            q4_g_tol = q4_g_tol,
            q4_iterations = q4_iterations,
            q4_n_newton = q4_n_newton,
            q4_vcov = q4_vcov,
            method = method,
        )
    elseif structured_marker !== nothing && structured_marker[1] === :structured_q2
        # No route-wide `coords` guard here: each provider branch below refuses
        # when ITS OWN input is missing (relmat needs `K`, animal needs `A`,
        # spatial needs `coords`), so the fail-closed rule is stated once, next
        # to the provider it constrains. The guard this replaced fired on the
        # presence of `coords` alone and so refused a perfectly ordinary
        # relmat/animal q=2 fit that merely carried a stray `coords =` kwarg.
        return _fit_bivariate_q2_structured(
            f, fam, data, fixed, structured_marker, tree, K, A, coords;
            spatial_range = spatial_range,
            g_tol = Float64(g_tol),
            method = method,
        )
    end
    # The residual-only bivariate route carries no random effects, but it DOES
    # have a restricted likelihood (#624 / drmTMB #1142): REML here integrates the
    # MEAN fixed effects beta_mu1 and beta_mu2 out of the Gaussian likelihood
    # under the row-wise 2x2 residual covariance — the classical
    # Patterson–Thompson/SUR case, and exactly the set native drmTMB hands to
    # TMB's Laplace approximation for this cell. `V !== nothing` is refused above
    # and stays refused: that route marginalises nothing and keeps its permanent
    # scope boundary.
    method === :REML && return _fit_bivariate_residual_reml(f, fam, data, rhs, g_tol)
    return _fit_bivariate_residual(f, fam, data, rhs, g_tol; V = V)
end

# Finite, data-scaled log-σ seed for a residual-σ intercept. `resid` is the OLS
# residual vector and `yobs` the observed response. The seed variance is floored
# at max(1e-6·var(yobs), 1e-8) so a saturated (zero-residual) fit does not seed
# log(eps()) ≈ −36; for a (near-)constant response (var ≈ 0) the 1e-8 backstop
# keeps the seed finite. Returns log(sqrt(seed_var)).
function _seed_ls(resid::AbstractVector, yobs::AbstractVector)
    v_resid = length(resid) == 0 ? 0.0 : mean(resid .^ 2)
    v_y = length(yobs) > 1 ? Statistics.var(yobs) : 0.0
    v_floor = max(1e-6 * v_y, 1e-8)
    return 0.5 * log(max(v_resid, v_floor))
end

# Designs, observation masks and the theta block layout shared by the
# residual-only bivariate Gaussian route's ML fitter and its REML fitter
# (#624 / drmTMB #1142). Moved out of `_fit_bivariate_residual` unchanged so the
# two estimators build ONE design and ONE parameter layout, not two that drift.
function _biv_residual_designs(f::BivariateDrmFormula, rhs, data)
    y1, X1, nm1 = _design(f.response1, rhs[:mu1], data)
    y2, X2, nm2 = _design(f.response2, rhs[:mu2], data)
    _, Xs1, nms1 = _design(f.response1, rhs[:sigma1], data)   # reuse a real LHS;
    _, Xs2, nms2 = _design(f.response1, rhs[:sigma2], data)   # only the matrix is kept
    _, Xr, nmr = _design(f.response1, rhs[:rho12], data)

    n = length(y1)
    obs1 = _observed_response_mask(y1)
    obs2 = _observed_response_mask(y2)
    n_like = count(obs1 .| obs2)
    n_like > 0 ||
        throw(ArgumentError("drm: at least one bivariate Gaussian response cell must be observed"))
    count(obs1) >= size(X1, 2) ||
        throw(ArgumentError("drm: observed `$(f.response1)` values are fewer than the mu1 coefficients"))
    count(obs2) >= size(X2, 2) ||
        throw(ArgumentError("drm: observed `$(f.response2)` values are fewer than the mu2 coefficients"))
    count(obs1 .& obs2) > 0 ||
        throw(ArgumentError("drm: at least one row must observe both bivariate Gaussian responses to estimate rho12"))

    ps = (size(X1, 2), size(X2, 2), size(Xs1, 2), size(Xs2, 2), size(Xr, 2))
    offs = cumsum([0, ps...])
    return (; y1, X1, nm1, y2, X2, nm2, Xs1, nms1, Xs2, nms2, Xr, nmr,
            n, obs1, obs2, n_like, ps, offs)
end

function _fit_bivariate_residual(f::BivariateDrmFormula, fam::Gaussian, data, rhs, g_tol::Real; V = nothing)
    d = _biv_residual_designs(f, rhs, data)
    y1, X1, nm1 = d.y1, d.X1, d.nm1
    y2, X2, nm2 = d.y2, d.X2, d.nm2
    Xs1, nms1 = d.Xs1, d.nms1
    Xs2, nms2 = d.Xs2, d.nms2
    Xr, nmr = d.Xr, d.nmr
    n, obs1, obs2, n_like = d.n, d.obs1, d.obs2, d.n_like
    offs = d.offs
    rng(k) = (offs[k]+1):offs[k+1]

    # A8: known per-row 2x2 sampling covariance (bivariate meta-analysis).
    # `nothing` keeps the original per-row expressions byte-identical below; a
    # resolved V routes every row through the general 2x2 form
    #   S = V_i + Sigma_het,i,  Sigma_het = [[s1^2, rho*s1*s2], [., s2^2]]
    # with a positive-definiteness sentinel where the guarded rho cannot save us
    # (V's own correlation plus the heterogeneity correlation can exceed 1).
    meta_v = _resolve_biv_meta_v(V, n)

    function nll(θ)
        b1 = θ[rng(1)]; b2 = θ[rng(2)]; bs1 = θ[rng(3)]; bs2 = θ[rng(4)]; br = θ[rng(5)]
        η1 = X1 * b1; η2 = X2 * b2; ls1 = Xs1 * bs1; ls2 = Xs2 * bs2; ηr = Xr * br
        s = zero(eltype(θ))
        if meta_v === nothing
            @inbounds for i in 1:n
                if obs1[i] && obs2[i]
                    ρ = RHO_GUARD * tanh(ηr[i])    # guard ρ off ±1 (tanh saturates to ±1.0 in
                    om = 1 - ρ * ρ                 # Float64 for large η → om=0 → NaN); matches the
                                                   # q4 engine's RHO_GUARD + drmTMB's guarded link
                    z1 = (y1[i] - η1[i]) * exp(-ls1[i])     # standardised residuals
                    z2 = (y2[i] - η2[i]) * exp(-ls2[i])
                    # −log φ₂ = log(2π) + (½ log|Σ|) + (½ rᵀΣ⁻¹r)
                    s += log(2π) + ls1[i] + ls2[i] + 0.5 * log(om) +
                         0.5 * (z1 * z1 - 2ρ * z1 * z2 + z2 * z2) / om
                elseif obs1[i]
                    z1 = (y1[i] - η1[i]) * exp(-ls1[i])
                    s += 0.5 * log(2π) + ls1[i] + 0.5 * z1 * z1
                elseif obs2[i]
                    z2 = (y2[i] - η2[i]) * exp(-ls2[i])
                    s += 0.5 * log(2π) + ls2[i] + 0.5 * z2 * z2
                end
            end
        else
            @inbounds for i in 1:n
                if obs1[i] && obs2[i]
                    ρ = RHO_GUARD * tanh(ηr[i])
                    σ1 = exp(ls1[i]); σ2 = exp(ls2[i])
                    S11 = meta_v.v1[i] + σ1 * σ1
                    S22 = meta_v.v2[i] + σ2 * σ2
                    S12 = meta_v.cov12[i] + ρ * σ1 * σ2
                    det = S11 * S22 - S12 * S12
                    # V's correlation and rho can conspire past PD; the RHO_GUARD
                    # alone cannot prevent it here. Sentinel, matching the coupled
                    # phylo block's house style.
                    det > 0 || return oftype(s, 1e18)
                    r1 = y1[i] - η1[i]; r2 = y2[i] - η2[i]
                    s += log(2π) + 0.5 * log(det) +
                         0.5 * (S22 * r1 * r1 - 2 * S12 * r1 * r2 + S11 * r2 * r2) / det
                elseif obs1[i]
                    var1 = meta_v.v1[i] + exp(2 * ls1[i])
                    r1 = y1[i] - η1[i]
                    s += 0.5 * (log(2π) + log(var1) + r1 * r1 / var1)
                elseif obs2[i]
                    var2 = meta_v.v2[i] + exp(2 * ls2[i])
                    r2 = y2[i] - η2[i]
                    s += 0.5 * (log(2π) + log(var2) + r2 * r2 / var2)
                end
            end
        end
        return s
    end

    θ0 = zeros(offs[end])
    X1_obs = Matrix{Float64}(X1[obs1, :])
    X2_obs = Matrix{Float64}(X2[obs2, :])
    y1_obs = Vector{Float64}(y1[obs1])
    y2_obs = Vector{Float64}(y2[obs2])
    β1 = X1_obs \ y1_obs
    β2 = X2_obs \ y2_obs
    θ0[rng(1)] .= β1
    θ0[rng(2)] .= β2
    # Residual-σ seeds. When the mean design is saturated (observed rows == mu
    # coefficients) the residuals are exactly 0, so `log(sqrt(mean(r²)) + eps())`
    # collapses to log(eps()) ≈ −36 — an extreme start where exp(−ls) overflows on
    # the first nll evaluation. Floor the seed variance at a small fraction of the
    # response variance so the start stays finite and on the data scale. `_seed_ls`
    # falls back to a scale-1 floor when the response is (near-)constant.
    θ0[offs[3]+1] = _seed_ls(y1_obs - X1_obs * β1, y1_obs)     # σ1 intercept (log scale)
    θ0[offs[4]+1] = _seed_ls(y2_obs - X2_obs * β2, y2_obs)     # σ2 intercept (log scale)
    # rho12 block (rng(5)) intentionally left at 0 (atanh scale ⇒ ρ = 0).
    θ0[rng(5)] .= 0.0

    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂); context = "bivariate Gaussian")

    blocks = [:mu1 => rng(1), :mu2 => rng(2), :sigma1 => rng(3), :sigma2 => rng(4), :rho12 => rng(5)]
    names = [:mu1 => nm1, :mu2 => nm2, :sigma1 => nms1, :sigma2 => nms2, :rho12 => nmr]
    means = Dict(:mu1 => X1 * θ̂[rng(1)], :mu2 => X2 * θ̂[rng(2)])
    obs = Dict(:mu1 => Vector{Float64}(y1), :mu2 => Vector{Float64}(y2))
    scales = Dict(:sigma1 => exp.(Xs1 * θ̂[rng(3)]),
                  :sigma2 => exp.(Xs2 * θ̂[rng(4)]),
                  :rho12 => RHO_GUARD .* tanh.(Xr * θ̂[rng(5)]))   # report the model's guarded ρ
    return _withiterations(
        _withformula(_withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n_like, Optim.converged(res), means, obs, scales), nll), f),
        Optim.iterations(res))
end

# ---------------------------------------------------------------------------
# REML for the residual-only bivariate Gaussian route (#624 / drmTMB #1142).
#
# THE MODEL is the one `_fit_bivariate_residual` fits by ML: row i is bivariate
# Gaussian with mean (x1_i'b_mu1, x2_i'b_mu2) and covariance
#
#     S_i = [ s1_i^2            rho_i*s1_i*s2_i
#             rho_i*s1_i*s2_i   s2_i^2          ],
#
# log s1_i = xs1_i'b_sigma1, log s2_i = xs2_i'b_sigma2,
# rho_i = RHO_GUARD*tanh(xr_i'b_rho) — same guarded rho link, same
# partially-observed row handling (a row observing one response contributes its
# univariate factor).
#
# WHICH AXES ARE MARGINALISED. `reml_q2.jl`'s rule — an axis is profiled out of
# the restricted likelihood iff there is something to integrate it against —
# reads here as: b_mu1 and b_mu2 enter the leaf LINEARLY through a Gaussian
# likelihood, so the objective is EXACTLY quadratic in them and they integrate
# in closed form; b_sigma1, b_sigma2 and b_rho enter nonlinearly (through
# exp(.) and tanh(.)) and stay OUTER, exactly the role b_rho plays in
# `reml_q4.jl` and b_sigma1/b_sigma2/b_rho play in `reml_q2.jl`.
#
# That marginalised set {b_mu1, b_mu2} is the SAME set native drmTMB hands to
# TMB for this cell: `drm_apply_estimator_spec()` (drmTMB R/drmTMB.R) sets
# `tmb_random_names = c("beta_mu1", "beta_mu2")` when the bivariate model has no
# sigma-side random effect, and TMB's Laplace approximation of a quadratic
# integrand is EXACT — so the two engines evaluate the same integral rather than
# two approximations of it.
#
# THE OBJECTIVE AND ITS CONSTANT. With Z_i the block-diagonal row design
# [x1_i' 0; 0 x2_i'], H(phi) = sum_i Z_i' S_i^-1 Z_i the joint fixed-effect
# Hessian, and bhat(phi) = H^-1 sum_i Z_i' S_i^-1 y_i the GLS profile,
#
#     l_R(phi) = l_ML(bhat(phi), phi) - 0.5*logdet H(phi) + (p_beta/2)*log(2*pi).
#
# The +(p_beta/2)*log(2*pi) term is the NORMALISED Patterson–Thompson convention
# this package standardised on in #477, and it is the SAME constant TMB's
# Laplace approximation carries, since
# -log(int exp(-f(b)) db) ~= f(bhat) + 0.5*logdet(H/(2*pi)). Keeping it means
# `loglik()` for this fit is directly comparable to drmTMB's `logLik()` on the
# same cell: there is no leftover constant to subtract before comparing.
#
# vcov. bhat depends on phi, so the mean block carries phi's uncertainty:
#
#     Var(bhat) = H^-1 + G*Var(phihat)*G',  G = d bhat / d phi,
#     Cov(bhat, phihat) = G*Var(phihat),
#
# which is the (beta, beta) block of the inverse joint precision — the quantity
# `TMB::sdreport()` reports for an ADREPORTed function of the marginalised
# block, and the one drmTMB reads under REML (`vcov.drmTMB` -> `sdr$cov`). On
# the ledger's cell G is exactly zero (mu1 and mu2 share one design, so the
# SUR/GLS profile is per-equation OLS and does not move with phi) and the
# formula collapses to H^-1; it is written in full so that a cell with different
# mu1/mu2 designs stays correct rather than silently under-reporting.
function _fit_bivariate_residual_reml(f::BivariateDrmFormula, fam::Gaussian, data, rhs, g_tol::Real)
    d = _biv_residual_designs(f, rhs, data)
    y1, X1, nm1 = d.y1, d.X1, d.nm1
    y2, X2, nm2 = d.y2, d.X2, d.nm2
    Xs1, nms1 = d.Xs1, d.nms1
    Xs2, nms2 = d.Xs2, d.nms2
    Xr, nmr = d.Xr, d.nmr
    n, obs1, obs2, n_like = d.n, d.obs1, d.obs2, d.n_like
    offs = d.offs
    rng(k) = (offs[k]+1):offs[k+1]

    p1, p2 = size(X1, 2), size(X2, 2)
    pβ = p1 + p2
    ps1, ps2, pr = size(Xs1, 2), size(Xs2, 2), size(Xr, 2)
    pφ = ps1 + ps2 + pr

    # An unobserved cell must never enter an arithmetic expression (it can be
    # NaN-coded). Its S^-1 weights are identically 0 below, so substituting 0 for
    # the value is safe and keeps the loops branch-light.
    ỹ1 = [obs1[i] ? Float64(y1[i]) : 0.0 for i in 1:n]
    ỹ2 = [obs2[i] ? Float64(y2[i]) : 0.0 for i in 1:n]
    n_cells = count(obs1) + count(obs2)      # observed RESPONSE CELLS, not rows
    const_2pi = 0.5 * n_cells * log(2π)

    # Per-row S^-1 entries (a, b, c) = (S^-1_11, S^-1_12, S^-1_22) and
    # sum_i log|S_i| over the observed pattern. A row observing one response
    # contributes its scalar precision on that axis and nothing off-diagonal.
    function _weights(φ)
        ls1 = Xs1 * φ[1:ps1]
        ls2 = Xs2 * φ[(ps1+1):(ps1+ps2)]
        ηr = Xr * φ[(ps1+ps2+1):pφ]
        T = promote_type(eltype(ls1), eltype(ls2), eltype(ηr))
        a = zeros(T, n); b = zeros(T, n); c = zeros(T, n)
        logdetS = zero(T)
        @inbounds for i in 1:n
            if obs1[i] && obs2[i]
                ρ = RHO_GUARD * tanh(ηr[i])
                om = 1 - ρ * ρ
                a[i] = exp(-2 * ls1[i]) / om
                b[i] = -ρ * exp(-ls1[i] - ls2[i]) / om
                c[i] = exp(-2 * ls2[i]) / om
                logdetS += 2 * ls1[i] + 2 * ls2[i] + log(om)
            elseif obs1[i]
                a[i] = exp(-2 * ls1[i])
                logdetS += 2 * ls1[i]
            elseif obs2[i]
                c[i] = exp(-2 * ls2[i])
                logdetS += 2 * ls2[i]
            end
        end
        return a, b, c, logdetS
    end

    # H = sum_i Z_i' S_i^-1 Z_i and the GLS right-hand side sum_i Z_i' S_i^-1 y_i.
    function _hessian_rhs(a, b, c)
        T = eltype(a)
        H = zeros(T, pβ, pβ)
        rhsv = zeros(T, pβ)
        @inbounds for i in 1:n
            (obs1[i] || obs2[i]) || continue
            for r in 1:p1
                x1r = X1[i, r]
                for s in 1:p1
                    H[r, s] += a[i] * x1r * X1[i, s]
                end
                for s in 1:p2
                    H[r, p1+s] += b[i] * x1r * X2[i, s]
                end
                rhsv[r] += (a[i] * ỹ1[i] + b[i] * ỹ2[i]) * x1r
            end
            for r in 1:p2
                x2r = X2[i, r]
                for s in 1:p2
                    H[p1+r, p1+s] += c[i] * x2r * X2[i, s]
                end
                rhsv[p1+r] += (b[i] * ỹ1[i] + c[i] * ỹ2[i]) * x2r
            end
        end
        @inbounds for r in 1:p1, s in 1:p2
            H[p1+s, r] = H[r, p1+s]
        end
        return H, rhsv
    end

    # GLS profile of (b_mu1, b_mu2) at fixed phi.
    _profile(φ) = begin
        a, b, c, logdetS = _weights(φ)
        H, rhsv = _hessian_rhs(a, b, c)
        (H \ rhsv, a, b, c, logdetS, H)
    end

    # Plain ML negative log-likelihood, assembled from the same pieces so the
    # restricted objective and the reported `ml_loglik` cannot drift apart.
    # Identical term by term to `_fit_bivariate_residual`'s `nll` on the
    # V === nothing branch: log(2*pi) + ls1 + ls2 + 0.5*log(1-rho^2) + 0.5*q for a
    # doubly observed row, 0.5*log(2*pi) + ls + 0.5*z^2 for a singly observed one.
    function _ml_nll(β, a, b, c, logdetS)
        r1 = ỹ1 .- X1 * β[1:p1]
        r2 = ỹ2 .- X2 * β[(p1+1):pβ]
        q = zero(promote_type(eltype(r1), eltype(r2), eltype(a)))
        @inbounds for i in 1:n
            q += a[i] * r1[i] * r1[i] + 2 * b[i] * r1[i] * r2[i] + c[i] * r2[i] * r2[i]
        end
        return const_2pi + 0.5 * logdetS + 0.5 * q
    end

    # Restricted NEGATIVE log-likelihood over phi alone.
    function nll_reml(φ)
        β, a, b, c, logdetS, H = _profile(φ)
        return _ml_nll(β, a, b, c, logdetS) + 0.5 * logdet(H) - 0.5 * pβ * log(2π)
    end

    # Warm start: the ML fitter's own residual-sigma seeds, rho at 0.
    X1_obs = Matrix{Float64}(X1[obs1, :])
    X2_obs = Matrix{Float64}(X2[obs2, :])
    y1_obs = Vector{Float64}(y1[obs1])
    y2_obs = Vector{Float64}(y2[obs2])
    β1_ols = X1_obs \ y1_obs
    β2_ols = X2_obs \ y2_obs
    φ0 = zeros(pφ)
    φ0[1] = _seed_ls(y1_obs - X1_obs * β1_ols, y1_obs)
    φ0[ps1+1] = _seed_ls(y2_obs - X2_obs * β2_ols, y2_obs)

    res = Optim.optimize(nll_reml, φ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    φ̂ = Optim.minimizer(res)
    β̂, â, b̂, ĉ, logdetŜ, Ĥ = _profile(φ̂)
    θ̂ = vcat(β̂, φ̂)

    Vφ = inv(Symmetric(ForwardDiff.hessian(nll_reml, φ̂)))
    G = ForwardDiff.jacobian(φ -> _profile(φ)[1], φ̂)      # d bhat / d phi
    Vβ = Matrix(inv(Symmetric(Ĥ))) .+ G * Vφ * transpose(G)
    Cβφ = G * Vφ
    V = zeros(pβ + pφ, pβ + pφ)
    V[1:pβ, 1:pβ] .= Vβ
    V[1:pβ, (pβ+1):(pβ+pφ)] .= Cβφ
    V[(pβ+1):(pβ+pφ), 1:pβ] .= transpose(Cβφ)
    V[(pβ+1):(pβ+pφ), (pβ+1):(pβ+pφ)] .= Vφ

    reml_ll = -nll_reml(φ̂)
    ml_ll = -_ml_nll(β̂, â, b̂, ĉ, logdetŜ)

    blocks = [:mu1 => rng(1), :mu2 => rng(2), :sigma1 => rng(3), :sigma2 => rng(4), :rho12 => rng(5)]
    names = [:mu1 => nm1, :mu2 => nm2, :sigma1 => nms1, :sigma2 => nms2, :rho12 => nmr]
    means = Dict(:mu1 => X1 * θ̂[rng(1)], :mu2 => X2 * θ̂[rng(2)])
    obs = Dict(:mu1 => Vector{Float64}(y1), :mu2 => Vector{Float64}(y2))
    scales = Dict(:sigma1 => exp.(Xs1 * θ̂[rng(3)]),
                  :sigma2 => exp.(Xs2 * θ̂[rng(4)]),
                  :rho12 => RHO_GUARD .* tanh.(Xr * θ̂[rng(5)]))

    # Full-theta PLAIN ML objective for profile intervals, in the same theta
    # layout the ML fitter uses (so downstream inference code sees one layout).
    function nll_full(θ)
        a, b, c, logdetS = _weights(θ[(pβ+1):(pβ+pφ)])
        return _ml_nll(θ[1:pβ], a, b, c, logdetS)
    end

    fit = DrmFit(fam, blocks, names, θ̂, V, reml_ll, n_like, Optim.converged(res), means, obs, scales)
    return _withiterations(
        _withformula(_withreml(_withnll(fit, nll_full), reml_ll, ml_ll), f),
        Optim.iterations(res))
end

function _bivariate_q4_marker(rhs)
    params = (:mu1, :mu2, :sigma1, :sigma2)
    fixed = Dict{Symbol,Any}()
    markers = Dict{Symbol,Any}()
    for p in params
        fixed_p, structured = _split_bivariate_q4_rhs(rhs[p], p)
        fixed[p] = fixed_p
        structured !== nothing && (markers[p] = structured)
    end
    fixed_rho, re_rho, metav_rho, structured_rho = _split_ranef(rhs[:rho12])
    isempty(re_rho) || error("`rho12` is the residual-correlation formula and cannot contain random-effect terms")
    metav_rho === nothing || error("`rho12` is the residual-correlation formula and cannot contain `meta_V`")
    structured_rho === nothing || error("`rho12` is the residual correlation; group-level coevolution lives in the shared `phylo` block on mu1/mu2/sigma1/sigma2")
    fixed[:rho12] = fixed_rho

    isempty(markers) && return fixed, nothing
    marker_keys = Set(keys(markers))
    if marker_keys == Set((:mu1, :mu2))
        marker_vals = [markers[p] for p in (:mu1, :mu2)]
        kind = marker_vals[1][1]
        kind in (:phylo, :relmat, :animal, :spatial) ||
            error("the bivariate q=2 front end currently supports only `phylo(...)`, `relmat(...)`, `animal(...)`, or `spatial(...)` markers")
        all(m -> m[1] === kind, marker_vals) ||
            error("the q=2 structured markers on mu1 and mu2 must use the same structured type")
        groups = [m[2] for m in marker_vals]
        all(==(groups[1]), groups) ||
            error("the q=2 structured markers on mu1 and mu2 must use the same grouping variable")
        tags = [m[3] for m in marker_vals]
        (all(isnothing, tags) || (tags[1] == tags[2])) ||
            error("the q=2 structured markers on mu1 and mu2 must share the same correlation tag")
        return fixed, (:structured_q2, kind, groups[1])
    end
    length(markers) == length(params) ||
        error("bivariate structured Julia fits require either q=2 markers on mu1 and mu2 only, or q=4 markers on mu1, mu2, sigma1, and sigma2")
    marker_vals = [markers[p] for p in params]   # one (kind, group, tag) per axis
    kind = marker_vals[1][1]
    kind in (:phylo, :relmat, :animal, :spatial) ||
        error("the bivariate q=4 front end supports `phylo(...)`, `relmat(...)`, `animal(...)`, or `spatial(...)` markers")
    all(m -> m[1] === kind, marker_vals) ||
        error("the q=4 structured markers on mu1, mu2, sigma1, and sigma2 must use the same structured type")
    groups = [m[2] for m in marker_vals]
    all(==(groups[1]), groups) ||
        error("the q=4 structured markers on mu1, mu2, sigma1, and sigma2 must use the same grouping variable")
    tags = [m[3] for m in marker_vals]           # axis order: mu1, mu2, sigma1, sigma2
    lc_zero = _q4_block_lc_zero(tags)
    kind === :phylo && return fixed, (:phylo_q4, groups[1], lc_zero)
    return fixed, (:structured_q4, kind, groups[1], lc_zero)
end

# Translate the per-axis correlation tags into the set of log-Cholesky indices to
# pin at 0 so Σ_a is block-diagonal across distinct tags.
#
# Axis order = (mu1, mu2, sigma1, sigma2) = Σ_a rows/cols 1..4. The 10-vec lc is
# the column-major lower triangle of the Cholesky factor C (lc_to_Λ): index→(i,j)
#   1→(1,1) 2→(2,1) 3→(3,1) 4→(4,1) 5→(2,2) 6→(3,2) 7→(4,2) 8→(3,3) 9→(4,3) 10→(4,4)
# Zeroing every off-diagonal C[i,j] (i>j) whose two axes carry DIFFERENT tags
# makes C block-lower-triangular within each tag-group ⇒ C C' is exactly
# block-diagonal (the cross-tag covariance is identically 0).
#
# All-`nothing` tags (every axis written `phylo(1 | group)`) ⇒ spec E, the full
# correlated 4×4 Σ_a ⇒ `lc_zero = Int[]` (no constraint). A mix of tagged and
# untagged axes is rejected (ambiguous block membership).
function _q4_block_lc_zero(tags)
    all(isnothing, tags) && return Int[]
    any(isnothing, tags) &&
        error("the q=4 block-diagonal form needs a correlation tag on EVERY axis: " *
              "write `phylo(1 | tag | group)` on mu1, mu2, sigma1, sigma2 (got a mix " *
              "of tagged and untagged markers)")
    # column-major lower-tri (i, j) for each lc index 1..10
    ij = [(1,1),(2,1),(3,1),(4,1),(2,2),(3,2),(4,2),(3,3),(4,3),(4,4)]
    lc_zero = Int[]
    for (k, (i, j)) in enumerate(ij)
        i == j && continue                 # never pin a diagonal (block variances)
        tags[i] == tags[j] || push!(lc_zero, k)
    end
    return lc_zero
end

function _split_bivariate_q4_rhs(rhs, param::Symbol)
    terms = rhs isa Tuple ? collect(rhs) : Any[rhs]
    fixed = Any[]
    structured = nothing
    for t in terms
        if t isa FunctionTerm && t.f === (|)
            error("bivariate q=4 structured fits support only `phylo`/`relmat`/`animal`/`spatial(1 | group)` markers, not ordinary random effects")
        elseif t isa FunctionTerm && t.f === meta_V
            error("bivariate q=4 structured fits do not support `meta_V` markers")
        elseif t isa FunctionTerm && (t.f === relmat || t.f === animal || t.f === phylo || t.f === spatial)
            structured === nothing ||
                error("`$param` contains multiple structured markers; the q=4 front end accepts exactly one structured intercept marker per predictor")
            grp, tag = _q4_marker_group(t, param)
            structured = (_structured_marker_kind(t), grp, tag)
        else
            push!(fixed, t)
        end
    end
    fixed_rhs = isempty(fixed) ? ConstantTerm(1) :
                length(fixed) == 1 ? fixed[1] : Tuple(fixed)
    return fixed_rhs, structured
end

# Parse the inner term of a structured q=4 marker. Returns `(group, tag)` where
# `tag` is `nothing` for the 2-arg `phylo(1 | group)` (spec E, one shared 4×4 Σ_a)
# or the correlation-group symbol for the 3-arg `phylo(1 | tag | group)` (spec D,
# block-diagonal Σ_a — axes sharing a `tag` form one block). Julia parses
# `1 | tag | group` left-associatively as `|(|(1, tag), group)`.
function _q4_marker_group(t, param::Symbol)
    kind = _structured_marker_kind(t)
    inner = t.args[1]
    inner isa FunctionTerm && inner.f === (|) ||
        error("`$param` structured marker must be written as `$kind(1 | group)` or `$kind(1 | tag | group)`")
    if inner.args[1] isa FunctionTerm && inner.args[1].f === (|)
        # 3-arg tagged form: |(|(1, tag), group)
        coef = inner.args[1].args[1]
        (coef isa ConstantTerm && coef.n == 1) ||
            error("`$param` uses `$kind`, but the bivariate q=4 front end supports only intercept markers (`$kind(1 | tag | group)`)")
        inner.args[1].args[2] isa Term ||
            error("`$param` correlation tag must be a bare symbol in `$kind(1 | tag | group)`")
        inner.args[2] isa Term ||
            error("`$param` group must be a bare grouping variable in `$kind(1 | tag | group)`")
        return inner.args[2].sym, inner.args[1].args[2].sym
    end
    # 2-arg form: |(1, group)
    length(inner.args) == 2 && inner.args[2] isa Term ||
        error("`$param` structured marker must be written as `$kind(1 | group)` or `$kind(1 | tag | group)`")
    lhs = inner.args[1]
    (lhs isa ConstantTerm && lhs.n == 1) ||
        error("`$param` uses `$kind`, but the bivariate q=4 front end supports only intercept markers (`$kind(1 | group)`)")
    return inner.args[2].sym, nothing
end

function _structured_marker_kind(t)
    t.f === relmat && return :relmat
    t.f === animal && return :animal
    t.f === phylo && return :phylo
    t.f === spatial && return :spatial
    error("unsupported structured marker")
end

# Reorder the ML driver's retained coordinates instead of re-factorizing Λ:
# rounding can make an extreme log-Cholesky covariance numerically non-PD.
# Even a failed fit must remain packable so its nonconvergence can be reported.
function _q2_structured_theta(fit_q2, k::Int, method::Symbol)
    if method === :ML
        θ = fit_q2.θ
        return vcat(θ[1:(2k)], θ[(2k + 4):(2k + 6)], θ[(2k + 1):(2k + 3)])
    end
    return vcat(fit_q2.β[:, 1], fit_q2.β[:, 2],
                log.(fit_q2.σ_res), atanh(fit_q2.rho12 / RHO_GUARD),
                cov_to_lc(fit_q2.Λ))
end

function _fit_bivariate_q2_structured(f::BivariateDrmFormula, fam::Gaussian, data,
                                      fixed, marker, tree, K, A, coords;
                                      spatial_range = nothing,
                                      g_tol::Float64, method::Symbol = :ML)
    marker[1] === :structured_q2 || error("internal error: expected q2 structured marker")
    kind = marker[2]
    grp = marker[3]
    y1, X1, nm1 = _design(f.response1, fixed[:mu1], data)
    y2, X2, nm2 = _design(f.response2, fixed[:mu2], data)
    _, Xs1, nms1 = _design(f.response1, fixed[:sigma1], data)
    _, Xs2, nms2 = _design(f.response1, fixed[:sigma2], data)
    _, Xr, nmr = _design(f.response1, fixed[:rho12], data)
    all(isfinite, y1) && all(isfinite, y2) ||
        throw(ArgumentError("drm: bivariate q=2 structured Julia route currently requires complete responses"))
    size(Xs1, 2) == 1 && size(Xs2, 2) == 1 && size(Xr, 2) == 1 ||
        throw(ArgumentError("drm: bivariate q=2 structured Julia route currently supports intercept-only sigma1, sigma2, and rho12 formulas"))

    Y = hcat(Vector{Float64}(y1), Vector{Float64}(y2))
    size(X2, 2) == size(X1, 2) && X2 ≈ X1 ||
        throw(ArgumentError("drm: bivariate q=2 structured Julia route currently requires mu1 and mu2 to use the same fixed-effect design"))

    group_values = getproperty(data, grp)
    gidx, G = _group_index(group_values)
    phy = nothing
    species = Int[]
    prob, Q_cond = if kind === :phylo
        tree === nothing && error("phylo(1 | $grp) needs `tree = …`")
        phy_obj = _as_augmented_phy(tree)
        sp = _phylo_species_index(phy_obj, group_values)
        phy = phy_obj
        species = sp
        make_coevo_problem(phy_obj, Y, Matrix{Float64}(X1); species = sp)
    elseif kind === :relmat
        K === nothing && error("relmat(1 | $grp) needs `K = …`")
        C = Matrix{Float64}(K)
        size(C) == (G, G) || error("relmat structured matrix must be $(G)×$(G) (the number of `$grp` levels)")
        make_coevo_problem_from_covariance(C, Y, Matrix{Float64}(X1); group = gidx)
    elseif kind === :animal
        A === nothing && error("animal(1 | $grp) needs `A = …`")
        C = Matrix{Float64}(A)
        size(C) == (G, G) || error("animal relatedness matrix must be $(G)×$(G) (the number of `$grp` levels)")
        make_coevo_problem_from_covariance(C, Y, Matrix{Float64}(X1); group = gidx)
    elseif kind === :spatial
        # `spatial` supplies a covariance exactly as `relmat`/`animal` do; the
        # only difference is that the G-by-G matrix is BUILT from the coordinates
        # by the same fixed-range rule the q=4 route uses. Identical solver call
        # below, so `spatial(coords)` and `relmat(K = that same matrix)` are one
        # model reached by two names.
        C = _spatial_covariance_from_coords(grp, G, coords, spatial_range)
        make_coevo_problem_from_covariance(C, Y, Matrix{Float64}(X1); group = gidx)
    else
        error("internal error: unsupported q2 structured marker `$kind`")
    end

    β0 = hcat(X1 \ y1, X2 \ y2)
    r1 = y1 .- X1 * β0[:, 1]
    r2 = y2 .- X2 * β0[:, 2]
    ρ0 = if std(r1) > 0 && std(r2) > 0
        clamp(cor(r1, r2), -0.5, 0.5)
    else
        0.0
    end
    fit_q2 = if method === :REML
        fit_coevolution_q2_reml(
            prob,
            Q_cond;
            β0 = β0,
            Λ0 = Matrix(Symmetric([0.25 0.02; 0.02 0.25])),
            σ0 = [std(r1) + eps(), std(r2) + eps()],
            rho0 = ρ0,
            g_tol = g_tol,
            iterations = 300,
        )
    else
        fit_coevolution_q2_residual(
            prob,
            Q_cond;
            β0 = β0,
            Λ0 = Matrix(Symmetric([0.25 0.02; 0.02 0.25])),
            σ0 = [std(r1) + eps(), std(r2) + eps()],
            rho0 = ρ0,
            g_tol = g_tol,
            iterations = 300,
        )
    end

    k = size(X1, 2)
    blocks = [
        :mu1 => 1:k,
        :mu2 => (k + 1):(2k),
        :sigma1 => (2k + 1):(2k + 1),
        :sigma2 => (2k + 2):(2k + 2),
        :rho12 => (2k + 3):(2k + 3),
        :phylocov => (2k + 4):(2k + 6),
    ]
    names = [
        :mu1 => nm1,
        :mu2 => nm2,
        :sigma1 => nms1,
        :sigma2 => nms2,
        :rho12 => nmr,
        :phylocov => _q2_phylocov_names(),
    ]
    θ̂ = _q2_structured_theta(fit_q2, k, method)
    nll = function (θ)
        β = hcat(θ[blocks[1].second], θ[blocks[2].second])
        Λ = lc_to_cov(θ[blocks[6].second], 2)
        σ1 = exp(θ[blocks[3].second][1])
        σ2 = exp(θ[blocks[4].second][1])
        ρ = RHO_GUARD * tanh(θ[blocks[5].second][1])
        D = Matrix(Symmetric([σ1^2 ρ * σ1 * σ2; ρ * σ1 * σ2 σ2^2]))
        ℓ, = coevo_marginal_cov(prob, Q_cond, β, Λ, D)
        return -ℓ
    end
    # Observed-information vcov by finite differences of the marginal ML NLL.
    # ForwardDiff is not available on this route: `coevo_marginal_cov` factorises a
    # sparse H_uu through CHOLMOD (coevolution_q.jl), which accepts Float64 and
    # ComplexF64 only, so even a first-order Dual throws a TypeError. Second-
    # differencing the objective is the same fallback the sparse-LSS routes take
    # for the same reason (gaussian_sparse_lss.jl), through the shared helper; the
    # invert-vs-pseudo-invert decision and its boundary warning are the same guard
    # the dense ML bivariate sibling uses above.
    #
    # STEP: `_fd_hessian_step` is calibrated in the number of scalar observations
    # summed into the objective. Here that is 2 per row (two responses), so pass
    # 2 * length(y1), not length(y1). Measured 2026-09-05 against the EXACT
    # beta-block oracle `_q2_profile_and_schur(...).S`: at 1200 rows the 2n step
    # gives 1.17e-9 max relative error against 5.54e-9 for the n step (ML arm);
    # below 400 rows both clamp to 1e-4 and are bit-identical.
    #
    # ESTIMATOR: under method = :REML this is the ML curvature evaluated at the
    # REML point -- the restricted-penalty curvature is omitted, exactly as the q=4
    # sibling documents at its `_q4_fd_vcov` call site below. REML Wald intervals
    # on this route are therefore mildly anti-conservative.
    #
    # KNOWN WART: `_finite_hessian` hardcodes the prefix "sparse-Laplace vcov:" on
    # its own FD-quality warnings (sparse_laplace_glmm.jl). On this route that label
    # is wrong. The route-naming `context` below reaches the user through the
    # singularity warning; correcting the helper's prefix needs a shared file that
    # is out of scope for this change.
    V = if isfinite(fit_q2.loglik)
        _vcov_from_hessian(_finite_hessian(nll, θ̂; h = _fd_hessian_step(2 * length(y1)));
                           context = "bivariate Gaussian q=2 structured ($kind) " *
                                     "finite-difference Hessian")
    else
        # No curvature exists at a rejected final state. Preserve the failed
        # fit and its convergence flag without attempting an Inf/NaN Hessian.
        fill(NaN, length(θ̂), length(θ̂))
    end
    means = Dict(:mu1 => X1 * fit_q2.β[:, 1], :mu2 => X2 * fit_q2.β[:, 2])
    obs = Dict(:mu1 => Vector{Float64}(y1), :mu2 => Vector{Float64}(y2))
    scales = Dict(
        :sigma1 => fill(fit_q2.σ_res[1], length(y1)),
        :sigma2 => fill(fit_q2.σ_res[2], length(y1)),
        :rho12 => fill(fit_q2.rho12, length(y1)),
    )

    all_blups = reshape(Vector{Float64}(fit_q2.u_hat), 2, prob.N)
    blups = if kind === :phylo
        keep = setdiff(1:phy.n_total, [phy.root_index])
        node_pos = Dict(node => i for (i, node) in enumerate(keep))
        leaf_pos = [node_pos[phy.leaf_indices[k]] for k in 1:phy.n_leaves]
        all_blups[:, leaf_pos]
    else
        all_blups
    end
    re = (;
        effects = Dict(Symbol(grp) => blups),
        Sigma_a = Matrix{Float64}(fit_q2.Λ),
        axes = (:mu1, :mu2),
        structured_type = kind,
        Q_cond = Q_cond,
        phy = phy,
        group = grp,
        group_index = gidx,
        species = species,
        # The one free choice this cell makes is the FIXED range, and its
        # default is data-dependent, so it must be visible on the fit rather
        # than inferred. Same field, same rule, same default helper as the q=4
        # route records at its own `re`.
        spatial_range = kind === :spatial ?
            (spatial_range === nothing ?
                _q4_default_spatial_range(coords, G) : Float64(spatial_range)) :
            nothing,
        prob = prob,
    )
    fit = DrmFit(fam, blocks, names, θ̂, V, fit_q2.loglik, length(y1),
                 fit_q2.converged, means, obs, scales)
    fit = method === :REML ? _withreml(fit, fit_q2.reml_loglik, fit_q2.ml_loglik) : fit
    return _withranef(_withformula(_withnll(fit, nll), f), re)
end

# Exponential spatial covariance over the G levels of `grp`, at a FIXED range.
#
# Shared by the q=2 and q=4 bivariate structured routes so both cells build the
# same G-by-G matrix from the same coordinates, and so every fail-closed check on
# `coords` is stated exactly once. The range is `spatial_range` when supplied,
# else the mean off-diagonal pairwise distance; the 1e-8 ridge keeps the kernel
# positive definite at near-coincident sites. Row k of `coords` is the k-th
# level of `grp` in `_group_index` order (stable FIRST-SEEN order in the data),
# which is the same contract the q=4 route has always had.
#
# This is NOT the univariate spatial route: `_fit_spatial_gaussian`
# (gaussian_structured.jl) carries log rho as a free parameter and rebuilds the
# kernel each evaluation. Here rho is fixed, so the two are different models.
function _spatial_covariance_from_coords(grp::Symbol, G::Int, coords, spatial_range)
    coords === nothing && error("spatial(1 | $grp) needs `coords = …`")
    G >= 2 || error("spatial(1 | $grp) needs at least 2 distinct sites; got G=$G")
    # A 1-D transect is naturally written as a plain Vector (`coords = [0.0, 1.0, 2.0]`),
    # but `Matrix{Float64}(::Vector)` has no method and raised a bare MethodError. Accept it
    # as the G-by-1 column it obviously means. UNAMBIGUOUS: a flattened 2-D layout has length
    # 2G, so it reshapes to a 2G-by-1 and still trips the row check below with a message
    # naming the expected shape (measured: G=4 flattened G-by-2 has length 8, rows 8 != 4).
    # Anything neither a real vector nor matrix-convertible gets a shape error, not a
    # MethodError. This helper is SHARED by the q=2 and q=4 spatial routes, so both gain it.
    Cmat = if coords isa AbstractVector{<:Real}
        reshape(Float64.(coords), :, 1)
    else
        try
            Matrix{Float64}(coords)
        catch
            error("spatial(1 | $grp) needs `coords` as a $(G)-by-d matrix of site " *
                  "coordinates, or a length-$(G) vector for a 1-D transect; got a " *
                  "$(typeof(coords)) that is neither")
        end
    end
    size(Cmat, 1) == G ||
        error("spatial coords must have $G rows (one per `$grp` level); got $(size(Cmat, 1))")
    size(Cmat, 2) >= 1 || error("spatial coords must have at least one coordinate column")
    Ddist = [sqrt(sum(abs2, Cmat[k, :] .- Cmat[l, :])) for k in 1:G, l in 1:G]
    any(Ddist .> 0) ||
        error("spatial(1 | $grp): all site coordinates coincide; the spatial range is not identified")
    rho = if spatial_range === nothing
        sum(Ddist) / (G^2 - G)
    else
        Float64(spatial_range)
    end
    rho > 0 || error("spatial_range must be positive (got $rho)")
    return Matrix{Float64}(exp.(-Ddist ./ rho) + 1e-8 * I)
end

# Build a G×G SPD precision for level-indexed q=4 structured providers (#189).
# Spatial uses a *fixed* range (keyword `spatial_range`, else mean pairwise
# distance) — joint ρ estimation is deferred.
function _q4_structured_precision(kind::Symbol, grp::Symbol, G::Int;
                                  K, A, coords, spatial_range)
    if kind === :relmat
        K === nothing && error("relmat(1 | $grp) needs `K = …`")
        C = Matrix{Float64}(K)
        size(C) == (G, G) || error("relmat structured matrix must be $(G)×$(G) (the number of `$grp` levels)")
        isposdef(Symmetric(C)) || error("relmat K must be positive definite")
        return Matrix(inv(cholesky(Symmetric(C))))
    elseif kind === :animal
        A === nothing && error("animal(1 | $grp) needs `A = …`")
        C = Matrix{Float64}(A)
        size(C) == (G, G) || error("animal relatedness matrix must be $(G)×$(G) (the number of `$grp` levels)")
        isposdef(Symmetric(C)) || error("animal A must be positive definite")
        return Matrix(inv(cholesky(Symmetric(C))))
    elseif kind === :spatial
        Ksp = _spatial_covariance_from_coords(grp, G, coords, spatial_range)
        return Matrix(inv(cholesky(Symmetric(Matrix{Float64}(Ksp)))))
    else
        error("internal error: unsupported q4 structured kind `$kind`")
    end
end

"""
    _fit_bivariate_q4_structured(...)

q=4 PLSM front end for level-indexed structured providers (`relmat` / `animal` /
`spatial`) — issue #189. Reuses [`fit_q4_sparse_tmb`](@ref) via
[`make_problem_from_Q`](@ref); does not rewrite the verified Laplace engine.

Spatial uses a fixed range (`spatial_range`, default = mean pairwise distance).
Non-tree `bootstrap_sigma_a` is deliberately unsupported.
"""
function _fit_bivariate_q4_structured(f::BivariateDrmFormula, fam::Gaussian, data, fixed, marker,
                                      tree, K, A, coords;
                                      spatial_range = nothing,
                                      q4_g_tol::Real, q4_iterations::Int,
                                      q4_n_newton::Int, q4_vcov::Bool, method::Symbol = :ML)
    marker[1] === :structured_q4 || error("internal error: expected q4 structured marker")
    kind = marker[2]
    grp = marker[3]
    lc_zero = length(marker) >= 4 ? marker[4] : Int[]
    _ = tree   # unused for non-phylo providers; accepted for API symmetry
    y1, X1, nm1 = _design(f.response1, fixed[:mu1], data)
    y2, X2, nm2 = _design(f.response2, fixed[:mu2], data)
    _, Xs1, nms1 = _design(f.response1, fixed[:sigma1], data)
    _, Xs2, nms2 = _design(f.response1, fixed[:sigma2], data)
    _, Xr, nmr = _design(f.response1, fixed[:rho12], data)
    obs1 = _observed_response_mask(y1)
    obs2 = _observed_response_mask(y2)
    (count(obs1) >= size(X1, 2) && count(obs2) >= size(X2, 2)) ||
        throw(ArgumentError("drm: too few observed `$(f.response1)`/`$(f.response2)` rows " *
            "for the bivariate q=4 mean coefficients"))

    group_values = getproperty(data, grp)
    gidx, G = _group_index(group_values)
    Qdense = _q4_structured_precision(kind, grp, G;
                                      K = K, A = A, coords = coords,
                                      spatial_range = spatial_range)
    prob, Q_cond = make_problem_from_Q(Qdense, y1, y2, X1, X2, Xs1, Xs2, Xr; group = gidx)

    β1 = X1[obs1, :] \ y1[obs1]
    β2 = X2[obs2, :] \ y2[obs2]
    res1 = y1[obs1] .- X1[obs1, :] * β1
    res2 = y2[obs2] .- X2[obs2, :] * β2
    β0 = (
        mu1 = β1,
        mu2 = β2,
        s1 = _initial_scale_beta(Xs1, res1),
        s2 = _initial_scale_beta(Xs2, res2),
        rho = zeros(size(Xr, 2)),
    )
    Λ0 = Matrix(Symmetric([
        0.30 0.02 0.01 0.010
        0.02 0.30 0.01 0.010
        0.01 0.01 0.08 0.005
        0.01 0.01 0.005 0.080
    ]))
    if !isempty(lc_zero)
        lc0 = Λ_to_lc(Λ0)
        lc0[lc_zero] .= 0.0
        Λ0 = lc_to_Λ(lc0)
    end
    reml_ll = NaN
    ml_ll = NaN
    if method === :REML
        rr = fit_q4_reml(
            prob, Q_cond;
            beta0 = β0,
            Lambda0 = Λ0,
            g_tol = Float64(q4_g_tol),
            iterations = q4_iterations,
            n_newton = q4_n_newton,
            lc_zero = lc_zero,
        )
        β_reml = (mu1 = rr.beta.mu1, mu2 = rr.beta.mu2,
                  s1 = rr.beta.s1, s2 = rr.beta.s2, rho = rr.beta.rho)
        θ_reml = pack_theta(β_reml, rr.Lambda)
        reml_ll = rr.reml_loglik
        ml_ll = rr.ml_loglik
        r = (θ = θ_reml, β = β_reml, Λ = Matrix(rr.Lambda),
             loglik = rr.reml_loglik, converged = rr.converged)
    else
        r = fit_q4_sparse_tmb(
            prob, Q_cond;
            β0 = β0,
            Λ0 = Λ0,
            g_tol = Float64(q4_g_tol),
            iterations = q4_iterations,
            n_newton = q4_n_newton,
            lc_zero = lc_zero,
        )
    end

    k1, k2, ks1, ks2, kr = beta_widths(prob)
    offs = cumsum([0, k1, k2, ks1, ks2, kr, 10])
    rng(k) = (offs[k] + 1):offs[k + 1]
    blocks = [
        :mu1 => rng(1),
        :mu2 => rng(2),
        :sigma1 => rng(3),
        :sigma2 => rng(4),
        :rho12 => rng(5),
        :phylocov => rng(6),
    ]
    names = [
        :mu1 => nm1,
        :mu2 => nm2,
        :sigma1 => nms1,
        :sigma2 => nms2,
        :rho12 => nmr,
        :phylocov => _q4_phylocov_names(),
    ]
    θ̂ = Vector{Float64}(r.θ)
    nll(θ) = marginal_nll(prob, Q_cond, Vector{Float64}(θ); n_newton = q4_n_newton)[1]
    nllgrad! = function (g, θ)
        _, gg, _, _ = marginal_and_exact_grad(prob, Q_cond, Vector{Float64}(θ); n_newton = q4_n_newton)
        copyto!(g, gg)
        return g
    end
    V = q4_vcov ? _q4_fd_vcov(prob, Q_cond, θ̂; n_newton = q4_n_newton) :
        fill(NaN, length(θ̂), length(θ̂))

    β̂ = r.β
    means = Dict(:mu1 => X1 * β̂.mu1, :mu2 => X2 * β̂.mu2)
    obs = Dict(:mu1 => Vector{Float64}(y1), :mu2 => Vector{Float64}(y2))
    scales = Dict(
        :sigma1 => exp.(Xs1 * β̂.s1),
        :sigma2 => exp.(Xs2 * β̂.s2),
        :rho12 => RHO_GUARD .* tanh.(Xr * β̂.rho),
    )
    _, u_hat, _, _ = marginal_nll(prob, Q_cond, θ̂; n_newton = q4_n_newton)
    blups = reshape(Vector{Float64}(u_hat), 4, prob.n_total)   # 4 × G levels
    re = (;
        effects = Dict(Symbol(grp) => blups),
        Sigma_a = Matrix{Float64}(r.Λ),
        axes = (:mu1, :mu2, :sigma1, :sigma2),
        structured_type = kind,
        Q_cond = Q_cond,
        phy = nothing,
        group = grp,
        group_index = gidx,
        species = Int[],
        spatial_range = kind === :spatial ?
            (spatial_range === nothing ?
                _q4_default_spatial_range(coords, G) : Float64(spatial_range)) :
            nothing,
        prob = prob,
        n_newton = q4_n_newton,
    )
    # #509: the optimiser reported success at a numerically singular Λ
    # (saturated fixture, cond(Λ) = 1.3e12). Gate the public flag on the same
    # Λ-admissibility notion the q2 route's #503 guard uses — the estimates
    # stay available; success is just not claimed at an inadmissible Λ.
    q4_converged = r.converged && _q2_lambda_admissible(Matrix{Float64}(r.Λ))
    fit = DrmFit(fam, blocks, names, θ̂, V, r.loglik, length(y1), q4_converged, means, obs, scales)
    fit = method === :REML ? _withreml(fit, reml_ll, ml_ll) : fit
    return _withranef(_withformula(_withnll(fit, nll, nllgrad!), f), re)
end

function _q4_default_spatial_range(coords, G::Int)
    Cmat = Matrix{Float64}(coords)
    Ddist = [sqrt(sum(abs2, Cmat[k, :] .- Cmat[l, :])) for k in 1:G, l in 1:G]
    return sum(Ddist) / (G^2 - G)
end

function _fit_bivariate_q4_phylo(f::BivariateDrmFormula, fam::Gaussian, data, fixed, marker, tree;
                                 q4_g_tol::Real, q4_iterations::Int,
                                 q4_n_newton::Int, q4_vcov::Bool, method::Symbol = :ML)
    marker[1] === :phylo_q4 || error("internal error: expected q4 phylo marker")
    grp = marker[2]
    lc_zero = length(marker) >= 3 ? marker[3] : Int[]   # block-diagonal Σ_a pins
    tree === nothing && error("phylo(1 | $grp) needs `tree = …`")
    phy = _as_augmented_phy(tree)

    y1, X1, nm1 = _design(f.response1, fixed[:mu1], data)
    y2, X2, nm2 = _design(f.response2, fixed[:mu2], data)
    _, Xs1, nms1 = _design(f.response1, fixed[:sigma1], data)
    _, Xs2, nms2 = _design(f.response1, fixed[:sigma2], data)
    _, Xr, nmr = _design(f.response1, fixed[:rho12], data)
    # #19: missing responses are supported via a per-cell observed mask — observed
    # cells enter the likelihood, the full tree is kept. make_problem records the mask
    # (NaN ⇒ missing) and zeroes the missing values; here we only guard identifiability
    # and seed the optimiser from the OBSERVED rows.
    obs1 = _observed_response_mask(y1)
    obs2 = _observed_response_mask(y2)
    (count(obs1) >= size(X1, 2) && count(obs2) >= size(X2, 2)) ||
        throw(ArgumentError("drm: too few observed `$(f.response1)`/`$(f.response2)` rows " *
            "for the bivariate q=4 mean coefficients"))

    species = _phylo_species_index(phy, getproperty(data, grp))
    prob, Q_cond = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr; species = species)

    β1 = X1[obs1, :] \ y1[obs1]        # seed from observed rows (X1 \ y1 would be NaN)
    β2 = X2[obs2, :] \ y2[obs2]
    res1 = y1[obs1] .- X1[obs1, :] * β1
    res2 = y2[obs2] .- X2[obs2, :] * β2
    β0 = (
        mu1 = β1,
        mu2 = β2,
        s1 = _initial_scale_beta(Xs1, res1),
        s2 = _initial_scale_beta(Xs2, res2),
        rho = zeros(size(Xr, 2)),
    )
    Λ0 = Matrix(Symmetric([
        0.30 0.02 0.01 0.010
        0.02 0.30 0.01 0.010
        0.01 0.01 0.08 0.005
        0.01 0.01 0.005 0.080
    ]))
    if !isempty(lc_zero)
        # Make the start factor block-consistent with the EXACT pinned pattern the
        # fit enforces (`fit_q4_sparse_tmb`/`fit_q4_reml` force-zero `lc[lc_zero]`).
        # Zeroing the same log-Cholesky indices — rather than a hard-coded mu↔sigma
        # cross block (axes 1,2 vs 3,4) — keeps the start correct for ANY tag layout
        # (e.g. `{mu1,sigma1}` vs `{mu2,sigma2}` blocks), not just the default split.
        lc0 = Λ_to_lc(Λ0)
        lc0[lc_zero] .= 0.0
        Λ0 = lc_to_Λ(lc0)
    end
    reml_ll = NaN
    ml_ll = NaN
    if method === :REML
        # REML (Patterson–Thompson): β_μ profiled out via the bordered augmented
        # state; fit_q4_reml warm-starts from an internal ML fit. We rebuild a
        # unified `r` so the DrmFit construction below is identical to the ML path.
        rr = fit_q4_reml(
            prob, Q_cond;
            beta0 = β0,
            Lambda0 = Λ0,
            g_tol = Float64(q4_g_tol),
            iterations = q4_iterations,
            n_newton = q4_n_newton,
            lc_zero = lc_zero,
        )
        β_reml = (mu1 = rr.beta.mu1, mu2 = rr.beta.mu2,
                  s1 = rr.beta.s1, s2 = rr.beta.s2, rho = rr.beta.rho)
        θ_reml = pack_theta(β_reml, rr.Lambda)   # full 17-vec θ̂ (β + log-Cholesky Λ)
        reml_ll = rr.reml_loglik
        ml_ll = rr.ml_loglik
        r = (θ = θ_reml, β = β_reml, Λ = Matrix(rr.Lambda),
             loglik = rr.reml_loglik, converged = rr.converged)
    else
        r = fit_q4_sparse_tmb(
            prob, Q_cond;
            β0 = β0,
            Λ0 = Λ0,
            g_tol = Float64(q4_g_tol),
            iterations = q4_iterations,
            n_newton = q4_n_newton,
            lc_zero = lc_zero,
        )
    end

    k1, k2, ks1, ks2, kr = beta_widths(prob)
    offs = cumsum([0, k1, k2, ks1, ks2, kr, 10])
    rng(k) = (offs[k] + 1):offs[k + 1]
    blocks = [
        :mu1 => rng(1),
        :mu2 => rng(2),
        :sigma1 => rng(3),
        :sigma2 => rng(4),
        :rho12 => rng(5),
        :phylocov => rng(6),
    ]
    names = [
        :mu1 => nm1,
        :mu2 => nm2,
        :sigma1 => nms1,
        :sigma2 => nms2,
        :rho12 => nmr,
        :phylocov => _q4_phylocov_names(),
    ]
    θ̂ = Vector{Float64}(r.θ)
    nll(θ) = marginal_nll(prob, Q_cond, Vector{Float64}(θ); n_newton = q4_n_newton)[1]
    nllgrad! = function (g, θ)
        _, gg, _, _ = marginal_and_exact_grad(prob, Q_cond, Vector{Float64}(θ); n_newton = q4_n_newton)
        copyto!(g, gg)
        return g
    end
    # S5 change (c): compute û at θ̂ ONCE, before the vcov call, and reuse it as
    # a warm u0 for every _q4_fd_vcov perturbed evaluation (moved up from
    # below, where it ran AFTER the vcov call and so could not help it). Each
    # of the 2*n_theta perturbed evaluations previously started its inner
    # Newton cold (u0 = nothing, >= 200 iterations); a perturbation of θ̂ by
    # h = 1e-4 lands very close to θ̂ itself, so û(θ̂) is an excellent warm
    # start for û(θ̂ ± h e_k) too.
    #
    # The warm and cold inner Newton do NOT share a stopping criterion: a
    # warm u0 routes `estep_mode` to `_estep_fast`, which exits at
    # ‖∇J‖ < ftol = 1e-6 (sparse_aug_plsm.jl); the cold path (u0 = nothing,
    # every OTHER caller of estep_mode, and every perturbed evaluation before
    # this S5 change) goes to `_estep_robust`, which exits at tol = 1e-8. So
    # û(θ̂ ± h e_k) computed here is converged one order of magnitude looser
    # than the pre-S5 cold reference (Noether audit, Q3) -- exactly the
    # mechanism test/test_q4_perf_identities.jl's G5d.1/G5d.2 characterise
    # (u_hat gap ≤ ftol, amplified by the FD Hessian's 1/2h).
    _, u_hat, _, _ = marginal_nll(prob, Q_cond, θ̂; n_newton = q4_n_newton)
    # V is the ML observed-information vcov (FD of the marginal ML NLL) evaluated
    # at θ̂. Under method = :REML this is θ̂_reml, so V is the ML curvature at the
    # REML point; the restricted-penalty curvature (−0.5·∂²logdet S/∂θ²) is
    # omitted, mirroring the q=2 σ-phylo REML route (gaussian_locscale_phylo.jl).
    V = q4_vcov ? _q4_fd_vcov(prob, Q_cond, θ̂; n_newton = q4_n_newton, u0 = u_hat) :
        fill(NaN, length(θ̂), length(θ̂))

    β̂ = r.β
    means = Dict(:mu1 => X1 * β̂.mu1, :mu2 => X2 * β̂.mu2)
    obs = Dict(:mu1 => Vector{Float64}(y1), :mu2 => Vector{Float64}(y2))
    scales = Dict(
        :sigma1 => exp.(Xs1 * β̂.s1),
        :sigma2 => exp.(Xs2 * β̂.s2),
        :rho12 => RHO_GUARD .* tanh.(Xr * β̂.rho),   # report the model's guarded ρ (engine uses RHO_GUARD)
    )
    all_blups = reshape(Vector{Float64}(u_hat), 4, prob.n_total)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    node_pos = Dict(node => i for (i, node) in enumerate(keep))
    leaf_pos = [node_pos[phy.leaf_indices[k]] for k in 1:phy.n_leaves]
    blups = all_blups[:, leaf_pos]
    re = (;
        effects = Dict(Symbol(grp) => blups),
        Sigma_a = Matrix{Float64}(r.Λ),
        axes = (:mu1, :mu2, :sigma1, :sigma2),
        Q_cond = Q_cond,
        phy = phy,
        group = grp,
        species = species,
        prob = prob,                 # AugProblem — lets profile_sigma_a re-optimise the marginal
        n_newton = q4_n_newton,      # the inner-mode iteration count this fit used
    )
    # #509: same Λ-admissibility gate as the structured constructor above.
    q4_converged = r.converged && _q2_lambda_admissible(Matrix{Float64}(r.Λ))
    fit = DrmFit(fam, blocks, names, θ̂, V, r.loglik, length(y1), q4_converged, means, obs, scales)
    fit = method === :REML ? _withreml(fit, reml_ll, ml_ll) : fit
    return _withranef(_withformula(_withnll(fit, nll, nllgrad!), f), re)
end

_as_augmented_phy(tree::AugmentedPhy) = tree
_as_augmented_phy(tree::AbstractString) = augmented_phy(tree)
_as_augmented_phy(tree) = error("tree must be an AugmentedPhy or Newick string for bivariate q=4 phylogenetic fits")

function _phylo_species_index(phy::AugmentedPhy, labels)
    length(labels) > 0 || error("phylo grouping column is empty")
    if all(l -> l isa Integer, labels)
        idx = Int.(labels)
        all(i -> 1 <= i <= phy.n_leaves, idx) ||
            error("integer phylo group labels must be in 1:$(phy.n_leaves)")
        return idx
    end
    name_to_idx = Dict(name => i for (i, name) in enumerate(phy.leaf_names))
    idx = Vector{Int}(undef, length(labels))
    for (i, label) in enumerate(labels)
        key = String(label)
        haskey(name_to_idx, key) ||
            error("phylo group label `$key` is not present in the tree tip names")
        idx[i] = name_to_idx[key]
    end
    return idx
end

function _initial_scale_beta(X, residual)
    y = fill(log(std(residual) + eps()), size(X, 1))
    return X \ y
end

function _q4_phylocov_names()
    ["Sigma_a:L11", "Sigma_a:L21", "Sigma_a:L31", "Sigma_a:L41", "Sigma_a:L22",
     "Sigma_a:L32", "Sigma_a:L42", "Sigma_a:L33", "Sigma_a:L43", "Sigma_a:L44"]
end

_q2_phylocov_names() = ["Sigma_a:L11", "Sigma_a:L21", "Sigma_a:L22"]

function _q4_fd_vcov(prob::AugProblem, Q_cond::SparseMatrixCSC, θ::Vector{Float64};
                     h::Real = 1e-4, n_newton::Int = 40, u0 = nothing)
    nθ = length(θ)
    H = zeros(nθ, nθ)
    for k in 1:nθ
        θp = copy(θ); θp[k] += h
        θm = copy(θ); θm[k] -= h
        # S5 change (c): warm u0 into every perturbed evaluation's inner Newton
        # (u0 = nothing, the pre-S5 default, is unaffected -- still goes
        # straight to the cold robust path).
        _, gp, _, _ = marginal_and_exact_grad(prob, Q_cond, θp; u0 = u0, n_newton = n_newton)
        _, gm, _, _ = marginal_and_exact_grad(prob, Q_cond, θm; u0 = u0, n_newton = n_newton)
        H[:, k] .= (gp .- gm) ./ (2h)
    end
    return _vcov_from_hessian(H; context = "q=4 finite-difference Hessian")
end
