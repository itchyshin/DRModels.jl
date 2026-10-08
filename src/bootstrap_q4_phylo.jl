# Parametric bootstrap of the q=4 phylogenetic location–scale among-axis SDs.
#
# The single-tree complement to Ayumi's across-tree sweep. A bivariate q=4
# σ-phylo fit reports the 4×4 among-axis covariance Σ_a; its diagonal SDs
# sqrt.(diag(Σ_a)) for axes (μ1, μ2, σ1, σ2) are the quantities of interest
# ("is there phylogenetic signal in this trait's mean / in its variance?").
# When an axis collapses the boundary makes a Wald SE or a profile singular
# (profile_result → SingularException) — so neither gives an honest interval.
#
# This bootstrap does: each replicate redraws tip random effects from the fitted
# N(0, Q_cond⁻¹ ⊗ Σ̂_a) on the SAME tree (the exact precision-Cholesky draw the
# verified `simulate_coevolution` uses), adds them to the fitted fixed effects on
# all four axes, regenerates (y1, y2) with the fitted residual ρ12, refits the
# verified q=4 engine with the original estimator (ML or REML), and records
# sqrt.(diag(Σ̂_a)). Percentile CIs respect the
# SD ≥ 0 boundary automatically: a collapsing axis yields an interval whose lower
# end sits at ~0 — the honest "no detectable phylogenetic signal in that axis".

"""
    bootstrap_sigma_a(fit; data, B = 300, level = 0.95, rng = default_rng(),
                      failures = :warn, check_converged = true,
                      q4_g_tol = 1e-3, q4_iterations = 300) -> NamedTuple

Parametric bootstrap of the among-axis standard deviations `sqrt.(diag(Σ_a))`
for a bivariate q=4 phylogenetic location–scale fit
(`drm(bf(mu1=…, mu2=…, sigma1=…, sigma2=…, rho12=…), Gaussian(); tree = …)`).

Returns a `NamedTuple` with

- `summary` — a vector of rows `(param, coef, estimate, std_error, lower, upper)`
  for `param ∈ (:sd_mu1, :sd_mu2, :sd_sigma1, :sd_sigma2)`. `estimate` is the
  fitted SD; `lower`/`upper` are the `level` percentile interval over the
  successful replicates.
- `cor_summary` — the same rows for the 6 among-axis **correlations**
  `(:cor_mu1_mu2, :cor_mu1_sigma1, :cor_mu1_sigma2, :cor_mu2_sigma1,
  :cor_mu2_sigma2, :cor_sigma1_sigma2)` — the coevolutionary correlations of
  `coevolution_cor`, now with CIs. A correlation whose axis collapses is
  unidentified and comes back with a wide interval (e.g. ρ_a(μ1,μ2) spanning the
  sign when a σ-axis pins).
- `status` — `"bootstrap"` when every replicate is kept, `"bootstrap_incomplete"`
  when any is dropped and at least two remain, `"bootstrap_at_boundary"` when the
  retained draws also sit on an SD or correlation bound, or
  `"bootstrap_unavailable"` when fewer than two remain. A dropped replicate warns.
- `attempted`, `used`, `failed`, `failures`, `level`, `draws` (the `used × 4`
  matrix of replicate SDs), `cor_draws` (`used × 6`), `axes`, `cor_pairs`, `elapsed`.

The interval is the honest report at a boundary: when a scale axis carries no
phylogenetic signal its SD collapses and the percentile interval includes ~0,
where the Hessian-based SE / profile is undefined. `data` must be the table the
fit was estimated on (the covariate / grouping columns are reused; the two
response columns are overwritten per replicate).

With `failures = :warn` (default) failed refits are dropped and reported in
`failures`; `failures = :error` rethrows on the first failure.
"""
function bootstrap_sigma_a(fit::DrmFit; data, B::Real = 300, level::Real = 0.95,
                           rng = Random.default_rng(), failures::Symbol = :warn,
                           check_converged::Bool = true,
                           q4_g_tol::Real = 1e-3, q4_iterations::Int = 300)
    B = _validate_bootstrap_B(B)
    (failures === :warn || failures === :error) ||
        throw(ArgumentError("failures must be :warn or :error"))
    level = _validate_ci_level(level; what="bootstrap_sigma_a")
    re = fit.ranef
    (re isa NamedTuple && haskey(re, :Sigma_a) && haskey(re, :Q_cond) &&
     haskey(re, :phy) && re.phy isa AugmentedPhy && haskey(re, :species) &&
     !isempty(re.species)) || throw(ArgumentError(
        "bootstrap_sigma_a currently supports only tree-driven q=4 phylo fits " *
        "(fit.ranef with Sigma_a / Q_cond / AugmentedPhy phy / species). " *
        "Non-tree structured providers (spatial / relmat / animal; issue #189) " *
        "do not yet have a bootstrap path."))
    fit.formula isa BivariateDrmFormula || throw(ArgumentError(
        "bootstrap_sigma_a requires a BivariateDrmFormula fit created by drm"))

    Σa = Matrix{Float64}(re.Sigma_a)
    Q_cond = re.Q_cond
    phy = re.phy
    species = collect(Int, re.species)
    form = fit.formula
    fam = fit.family

    # Fixed (RE-free) parts per observation. fit.means/scales hold Xβ only (no
    # BLUP); log σ axes are recovered as log of the reported σ.
    μ1f = Vector{Float64}(fit.means[:mu1])
    μ2f = Vector{Float64}(fit.means[:mu2])
    logσ1f = log.(Vector{Float64}(fit.scales[:sigma1]))
    logσ2f = log.(Vector{Float64}(fit.scales[:sigma2]))
    ρ = Vector{Float64}(fit.scales[:rho12])
    n = length(μ1f)

    # leaf → augmented-node position, EXACTLY as the fit extracted its BLUPs
    # (gaussian_bivariate.jl) — guarantees consistency with this re.Q_cond.
    keep = setdiff(1:phy.n_total, [phy.root_index])
    node_pos = Dict(node => i for (i, node) in enumerate(keep))
    leaf_pos = [node_pos[phy.leaf_indices[k]] for k in 1:phy.n_leaves]

    Fchol = _q4_re_prior_chol(Q_cond, Σa)
    cols0 = Tables.columntable(data)

    axes = (:sd_mu1, :sd_mu2, :sd_sigma1, :sd_sigma2)
    # the 6 unique among-axis correlations (axis order mu1=1, mu2=2, σ1=3, σ2=4)
    cor_pairs = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    cor_names = (:cor_mu1_mu2, :cor_mu1_sigma1, :cor_mu1_sigma2,
                 :cor_mu2_sigma1, :cor_mu2_sigma2, :cor_sigma1_sigma2)
    est = sqrt.(max.(diag(Σa), 0.0))
    cor_est = _q4_cor_offdiag(Σa, cor_pairs)
    sd_draws = Matrix{Float64}(undef, B, 4)
    cor_draws = Matrix{Float64}(undef, B, 6)
    ok = falses(B)
    messages = Vector{Union{Nothing,String}}(nothing, B)
    refit_error = fill(false, B)
    seeds = rand(rng, UInt, B)

    function run_one!(b)
        rr = Random.MersenneTwister(seeds[b])
        try
            u = Fchol.UP \ randn(rr, size(Fchol, 1))
            U = reshape(u, 4, :)            # 4 × length(keep), axis-inner
            Uleaf = U[:, leaf_pos]          # 4 × n_leaves
            μ1 = μ1f .+ Uleaf[1, species]
            μ2 = μ2f .+ Uleaf[2, species]
            σ1 = exp.(logσ1f .+ Uleaf[3, species])
            σ2 = exp.(logσ2f .+ Uleaf[4, species])
            e1 = randn(rr, n)
            e2 = randn(rr, n)
            y1 = μ1 .+ σ1 .* e1
            y2 = μ2 .+ σ2 .* (ρ .* e1 .+ sqrt.(max.(1 .- ρ .^ 2, 0.0)) .* e2)
            datab = merge(cols0,
                          NamedTuple{(form.response1, form.response2)}((y1, y2)))
            fitb = drm(form, fam; data = datab, tree = phy,
                       q4_g_tol = q4_g_tol, q4_iterations = q4_iterations,
                       q4_vcov = false, method = fit.estim_method)
            if check_converged && !is_converged(fitb)
                messages[b] = _BOOTSTRAP_UNCONVERGED
                return nothing
            end
            Σb = Matrix{Float64}(fitb.ranef.Sigma_a)
            sd_draws[b, :] = sqrt.(max.(diag(Σb), 0.0))
            cor_draws[b, :] = _q4_cor_offdiag(Σb, cor_pairs)
            ok[b] = true
        catch err
            messages[b] = sprint(showerror, err)
            refit_error[b] = true
        end
        return nothing
    end

    elapsed = @elapsed for b in 1:B
        run_one!(b)
    end

    failure_rows = NamedTuple{(:replicate, :seed, :message),Tuple{Int,UInt,String}}[]
    for b in 1:B
        messages[b] === nothing && continue
        push!(failure_rows, (replicate = b, seed = seeds[b], message = messages[b]::String))
    end
    if failures === :error && any(refit_error)
        first_b = findfirst(refit_error)
        throw(ErrorException("bootstrap_sigma_a failed in $(count(refit_error)) of " *
            "$B replicates; first failure replicate $(first_b), seed $(seeds[first_b]): $(messages[first_b])"))
    end
    used = count(ok)
    if used == 0
        detail = isempty(failure_rows) ? "" : begin
            f = first(failure_rows)
            "; first failure replicate $(f.replicate), seed $(f.seed): $(f.message)"
        end
        throw(ErrorException("all $B bootstrap_sigma_a replicates failed$detail"))
    end

    used_sd = sd_draws[ok, :]
    used_cor = cor_draws[ok, :]
    α = (1 - level) / 2
    RowT = NamedTuple{(:param, :coef, :estimate, :std_error, :lower, :upper),
                      Tuple{Symbol,String,Float64,Float64,Float64,Float64}}
    summary = RowT[]
    for a in 1:4
        col = @view used_sd[:, a]
        push!(summary, (param = axes[a], coef = String(axes[a]), estimate = est[a],
                        std_error = Statistics.std(col),
                        lower = Statistics.quantile(col, α),
                        upper = Statistics.quantile(col, 1 - α)))
    end
    cor_summary = RowT[]
    for c in 1:6
        col = @view used_cor[:, c]
        push!(cor_summary, (param = cor_names[c], coef = String(cor_names[c]),
                            estimate = cor_est[c], std_error = Statistics.std(col),
                            lower = Statistics.quantile(col, α),
                            upper = Statistics.quantile(col, 1 - α)))
    end

    boundary_params = String[]
    if used >= _BOOTSTRAP_BOUNDARY_MIN_DRAWS
        for a in 1:4
            _bootstrap_param_at_boundary(axes[a], @view used_sd[:, a]) || continue
            push!(boundary_params, String(axes[a]))
        end
        for c in 1:6
            _bootstrap_param_at_boundary(cor_names[c], @view used_cor[:, c]) || continue
            push!(boundary_params, String(cor_names[c]))
        end
    end
    failed_n = length(failure_rows)
    status = _finish_bootstrap_status(used, failed_n, boundary_params)

    return (summary = summary, cor_summary = cor_summary, failures = failure_rows,
            attempted = B, used = used, failed = failed_n, seeds = seeds,
            level = level, draws = used_sd, cor_draws = used_cor, axes = axes,
            cor_pairs = cor_names, elapsed = elapsed, status = status,
            boundary_params = boundary_params, check_converged = check_converged)
end

# The 6 unique among-axis correlations from a 4×4 Σ_a, in `pairs` order. A
# collapsed axis (SD ≈ 0) makes its correlations unidentified — they come back
# wide under the bootstrap: the honest "this coevolution correlation isn't
# estimable here" report (e.g. ρ_a(μ1,μ2) riding toward ±1 when a σ-axis pins).
function _q4_cor_offdiag(Σ::AbstractMatrix, pairs)
    sd = sqrt.(max.(diag(Σ), 0.0))
    return Float64[(sd[i] > 0 && sd[j] > 0) ?
                   clamp(Σ[i, j] / (sd[i] * sd[j]), -1.0, 1.0) : 0.0
                   for (i, j) in pairs]
end

# Cholesky of the tip-RE prior precision kron(Q_cond, Σ_a⁻¹). Σ_a may be near
# singular at a σ-collapse; add a vanishing ridge only if the bare inverse fails,
# so the common path is byte-identical to the engine's own prior_precision draw.
function _q4_re_prior_chol(Q_cond, Σa::AbstractMatrix)
    Σ = Matrix{Float64}(Σa)
    for ridge in (0.0, 1e-10, 1e-8, 1e-6)
        Σr = ridge == 0.0 ? Σ : Σ + ridge * I
        try
            P = prior_precision(Q_cond, inv(Symmetric(Σr)))
            return cholesky(Symmetric(P))
        catch
            continue
        end
    end
    error("bootstrap_sigma_a: could not form a positive-definite tip-RE prior " *
          "precision from Σ_a (an axis is fully collapsed); try fewer axes")
end
