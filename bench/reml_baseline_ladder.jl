using DRModels
using LinearAlgebra
using Random

const REML_BASELINE_LADDER_REPORT_FIELDS = (
    :sha,
    :dirty,
    :julia_version,
    :threads,
    :blas,
    :fixture,
    :protocol,
    :accounting,
    :evidence_class,
    :fits,
)

const REML_BASELINE_LADDER_FIT_FIELDS = (
    :method,
    :seconds,
    :converged,
    :objective,
    :estimates,
    :interval_status,
    :timing_class,
    :fit_order,
)

# A deterministic next rung beyond the p=8 smoke fixture. It remains intentionally
# small enough for local warm-harness runs, rather than implying an Ayumi/10k scale test.
const REML_BASELINE_LADDER_INTERMEDIATE_FIXTURE = (p = 16, nrep = 3, seed = 291)

"""
Report-only structural accounting for the current central-FD REML outer loop.

For the Arc 0/1 q4 fixture, outer φ = (β_ρ, lc(Λ)) with `n_lc = 10` for a 4×4
Λ. Each gradient request evaluates the restricted objective once at the base
point and twice per outer coordinate. This is an upper-bound route count before
line search — not a measured call counter from the optimizer.
"""
function reml_baseline_q4_structural_accounting(; n_rho::Int = 1, n_lc::Int = 10)
    n_rho >= 1 || throw(ArgumentError("n_rho must be at least 1"))
    n_lc == 10 || throw(ArgumentError("q=4 Λ uses exactly 10 log-Cholesky coordinates"))
    n_outer = n_rho + n_lc
    return (
        n_rho = n_rho,
        n_lc = n_lc,
        n_outer_phi = n_outer,
        structural_evals_per_gradient_request = 1 + 2 * n_outer,
        source = :central_fd_route_upper_bound,
    )
end

function _reml_baseline_ladder_git(args...; default = "unavailable")
    try
        return strip(read(Cmd(["git", args...]), String))
    catch
        return default
    end
end

function reml_baseline_ladder_metadata()
    return (
        sha = _reml_baseline_ladder_git("rev-parse", "HEAD"),
        # Ignore untracked files so local worktree roots do not make a committed
        # checkout appear source-dirty; tracked modifications remain visible.
        dirty = !isempty(_reml_baseline_ladder_git(
            "status", "--porcelain", "--untracked-files=no"; default = "",
        )),
        julia_version = string(VERSION),
        threads = Threads.nthreads(),
        blas = sprint(show, BLAS.get_config()),
    )
end

function _reml_baseline_ladder_fixture(; p::Int, nrep::Int, seed::Int)
    rng = MersenneTwister(seed)
    phy = random_balanced_tree(p; branch_length = 0.2)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    Λ = [
        0.25 0.10 0.05 0.00
        0.10 0.25 0.00 0.04
        0.05 0.00 0.09 0.02
        0.00 0.04 0.02 0.09
    ]
    P = prior_precision(phy.Q_topology[keep, keep], inv(Λ))
    u_aug = cholesky(Symmetric(P)).UP \ randn(rng, size(P, 1))
    pos = Dict(node => i for (i, node) in enumerate(keep))
    leaf_pos = [pos[node] for node in phy.leaf_indices]
    species_index = repeat(1:p, inner = nrep)
    species = [phy.leaf_names[i] for i in species_index]
    n = length(species_index)
    x = randn(rng, n)
    y1 = Vector{Float64}(undef, n)
    y2 = Vector{Float64}(undef, n)
    rho = DRModels.RHO_GUARD * tanh(0.3)

    for i in eachindex(y1)
        u = @view u_aug[(4 * (leaf_pos[species_index[i]] - 1) + 1):(4 * leaf_pos[species_index[i]])]
        sigma1 = exp(-0.4 + u[3])
        sigma2 = exp(-0.5 + u[4])
        e = cholesky(Symmetric([sigma1^2 rho * sigma1 * sigma2;
                                rho * sigma1 * sigma2 sigma2^2])).L * randn(rng, 2)
        y1[i] = 1.0 + 0.5 * x[i] + u[1] + e[1]
        y2[i] = -0.3 + 0.4 * x[i] + u[2] + e[2]
    end

    formula = bf(
        mu1 = @formula(y1 ~ x + phylo(1 | species)),
        mu2 = @formula(y2 ~ x + phylo(1 | species)),
        sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
        sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
        rho12 = @formula(rho12 ~ 1),
    )
    return (; formula, data = (; y1, y2, x, species), phy)
end

function _reml_baseline_ladder_fit_record(method, fit, seconds;
                                         timing_class::Symbol,
                                         fit_order::Int)
    return (
        method = method,
        seconds = seconds,
        converged = fit.converged,
        objective = (
            reported_loglik = fit.loglik,
            ml_loglik = fit.ml_loglik,
            reml_loglik = fit.reml_loglik,
        ),
        estimates = (
            theta = copy(fit.theta),
            sigma_a_diagonal = diag(fit.ranef.Sigma_a),
        ),
        # q=4 profile/bootstrap inference is intentionally not part of this
        # timing harness; recording that fact prevents a silent CI comparison.
        interval_status = :not_evaluated,
        timing_class = timing_class,
        fit_order = fit_order,
    )
end

function _reml_baseline_ladder_fit!(fixture, method; q4_iterations, q4_n_newton)
    elapsed = @elapsed fit = drm(
        fixture.formula, Gaussian();
        data = fixture.data,
        tree = fixture.phy,
        method,
        q4_iterations,
        q4_n_newton,
        q4_vcov = false,
    )
    return fit, elapsed
end

function reml_baseline_ladder_markdown(report)
    lines = String[
        "# Issue #291 — small-fixture ML vs baseline REML",
        "",
        "This is a sparse-baseline characterization record, not an acceleration",
        "or AI-REML claim.",
        "",
        "## Environment",
        "- SHA: `$(report.sha)`",
        "- Dirty tracked tree: `$(report.dirty)`",
        "- Julia: `$(report.julia_version)`",
        "- Threads: `$(report.threads)`",
        "- BLAS: `$(report.blas)`",
        "- Fixture: p=$(report.fixture.p), nrep=$(report.fixture.nrep), seed=$(report.fixture.seed)",
        "- Protocol: `$(report.protocol)`",
        "- Evidence class: `$(report.evidence_class)`",
        "",
        "## Structural accounting (report-only)",
        "- n_rho: `$(report.accounting.n_rho)`",
        "- n_lc: `$(report.accounting.n_lc)`",
        "- n_outer_phi: `$(report.accounting.n_outer_phi)`",
        "- structural_evals_per_gradient_request: `$(report.accounting.structural_evals_per_gradient_request)`",
        "- accounting_source: `$(report.accounting.source)`",
        "",
        "## Fits",
    ]
    for record in report.fits
        push!(lines, "- `$(record.method)`: seconds=$(record.seconds), converged=$(record.converged), " *
                     "timing_class=$(record.timing_class), fit_order=$(record.fit_order), " *
                     "reported_loglik=$(record.objective.reported_loglik), " *
                     "ml_loglik=$(record.objective.ml_loglik), " *
                     "reml_loglik=$(record.objective.reml_loglik), " *
                     "interval_status=$(record.interval_status)")
        push!(lines, "  - theta=$(repr(record.estimates.theta))")
        push!(lines, "  - sigma_a_diagonal=$(repr(record.estimates.sigma_a_diagonal))")
    end
    if report.evidence_class === :diagnostic_only
        push!(lines, "")
        push!(lines, "## Evidence note")
        push!(lines, "Cold / compile-contaminated elapsed times are diagnostic only and")
        push!(lines, "must not be used to rank ML against REML or to claim sparse-vs-AI speed.")
    elseif report.evidence_class === :warm_comparable
        push!(lines, "")
        push!(lines, "## Evidence note")
        push!(lines, "Warm timed passes exclude an initial compile/warmup pair. Relative seconds")
        push!(lines, "may be compared across methods **within this fixture only**; they are still")
        push!(lines, "not a public speed headline and not an AI-REML result.")
    end
    return join(lines, "\n") * "\n"
end

"""
    reml_baseline_ladder_report(; p=8, nrep=3, seed=291,
                                protocol=:warm_bidirectional,
                                q4_iterations=100, q4_n_newton=30)

Fit the same small, deterministic Gaussian q=4 phylogenetic location-scale
fixture by ML and by the existing baseline `method=:REML` optimizer.

`protocol`:

- `:cold_ml_first` — historical Arc 0 order (ML then REML). Evidence class is
  always `:diagnostic_only` because the first fit absorbs compile cost.
- `:warm_bidirectional` — discard one ML+REML warmup pair, then time REML→ML
  and ML→REML. Evidence class is `:warm_comparable` when all timed fits converge.

Structural FD accounting is report-only and does not instrument `src/`.

Run with:

    julia --project=. bench/reml_baseline_ladder.jl

Set `DRM_REML_LADDER_OUTPUT=/path/to/report.md` to also write the Markdown
record to a durable artifact. Set `DRM_REML_LADDER_PROTOCOL=cold_ml_first` to
force the cold protocol from the CLI entrypoint.

The named intermediate rung preserves the seed family while increasing the
number of tips:

    reml_baseline_ladder_report(; REML_BASELINE_LADDER_INTERMEDIATE_FIXTURE...)
"""
function reml_baseline_ladder_report(; p::Int = 8, nrep::Int = 3, seed::Int = 291,
                                     protocol::Symbol = :warm_bidirectional,
                                     q4_iterations::Int = 100, q4_n_newton::Int = 30,
                                     n_rho::Int = 1)
    p >= 4 || throw(ArgumentError("p must be at least 4 for a balanced q4 tree"))
    nrep >= 2 || throw(ArgumentError("nrep must be at least 2 for scale-RE identifiability"))
    protocol in (:cold_ml_first, :warm_bidirectional) ||
        throw(ArgumentError("protocol must be :cold_ml_first or :warm_bidirectional"))

    fixture = _reml_baseline_ladder_fixture(; p, nrep, seed)
    accounting = reml_baseline_q4_structural_accounting(; n_rho)
    fits = NamedTuple[]
    order = 0

    if protocol === :cold_ml_first
        for method in (:ML, :REML)
            fit, elapsed = _reml_baseline_ladder_fit!(
                fixture, method; q4_iterations, q4_n_newton,
            )
            order += 1
            push!(fits, _reml_baseline_ladder_fit_record(
                method, fit, elapsed;
                timing_class = order == 1 ? :cold_includes_compile : :after_cold_companion,
                fit_order = order,
            ))
        end
        evidence_class = :diagnostic_only
    else
        # Warmup pair — elapsed times discarded from comparable ranking.
        for method in (:ML, :REML)
            fit, elapsed = _reml_baseline_ladder_fit!(
                fixture, method; q4_iterations, q4_n_newton,
            )
            order += 1
            push!(fits, _reml_baseline_ladder_fit_record(
                method, fit, elapsed;
                timing_class = :warmup_discarded,
                fit_order = order,
            ))
        end
        for method in (:REML, :ML, :ML, :REML)
            fit, elapsed = _reml_baseline_ladder_fit!(
                fixture, method; q4_iterations, q4_n_newton,
            )
            order += 1
            push!(fits, _reml_baseline_ladder_fit_record(
                method, fit, elapsed;
                timing_class = :warm_timed,
                fit_order = order,
            ))
        end
        warm = filter(r -> r.timing_class === :warm_timed, fits)
        evidence_class = all(r -> r.converged, warm) ? :warm_comparable : :diagnostic_only
    end

    return (;
        reml_baseline_ladder_metadata()...,
        fixture = (; p, nrep, seed),
        protocol,
        accounting,
        evidence_class,
        fits,
    )
end

function reml_baseline_ladder_main()
    proto = get(ENV, "DRM_REML_LADDER_PROTOCOL", "warm_bidirectional")
    protocol = Symbol(proto)
    report = reml_baseline_ladder_report(; protocol)
    markdown = reml_baseline_ladder_markdown(report)
    print(markdown)
    output = get(ENV, "DRM_REML_LADDER_OUTPUT", "")
    isempty(output) || write(output, markdown)
    return report
end

if abspath(PROGRAM_FILE) == @__FILE__
    reml_baseline_ladder_main()
end
