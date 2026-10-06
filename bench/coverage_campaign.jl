# coverage_campaign.jl — interval-coverage campaign runner (issue #468).
#
# Measures empirical coverage of DRModels.jl's public confint() Wald / profile
# intervals against known truth, on the two cell shapes the standing fences
# gate: test/test_parity_gaussian_phylo_mean.jl:77 and
# test/test_parity_biv_q4_phylo_reml.jl:72 both assert
# `interval_status != "coverage_claimed"`. This runner produces the evidence
# a fence-removal PR would need; it does not touch the fences itself
# (measurement first, fence PR second, Rose audit between it —
# docs/dev-log/evidence/2026-08-24-coverage-prerun.md §2).
#
# THE DESIGN IS FIXED — this file implements, and must not redesign, the two
# docs it was written from:
#   docs/dev-log/evidence/2026-08-24-coverage-prerun.md      (ADEMP spec, §2-5)
#   docs/dev-log/plans/2026-08-19-cell-d-ademp-pre-run.md    (DGP spine, #454)
#
# Sharded invocation (one process per seed block, up to 150 in parallel; the
# caller sets OPENBLAS_NUM_THREADS=1 in the env):
#
#   julia --project=<proj> --startup-file=no bench/coverage_campaign.jl \
#       --cell U --ntip 16 --per 4 --seed-start 1 --seed-count 25 --out /path/rows.tsv
#   julia --project=<proj> --startup-file=no bench/coverage_campaign.jl \
#       --cell B --seed-start 1 --seed-count 25 --out /path/rows.tsv
#   julia --project=<proj> --startup-file=no bench/coverage_campaign.jl \
#       --cell U --ntip 16 --per 4 --smoke
#
# Cell B has no ntip ladder (--ntip / --per are ignored for it); it always
# uses the fixture-sized shape (ntip=16, per=8 -> N=128), matching
# test/parity/q4-reml/biv-q4-phylo-reml.
#
# `include`ing this file runs nothing — `main()` only fires under
# `abspath(PROGRAM_FILE) == @__FILE__` (the include-runs-main() trap).
#
# RESTARTABILITY: each process writes ONLY its own --out file, one whole rep
# (every row for that seed) at a time, flushed immediately — a rep is either
# fully in the file or not present at all, never partially written. A
# killed/restarted process re-derives which seeds are already done by reading
# --out's `seed` column and skips them, so re-running the same shard command
# is idempotent.
#
# THE OUTPUT CONTRACT (tab-separated). The first 10 columns are the
# pre-run's validated smoke contract, fixed by design; the last 3 are
# appended, never reordered:
#   seed  conv  param  coef  method  estimate  lower  upper  truth  covered  t_fit_s  ntip  cell  jit
#
# `jit` is 1 for the very first rep this PROCESS executed (pays JIT
# compilation on top of the true fit cost) and 0 otherwise — the columns
# still carry that rep's real numbers (it is not dropped, per Williams 10b),
# but downstream per-rep timing analysis must exclude jit=1 rows or it will
# report the JIT-vs-warm timing trap as a real cost (docs/dev-log/plans/
# 2026-08-19-cell-d-ademp-pre-run.md, failure mode 3).
#
# SCALES (load-bearing, see the evidence doc §3): mu-block truths are on the
# identity scale; sigma / resd / sigma1 / sigma2 truths are on the LOG scale
# (matching each fit's own `theta` parameterisation); rho12 truth is on the
# atanh scale (0 either way, since the DGP's residual noise is independent);
# phylocov truths are the log-Cholesky (`cov_to_lc`) entries of the SAME 4x4
# among-axis covariance used to simulate the correlated phylo effects.

using DRModels
using LinearAlgebra
using Random
using Printf

LinearAlgebra.BLAS.set_num_threads(1)   # belt-and-braces; the caller also sets OPENBLAS_NUM_THREADS=1

# ---------------------------------------------------------------------------
# Cell U — univariate phylo mean+scale, Gaussian ML.
# ---------------------------------------------------------------------------

const TRUTH_U_BETA0 = 0.3
const TRUTH_U_BETA1 = 0.25
const TRUTH_U_SIGMA_PHY = 0.7      # RAW covariance scale — see the DGP note below
const TRUTH_U_SIGMA_RESID = 0.5

# Wald is restricted to the mu block ONLY (finding F2, pre-run §3): the
# sigma/resd axes have Inf stderror on this route -- (-Inf, Inf) silently.
# Excluded by design; never called, never emitted as a row.
const WALD_TARGETS_U = [(:mu, "(Intercept)"), (:mu, "x")]
const PROFILE_TARGETS_U =
    [(:mu, "(Intercept)"), (:mu, "x"), (:sigma, "(Intercept)"), (:resd, "species")]

function truth_lookup_u(param::Symbol, coefname::AbstractString)
    param === :mu && coefname == "(Intercept)" && return TRUTH_U_BETA0
    param === :mu && coefname == "x" && return TRUTH_U_BETA1
    param === :sigma && coefname == "(Intercept)" && return log(TRUTH_U_SIGMA_RESID)
    param === :resd && coefname == "species" && return log(TRUTH_U_SIGMA_PHY)
    error("no truth defined for Cell U target ($param, $coefname)")
end

# DGP spine (#454, reused not reinvented): simulate the phylo effect on the
# SAME raw covariance the fitter uses (`sigma_phy_dense`, diagonal = tree
# height h, NOT 1), scaled by sigma_phy. The fitter's own `resd` parameter is
# defined against that same raw covariance, so truth(resd) = log(sigma_phy)
# holds at every ntip -- h cancels because both sides of the comparison use
# the identical Q. h = 1.0 exactly at ntip=16 (depth 4, branch_length=0.25),
# 1.25 at ntip=32, 1.5 at ntip=64 (measured) -- the ntip=16 cell cannot tell
# a correct raw-scale draw from a wrong unit-height one; ntip=32/64 can.
function simulate_cell_u(seed::Integer, ntip::Integer, per::Integer)
    rng = Random.MersenneTwister(seed)
    phy = random_balanced_tree(ntip; branch_length = 0.25)
    Kraw = sigma_phy_dense(phy; σ²_phy = 1.0)
    Ltree = cholesky(Symmetric(Kraw)).L
    u = TRUTH_U_SIGMA_PHY .* (Ltree * randn(rng, ntip))
    species_idx = repeat(1:ntip, inner = per)
    n = ntip * per
    x = randn(rng, n)
    y = TRUTH_U_BETA0 .+ TRUTH_U_BETA1 .* x .+ u[species_idx] .+
        TRUTH_U_SIGMA_RESID .* randn(rng, n)
    species = phy.leaf_names[species_idx]
    return (; y, x, species, phy)
end

function fit_cell_u(dat)
    return drm(
        bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
        Gaussian();
        data = (; y = dat.y, x = dat.x, species = dat.species),
        tree = dat.phy,
    )
end

# ---------------------------------------------------------------------------
# Cell B — bivariate q4 phylo REML, fixture-sized (no ntip ladder).
# ---------------------------------------------------------------------------

const NTIP_B = 16     # matches test/parity/q4-reml/biv-q4-phylo-reml's n_tip
const PER_B = 8        # matches its n_each -> N = 128

const TRUTH_B_MU1 = (β0 = 0.5, β1 = 0.3)
const TRUTH_B_MU2 = (β0 = -0.2, β1 = 0.4)
const TRUTH_B_LOGSIGMA1 = -0.6
const TRUTH_B_LOGSIGMA2 = -0.6

# The among-axis (mu1, mu2, sigma1, sigma2) phylo covariance, taken from the
# fixture generator's fixed parameter set (test/parity/gen_biv_q4_phylo_reml.R,
# its `Lam` matrix) -- reused, not reinvented, per the pre-run's "truth taken
# from a fixed generator parameter set" instruction.
#
# JUDGEMENT CALL: the R generator draws U <- L %*% Z %*% t(chol(Lam)), which
# is R's chol() (upper, C'C = Lam) transposed and RIGHT-multiplied. Verified
# numerically here (2026-08-25, n = 2e6 draws) that this recipe's empirical
# column covariance is NOT Lam (e.g. Lam[1,1]=0.25 vs empirical 0.300,
# Lam[1,4]=0.00 vs empirical -5e-5..ok, but Lam[2,4]=0.04 vs empirical
# 0.0339 -- the whole matrix is off, not just noise). It does not matter for
# the fixture's own point-estimate parity check (which only compares
# DRModels.jl's refit of the data drmTMB itself refit -- no ground truth is
# invoked), but it matters here: coverage needs an honest truth. This runner
# instead draws U = Ltree * Z * Llam' with Llam = cholesky(Lam).L, which is
# the textbook matrix-normal identity Cov(vec(U)) = Lam ⊗ Kraw (re-verified
# the same way: empirical cov within Monte Carlo noise of Lam). Truth is
# `Lam` itself on DRModels.jl's raw covariance scale; at ntip=16 (this cell's
# fixed size) the balanced tree's height is exactly 1.0, so raw and
# correlation scale coincide and no rescale is needed either way.
const LAM_B = Symmetric([
    0.25 0.10 0.05 0.00
    0.10 0.25 0.00 0.04
    0.05 0.00 0.16 0.02
    0.00 0.04 0.02 0.16
])

# Column-major lower-triangle order, identical to `cov_to_lc`'s own
# convention (and to `DRModels`'s internal `_q4_phylocov_names()`, not called here
# since it is private -- these are just row labels).
const PHYLOCOV_NAMES_B = [
    "Sigma_a:L11", "Sigma_a:L21", "Sigma_a:L31", "Sigma_a:L41",
    "Sigma_a:L22", "Sigma_a:L32", "Sigma_a:L42",
    "Sigma_a:L33", "Sigma_a:L43",
    "Sigma_a:L44",
]
const PHYLOCOV_TRUTH_B = Dict(zip(PHYLOCOV_NAMES_B, cov_to_lc(Matrix(LAM_B))))

const WALD_TARGETS_B = [
    (:mu1, "(Intercept)"), (:mu1, "x"),
    (:mu2, "(Intercept)"), (:mu2, "x"),
    (:sigma1, "(Intercept)"), (:sigma2, "(Intercept)"),
    (:rho12, "(Intercept)"),
    [(:phylocov, nm) for nm in PHYLOCOV_NAMES_B]...,
]
# Profile is rho12 ONLY (finding F4: full-profile over all 17 targets is a
# measured runaway, >12 CPU-min/rep; per-target cost ~85 s).
const PROFILE_TARGETS_B = [(:rho12, "(Intercept)")]

function truth_lookup_b(param::Symbol, coefname::AbstractString)
    param === :mu1 && coefname == "(Intercept)" && return TRUTH_B_MU1.β0
    param === :mu1 && coefname == "x" && return TRUTH_B_MU1.β1
    param === :mu2 && coefname == "(Intercept)" && return TRUTH_B_MU2.β0
    param === :mu2 && coefname == "x" && return TRUTH_B_MU2.β1
    param === :sigma1 && coefname == "(Intercept)" && return TRUTH_B_LOGSIGMA1
    param === :sigma2 && coefname == "(Intercept)" && return TRUTH_B_LOGSIGMA2
    param === :rho12 && coefname == "(Intercept)" && return 0.0  # atanh(0); residual noise is drawn independently
    if param === :phylocov && haskey(PHYLOCOV_TRUTH_B, coefname)
        return PHYLOCOV_TRUTH_B[coefname]
    end
    error("no truth defined for Cell B target ($param, $coefname)")
end

function simulate_cell_b(seed::Integer)
    rng = Random.MersenneTwister(seed)
    phy = random_balanced_tree(NTIP_B; branch_length = 0.25)   # height == 1.0 exactly here
    Kraw = sigma_phy_dense(phy; σ²_phy = 1.0)
    Ltree = cholesky(Symmetric(Kraw)).L
    Llam = cholesky(LAM_B).L
    Z = randn(rng, NTIP_B, 4)
    U = Ltree * Z * transpose(Llam)     # Cov(vec(U)) = LAM_B ⊗ Kraw (verified, see note above)
    tip = repeat(1:NTIP_B, inner = PER_B)
    n = NTIP_B * PER_B
    x = randn(rng, n)
    y1 = TRUTH_B_MU1.β0 .+ TRUTH_B_MU1.β1 .* x .+ U[tip, 1] .+
         exp.(TRUTH_B_LOGSIGMA1 .+ U[tip, 3]) .* randn(rng, n)
    y2 = TRUTH_B_MU2.β0 .+ TRUTH_B_MU2.β1 .* x .+ U[tip, 2] .+
         exp.(TRUTH_B_LOGSIGMA2 .+ U[tip, 4]) .* randn(rng, n)
    species = phy.leaf_names[tip]
    return (; y1, y2, x, species, phy)
end

function fit_cell_b(dat)
    form = bf(
        mu1 = @formula(y1 ~ x + phylo(1 | species)),
        mu2 = @formula(y2 ~ x + phylo(1 | species)),
        sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
        sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
        rho12 = @formula(rho12 ~ 1),
    )
    return drm(
        form, Gaussian();
        data = (; y1 = dat.y1, y2 = dat.y2, x = dat.x, species = dat.species),
        tree = dat.phy, method = :REML, q4_vcov = true,   # F3: q4_vcov=false silently gives 17/17 (-Inf,Inf)
    )
end

# ---------------------------------------------------------------------------
# Cell-generic dispatch.
# ---------------------------------------------------------------------------

wald_targets(cell::Symbol) = cell === :U ? WALD_TARGETS_U : WALD_TARGETS_B
profile_targets(cell::Symbol) = cell === :U ? PROFILE_TARGETS_U : PROFILE_TARGETS_B
all_targets(cell::Symbol) =
    vcat([(p, c, "wald") for (p, c) in wald_targets(cell)],
         [(p, c, "profile") for (p, c) in profile_targets(cell)])
truth_lookup(cell::Symbol, param::Symbol, coefname::AbstractString) =
    cell === :U ? truth_lookup_u(param, coefname) : truth_lookup_b(param, coefname)
wald_parm(cell::Symbol) = cell === :U ? :mu : nothing
profile_parm(cell::Symbol) = cell === :U ? [:mu, :sigma, :resd] : :rho12

const Row = NamedTuple{
    (:seed, :conv, :param, :coef, :method, :estimate, :lower, :upper, :truth, :covered,
     :t_fit_s, :ntip, :cell, :jit),
    Tuple{Int,Int,Symbol,String,String,Float64,Float64,Float64,Float64,Int,Float64,Int,Symbol,Bool},
}

function mkrow(seed, conv, param, coefname, method, estimate, lower, upper, truth, covered,
               t_fit_s, ntip, cell, jit)
    return Row((seed, conv, param, String(coefname), method, Float64(estimate), Float64(lower),
                Float64(upper), Float64(truth), covered, Float64(t_fit_s), ntip, cell, jit))
end

# Run one replicate: simulate -> fit -> Wald -> profile, all through the
# PUBLIC API only (drm / confint) -- never a private refit. Every failure
# mode still emits the cell's full, fixed row set (never a silently dropped
# or silently NA row); only the numbers differ:
#   - fit throws:            all rows, conv=0, estimate/lower/upper=NaN, covered=0
#   - fit returns (converged or not): conv = fit.converged; Wald/profile each
#     attempted independently, and if ONE of the two throws, only THAT
#     method's rows fall back to NaN/covered=0 (the other method's rows, if
#     they succeeded, are kept).
function run_rep(cell::Symbol, seed::Integer, ntip::Integer, per::Integer, jit::Bool)
    dat = cell === :U ? simulate_cell_u(seed, ntip, per) : simulate_cell_b(seed)
    reported_ntip = cell === :U ? ntip : NTIP_B

    t_fit = NaN
    fit = nothing
    ok = true
    try
        t_fit = @elapsed begin
            fit = cell === :U ? fit_cell_u(dat) : fit_cell_b(dat)
        end
    catch err
        ok = false
        @warn "fit threw; recording as a failed rep (stays in the denominator, Williams 10b)" cell seed exception =
            (err, catch_backtrace())
    end

    if !ok
        return [
            mkrow(seed, 0, param, coefname, method, NaN, NaN, NaN,
                  truth_lookup(cell, param, coefname), 0, t_fit, reported_ntip, cell, jit)
            for (param, coefname, method) in all_targets(cell)
        ]
    end

    conv = Int(fit.converged)
    rows = Row[]

    try
        for r in confint(fit; method = :wald, parm = wald_parm(cell))
            tr = truth_lookup(cell, r.param, r.coef)
            if isfinite(r.lower) && isfinite(r.upper)
                covered = Int(r.lower <= tr <= r.upper)
                push!(rows, mkrow(seed, conv, r.param, r.coef, "wald", r.estimate, r.lower, r.upper,
                                   tr, covered, t_fit, reported_ntip, cell, jit))
            else
                # Per-coefficient boundary Inf stderror (stderror()'s own documented
                # behaviour, src/inference.jl): a non-finite bound trivially "covers"
                # any finite truth (-Inf <= tr <= Inf), which would silently inflate
                # coverage exactly like Cell U's excluded-by-design sigma/resd axes.
                # Cell B's phylocov axes are NOT excluded by design (q4_vcov=true only
                # guards the whole-vcov case, not a per-axis boundary) so this can occur
                # dynamically per seed; treat it as an interval failure (NaN, covered=0,
                # Williams 10b: stays in the denominator, never silently dropped).
                push!(rows, mkrow(seed, conv, r.param, r.coef, "wald", NaN, NaN, NaN,
                                   tr, 0, t_fit, reported_ntip, cell, jit))
            end
        end
    catch err
        @warn "wald confint threw; that method's rows marked failed (interval failed, not dropped)" cell seed exception =
            (err, catch_backtrace())
        for (param, coefname) in wald_targets(cell)
            push!(rows, mkrow(seed, conv, param, coefname, "wald", NaN, NaN, NaN,
                               truth_lookup(cell, param, coefname), 0, t_fit, reported_ntip, cell, jit))
        end
    end

    try
        for r in confint(fit; method = :profile, parm = profile_parm(cell))
            tr = truth_lookup(cell, r.param, r.coef)
            if isfinite(r.lower) && isfinite(r.upper)
                covered = Int(r.lower <= tr <= r.upper)
                push!(rows, mkrow(seed, conv, r.param, r.coef, "profile", r.estimate, r.lower, r.upper,
                                   tr, covered, t_fit, reported_ntip, cell, jit))
            else
                # Same guard the Wald branch above already carries, and it became
                # load-bearing here after DRModels.jl#493: a failed profile endpoint now
                # returns +/-Inf (rather than the old fabricated near-zero step), so
                # an unguarded `r.lower <= tr <= r.upper` would be TRIVIALLY TRUE and
                # would report ~100% coverage on exactly the reps whose endpoint
                # search failed -- inverting the signal the campaign exists to measure.
                # Recorded as an interval failure: stays in the denominator
                # (Williams 10b), never silently dropped and never counted as covered.
                push!(rows, mkrow(seed, conv, r.param, r.coef, "profile", NaN, NaN, NaN,
                                   tr, 0, t_fit, reported_ntip, cell, jit))
            end
        end
    catch err
        @warn "profile confint threw; that method's rows marked failed (interval failed, not dropped)" cell seed exception =
            (err, catch_backtrace())
        for (param, coefname) in profile_targets(cell)
            push!(rows, mkrow(seed, conv, param, coefname, "profile", NaN, NaN, NaN,
                               truth_lookup(cell, param, coefname), 0, t_fit, reported_ntip, cell, jit))
        end
    end

    return rows
end

# ---------------------------------------------------------------------------
# TSV I/O.
# ---------------------------------------------------------------------------

const HEADER =
    join(("seed", "conv", "param", "coef", "method", "estimate", "lower", "upper", "truth",
          "covered", "t_fit_s", "ntip", "cell", "jit"), '\t')

_fmt(x::Real) = isnan(x) ? "NaN" : (isfinite(x) ? @sprintf("%.6f", x) : (x > 0 ? "Inf" : "-Inf"))
_fmt_t(x::Real) = isnan(x) ? "NaN" : @sprintf("%.3f", x)

function tsv_line(r::Row)
    return join((
        string(r.seed), string(r.conv), string(r.param), r.coef, r.method,
        _fmt(r.estimate), _fmt(r.lower), _fmt(r.upper), _fmt(r.truth),
        string(r.covered), _fmt_t(r.t_fit_s), string(r.ntip), string(r.cell), string(Int(r.jit)),
    ), '\t')
end

# Seeds already fully present in an existing --out file for THIS (cell, ntip)
# (a whole rep is written and flushed atomically, so "present" == "done" -- no
# partial-rep corruption to guard against on restart). Keyed on (seed, cell,
# ntip), not seed alone: the ntip ladder (Cell U: 16/32/64) means the same
# seed number is legitimately reused across grid points, and if two shard
# commands ever point at the same --out path (a naming-template slip is not
# guarded against anywhere else), keying on seed alone would silently treat
# the second grid point's seeds as "already done" and skip them with no
# error -- exactly the kind of gap this campaign exists to avoid produce.
# Also warns (not errors -- an operator may deliberately share one file
# across a whole cell's ntip runs) if the file holds rows for a different
# cell than requested, since that combination is never expected by design.
function existing_seeds(path::AbstractString, cell::Symbol, ntip::Integer)
    seeds = Set{Int}()
    isfile(path) || return seeds
    other_cells = Set{Symbol}()
    open(path, "r") do io
        for (i, line) in enumerate(eachline(io))
            i == 1 && continue   # header
            isempty(line) && continue
            cols = split(line, '\t')
            row_seed = parse(Int, cols[1])
            row_ntip = parse(Int, cols[12])
            row_cell = Symbol(cols[13])
            row_cell == cell || push!(other_cells, row_cell)
            if row_cell === cell && row_ntip == ntip
                push!(seeds, row_seed)
            end
        end
    end
    isempty(other_cells) ||
        @warn "existing --out file also contains rows for a different cell; only this cell/ntip's seeds are treated as done" path cell ntip other_cells
    return seeds
end

# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

function parse_cli(args)
    opts = Dict{String,String}()
    smoke = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--smoke"
            smoke = true
            i += 1
        elseif startswith(a, "--")
            key = a[3:end]
            i + 1 <= length(args) || error("coverage_campaign: missing value for --$key")
            opts[key] = args[i + 1]
            i += 2
        else
            error("coverage_campaign: unrecognized argument '$a'")
        end
    end
    return opts, smoke
end

function main(args)
    opts, smoke = parse_cli(args)
    haskey(opts, "cell") || error("coverage_campaign: --cell U|B is required")
    cell = Symbol(opts["cell"])
    cell in (:U, :B) || error("coverage_campaign: --cell must be U or B (got '$(opts["cell"])')")

    ntip = parse(Int, get(opts, "ntip", "16"))
    per = parse(Int, get(opts, "per", "4"))

    if smoke
        seed = parse(Int, get(opts, "seed-start", "1"))
        t0 = time()
        rows = run_rep(cell, seed, ntip, per, true)
        wall = time() - t0
        println(HEADER)
        for r in rows
            println(tsv_line(r))
        end
        @info "smoke done" cell seed nrows = length(rows) wall_s = round(wall; digits = 3)
        return
    end

    haskey(opts, "seed-start") || error("coverage_campaign: --seed-start is required (or pass --smoke)")
    haskey(opts, "seed-count") || error("coverage_campaign: --seed-count is required (or pass --smoke)")
    haskey(opts, "out") || error("coverage_campaign: --out is required (or pass --smoke)")
    seed_start = parse(Int, opts["seed-start"])
    seed_count = parse(Int, opts["seed-count"])
    seed_count >= 1 || error("coverage_campaign: --seed-count must be >= 1")
    out = opts["out"]

    reported_ntip = cell === :U ? ntip : NTIP_B
    seeds = seed_start:(seed_start + seed_count - 1)
    done = existing_seeds(out, cell, reported_ntip)
    todo = [s for s in seeds if !(s in done)]
    @info "coverage_campaign starting" cell ntip = (cell === :U ? ntip : NTIP_B) per =
        (cell === :U ? per : PER_B) n_total = length(seeds) n_done = length(seeds) - length(todo) n_todo =
        length(todo) out

    isempty(todo) && (@info "nothing to do; all requested seeds already in $out"; return)

    outdir = dirname(out)
    isempty(outdir) || mkpath(outdir)
    write_header = !isfile(out) || filesize(out) == 0

    open(out, "a") do io
        write_header && println(io, HEADER)
        jit = true   # first fit this PROCESS executes pays JIT compilation
        for seed in todo
            rows = run_rep(cell, seed, ntip, per, jit)
            jit = false
            for r in rows
                println(io, tsv_line(r))
            end
            flush(io)
        end
    end
    @info "coverage_campaign done" cell out n_ran = length(todo)
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
