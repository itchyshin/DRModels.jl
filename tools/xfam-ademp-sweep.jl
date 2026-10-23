#!/usr/bin/env julia
#
# xfam-ademp-sweep.jl — ADEMP recovery + profile-CI coverage sweep for DRModels.jl's
# CROSS-FAMILY bivariate model (`DRModels.fit_mixed_family`, shared-latent GHQ).
#
# ADEMP (Morris, White & Crowther 2019):
#   Aims       Quantify, per cross-family pair, whether `fit_mixed_family`
#              recovers the latent-scale correlation ρ (low bias, calibrated CI).
#   Data-gen.  Shared per-observation latent  u_i ~ N(0,1);  η_k = X_k β_k + λ_k u,
#              y_k ~ fam_k(η_k) via DRModels's OWN per-family sampler `_mf_rand` (the
#              exact code path the parametric bootstrap uses). n ≈ 500, p = 1
#              covariate + intercept on each axis. Everything seeded.
#   Estimand   The engine-consistent latent-scale correlation
#                  ρ = λ1 λ2 / sqrt((λ1²+v1)(λ2²+v2)),  v_k = link_residual(fam_k, …),
#              evaluated at the TRUE parameters on the realised design — i.e. the
#              population value `fit_mixed_family.rho_latent` is consistent for
#              under this engine's reporting convention (`rho_of` at θ_true). This
#              makes profile-CI coverage well-posed (we cover the value the point
#              estimator targets, not a moment-ρ from a different convention).
#   Methods    `fit_mixed_family(...; profile=true)` — the shared-latent GHQ MLE
#              and its profile-likelihood CI on ρ (the recommended interval).
#   Performance  Per cell: convergence rate; ρ bias, empirical SD, RMSE; and
#              profile-CI coverage of the true ρ with a Wilson 95% interval.
#
# BUDGET NOTE (honest reporting, measured on Julia 1.10, this machine):
#   The POINT fit is cheap (~0.3–0.7 s); bias/RMSE/convergence run at the full
#   rep count for ALL six pairs. The PROFILE-CI is the bottleneck and its cost is
#   wildly heterogeneous — ~10 s (Gaussian×Poisson) to ~125 s (NB2×Gaussian) and
#   ~375 s (Gamma×Poisson) PER replicate, with a heavy right tail. A few-minutes
#   budget therefore cannot deliver a fixed, large coverage rep count for the slow
#   pairs. We instead run coverage under a per-pair WALL-CLOCK CAP: each pair gets
#   as many seeded profile reps as fit under its time budget, and we report the
#   ACHIEVED rep count + Monte-Carlo (Wilson) uncertainty. Fast pairs accrue a
#   usable coverage estimate; slow pairs get a small, explicitly low-precision one
#   (or none if even one rep would blow the cap). Nothing is extrapolated.
#
#   Run from the repo root:
#       julia --project=. tools/xfam-ademp-sweep.jl
#   Optional env overrides (defaults keep the total to a few minutes):
#       XFAM_NPOINT=100   XFAM_COVCAP=60   XFAM_N=500   XFAM_SEED=20260610

using DRModels
using Random, Statistics, Printf
using SpecialFunctions: trigamma
import DRModels: _mf_rand, _mf_mean, link_residual

# ----------------------------------------------------------------------------
# Config (seeded; env-overridable so the cost can be tuned without code edits).
# ----------------------------------------------------------------------------
const N        = parse(Int,     get(ENV, "XFAM_N",      "500"))   # obs / replicate
const N_POINT  = parse(Int,     get(ENV, "XFAM_NPOINT", "60"))    # reps for bias/RMSE (point fit; cheap)
const COV_CAP  = parse(Float64, get(ENV, "XFAM_COVCAP", "90.0"))  # default wall-clock cap (s) / coverage cell
const COV_MAX  = parse(Int,     get(ENV, "XFAM_COVMAX", "40"))    # hard rep ceiling / coverage cell
# Per-pair coverage wall-clock caps (s). Gaussian×Poisson — the flagship, fastest,
# fully-identified pair — gets the largest cap so its coverage reaches a genuinely
# usable rep count (the headline number); the others get a smaller supporting cap.
# Falls back to COV_CAP for any pair not listed.
const COV_CAP_BY_PAIR = Dict(
    "Gaussian x Poisson"  => 200.0,
    "Gaussian x Binomial" => 100.0,
    "Poisson x Binomial"  => 100.0,
    "Beta x Binomial"     => 70.0,
)
# Profile coverage is the budget bottleneck (~10 s/rep minimum). To get a USABLE
# rep count per cell within a few minutes we run coverage at ONE representative ρ
# (the "moderate" loading) per eligible pair, rather than spreading a tiny n over
# both ρ values. Set XFAM_COVALL=1 to attempt coverage at every loading (much slower).
const COV_ONLY_MODERATE = get(ENV, "XFAM_COVALL", "0") != "1"
const BASESEED = parse(Int,     get(ENV, "XFAM_SEED",   "20260610"))
const LEVEL    = 0.95
const K_GHQ    = 32                                               # GHQ nodes (fit default)

# Fixed true fixed-effects (kept modest so means stay in a numerically benign
# range for every link). Same on both axes; the covariate is standard normal.
const Β1 = [0.4, 0.5]
const Β2 = [0.3, -0.4]
const NTRIALS = 12.0      # Binomial denominator (when an axis is Binomial)

# Per-family NATURAL dispersion used by the sampler `_mf_rand` (σ for Gaussian/
# Beta/Gamma, size θ for NB2; ignored by Poisson/Binomial, passed as 1.0).
nat_disp(::Gaussian)     = 0.5
nat_disp(::Beta)         = 0.35
nat_disp(::Gamma)        = 0.40
nat_disp(::NegBinomial2) = 6.0
nat_disp(::Poisson)      = 1.0
nat_disp(::Binomial)     = 1.0

has_disp(f) = !(f isa Poisson || f isa Binomial)

# The six requested cross-family pairs. `prof` = profile-coverage eligible.
# The two pairs flagged false are EXCLUDED from profile coverage on the basis of
# MEASURED per-replicate profile-CI cost on this machine/Julia (calibration runs):
#     Gamma x Poisson  ≈ 375 s / rep   NB2 x Gaussian ≈ 125 s / rep
# At those costs even ONE profile rep blows a few-minutes budget (the profile
# bisection re-optimises the full model ~80× at g_tol=1e-9, and the Gamma/NB2
# loggamma kernels under ForwardDiff are slow with a flat-likelihood tail). They
# still get FULL bias/RMSE/convergence from the cheap point fit; their profile
# coverage is reported as "out of budget" with the measured cost as evidence —
# never extrapolated. Flip `prof` to true (and raise XFAM_COVCAP) to include them.
const PAIRS = [
    ("Gaussian x Poisson",   Gaussian(),     Poisson(),      true),
    ("Gaussian x Binomial",  Gaussian(),     Binomial(),     true),
    ("Poisson x Binomial",   Poisson(),      Binomial(),     true),
    ("NB2 x Gaussian",       NegBinomial2(), Gaussian(),     false),
    ("Gamma x Poisson",      Gamma(),        Poisson(),      false),
    ("Beta x Binomial",      Beta(),         Binomial(),     true),
]

# Two true-ρ targets per pair. ρ is set by the loadings (λ1,λ2) given the families'
# link-scale residual variances. We hold λ_k equal across a pair (λ1=λ2=λ) and pick
# two λ values giving a "moderate" and a "strong" latent correlation; the exact
# realised true ρ is computed per design (see `true_rho`).
const LOADINGS = [("moderate", 0.55), ("strong", 0.90)]   # λ1 = λ2 = λ

# ----------------------------------------------------------------------------
# Estimand: the engine-consistent latent-scale ρ at the TRUE parameters, using the
# SAME representative-mean + link_residual mapping `fit_mixed_family.rho_of` uses
# (mean of the response-scale mean at u=0, i.e. mean(_mf_mean(fam, Xβ))). This is
# the population value rho_latent is consistent for, hence the coverage target.
# ----------------------------------------------------------------------------
disp_v(f::Gaussian,     σ, μ̄) = link_residual(f;    dispersion = σ^2)        # σ²
disp_v(f::Gamma,        σ, μ̄) = link_residual(f;    dispersion = σ^2)        # trigamma(1/σ²)
disp_v(f::Beta,         σ, μ̄) = link_residual(f, μ̄; dispersion = inv(σ^2))   # φ = 1/σ²
disp_v(f::NegBinomial2, σ, μ̄) = link_residual(f;    dispersion = σ)          # θ
disp_v(f::Poisson,      σ, μ̄) = link_residual(f, μ̄)
disp_v(f::Binomial,     σ, μ̄) = link_residual(f, μ̄)

function true_rho(fam1, fam2, X1, X2, λ1, λ2)
    μ̄1 = mean(_mf_mean.(Ref(fam1), X1 * Β1))
    μ̄2 = mean(_mf_mean.(Ref(fam2), X2 * Β2))
    v1 = disp_v(fam1, nat_disp(fam1), μ̄1)
    v2 = disp_v(fam2, nat_disp(fam2), μ̄2)
    return λ1 * λ2 / sqrt((λ1^2 + v1) * (λ2^2 + v2))
end

# ----------------------------------------------------------------------------
# One replicate's DATA: shared latent + per-family sampler. Fully seeded by `rep`.
# ----------------------------------------------------------------------------
function gen_data(fam1, fam2, λ1, λ2, rep::Int)
    rng = MersenneTwister(BASESEED + 1009 * rep)   # per-rep data stream
    x   = randn(rng, N)
    X1  = hcat(ones(N), x)
    X2  = hcat(ones(N), x)
    u   = randn(rng, N)
    η1  = X1 * Β1 .+ λ1 .* u
    η2  = X2 * Β2 .+ λ2 .* u
    d1  = nat_disp(fam1); d2 = nat_disp(fam2)
    tr  = fill(NTRIALS, N)
    y1  = [_mf_rand(fam1, η1[i], tr[i], d1, rng) for i in 1:N]
    y2  = [_mf_rand(fam2, η2[i], tr[i], d2, rng) for i in 1:N]
    return (; X1, X2, y1, y2, tr)
end

# Point fit (no CI) — cheap; used for convergence + bias + RMSE.
function fit_point(fam1, fam2, d)
    DRModels.fit_mixed_family(; y1 = d.y1, X1 = d.X1, fam1 = fam1,
                           y2 = d.y2, X2 = d.X2, fam2 = fam2,
                           trials1 = d.tr, trials2 = d.tr,
                           K = K_GHQ, confint = false)
end

# Profile fit (CI only on ρ) — expensive; used for coverage.
function fit_profile(fam1, fam2, d)
    DRModels.fit_mixed_family(; y1 = d.y1, X1 = d.X1, fam1 = fam1,
                           y2 = d.y2, X2 = d.X2, fam2 = fam2,
                           trials1 = d.tr, trials2 = d.tr,
                           K = K_GHQ, confint = false, profile = true)
end

# Wilson score 95% interval for a binomial proportion (coverage MC error).
function wilson(k::Int, n::Int; z = 1.959963984540054)
    n == 0 && return (NaN, NaN)
    p̂ = k / n
    den = 1 + z^2 / n
    centre = (p̂ + z^2 / (2n)) / den
    halfw  = z * sqrt(p̂ * (1 - p̂) / n + z^2 / (4n^2)) / den
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))
end

# ----------------------------------------------------------------------------
# Sweep.
# ----------------------------------------------------------------------------
struct Cell
    pair::String
    loadlabel::String
    λ::Float64
    ρtrue::Float64
    nconv::Int
    nrep::Int
    bias::Float64
    emp_sd::Float64
    rmse::Float64
    cov::Float64          # profile-CI coverage (NaN if no coverage reps ran)
    cov_lo::Float64
    cov_hi::Float64
    cov_n::Int            # achieved profile-coverage reps
    cov_t::Float64        # wall-clock spent on coverage for this cell (s)
    cov_eligible::Bool    # was this pair attempted for profile coverage at all?
end

function run_point_stats(fam1, fam2, λ, ρtrue)
    ρhat = Float64[]
    nconv = 0
    for rep in 1:N_POINT
        d = gen_data(fam1, fam2, λ, λ, rep)
        f = try
            fit_point(fam1, fam2, d)
        catch
            nothing
        end
        f === nothing && continue
        f.converged && (nconv += 1)
        isfinite(f.rho_latent) && push!(ρhat, f.rho_latent)
    end
    n = length(ρhat)
    if n == 0
        return (nconv, 0, NaN, NaN, NaN)
    end
    bias   = mean(ρhat) - ρtrue
    emp_sd = n > 1 ? std(ρhat) : NaN
    rmse   = sqrt(mean((ρhat .- ρtrue) .^ 2))
    return (nconv, n, bias, emp_sd, rmse)
end

# Wall-clock-capped profile coverage. Runs seeded reps (reusing the SAME data
# streams as the point stage) until either COV_MAX reps or the per-pair time cap
# is reached; returns coverage k/n + achieved n + elapsed.
function run_coverage(fam1, fam2, λ, ρtrue, cap_s::Float64)
    covered = 0
    n = 0
    t0 = time()
    for rep in 1:COV_MAX
        # stop if even an *average*-cost next rep would breach the cap (use the
        # running mean rep cost; first rep always runs so the cap can't zero-out).
        if n > 0
            mean_cost = (time() - t0) / n
            (time() - t0) + mean_cost > cap_s && break
        end
        d = gen_data(fam1, fam2, λ, λ, rep)
        f = try
            fit_profile(fam1, fam2, d)
        catch
            nothing
        end
        if f !== nothing
            lo, hi = f.rho_ci_profile
            if isfinite(lo) && isfinite(hi)
                n += 1
                (lo <= ρtrue <= hi) && (covered += 1)
            end
        end
        # hard stop the instant we've exceeded the cap (protects against one slow rep)
        (time() - t0) > cap_s && break
    end
    elapsed = time() - t0
    cov = n == 0 ? NaN : covered / n
    lo, hi = wilson(covered, n)
    return (cov, lo, hi, n, elapsed, covered)
end

function main()
    println("# DRModels.jl cross-family ADEMP recovery + profile-coverage sweep")
    println("# n=$N  point-reps=$N_POINT  cov-cap=$(COV_CAP)s/pair  cov-max=$COV_MAX  seed=$BASESEED  level=$LEVEL")
    println("# Julia $(VERSION)")
    # The profile budget is split: the two known-slow pairs (Gamma×P ~375 s/rep,
    # NB2×G ~125 s/rep) get the same wall-clock cap, but at those costs the cap
    # admits 0–1 reps — reported honestly as low/zero-precision rather than skipped
    # silently. Fast pairs accrue many reps under the same cap.
    cells = Cell[]
    t_start = time()
    for (pname, fam1, fam2, prof_ok) in PAIRS
        for (llabel, λ) in LOADINGS
            # ρ_true on a representative realised design (rep=1) so the mean-
            # dependent link_residual (Poisson/Beta) uses the actual covariate.
            d1 = gen_data(fam1, fam2, λ, λ, 1)
            ρt = true_rho(fam1, fam2, d1.X1, d1.X2, λ, λ)

            @printf("\n[%s | %s λ=%.2f]  ρ_true=%.4f\n", pname, llabel, λ, ρt)
            tp = time()
            nconv, nrep, bias, emp_sd, rmse = run_point_stats(fam1, fam2, λ, ρt)
            @printf("  point: conv=%d/%d  used=%d  bias=%+.4f  emp_sd=%.4f  rmse=%.4f  (%.1fs)\n",
                    nconv, N_POINT, nrep, bias, emp_sd, rmse, time() - tp)

            run_cov_here = prof_ok && (!COV_ONLY_MODERATE || llabel == "moderate")
            if run_cov_here
                cap = get(COV_CAP_BY_PAIR, pname, COV_CAP)
                cov, clo, chi, cn, ct, ck = run_coverage(fam1, fam2, λ, ρt, cap)
                if cn == 0
                    @printf("  cover: NO profile reps fit under %.0fs cap (per-rep cost > cap)\n", cap)
                else
                    @printf("  cover: %d/%d = %.3f  [%.3f, %.3f] (Wilson95)  reps=%d  (%.1fs)\n",
                            ck, cn, cov, clo, chi, cn, ct)
                end
                push!(cells, Cell(pname, llabel, λ, ρt, nconv, nrep, bias, emp_sd, rmse,
                                  cov, clo, chi, cn, ct, true))
            elseif !prof_ok
                @printf("  cover: SKIPPED (profile-CI out of budget; measured ~%s/rep)\n",
                        startswith(pname, "Gamma") ? "375 s" : "125 s")
                push!(cells, Cell(pname, llabel, λ, ρt, nconv, nrep, bias, emp_sd, rmse,
                                  NaN, NaN, NaN, 0, 0.0, false))
            else
                @printf("  cover: (coverage measured at the moderate-ρ cell only)\n")
                push!(cells, Cell(pname, llabel, λ, ρt, nconv, nrep, bias, emp_sd, rmse,
                                  NaN, NaN, NaN, 0, 0.0, true))
            end
        end
    end
    @printf("\n# total wall-clock: %.1fs\n", time() - t_start)

    # Emit a machine-readable block the report builder consumes.
    println("\n===CELLS-TSV===")
    println(join(["pair","load","lambda","rho_true","nconv","npoint","nrep",
                  "bias","emp_sd","rmse","cov","cov_lo","cov_hi","cov_n","cov_t","cov_eligible"], "\t"))
    for c in cells
        println(join((c.pair, c.loadlabel, @sprintf("%.2f", c.λ), @sprintf("%.4f", c.ρtrue),
                      c.nconv, N_POINT, c.nrep,
                      @sprintf("%.4f", c.bias), @sprintf("%.4f", c.emp_sd), @sprintf("%.4f", c.rmse),
                      isnan(c.cov) ? "NA" : @sprintf("%.3f", c.cov),
                      isnan(c.cov_lo) ? "NA" : @sprintf("%.3f", c.cov_lo),
                      isnan(c.cov_hi) ? "NA" : @sprintf("%.3f", c.cov_hi),
                      c.cov_n, @sprintf("%.1f", c.cov_t), c.cov_eligible), "\t"))
    end
    println("===END-CELLS-TSV===")
end

main()
