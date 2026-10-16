# test_1088_nb2_crossed_runoff.jl -- #1088: NB2 crossed random intercepts `(1 | g) + (1 | h)`
# at heavy overdispersion (size ~0.002): the outer optimiser ran off past the clamps.
#
# Mechanism (the issue's three sets, Julia 1.12.6): the Method-of-Moments start caps the NB2
# size at 0.5 (log sigma 0.35) while the MLE is ~0.0025 (log sigma ~3), so the start gradient
# is ~3e3 in log sigma and ~6e2 in each log SD. LBFGS's first, unscaled step lands at
# log sigma = 3021.5 and log SDs 616 / 687, past every clamp. The objective there (1034.05) is
# flat but still below the start's (2979.8), so the line search accepts the step; the analytic
# gradient there is not zero, so the next line search fails. PR #1075's head `740521f4` returns
# that point with converged = false.
#
# Fix (`_fit_crossed_mean_laplace_nuisance`): a first pass that ends past an UPPER clamp is
# re-fitted inside the clamp box (`_crossed_nuisance_box_refit`) and the re-fit is adopted
# only if it strictly lowers the objective. A fit that still ends past an upper clamp warns
# and is marked not converged (`_nuisance_runoff`, `_warn_nuisance_runoff`).
#
# Reference: native drmTMB ca3ddef (0.7.1), one #1073 comparison row per set, written by
# test/parity/gen_1088_nb2_runoff_reference.R into test/fixtures/j3_se/<route>/ref.toml
# (opt$convergence, sdr$pdHess, fitted log-dispersion, opt$par, sdreport SEs, logLik) from
# the data written by test/parity/gen_1088_nb2_runoff_data.jl.
using DRModels
using Test, TOML, LinearAlgebra, Logging

const _R1088_DIR = joinpath(@__DIR__, "fixtures", "j3_se")
const _R1088_DRMTMB = "ca3ddefa5ea11bbfd3b765615dfbb89305d2485a"

function _r1088_data(route)
    lines = readlines(joinpath(_R1088_DIR, route, "data.csv"))
    rows = [split(l, ',') for l in lines[2:end]]
    return (; y = [parse(Float64, r[1]) for r in rows], x = [parse(Float64, r[2]) for r in rows],
              g = String[r[3] for r in rows], h = String[r[4] for r in rows])
end
_r1088_ref(route) = TOML.parsefile(joinpath(_R1088_DIR, route, "ref.toml"))
_r1088_fit(d) = drm(bf(@formula(y ~ 1 + x + (1 | g) + (1 | h)), @formula(sigma ~ 1)),
                    NegBinomial2(); data = d)

# One #1073 comparison row per set, printed at the end of this file.
const _R1088_ROWS = NamedTuple[]

@testset "#1088: NB2 crossed fit reaches drmTMB's point (size ~0.002)" begin
    for route in ("nb2_runoff_s7_n1500", "nb2_runoff_s8_n1500", "nb2_runoff_s7_n800")
        ref = _r1088_ref(route)
        fit = _r1088_fit(_r1088_data(route))
        θ = fit.theta
        θt = Float64.(ref["theta"])
        # A collapsed log SD sends this route to the value-based Hessian, whose entry for that
        # coordinate is noise-dominated and BLAS-kernel dependent; a negative diagonal there
        # must read as "no SE", not throw.
        se = [v >= 0 ? sqrt(v) : NaN for v in diag(vcov(fit))]
        se_t = Float64.(ref["se"])
        # SEs are compared only on coordinates that are not collapsed on either side.
        live = [k for k in eachindex(θt) if k <= 3 || (θt[k] > -8.0 && θ[k] > -8.0 + 1e-3)]
        se_ok = ref["comparable"] && all(isfinite, se[live])
        push!(_R1088_ROWS, (route = route, drmTMB_commit = ref["drmTMB_commit"],
                            opt_convergence = ref["opt_convergence"], pd_hess = ref["pd_hess"],
                            logdisp_drmTMB = ref["log_dispersion"], logdisp_DRModels = θ[3],
                            nll_drmTMB = -ref["loglik"], nll_DRModels = -loglik(fit),
                            converged = is_converged(fit),
                            status = ref["point_comparable"] ? "optimum compared" :
                                     "not comparable (drmTMB opt\$convergence = $(ref["opt_convergence"]))",
                            max_rel_se = se_ok ? maximum(abs.(se[live] ./ se_t[live] .- 1)) : NaN))
        @testset "$route" begin
            @test ref["drmTMB_commit"] == _R1088_DRMTMB
            @test ref["log_dispersion"] == θt[3]
            # Inside the box and at a real optimum (base: log sigma 3021.5 / 2902.0 / 1542.6).
            @test θ[3] < DRModels._dispersion_clamp(Val(:nb2_fixed))
            @test all(<(3.0), θ[4:5])
            @test is_converged(fit)
            if ref["point_comparable"]
                # drmTMB opt$convergence == 0: its optimum is the reference.
                @test maximum(abs.(θ[1:3] .- θt[1:3])) < 2e-4        # beta and log-dispersion
                for k in 4:5
                    if θt[k] < -8.0
                        # collapsed in drmTMB (-64.4 / -8.34 / -13.9): at DRModels' floor too
                        @test θ[k] <= -8.0 + 1e-3
                    else
                        @test abs(θ[k] - θt[k]) < 2e-4
                    end
                end
                @test loglik(fit) ≈ ref["loglik"] atol = 1e-5
            else
                # drmTMB code 1 (singular convergence): its point is not an optimum and is not
                # a pass/fail reference. DRModels must not end worse than it.
                @test -loglik(fit) <= -ref["loglik"] + 1e-6
            end
            if se_ok
                # A collapsed log SD sends this route to the value-based (known-noisy) Hessian,
                # so only a loose band is asserted (the J3 convention for that path).
                @test all(abs.(se[live] ./ se_t[live] .- 1) .< 0.10)
            end
        end
    end
end

@testset "#1088: a fit still past an upper clamp warns and is not converged" begin
    pμ = 2
    b = DRModels._dispersion_clamp(Val(:nb2_fixed))
    # base's end point on nb2_runoff_s7_n1500
    hits = DRModels._nuisance_runoff([-222.18, -24.367, 3021.5, 616.16, 686.82], pμ, b)
    @test length(hits) == 3
    @test DRModels._nuisance_runoff([3.862, 2.160, 3.0036, -64.4, -8.34], pμ, b) === nothing
    # the lower side is a legitimate boundary (Poisson limit, variance collapse), not a run-off
    @test DRModels._nuisance_runoff([0.1, 0.2, -6.0, -12.0, -30.0], pμ, b) === nothing
    @test DRModels._nuisance_runoff([0.1, 0.2, 3.9999, 2.9999, 0.0], pμ, b) === nothing
    @test DRModels._nuisance_runoff([0.1, 0.2, NaN, 0.0, 0.0], pμ, b) !== nothing
    @test DRModels._nuisance_runoff([0.1, 0.2, 5.0, 0.0, 0.0], pμ,
                                    DRModels._dispersion_clamp(Val(:tweedie_fixed))) === nothing
    @test_logs (:warn, r"ended past an upper clamp") DRModels._warn_nuisance_runoff("test", hits)
    @test_logs DRModels._without_boundary_warnings(() -> DRModels._warn_nuisance_runoff("test", hits))

    # End to end: with the re-fit switched off, the first pass reproduces the issue's end point,
    # and the guard flags it instead of returning it as a fit.
    d = _r1088_data("nb2_runoff_s7_n1500")
    n = length(d.y)
    gidx, G = DRModels._group_index(d.g)
    hidx, H = DRModels._group_index(d.h)
    comps = [(ones(n), gidx, G, "g"), (ones(n), hidx, H, "h")]
    f0 = @test_logs (:warn, r"ended past an upper clamp") match_mode = :any DRModels._fit_nb2_crossed_laplace(
        NegBinomial2(), d.y, hcat(ones(n), d.x), ones(n, 1), comps, ["(Intercept)", "x"],
        ["(Intercept)"], 1e-8; recover_runoff = false)
    @test !f0.converged
    @test f0.theta[3] > 100                          # 3021.5 on Julia 1.12.6
    @test -f0.loglik > -_r1088_ref("nb2_runoff_s7_n1500")["loglik"] + 100   # 1034.05 vs 426.15
end

println("\n#1088 / #1073 comparison rows (drmTMB ", first(_R1088_DRMTMB, 7), "):")
println("| route | drmTMB commit | opt\$convergence | sdr\$pdHess | log sigma drmTMB | log sigma DRModels | nll drmTMB | nll DRModels | converged | status | max rel. SE diff |")
println("|---|---|---|---|---|---|---|---|---|---|---|")
for r in _R1088_ROWS
    println("| ", r.route, " | `", first(r.drmTMB_commit, 7), "` | ", r.opt_convergence, " | ", r.pd_hess, " | ",
            round(r.logdisp_drmTMB; digits = 5), " | ", round(r.logdisp_DRModels; digits = 5), " | ",
            round(r.nll_drmTMB; digits = 6), " | ", round(r.nll_DRModels; digits = 6), " | ", r.converged, " | ",
            r.status, " | ", isnan(r.max_rel_se) ? "n/a" : string(round(r.max_rel_se; sigdigits = 2)), " |")
end
