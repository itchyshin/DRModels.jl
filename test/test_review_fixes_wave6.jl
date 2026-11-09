# Wave-6 adversarial-review fixes (2026-11-08). One testset per fix.
using DRModels, Test, LinearAlgebra, DelimitedFiles, Logging, StableRNGs
import Distributions

const _FX6 = joinpath(@__DIR__, "fixtures", "entry_robust")
_quiet6(f) = Logging.with_logger(f, Logging.NullLogger())

@testset "(a) information criteria and LR guards share loglik()'s failed-fit rule" begin
    g, _ = readdlm(joinpath(_FX6, "gaussian_q2_x1e6_data.csv"), ','; header = true)
    hdr = vec(readdlm(joinpath(_FX6, "gaussian_q2_x1e6_data.csv"), ','; header = true)[2])
    col(n) = g[:, findfirst(==(n), String.(hdr))]
    dq = (; y = Float64.(col("y")), x = Float64.(col("x")), z0 = Float64.(col("z0")),
          z1 = Float64.(col("z1")), g = string.(col("g")))
    fq = _quiet6(() -> drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1),
                             @formula(sd(g, coef(intercept)) ~ z0), @formula(sd(g, coef(x)) ~ z1)),
                          Gaussian(); data = dq))
    @test fq.converged == false
    @test abs(fq.loglik) >= 1e15                       # the runaway +3.6e76 stored value
    for f in (aic, bic, aicc, deviance)
        @test isnan(_quiet6(() -> f(fq)))              # never a finite, best-looking value
    end
    ok = _quiet6(() -> drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = dq))
    ok2 = _quiet6(() -> drm(bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = dq))
    @test_throws ArgumentError lrtest(ok2, fq)
    @test_throws ArgumentError lrt_boundary(fq, ok2)
    @test_throws ArgumentError lrtest(fq, ok2)
end

@testset "(b) unlabelled A / coords warning: once per bootstrap_ci call, every ordinary drm() call" begin
    G, m = 6, 5
    rng = StableRNG(11)
    g = repeat(["a$i" for i in 1:G], inner = m)
    x = randn(rng, G * m)
    y = 1 .+ 0.4 .* x .+ repeat(randn(rng, G), inner = m) .+ 0.5 .* randn(rng, G * m)
    d = (; y, x, g)
    R0 = [0.4^abs(i - j) for i in 1:G, j in 1:G]
    f = bf(@formula(y ~ x + animal(1 | g)), @formula(sigma ~ 1))
    nlab(logs) = count(l -> occursin("has no labels", l.message), logs)
    # ordinary calls: one warning each
    for _ in 1:2
        logs, _ = Test.collect_test_logs(() -> drm(f, Gaussian(); data = d, A = R0))
        @test nlab(logs) == 1
    end
    # bootstrap_ci: the base fit warns once, the B refits stay quiet
    B = 4
    logs, _ = Test.collect_test_logs(() -> bootstrap_ci(f, Gaussian(); data = d, A = R0, B,
                                                      rng = StableRNG(3), failures = :skip, check_converged = false))
    @test nlab(logs) == 1
end

function _refusal6(f, fam, d; kw...)
    try
        drm(f, fam; data = d, kw...)
    catch e
        e isa ArgumentError && return e.msg
        rethrow()
    end
    return ""
end

@testset "(c) the drmTMB note only claims what the refusal-gap map shows" begin
    rng = StableRNG(21)
    n = 60
    g = repeat(["a$i" for i in 1:6], inner = 10)
    x = randn(rng, n)
    y = randn(rng, n)
    d = (; y, x, g)
    f = bf(@formula(y ~ x + (1 | g)))
    msg = _refusal6(f, Student(), d; tree = 1)
    @test occursin("does not accept the keyword `tree`", msg)
    @test !occursin("refuses this model too", msg)              # the old, over-broad claim
    @test occursin("refused all 50", msg)                       # what the grid shows
    @test occursin("nu ~ phylo(1 | id, tree = tree)", msg)      # drmTMB does fit this Student model
    @test occursin("not verified against drmTMB", msg)
    # a family with R fits in the grid says so, with the count
    mlog = _refusal6(bf(@formula(y ~ x + (1 | g))), DRModels.LogNormal(), d; coords = 1)
    @test occursin("refused all 50", mlog)
    @test occursin("fitted 3 of the 50", DRModels._kw_r_note(:tree, DRModels.Tweedie()))
    @test occursin("fitted 2 of the 50", DRModels._kw_r_note(:coords, Student()))
    # a family/keyword outside the grid makes no claim about drmTMB
    @test DRModels._kw_r_note(:tree, Gaussian()) == " Not verified against drmTMB for this model."
end

@testset "(d) se= refusal names control = DRMControl(se = false); accepted lists carry control and weights" begin
    rng = StableRNG(22)
    n = 60
    g = repeat(["a$i" for i in 1:6], inner = 10)
    x = randn(rng, n)
    y = randn(rng, n)
    d = (; y, x, g)
    msg = _refusal6(bf(@formula(y ~ x + (1 | g))), Gaussian(), d; se = false)
    @test occursin("does not accept the keyword `se`", msg)
    @test occursin("control = DRMControl(se = false)", msg)
    @test occursin("works on every route", msg)
    acc = last(split(msg, "Keywords this route accepts: "))
    @test occursin("`control`", acc) && occursin("`weights`", acc)
    # derived from the signatures: a splat route and a fixed-effect route agree
    msg2 = _refusal6(bf(@formula(y ~ x + (1 | g))), Student(), d; tree = 1)
    acc2 = last(split(msg2, "Keywords this route accepts: "))
    @test occursin("`control`", acc2) && occursin("`weights`", acc2)
    # the advice is true: control = DRMControl(se = false) fits where `se = false` was refused
    fit = _quiet6(() -> drm(bf(@formula(y ~ x + (1 | g))), Gaussian(); data = d, control = DRMControl(se = false)))
    @test fit isa DRModels.DrmFit
end

using SparseArrays

# (e1) noise floor of the Binomial animal q2 inner mode solve. Synthetic near-singular joint
# precision (gamma = atanh(rho) = 10, the cap): without the noise-floor acceptance the Newton
# line search cannot lower `joint` and the solve reports failure; with it the mode is accepted.
function _nf_mode(seed, G, gamma, sd1, trials)
    rng = StableRNG(seed)
    m = 6; n = G * m
    gidx = repeat(1:G, inner = m); x = randn(rng, n)
    ntr = fill(trials, n); s = [rand(rng, 0:trials) for _ in 1:n]
    Xmu = hcat(ones(n), randn(rng, n))
    Q = sparse(1.0I, G, G) + spdiagm(1 => fill(-0.3, G - 1), -1 => fill(-0.3, G - 1))
    rho = tanh(gamma); sech = DRModels._binomial_stable_sech(gamma)
    Qj = kron(Q, sparse([1.0 -rho; -rho 1.0] ./ sech^2))
    aux = (s = s, ntr = ntr, logchoose = zeros(n))
    return DRModels._binomial_animal_q2_mode(aux, Float64.(s), Float64.(ntr), Xmu, x, gidx, Qj,
        fill(0.5, G), x .* sd1, [0.1, 0.2])
end

@testset "(e1) _binomial_animal_q2_mode accepts the mode at the rounding noise floor" begin
    # (seed, G, gamma, sd1, trials): each reported ok = false with the noise-floor line removed
    for (seed, G, gamma, sd1, trials) in ((2, 25, 10.0, 0.001, 1), (3, 25, 10.0, 0.05, 1),
                                          (4, 25, 10.0, 0.05, 1), (5, 25, 10.0, 0.001, 10),
                                          (6, 25, 10.0, 0.05, 1), (7, 10, 10.0, 0.001, 1),
                                          (7, 10, 10.0, 0.001, 10))
        b, ch, ok = _nf_mode(seed, G, gamma, sd1, trials)
        @test ok
        @test all(isfinite, b)
    end
end

@testset "(e2) restart bookkeeping: info record, fit.scales[:q2_restarts], skipped by simulate" begin
    Optim = DRModels.Optim
    f(th) = th[1] < -2 ? 1.0 + 0.05 * (th[1] + 6)^2 : 0.5 + 0.1 * (th[1] - 2)^2 + (th[1] < 0.5 ? 0.6 : 0.0)
    run_from(th0) = Optim.optimize(th -> f(th), th0, Optim.NelderMead(), Optim.Options(iterations = 200))
    first = run_from([-5.0])
    start_from(sd0, sd1, het) = [sd1 > 0.5 ? 3.0 : -5.0]
    sd_min(th) = th[1] < -2 ? 1e-4 : 1.0
    info = Ref{Any}()
    best = DRModels._q2_boundary_restarts(run_from, [0.0], first, start_from, sd_min, r -> true; info)
    @test info[].ran
    @test 1 <= info[].winner <= length(DRModels._Q2_ALT_STARTS)
    @test info[].gain > 0.4 && info[].gain ≈ Optim.minimum(first) - Optim.minimum(best)
    @test !info[].flipped
    # an interior first fit runs nothing and says so
    interior = run_from([3.0])
    DRModels._q2_boundary_restarts(run_from, [0.0], interior, start_from, sd_min, r -> true; info)
    @test info[] == (ran = false, winner = 0, gain = 0.0, flipped = false)
    # a real fit records it, and simulate / predict ignore the key
    wd = joinpath(@__DIR__, "fixtures", "weak_optimum")
    m, h = readdlm(joinpath(wd, "binomial_animal_q2_boundary.tsv"), '\t', Any; header = true, comments = true)
    cols = Dict(String(h[j]) => m[:, j] for j in eachindex(h))
    A = readdlm(joinpath(wd, "binomial_animal_q2_boundary_A.tsv"), '\t', Float64; comments = true)
    data = (; y = Float64.(cols["y"]), x = Float64.(cols["x"]), z0 = Float64.(cols["z0"]),
            z1 = Float64.(cols["z1"]), group = String.(cols["group"]))
    fm = bf(@formula(y ~ x + animal(1 + x | p | group)),
            @formula(sd(group, animal, coef(intercept)) ~ z0), @formula(sd(group, animal, coef(x)) ~ z1))
    fit = _quiet6(() -> drm(fm, DRModels.Binomial(); data, A, se = false))
    rec = fit.scales[:q2_restarts]
    @test length(rec) == 4 && all(isfinite, rec)
    @test rec[1] == 1.0                                   # the boundary fixture does restart
    @test 0.0 <= rec[2] <= length(DRModels._Q2_ALT_STARTS)
    @test _quiet6(() -> length(simulate(fit; rng = StableRNG(5), re = :zero))) == length(data.y)
    nd = (; x = data.x[1:5], z0 = data.z0[1:5], z1 = data.z1[1:5], group = data.group[1:5])
    @test _quiet6(() -> length(simulate(fit; rng = StableRNG(5), newdata = nd))) == 5
    @test !haskey(DRModels._bridge_dpars(fit), "q2_restarts")
end
