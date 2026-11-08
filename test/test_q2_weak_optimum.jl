# Weak-optimum fixes for the structured intercept-and-slope (q2) Laplace fitters.
#
#  (a) NB2 phylo q2, seed 55 of the scale-extension study: the single default start slid
#      down a flat slope-SD ridge to the boundary (log-lik -1637.421869) while drmTMB 0.8.1
#      reaches the interior optimum (-1637.420775). `_q2_boundary_restarts` re-runs L-BFGS
#      from alternative SD starts when a fit ends near the boundary.
#  (c) Binomial animal q2, the `test_stored_loglik_at_optimum` dataset on Julia 1.10: slope SD
#      and rho run to the boundary, a stencil probe of the finite-difference gradient hit
#      the 1e18 failed-inner-solve sentinel, and the fit stopped unconverged.
#  The datasets are committed (test/fixtures/weak_optimum), so the numbers do not depend on
#  the Julia version's random stream. The R reference value is the drmTMB 0.8.1 ML log-lik
#  on the same file (`ll_R` of `scale_ext/per_seed.csv`, seed 55).
using DRModels, Test, DelimitedFiles, LinearAlgebra

const _WO_DIR = joinpath(@__DIR__, "fixtures", "weak_optimum")

function _wo_read(file)
    m, h = readdlm(joinpath(_WO_DIR, file), '\t', Any; header = true, comments = true)
    cols = Dict(String(h[j]) => m[:, j] for j in eachindex(h))
    return cols
end

@testset "_q2_boundary_restarts selection rule" begin
    Optim = DRModels.Optim
    # Two basins in theta[1]: a boundary ridge basin around -6 (objective 1.0) and an
    # interior basin around 2 (objective 0.5). Start 0.0 falls to the ridge, start 3.0 to the interior.
    f(th) = th[1] < -2 ? 1.0 + 0.05 * (th[1] + 6)^2 : 0.5 + 0.1 * (th[1] - 2)^2 + (th[1] < 0.5 ? 0.6 : 0.0)
    last_b = [1.0, 2.0, 3.0]
    run_from(th0) = Optim.optimize(th -> f(th), th0, Optim.NelderMead(), Optim.Options(iterations = 200))
    first = run_from([-5.0])
    @test Optim.minimizer(first)[1] < -2 && Optim.minimum(first) > 0.99
    start_from(sd0, sd1, het) = [sd1 > 0.5 ? 3.0 : -5.0]
    sd_min(th) = th[1] < -2 ? 1e-4 : 1.0           # collapsed SD <=> boundary basin
    conv(r) = true
    best = DRModels._q2_boundary_restarts(run_from, last_b, first, start_from, sd_min, conv)
    @test best !== first
    @test Optim.minimum(best) < 0.51
    # An interior first fit is returned unchanged with the warm-start buffer untouched.
    interior = run_from([3.0])
    stamp = copy(last_b)
    ran = Ref(0)
    counting(th0) = (ran[] += 1; run_from(th0))
    kept = DRModels._q2_boundary_restarts(counting, last_b, interior, start_from, sd_min, conv)
    @test kept === interior
    @test ran[] == 0
    @test last_b == stamp
    # A boundary first fit that no alternative beats is also returned unchanged, buffer restored.
    flat(th) = 1.0 + 0.05 * (th[1] + 6)^2
    run_flat(th0) = Optim.optimize(th -> flat(th), th0, Optim.NelderMead(), Optim.Options(iterations = 200))
    first_flat = run_flat([-5.0])
    last_b .= [4.0, 5.0, 6.0]; stamp = copy(last_b)
    kept = DRModels._q2_boundary_restarts(run_flat, last_b, first_flat, start_from, sd_min, conv)
    @test kept === first_flat
    @test last_b == stamp
    # An unconverged first fit yields to a converged alternative at an objective up to 1e-5 higher.
    r1 = run_flat([-5.0])
    call = Ref(0)
    conv_alt(r) = (call[] += 1; call[] > 1)         # first call (the first fit) unconverged, later calls converged
    run_shift(th0) = Optim.optimize(th -> flat(th) + 5e-6, th0, Optim.NelderMead(), Optim.Options(iterations = 200))
    chosen = DRModels._q2_boundary_restarts(run_shift, last_b, r1, start_from, th -> 1.0, conv_alt)
    @test chosen !== r1
end

@testset "_q2_alt_theta builds constant and heterogeneous slope-SD starts" begin
    Z0 = [1.0 -1.0; 1.0 1.0; 1.0 -1.0; 1.0 1.0]
    Z1 = [1.0 -0.5; 1.0 0.25; 1.0 -0.5; 1.0 0.5]
    ranges = (3:4, 5:6)
    th0 = fill(9.0, 6)
    th = DRModels._q2_alt_theta(th0, ranges, (Z0, Z1), 0.2, 0.1, 0)
    @test th[1:2] == [9.0, 9.0]
    @test th[3:4] ≈ [log(0.2), 0.0] atol = 1e-12
    @test th[5:6] ≈ [log(0.1), 0.0] atol = 1e-12
    for het in (-1, 1)
        th = DRModels._q2_alt_theta(th0, ranges, (Z0, Z1), 0.4, 0.22, het)
        @test th[5] ≈ log(0.22)
        @test th[6] ≈ het * 2 / 0.5          # 2 / max |z1|
    end
    # an intercept-only slope-SD design has no predictor to make heterogeneous
    th = DRModels._q2_alt_theta([9.0, 9.0, 9.0, 0.0, 0.0], (3:4, 5:5), (Z0, ones(4, 1)), 0.4, 0.22, 1)
    @test th[5] ≈ log(0.22)
end

@testset "NB2 phylo q2 seed 55 reaches the interior optimum (case a)" begin
    d = _wo_read("nb2_phylo_s055.tsv")
    data = (; y = Float64.(d["y"]), x = Float64.(d["x"]), z0 = Float64.(d["z0"]),
              z1 = Float64.(d["z1"]), g = String.(d["g"]))
    tree = strip(read(joinpath(_WO_DIR, "nb2_phylo_tree.nwk"), String))
    f = bf(@formula(y ~ x + phylo(1 + x | g)), @formula(sigma ~ 1),
           @formula(sd(g, phylogenetic, coef(intercept)) ~ z0),
           @formula(sd(g, phylogenetic, coef(x)) ~ z1))
    fit = drm(f, NegBinomial2(); data, tree, se = false)
    ll_R = -1637.42077452727
    boundary_ll = -1637.42186938546      # value of the single-start fit that stopped on the ridge
    @test fit.converged
    @test abs(loglikelihood(fit) - ll_R) < 1e-6
    @test loglikelihood(fit) > boundary_ll + 5e-4
    # interior optimum: the slope SD is small but not collapsed (drmTMB: coefficients -4.363, -1.417)
    @test fit.theta[6] > -6
end

@testset "Binomial animal q2 boundary fit converges (case c)" begin
    d = _wo_read("binomial_animal_q2_boundary.tsv")
    A = readdlm(joinpath(_WO_DIR, "binomial_animal_q2_boundary_A.tsv"), '\t', Float64; comments = true)
    data = (; y = Float64.(d["y"]), x = Float64.(d["x"]), z0 = Float64.(d["z0"]),
              z1 = Float64.(d["z1"]), group = String.(d["group"]))
    f = bf(@formula(y ~ x + animal(1 + x | group)),
           @formula(sd(group, animal, coef(intercept)) ~ z0),
           @formula(sd(group, animal, coef(x)) ~ z1))
    fit = drm(f, DRModels.Binomial(); data, A, se = false)
    @test fit.converged
    # boundary optimum: the slope SD has collapsed; several starts agree on this objective
    @test abs(loglikelihood(fit) + 263.18498) < 5e-5
end
