# test_1190_clamped_logsd_vcov.jl -- #1190: a random-effect log SD at its clamp must not
# blank or corrupt the standard errors of the identified parameters.
#
# NB2 crossed random intercepts `(1 | g) + (1 | h)` at heavy overdispersion (the #1088 sets):
# at the -8 floor the objective is flat in that log SD, so its Hessian row is near zero.
# On base the value-path pseudo-inverse reported a meaningless SE for the clamped coordinate
# (114, 130 on s7 / n = 1500) and, when finite-difference noise tipped one eigenvalue
# negative, an all-NaN covariance for a fit with `converged = true` (s7 / n = 800). Now the
# clamped log SD is held fixed at the boundary, the Hessian of the remaining coordinates is
# inverted, and the clamped coordinate has NaN (NA) SE with a boundary warning.
#
# Reference: native drmTMB ca3ddef (0.7.1), test/fixtures/j3_se/<route>/ref.toml (written by
# test/parity/gen_1088_nb2_runoff_reference.R). On s8 / n = 1500 drmTMB's sdreport is finite
# for the fixed effects and the dispersion; on s7 its sdreport is all NaN (drmTMB #1594).
using DRModels
using Test, TOML, LinearAlgebra, Logging

const _R1190_DIR = joinpath(@__DIR__, "fixtures", "j3_se")

function _r1190_data(route)
    lines = readlines(joinpath(_R1190_DIR, route, "data.csv"))
    rows = [split(l, ',') for l in lines[2:end]]
    return (; y = [parse(Float64, r[1]) for r in rows], x = [parse(Float64, r[2]) for r in rows],
              g = String[r[3] for r in rows], h = String[r[4] for r in rows])
end
_r1190_fit(route) = drm(bf(@formula(y ~ 1 + x + (1 | g) + (1 | h)), @formula(sigma ~ 1)),
                        NegBinomial2(); data = _r1190_data(route))
_r1190_se(fit) = [v >= 0 ? sqrt(v) : NaN for v in diag(vcov(fit))]

@testset "#1190: helper holds a clamped log SD fixed" begin
    θ = [0.5, -0.2, 3.0, -8.0, 0.4]          # beta, beta, log sigma, log SD g (floor), log SD h
    H = [4.0 0.5 0.1 0.0 0.2;
         0.5 3.0 0.0 0.0 0.1;
         0.1 0.0 2.0 0.0 0.0;
         0.0 0.0 0.0 0.0 0.0;
         0.2 0.1 0.0 0.0 1.5]
    free = [1, 2, 3, 5]
    V = @test_logs (:warn, r"treated as fixed") DRModels._vcov_with_clamped_logsd_fixed(
        H, θ, 4:5; context = "unit")
    @test V[free, free] ≈ inv(H[free, free])
    @test all(isnan, V[4, :]) && all(isnan, V[:, 4])
    # Noise that tips the flat coordinate negative must not blank the other SEs.
    Hn = copy(H); Hn[4, 4] = -5e-4
    Vn = @test_logs (:warn, r"treated as fixed") DRModels._vcov_with_clamped_logsd_fixed(
        Hn, θ, 4:5; context = "unit")
    @test Vn[free, free] ≈ inv(H[free, free])
    # Silent inside bootstrap / profile refits.
    @test_logs DRModels._without_boundary_warnings(
        () -> DRModels._vcov_with_clamped_logsd_fixed(H, θ, 4:5; context = "unit"))
    # Nothing clamped: exactly the ordinary guarded inverse.
    θi = [0.5, -0.2, 3.0, -1.0, 0.4]
    Hi = copy(H); Hi[4, 4] = 2.0
    @test @test_logs(DRModels._vcov_with_clamped_logsd_fixed(Hi, θi, 4:5; context = "unit")) ≈ inv(Hi)
    # The upper clamp is held fixed too.
    θu = [0.5, -0.2, 3.0, 0.1, 3.0]
    Hu = copy(H); Hu[5, :] .= 0; Hu[:, 5] .= 0; Hu[4, 4] = 2.0
    Vu = @test_logs (:warn, r"treated as fixed") DRModels._vcov_with_clamped_logsd_fixed(
        Hu, θu, 4:5; context = "unit")
    @test all(isnan, Vu[5, :]) && all(isfinite, Vu[1:4, 1:4])
end

@testset "#1190: gradient-difference Hessian of a coordinate subset" begin
    A = [3.0 0.4 0.0; 0.4 2.0 0.3; 0.0 0.3 1.0]
    g!(G, x) = (G .= A * x; G)                     # f = x'Ax/2, so H = A everywhere
    x0 = [0.2, -0.1, 0.7]
    @test DRModels._finite_hessian_from_gradient(g!, x0) ≈ A atol = 1e-6
    Hsub = DRModels._finite_hessian_from_gradient(g!, x0; coords = [1, 3])
    @test size(Hsub) == (2, 2)
    @test Hsub ≈ A[[1, 3], [1, 3]] atol = 1e-6
end

@testset "#1190: NB2 crossed fit with a log SD at the floor" begin
    for route in ("nb2_runoff_s7_n1500", "nb2_runoff_s8_n1500", "nb2_runoff_s7_n800")
        @testset "$route" begin
            fit = Logging.with_logger(Logging.NullLogger()) do
                _r1190_fit(route)
            end
            se = _r1190_se(fit)
            floor_k = [k for k in 4:5 if fit.theta[k] <= -8.0 + 1e-3]
            @test !isempty(floor_k)
            # The identified coordinates (beta, log sigma, and any interior log SD) have a SE ...
            ok = setdiff(1:5, floor_k)
            @test all(isfinite, se[ok]) && all(>(0), se[ok])
            # ... and the clamped coordinate has none, rather than a meaningless one.
            @test all(isnan, se[floor_k])
            # Never `converged` with an all-NaN covariance, and a fit at a clamp is a boundary
            # fit (drmTMB reads the same collapsed SD, ~1e-28, as `:boundary`).
            @test !all(isnan, se)
            @test convergence_status(fit) === :boundary
            @test !is_converged(fit)
        end
    end

    # drmTMB (s8 / n = 1500, opt$convergence 0, pdHess TRUE): fixed-effect and log-dispersion
    # SEs. DRModels now holds the collapsed log SD fixed where drmTMB runs it to -13.9, so the
    # two Hessians are those of the same remaining parameters.
    ref = TOML.parsefile(joinpath(_R1190_DIR, "nb2_runoff_s8_n1500", "ref.toml"))
    @test ref["comparable"]
    fit = Logging.with_logger(Logging.NullLogger()) do
        _r1190_fit("nb2_runoff_s8_n1500")
    end
    se = _r1190_se(fit)
    se_t = Float64.(ref["se"])
    live = [1, 2, 3, 5]                      # log SD h is interior in both
    @test maximum(abs.(se[live] ./ se_t[live] .- 1)) < 1e-4

    # s7 / n = 1500: drmTMB's sdreport is all NaN (drmTMB #1594), so there is no reference
    # SE. The fixed-effect SEs are the same as the pseudo-inverse reported on base to 1e-3.
    fit7 = Logging.with_logger(Logging.NullLogger()) do
        _r1190_fit("nb2_runoff_s7_n1500")
    end
    @test _r1190_se(fit7)[1:3] ≈ [0.52051, 0.69901, 0.087961] rtol = 1e-3
end
