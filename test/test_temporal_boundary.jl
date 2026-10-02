# Temporal boundary diagnostic (twin of drmTMB's `temporal_boundary` check,
# drmTMB #1447 / #1448). A temporal parameter can reach an interpretability
# boundary while the optimiser and Hessian look regular; such fits are flagged
# at fit time and in `check_drm(fit).temporal_boundary`.
using DRModels
using Test, Random, LinearAlgebra, StableRNGs, Logging, Statistics

@testset "temporal boundary diagnostic" begin
    rules = DRModels._temporal_boundary_rules
    gaps = [[0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 4.0]]          # spans 6 and 4.5, min gap 0.5
    @testset "each rule fires at drmTMB's threshold" begin
        @test isempty(rules(:ou, 0.4, 1.0, 0.4, gaps))
        @test only(rules(:ou, 0.9e-3, 1.0, 0.4, gaps)) |> s -> startswith(s, "sigma_ratio=")
        @test isempty(rules(:ou, 1.1e-3, 1.0, 0.4, gaps))
        @test only(rules(:ou, 0.4, 1.0, 1e-4 / 6 * 0.99, gaps)) |> s -> startswith(s, "decay_x_max_span=")
        @test isempty(rules(:ou, 0.4, 1.0, 1e-4 / 6 * 1.01, gaps))
        @test only(rules(:ou, 0.4, 1.0, 30 / 0.5 * 1.01, gaps)) |> s -> startswith(s, "decay_x_min_gap=")
        @test isempty(rules(:ou, 0.4, 1.0, 30 / 0.5 * 0.99, gaps))
        @test only(rules(:ar1, 0.4, 1.0, -0.9995, gaps)) |> s -> startswith(s, "phi=")
        @test isempty(rules(:ar1, 0.4, 1.0, 0.99, gaps))
        @test isempty(rules(:homtoep, 0.4, 1.0, nothing, gaps))
        @test only(rules(:homtoep, 1e-4, 1.0, nothing, gaps)) |> s -> startswith(s, "sigma_ratio=")
    end

    # drmTMB's own boundary panel: stable series intercepts, a weak slowly
    # decaying OU process and no (1 | id), so the OU decay absorbs the
    # intercept and runs to ~0.
    function panel(rng; n = 20, times = [0.0, 1, 3, 6], b_sd = 0.0, ou_sd = 0.8,
                   decay = 0.4, noise = 0.3)
        R = [exp(-decay * abs(a - b)) for a in times, b in times]
        L = cholesky(Symmetric(R)).L
        id = String[]; t = Float64[]; x = Float64[]; y = Float64[]
        for i in 1:n
            xs = randn(rng, length(times))
            z = b_sd * randn(rng) .+ ou_sd .* (L * randn(rng, length(times)))
            append!(id, fill("s$i", length(times))); append!(t, times); append!(x, xs)
            append!(y, 0.5 .+ 0.3 .* xs .+ z .+ noise .* randn(rng, length(times)))
        end
        return (y = y, x = x, id = id, t = t)
    end
    fOU = bf(@formula(y ~ x + temporal(1 | id, t, ou)), @formula(sigma ~ 1))

    @testset "healthy OU fit: no finding, no warning" begin
        d = panel(StableRNG(1))
        fit = @test_logs min_level = Logging.Warn drm(fOU, Gaussian(); data = d)
        tb = with_logger(NullLogger()) do; check_drm(fit).temporal_boundary; end
        @test tb.at_boundary == false && isempty(tb.findings)
        # non-temporal fits carry `nothing`
        f0 = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = d)
        @test with_logger(() -> check_drm(f0), NullLogger()).temporal_boundary === nothing
    end

    # drmTMB's own boundary panel (its test-temporal-boundary.R DGP, R
    # `set.seed(2)`: stable series intercepts SD 0.6, a weak OU process SD 0.2
    # with decay 0.01, noise 0.3, 20 series at times 0, 1, 3, 6, no (1 | id)).
    # The data were generated in R by base RNG calls only and committed as
    # test/fixtures/temporal/ou_boundary_panel.csv. drmTMB (#1448 head
    # 012258e9f) fits decay 5.0e-11 with the same β, σ and process SD (to
    # 1e-9) and reports convergence_status "boundary". Its printed logLik,
    # −45.5124224656, is 2.5e-6 above the EXACT value at its own estimates:
    # a 256-bit dense Cholesky gives −45.5124249517 there (measured
    # 2026-10-02), so drmTMB's objective loses digits as decay → 0. The
    # exact value is the reference below.
    @testset "OU decay running to zero is flagged (drmTMB's boundary panel)" begin
        L = readlines(joinpath(@__DIR__, "fixtures", "temporal", "ou_boundary_panel.csv"))
        h = split(L[1], ','); r = split.(L[2:end], ',')
        col(n) = getindex.(r, findfirst(==(n), h))
        d = (id = String.(col("id")), t = parse.(Float64, col("t")),
             x = parse.(Float64, col("x")), y = parse.(Float64, col("y")))
        fit = @test_logs (:warn, r"interpretability boundary") match_mode = :any drm(fOU, Gaussian(); data = d)
        λ = temporal_parameters(fit).decay
        println("drmTMB boundary panel: OU decay = ", λ, ", logLik = ", loglik(fit))
        @test λ * 6 < 1e-4
        @test loglik(fit) >= -45.5124249517 - 1e-9      # at least the exact value at drmTMB's θ̂
        tb = with_logger(NullLogger()) do; check_drm(fit).temporal_boundary; end
        @test tb.at_boundary && any(startswith("decay_x_max_span="), tb.findings)
        @test fit.converged                             # still converged, as drmTMB
        # bootstrap / profile refits stay quiet (`_without_boundary_warnings`)
        @test_logs min_level = Logging.Warn DRModels._without_boundary_warnings(
            () -> DRModels._warn_temporal_boundary(fit))
    end

    @testset "residual SD collapsing into the process is flagged" begin
        d = panel(StableRNG(3); n = 60, noise = 0.0)    # no observation noise at all
        fit = with_logger(NullLogger()) do; drm(fOU, Gaussian(); data = d); end
        tp = temporal_parameters(fit)
        println("no-noise panel: sigma = ", tp.sigma)
        tb = with_logger(NullLogger()) do; check_drm(fit).temporal_boundary; end
        @test tp.sigma < 1e-3 * std(d.y)
        @test tb.at_boundary && any(startswith("sigma_ratio="), tb.findings)
    end
end
