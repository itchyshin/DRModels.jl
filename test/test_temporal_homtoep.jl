# temporal(1 | id, occ, homtoep) on the Gaussian mean (wave 2; twin of drmTMB's
# `temporal(1 | id, time = occasion, structure = "homtoep")`): every series
# observes the same K equally spaced occasions, and its K responses have
# covariance σ² R, R a free positive-definite Toeplitz correlation matrix
# parameterised by partial autocorrelations. σ is the TOTAL within-series SD.
#
# Independent checks: (1) the partial autocorrelations of the package's R,
# recomputed from the dense matrix by Schur complements in BigFloat, are
# tanh κ; (2) the objective equals the dense Gaussian −log-likelihood built
# from that R, evaluated in BigFloat (strong partial autocorrelations make R
# nearly singular, and a Float64 dense oracle then loses more digits than the
# recursion it is checking).
using DRModels
using Test, Random, LinearAlgebra, StableRNGs, Statistics, Logging

@testset "temporal homogeneous Toeplitz (Gaussian mean)" begin
    toep(ρ) = [ρ[abs(i - j) + 1] for i in eachindex(ρ), j in eachindex(ρ)]
    rho_of(κ) = [1.0; DRModels._homtoep_correlations(κ)]
    # Partial autocorrelation at lag m of a dense correlation matrix: the
    # correlation of x_1 and x_{m+1} given x_2..x_m (Schur complement).
    function dense_pacf(R)
        K = size(R, 1)
        map(1:(K-1)) do m
            a = [1, m + 1]; b = collect(2:m)
            C = R[a, a] - (isempty(b) ? zeros(2, 2) : R[a, b] * (R[b, b] \ R[b, a]))
            C[1, 2] / sqrt(C[1, 1] * C[2, 2])
        end
    end
    # θ = [β; log σ; κ]; dense Cholesky in BigFloat
    function dense_nll(θ, y, X, id, occ)
        p = size(X, 2)
        θb = big.(θ)
        σ2 = exp(2θb[p+1])
        ρ = rho_of(θb[p+2:end])
        lev = sort(unique(occ)); step = lev[2] - lev[1]
        n = length(y)
        V = [id[i] == id[j] ? σ2 * ρ[Int(abs(occ[i] - occ[j]) ÷ step) + 1] : zero(σ2) for i in 1:n, j in 1:n]
        C = cholesky(Symmetric(V))
        r = big.(y) .- big.(X) * θb[1:p]
        return Float64((logdet(C) + dot(r, C \ r) + n * log(2 * big(pi))) / 2)
    end
    # Complete panel of `S` series × occasions `occ`, true lag correlations ρ,
    # total SD σ, rows shuffled.
    # A non-exponential, positive-definite lag pattern: PACs (0.6, −0.3, 0.35, 0, 0.15).
    ρ6 = rho_of(atanh.([0.6, -0.3, 0.35, 0.0, 0.15]))
    function sim_data(rng; S = 30, occ = collect(0:5), ρ = ρ6[1:length(occ)],
                      σ = 0.9, β = (0.2, 0.4))
        K = length(occ)
        L = cholesky(Symmetric(toep(ρ))).L
        id = String[]; t = Int[]; x = Float64[]; y = Float64[]
        for s in 1:S
            xs = randn(rng, K)
            append!(id, fill("site$(lpad(s, 3, '0'))", K)); append!(t, occ); append!(x, xs)
            append!(y, β[1] .+ β[2] .* xs .+ σ .* (L * randn(rng, K)))
        end
        p = randperm(rng, length(y))
        return (y = y[p], x = x[p], id = id[p], occ = t[p])
    end
    fT = bf(@formula(y ~ x + temporal(1 | id, occ, homtoep)), @formula(sigma ~ 1))
    quiet(f) = with_logger(f, NullLogger())

    @testset "partial-autocorrelation parameterisation (dense Schur check)" begin
        rng = StableRNG(41)
        worst = 0.0
        for K in (3, 6, 12), _ in 1:4
            κ = 1.5 .* randn(rng, K - 1)
            ρbig = rho_of(big.(κ))
            @test dense_pacf(toep(ρbig)) ≈ tanh.(big.(κ)) atol = 1e-40   # the recursion's algebra
            @test isposdef(Symmetric(toep(rho_of(κ))))
            worst = max(worst, maximum(abs.(rho_of(κ) .- ρbig)))           # its Float64 accuracy
        end
        println("homtoep lag correlations: worst |Float64 − BigFloat| = ", worst)
        @test worst < 1e-12
        # an AR1 correlation pattern has PACs (φ, 0, …, 0)
        @test rho_of([atanh(0.7), 0, 0, 0]) ≈ 0.7 .^ (0:4) atol = 1e-14
        # near-singular extremes stay PD and finite
        κ = [5.0, -5.0, 5.0]
        @test all(isfinite, rho_of(κ)) && isposdef(Symmetric(toep(rho_of(κ))))
    end

    @testset "dense oracle (K = 3, 6, 12; random θ and |π| → 1)" begin
        rng = StableRNG(42)
        worst = 0.0
        for occ in ([0, 1, 2], [3, 5, 7, 9, 11, 13], collect(0:11))
            K = length(occ)
            d = sim_data(rng; S = max(8, K), occ = occ, ρ = rho_of(0.6 .* randn(rng, K - 1)))
            fit = quiet(() -> drm(fT, Gaussian(); data = d))
            X = hcat(ones(length(d.y)), d.x)
            @test length(fit.theta) == 3 + K - 1
            @test loglik(fit) ≈ -dense_nll(fit.theta, d.y, X, d.id, d.occ) atol = 1e-10
            for _ in 1:5
                θ = [randn(rng); randn(rng); 0.5randn(rng); 1.2 .* randn(rng, K - 1)]
                ref = dense_nll(θ, d.y, X, d.id, d.occ)
                e = abs(fit.nll(θ) - ref) / max(1, abs(ref))
                worst = max(worst, e)
                @test e < 1e-10
            end
            # strong but admissible partial autocorrelations (|π| up to 0.995)
            θ = [0.1; 0.3; 0.0; fill(3.0, K - 1) .* (-1) .^ (1:K-1)]
            @test fit.nll(θ) ≈ dense_nll(θ, d.y, X, d.id, d.occ) rtol = 1e-10
        end
        println("homtoep dense oracle: worst |Δnll| / max(1, |nll|) = ", worst)
    end

    # drmTMB #1449's stability points: partial autocorrelations at and beyond
    # the Float64 limit of tanh (|κ| ≥ 19 rounds tanh to ±1). The reference is
    # an independent prediction-error evaluation in 2048-bit arithmetic (|κ| = 40
    # makes R singular to ~1e-170): the
    # Yule–Walker system R φ = r is solved at every order by dense LU for the
    # predictor and the innovation variance, using ρ computed in BigFloat.
    @testset "extreme partial autocorrelations vs a BigFloat prediction-error reference" begin
        function pe_reference(θ, y, X, id, occ)
            setprecision(BigFloat, 2048) do
                p = size(X, 2)
                θb = big.(θ)
                ρ = rho_of(θb[p+2:end]); K = length(ρ)
                @assert maximum(abs.(dense_pacf(toep(ρ)) .- tanh.(θb[p+2:end]))) < big(1e-300)
                σ2 = exp(2θb[p+1])
                r = big.(y) .- big.(X) * θb[1:p]
                tot = big(0)
                for s in unique(id)
                    rows = findall(==(s), id); rows = rows[sortperm(occ[rows])]
                    rs = r[rows]
                    for t in 1:K
                        if t == 1
                            e, v = rs[1], big(1)
                        else
                            Rm = toep(ρ[1:t-1]); rv = ρ[2:t]
                            φ = Rm \ rv                      # Yule–Walker, order t − 1
                            e = rs[t] - sum(φ[j] * rs[t-j] for j in 1:t-1)
                            v = 1 - dot(φ, rv)
                        end
                        tot += log(σ2 * v) + e^2 / (σ2 * v)
                    end
                end
                Float64((tot + length(y) * log(2 * big(pi))) / 2)
            end
        end
        d = sim_data(StableRNG(53); S = 10)
        fit = quiet(() -> drm(fT, Gaussian(); data = d))
        X = hcat(ones(length(d.y)), d.x)
        for κ in ([8, -8, 5, 0, 3], [15, 15, -15, 12, 0.1], [18.5, 0, 0, 0, 0],
                  fill(40.0, 5), fill(-40.0, 5), [40.0, -40, 40, -40, 40])
            θ = [0.2; 0.4; log(0.9); Float64.(κ)]
            v = fit.nll(θ)
            ref = pe_reference(θ, d.y, X, d.id, d.occ)
            # the true objective is huge here (innovation variances down to
            # ~1e-170); it must be finite and equal the exact reference, not the
            # route's 1e18 failure sentinel
            println("homtoep extreme κ = ", κ, ": nll = ", v, ", rel. error = ", abs(v - ref) / abs(ref))
            @test isfinite(v)
            @test v ≈ ref rtol = 1e-10
        end
    end

    @testset "recovery (S = 400 sites × 6 occasions, non-exponential ρ)" begin
        ρtrue = ρ6
        d = sim_data(StableRNG(2032); S = 400, ρ = ρtrue)
        fit = drm(fT, Gaussian(); data = d)
        tp = temporal_parameters(fit)
        println("homtoep recovery (S = 400): cor = ", round.(tp.cor; digits = 3), ", sigma = ",
                tp.sigma, ", beta = ", coef(fit, :mu))
        @test fit.converged
        @test maximum(abs.(tp.cor .- ρtrue[2:end])) < 0.08          # ≈ 3 SE at S = 400
        @test tp.sigma ≈ 0.9 atol = 0.06
        @test coef(fit, :mu) ≈ [0.2, 0.4] atol = 0.08
        @test tp.pac ≈ dense_pacf(toep([1; tp.cor])) atol = 1e-10
    end

    @testset "invariances" begin
        d = sim_data(StableRNG(43); S = 25)
        fit = drm(fT, Gaussian(); data = d)
        p = reverse(eachindex(d.y))
        @test drm(fT, Gaussian(); data = map(c -> c[p], d)).nll(fit.theta) ≈ fit.nll(fit.theta) atol = 1e-9
        # only the lag index matters: occasions 10, 13, …, 25 fit the same model
        d2 = merge(d, (occ = 10 .+ 3 .* d.occ,))
        @test drm(fT, Gaussian(); data = d2).nll(fit.theta) ≈ fit.nll(fit.theta) atol = 1e-9
    end

    @testset "accessors" begin
        d = sim_data(StableRNG(44); S = 30)
        fit = drm(fT, Gaussian(); data = d)
        n = length(d.y); X = hcat(ones(n), d.x)
        @test first.(fit.blocks) == [:mu, :sigma, :temporal_pac]
        @test Dict(fit.coefnames)[:temporal_pac] == ["pac_lag$m" for m in 1:5]
        tp = temporal_parameters(fit)
        @test tp.structure === :homtoep && tp.sd === nothing && tp.phi === nothing &&
              tp.decay === nothing && tp.sd_iid === nothing && tp.sd_phylo === nothing
        @test length(tp.cor) == 5 && tp.pac ≈ tanh.(coef(fit, :temporal_pac))
        @test tp.sigma ≈ exp(only(coef(fit, :sigma)))
        @test predict(fit, d) ≈ fitted(fit) ≈ X * coef(fit, :mu)
        pp = predict_parameters(fit, d)
        @test pp[:mu] ≈ fitted(fit) && all(≈(tp.sigma), pp[:sigma])
        @test isempty(ranef(fit))                       # no latent states (as drmTMB)
        @test structured_effects(fit) == [(dpar = :mu, kind = :temporal, grouping = :id)]
        @test any(r -> r.param === :temporal_pac && r.scale === :atanh, profile_targets(fit))
        @test length(coeftable(fit).rownms) == 8 && dof(fit) == 8 && isfinite(aic(fit))
        @test residuals(fit) ≈ d.y .- fitted(fit)
        @test check_drm(fit) !== nothing
        @test sprint(show, MIME("text/plain"), fit) isa String
        pr = confint(fit; method = :profile, parm = :mu => "x")
        @test only(pr).lower < coef(fit, :mu)[2] < only(pr).upper
        # drmTMB #1449 scope: mean-coefficient PROFILES only. Wald covariance and
        # intervals, and profiles of σ or the lag correlations, are refused.
        ae = ArgumentError
        werr = try vcov(fit); "" catch e; e.msg end
        @test occursin("Wald coefficient covariance is unavailable", werr)
        @test_throws ae stderror(fit)
        @test_throws ae confint(fit)                               # Wald
        @test_throws ae confint(fit; parm = :mu)
        @test_throws ae predict(fit, d; se = true)
        perr = try confint(fit; method = :profile, parm = :sigma); "" catch e; e.msg end
        @test occursin("mean regression coefficients only", perr)
        @test_throws ae confint(fit; method = :profile, parm = :temporal_pac)
        @test_throws ae profile_result(fit; parm = :temporal_pac)
        allmu = confint(fit; method = :profile)                    # default: the mean block
        @test length(allmu) == 2 && all(r -> r.param === :mu, allmu)
        tg = profile_targets(fit)
        @test all(r -> r.profile_ready == (r.param === :mu), tg)
        @test all(r -> r.profile_note == "temporal_homtoep_nonmean_intervals_deferred",
                  filter(r -> r.param !== :mu, tg))
        ct = coeftable(fit)
        @test all(isnan, ct.cols[2])                               # SEs withheld
        @test occursin("Wald SEs withheld", sprint(show, MIME("text/plain"), fit))
        # whitened residuals = L⁻¹ r per series, L = chol(σ²R) (drmTMB's Pearson)
        ρ = [1.0; tp.cor]; Lc = cholesky(Symmetric(tp.sigma^2 .* toep(ρ))).L
        rq = residuals(fit; type = :quantile); rr = residuals(fit)
        for s in unique(d.id)
            rows = findall(==(s), d.id); rows = rows[sortperm(d.occ[rows])]
            @test rq[rows] ≈ Lc \ rr[rows] atol = 1e-10
        end
        # temporal_parameters of a wave-1 fit carries `cor = pac = nothing`
        fa = drm(bf(@formula(y ~ x + temporal(1 | id, occ, ar1)), @formula(sigma ~ 1)), Gaussian(); data = d)
        @test temporal_parameters(fa).cor === nothing && temporal_parameters(fa).pac === nothing
    end

    @testset "simulate draws σ²R per series" begin
        ρtrue = ρ6
        d = sim_data(StableRNG(45); S = 40, ρ = ρtrue)
        fit = drm(fT, Gaussian(); data = d)
        tp = temporal_parameters(fit)
        E = simulate(fit; nsim = 3000, rng = StableRNG(46)) .- fitted(fit)
        ρ̂ = [1; tp.cor]
        for lag in 0:5
            pairs = [(i, j) for i in eachindex(d.id), j in eachindex(d.id)
                     if d.id[i] == d.id[j] && d.occ[j] - d.occ[i] == lag]
            c = mean(E[i, s] * E[j, s] for (i, j) in pairs, s in 1:size(E, 2))
            @test c ≈ tp.sigma^2 * ρ̂[lag + 1] atol = 0.03
        end
        # different series are independent
        other = [(i, j) for i in 1:20, j in 1:20 if d.id[i] != d.id[j]]
        @test abs(mean(E[i, s] * E[j, s] for (i, j) in other, s in 1:size(E, 2))) < 0.02
        bc = quiet(() -> bootstrap_ci(fit; data = d, B = 6, rng = StableRNG(47)))
        @test length(bc) == 8 && all(r -> isfinite(r.lower) && isfinite(r.upper), bc)
    end

    @testset "refusals (drmTMB's panel rules)" begin
        d = sim_data(StableRNG(48); S = 6)
        msg(f, data; kw...) = try
            drm(f, Gaussian(); data = data, kw...); ""
        catch e
            e isa ArgumentError ? e.msg : rethrow()
        end
        drop1 = map(c -> c[2:end], d)
        @test occursin("complete retained schedule", msg(fT, drop1))
        @test occursin(d.id[1], msg(fT, drop1))                 # names the incomplete series
        uneq = merge(d, (occ = [o == 5 ? 6 : o for o in d.occ],))
        @test occursin("equally spaced", msg(fT, uneq))
        two = map(c -> c[d.occ .< 2], d)
        @test occursin("at least three common occasions", msg(fT, two))
        big = sim_data(StableRNG(49); S = 3, occ = collect(0:12), ρ = [1.0; zeros(12)])
        @test occursin("at most 12", msg(fT, big))
        few = sim_data(StableRNG(51); S = 5)                    # 5 series, 6 occasions
        @test occursin("at least as many series as occasions", msg(fT, few))
        @test drm(fT, Gaussian(); data = sim_data(StableRNG(52); S = 6)) isa DrmFit   # S = K is allowed
        @test occursin("must be finite integers", msg(fT, merge(d, (occ = d.occ .+ 0.5,))))
        oerr = msg(bf(@formula(y ~ x + (1 | id) + temporal(1 | id, occ, homtoep)), @formula(sigma ~ 1)), d)
        @test occursin("does not allow an ordinary", oerr) && !occursin("(1 | id)", oerr)
        dup = map(c -> [c; c[1]], d)
        @test occursin("must be unique", msg(fT, dup))
        @test_throws ArgumentError drm(fT, Gaussian(); data = d, method = :REML)
        @test_throws ArgumentError drm(bf(@formula(y ~ x + temporal(1 | id, occ, homtoep)),
                                          @formula(sigma ~ x)), Gaussian(); data = d)
        # the paired phylo() provider is OU-only
        nwk = "(" * join(["$(l):1.0" for l in unique(d.id)], ",") * ");"
        @test occursin("requires structure `ou`", msg(bf(@formula(y ~ x + phylo(1 | id) +
            temporal(1 | id, occ, homtoep)), @formula(sigma ~ 1)), d; tree = nwk))
        @test_throws ArgumentError drm(bf(@formula(y ~ x + temporal(1 | id, occ, toep)),
                                          @formula(sigma ~ 1)), Gaussian(); data = d)
    end

    # drmTMB: id / time are validated on all rows, the missing-response rows are
    # dropped, and the panel rules then apply to the retained rows (checked
    # against drmTMB #1449 head 90c740791 on Totoro, 2026-10-02).
    @testset "missing responses (drmTMB's response omission)" begin
        raw = readlines(joinpath(@__DIR__, "fixtures", "temporal", "homtoep_panel6.csv"))
        h = split(raw[1], ','); r = split.(raw[2:end], ',')
        col(n) = getindex.(r, findfirst(==(n), h))
        d = (y = Union{Missing,Float64}[parse(Float64, v) for v in col("y")],
             x = parse.(Float64, col("x")), id = String.(col("id")), occ = parse.(Int, col("occ")))
        msg(data) = try
            with_logger(NullLogger()) do; drm(fT, Gaussian(); data = data); end; ""
        catch e
            e isa ArgumentError ? e.msg : rethrow()
        end
        one = merge(d, (y = [i == 1 ? missing : v for (i, v) in enumerate(d.y)],))
        @test occursin("complete retained schedule", msg(one))
        @test occursin(d.id[1], msg(one))
        # occasion 5 missing for every site: fits on occasions 0–4 (K = 5);
        # drmTMB logLik −213.499788672539
        last = merge(d, (y = [o == 5 ? missing : v for (o, v) in zip(d.occ, d.y)],))
        fit = @test_logs (:warn, r"missing response") match_mode = :any drm(fT, Gaussian(); data = last)
        @test nobs(fit) == 200 && length(temporal_parameters(fit).cor) == 4
        @test loglik(fit) ≈ -213.499788672539 atol = 1e-8
        keep = d.occ .!= 5
        sub = (y = Float64.(d.y[keep]), x = d.x[keep], id = d.id[keep], occ = d.occ[keep])
        @test loglik(fit) ≈ loglik(drm(fT, Gaussian(); data = sub)) atol = 1e-10
        # occasion 2 missing for every site: 0, 1, 3, 4, 5 is not equally spaced
        mid = merge(d, (y = [o == 2 ? missing : v for (o, v) in zip(d.occ, d.y)],))
        @test occursin("equally spaced", msg(mid))
        # duplicate keys are refused even when the duplicate's response is missing
        dup = (y = [d.y; missing], x = [d.x; 0.0], id = [d.id; d.id[1]], occ = [d.occ; d.occ[1]])
        @test occursin("must be unique", msg(dup))
    end

    @testset "R bridge spelling (drmTMB keyword form)" begin
        bt = DRModels._bridge_temporal_expr
        @test bt(Meta.parse("temporal(1 | id, time = occ, structure = \"homtoep\")")) ==
              :(temporal(1 | id, occ, homtoep))
        @test_throws ArgumentError bt(Meta.parse("temporal(1 | id, time = occ, structure = \"toep\")"))
        d = sim_data(StableRNG(50); S = 12)
        out = drm_bridge(; formula = "y ~ x + temporal(1 | id, time = occ, structure = \"homtoep\"); sigma ~ 1",
                         family = "gaussian", data = d)
        @test out["loglik"] ≈ loglik(drm(fT, Gaussian(); data = d)) atol = 1e-8
        @test all(isnan, out["vcov"])                  # Wald covariance withheld, as drmTMB
    end
end
