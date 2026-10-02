# temporal(1 | id, time, ou) on the Gaussian mean (D-310; twin of drmTMB's
# `temporal(1 | id, time = elapsed, structure = "ou")`): an Ornstein–Uhlenbeck
# (continuous-time AR1) process over numeric elapsed time, Corr = exp(−λ|Δt|).
# Every likelihood check is against an independent dense construction.
using DRModels
using Test, Random, LinearAlgebra, DelimitedFiles, StableRNGs, Statistics, Logging
using Distributions: MvNormal, Normal, logpdf

@testset "temporal OU (Gaussian mean)" begin
    # θ = [β; log σ; (log σ_b); log σ_t; log λ]
    function dense_nll(θ, y, X, id, t; ordinary = false)
        p = size(X, 2); n = length(y)
        σ2 = exp(2θ[p+1])
        sb2 = ordinary ? exp(2θ[p+2]) : 0.0
        st2 = exp(2θ[p + (ordinary ? 3 : 2)])
        λ = exp(θ[p + (ordinary ? 4 : 3)])
        V = zeros(n, n)
        for i in 1:n, j in 1:n
            id[i] == id[j] && (V[i, j] = st2 * exp(-λ * abs(t[i] - t[j])) + sb2)
        end
        V += σ2 * I
        return -logpdf(MvNormal(X * θ[1:p], Symmetric(V)), y)
    end
    # Irregular elapsed times in (0, 10), unequal series lengths, rows shuffled.
    function sim_data(rng; S = 12, λ = 0.45, st = 0.65, σ = 0.4, sb = 0.0, origin = 0.0)
        id = String[]; t = Float64[]; x = Float64[]; y = Float64[]
        for s in 1:S
            k = rand(rng, 2:7)
            ts = sort(10 .* rand(rng, k)) .+ origin
            R = [exp(-λ * abs(a - b)) for a in ts, b in ts]
            xs = randn(rng, k)
            a = st .* (cholesky(Symmetric(R)).L * randn(rng, k))
            append!(id, fill("site$s", k)); append!(t, ts); append!(x, xs)
            append!(y, 0.8 .+ 0.35 .* xs .+ a .+ sb * randn(rng) .+ σ .* randn(rng, k))
        end
        p = sortperm(rand(rng, length(y)))
        return (y = y[p], x = x[p], id = id[p], elapsed = t[p])
    end
    fOU = bf(@formula(y ~ x + temporal(1 | id, elapsed, ou)), @formula(sigma ~ 1))
    fOUi = bf(@formula(y ~ x + (1 | id) + temporal(1 | id, elapsed, ou)), @formula(sigma ~ 1))

    @testset "dense oracle to 1e-10 (irregular times, ± ordinary intercept)" begin
        rng = StableRNG(21)
        for rep in 1:3, ordinary in (false, true)
            d = sim_data(rng; S = 8 + 3rep, sb = ordinary ? 0.4 : 0.0)
            fit = with_logger(NullLogger()) do      # only the objective is used
                drm(ordinary ? fOUi : fOU, Gaussian(); data = d)
            end
            X = hcat(ones(length(d.y)), d.x)
            np = length(fit.theta)
            @test np == (ordinary ? 6 : 5)
            for _ in 1:4
                θ = [randn(rng); randn(rng); 0.5randn(rng); 0.5 .* randn(rng, np - 4); 1.5randn(rng)]
                @test abs(fit.nll(θ) - dense_nll(θ, d.y, X, d.id, d.elapsed; ordinary)) < 1e-10
            end
            # very fast decay (near-independent states) and very slow decay
            for ψ in (4.0, -5.0)
                θ = copy(fit.theta); θ[end] = ψ
                @test abs(fit.nll(θ) - dense_nll(θ, d.y, X, d.id, d.elapsed; ordinary)) < 1e-10
            end
            # decay → 0 (λ = e^-20): a precision-matrix evaluation measured a negative
            # 1ᵀV⁻¹1 here during a line search; the filter must match the dense oracle
            # (σ is reset to 0.3: a σ̂ that sits at its boundary would make the DENSE
            # oracle's own V near-singular, and the comparison would test the oracle)
            θ = copy(fit.theta); θ[end] = -20.0; θ[3] = log(0.3)
            @test fit.nll(θ) ≈ dense_nll(θ, d.y, X, d.id, d.elapsed; ordinary) rtol = 1e-8
            @test loglik(fit) ≈ -dense_nll(fit.theta, d.y, X, d.id, d.elapsed; ordinary) atol = 1e-10
        end
    end

    @testset "fit recovery (S = 800 series, λ = 0.45, σ_t = 0.65, σ = 0.4)" begin
        d = sim_data(StableRNG(2027); S = 800)
        fit = drm(fOU, Gaussian(); data = d)
        tp = temporal_parameters(fit)
        println("temporal OU recovery (S = 800): decay = ", tp.decay, ", sd = ", tp.sd, ", sigma = ", tp.sigma,
                ", beta = ", coef(fit, :mu))
        @test fit.converged
        @test tp.structure === :ou && tp.phi === nothing && tp.sd_iid === nothing
        @test tp.decay ≈ 0.45 atol = 0.15
        @test tp.sd ≈ 0.65 atol = 0.12
        @test tp.sigma ≈ 0.4 atol = 0.12
        @test coef(fit, :mu) ≈ [0.8, 0.35] atol = 0.1
    end

    @testset "ordinary (1 | id) + OU" begin
        d = sim_data(StableRNG(77); S = 800, sb = 0.45)
        tp = temporal_parameters(drm(fOUi, Gaussian(); data = d))
        println("temporal OU + (1 | id) recovery (S = 800): decay = ", tp.decay, ", sd = ", tp.sd,
                ", sd_iid = ", tp.sd_iid, ", sigma = ", tp.sigma)
        @test tp.sd_iid ≈ 0.45 atol = 0.15
        @test tp.decay ≈ 0.45 atol = 0.25
    end

    @testset "invariances" begin
        d = sim_data(StableRNG(8); S = 20)
        fit = drm(fOU, Gaussian(); data = d)
        # time-origin shift: only elapsed gaps enter
        d2 = merge(d, (elapsed = d.elapsed .+ 1234.5,))
        fit2 = drm(fOU, Gaussian(); data = d2)
        @test fit2.nll(fit.theta) ≈ fit.nll(fit.theta) atol = 1e-9
        @test loglik(fit2) ≈ loglik(fit) atol = 1e-6
        # time-unit change: hours instead of days rescales λ by 1/24, logLik unchanged
        d3 = merge(d, (elapsed = 24 .* d.elapsed,))
        θ3 = copy(fit.theta); θ3[end] -= log(24)
        @test drm(fOU, Gaussian(); data = d3).nll(θ3) ≈ fit.nll(fit.theta) atol = 1e-9
        # OU on integer occasions is AR1 with φ = exp(−λ) > 0
        da = sim_data(StableRNG(9); S = 20)
        da = merge(da, (elapsed = round.(da.elapsed),))
        keep = [findfirst(i -> da.id[i] == da.id[j] && da.elapsed[i] == da.elapsed[j], eachindex(da.id)) == j
                for j in eachindex(da.id)]                    # drop (id, time) ties made by rounding
        da = map(c -> c[keep], da)
        fo = drm(fOU, Gaussian(); data = da)
        fa = drm(bf(@formula(y ~ x + temporal(1 | id, occ, ar1)), @formula(sigma ~ 1)), Gaussian();
                 data = merge(da, (occ = Int.(da.elapsed),)))
        θa = copy(fo.theta); θa[end] = atanh(exp(-exp(fo.theta[end])))
        @test fa.nll(θa) ≈ fo.nll(fo.theta) atol = 1e-9
        # row order
        p = reverse(eachindex(d.y))
        @test loglik(drm(fOU, Gaussian(); data = map(c -> c[p], d))) ≈ loglik(fit) atol = 1e-6
    end

    @testset "accessors" begin
        d = sim_data(StableRNG(32); S = 20, sb = 0.4)
        fit = drm(fOUi, Gaussian(); data = d)
        n = length(d.y); X = hcat(ones(n), d.x)
        lbl = "temporal(1 | id, time = elapsed, structure = \"ou\")"
        @test first.(fit.blocks) == [:mu, :sigma, :resd, :temporal_decay]
        @test Dict(fit.coefnames)[:resd] == ["id_iid", "id"]
        @test Dict(fit.coefnames)[:temporal_decay] == [lbl]
        tp = temporal_parameters(fit)
        @test tp.decay ≈ exp(only(coef(fit, :temporal_decay)))
        @test predict(fit, d) ≈ fitted(fit) ≈ X * coef(fit, :mu)
        @test structured_effects(fit) == [(dpar = :mu, kind = :temporal, grouping = :id)]
        # conditional modes against the dense model: temporal σ_t² R V⁻¹ r (rows),
        # intercept σ_b² Zᵀ V⁻¹ r (series in first-seen order)
        st2 = exp(2fit.theta[5]); sb2 = exp(2fit.theta[4]); λ = tp.decay
        R = [d.id[i] == d.id[j] ? exp(-λ * abs(d.elapsed[i] - d.elapsed[j])) : 0.0 for i in 1:n, j in 1:n]
        J = [d.id[i] == d.id[j] ? 1.0 : 0.0 for i in 1:n, j in 1:n]
        w = (st2 * R + sb2 * J + exp(2fit.theta[3]) * I) \ (d.y - X * coef(fit, :mu))
        @test ranef(fit)[:id] ≈ st2 * R * w atol = 1e-8
        levels = unique(d.id)
        @test ranef(fit)[:id_iid] ≈ [sb2 * sum(w[d.id .== l]) for l in levels] atol = 1e-8
        @test any(r -> r.param === :temporal_decay && r.scale === :log, profile_targets(fit))
        @test length(coeftable(fit).rownms) == 6
    end

    @testset "refusals" begin
        d = sim_data(StableRNG(4); S = 10)
        ae = ArgumentError
        dup = map(c -> [c; c[1]], d)
        @test_throws ae drm(fOU, Gaussian(); data = dup)                    # duplicate (id, time)
        # one distinct lag only (every series observed at 0 and 1)
        one_lag = (y = randn(StableRNG(1), 20), x = randn(StableRNG(2), 20),
                   id = repeat(["a$i" for i in 1:10], inner = 2), elapsed = repeat([0.0, 1.0], 10))
        @test_throws ae drm(fOU, Gaussian(); data = one_lag)
        # with (1 | id) three distinct lags are needed
        two_lags = (y = randn(StableRNG(3), 30), x = randn(StableRNG(4), 30),
                    id = repeat(["a$i" for i in 1:10], inner = 3), elapsed = repeat([0.0, 1.0, 3.0], 10))
        @test drm(fOU, Gaussian(); data = two_lags) isa DrmFit
        two_lags2 = merge(two_lags, (elapsed = repeat([0.0, 1.0, 2.0], 10),))
        @test_throws ae drm(fOUi, Gaussian(); data = two_lags2)
        # one series cannot carry an ordinary intercept
        one_series = (y = randn(StableRNG(5), 6), x = randn(StableRNG(6), 6), id = fill("a", 6),
                      elapsed = [0.0, 0.5, 2.0, 3.5, 7.0, 8.0])
        @test_throws ae drm(fOUi, Gaussian(); data = one_series)
        @test_throws ae drm(fOU, Gaussian(); data = merge(d, (elapsed = [i == 1 ? Inf : v for (i, v) in enumerate(d.elapsed)],)))
        @test_throws ae drm(bf(@formula(y ~ x + temporal(1 | id, days, ou)), @formula(sigma ~ 1)),
                            Gaussian(); data = d)                    # time column absent
        @test_throws ae drm(fOU, Gaussian(); data = d, method = :REML)
        @test_throws ae drm(bf(@formula(y ~ x + temporal(1 | id, elapsed, ou)), @formula(sigma ~ x)),
                            Gaussian(); data = d)
        @test_throws ae drm(bf(@formula(y ~ x + temporal(1 | id, elapsed, ou)), @formula(sigma ~ 1)),
                            DRModels.Gamma(); data = merge(d, (y = exp.(d.y),)))
    end

    @testset "fixture ou_irregular.csv" begin
        raw, hdr = readdlm(joinpath(@__DIR__, "fixtures", "temporal", "ou_irregular.csv"), ','; header = true)
        col(nm) = raw[:, findfirst(==(nm), vec(hdr))]
        d = (y = Float64.(col("y")), x = Float64.(col("x")), id = String.(col("id")),
             elapsed = Float64.(col("elapsed")))
        @test length(unique(d.id)) == 24
        fit = drm(fOUi, Gaussian(); data = d)
        @test fit.converged
        X = hcat(ones(length(d.y)), d.x)
        @test loglik(fit) ≈ -dense_nll(fit.theta, d.y, X, d.id, d.elapsed; ordinary = true) atol = 1e-10
        @test temporal_parameters(fit).decay > 0
    end
end
