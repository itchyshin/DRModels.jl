# Prior `weights` (drmTMB: row log-likelihood multipliers) on the latent-term routes that used to refuse them:
#   Gaussian `(1 | g)` (ML and REML), Poisson `(1 | g)`, Gaussian `meta_V`, bivariate Gaussian (residual route).
# Reference: test/fixtures/weights_re/r_reference.tsv, drmTMB 0.8.1, made by test/fixtures/weights_re/make_r.R on
# the committed data_weights.csv. Bounds fixed before the reference was read:
#   |loglik - R| <= 1e-4;  fixed effects <= 1e-3 R SE;  log SD <= 5e-3;  SEs <= 1e-3 relative.
# Independent oracles (no R): integer weights equal replicated rows (a replicated row shares its group effect),
# unit weights equal no weights, and every shape that is not wired keeps a refusal.
using DRModels, Test, DelimitedFiles, LinearAlgebra, Random, QuadGK

const _WR_DIR = joinpath(@__DIR__, "fixtures", "weights_re")

function _wr_data()
    lines = readlines(joinpath(_WR_DIR, "data_weights.csv"))
    hdr = String.(strip.(split(lines[1], ','), '"'))
    rows = [split(l, ',') for l in lines[2:end]]
    col(c) = [r[findfirst(==(c), hdr)] for r in rows]
    num(c) = parse.(Float64, col(c))
    return (; id = num("id"), g = String.(strip.(col("g"), '"')), x1 = num("x1"), x2 = num("x2"), w = num("w"),
              wi = num("wi"), wz = num("wz"), y_g = num("y_g"), y_pois = num("y_pois"), y1 = num("y1"),
              y2 = num("y2"), vi = num("vi"), yi = num("yi"))
end

function _wr_reference()
    ref = Dict{String,Vector{Tuple{String,Float64,Float64}}}()
    for l in readlines(joinpath(_WR_DIR, "r_reference.tsv"))[2:end]
        f = split(l, '\t')
        push!(get!(ref, String(f[1]), Tuple{String,Float64,Float64}[]),
              (String(f[2]), parse(Float64, f[3]), f[4] == "NA" ? NaN : parse(Float64, f[4])))
    end
    return ref
end

const _WR_D = _wr_data()
const _WR_REF = _wr_reference()
_wr_get(model, key) = (r = _WR_REF[model]; i = findfirst(t -> t[1] == key, r); r[i][2:3])

# `sdkeys`: R rows holding a natural-scale SD (compared on the log scale, SE by the delta rule).
function _wr_check(fit, model; sdkeys = String[], se_rtol = 1e-3)
    rows = _WR_REF[model]
    coefs = [t for t in rows if startswith(t[1], "coefficients:")]
    sds = [_wr_get(model, k) for k in sdkeys]
    th = fit.theta
    se = stderror(fit)
    @test length(th) == length(coefs) + length(sds)
    @test abs(loglikelihood(fit) - _wr_get(model, "ll")[1]) <= 1e-4
    @test _wr_get(model, "conv")[1] == 1 && fit.converged
    for (k, t) in enumerate(coefs)
        @test abs(th[k] - t[2]) <= 1e-3 * t[3]
        @test abs(se[k] - t[3]) <= se_rtol * t[3]
    end
    for (j, (est, sdse)) in enumerate(sds)
        k = length(coefs) + j
        @test abs(th[k] - log(est)) <= 5e-3
        @test abs(se[k] - sdse / est) <= se_rtol * sdse / est
    end
end

_wr_rep(d, w) = (idx = reduce(vcat, [fill(i, Int(w[i])) for i in eachindex(w)]);
                 NamedTuple{keys(d)}(map(c -> c[idx], values(d))))

@testset "prior weights on latent-term routes (drmTMB 0.8.1 parity)" begin
    d = _WR_D
    sd_g = "parameters:sd:mu:(1 | g)"

    @testset "Gaussian (1 | g): ML, REML, sigma ~ x2, integer and zero weights" begin
        f = bf(@formula(y_g ~ x1 + (1 | g)), @formula(sigma ~ 1))
        _wr_check(drm(f, Gaussian(); data = d, weights = d.w), "gauss_re_ml"; sdkeys = [sd_g])
        _wr_check(drm(f, Gaussian(); data = d, weights = d.w, method = :REML), "gauss_re_reml"; sdkeys = [sd_g])
        _wr_check(drm(f, Gaussian(); data = d, weights = d.wi), "gauss_re_int"; sdkeys = [sd_g])
        _wr_check(drm(f, Gaussian(); data = d, weights = d.wz), "gauss_re_zero"; sdkeys = [sd_g])
        f2 = bf(@formula(y_g ~ x1 + (1 | g)), @formula(sigma ~ x2))
        _wr_check(drm(f2, Gaussian(); data = d, weights = d.w), "gauss_re_sig"; sdkeys = [sd_g])
    end

    @testset "Gaussian (1 | g): oracles" begin
        f = bf(@formula(y_g ~ x1 + (1 | g)), @formula(sigma ~ 1))
        base = drm(f, Gaussian(); data = d)
        ones_fit = drm(f, Gaussian(); data = d, weights = ones(length(d.y_g)))
        @test ones_fit.theta ≈ base.theta rtol = 1e-10
        @test loglikelihood(ones_fit) ≈ loglikelihood(base) rtol = 1e-12
        # integer weights = replicated rows (replicates share the group effect)
        dr = _wr_rep(d, d.wi)
        for m in (:ML, :REML)
            wf = drm(f, Gaussian(); data = d, weights = d.wi, method = m)
            rf = drm(f, Gaussian(); data = dr, method = m)
            @test loglikelihood(wf) ≈ loglikelihood(rf) atol = 1e-7
            @test wf.theta ≈ rf.theta atol = 1e-5
        end
    end

    @testset "weights follow rows dropped for a missing response" begin
        f = bf(@formula(y_g ~ x1 + (1 | g)), @formula(sigma ~ 1))
        ym = Union{Missing,Float64}[i % 11 == 3 ? missing : v for (i, v) in enumerate(d.y_g)]
        dm = merge(d, (; y_g = ym))
        keep = findall(!ismissing, ym)
        dk = NamedTuple{keys(d)}(map(c -> c[keep], values(dm)))
        dk = merge(dk, (; y_g = Float64.(ym[keep])))
        fa = DRModels.drm(f, Gaussian(); data = dm, weights = d.w)
        fb = DRModels.drm(f, Gaussian(); data = dk, weights = d.w[keep])
        @test fa.theta ≈ fb.theta rtol = 1e-8
        @test loglikelihood(fa) ≈ loglikelihood(fb) atol = 1e-8
    end

    @testset "Poisson (1 | g), marginal = :Laplace (the TMB convention drmTMB uses)" begin
        f = bf(@formula(y_pois ~ x1 + (1 | g)))
        _wr_check(drm(f, Poisson(); data = d, weights = d.w, marginal = :Laplace), "pois_re"; sdkeys = [sd_g])
        _wr_check(drm(f, Poisson(); data = d, weights = d.wi, marginal = :Laplace), "pois_re_int"; sdkeys = [sd_g])
        _wr_check(drm(f, Poisson(); data = d, weights = d.wz, marginal = :Laplace), "pois_re_zero"; sdkeys = [sd_g])
    end

    @testset "Poisson (1 | g): default marginal and oracles" begin
        f = bf(@formula(y_pois ~ x1 + (1 | g)))
        base = drm(f, Poisson(); data = d)
        @test drm(f, Poisson(); data = d, weights = ones(length(d.y_pois))).theta ≈ base.theta rtol = 1e-8
        dr = _wr_rep(d, d.wi)
        for marg in (:LA, :Laplace)
            wf = drm(f, Poisson(); data = d, weights = d.wi, marginal = marg)
            rf = drm(f, Poisson(); data = dr, marginal = marg)
            @test loglikelihood(wf) ≈ loglikelihood(rf) atol = 1e-7
            @test wf.theta ≈ rf.theta atol = 1e-5
        end
        # the default (exact-integral) marginal differs from R's Laplace by the marginal, not by the weights
        @test abs(loglikelihood(drm(f, Poisson(); data = d, weights = d.w)) - _wr_get("pois_re", "ll")[1]) < 0.05
        # not wired: refused by name, never run with the weights ignored
        @test_throws ArgumentError drm(f, Poisson(); data = d, weights = d.w, method = :REML)
        @test_throws ArgumentError drm(f, Poisson(); data = d, weights = d.w, marginal = :AGHQ)
    end

    @testset "Gaussian meta_V(vi)" begin
        f = bf(@formula(yi ~ x1 + meta_V(vi)), @formula(sigma ~ 1))
        _wr_check(drm(f, Gaussian(); data = d, weights = d.w), "meta")
        _wr_check(drm(f, Gaussian(); data = d, weights = d.wi), "meta_int")
        _wr_check(drm(f, Gaussian(); data = d, weights = d.wz), "meta_zero")
        base = drm(f, Gaussian(); data = d)
        @test drm(f, Gaussian(); data = d, weights = ones(length(d.yi))).theta ≈ base.theta rtol = 1e-10
        rf = drm(f, Gaussian(); data = _wr_rep(d, d.wi))
        @test loglikelihood(drm(f, Gaussian(); data = d, weights = d.wi)) ≈ loglikelihood(rf) atol = 1e-8
        # a meta model with a random effect is still refused, as is REML
        @test_throws ArgumentError drm(bf(@formula(yi ~ x1 + meta_V(vi) + (1 | g)), @formula(sigma ~ 1)),
                                       Gaussian(); data = d, weights = d.w)
        # `weights` with `meta_V` and REML is refused by name (a weighted restriction term is not
        # implemented), with or without a mean random intercept, whether or not `meta_V` REML exists.
        for ff in (f, bf(@formula(yi ~ x1 + meta_V(vi) + (1 | g)), @formula(sigma ~ 1)))
            err = try drm(ff, Gaussian(); data = d, weights = d.w, method = :REML); nothing catch e; e end
            @test err isa ArgumentError
            msg = err isa Exception ? sprint(showerror, err) : ""
            @test occursin("`weights` together with `meta_V` and method = :REML", msg)
        end
    end

    @testset "bivariate Gaussian (residual route)" begin
        f = bf(mu1 = @formula(y1 ~ x1), mu2 = @formula(y2 ~ x2), sigma1 = @formula(sigma1 ~ 1),
               sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
        _wr_check(drm(f, Gaussian(); data = d, weights = d.w), "biv")
        _wr_check(drm(f, Gaussian(); data = d, weights = d.wz), "biv_zero")
        base = drm(f, Gaussian(); data = d)
        @test drm(f, Gaussian(); data = d, weights = ones(length(d.y1))).theta ≈ base.theta rtol = 1e-10
        rf = drm(f, Gaussian(); data = _wr_rep(d, d.wi))
        @test loglikelihood(drm(f, Gaussian(); data = d, weights = d.wi)) ≈ loglikelihood(rf) atol = 1e-8
        @test_throws ArgumentError drm(f, Gaussian(); data = d, weights = d.w, method = :REML)
    end

    @testset "input checks and the shapes that stay refused" begin
        f = bf(@formula(y_g ~ x1 + (1 | g)), @formula(sigma ~ 1))
        n = length(d.y_g)
        @test_throws ArgumentError drm(f, Gaussian(); data = d, weights = fill(0.0, n))            # all zero
        @test_throws ArgumentError drm(f, Gaussian(); data = d, weights = vcat(-1.0, d.w[2:end]))   # negative
        @test_throws ArgumentError drm(f, Gaussian(); data = d, weights = vcat(NaN, d.w[2:end]))    # non-finite
        @test_throws DimensionMismatch drm(f, Gaussian(); data = d, weights = d.w[1:end-1])         # wrong length
        # random slope, crossed intercepts and the optimiser fields keep the original refusal
        @test_throws ArgumentError drm(bf(@formula(y_g ~ x1 + (1 + x1 | g)), @formula(sigma ~ 1)), Gaussian();
                                       data = d, weights = d.w)
        @test_throws ArgumentError drm(f, Gaussian(); data = d, weights = d.w, control = DRMControl(multi_start = 2))
        # bivariate with a random effect: refused (the ordinary-random-effect bivariate route is not available)
        fb = bf(mu1 = @formula(y1 ~ x1 + (1 | p | g)), mu2 = @formula(y2 ~ x2 + (1 | p | g)),
                sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
        @test_throws ArgumentError drm(fb, Gaussian(); data = d, weights = d.w)
    end
end

@testset "weights reach refit helpers (bootstrap, update) and edge weights" begin
    d = _WR_D
    n = length(d.y_g)
    fb = bf(mu1 = @formula(y1 ~ x1), mu2 = @formula(y2 ~ x2), sigma1 = @formula(sigma1 ~ 1),
            sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))

    @testset "bivariate bootstrap refits keep the weights" begin
        fw = drm(fb, Gaussian(); data = d, weights = d.w)
        @test DRModels._bootstrap_refit_kwargs(fw).weights == d.w
        @test get(DRModels._bootstrap_bivariate_refit_kwargs(DRModels._bootstrap_refit_kwargs(fw)), :weights, nothing) == d.w
        bs(fit) = bootstrap_result(fit; data = d, B = 6, rng = MersenneTwister(5), check_converged = false).summary
        f0 = drm(fb, Gaussian(); data = d)
        f1 = drm(fb, Gaussian(); data = d, weights = ones(n))
        @test bs(f1) == bs(f0)            # unit weights = no weights, same seed
        @test bs(fw) != bs(f0)            # real weights change the refits
        # Sensitive to the refit closure: replay the two replicates by hand (same seeds, same simulated
        # data) and refit WITH and WITHOUT the weights. The bootstrap std_error of the first coefficient
        # must equal the weighted replay, and the unweighted replay must differ from it.
        br = bootstrap_result(fw; data = d, B = 2, rng = MersenneTwister(5), check_converged = false)
        simfn = DRModels._marginal_simulator(fw, d; K = nothing, A = nothing, tree = nothing, coords = nothing)
        function replay(w)
            est = map(br.seeds) do sd
                rr = MersenneTwister(sd)
                ys = simfn === nothing ? simulate(fw; rng = rr, re = :zero) : simfn(rr)
                datab = DRModels._bootstrap_data(fb, d, ys)
                coef(drm(fb, Gaussian(); data = datab, weights = w))[1]
            end
            return abs(est[1] - est[2]) / sqrt(2)             # std of two draws
        end
        @test br.summary[1].std_error ≈ replay(d.w) rtol = 1e-6
        @test abs(replay(nothing) - br.summary[1].std_error) > 1e-3 * br.summary[1].std_error
    end

    @testset "update never refits a weighted model unweighted" begin
        f = bf(@formula(y_g ~ x1 + (1 | g)), @formula(sigma ~ 1))
        ffix = bf(@formula(y_g ~ x1), @formula(sigma ~ 1))
        for (fm, fit) in ((ffix, drm(ffix, Gaussian(); data = d, weights = d.w)),
                          (f, drm(f, Gaussian(); data = d, weights = d.w)))
            @test_throws ArgumentError update(fit, fm; data = d)                       # stored weights, none given
            up = update(fit, fm; data = d, weights = d.w)                               # explicit weights pass through
            @test up.theta ≈ fit.theta rtol = 1e-6
            @test weights(up) == d.w
            @test weights(update(fit, fm; data = d, weights = nothing)) === nothing     # explicit opt-out is allowed
        end
        fu = drm(f, Gaussian(); data = d)
        @test update(fu, f; data = d).theta ≈ fu.theta rtol = 1e-6                     # unweighted: unchanged
    end

    @testset "zero weights equal dropped rows, REML included; an all-zero group" begin
        f = bf(@formula(y_g ~ x1 + (1 | g)), @formula(sigma ~ 1))
        keep = findall(>(0), d.wz)
        dk = NamedTuple{keys(d)}(map(c -> c[keep], values(d)))
        for m in (:ML, :REML)
            a = drm(f, Gaussian(); data = d, weights = d.wz, method = m)
            b = drm(f, Gaussian(); data = dk, weights = d.w[keep], method = m)
            @test loglikelihood(a) ≈ loglikelihood(b) atol = 1e-7
            @test a.theta ≈ b.theta atol = 1e-5
        end
        wg = d.w .* (d.g .!= "g01")                          # every row of group g01 has weight 0
        keepg = findall(>(0), wg)
        dg = NamedTuple{keys(d)}(map(c -> c[keepg], values(d)))
        a = drm(f, Gaussian(); data = d, weights = wg)
        b = drm(f, Gaussian(); data = dg, weights = d.w[keepg])
        @test loglikelihood(a) ≈ loglikelihood(b) atol = 1e-7
        @test a.theta ≈ b.theta atol = 1e-5
        fp = bf(@formula(y_pois ~ x1 + (1 | g)))
        for kw in ((;), (; marginal = :Laplace))
            a = drm(fp, Poisson(); data = d, weights = wg, kw...)
            b = drm(fp, Poisson(); data = dg, weights = d.w[keepg], kw...)
            @test loglikelihood(a) ≈ loglikelihood(b) atol = 1e-6
            @test a.theta ≈ b.theta atol = 1e-4
        end
    end

    @testset "large non-integer weights, Poisson default marginal vs brute-force quadrature" begin
        # Gap (fit minus brute-force quadgk of the weighted likelihood at the fitted parameters), measured
        # on Totoro for max weight 1.97 / 9.4 / 37 / 71 / 187: -3.7e-5, -2.9e-4, -1.9e-4, -1.6e-4, -1.6e-4.
        # It does not grow with the weights (finite-node adaptive quadrature), so 1e-3 is the largest
        # measured gap (2.9e-4) with 3x headroom.
        fp = bf(@formula(y_pois ~ x1 + (1 | g)))
        for s in (1, 5, 20, 38, 100)
            wbig = d.w .* (1 .+ (s - 1) .* (d.id .% 5 .== 0))
            fit = drm(fp, Poisson(); data = d, weights = wbig)
            β = fit.theta[1:2]; σb = exp(fit.theta[3])
            ll = 0.0
            for gl in unique(d.g)
                idx = findall(==(gl), d.g)
                η0 = β[1] .+ β[2] .* d.x1[idx]
                lf(b) = sum(wbig[idx] .* (d.y_pois[idx] .* (η0 .+ b) .- exp.(η0 .+ b))) -
                        sum(wbig[idx] .* [sum(log, 1:Int(y); init = 0.0) for y in d.y_pois[idx]]) - b^2 / (2σb^2) - log(σb * sqrt(2π))
                m = maximum(lf.(range(-12σb, 12σb; length = 4001)))
                val, _ = quadgk(b -> exp(lf(b) - m), collect(range(-12σb, 12σb; length = 801))...; rtol = 1e-12, atol = 1e-14)
                ll += m + log(val)
            end
            @test loglikelihood(fit) ≈ ll atol = 1e-3
        end
    end
end
