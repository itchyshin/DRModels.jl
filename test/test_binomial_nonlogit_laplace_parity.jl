# test_binomial_nonlogit_laplace_parity.jl -- Binomial with a :probit or :cloglog link, one ordinary
# random intercept `(1 | g)` or one correlated block `(1 + x1 | g)`, and `marginal = :Laplace`, against
# drmTMB 0.8.1. References: `test/fixtures/laplace_more_shapes_parity/` written by
# `test/parity/gen_laplace_more_shapes_parity.R` (max|grad| < 1e-6, all SEs finite; numbers and
# simulated data only). Bounds are the lane's and are never loosened:
#   logLik 1e-4 (absolute) | fixed effects 1e-3 R SE | log SD 5e-3.
# `fitted(fit)` (conditional: the stored random-effect modes are included) is compared with drmTMB's
# `fitted()` at 1e-6; `fitted(fit; re = :zero)` is the population value, pinned against the inverse
# link of the linear predictor at 1e-12 as before.
using DRModels, Test, TOML, Random, LinearAlgebra, Statistics
using SpecialFunctions: erf

const _BNL_DIR = joinpath(@__DIR__, "fixtures", "laplace_more_shapes_parity")

function _bnl_data()
    L = readlines(joinpath(_BNL_DIR, "data.csv"))
    hdr = [strip(c, '"') for c in split(L[1], ',')]
    cols = Dict{Symbol,Any}()
    for (j, h) in enumerate(hdr)
        raw = [strip(split(l, ',')[j], '"') for l in L[2:end]]
        cols[Symbol(h)] = all(s -> tryparse(Float64, s) !== nothing, raw) ? parse.(Float64, raw) : String.(raw)
    end
    for k in (:y_bin, :s, :fl)
        cols[k] = round.(Int, cols[k])
    end
    return NamedTuple{Tuple(keys(cols))}(Tuple(values(cols)))
end

const _BNL_CELLS = (
    (id = "binom_probit_ri",     link = :probit,  form = bf(@formula(y_bin ~ x1 + (1 | g))), npar_free = 3),
    (id = "binom_cloglog_ri",    link = :cloglog, form = bf(@formula(y_bin ~ x1 + (1 | g))), npar_free = 3),
    # the correlation of the probit cell sits on its boundary in both engines (R: eta_cor = -10.8,
    # SE 1e4; Julia: about -14), so only the two fixed effects and the two log SDs are compared there
    (id = "binom_probit_slope",  link = :probit,  form = bf(@formula(cbind(s, fl) ~ x1 + (1 + x1 | g))), npar_free = 4),
    (id = "binom_cloglog_slope", link = :cloglog, form = bf(@formula(cbind(s, fl) ~ x1 + (1 + x1 | g))), npar_free = 5),
)

@testset "Binomial probit/cloglog marginal = :Laplace vs drmTMB 0.8.1" begin
    d = _bnl_data()
    for c in _BNL_CELLS
        ref = TOML.parsefile(joinpath(_BNL_DIR, "ref_$(c.id).toml"))
        @test ref["max_grad"] < 1e-6 && ref["drmTMB_version"] == "0.8.1"
        fam = DRModels.Binomial(c.link)
        fit = drm(c.form, fam; data = d, marginal = :Laplace)
        @testset "$(c.id)" begin
            @test fit.marginal === :Laplace
            @test abs(loglik(fit) - ref["loglik"]) <= 1e-4
            θ = coef(fit); V = vcov(fit)
            pfix = 2
            for k in 1:c.npar_free
                r = ref["theta"][k]
                tol = k <= pfix ? 1e-3 * ref["se"][k] : 5e-3
                @test abs(θ[k] - r) <= tol
            end
            Xβ = θ[1] .+ θ[2] .* d.x1
            invlink = c.link === :probit ? (η -> 0.5 * (1 + erf(η / sqrt(2)))) :
                                           (η -> -expm1(-exp(η)))
            @test maximum(abs.(fitted(fit; re = :zero) .- invlink.(Xβ))) <= 1e-12
            @test maximum(abs.(fitted(fit) .- Float64.(ref["fitted"]))) <= 1e-6
            # the default integrator is unchanged and gives a different number
            fla = drm(c.form, fam; data = d)
            @test fla.marginal === :LA
            @test abs(loglik(fla) - loglik(fit)) > 1e-3
            fex = drm(c.form, fam; data = d, marginal = :LA)
            @test coef(fex) == coef(fla) && loglik(fex) == loglik(fla)
        end
    end
end

@testset "Binomial probit :Laplace (1 | g) post-fit functions" begin
    d = _bnl_data(); n = length(d.y_bin)
    form = bf(@formula(y_bin ~ x1 + (1 | g)))
    fam = DRModels.Binomial(:probit)
    fit = drm(form, fam; data = d, marginal = :Laplace)
    fla = drm(form, fam; data = d)
    p = length(coef(fit))
    @test nobs(fit) == n
    @test loglikelihood(fit) == loglik(fit)
    V = vcov(fit)
    @test size(V) == (p, p) && issymmetric(V) && minimum(eigvals(Symmetric(V))) > 0
    @test aic(fit) ≈ -2 * loglik(fit) + 2p
    # the objective stored on the fit is the Laplace one, not the 32-node quadrature one
    θ̂ = coef(fit)
    @test fit.nll(θ̂) ≈ -loglik(fit) atol = 1e-9
    @test abs(fla.nll(θ̂) - fit.nll(θ̂)) > 1e-3
    @test length(predict(fit, d)) == n
    @test length(simulate(fit; data = d, rng = Random.Xoshiro(1))) == n
    @test length(residuals(fit; type = :quantile)) == n
    @test re_sd(fit)[:g] ≈ exp(coef(fit, :resd)[1])
    ci = confint(fit)
    @test all(c -> c.lower < c.estimate < c.upper, ci)
    cp = confint(fit; method = :profile)
    cpl = confint(fla; method = :profile)
    @test all(c -> c.lower < c.estimate < c.upper, cp)
    @test maximum(abs.([a.lower - b.lower for (a, b) in zip(cp, cpl)])) > 1e-6
    # lrtest against a :LA fit is refused
    @test_throws ArgumentError lrtest(fla, fit)
    # bootstrap: every replicate is refitted with :Laplace
    br = bootstrap_result(fit; data = d, B = 3, rng = Random.Xoshiro(1))
    @test br.used == 3 && br.marginal === :Laplace && br.refit_marginal === :Laplace
    sim = DRModels._marginal_simulator(fit, d)
    reps = map(br.seeds) do seed
        datab = DRModels._bootstrap_data(fit.formula, d, sim(Random.MersenneTwister(seed)))
        (lap = drm(form, fam; data = datab, marginal = :Laplace), la = drm(form, fam; data = datab))
    end
    @test all(r -> r.lap.marginal === :Laplace, reps)
    rep_sd(k) = vec(Statistics.std(permutedims(reduce(hcat, [coef(getproperty(r, k)) for r in reps])); dims = 1))
    boot_sd = [row.std_error for row in br.summary]
    @test boot_sd ≈ rep_sd(:lap) rtol = 1e-8
    @test !isapprox(boot_sd, rep_sd(:la); rtol = 1e-6)
end

@testset "Binomial cloglog :Laplace (1 + x1 | g) post-fit functions" begin
    d = _bnl_data(); n = length(d.y_bin)
    form = bf(@formula(cbind(s, fl) ~ x1 + (1 + x1 | g)))
    fam = DRModels.Binomial(:cloglog)
    fit = drm(form, fam; data = d, marginal = :Laplace)
    @test fit.marginal === :Laplace && is_converged(fit)
    @test isfinite(loglik(fit)) && length(residuals(fit; type = :quantile)) == n
    # a correlated block has no fresh random-effect draw: the bootstrap is refused by name (bootstrap
    # refits keeping the option are pinned on the `(1 | g)` cells above)
    for call in (() -> bootstrap_result(fit; data = d, B = 2, rng = Random.Xoshiro(1)),
                 () -> bootstrap_summary(fit; data = d, B = 2, rng = Random.Xoshiro(1)))
        err = try; call(); nothing; catch e; e; end
        @test err isa ArgumentError && occursin("no fresh random-effect draw", sprint(showerror, err))
    end
    # not available on this shape, refused by name rather than computed another way
    @test_throws ArgumentError ranef(fit)
    @test_throws ArgumentError simulate(fit; data = d, rng = Random.Xoshiro(1))
end

@testset "Binomial :Laplace refusals and the logit route" begin
    d = _bnl_data()
    # still refused for non-logit links outside the admitted shapes
    for (what, bad) in (
            "random slope" => bf(@formula(y_bin ~ x1 + (0 + x1 | g))),
            "fixed effects only" => bf(@formula(y_bin ~ x1)))
        @testset "$what" begin
            @test_throws ArgumentError drm(bad, DRModels.Binomial(:probit); data = d, marginal = :Laplace)
        end
    end
    @test_throws ArgumentError drm(bf(@formula(y_bin ~ x1 + (1 | g))), DRModels.Binomial(:probit);
                                   data = d, marginal = :VA)
    @test_throws ArgumentError drm(bf(@formula(y_bin ~ x1 + (1 | g))), DRModels.Binomial(:probit);
                                   data = d, marginal = :Laplace, method = :REML)
    # the logit link keeps its own (sparse) Laplace route, tagged the same way
    fl = drm(bf(@formula(y_bin ~ x1 + (1 | g))), DRModels.Binomial(); data = d, marginal = :Laplace)
    @test fl.marginal === :Laplace && isfinite(loglik(fl))
end
