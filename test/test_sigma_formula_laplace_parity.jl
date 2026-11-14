# test_sigma_formula_laplace_parity.jl -- Gamma, Beta, NegBinomial2 and ZeroOneBeta with one ordinary
# `(1 | g)` on the mean, `sigma ~ x1`, and `marginal = :Laplace`, against drmTMB 0.8.1. References:
# `test/fixtures/laplace_sigma_formula_parity/` written by
# `test/parity/gen_laplace_sigma_formula_parity.R` (max|grad| < 1e-6, all SEs finite; numbers and
# simulated data only). Bounds are the lane's and are never loosened:
#   logLik 1e-4 (absolute) | fixed effects 1e-3 R SE | log SD 5e-3.
# `fitted(fit)` (conditional: the stored random-effect modes are included) is compared with drmTMB's
# `fitted()` at 1e-6; `fitted(fit; re = :zero)` is the population value, pinned against the inverse
# link of the linear predictor at 1e-12 as before.
using DRModels, Test, TOML, Random, LinearAlgebra, Statistics

const _SFL_DIR = joinpath(@__DIR__, "fixtures", "laplace_sigma_formula_parity")

function _sfl_data()
    L = readlines(joinpath(_SFL_DIR, "data.csv"))
    hdr = [strip(c, '"') for c in split(L[1], ',')]
    cols = Dict{Symbol,Any}()
    for (j, h) in enumerate(hdr)
        raw = [strip(split(l, ',')[j], '"') for l in L[2:end]]
        cols[Symbol(h)] = all(s -> tryparse(Float64, s) !== nothing, raw) ? parse.(Float64, raw) : String.(raw)
    end
    cols[:y_nb] = round.(Int, cols[:y_nb])
    return NamedTuple{Tuple(keys(cols))}(Tuple(values(cols)))
end

const _SFL_CELLS = (
    (id = "gamma_sigma_x1", fam = DRModels.Gamma(), form = bf(@formula(y_gam ~ x1 + (1 | g)), @formula(sigma ~ x1)),
     mean = η -> exp.(η)),
    (id = "beta_sigma_x1", fam = DRModels.Beta(), form = bf(@formula(y_beta ~ x1 + (1 | g)), @formula(sigma ~ x1)),
     mean = η -> 1 ./ (1 .+ exp.(-η))),
    (id = "nb2_sigma_x1", fam = DRModels.NegBinomial2(), form = bf(@formula(y_nb ~ x1 + (1 | g)), @formula(sigma ~ x1)),
     mean = η -> exp.(η)),
    (id = "zob_sigma_x1", fam = DRModels.ZeroOneBeta(), form = bf(@formula(y_zoib ~ x1 + (1 | g)), @formula(sigma ~ x1)),
     mean = nothing),
)

@testset "sigma ~ x1 with (1 | g): marginal = :Laplace vs drmTMB 0.8.1" begin
    d = _sfl_data()
    for c in _SFL_CELLS
        ref = TOML.parsefile(joinpath(_SFL_DIR, "ref_$(c.id).toml"))
        @test ref["max_grad"] < 1e-6 && ref["drmTMB_version"] == "0.8.1"
        fit = drm(c.form, c.fam; data = d, marginal = :Laplace)
        @testset "$(c.id)" begin
            @test is_converged(fit)
            @test fit.marginal === :Laplace
            @test abs(loglik(fit) - ref["loglik"]) <= 1e-4
            θ = coef(fit); np = length(θ)
            @test np == length(ref["theta"])
            for k in 1:np
                tol = k == np ? 5e-3 : 1e-3 * ref["se"][k]
                @test abs(θ[k] - ref["theta"][k]) <= tol
            end
            if c.mean !== nothing
                @test maximum(abs.(fitted(fit; re = :zero) .- c.mean(θ[1] .+ θ[2] .* d.x1))) <= 1e-12
            end
            @test maximum(abs.(fitted(fit) .- Float64.(ref["fitted"]))) <= 1e-6
            # the default integrator is unchanged and gives a different number
            fla = drm(c.form, c.fam; data = d)
            @test fla.marginal === :LA
            @test abs(loglik(fla) - loglik(fit)) > 1e-3
            fex = drm(c.form, c.fam; data = d, marginal = :LA)
            @test coef(fex) == coef(fla) && loglik(fex) == loglik(fla)
        end
    end
end

@testset "sigma ~ x1 :Laplace post-fit functions" begin
    d = _sfl_data(); n = length(d.g)
    for c in (_SFL_CELLS[1], _SFL_CELLS[4])
        @testset "$(c.id)" begin
            fit = drm(c.form, c.fam; data = d, marginal = :Laplace)
            fla = drm(c.form, c.fam; data = d)
            p = length(coef(fit))
            @test nobs(fit) == n
            V = vcov(fit)
            @test size(V) == (p, p) && issymmetric(V) && minimum(eigvals(Symmetric(V))) > 0
            @test aic(fit) ≈ -2 * loglik(fit) + 2p
            # the objective stored on the fit is the Laplace one, not the 5-node quadrature one
            θ̂ = coef(fit)
            @test fit.nll(θ̂) ≈ -loglik(fit) atol = 1e-9
            @test abs(fla.nll(θ̂) - fit.nll(θ̂)) > 1e-3
            @test length(predict(fit, d)) == n
            @test length(residuals(fit; type = :quantile)) == n
            @test re_sd(fit)[:g] ≈ exp(coef(fit, :resd)[1])
            @test_throws ArgumentError lrtest(fla, fit)
            # bootstrap: every replicate is refitted with :Laplace
            br = bootstrap_result(fit; data = d, B = 2, rng = Random.Xoshiro(1))
            @test br.used == 2 && br.marginal === :Laplace && br.refit_marginal === :Laplace
            sim = DRModels._marginal_simulator(fit, d)
            reps = map(br.seeds) do seed
                datab = DRModels._bootstrap_data(fit.formula, d, sim(Random.MersenneTwister(seed)))
                (lap = drm(c.form, c.fam; data = datab, marginal = :Laplace), la = drm(c.form, c.fam; data = datab))
            end
            @test all(r -> r.lap.marginal === :Laplace, reps)
            rep_sd(k) = vec(Statistics.std(permutedims(reduce(hcat, [coef(getproperty(r, k)) for r in reps])); dims = 1))
            boot_sd = [row.std_error for row in br.summary]
            @test boot_sd ≈ rep_sd(:lap) rtol = 1e-8
            @test !isapprox(boot_sd, rep_sd(:la); rtol = 1e-6)
        end
    end
end

@testset "sigma ~ x1 :Laplace refusals" begin
    d = _sfl_data()
    # shapes outside the covered ones keep refusing by name
    @test_throws ArgumentError drm(bf(@formula(y_gam ~ x1 + (1 | g)), @formula(sigma ~ x1 + (1 | g))), DRModels.Gamma();
                                   data = d, marginal = :Laplace)
    @test_throws ArgumentError drm(bf(@formula(y_gam ~ x1 + offset(x1) + (1 | g)), @formula(sigma ~ x1)), DRModels.Gamma();
                                   data = d, marginal = :Laplace)
    zf(f) = drm(f, DRModels.ZeroOneBeta(); data = d, marginal = :Laplace)
    @test_throws ArgumentError zf(bf(@formula(y_zoib ~ x1), @formula(sigma ~ x1)))
    @test_throws ArgumentError zf(bf(@formula(y_zoib ~ x1 + (0 + x1 | g)), @formula(sigma ~ x1)))
    @test_throws ArgumentError zf(bf(@formula(y_zoib ~ x1 + (1 | g)), @formula(sd(g) ~ x1)))
    for m in (:VA, :AGHQ, :bogus)
        @test_throws ArgumentError drm(bf(@formula(y_zoib ~ x1 + (1 | g)), @formula(sigma ~ x1)),
                                       DRModels.ZeroOneBeta(); data = d, marginal = m)
    end
    e = try zf(bf(@formula(y_zoib ~ x1), @formula(sigma ~ x1))) catch err; sprint(showerror, err) end
    @test occursin("marginal = :Laplace", e) && occursin("ZeroOneBeta", e)
end
