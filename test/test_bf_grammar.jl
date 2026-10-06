# bf() grammar parity: the formula front end rejects the same reserved / mis-typed
# syntax drmTMB rejects, with clear parallel error messages. Front-end only — no
# fitting; bf builds the formula bundle from the parameter on each LHS.
using DRModels
using Test

@testset "bf() grammar: valid forms still build" begin
    @test bf(@formula(y ~ x)) isa DrmFormula
    @test bf(@formula(y ~ x), @formula(sigma ~ x)) isa DrmFormula
    @test bf(@formula(y ~ x), @formula(sigma ~ x), @formula(nu ~ 1)) isa DrmFormula
    # zero-inflation / hurdle / zero-one-inflation parameters are accepted
    @test bf(@formula(y ~ x), @formula(zi ~ x)) isa DrmFormula
    @test bf(@formula(y ~ x), @formula(hu ~ 1)) isa DrmFormula
    @test bf(@formula(y ~ x), @formula(zoi ~ 1), @formula(coi ~ 1)) isa DrmFormula
    # cbind two-column response still parses
    @test bf(@formula(cbind(s, f) ~ x), @formula(sigma ~ 1)) isa DrmFormula
    # sigma defaults to ~ 1 when omitted
    @test any(p -> first(p) === :sigma, bf(@formula(y ~ x)).forms)
end

@testset "bf() grammar: reserved-syntax rejections" begin
    # `tau` is not a parameter name — the scale is `sigma`
    @test_throws ArgumentError bf(@formula(y ~ x), @formula(tau ~ x))
    # μ comes from the response formula, not a separate `mu ~ …`
    @test_throws ArgumentError bf(@formula(y ~ x), @formula(mu ~ x))
    # bivariate parameters are invalid in the positional (univariate) form
    @test_throws ArgumentError bf(@formula(y ~ x), @formula(rho12 ~ x))
    @test_throws ArgumentError bf(@formula(y ~ x), @formula(sigma1 ~ x))
    @test_throws ArgumentError bf(@formula(y ~ x), @formula(mu2 ~ x))
    # unknown distributional parameter
    @test_throws ArgumentError bf(@formula(y ~ x), @formula(theta ~ x))
    # a parameter given twice
    @test_throws ArgumentError bf(@formula(y ~ x), @formula(sigma ~ x), @formula(sigma ~ z))
end

@testset "bf() grammar: bivariate keyword form" begin
    # valid bivariate form still builds (placeholder LHS = parameter name)
    @test bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
             sigma1 = @formula(sigma1 ~ x), sigma2 = @formula(sigma2 ~ x),
             rho12 = @formula(rho12 ~ x)) isa BivariateDrmFormula
    # σ1/σ2/ρ12 default to ~ 1 when omitted
    @test bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x)) isa BivariateDrmFormula

    # a placeholder LHS that is not its own parameter name is rejected
    @test_throws ArgumentError bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                                  sigma1 = @formula(tau ~ x))
    @test_throws ArgumentError bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                                  rho12 = @formula(rho ~ x))
    # swapped sigma1/sigma2 placeholders are caught
    @test_throws ArgumentError bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                                  sigma1 = @formula(sigma2 ~ x))
    # a two-column cbind response is univariate-only
    @test_throws ArgumentError bf(mu1 = @formula(cbind(s, f) ~ x), mu2 = @formula(y2 ~ x))

    # the tau→sigma message points at the right placeholder
    e_tau = try; bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                    sigma1 = @formula(tau ~ x)); catch e; e; end
    @test e_tau isa ArgumentError && occursin("sigma1", e_tau.msg)
end

@testset "bf() grammar: error messages are actionable" begin
    # the `tau` error points at `sigma`
    e_tau = try; bf(@formula(y ~ x), @formula(tau ~ x)); catch e; e; end
    @test e_tau isa ArgumentError && occursin("sigma", e_tau.msg)
    # the bivariate-parameter error points at the keyword form
    e_rho = try; bf(@formula(y ~ x), @formula(rho12 ~ x)); catch e; e; end
    @test e_rho isa ArgumentError && occursin("mu1", e_rho.msg)
    # the unknown-parameter error lists the valid names
    e_unk = try; bf(@formula(y ~ x), @formula(theta ~ x)); catch e; e; end
    @test e_unk isa ArgumentError && occursin("sigma", e_unk.msg)
end
