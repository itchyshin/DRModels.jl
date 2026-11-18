# fitted() / residuals() / predict() mean the same thing as drmTMB 0.8.1 on the
# same fitted model (gap map v2, 2026-11-09). R's `fitted()` is the response mean
# E[Y]; `predict(type = "response")` is the response scale of the location
# parameter (the count component mean, and the log-scale mu for lognormal);
# `predict(type = "mean")` is E[Y]. Reference values and the data are committed
# under test/fixtures/fitted_semantics/ (make_reference.R, run on drmTMB 0.8.1);
# nothing here draws random numbers.
using DRModels
using Test
using DelimitedFiles

const _FS_DIR = joinpath(@__DIR__, "fixtures", "fitted_semantics")

function _fs_data()
    m, h = readdlm(joinpath(_FS_DIR, "data.csv"), ','; header = true)
    return NamedTuple(Symbol(h[j]) => Float64.(m[:, j]) for j in eachindex(vec(h)))
end

function _fs_reference()
    m = readdlm(joinpath(_FS_DIR, "reference.csv"), ','; skipstart = 1)
    ref = Dict{Tuple{String,String},Vector{Float64}}()
    for i in 1:size(m, 1)
        push!(get!(ref, (String(m[i, 1]), String(m[i, 2])), Float64[]), Float64(m[i, 4]))
    end
    return ref
end

const _FS_NEW = (; x = [-0.9, -0.2, 0.3, 0.8], z = [0.5, -0.7, 0.1, 0.9],
                 ln = ones(4), zip = zeros(4), zinb = zeros(4), tnb = ones(4), hnb = zeros(4))

@testset "fitted / predict semantics against drmTMB 0.8.1" begin
    d = _fs_data()
    ref = _fs_reference()
    fits = Dict(
        "lognormal" => drm(bf(@formula(ln ~ x), @formula(sigma ~ z)), LogNormal(); data = d),
        "zi_poisson" => drm(bf(@formula(zip ~ x), @formula(zi ~ z)), Poisson(); data = d),
        "zi_nbinom2" => drm(bf(@formula(zinb ~ x), @formula(sigma ~ 1), @formula(zi ~ z)),
                            NegBinomial2(); data = d),
        "truncated_nbinom2" => drm(bf(@formula(tnb ~ x), @formula(sigma ~ 1)),
                                   TruncatedNegBinomial2(); data = d),
        "hurdle_nbinom2" => drm(bf(@formula(hnb ~ x), @formula(sigma ~ 1), @formula(hu ~ z)),
                                TruncatedNegBinomial2(); data = d),
    )
    for name in ("lognormal", "zi_poisson", "zi_nbinom2", "truncated_nbinom2", "hurdle_nbinom2")
        fit = fits[name]
        @testset "$name" begin
            @test loglikelihood(fit) ≈ only(ref[(name, "logLik")]) atol = 1e-3
            @test fitted(fit) ≈ ref[(name, "fitted")] rtol = 1e-4
            @test residuals(fit) ≈ ref[(name, "residuals_response")] atol = 1e-3
            @test predict(fit, _FS_NEW; type = :response) ≈ ref[(name, "predict_response_new")] rtol = 1e-4
            @test predict(fit, _FS_NEW; type = :mean) ≈ ref[(name, "predict_mean_new")] rtol = 1e-4
            @test predict(fit, _FS_NEW; type = :link) ≈ ref[(name, "predict_link_new")] rtol = 1e-4
        end
    end
    @testset "the component mean stays reachable" begin
        # fit.means[:mu] is what simulate / marginal_parameters read; `:median`
        # keeps the exp(mu) value the lognormal `:response` used to return.
        ln = fits["lognormal"]
        @test predict(ln, _FS_NEW; type = :median) ≈ exp.(predict(ln, _FS_NEW; type = :link))
        @test_throws ArgumentError predict(fits["zi_poisson"], _FS_NEW; type = :median)
        zip = fits["zi_poisson"]
        @test all(fitted(zip) .< zip.means[:mu])
        tn = fits["truncated_nbinom2"]
        s2 = tn.scales[:sigma] .^ 2
        p0 = exp.(-log1p.(tn.means[:mu] .* s2) ./ s2)      # NB2 P(0)
        @test fitted(tn) ≈ tn.means[:mu] ./ (1 .- p0)
    end
end
