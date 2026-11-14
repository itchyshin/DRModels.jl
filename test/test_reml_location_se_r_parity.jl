# Gaussian REML location SEs against drmTMB 0.8.1 (values pinned in
# fixtures/reml_location_se/r_reference.csv, made by gen_ref.R next to it).
#
# drmTMB reports, under REML, the covariance of the location coefficients as
#     (X' W X)^-1 + J V_sigma J'      with J = d beta_hat / d beta_sigma,
# and the location-scale covariance block J V_sigma. The plug-in GLS covariance
# (X' W X)^-1 alone is 0.5 percent too small when sigma has a regression.
using DRModels, Test, LinearAlgebra
using Distributions: Normal, quantile

function _remlse_data(name)
    if name == "a"
        i = 1:24; x = sin.(i); u = mod.(0.6180339887 .* i, 1)
        return (; y = 0.5 .+ 0.8 .* x .+ exp.(1.2 .* x) .* 1.5 .* (2 .* u .- 1), x)
    elseif name == "b"
        i = 1:40; x = sin.(i); z = cos.(1.3 .* i); u = mod.(0.7548776662 .* i, 1)
        return (; y = 1 .+ 0.6 .* x .- 0.4 .* z .+ exp.(0.5 .* x .- 0.3 .* z) .* 0.8 .* (2 .* u .- 1), x, z)
    else
        i = 1:100; g = string.(mod.(i .- 1, 20) .+ 1)
        x = sin.(0.7 .* i); u = mod.(0.6180339887 .* i, 1); ge = 0.8 .* sin.((1:20) .* 2.3)
        gi = mod.(i .- 1, 20) .+ 1
        return (; y = 1 .+ 0.5 .* x .+ ge[gi] .+ exp.(0.5 .* x) .* 0.9 .* (2 .* u .- 1), x, g)
    end
end

function _remlse_reference()
    lines = readlines(joinpath(@__DIR__, "fixtures", "reml_location_se", "r_reference.csv"))[2:end]
    ref = Dict{String,Dict{String,Any}}()
    for ln in lines
        f = split(replace(ln, "\"" => ""), ",")
        d = get!(ref, f[1], Dict{String,Any}("vcov" => Dict{Tuple{String,String},Float64}(),
                                              "coef" => Dict{String,Float64}(), "names" => String[]))
        v = parse(Float64, f[5])
        if f[2] == "logLik"
            d["logLik"] = v
        elseif f[2] == "coef"
            d["coef"][f[3]] = v; push!(d["names"], f[3])
        else
            d["vcov"][(f[3], f[4])] = v
        end
    end
    return ref
end

const _REMLSE_FORMS = Dict(
    "a_s1" => ("a", bf(@formula(y ~ x), @formula(sigma ~ 1))),
    "a_sx" => ("a", bf(@formula(y ~ x), @formula(sigma ~ x))),
    "b_sxz" => ("b", bf(@formula(y ~ x + z), @formula(sigma ~ x + z))),
    "c_re_sx" => ("c", bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x))),
)

@testset "REML location SEs match drmTMB (sigma uncertainty term)" begin
    ref = _remlse_reference()
    for tag in ("a_s1", "a_sx", "b_sxz", "c_re_sx")
        dname, form = _REMLSE_FORMS[tag]
        r = ref[tag]
        fit = drm(form, Gaussian(); data = _remlse_data(dname), method = :REML)
        # The route's fixed-effect names, in R's order (mu then sigma).
        nms = r["names"]
        k = length(nms)
        V = vcov(fit)[1:k, 1:k]     # the (1 | g) route appends its group SD after the fixed effects
        @testset "$tag" begin
            @test isapprox(loglikelihood(fit), r["logLik"]; atol = 1e-5)
            @test isapprox(coef(fit)[1:k], [r["coef"][n] for n in nms]; rtol = 1e-5, atol = 1e-6)
            R = [r["vcov"][(a, b)] for a in nms, b in nms]
            sdR = sqrt.(diag(R))
            # every entry within 1e-4 of the geometric mean SE (the parity bound is 1e-3)
            @test maximum(abs.(V .- R) ./ (sdR * sdR')) < 1e-4
            # standard errors, relative 1e-4
            @test maximum(abs.(sqrt.(diag(V)) ./ sdR .- 1)) < 1e-4
            se = stderror(fit)[1:k]
            @test se ≈ sqrt.(diag(V)) rtol = 1e-12
            ct = coeftable(fit)
            @test ct.cols[2][1:k] ≈ sdR rtol = 1e-4
            ci = confint(fit; method = :wald)
            zc = quantile(Normal(), 0.975)
            @test [row.upper - row.estimate for row in ci][1:k] ≈ zc .* sdR rtol = 1e-4
        end
    end
end

@testset "REML location SEs: formula and unchanged parts" begin
    data = _remlse_data("a")
    f = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data, method = :REML)
    X = [ones(24) data.x]; w = exp.(-2 .* (X * coef(f, :sigma)))
    A = X' * (w .* X)
    r = data.y .- X * coef(f, :mu)
    # J by central differences of the GLS fit in beta_sigma
    gls(bs) = (ww = exp.(-2 .* (X * bs)); (X' * (ww .* X)) \ (X' * (ww .* data.y)))
    bs = coef(f, :sigma); h = 1e-5
    J = hcat([(gls(bs .+ h .* (1:2 .== j)) .- gls(bs .- h .* (1:2 .== j))) ./ (2h) for j in 1:2]...)
    Vs = vcov(f)[3:4, 3:4]
    @test vcov(f)[1:2, 1:2] ≈ inv(A) + J * Vs * J' rtol = 1e-6
    @test vcov(f)[1:2, 3:4] ≈ J * Vs rtol = 1e-6
    # the sigma block and the estimates are what they were
    @test sqrt.(diag(Vs)) ≈ [0.1508986, 0.2005000] rtol = 1e-5
    # constant sigma: J = 0, so the plug-in covariance is the whole answer
    f1 = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data, method = :REML)
    w1 = exp(-2 * coef(f1, :sigma)[1])
    @test vcov(f1)[1:2, 1:2] ≈ inv(w1 .* (X' * X)) rtol = 1e-10
    @test maximum(abs.(vcov(f1)[1:2, 3])) < 1e-12
    # ... and it is the plug-in covariance bit for bit: the mean block is the formula without the
    # extra term and the cross block is exactly zero, not rounding noise
    wv = exp.(-2 .* fill(coef(f1, :sigma)[1], 24))
    @test vcov(f1)[1:2, 1:2] == DRModels._vcov_from_gram(X' * (wv .* X); context = "Gaussian REML (mean WLS)")
    @test all(iszero, vcov(f1)[1:2, 3])
end
