# drmTMB sends `coef_labels["recov"] = "<group>:atanh_rho"` for a Binomial
# animal intercept-plus-slope, including the unlabelled `animal(1 + x | id)`
# form (`drm_julia_ordinary_binomial_recov_label`). That form pins the
# correlation at 0, so the fitted blocks are `mu` and two `sdtarget_*` SD
# regressions and there is no `:recov` coefficient (#1401). The bridge must
# accept that one pinned name and still reject every other unknown dpar.
using DRModels
using Test, LinearAlgebra, Random

function _baq2_recov_data(; G = 8, m = 6, seed = 20261116)
    rng = MersenneTwister(seed)
    ids = ["id_$(i)" for i in 1:G]
    A = [0.4^abs(i - j) for i in 1:G, j in 1:G]
    gidx = repeat(1:G, inner = m)
    id = ids[gidx]
    z0 = ifelse.(gidx .<= G ÷ 2, -1.0, 1.0)
    z1 = collect(range(-1.0, 1.0; length = G))[gidx]
    x = repeat(collect(range(-1.0, 1.0; length = m)), outer = G)
    eta = -0.2 .+ 0.3 .* x .+ 0.15 .* randn(rng, length(x))
    y = Float64.(rand.(Ref(rng), DRModels.Distributions.Bernoulli.(1 ./ (1 .+ exp.(-eta)))))
    return (; data = (; y, x, z0, z1, id), A)
end

function _baq2_recov_labels(fit, recov)
    labels = Dict{String,Any}(String(param) => names for (param, names) in fit.coefnames)
    labels["recov"] = recov
    return labels
end

@testset "Binomial animal q2 bridge accepts drmTMB's pinned recov label (#1401)" begin
    fx = _baq2_recov_data()
    formula = bf(
        @formula(y ~ x + animal(1 + x | id)),
        @formula(sd(id, animal, coef(intercept)) ~ z0),
        @formula(sd(id, animal, coef(x)) ~ z1),
    )
    fit = drm(formula, Binomial(); data = fx.data, A = fx.A, se = false)
    @test fit.family isa Binomial
    @test !any(pair -> first(pair) === :recov, fit.blocks)
    @test count(pair -> startswith(String(first(pair)), "sdtarget_"), fit.blocks) == 2
    @test DRModels._bridge_pinned_binomial_animal_q2_group(fit) === :id

    pinned = "id:atanh_rho"
    good = _baq2_recov_labels(fit, [pinned])
    names, values, raw, _ = DRModels._bridge_coef_vector(fit; coef_labels = good)
    @test values == coef(fit)
    @test length(names) == length(raw) == length(coef(fit))
    @test !any(name -> startswith(name, "recov_"), names)
    for (param, coef_names) in fit.coefnames
        for nm in coef_names
            @test "$(param)_$(nm)" in names
        end
    end
    # JuliaCall can unbox a length-1 R character vector to a bare String.
    scalar_names, scalar_values, _, _ = DRModels._bridge_coef_vector(fit;
        coef_labels = _baq2_recov_labels(fit, pinned))
    @test scalar_names == names
    @test scalar_values == values

    wrong = _baq2_recov_labels(fit, ["id:atanh_partial_rho_21"])
    wrong_err = try
        DRModels._bridge_coef_vector(fit; coef_labels = wrong)
        nothing
    catch err
        err
    end
    @test wrong_err isa ErrorException
    @test occursin("unknown dpar \"recov\"", sprint(showerror, wrong_err))
    @test occursin("sdtarget_", sprint(showerror, wrong_err))

    other = Dict{String,Any}(String(param) => names for (param, names) in fit.coefnames)
    other["notadpar"] = ["whatever"]
    other_err = try
        DRModels._bridge_coef_vector(fit; coef_labels = other)
        nothing
    catch err
        err
    end
    @test other_err isa ErrorException
    @test occursin("unknown dpar \"notadpar\"", sprint(showerror, other_err))
    @test !occursin("unknown dpar \"recov\"", sprint(showerror, other_err))

    bridge_formula = "y ~ x + animal(1 + x | id); " *
        "sd(id, animal, coef(intercept)) ~ z0; sd(id, animal, coef(x)) ~ z1"
    out = drm_bridge(; formula = bridge_formula, family = "binomial", data = fx.data,
                     A = fx.A, options = Dict{String,Any}(
                         "se" => false,
                         "coef_labels" => good,
                     ))
    @test out["coef_names"] == out["vcov_names"]
    @test length(out["coefficients"]) == length(out["coef_names"]) == length(coef(fit))
    @test !any(name -> startswith(name, "recov_"), out["coef_names"])
    @test out["coefficients"] ≈ coef(fit) atol = 1e-8
    @test out["loglik"] ≈ loglikelihood(fit) atol = 1e-8
end
