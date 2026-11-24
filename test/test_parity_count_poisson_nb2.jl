# test_parity_count_poisson_nb2.jl -- Poisson and NB2 parity vs drmTMB: logLik, coef, SE.
# Refs #1073, #1072.
#
# References are generated drmTMB numbers only (no drmTMB source; see test/parity/GENERATING.md):
#   * fixed-effects fits  -- test/parity/fixtures/{count-poisson,count-nbinom2}/expected.toml
#                            (drmTMB 0.7.0, committed with their data.csv)
#   * ordinary (1 | g)    -- test/fixtures/count_parity/{poisson_g,nb2_g}/ref.toml, written by
#                            test/parity/gen_count_ranef_parity.R (drmTMB 0.7.1; TMB max|grad|
#                            at opt$par < 1e-6, all SEs finite)
#
# Each quantity is its own check, so a failure names the quantity. A quantity that differs
# from drmTMB beyond tolerance is a `@test_broken` (it flips to an "Unexpected Pass" error
# when fixed), never a loosened tolerance.
#
# Known, documented mismatch (docs/src/drmtmb-parity.md, "Default marginal approximation"):
# the DEFAULT `marginal = :LA` for an ordinary `(1 | g)` on a Poisson/NB2 mean is 5-node
# per-group adaptive quadrature, whereas drmTMB uses the Laplace approximation (one node).
# The two fit the same model with different integrators, so logLik and the working-scale
# coefficients differ at the 1e-2 / 1e-4..1e-3 level. `marginal = :Laplace` is the
# documented way to match drmTMB and must agree to ~1e-9. No default is changed here.
using DRModels, Test, TOML, DelimitedFiles

const _PC_FIXED = joinpath(@__DIR__, "parity", "fixtures")
const _PC_GROUP = joinpath(@__DIR__, "fixtures", "count_parity")

# tolerances
const _PC_ATOL_LL = 1e-3          # logLik (same as the parity fixtures' [tol])
const _PC_ATOL_COEF = 1e-5        # coef: |d| <= atol + rtol*|ref|
const _PC_RTOL_COEF = 1e-5
const _PC_RTOL_SE = 1e-3          # SE, same integrator as drmTMB
const _PC_RTOL_SE_DEFAULT = 1e-2  # SE, default (1 | g) integrator: the #1073 ~1% SE bar

# flat `<block>_<coef>` -> (estimate, SE), walking fit.blocks / fit.coefnames in theta order
function _pc_named(fit)
    θ = coef(fit); V = vcov(fit)
    names = Dict(p => ns for (p, ns) in fit.coefnames)
    est = Dict{String,Float64}(); se = Dict{String,Float64}(); i = 0
    for (param, r) in fit.blocks
        haskey(names, param) || continue
        for (nm, v) in zip(names[param], θ[r])
            i += 1
            est["$(param)_$(nm)"] = v
            se["$(param)_$(nm)"] = V[i, i] > 0 ? sqrt(V[i, i]) : NaN
        end
    end
    return est, se
end

# One @test (or @test_broken, for a label in `broken`) per quantity.
function _pc_check(case::AbstractString, fit, ref_ll, ref_coef, ref_se; se_rtol, broken = String[])
    est, se = _pc_named(fit)
    function chk(label, ok)
        full = "$case: $label"
        if label in broken
            @test_broken ok          # documented mismatch; see header
        else
            @test ok
        end
        return nothing
    end
    @testset "$case" begin
        @test is_converged(fit)
        chk("logLik", abs(loglik(fit) - ref_ll) <= _PC_ATOL_LL)
        for (k, r) in ref_coef
            chk("coef $k", haskey(est, k) && abs(est[k] - r) <= _PC_ATOL_COEF + _PC_RTOL_COEF * abs(r))
        end
        for (k, r) in ref_se
            chk("se $k", haskey(se, k) && isfinite(se[k]) && abs(se[k] / r - 1) <= se_rtol)
        end
    end
end

# fixed-effects fixture: committed expected.toml + data.csv
function _pc_fixed(case, fam, forms)
    dir = joinpath(_PC_FIXED, case)
    t = TOML.parsefile(joinpath(dir, "expected.toml"))
    m, h = readdlm(joinpath(dir, "data.csv"), ','; header = true)
    d = NamedTuple{Tuple(Symbol.(strip.(string.(vec(h)), '"')))}(Tuple(Float64.(m[:, j]) for j in 1:size(m, 2)))
    fit = drm(bf(forms...), fam; data = d)
    _pc_check(case, fit, t["fit"]["loglik"], Dict(t["coef"]), Dict(t["se"]); se_rtol = _PC_RTOL_SE)
end

# (1 | g) reference: ref.toml (names / theta / se / loglik) + data.csv (y, x, g)
function _pc_group_data(route)
    L = readlines(joinpath(_PC_GROUP, route, "data.csv"))
    hdr = [strip(c, '"') for c in split(L[1], ',')]
    col(name) = [strip(split(l, ',')[findfirst(==(name), hdr)], '"') for l in L[2:end]]
    return (; y = parse.(Float64, col("y")), x = parse.(Float64, col("x")), g = String.(col("g")))
end
function _pc_group(route, fam, forms; kw...)
    ref = TOML.parsefile(joinpath(_PC_GROUP, route, "ref.toml"))
    @test ref["max_grad"] < 1e-6                       # reference usable
    fit = drm(bf(forms...), fam; data = _pc_group_data(route), kw...)
    return fit, ref
end

@testset "Poisson / NB2 parity vs drmTMB: logLik, coef, SE (Refs #1073, #1072)" begin
    @testset "fixed effects" begin
        _pc_fixed("count-poisson", Poisson(), (@formula(y ~ x),))
        _pc_fixed("count-nbinom2", NegBinomial2(), (@formula(y ~ x), @formula(sigma ~ 1)))
    end

    @testset "(1 | g), marginal = :Laplace (the documented drmTMB-matching route)" begin
        for (route, fam, forms) in (
                ("poisson_g", Poisson(), (@formula(y ~ x + (1 | g)),)),
                ("nb2_g", NegBinomial2(), (@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))))
            fit, ref = _pc_group(route, fam, forms; marginal = :Laplace)
            _pc_check("$route :Laplace", fit, ref["loglik"], Dict(zip(ref["names"], ref["theta"])),
                      Dict(zip(ref["names"], ref["se"])); se_rtol = _PC_RTOL_SE)
        end
    end

    @testset "(1 | g), default marginal (5-node AGHQ vs drmTMB Laplace)" begin
        for (route, fam, forms) in (
                ("poisson_g", Poisson(), (@formula(y ~ x + (1 | g)),)),
                ("nb2_g", NegBinomial2(), (@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))))
            fit, ref = _pc_group(route, fam, forms)
            coefs = Dict(zip(ref["names"], ref["theta"]))
            # logLik (~0.02 / 0.08 nat) and every working-scale coefficient differ by more than
            # tolerance: different integrators, documented in drmtmb-parity.md. SEs agree within
            # the 1% bar.
            broken = vcat(["logLik"], ["coef $k" for k in ref["names"]])
            _pc_check("$route default", fit, ref["loglik"], coefs, Dict(zip(ref["names"], ref["se"]));
                      se_rtol = _PC_RTOL_SE_DEFAULT, broken = broken)
        end
    end
end
