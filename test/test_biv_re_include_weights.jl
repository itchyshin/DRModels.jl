# Bivariate Gaussian `(1 | p | g)` with `method = :REML`, `miss_control(response = "include")` (rows with one
# or both responses missing are kept) and prior `weights`.
# drmTMB 0.8.1 refuses REML with kept missing rows, so there is no drmTMB reference. The references are
# independent of drmTMB and of this package, made by fixtures/biv_re_include/reml_weights.R (R 4.6.1) on
# reml_weights_data.csv (reml_data1.csv plus two weight columns):
#   w_int  integer weights 0 to 3 (25 of 200 rows have weight 0): nlme::lme REML on the long-format data with
#          every row repeated w_int times (a weight multiplies the row's whole log-density, the same as repeating
#          the row, and weight 0 removes it), and a dense restricted likelihood written from the formula;
#   w_real non-integer weights (8 rows with weight 0): the dense restricted likelihood only.
# Bounds are set from the measured gaps (Julia 1.13.3) with a margin of about 3 to 4. The nlme optimiser
# stops slightly short of the optimum; the dense maximum is the tighter reference.
#                     measured gap (nlme | dense w_int | dense w_real)      bound (nlme | dense)
#   log-likelihood    9.1e-9 | 4.5e-11 | 2.5e-11                            3e-8 | 2e-10
#   fixed effects     3.3e-7 | 9.0e-10 | 1.5e-8                             1e-6 | 5e-8
#   Sigma entries     1.6e-5 | 1.1e-7  | 2.6e-7                             5e-5 | 1e-6
#   sigma1, sigma2    7.2e-7 | 5.7e-8  | 7.3e-8                             3e-6 | 3e-7
#   rho               3.3e-6 | 3.5e-8  | 5.2e-8                             1e-5 | 2e-7
using DRModels
using Test
using DelimitedFiles

function _bw_data()
    m, h = readdlm(joinpath(@__DIR__, "fixtures", "biv_re_include", "reml_weights_data.csv"), ',', Any; header = true)
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(String.(vec(h)))
        c = m[:, j]
        v = n == "g" ? string.(c) :
            any(x -> x == "NA", c) ? Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c] : Float64.(c)
        push!(cols, (n == "y1" ? :y1m : n == "y2" ? :y2m : Symbol(n)) => v)
    end
    return (; cols...)
end

function _bw_reference()
    m = readdlm(joinpath(@__DIR__, "fixtures", "biv_re_include", "reml_weights_reference.csv"), ',', Any; skipstart = 1)
    ref = Dict{Int,Dict{String,Float64}}()
    for r in 1:size(m, 1)
        get!(ref, Int(m[r, 1]), Dict{String,Float64}())[String(m[r, 2])] = Float64(m[r, 3])
    end
    return ref
end

function _bw_gaps(fit, r)
    b = vcat(coef(fit, :mu1), coef(fit, :mu2))
    bn = [r["r1"], r["x1"], r["r2"], r["x2"]]
    Lam = Matrix(fit.ranef.Sigma_g)
    return (ll = abs(loglikelihood(fit) - r["ll"]), beta = maximum(abs.(b .- bn)),
            Sigma = max(abs(Lam[1, 1] - r["Sigma11"]), abs(Lam[1, 2] - r["Sigma12"]), abs(Lam[2, 2] - r["Sigma22"])),
            sigma = max(abs(exp(only(coef(fit, :sigma1))) - r["sigma1"]), abs(exp(only(coef(fit, :sigma2))) - r["sigma2"])),
            rho = abs(DRModels.RHO_GUARD * tanh(only(coef(fit, :rho12))) - r["rho"]))
end

@testset "bivariate (1 | p | g): REML with kept missing rows and weights equals independent references" begin
    ref = _bw_reference(); d = _bw_data()
    inc = miss_control(response = "include")
    f = bf(mu1 = @formula(y1m ~ x1 + (1 | p | g)), mu2 = @formula(y2m ~ x2 + (1 | p | g)),
           sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
    @test any(ismissing, d.y1m) && any(ismissing, d.y2m)
    @test count(==(0.0), d.w_int) == 25 && count(==(0.0), d.w_real) >= 8
    nobs_w(w) = sum(w[i] * (!ismissing(d.y1m[i]) + !ismissing(d.y2m[i])) for i in eachindex(w))
    @testset "integer weights with zeros: nlme on repeated rows" begin
        fit = drm(f, Gaussian(); data = d, weights = d.w_int, missing = inc, method = :REML)
        @test fit.converged
        @test nobs_w(d.w_int) == ref[1]["N"]
        g = _bw_gaps(fit, ref[1])
        @test g.ll < 3e-8 && g.beta < 1e-6 && g.Sigma < 5e-5 && g.sigma < 3e-6 && g.rho < 1e-5
    end
    @testset "$(k == 2 ? "integer" : "non-integer") weights with zeros: dense restricted likelihood" for (k, wc) in ((2, :w_int), (3, :w_real))
        fit = drm(f, Gaussian(); data = d, weights = getproperty(d, wc), missing = inc, method = :REML)
        @test fit.converged
        @test nobs_w(getproperty(d, wc)) ≈ ref[k]["N"] atol = 1e-9
        g = _bw_gaps(fit, ref[k])
        @test g.ll < 2e-10 && g.beta < 5e-8 && g.Sigma < 1e-6 && g.sigma < 3e-7 && g.rho < 2e-7
    end
end
