# `vc` and `re_sd` on the correlated block `(1 + x | g)` of the `marginal = :Laplace` routes.
#
# These routes store the block as (log SD, log SD, atanh r), not as a log-Cholesky factor
# (`g:log_sd_intercept`, `g:log_sd_x`, `g:atanh_rho`). `vc` and `re_sd` used to read the Binomial
# triple as a Cholesky factor (a wrong slope SD and correlation) and refused the Poisson and
# NegBinomial2 pair with a message about `sd(group) ~ ...`. A Binomial block of three coefficients
# (stored as log SDs and partial correlations) was read as a Cholesky triple too; it is now refused.
#
# Reference: drmTMB 0.8.1 (ML), made by test/fixtures/vc_laplace_corr_slope/gen_ref.R on the datasets
# made by gen_data.jl. The tolerances were fixed before the fits were read: SDs within 1e-3 relative,
# the correlation within 2e-3, the log-likelihood within 1e-3.
using DRModels, Test

const _VCL_DIR = joinpath(@__DIR__, "fixtures", "vc_laplace_corr_slope")

function _vcl_data(name)
    lines = readlines(joinpath(_VCL_DIR, "data_$name.csv"))
    hdr = split(lines[1], ',')
    rows = [split(l, ',') for l in lines[2:end]]
    cols = Pair{Symbol,Any}[]
    for (j, h) in enumerate(hdr)
        v = [r[j] for r in rows]
        push!(cols, Symbol(h) => (h == "g" ? String.(v) : parse.(Float64, v)))
    end
    return NamedTuple(cols)
end

function _vcl_reference()
    lines = readlines(joinpath(_VCL_DIR, "r_reference.csv"))
    hdr = String.(strip.(split(lines[1], ','), '"'))
    ref = Dict{String,Any}()
    for l in lines[2:end]
        f = split(l, ',')
        val(h) = (s = f[findfirst(==(h), hdr)]; s == "NA" ? NaN : parse(Float64, s))
        ref[String(strip(f[1], '"'))] = (; sd = [val("sd1"), val("sd2"), val("sd3")],
            rho = [val("rho12"), val("rho13"), val("rho23")], ll = val("logLik"))
    end
    return ref
end

const _VCL_REF = _vcl_reference()

const _VCL_FAMILIES = (
    binomial = (Binomial(), bf(@formula(cbind(s, fl) ~ x + (1 + x | g)))),
    poisson = (Poisson(), bf(@formula(y ~ x + (1 + x | g)))),
    nb2 = (NegBinomial2(), bf(@formula(y ~ x + (1 + x | g)))),
)

# The three stored parameters of the `(1 + x | g)` block, in (log SD, log SD, atanh r) order.
_vcl_theta(fit::DrmFit{Binomial}) = coef(fit, :recov)
_vcl_theta(fit) = [only(coef(fit, :sd_intercept)), only(coef(fit, :sd_slope)), only(coef(fit, :recov))]

@testset "vc and re_sd on the marginal = :Laplace correlated block" begin
    @testset "$fam, data set $seed" for fam in keys(_VCL_FAMILIES), seed in 1:4
        family, form = _VCL_FAMILIES[fam]
        ref = _VCL_REF["$(fam)_s$(seed)"]
        fit = drm(form, family; data = _vcl_data("$(fam)_s$(seed)"), marginal = :Laplace)
        @test fit.converged
        @test fit.marginal === :Laplace
        @test abs(loglikelihood(fit) - ref.ll) <= 1e-3

        sd = re_sd(fit)
        V = vc(fit)[:g]
        r = V[1, 2] / sqrt(V[1, 1] * V[2, 2])
        # against drmTMB
        @test sd[:g_intercept] ≈ ref.sd[1] rtol = 1e-3
        @test sd[:g_slope] ≈ ref.sd[2] rtol = 1e-3
        @test sqrt(V[1, 1]) ≈ ref.sd[1] rtol = 1e-3
        @test sqrt(V[2, 2]) ≈ ref.sd[2] rtol = 1e-3
        @test abs(r - ref.rho[1]) <= 2e-3
        # against a direct transform of the stored coefficients
        th = _vcl_theta(fit)
        @test sd[:g_intercept] ≈ exp(th[1]) rtol = 1e-12
        @test sd[:g_slope] ≈ exp(th[2]) rtol = 1e-12
        bound = fam === :binomial ? 1.0 : 0.999999       # the count routes bound r as 0.999999 tanh(eta)
        @test r ≈ bound * tanh(th[3]) rtol = 1e-9
        @test V[1, 1] ≈ exp(2th[1]) rtol = 1e-12
        @test V[2, 2] ≈ exp(2th[2]) rtol = 1e-12
        # the printed block says what it holds
        txt = sprint(show, MIME("text/plain"), fit)
        @test !occursin("Cholesky", txt)
        @test occursin(fam === :binomial ? "Random-effect SDs (log σ) and correlations (atanh)" :
                                           "Random-effect correlation (atanh)", txt)
    end

    @testset "Binomial (1 + x + x2 | g): refused, parameters agree with drmTMB, data set $seed" for seed in 1:2
        d = _vcl_data("binomial_q3_s$(seed)")
        ref = _VCL_REF["binomial_q3_s$(seed)"]
        f = bf(@formula(cbind(s, fl) ~ x + x2 + (1 + x + x2 | g)))
        for marginal in (:Laplace, nothing)
            fit = marginal === nothing ? drm(f, Binomial(); data = d) : drm(f, Binomial(); data = d, marginal)
            @test_throws ArgumentError re_sd(fit)
            @test_throws ArgumentError vc(fit)
            th = coef(fit, :recov)
            # the default route integrates by quadrature, so only the Laplace route matches drmTMB's Laplace fit tightly
            tol = marginal === :Laplace ? 1e-3 : 1e-2
            @test exp.(th[1:3]) ≈ ref.sd[1:3] rtol = tol
            @test maximum(abs, DRModels._binomial_pairwise_correlations(th[4:6], 3) .- ref.rho) <= 2tol
        end
    end

    @testset "default route: the log-Cholesky block is read as before" begin
        d = _vcl_data("binomial_s1")
        fit = drm(bf(@formula(cbind(s, fl) ~ x + (1 + x | g))), Binomial(); data = d, marginal = :LA)
        th = coef(fit, :recov)
        l11, l22, l21 = exp(th[1]), exp(th[2]), th[3]
        @test vc(fit)[:g] == [l11^2 l21*l11; l21*l11 l21^2+l22^2]
        @test re_sd(fit)[:g_intercept] == l11
        @test re_sd(fit)[:g_slope] == sqrt(l21^2 + l22^2)
        @test occursin("Random-effect covariance (Cholesky)", sprint(show, MIME("text/plain"), fit))
    end
end
