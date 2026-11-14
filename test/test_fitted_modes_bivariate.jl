# fitted() keeps the conditional modes on the bivariate Gaussian route with an ordinary
# correlated random intercept `(1 | p | g)` in mu1 and mu2.
#
# Before, the fit kept only the fixed-effect means: logLik agreed with drmTMB 0.8.1 but `fitted(fit)`
# either returned the population values (0.75 away from drmTMB) or, once the conditional default of
# `fitted()` was in place, refused. Now `fitted(fit)` is drmTMB's `fitted(fit)`: a Dict with the
# two response means (`:mu1`, `:mu2`; drmTMB returns an n x 2 matrix) that include the group modes.
# For a Gaussian model the modes are the exact BLUPs, in closed form at the fitted parameters.
#
# References: test/fixtures/fitted_modes_bivariate/fitted_modes_reference.csv and ranef_reference.csv,
# written by fitted_modes_r.R in the same folder (drmTMB 0.8.1, R 4.6.1, Totoro) from data.csv.
# Bounds follow test_fitted_conditional.jl and are not loosened: 1e-6 on fitted values and modes,
# 1e-4 on logLik. The modes are also checked against the joint density maximised here by Newton
# steps, independently of the package.
using DRModels
using Test
using DelimitedFiles
using ForwardDiff
using LinearAlgebra

const _FB_DIR = joinpath(@__DIR__, "fixtures", "fitted_modes_bivariate")

function _fb_data()
    m, h = readdlm(joinpath(_FB_DIR, "data.csv"), ',', Any; header = true)
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(String.(vec(h)))
        c = m[:, j]
        v = if n == "g"
            String.(c)
        elseif any(x -> x == "NA", c)
            Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c]
        else
            Float64.(c)
        end
        push!(cols, Symbol(n) => v)
    end
    return (; cols...)
end

function _fb_reference()
    m = readdlm(joinpath(_FB_DIR, "fitted_modes_reference.csv"), ',', Any; skipstart = 1)
    ids = String.(m[:, 1])
    return Dict(id => (f1 = Float64.(m[ids .== id, 3]), f2 = Float64.(m[ids .== id, 4]),
                       ll = Float64(m[findfirst(==(id), ids), 5])) for id in unique(ids))
end

function _fb_ranef()
    m = readdlm(joinpath(_FB_DIR, "ranef_reference.csv"), ',', Any; skipstart = 1)
    ids = String.(m[:, 1])
    return Dict(id => (group = String.(m[ids .== id, 2]), u = permutedims(Float64.(m[ids .== id, 3:4])))
                for id in unique(ids))
end

_fb_form(; s1 = :(1), rho = :(1)) = bf(mu1 = @formula(y1 ~ x1 + (1 | p | g)), mu2 = @formula(y2 ~ x2 + (1 | p | g)),
                                        sigma1 = s1 === :x1 ? @formula(sigma1 ~ x1) : @formula(sigma1 ~ 1),
                                        sigma2 = @formula(sigma2 ~ 1),
                                        rho12 = rho === :x1 ? @formula(rho12 ~ x1) : @formula(rho12 ~ 1))

# (cell id, formula, fit keywords, sigma1 uses x1, rho12 uses x1)
const _FB_CELLS = [
    ("G|gauss|RE=(1|p|g)|rho=1", _fb_form(), (;), false, false),
    ("G|gauss|RE=(1|p|g)|rho=x1", _fb_form(rho = :x1), (;), false, true),
    ("G|gauss|RE=(1|p|g)|sigma1=x1", _fb_form(s1 = :x1), (;), true, false),
    ("G|gauss|RE=(1|p|g)|REML", _fb_form(), (; method = :REML), false, false),
]

# Modes found by maximising sum_i log N(y_i - X_i b - u; 0, S_i) - z'z / 2 per group with u = Lg z,
# Lg Lg' = Sigma_g (own formulas, written from the model; Newton steps with ForwardDiff derivatives).
# The z scale needs no inverse of Sigma_g, which is singular for a group SD at its boundary.
function _fb_modes(fit, d, s1x, rhox)
    x1, x2, y1, y2, grp = d.x1, d.x2, d.y1, d.y2, d.g
    n = length(y1)
    b1 = coef(fit, :mu1); b2 = coef(fit, :mu2)
    c1 = coef(fit, :sigma1); c2 = coef(fit, :sigma2); cr = coef(fit, :rho12)
    s1 = exp.(s1x ? c1[1] .+ c1[2] .* x1 : fill(c1[1], n))
    s2 = fill(exp(c2[1]), n)
    rho = DRModels.RHO_GUARD .* tanh.(rhox ? cr[1] .+ cr[2] .* x1 : fill(cr[1], n))
    r1 = y1 .- (b1[1] .+ b1[2] .* x1)
    r2 = y2 .- (b2[1] .+ b2[2] .* x2)
    F = eigen(Symmetric(Matrix(fit.ranef.Sigma_g)))
    Lg = F.vectors * Diagonal(sqrt.(max.(F.values, 0.0)))
    out = Dict{String,Vector{Float64}}()
    for g in unique(grp)
        rows = findall(==(g), grp)
        function nlj(z)
            u = Lg * z
            v = 0.0
            for i in rows
                S = [s1[i]^2 rho[i]*s1[i]*s2[i]; rho[i]*s1[i]*s2[i] s2[i]^2]
                e = [r1[i] - u[1], r2[i] - u[2]]
                v += 0.5 * logdet(S) + 0.5 * dot(e, S \ e)
            end
            return v + 0.5 * dot(z, z)
        end
        z = zeros(2)
        for _ in 1:100
            st = ForwardDiff.hessian(nlj, z) \ ForwardDiff.gradient(nlj, z)
            z = z .- st
            norm(st) < 1e-13 && break
        end
        out[g] = Lg * z
    end
    return out
end

@testset "fitted(): modes on the bivariate Gaussian (1 | p | g) route" begin
    d = _fb_data(); ref = _fb_reference(); rref = _fb_ranef()
    for (id, f, kw, s1x, rhox) in _FB_CELLS
        @testset "$id" begin
            fit = drm(f, Gaussian(); data = d, kw...)
            r = ref[id]
            @test abs(loglikelihood(fit) - r.ll) < 1e-4
            cond = fitted(fit)
            @test cond isa AbstractDict && Set(keys(cond)) == Set([:mu1, :mu2])
            @test length(cond[:mu1]) == length(cond[:mu2]) == nobs(fit) == length(r.f1)
            @test maximum(abs.(cond[:mu1] .- r.f1)) < 1e-6          # drmTMB fitted(fit), column 1
            @test maximum(abs.(cond[:mu2] .- r.f2)) < 1e-6          # column 2
            @test fitted(fit; re = :conditional) == cond
            zero = fitted(fit; re = :zero)
            @test zero[:mu1] == fit.means[:mu1] && zero[:mu2] == fit.means[:mu2]
            @test maximum(abs.(zero[:mu1] .- cond[:mu1])) > 0.1      # the modes matter
            # the modes against drmTMB's ranef(fit)
            rr = rref[id]
            gl = unique(d.g)
            U = ranef(fit)[:g]
            @test size(U) == (2, length(gl))
            perm = [findfirst(==(g), gl) for g in rr.group]
            @test maximum(abs.(U[:, perm] .- rr.u)) < 1e-6
            # the modes against the joint density maximised independently
            um = _fb_modes(fit, d, s1x, rhox)
            @test maximum(maximum(abs.(U[:, k] .- um[g])) for (k, g) in enumerate(gl)) < 1e-6
            gi = [findfirst(==(g), gl) for g in d.g]
            @test maximum(abs.(cond[:mu1] .- (fit.means[:mu1] .+ U[1, gi]))) < 1e-12
        end
    end
    @testset "interior group covariance: modes against the joint density" begin
        m, h = readdlm(joinpath(@__DIR__, "fixtures", "biv_ordinary_re", "interior_bal.csv"), ',', Any; header = true)
        di = (; [Symbol(n) => (n == "g" ? String.(m[:, j]) : Float64.(m[:, j])) for (j, n) in enumerate(String.(vec(h)))]...)
        fit = drm(_fb_form(), Gaussian(); data = di)
        @test abs(fit.ranef.Sigma_g[1, 1]) > 0.05 && abs(fit.ranef.Sigma_g[2, 2]) > 0.05    # not at the boundary
        U = ranef(fit)[:g]; gl = unique(di.g)
        um = _fb_modes(fit, di, false, false)
        @test maximum(maximum(abs.(U[:, k] .- um[g])) for (k, g) in enumerate(gl)) < 1e-6
        cond = fitted(fit)
        gi = [findfirst(==(g), gl) for g in di.g]
        @test maximum(abs.(cond[:mu2] .- (fit.means[:mu2] .+ U[2, gi]))) < 1e-12
        @test maximum(abs.(cond[:mu2] .- fit.means[:mu2])) > 0.05
    end
    @testset "estimates and the fit record are not touched" begin
        fit = drm(_fb_form(), Gaussian(); data = d)
        @test !haskey(fit.scales, :cond_fit_eta_re)
        @test fit.means[:mu1] ≈ coef(fit, :mu1)[1] .+ coef(fit, :mu1)[2] .* d.x1
    end
end
