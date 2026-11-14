# fitted() and ranef() on the bivariate Gaussian (1 | p | g) route with prior weights.
#
# A weight multiplies a row's whole log density (drmTMB, and the replicated-row definition of
# test_biv_re_weights.jl), so the group modes are the BLUPs under the row precision w_i * S_i^-1, where
# S_i is the 2 x 2 residual covariance of row i (the observed cells only, for a row with one response
# missing under the include policy). A zero weight removes the row from its group's mode.
# Checks, per cell: (1) the log-likelihood and the row count against drmTMB 0.8.1; (2) fitted(fit) and the
# group modes against drmTMB's fitted() and ranef(); (3) the modes against an independent Newton
# maximisation of the weighted joint density (own code, no use of the fit's formula for the modes);
# (4) the log-likelihood evaluated by hand with a dense Gaussian covariance S_i / w_i plus the
# normalising term that turns it into the weighted density.
# Reference: test/fixtures/fitted_modes_bivariate_weights/{fitted_reference,ranef_reference}.csv, written
# by make_reference.R in the same folder (drmTMB 0.8.1) from test/fixtures/biv_re_weights data.
using DRModels
using Test, DelimitedFiles, LinearAlgebra, Logging, ForwardDiff

const _FMW_FIX = joinpath(@__DIR__, "fixtures", "fitted_modes_bivariate_weights")

function _fmw_data(path)
    m, h = readdlm(path, ',', Any; header = true)
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(String.(vec(h)))
        c = m[:, j]
        push!(cols, Symbol(n) => n == "g" ? String.(c) :
            any(x -> x == "NA", c) ? Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c] :
            Float64.(c))
    end
    return (; cols...)
end

const _FMW_D = _fmw_data(joinpath(@__DIR__, "fixtures", "biv_re_weights", "data_weights.csv"))
const _FMW_U = _fmw_data(joinpath(@__DIR__, "fixtures", "biv_re_weights", "unbal_weights.csv"))
const _FMW_F = bf(mu1 = @formula(y1 ~ x1 + (1 | p | g)), mu2 = @formula(y2 ~ x2 + (1 | p | g)),
                  sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
const _FMW_FM = bf(mu1 = @formula(y1m ~ x1 + (1 | p | g)), mu2 = @formula(y2m ~ x2 + (1 | p | g)),
                   sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))

function _fmw_fitted_ref()
    m = readdlm(joinpath(_FMW_FIX, "fitted_reference.csv"), ',', Any; skipstart = 1)
    out = Dict{String,NamedTuple}()
    for cell in unique(String.(m[:, 1]))
        r = m[String.(m[:, 1]) .== cell, :]
        out[cell] = (; fit = hcat(Float64.(r[:, 3]), Float64.(r[:, 4])), ll = Float64(r[1, 5]), nobs = Int(r[1, 6]))
    end
    return out
end
function _fmw_ranef_ref()
    m = readdlm(joinpath(_FMW_FIX, "ranef_reference.csv"), ',', Any; skipstart = 1)
    return Dict(cell => (; groups = String.(m[String.(m[:, 1]) .== cell, 2]),
                         U = permutedims(hcat(Float64.(m[String.(m[:, 1]) .== cell, 3]),
                                              Float64.(m[String.(m[:, 1]) .== cell, 4]))))
                for cell in unique(String.(m[:, 1])))
end
const _FMW_REF = _fmw_fitted_ref()
const _FMW_RREF = _fmw_ranef_ref()

# One row's log density kernel (observed cells only) at residual r = y - mean - u.
function _fmw_rowlogdens(r, o1, o2, s1, s2, ρ)
    if o1 && o2
        om = 1 - ρ^2
        q = (r[1]^2 / s1^2 - 2ρ * r[1] * r[2] / (s1 * s2) + r[2]^2 / s2^2) / om
        return -0.5 * (q + 2log(s1) + 2log(s2) + log(om) + 2log(2π))
    elseif o1
        return -0.5 * (r[1]^2 / s1^2 + 2log(s1) + log(2π))
    elseif o2
        return -0.5 * (r[2]^2 / s2^2 + 2log(s2) + log(2π))
    end
    return zero(r[1])
end

# Newton maximisation of sum_i w_i log f_i(y_i - mean_i - u) - u' Sigma_g^-1 u / 2 over u, one group at a
# time, in z coordinates (u = L z, prior -z'z/2; Sigma_g can sit near a boundary). Uses only the fitted
# parameters, the weights and the data.
function _fmw_modes(fit, y1, y2, w)
    n = length(w)
    mu1, mu2 = fit.means[:mu1], fit.means[:mu2]
    s1, s2, ρ = fit.scales[:sigma1], fit.scales[:sigma2], fit.scales[:rho12]
    o1 = .!ismissing.(y1); o2 = .!ismissing.(y2)
    Y1 = [o1[i] ? Float64(y1[i]) : 0.0 for i in 1:n]; Y2 = [o2[i] ? Float64(y2[i]) : 0.0 for i in 1:n]
    F = eigen(Symmetric(Matrix{Float64}(fit.ranef.Sigma_g)))
    L = F.vectors * Diagonal(sqrt.(max.(F.values, 0.0)))
    gi = fit.ranef.group_index
    G = maximum(gi)
    U = zeros(2, G)
    for g in 1:G
        rows = findall(==(g), gi)
        obj(z) = begin
            u = L * z
            s = -0.5 * dot(z, z)
            for i in rows
                w[i] > 0 || continue
                s += w[i] * _fmw_rowlogdens((Y1[i] - mu1[i] - u[1], Y2[i] - mu2[i] - u[2]), o1[i], o2[i], s1[i], s2[i], ρ[i])
            end
            return s
        end
        z = zeros(2)
        for _ in 1:50
            gr = ForwardDiff.gradient(obj, z); H = ForwardDiff.hessian(obj, z)
            dz = -(H \ gr)
            z += dz
            norm(dz) < 1e-13 && break
        end
        U[:, g] = L * z
    end
    return U
end

# Dense Gaussian log-likelihood of the weighted density: per group, log N(r; 0, blockdiag(S_i / w_i) + Z Sigma_g Z')
# plus the term that turns the Gaussian with covariance S_i / w_i into the weighted density w_i log f_i.
function _fmw_loglik_dense(fit, y1, y2, w)
    n = length(w)
    mu1, mu2 = fit.means[:mu1], fit.means[:mu2]
    s1, s2, ρ = fit.scales[:sigma1], fit.scales[:sigma2], fit.scales[:rho12]
    Sg = Matrix{Float64}(fit.ranef.Sigma_g)
    gi = fit.ranef.group_index
    total = 0.0
    for g in 1:maximum(gi)
        cells = Tuple{Int,Int,Float64}[]          # (row, response, residual)
        c = 0.0
        for i in findall(==(g), gi)
            w[i] > 0 || continue
            o = (!ismissing(y1[i]), !ismissing(y2[i]))
            (o[1] || o[2]) || continue
            S = [s1[i]^2 ρ[i]*s1[i]*s2[i]; ρ[i]*s1[i]*s2[i] s2[i]^2]
            idx = [k for k in 1:2 if o[k]]
            Ss = S[idx, idx]
            c += -0.5 * ((w[i] - 1) * (logdet(Ss) + length(idx) * log(2π)) + length(idx) * log(w[i]))
            for k in idx
                push!(cells, (i, k, (k == 1 ? Float64(y1[i]) - mu1[i] : Float64(y2[i]) - mu2[i])))
            end
        end
        isempty(cells) && continue
        m = length(cells)
        V = zeros(m, m)
        for a in 1:m, b in 1:m
            ia, ka, _ = cells[a]; ib, kb, _ = cells[b]
            V[a, b] = Sg[ka, kb]
            if ia == ib
                S = [s1[ia]^2 ρ[ia]*s1[ia]*s2[ia]; ρ[ia]*s1[ia]*s2[ia] s2[ia]^2]
                V[a, b] += S[ka, kb] / w[ia]
            end
        end
        r = [x[3] for x in cells]
        total += -0.5 * (logdet(Symmetric(V)) + dot(r, Symmetric(V) \ r) + m * log(2π)) + c
    end
    return total
end

# (cell, formula, data, weight column, method, include?, response columns, ML?)
const _FMW_CELLS = [
    ("w_ml",      _FMW_F,  _FMW_D, :w,  :ML,   false, (:y1, :y2),   true),
    ("w_reml",    _FMW_F,  _FMW_D, :w,  :REML, false, (:y1, :y2),   false),
    ("wz_ml",     _FMW_F,  _FMW_D, :wz, :ML,   false, (:y1, :y2),   true),
    ("wunbal_ml", _FMW_F,  _FMW_U, :uw, :ML,   false, (:y1, :y2),   true),
    ("wincl_ml",  _FMW_FM, _FMW_D, :w,  :ML,   true,  (:y1m, :y2m), true),
]

@testset "bivariate (1 | p | g) with weights: conditional fitted() and ranef()" begin
    for (cell, f, d, wcol, method, incl, (c1, c2), isml) in _FMW_CELLS
        @testset "$cell" begin
            w = getproperty(d, wcol)
            kw = incl ? (; missing = miss_control(response = "include")) : (;)
            fit = with_logger(NullLogger()) do
                drm(f, Gaussian(); data = d, weights = w, method = method, kw...)
            end
            ref, rref = _FMW_REF[cell], _FMW_RREF[cell]
            y1, y2 = getproperty(d, c1), getproperty(d, c2)
            U = fit.ranef.effects[Symbol(fit.ranef.group)]
            # (1) log-likelihood and row count against R
            @test abs(fit.loglik - ref.ll) <= 1e-4
            @test fit.nobs == ref.nobs
            # (2) fitted() against R's fitted(), modes against R's ranef()
            ft = fitted(fit)
            @test ft isa AbstractDict && Set(keys(ft)) == Set([:mu1, :mu2])
            gap_fit = maximum(abs.(hcat(ft[:mu1], ft[:mu2]) .- ref.fit))
            @test gap_fit <= 1e-6
            @test sort(unique(String.(d.g))) == rref.groups       # Julia group order = R factor level order
            gap_ranef = maximum(abs.(U .- rref.U))
            @test gap_ranef <= 1e-6
            # (3) modes against the independent maximisation of the weighted joint density
            Ui = _fmw_modes(fit, y1, y2, w)
            gap_ind = maximum(abs.(U .- Ui))
            @test gap_ind <= 1e-6
            # fitted() is the fixed-effect mean plus the group mode, row by row
            gi = fit.ranef.group_index
            @test ft[:mu1] ≈ fit.means[:mu1] .+ U[1, gi] rtol = 1e-12
            @test ft[:mu2] ≈ fit.means[:mu2] .+ U[2, gi] rtol = 1e-12
            @test fitted(fit; re = :zero)[:mu1] == fit.means[:mu1]
            # (4) log-likelihood by hand (ML cells; the REML objective is not this quantity)
            gap_hand = isml ? abs(_fmw_loglik_dense(fit, y1, y2, w) - fit.loglik) : NaN
            isml && @test gap_hand <= 1e-8
        end
    end
end
