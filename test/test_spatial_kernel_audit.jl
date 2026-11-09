using DRModels, Test, LinearAlgebra, Statistics
import DelimitedFiles

# Spatial-kernel audit: the bivariate q=2 / q=4 spatial routes build their fixed-range
# covariance from drmTMB's native kernel exp(-d / median positive distance) + 1e-6 I when
# `spatial_range` is not given (it used to be the MEAN distance with 1e-8 jitter).
# Reference values are committed R/TMB fits (`engine = "tmb"`) of the committed data in
# test/data/spatial_kernel_audit; the R generator is test/data/spatial_kernel_audit/r_gen_fit.R.
# Nothing here depends on the Julia version's RNG.

const _SKA_DIR = joinpath(@__DIR__, "data", "spatial_kernel_audit")

function _ska_read(path)
    raw, hdr = DelimitedFiles.readdlm(path, ','; header = true)
    return Dict(String(h) => raw[:, j] for (j, h) in enumerate(vec(hdr)))
end

# R writes NaN standard errors (rows we do not compare) as `NA`.
_ska_num(x) = x isa Number ? Float64(x) : (t = strip(string(x)); t == "NA" ? NaN : parse(Float64, t))

function _ska_load(tag)
    dir = joinpath(_SKA_DIR, tag)
    d = _ska_read(joinpath(dir, "data.csv")); cc = _ska_read(joinpath(dir, "coords.csv"))
    data = (; y1 = Float64.(d["y1"]), y2 = Float64.(d["y2"]), x1 = Float64.(d["x1"]),
            site = String.(d["site"]))
    coords = hcat(Float64.(cc["x"]), Float64.(cc["y"]))
    @assert String.(cc["site"]) == unique(data.site)
    rf = _ska_read(joinpath(dir, "r_fixed.csv")); rc = _ska_read(joinpath(dir, "r_cor.csv"))
    rl = readlines(joinpath(dir, "r_ll.txt"))
    return (; data, coords, r_est = _ska_num.(rf["est"]), r_se = _ska_num.(rf["se"]),
            r_cor = Dict((String(a), String(b)) => _ska_num(v)
                         for (a, b, v) in zip(rc["from_dpar"], rc["to_dpar"], rc["estimate"])),
            r_ll = parse(Float64, rl[1]), r_conv = rl[2])
end

_ska_form(q4::Bool) = q4 ?
    bf(mu1 = @formula(y1 ~ x1 + spatial(1 | site)), mu2 = @formula(y2 ~ x1 + spatial(1 | site)),
       sigma1 = @formula(sigma1 ~ 1 + spatial(1 | site)), sigma2 = @formula(sigma2 ~ 1 + spatial(1 | site)),
       rho12 = @formula(rho12 ~ 1)) :
    bf(mu1 = @formula(y1 ~ x1 + spatial(1 | site)), mu2 = @formula(y2 ~ x1 + spatial(1 | site)),
       sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))

# Fit with DIRECT `coords` input; return the quantities R reports, in R's order.
function _ska_fit(fx, q4::Bool; spatial_range = nothing, q4_f_reltol = nothing)
    kw = spatial_range === nothing ? (;) : (; spatial_range)
    q4_f_reltol === nothing || (kw = merge(kw, (; q4_f_reltol)))
    fit = q4 ?
        drm(_ska_form(true), Gaussian(); data = fx.data, coords = fx.coords, q4_g_tol = 1e-6,
            q4_iterations = 2000, q4_n_newton = 40, q4_vcov = false, kw...) :
        drm(_ska_form(false), Gaussian(); data = fx.data, coords = fx.coords, g_tol = 1e-8, kw...)
    S = fit.ranef.Sigma_a; sd = sqrt.(diag(S)); Rm = S ./ (sd * sd'); ax = fit.ranef.axes
    est = Float64[]
    for blk in (:mu1, :mu2, :sigma1, :sigma2, :rho12); append!(est, coef(fit, blk)); end
    append!(est, log.(sd))
    cors = Dict((String(ax[i]), String(ax[j])) => Rm[i, j] for i in eachindex(ax) for j in (i+1):length(ax))
    return (; fit, est, cors, ll = loglik(fit))
end

function _ska_compare(fx, r)
    n = length(r.est)                      # R's table also lists the correlation link rows after these
    d = r.est .- fx.r_est[1:n]
    dc = maximum(abs(v - get(fx.r_cor, k, get(fx.r_cor, (k[2], k[1]), NaN))) for (k, v) in r.cors)
    return (; abs = maximum(abs, d), se_units = maximum(abs.(d) ./ fx.r_se[1:n]), cor = dc, ll = abs(r.ll - fx.r_ll))
end

@testset "spatial kernel audit: bivariate default kernel is drmTMB's native kernel" begin
    fx = _ska_load("q2a"); G = size(fx.coords, 1)
    D = [norm(fx.coords[k, :] - fx.coords[l, :]) for k in 1:G, l in 1:G]
    pos = [D[k, l] for k in 1:G for l in (k+1):G]
    expected = exp.(-D ./ median(pos)) + 1e-6I           # drmTMB: drm_spatial_coords_precision
    @test maximum(abs.(DRModels._spatial_covariance_from_coords(:site, G, fx.coords, nothing) .- expected)) < 1e-14
    @test DRModels._spatial_covariance_from_coords(:site, G, fx.coords, nothing) ==
          DRModels._gaussian_spatial_sd_kernel(fx.coords, G, :site)
    @test DRModels._q4_default_spatial_range(fx.coords, G) == median(pos)
    # An explicit range and the legacy mean-range rule are unchanged.
    @test DRModels._spatial_covariance_from_coords(:site, G, fx.coords, 0.5) == exp.(-D ./ 0.5) + 1e-8I
    @test DRModels._spatial_covariance_from_coords(:site, G, fx.coords, nothing; legacy_mean_range = true) ==
          exp.(-D ./ (sum(D) / (G^2 - G))) + 1e-8I
    # 1-D transect given as a plain vector works on every default path.
    v = [0.0, 1.0, 3.0, 7.0]
    @test DRModels._spatial_covariance_from_coords(:s, 4, v, nothing) ==
          DRModels._spatial_covariance_from_coords(:s, 4, reshape(v, 4, 1), nothing)
    @test DRModels._gaussian_spatial_sd_kernel(v, 4, :s) == DRModels._gaussian_spatial_sd_kernel(reshape(v, 4, 1), 4, :s)
    @test DRModels._q4_default_spatial_range(v, 4) == median([1.0, 3.0, 7.0, 2.0, 6.0, 4.0])
    # Duplicate sites: median over POSITIVE distances only.
    @test DRModels._q4_default_spatial_range([0.0, 0.0, 1.0, 4.0], 4) == median([1.0, 4.0, 1.0, 4.0, 3.0])   # the zero distance is excluded
end

@testset "spatial kernel audit: bivariate q=2 spatial agrees with drmTMB at the optimum" begin
    for tag in ("q2a", "q2b", "q2c")
        fx = _ska_load(tag); @test fx.r_conv == "0"
        r = _ska_fit(fx, false); @test r.fit.converged
        c = _ska_compare(fx, r)
        @test c.abs <= 1e-3
        @test c.se_units <= 0.01
        @test c.cor <= 5e-3
        @test c.ll <= 1e-3
    end
    # Control: the retired mean-distance rule (explicit range) is measurably off R on the
    # same data, so the assertions above are not insensitive to the kernel.
    fx = _ska_load("q2a"); G = size(fx.coords, 1)
    D = [norm(fx.coords[k, :] - fx.coords[l, :]) for k in 1:G, l in 1:G]
    old = _ska_compare(fx, _ska_fit(fx, false; spatial_range = sum(D) / (G^2 - G)))
    @test old.se_units > 0.05
    @test old.ll > 0.05
end

@testset "spatial kernel audit: bivariate q=4 spatial agrees with drmTMB at the optimum" begin
    # R/drmTMB is already at its optimum (r_tight_ll.txt: re-optimising with tolerances of 1e-15
    # moves its loglik by <= 3e-7). The Julia DEFAULT stops on a relative objective change of 1e-7
    # and so can sit up to ~1.2e-3 below the optimum on these flat ridges (1.10.12: 1.18e-3 on q4c,
    # which broke an earlier 1e-3 bound). With `q4_f_reltol = 1e-12` it reaches R to <= 4e-5.
    for tag in ("q4a", "q4b", "q4c")
        fx = _ska_load(tag); @test fx.r_conv == "0"
        r_tight = parse(Float64, strip(read(joinpath(_SKA_DIR, tag, "r_tight_ll.txt"), String)))
        G = size(fx.coords, 1)
        med = median([norm(fx.coords[k, :] - fx.coords[l, :]) for k in 1:G for l in (k+1):G])
        r = _ska_fit(fx, true; q4_f_reltol = 1e-12); @test r.fit.converged
        c = _ska_compare(fx, r)
        @test c.se_units <= 0.01
        @test c.abs <= 1e-2
        @test c.cor <= 5e-3
        @test abs(r.ll - r_tight) <= 1e-4
        @test r.fit.ranef.spatial_range == med
        # Default tolerance stays covered, with the documented looser loglik bound.
        r0 = _ska_fit(fx, true); @test r0.fit.converged
        @test abs(r0.ll - fx.r_ll) <= 3e-3
        # The default-stop fit sits on the same flat ridge, so where it stops (and its SE
        # agreement with R) moves with the Julia version: max over q4a/q4b/q4c of se_units is
        # 0.0068 on 1.12.7 but 0.0106 on 1.13.2 (q4c). 0.015 still checks that the SEs track R
        # without asserting the tight-tolerance agreement the default stopping rule does not buy.
        @test _ska_compare(fx, r0).se_units <= 0.015
    end
end
