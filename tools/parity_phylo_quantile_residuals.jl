# parity_phylo_quantile_residuals.jl — the DRModels.jl half of
# tools/parity_phylo_quantile_residuals.R (#995). Run by that script; it can
# also be run by hand on a fixture directory the R script wrote:
#
#   julia --project=. tools/parity_phylo_quantile_residuals.jl <dir> <case> ...
#
# For each <case> it reads <dir>/<case>.tree.nwk and <dir>/<case>.data.csv
# (columns y, x, species; species are tip labels), fits
#
#   drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
#       Gaussian(); data, tree)
#
# by ML (the default), and writes <dir>/<case>.julia.csv with, per row:
#   q_julia     residuals(fit; type = :quantile)           (marginal PIT over the phylo effect)
#   fitted      fitted(fit) = Xβ̂ + û                        (conditional mean)
#   mu_fixed    Xβ̂
#   cii         C_ii, the row's raw branch-length tip variance
#   q_cond      (y − fitted) / σ̂_e                          (conditional form)
#   q_gh32      the same marginal PIT from the 32-node Gauss–Hermite
#               mixture used before the closed form (diagnostic only: what PR
#               #1078's head 8563231 returned)
# and <dir>/<case>.julia.toml with the scalars (estimates, logLik, convergence,
# the DRModels.jl commit, and the expected spread of sd(q_julia) under the fitted
# model: a plug-in value and a seeded 200-refit parametric bootstrap).
using DRModels, LinearAlgebra, Printf, Random, Statistics
import Distributions

const NBOOT = 200

function _read_fixture(path)
    lines = readlines(path)
    hdr = split(lines[1], ',')
    iy = findfirst(==("y"), hdr); ix = findfirst(==("x"), hdr); is = findfirst(==("species"), hdr)
    y = Float64[]; x = Float64[]; sp = String[]
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        f = split(ln, ',')
        push!(y, parse(Float64, f[iy])); push!(x, parse(Float64, f[ix]))
        push!(sp, strip(f[is], '"'))
    end
    return (; y, x, species = sp)
end

_git_head() = try
    root = pkgdir(DRModels)
    head = strip(read(`git -C $root rev-parse HEAD`, String))
    # `+dirty` when src/ differs from that commit, so the stamp names the code that ran.
    clean = success(`git -C $root diff --quiet HEAD -- src`)
    clean ? head : head * "+dirty"
catch
    "unknown"
end

function run_case(dir, case)
    nwk = strip(read(joinpath(dir, "$case.tree.nwk"), String))
    phy = augmented_phy(nwk)
    d = _read_fixture(joinpath(dir, "$case.data.csv"))
    f = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1))
    fit = drm(f, Gaussian(); data = d, tree = phy)
    fit.ranef isa DRModels.LocOnlyRanef ||
        error("$case: expected the sparse Gaussian phylo route (LocOnlyRanef), got $(typeof(fit.ranef))")
    q = residuals(fit; type = :quantile)
    μc = fitted(fit)
    μf = fit.ranef.mu_fixed
    cii = DRModels._loconly_row_tip_variance(fit.ranef)
    bl = Dict(fit.blocks)
    β = fit.theta[bl[:mu]]
    σe = exp(fit.theta[first(bl[:sigma])])
    σb = exp(fit.theta[first(bl[:resd])])
    qc = (d.y .- μc) ./ σe
    z, w = DRModels._gauss_hermite(32)
    N01 = Distributions.Normal()
    qgh = map(eachindex(q)) do i
        s = sqrt(2.0) * σb * sqrt(cii[i])
        F = sum(w[k] / sqrt(π) * Distributions.cdf(Distributions.Normal(μf[i] + s * z[k], σe), d.y[i])
                for k in eachindex(z))
        Distributions.quantile(N01, clamp(F, eps(), 1 - eps()))
    end
    open(joinpath(dir, "$case.julia.csv"), "w") do io
        println(io, "q_julia,fitted,mu_fixed,cii,q_cond,q_gh32")
        for i in eachindex(q)
            @printf(io, "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", q[i], μc[i], μf[i], cii[i], qc[i], qgh[i])
        end
    end
    open(joinpath(dir, "$case.julia.toml"), "w") do io
        println(io, "drmodels_commit = \"", _git_head(), "\"")
        println(io, "julia_version = \"", VERSION, "\"")
        println(io, "converged = ", fit.converged)
        @printf(io, "loglik = %.17g\n", fit.loglik)
        @printf(io, "beta0 = %.17g\nbeta1 = %.17g\n", β[1], β[2])
        @printf(io, "sigma_e = %.17g\nsd_phylo_raw = %.17g\n", σe, σb)
        @printf(io, "sd_phylo_drmtmb_scale = %.17g\n", re_sd(fit; scale = :drmtmb, tree = phy)[:species])
        @printf(io, "tree_height = %.17g\n", phylo_tree_height(phy))
        # Rows of related tips are correlated under the model, so even a
        # calibrated population-level PIT has a sample variance below 1:
        # E[s²] = (tr R − 1'R1/n) / (n − 1), R the fitted marginal correlation.
        C = sigma_phy_dense(phy)
        li = DRModels._phylo_mean_leaf_index(phy, d.species)
        V = σe^2 .* Matrix(I, length(d.y), length(d.y)) .+ σb^2 .* C[li, li]
        sd_v = sqrt.(diag(V))
        R = V ./ (sd_v * transpose(sd_v))
        n = length(d.y)
        # Plug-in value: the fitted V is taken as the truth, and the estimation of
        # β and of the variances is ignored.
        @printf(io, "pit_expected_sample_var = %.17g\n", (tr(R) - sum(R) / n) / (n - 1))
        # Spread: a parametric bootstrap at the fitted parameters (simulate y from
        # N(Xβ̂, V), refit, take sd of the quantile residuals), seeded.
        X = hcat(ones(n), d.x)
        Lv = cholesky(Symmetric(V)).L
        rng = Random.MersenneTwister(995)
        sds = map(1:NBOOT) do _
            ys = X * β .+ Lv * randn(rng, n)
            fb = drm(f, Gaussian(); data = (; y = ys, x = d.x, species = d.species), tree = phy)
            std(residuals(fb; type = :quantile))
        end
        qs = Statistics.quantile(sds, [0.05, 0.5, 0.95])
        @printf(io, "pit_sd_boot_n = %d\n", NBOOT)
        @printf(io, "pit_sd_boot_q05 = %.17g\npit_sd_boot_q50 = %.17g\npit_sd_boot_q95 = %.17g\n", qs...)
        @printf(io, "pit_sd_boot_frac_le_observed = %.17g\n", mean(sds .<= std(q)))
    end
    return nothing
end

dir = ARGS[1]
for case in ARGS[2:end]
    run_case(dir, case)
end
