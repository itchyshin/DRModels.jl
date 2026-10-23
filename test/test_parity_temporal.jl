# test_parity_temporal.jl — always-on check that DRModels.jl reproduces the
# stored drmTMB numbers for the temporal AR1 / OU parity cells (D-310).
#
# The cells and their provenance live in test/parity/temporal/; the helpers in
# test/parity/temporal_parity.jl are shared with the gated runner
# test/parity/runparity_temporal.jl (DRM_PARITY_TESTS=1), which additionally
# exercises drmTMB's keyword spelling through `drm_bridge` and prints the
# measured differences. Every cell is a closed-form Gaussian fit of at most a
# few hundred rows, so the whole set runs in seconds.
#
# Claim fence: point estimates at the ML optimum only (logLik, β, process SD,
# φ or decay, `(1 | id)` SD or phylogenetic stable SD, homtoep lag
# correlations, σ). No interval, coverage or calibration claim.

module TestParityTemporal

using DRModels
using Test
using TOML
using Logging

include(joinpath(@__DIR__, "parity", "temporal_parity.jl"))

@testset "temporal parity cells reproduce drmTMB (always on)" begin
    cells = temporal_parity_cells()
    @test length(cells) == 13
    @test Set(basename.(cells)) == Set(["ar1-gapped", "ar1-gapped-ri", "ou-irregular",
        "ou-irregular-ri", "vignette-ar1", "vignette-ar1-ri", "vignette-ou", "vignette-ou-ri",
        "phylo-ou-species", "vignette-phylo-ou",              # wave 2 (D-311): phylo + OU
        "homtoep-panel6", "homtoep-neg4", "vignette-homtoep"]) # wave 2: homogeneous Toeplitz
    for dir in cells
        @testset "$(basename(dir))" begin
            meta = TOML.parsefile(joinpath(dir, "expected.meta.toml"))
            @test length(meta["drmtmb_sha"]) == 40
            @test haskey(meta, "r_version") && haskey(meta, "r_call")
            ex = TOML.parsefile(joinpath(dir, "expected.toml"))
            @test ex["fit"]["method"] == "ML"
            @test ex["status"]["converged"] && ex["status"]["pdHess"]
            fit, rows = temporal_parity_check(dir)
            @test is_converged(fit)
            for r in rows
                @test r.pass
            end
            # Article cells: drmTMB's CONDITIONAL fitted values (all rows).
            # DRModels' fitted() is population-level, so add the modes back.
            if haskey(ex, "conditional")
                g = ex["fit"]["group"]
                ids = temporal_parity_data(ex["fit"]["data_file"]; group = g,
                    time = _temporal_time_column(ex["fit"]["julia_formula"]),
                    structure = ex["fit"]["structure"])[Symbol(g)]
                re = ranef(fit)
                cond = fitted(fit) .+ re[Symbol(g)]
                for k in (Symbol("$(g)_iid"), Symbol("$(g)_phylo"))   # per-series modes
                    haskey(re, k) && (cond = cond .+ re[k][indexin(ids, unique(ids))])
                end
                ref = Float64.(ex["conditional"]["fitted"])
                @test length(ref) == length(cond)
                @test maximum(abs.(cond .- ref)) <= 1e-6
            end
            # drmTMB's Pearson residuals; every cell carries them. AR1 / OU and
            # paired phylo() + OU cells: conditional on the fitted modes,
            # (y − Xβ̂ − modes)/σ̂. homtoep cells: the Levinson-whitened L⁻¹ r.
            # DRModels' `residuals(fit; type = :quantile)` is the same in all.
            # Measured on Julia 1.10.12 and 1.13: 6.9e-7 (vignette-ou), 2.9e-7
            # (vignette-ar1-ri), ≤ 2.1e-9 on every other cell. The two larger
            # gaps are drmTMB's, not DRModels': drmTMB takes its modes from the
            # random entries of TMB's `last.par.best` (drmTMB 07d1612ea
            # R/drmTMB.R:787), which on those two cells are not the modes at its
            # reported optimum, so its residual misses its own exact
            # σ̂V⁻¹(y − Xβ̂) by those amounts; rebuilt from `last.par` (the modes
            # re-solved at `opt$par`) it is exact to ≤ 3.1e-15. DRModels'
            # residual is exact, so the 1e-6 here only absorbs drmTMB's mode
            # noise. The tight check below does not depend on it.
            @test haskey(ex, "residuals")
            if haskey(ex, "residuals")
                z = residuals(fit; type = :quantile)
                ref = Float64.(ex["residuals"]["pearson"])
                @test maximum(abs.(z .- ref)) <= 1e-6
                # Tight, and immune to drmTMB's mode noise: the exact
                # conditional residual at drmTMB's stored estimates (AR1 / OU
                # cells without a tree). Measured ≤ 2.1e-9 (ar1-gapped-ri), the
                # agreement of the two packages' estimates.
                if ex["fit"]["structure"] in ("ar1", "ou") && !haskey(ex["fit"], "tree_file")
                    @test maximum(abs.(z .- temporal_dense_conditional_residuals(ex))) <= 1e-8
                end
            end
            # drmTMB's mean-coefficient profile intervals (article cells and the
            # homtoep cells). Each package locates the endpoints with its own
            # root search. Measured on Julia 1.10.12: 1.3e-6 on vignette-homtoep,
            # at most 6.6e-6 over all cells (homtoep-neg4, vignette-ou-ri);
            # enforced at 1e-5.
            for pr in get(ex, "profile", Any[])
                cname = replace(pr["parm"], "fixef:mu:" => "")
                ci = only(with_logger(NullLogger()) do
                    confint(fit; method = :profile, parm = :mu => cname)
                end)
                gap = max(abs(ci.lower - pr["lower"]), abs(ci.upper - pr["upper"]))
                println("  profile endpoint gap ", basename(dir), " ", cname, ": ", gap)
                @test gap <= 1e-5
            end
        end
    end
end

end # module
