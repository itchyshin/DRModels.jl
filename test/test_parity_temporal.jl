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
            # homtoep cells: drmTMB's Pearson residuals are the Levinson-whitened
            # L⁻¹ r; DRModels' `residuals(fit; type = :quantile)` is the same.
            if haskey(ex, "residuals")
                ref = Float64.(ex["residuals"]["pearson"])
                @test maximum(abs.(residuals(fit; type = :quantile) .- ref)) <= 1e-6
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
