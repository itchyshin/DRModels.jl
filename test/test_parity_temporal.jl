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
# φ or decay, `(1 | id)` SD or phylogenetic stable SD, σ). No interval, coverage or calibration claim.

module TestParityTemporal

using DRModels
using Test
using TOML

include(joinpath(@__DIR__, "parity", "temporal_parity.jl"))

@testset "temporal parity cells reproduce drmTMB (always on)" begin
    cells = temporal_parity_cells()
    @test length(cells) == 10
    @test Set(basename.(cells)) == Set(["ar1-gapped", "ar1-gapped-ri", "ou-irregular",
        "ou-irregular-ri", "vignette-ar1", "vignette-ar1-ri", "vignette-ou", "vignette-ou-ri",
        "phylo-ou-species", "vignette-phylo-ou"])            # last two: wave 2 (D-311)
    for dir in cells
        @testset "$(basename(dir))" begin
            meta = TOML.parsefile(joinpath(dir, "expected.meta.toml"))
            @test length(meta["drmtmb_sha"]) == 40
            @test haskey(meta, "r_version") && haskey(meta, "r_call")
            ex = TOML.parsefile(joinpath(dir, "expected.toml"))
            @test ex["fit"]["method"] == "ML"
            @test ex["status"]["converged"]
            # drmTMB's Hessian is singular exactly when an SD sits at its zero
            # boundary (`[tol].boundary`, e.g. σ̂ → 0 in drmTMB's 32-row
            # phylo + OU article data); otherwise it must be positive definite.
            @test ex["status"]["pdHess"] || !isempty(get(ex["tol"], "boundary", String[]))
            fit, rows = temporal_parity_check(dir)
            # A boundary SD leaves a singular Hessian, which `is_converged` flags;
            # the optimiser itself must still have converged.
            @test fit.converged
            @test is_converged(fit) ||
                  !isempty(get(TOML.parsefile(joinpath(dir, "expected.toml"))["tol"], "boundary", String[]))
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
        end
    end
end

end # module
