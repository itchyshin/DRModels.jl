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
# φ or decay, `(1 | id)` SD, σ). No interval, coverage or calibration claim.

module TestParityTemporal

using DRModels
using Test
using TOML

include(joinpath(@__DIR__, "parity", "temporal_parity.jl"))

@testset "temporal parity cells reproduce drmTMB (always on)" begin
    cells = temporal_parity_cells()
    @test length(cells) == 8
    @test Set(basename.(cells)) == Set(["ar1-gapped", "ar1-gapped-ri", "ou-irregular",
        "ou-irregular-ri", "vignette-ar1", "vignette-ar1-ri", "vignette-ou", "vignette-ou-ri"])
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
        end
    end
end

end # module
