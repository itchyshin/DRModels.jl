# runparity_temporal.jl — gated temporal AR1 / OU parity vs drmTMB (D-310).
# Runs under DRM_PARITY_TESTS=1 (wired from test/runtests.jl).
#
# For every cell in test/parity/temporal/: (1) the native Julia spelling
# `temporal(1 | id, time, ar1|ou)` reproduces drmTMB's logLik to 1e-8
# (absolute) and β, process SD, φ / decay, the `(1 | id)` SD and σ to 1e-6
# (relative) — the `[tol]` of each cell, tightened from the 1e-6 / 1e-5
# contract after the first run measured ≤ 1e-11 / ≤ 1e-8; (2) the same
# model sent through `drm_bridge` in drmTMB's own keyword spelling (`temporal(1 | id, time = occ, structure = "ar1")`, the
# `[fit].formula` string) reaches the same logLik. A table of the measured
# differences is printed so the achieved precision is on record.
#
# Wave 2 (D-311) adds the paired `phylo(1 | species) + temporal(…, ou)` cells
# (`[fit].tree_file`, a phylogenetic stable SD). A cell whose drmTMB fit put an
# SD at zero lists it in `[tol].boundary`: that SD is checked as "both engines
# below 1e-3" and the logLik tolerance is the cell's own (1e-6 there). The
# homogeneous Toeplitz cells compare the lag correlations (`cor`), on the
# absolute scale.
#
# The always-on twin, test/test_parity_temporal.jl, repeats check (1) so CI
# catches a drift without the gate.

using DRModels
using Test
using TOML
using Printf

isdefined(@__MODULE__, :temporal_parity_check) || include("temporal_parity.jl")

@testset "temporal AR1/OU (+ wave-2 phylo + OU, homtoep) parity vs drmTMB (D-310, D-311)" begin
    cells = temporal_parity_cells()
    @test length(cells) >= 4
    worst = Dict{String,Float64}()
    for dir in cells
        cell = basename(dir)
        @testset "$cell" begin
            fit, rows = temporal_parity_check(dir)
            tol = TOML.parsefile(joinpath(dir, "expected.toml"))["tol"]
            boundary = get(tol, "boundary", String[])
            # A boundary SD leaves a singular Hessian, which `is_converged` flags;
            # the optimiser itself must still have converged.
            @test fit.converged
            @test is_converged(fit) || !isempty(boundary)
            for r in rows
                @test r.pass
                r.pass || @error "temporal parity FAILED" cell r.quantity r.julia r.drmtmb r.absdiff r.reldiff
                r.quantity in boundary && continue          # checked as "both < 1e-3", not relative
                key = r.quantity == "logLik" ? "logLik (abs)" :
                      startswith(r.quantity, "mu_") ? "beta (rel)" : "$(r.quantity) (rel)"
                worst[key] = max(get(worst, key, 0.0),
                                 r.quantity == "logLik" ? r.absdiff : r.reldiff)
            end
            println(@sprintf("  %-16s ", cell),
                    join([@sprintf("%s %.1e", r.quantity, r.quantity == "logLik" ? r.absdiff : r.reldiff)
                          for r in rows], "  "))

            # drmTMB's keyword spelling through the bridge: same model, same optimum.
            ex = TOML.parsefile(joinpath(dir, "expected.toml"))["fit"]
            data = temporal_parity_data(ex["data_file"]; group = ex["group"],
                                        time = _temporal_time_column(ex["julia_formula"]),
                                        structure = ex["structure"])
            # drmTMB's R side strips `tree = tree` from `phylo()` and ships the
            # Newick separately (wave-2 paired cells); do the same here.
            form = replace(ex["formula"], r"phylo\(1 \| (\w+), tree = \w+\)" => s"phylo(1 | \1)")
            out = drm_bridge(; formula = form, family = "gaussian", data = data,
                             tree = _temporal_tree(ex))
            @test abs(out["loglik"] - Float64(ex["loglik"])) <= Float64(tol["atol_loglik"])
        end
    end
    println("  worst over cells: ",
            join([@sprintf("%s %.1e", k, worst[k]) for k in sort(collect(keys(worst)))], "  "))
end
