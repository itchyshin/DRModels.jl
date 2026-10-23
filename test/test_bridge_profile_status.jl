using DRModels, Test

@testset "bridge exposes selected profile failure" begin
    sdrows = [
        (param=:resd_mu, coef="phylo", estimate=1.1, lower=0.6, upper=1.8),
        (param=:resd_sigma, coef="phylo", estimate=0.9, lower=0.4, upper=1.5),
    ]
    @test DRModels._bridge_parse_sd_parm("sd:mu") === :resd_mu
    @test DRModels._bridge_parse_sd_parm("sd:sigma") === :resd_sigma
    @test DRModels._bridge_parse_sd_parm("sd:resd") === :resd
    @test DRModels._bridge_parse_sd_parm("sd:resd_mu") === :resd_mu
    @test DRModels._bridge_parse_sd_parm("sd:resd_sigma") === :resd_sigma
    @test DRModels._bridge_pick_sd_row(sdrows, :resd_sigma).param === :resd_sigma
    @test_throws ArgumentError DRModels._bridge_parse_sd_parm("sd:rho")
    row = (param=:mu, coef="x", estimate=0.0, lower=-Inf, upper=1.0)
    good = (param=:mu, coef="y", lower_endpoint_failed=false, upper_endpoint_failed=false,
        lower_unbounded=false, upper_unbounded=false)
    bad = (param=:mu, coef="x", lower_endpoint_failed=true, upper_endpoint_failed=false,
        lower_unbounded=false, upper_unbounded=false)
    result = (stats=[bad, good], failed=1)
    outcome = DRModels._bridge_profile_outcome(result, row)
    @test outcome.status == "profile_failed"
    @test occursin("failed", outcome.message)
    detailed = merge(bad,(lower_nuisance_reason=:not_converged,
        lower_nuisance_method=:lbfgs_finite,lower_nuisance_fallback=false))
    explanation = DRModels._bridge_profile_outcome((stats=[detailed],failed=1),row).message
    @test occursin("not_converged",explanation)
    @test occursin("lbfgs_finite",explanation)
    @test occursin("fallback=false",explanation)
    @test DRModels._bridge_profile_outcome(result, merge(row,(coef="y",))).status == "profile"
    # No-crossing within the searched range is different from a failed nuisance solve.
    searched = merge(bad, (lower_endpoint_failed=false,lower_unbounded=true))
    outcome = DRModels._bridge_profile_outcome((stats=[searched],failed=0), row)
    @test outcome.status == "profile"
    @test occursin("searched range",outcome.message)
    # Location-scale root diagnostics must reach the R-facing message, separately
    # from the nuisance solver's reason, and only for the selected coefficient.
    root_lower = (reason=:max_iterations, candidate=-3.0, residual=4.0)
    unrelated = (param=:mu, coef="y", lower=(reason=:unrelated_failure,))
    root_row = (param=:mu, coef="x", lower=root_lower)
    root_result = (stats=[detailed, good], failed=1,
                   endpoint_diagnostics=[unrelated, root_row])
    root_outcome = DRModels._bridge_profile_outcome(root_result, row)
    @test root_outcome.status == "profile_failed"
    @test occursin("endpoint=max_iterations", root_outcome.message)
    @test occursin("candidate=-3.0", root_outcome.message)
    @test occursin("residual=4.0", root_outcome.message)
    @test occursin("nuisance=not_converged", root_outcome.message)
    @test !occursin("unrelated_failure", root_outcome.message)
    @test DRModels._bridge_profile_outcome(root_result, merge(row,(coef="y",))).status == "profile"
    # A missing per-row diagnostic must not erase an aggregate failure.
    @test DRModels._bridge_profile_outcome((stats=NamedTuple[], failed=1),row).status == "profile_failed"
    @test DRModels._bridge_profile_outcome((stats=NamedTuple[], failed=0),row).status == "profile"
end
