using DRModels
using Test, LinearAlgebra

@testset "q4 direct export status contract" begin
    rows = DRModels._bridge_q4_direct_export_status()
    schema = DRModels._bridge_q4_direct_export_schema()

    @test length(rows) == 4
    @test all(row -> propertynames(row) == schema, rows)
    @test [row.axis for row in rows] == ["mu1", "mu2", "sigma1", "sigma2"]
    @test all(row -> row.dimension == "q4", rows)
    @test all(row -> row.route == "direct_drmjl", rows)
    @test all(row -> row.estimator == "ML", rows)
    @test [row.direct_sd_target for row in rows] == ["sd_mu1", "sd_mu2", "sd_sigma1", "sd_sigma2"]
    @test all(row -> row.sigma_a_source == "fit.ranef.Sigma_a", rows)
    @test all(row -> row.direct_status == "available_point_target", rows)
    @test all(row -> row.bridge_status == "experimental", rows)
    @test all(row -> row.inference_status == "point_target_only", rows)
    @test all(row -> occursin("no R-via-Julia q4 bridge parity", row.claim_boundary), rows)
    @test all(row -> occursin("interval coverage", row.claim_boundary), rows)

    validation = DRModels._bridge_q4_validate_direct_export_status(rows)
    @test validation.ok
    @test isempty(validation.errors)
    @test validation.n_rows == 4
    @test validation.schema == schema

    bad_rows = collect(rows)
    bad_rows[3] = merge(bad_rows[3], (direct_sd_target = "sd_mu1",))
    bad_validation = DRModels._bridge_q4_validate_direct_export_status(Tuple(bad_rows))
    @test !bad_validation.ok
    @test any(err -> occursin("direct_sd_target", err), bad_validation.errors)
end

@testset "q4 direct point export carries Sigma_a matrix" begin
    Σ = Matrix(Symmetric([
        0.1600 0.0400 0.0200 -0.0100
        0.0400 0.0900 0.0100 0.0150
        0.0200 0.0100 0.0400 0.0050
        -0.0100 0.0150 0.0050 0.0225
    ]))
    fake_fit = (
        ranef = (
            Sigma_a = Σ,
            axes = (:mu1, :mu2, :sigma1, :sigma2),
        ),
        estim_method = :ML,
    )

    point_export = DRModels._bridge_q4_point_export(fake_fit; family = "biv_gaussian")

    @test point_export["target"] == "gaussian_q4_phylo"
    @test point_export["dimension"] == "q4"
    @test point_export["family"] == "biv_gaussian"
    @test point_export["estimator"] == "ML"
    @test point_export["axes"] == ["mu1", "mu2", "sigma1", "sigma2"]
    @test point_export["sigma_a_source"] == "fit.ranef.Sigma_a"
    @test point_export["sigma_a"] ≈ Σ
    @test point_export["sd"]["mu1"] ≈ 0.4
    @test point_export["sd"]["mu2"] ≈ 0.3
    @test point_export["sd"]["sigma1"] ≈ 0.2
    @test point_export["sd"]["sigma2"] ≈ 0.15
    @test point_export["correlation"][1, 2] ≈ Σ[1, 2] / (0.4 * 0.3)
    @test point_export["correlation"][3, 4] ≈ Σ[3, 4] / (0.2 * 0.15)
    @test occursin("no R-via-Julia q4 bridge parity", point_export["claim_boundary"])
    @test occursin("interval coverage", point_export["claim_boundary"])

    non_q4 = (ranef = nothing, estim_method = :ML)
    @test isempty(DRModels._bridge_q4_point_export(non_q4; family = "gaussian"))
end

@testset "q4 phylocov names match log-Cholesky packing order" begin
    @test DRModels._q4_phylocov_names() == [
        "Sigma_a:L11",
        "Sigma_a:L21",
        "Sigma_a:L31",
        "Sigma_a:L41",
        "Sigma_a:L22",
        "Sigma_a:L32",
        "Sigma_a:L42",
        "Sigma_a:L33",
        "Sigma_a:L43",
        "Sigma_a:L44",
    ]
end
