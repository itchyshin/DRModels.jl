# refs #1378 — guard: fixture row orders must not use Julia-version-dependent RNG draws.
#
# `randperm` / `shuffle` on `MersenneTwister`, `Xoshiro`, or the default RNG can change
# between Julia versions and silently change test data. New tests should use `StableRNGs`
# or a fixed permutation (see test/README.md). This file lists known legacy call sites;
# shrink the set when migrating a file, and fail if a new file appears.
using Test

function _fixture_rng_violators(dir::AbstractString)
    violators = String[]
    for f in sort(filter(f -> occursin(r"^test_.*\.jl$", f), readdir(dir)))
        path = joinpath(dir, f)
        for line in eachline(path)
            s = strip(line)
            startswith(s, '#') && continue
            occursin(r"\b(randperm|shuffle)\s*\(", s) || continue
            occursin("StableRNG", s) && continue
            push!(violators, f)
            break
        end
    end
    return violators
end

# Frozen at 2026-11-23 on main (66 files). Remove a name when that file is migrated.
const _KNOWN_MT_FIXTURE_FILES = [
    "test_1089_missing_rows_dropped.jl",
    "test_1093_beta_laplace_sentinel.jl",
    "test_957_spatial_coords_labels.jl",
    "test_995_tip_variance_linear.jl",
    "test_aghq_1d.jl",
    "test_beta_structured_sd.jl",
    "test_beta_zob_raw_k_regression.jl",
    "test_binomial_animal_ainv_sd_regression.jl",
    "test_binomial_animal_ainv_slope_sd_regression.jl",
    "test_binomial_animal_iid_sd_regression.jl",
    "test_binomial_animal_q2_sd.jl",
    "test_binomial_animal_slope_iid_sd_regression.jl",
    "test_binomial_nonlogit_animal_slope_sd_regression.jl",
    "test_binomial_phylo_interaction_sd.jl",
    "test_binomial_phylo_interaction_slope_sd.jl",
    "test_binomial_phylo_sd_regression.jl",
    "test_binomial_phylo_spatial_q2_sd.jl",
    "test_binomial_probit_animal_sd_regression.jl",
    "test_binomial_relmat_sd_regression.jl",
    "test_binomial_relmat_slope_sd_regression.jl",
    "test_binomial_sd_regression.jl",
    "test_bridge_labelled_structured.jl",
    "test_gamma_animal_intercept_slope_sd.jl",
    "test_gamma_animal_sd_regression.jl",
    "test_gamma_animal_slope_sd_regression.jl",
    "test_gamma_relmat_sd_regression.jl",
    "test_gamma_spatial_sd_regression.jl",
    "test_gamma_structured_slope_sd_regression.jl",
    "test_gap_close_q2.jl",
    "test_gaussian_ordinary_q2_sd.jl",
    "test_gaussian_ordinary_q3_q4_sd.jl",
    "test_gaussian_sigma_phylo_spatial_sd.jl",
    "test_gaussian_structured_q2_reml.jl",
    "test_gaussian_structured_q2_sd.jl",
    "test_gaussian_structured_sd_intercept.jl",
    "test_gaussian_structured_sd_slope.jl",
    "test_iid_sd_quadrature_accuracy.jl",
    "test_labelled_ordinary_q2_sd.jl",
    "test_labelled_phylo_spatial_q2_sd.jl",
    "test_labelled_rho_q2_sd.jl",
    "test_metamorphic_sd_routes.jl",
    "test_missing_response_bivariate.jl",
    "test_nb2_animal_sd_regression.jl",
    "test_nb2_gamma_structured_sigma.jl",
    "test_nb2_relmat_sd_regression.jl",
    "test_nb2_spatial_slope_sd_regression.jl",
    "test_phylo_interaction.jl",
    "test_phylo_spatial_q2_sd.jl",
    "test_poisson_animal_intercept_slope_sd.jl",
    "test_poisson_animal_sd_regression.jl",
    "test_poisson_animal_slope_sd_regression.jl",
    "test_poisson_relmat_intercept_slope_sd.jl",
    "test_poisson_relmat_sd_regression.jl",
    "test_poisson_relmat_slope_sd_regression.jl",
    "test_q2_sd_simulate.jl",
    "test_sd_ids_labels.jl",
    "test_stored_loglik_at_optimum.jl",
    "test_structured_conditional_export.jl",
    "test_structured_plus_ordinary_bar.jl",
    "test_temporal_homtoep.jl",
    "test_temporal_phylo_ou.jl",
    "test_temporal_residuals.jl",
    "test_tweedie_relmat_sd_regression.jl",
    "test_tweedie_relmat_sd_validation.jl",
    "test_va_frontend_families.jl",
    "test_va_frontend_poisson.jl",
]

@testset "#1378 fixture RNG guard" begin
    found = _fixture_rng_violators(@__DIR__)
    @test sort(found) == sort(_KNOWN_MT_FIXTURE_FILES)
    extra = setdiff(found, _KNOWN_MT_FIXTURE_FILES)
    @test isempty(extra)
end