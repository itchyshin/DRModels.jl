# test_api_stability.jl — the API freeze gate for the v0.7 line (Wave A of the completion roadmap; release label v0.7.0 per D-181,
# docs/dev-log/plans/2026-08-28-v1.0-roadmap.md; decisions D-180).
#
# Every exported name is classified into exactly one of three tiers, and the
# union must EQUAL `names(DRModels)` — so a new export cannot appear without being
# classified, a stable name cannot vanish or be renamed without failing this
# test loudly, and a tier change is a reviewed edit here, never drift.
#
#   STABLE        — the SemVer promise: these names, their meanings, and their
#                   drmTMB-mirroring conventions (`sigma` never `tau`, `rho12`,
#                   `meta_V`) do not break within the 0.7 line.
#   EXPERIMENTAL  — exported and usable, EXEMPT from the promise, and said so
#                   in docs/src/api-stability.md: the R bridge (ledger
#                   r_bridge_status experimental), the cross-family surface
#                   (permanent claim_boundary, D-179 #3), association helpers,
#                   the penalized-MAP surface, and #136's VA scaffold edges.
#   ENGINE        — the computational spine exported for scripting and bench
#                   work (problem builders, mode-finders, packers, tree
#                   utilities). Exempt like Base internals: stable in practice,
#                   not promised.

using DRModels
using Test

const API_STABLE = [
    # grammar + front end
    "@formula", "bf", "drm_formula", "drm", "cbind",
    "meta_V", "relmat", "animal", "phylo", "spatial", "offset",
    "DrmFormula", "BivariateDrmFormula", "DrmFit",
    # families
    "Gaussian", "Student", "SkewNormal", "Poisson", "NegBinomial2",
    "TruncatedNegBinomial2", "TruncatedPoisson", "Beta", "BetaBinomial", "Binomial", "Gamma",
    "LogNormal", "ZeroOneBeta", "Tweedie", "CumulativeLogit",
    # StatsAPI-style accessors
    "coef", "vcov", "loglik", "loglikelihood", "nobs", "dof", "aic", "bic", "aicc",
    "deviance", "dof_residual", "weights", "update",
    "fixef", "re_sd", "vc", "ranef", "sigma", "corpairs", "rho12",
    "stderror", "confint", "coeftable", "fitted", "residuals",
    "predict", "predict_parameters", "marginal_parameters", "prediction_grid",
    "simulate", "family", "is_converged", "niterations",
    "estimation_method", "reml_loglik", "ml_loglik",
    # inference
    "bootstrap_ci", "bootstrap_summary", "bootstrap_result",
    "bootstrap_sigma_a", "check_drm",
    "profile_result", "profile_curve", "profile_targets", "profile_sigma_a",
    "parameter_surface", "corpairs_data",
    "lrtest", "anova", "chibar_pvalue", "lrt_boundary", "bias_correct",
    "heritability", "repeatability", "icc",
    "coevolution_cor", "coevolution_vc", "coevolution_summary",
    "structured_effects", "gaussian_locscale_phylo_sds",
    # figures
    "drm_figure", "plot_profile", "plot_parameter_surface", "plot_corpairs",
]

const API_EXPERIMENTAL = [
    # r2_constant_sigma (2026-09-06): EXPERIMENTAL on purpose. The value it
    # returns is settled -- it is the ordinary lm() R2 -- but its REFUSAL
    # BOUNDARIES are what a caller actually programs against, and those may widen:
    # a marginal/conditional R2 for random-effect fits would reshape this surface,
    # and that decision has not been made. Promising the boundaries under SemVer
    # before deciding them would be the wrong claim.
    "r2_constant_sigma",
    # R bridge (ledger r_bridge_status: experimental)
    "drm_bridge", "drm_bridge_inference", "drm_bridge_objective_at", "drm_listwise",
    "bridge_diagnostics",
    # cross-family surface (permanent claim_boundary, D-179 #3)
    "mf_coef", "mf_aic", "mf_bic", "mf_fitted", "mf_summary",
    "associate_pairs", "latent_normal", "association", "PairAssociation",
    "integration_diagnostics",
    # penalized MAP (A4c)
    "drm_phylo_penalty", "drm_phylo_penalty_sweep",
    "PhyloPenalty", "PhyloCorPenaltyNeedsTwoSD",
    # bivariate meta
    "meta_vcov_bivariate", "MetaVcovBivariate",
    # location-scale-scale (#544/#545)
    "sd", "sd_phylo",
    # prepared joint missing-predictor surface (post-v0.7 experimental)
    "PreparedJointModel", "PreparedJointFit",
    "PreparedFiniteJointModel", "PreparedFiniteJointFit",
    "JointDrmFit", "JointTwoDrmFit", "JointFiniteDrmFit",
    "JointMissingControl", "JointImputeModel", "CategoricalLogit",
    "prepared_joint_model", "prepared_joint_rowloglik", "prepared_joint_nll",
    "prepared_joint_conditional_moments", "fit_prepared_joint",
    "joint_missing_summary", "imputed", "mi", "miss_control",
    "impute_model", "cutpoints",
]

const API_ENGINE = [
    "AugProblem", "make_problem",
    "fit_q4_sparse_tmb", "marginal_and_exact_grad", "marginal_nll",
    "lc_metric",
    "fit_q4_sparse_fisherz", "fz_DRD", "fz_R", "fz_correlations",
    "fz_marginal_and_grad", "fz_phi_to_lc", "fz_init_from_Sigma",
    "estep_mode", "prior_precision", "build_Huu", "joint_grad", "joint_nll",
    "aug_prior_grad!", "reml_objective_at",
    "pack_theta", "unpack_theta", "lc_to_Λ", "Λ_to_lc",
    "lc_to_cov", "cov_to_lc", "lc_len",
    "augmented_phy", "random_balanced_tree", "random_caterpillar_tree",
    "phylo_tree_height", "augmented_tree_precision", "sigma_phy_dense",
    "takahashi_selinv",
    "fit_phylo_interaction", "phylo_interaction_nll", "phylo_correlation",
    "CoevoProblem", "make_coevo_problem", "make_coevo_problem_from_precision",
    "make_coevo_problem_from_covariance", "coevo_marginal",
    "coevo_marginal_cov", "fit_coevolution", "fit_coevolution_q2_residual",
    "fit_coevolution_q2_reml", "simulate_coevolution",
    "coevo_pack", "coevo_unpack", "coevo_theta_len",
]

@testset "API freeze gate (v0.7 line)" begin
    exported = Set(string.(names(DRModels)))
    classified = vcat(API_STABLE, API_EXPERIMENTAL, API_ENGINE, ["DRModels"])

    @testset "no name is classified twice" begin
        @test length(classified) == length(Set(classified))
    end

    @testset "classification is TOTAL: every export has a tier" begin
        unclassified = sort(collect(setdiff(exported, Set(classified))))
        isempty(unclassified) ||
            @info "UNCLASSIFIED exports (add each to a tier here, deliberately)" unclassified
        @test isempty(unclassified)
    end

    @testset "no phantom classifications" begin
        phantoms = sort(collect(setdiff(Set(classified), exported)))
        isempty(phantoms) ||
            @info "classified but NOT exported (removed or renamed?)" phantoms
        @test isempty(phantoms)
    end

    @testset "every STABLE name is exported and defined" begin
        for nm in API_STABLE
            sym = Symbol(nm)
            @test sym in names(DRModels)
            @test isdefined(DRModels, sym)
        end
    end

    @testset "naming contract holds on the stable tier" begin
        # drmTMB-mirroring conventions: sigma never tau; rho12 for the
        # bivariate residual correlation; meta_V not meta_known_V.
        @test !("tau" in API_STABLE)
        @test "sigma" in API_STABLE
        @test "rho12" in API_STABLE
        @test "meta_V" in API_STABLE
        @test !any(n -> occursin("meta_known", n), string.(names(DRModels)))
    end
end
