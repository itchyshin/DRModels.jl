"""
    DRModels

`DRModels.jl` — a Julia engine for distributional regression models, the Julia
twin of the R package **drmTMB**. Mirrors the gllvmTMB → GLLVModels.jl move.

The package covers univariate and bivariate distributional regression across
some twenty response families, with random, phylogenetic, spatial, pedigree and
supplied-matrix structure. Its origin is the **q=4 phylogenetic bivariate
location–scale model (PLSM)** — the selling-point model of drmTMB (Nakagawa et
al. 2025 MEE, Model 5). That marginal is a sparse **augmented-state Laplace
approximation** with an **exact O(p) gradient** (implicit-function / TMB-style,
via Takahashi selected inverse — never forms a dense p×p Σ_phy), optimised by
LBFGS with a fast-path-then-robust mode-finder.

For what is implemented and how far each route is tested, see the capability
matrix in the documentation. Measured comparisons against drmTMB include their
run conditions in `report/comparison-grid.md`; they are specific to the model
and data measured and are deliberately not quoted as package-wide figures here.

The engine retains the proof-of-concept's script-style organisation. Inference
provides Wald, profile, and parametric-bootstrap intervals. **Public:** opt-in
REML (`drm(method = :REML)`; its restricted correction covers all four
among-axis axes) and the conjugate-EM Gaussian phylo-mean solver
(`algorithm = :em`). A natural-gradient EM implementation failed the required
likelihood-parity check on `q4_p100`, so `algorithm = :natgrad` is not exposed;
the reusable Fisher metric is retained as engine infrastructure. **Experimental
prototypes are not wired:** SQUAREM EM, trust-region and line-search E-steps,
dense q=4 EM, and warm-start variants. Do not treat them as the public REML or
`:em` surface.
"""
module DRModels

# Soft source-level migration aid: users who have already loaded DRModels can
# qualify the old module name while moving calls to DRModels.  A renamed Julia
# package cannot keep `using DRM` alive because that resolves the package name
# before this module is loaded.
const DRM = DRModels

# Load the verified core engine. The relative @__DIR__ includes inside
# fit_q4_sparse_tmb.jl transitively pull the whole chain from this src/ dir.
include("fit_q4_sparse_tmb.jl")

# Guarded Hessian→covariance step shared by every family fitter. Must precede the
# family includes below; boundary-degenerate fits would otherwise hit an
# unguarded `inv` whose outcome depends on the LAPACK build.
include("vcov_guard.jl")

# Fisher / observed-information metric on log-Cholesky params (#13 S1b infra).
# Extracted after the natgrad solver failed the MLE-parity gate — not a public
# solver path. Feeds AI-REML / exact REML gradient follow-ups (#11 / #165).
include("lc_metric.jl")

# q=4 REML (Patterson–Thompson restricted likelihood; β_μ profiled out via a
# bordered augmented state). Additive — reuses the engine symbols above; ML
# remains the default and REML is opt-in (`method = :REML`). See #187.
include("reml_q4.jl")

# q=4 spherical/LKJ (D·R·D) OUTER reparameterization of the 4×4 among-axis covariance
# Σ_a (separation strategy: D = diag SDs, R = spherical/LKJ correlation-Cholesky).
# Additive — wraps the UNTOUCHED marginal_and_exact_grad; the engine still consumes
# the native log-Cholesky lc. Its value is conditioning/robustness at the σ-collapse
# boundary; it generalizes the verified q=2 atanh(ρ) bijection. See Ayumi #2.
include("fisherz_q4.jl")

# General-q multivariate-Brownian coevolution block (#188): reuses the q-agnostic
# sparse precision (kron(Q_tree, Λ⁻¹)) with a q×q Λ and an EXACT conjugate-Gaussian
# Laplace marginal. Independent of the q=4 location-scale leaf code above.
include("coevolution_q.jl")
include("reml_q2.jl")  # #470: REML for the bivariate q=2 structured residual-correlation route

# Gaussian location–scale front end (public bf()/drm() API).
include("gaussian_core.jl")
include("meta_vcov_bivariate.jl")  # A8: known bivariate sampling covariance (drmTMB meta_vcov_bivariate)
include("gaussian_bivariate.jl")
include("gaussian_ranef.jl")
include("gaussian_lss.jl")   # #544: location-scale-scale sd(g) ~ x on the (1|g) SD
include("aghq_1d.jl")            # #448: 1-D Liu–Pierce AGHQ around `_gauss_hermite`
include("gaussian_meta.jl")
include("gaussian_structured.jl")
include("gaussian_sparse_lss.jl")
include("phylo_interaction.jl")  # bipartite two-tree interaction RE: V = σ²(C_A⊗C_B) + σ_e²I
include("location_only.jl")      # #12: opt-in conjugate-EM for the Gaussian phylo-mean cell
include("student.jl")
include("skewnormal.jl")
include("poisson.jl")
include("sparse_laplace_glmm.jl")
include("negbinomial.jl")
include("beta.jl")
include("betabinomial.jl")
include("binomial.jl")
include("gamma.jl")
include("lognormal.jl")
include("bivariate_student.jl")  # drmTMB `biv_student()`: exact bivariate-t density,
                                 # ONE shared nu (logm2 link). Needs student.jl + gaussian_bivariate.jl.
include("bivariate_lognormal.jl")
include("associate_pairs.jl")    # drmTMB `associate_pairs()`: STAGED frozen-margin
                                 # latent-normal association (two-stage, not a joint fit). # drmTMB `biv_lognormal()`: closed form — delegates to the
                                 # Gaussian bivariate kernel on log(y) plus a Jacobian.
                                 # Must follow lognormal.jl (defines LogNormal) and
                                 # gaussian_bivariate.jl (defines the kernel it reuses).
include("zeroonebeta.jl")
include("tweedie.jl")
include("cumulative.jl")
include("link_residual.jl")      # S3: per-family link-scale variance for cross-family ρ
include("mixed_family.jl")       # S3: cross-family bivariate via shared-latent GHQ
include("mixed_family_postfit.jl") # post-fit accessors for fit_mixed_family
include("quantile_residuals.jl") # #183: per-family Dunn–Smyth quantile residuals
                                 # (included after all family types are defined)
include("locscale_kernels.jl")   # #202 groundwork: two-axis (mean+log-disp) kernels
include("locscale_inner.jl")     # #202 groundwork: q=2 augmented inner mode-finder
include("locscale_marginal.jl")  # #202 groundwork: q=2 Laplace marginal
include("locscale_fit.jl")       # #202 groundwork: end-to-end location–scale fit
include("locscale_grad.jl")      # #202 groundwork: exact O(p) outer gradient
include("locscale_infer.jl")     # #202 groundwork: Wald inference + RE summaries
include("locscale_whitened.jl")  # paired latent-coordinate value/gradient/information
include("locscale_profile.jl")   # #202: profile-likelihood CIs (trust-region inner solve)
include("locscale_frontend.jl")  # #202 slice 3b: drm() routing for (1|tag|group)
include("locscale_simulate.jl")  # marginal bootstrap draws both coupled latent axes
include("locscale_corr.jl")      # cluster ①: (1+x|g)/(0+x|g) reroute onto the q2 core
include("locscale_sigma.jl")     # cluster ②: standalone sigma ~ 1+(1|g) RE onto the q2 core
include("gaussian_locscale_phylo.jl")  # B1: Gaussian sigma~phylo(1|g) univariate route — separate/coupled/asymmetric + boundary CIs (Ayumi #2)
include("phylo_penalty.jl")      # A4c: penalized-MAP phylo variance components — drmTMB's drm_phylo_penalty()
include("inference.jl")
include("bias_correct.jl")       # TMB-style epsilon-method bias correction (#227 B11)
include("heritability.jl")       # comparative-biology derived ratios (h²/ICC) + CIs
include("coevo_accessors.jl")    # #188: q=4 coevolution among-axis correlation + variance accessors
include("profile_q4_phylo.jl")   # Ayumi #2: profile-likelihood CIs for the q=4 among-axis SDs (calibrated, no Hessian)
include("bootstrap_q4_phylo.jl") # Ayumi #2: parametric bootstrap of the q=4 among-axis SDs (boundary-honest CIs)
include("variational.jl")
include("ordinary_laplace.jl")      # Arc 2: marginal = :Laplace on ordinary (1 | g) (TMB convention)
include("summary.jl")
include("r2.jl")             # R2 for the constant-sigma Gaussian case ONLY; refuses elsewhere
include("visualization.jl")
include("plotting_ext.jl")   # #336: method-less drm_figure stub + thin plot_* (DRModelsMakieExt)
include("comparison.jl")
include("chibar.jl")             # chi-bar-square boundary p-values for variance-component LRTs
include("bridge.jl")
include("introspection.jl")     # A4d-2: profile_targets + structured_effects (drmTMB post-fit inventories)
include("missing_data.jl")       # #49: documented listwise-deletion preprocessing (no engine change)
include("joint_missing_predictor.jl") # #563: prepared exact joint-model prototypes
include("joint_missing_uncertainty.jl") # #563: native-shaped imputation summaries
include("joint_missing_two_predictor.jl") # #563: exact two-Gaussian prepared kernel
include("joint_missing_finite.jl") # #563: exact ordinal/categorical finite sums
include("joint_missing_frontend.jl") # #563: two fixed-effect joint formula routes
include("joint_missing_bridge.jl") # #563: primitive prepared-array transport
include("joint_missing_finite_bridge.jl") # #563: ordinal/categorical state transport

# Public API — the verified single-fit + scaling engine.
export AugProblem, make_problem,
       fit_q4_sparse_tmb, marginal_and_exact_grad, marginal_nll,
       lc_metric,
       fit_q4_sparse_fisherz, fz_DRD, fz_R, fz_correlations, fz_marginal_and_grad,
       fz_phi_to_lc, fz_init_from_Sigma,
       estep_mode, prior_precision, build_Huu, joint_grad, joint_nll, aug_prior_grad!,
       reml_objective_at,
       pack_theta, unpack_theta, lc_to_Λ, Λ_to_lc,
       augmented_phy, random_balanced_tree, random_caterpillar_tree, phylo_tree_height,
       augmented_tree_precision, sigma_phy_dense, takahashi_selinv,
       fit_phylo_interaction, phylo_interaction_nll, phylo_correlation,
       # general-q coevolution block (#188)
       CoevoProblem, make_coevo_problem, make_coevo_problem_from_precision,
       make_coevo_problem_from_covariance, coevo_marginal, coevo_marginal_cov,
       fit_coevolution, fit_coevolution_q2_residual, fit_coevolution_q2_reml,
       simulate_coevolution, coevo_pack, coevo_unpack, coevo_theta_len,
       lc_to_cov, cov_to_lc, lc_len

# Public API — the Gaussian distributional-regression front end.
export @formula, bf, drm_formula, drm, Gaussian, Student, SkewNormal, Poisson, NegBinomial2, TruncatedNegBinomial2, Beta, BetaBinomial, Binomial, Gamma, LogNormal, ZeroOneBeta, Tweedie, CumulativeLogit, cbind, meta_V, relmat, animal, phylo, spatial, sd, sd_phylo, DrmFormula, BivariateDrmFormula, DrmFit,
       coef, vcov, loglik, nobs, dof, aic, bic, fixef, re_sd, vc, ranef, sigma, corpairs, rho12, stderror, confint, coeftable, fitted, residuals, predict, predict_parameters, marginal_parameters, prediction_grid, simulate, bootstrap_ci, bootstrap_summary, bootstrap_result, bootstrap_sigma_a, check_drm, family,
       profile_result, profile_curve, parameter_surface, corpairs_data,
       drm_figure, plot_profile, plot_parameter_surface, plot_corpairs,
       gaussian_locscale_phylo_sds,
       profile_sigma_a,
       is_converged, niterations, deviance, dof_residual,
       lrtest, anova, aicc, weights, update,
       chibar_pvalue, lrt_boundary,
       bias_correct,
       heritability, repeatability, icc,
       coevolution_cor, coevolution_vc, coevolution_summary,
       reml_loglik, ml_loglik, estimation_method,
       drm_bridge, drm_bridge_inference, drm_bridge_objective_at,
       drm_listwise,
       associate_pairs, latent_normal, association, PairAssociation,
       integration_diagnostics,
       drm_phylo_penalty, drm_phylo_penalty_sweep, PhyloPenalty, PhyloCorPenaltyNeedsTwoSD,
       profile_targets, structured_effects,
       meta_vcov_bivariate, MetaVcovBivariate

# Public API — post-fit accessors for the cross-family bivariate fit
# (`fit_mixed_family`, currently reached as `DRModels.fit_mixed_family`).
export mf_coef, mf_aic, mf_bic, mf_fitted, mf_summary
export r2_constant_sigma

# Prepared joint missing-predictor API. This whole surface is Experimental: it
# accepts explicit numerical designs and includes bounded formula/mi and bridge
# wrappers. The admitted cells and limitations are documented separately.
export PreparedJointModel, PreparedJointFit, PreparedFiniteJointModel, PreparedFiniteJointFit, prepared_joint_model,
       prepared_joint_rowloglik, prepared_joint_nll,
       prepared_joint_conditional_moments, fit_prepared_joint,
       joint_missing_summary, imputed, mi, miss_control, impute_model,
       JointDrmFit, JointTwoDrmFit, JointFiniteDrmFit, JointMissingControl, JointImputeModel,
       CategoricalLogit, cutpoints

# Marginal method-selection surface (#136): VA/ELBO scaffold. Kept INTERNAL on
# purpose — the user-facing API is `method = :LA` / `:VA`, and exporting a bare
# `Laplace` would clash with `Distributions.Laplace`. Reach them as
# `DRModels.Variational` etc. if needed.

end # module DRModels
