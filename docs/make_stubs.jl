#!/usr/bin/env julia
#
# make_stubs.jl — generate the Phase 0 Documenter stub pages.
#
# SKIP-IF-EXISTS: never clobbers a page that already exists, so it is safe to
# re-run after Workflow D fills articles. It only creates missing stubs. The
# per-page table here is also the single place the navbar's status tags live.
# Run from the repo root:  julia docs/make_stubs.jl

const SRC     = joinpath(@__DIR__, "src")
const BASE    = "https://itchyshin.github.io/drmTMB/"
const ROADMAP = "https://github.com/itchyshin/DRModels.jl/blob/main/ROADMAP.md"

# (relpath under docs/src/, title, status, drmTMB url suffix ("" = DRModels.jl-only), what DRModels.jl has today)
pages = [
    # Model Guides
    ("model-guides/model-map.md", "What can I fit today?", "First slice", "articles/model-map.html",
        "the verified q=4 PLSM path; the full capability matrix fills as the API lands."),
    ("model-guides/which-scale.md", "Which scale are you modelling?", "Planned or reserved", "articles/which-scale.html",
        "the `sigma` (residual) vs group-level SD vs known-V distinction — documented as the API lands."),
    ("model-guides/distribution-families.md", "Choosing response families", "Planned or reserved", "articles/distribution-families.html",
        "Gaussian (the engine today); the other families arrive in Phase 2."),
    ("model-guides/model-workflow.md", "Checking and using fitted models", "Planned or reserved", "articles/model-workflow.html",
        "post-fit checks arrive with the public API (Phase 1.0+)."),
    ("model-guides/convergence.md", "Improving convergence", "First slice", "articles/convergence.html",
        "the mode-finder notes: off-diagonal Λ0 init (lc3/lc7), the Watanabe boundary, relative-objective stopping."),
    ("model-guides/large-data.md", "Working with large data", "Stable", "articles/large-data.html",
        "the O(p) precision sampler and near-linear scaling to p=10,000 (`bench/run_scaling.jl`)."),

    # Tutorials
    ("tutorials/location-scale.md", "When variance carries signal", "First slice", "articles/location-scale.html",
        "the q=4 location–scale engine fits this; the `bf()` front end lands in Phase 1.1."),
    ("tutorials/robust-student.md", "Robust continuous responses", "Planned or reserved", "articles/robust-student.html",
        "Student-t arrives in Phase 2."),
    ("tutorials/count-nbinom2.md", "Count abundance and extra zeros", "Planned or reserved", "articles/count-nbinom2.html",
        "Poisson / NB2 + zero-inflation arrive in Phase 2."),
    ("tutorials/proportion-beta-binomial.md", "Proportions and success rates", "Planned or reserved", "articles/proportion-beta-binomial.html",
        "beta / beta-binomial arrive in Phase 2."),
    ("tutorials/bivariate-coscale.md", "Changing residual coupling with rho12", "First slice", "articles/bivariate-coscale.html",
        "the verified bivariate q=4 PLSM — the DRModels.jl headline."),
    ("tutorials/meta-analysis.md", "Mean effects and residual heterogeneity", "Planned or reserved", "articles/meta-analysis.html",
        "`gaussian()` + `meta_V()` — planned."),
    ("tutorials/structural-dependence.md", "Structural dependence overview", "First slice", "articles/structural-dependence.html",
        "phylogenetic q=4 covariance (the engine); spatial / animal / relmat are planned."),
    ("tutorials/animal-models.md", "Animal models and additive relatedness", "Planned or reserved", "articles/animal-models.html",
        "`animal()` (pedigree / A / Ainv) — planned; needs the R-object marshalling design issue."),
    ("tutorials/phylogenetic-models.md", "Phylogenetic structured effects", "First slice", "articles/phylogenetic-models.html",
        "the phylogenetic q=4 engine; user-tree (Newick) I/O is a Phase 1.1 design issue."),
    ("tutorials/spatial-models.md", "Coordinate-spatial structured effects", "Planned or reserved", "articles/spatial-models.html",
        "`spatial(coords=)` — planned."),
    ("tutorials/relmat-known-matrices.md", "Known-matrix relatedness with relmat", "Planned or reserved", "articles/relmat-known-matrices.html",
        "`relmat(K=/Q=)` — planned."),
    ("tutorials/phylogenetic-spatial.md", "Structural dependence details", "Planned or reserved", "articles/phylogenetic-spatial.html",
        "the phylo × spatial theory — planned."),

    # Diagnostics & Validation
    ("diagnostics-and-validation/figure-gallery.md", "Figure gallery", "Planned or reserved", "articles/figure-gallery.html",
        "the Confidence Eye contract (Florence) arrives with the first figures in Phase 1.1."),
    ("diagnostics-and-validation/implementation-map.md", "Implementation map", "First slice", "articles/implementation-map.html",
        "the verified-engine map (see `report/q4-sparse-status.md`)."),
    ("diagnostics-and-validation/testing-likelihoods.md", "Testing likelihoods", "Planned or reserved", "articles/testing-likelihoods.html",
        "likelihood cross-checks arrive with the inference module."),
    ("diagnostics-and-validation/simulation-plot-grammar.md", "Simulation plot grammar", "Planned or reserved", "articles/simulation-plot-grammar.html",
        "the ADEMP harness — Curie, Phase 1+."),

    # Developer Notes
    ("developer-notes/formula-grammar.md", "Formula grammar", "First slice", "articles/formula-grammar.html",
        "Boole's drmTMB-exact contract (incl. the reserved-syntax rejections); the parser lands in Phase 1.1."),
    ("developer-notes/adding-families.md", "Adding distribution families", "Planned or reserved", "articles/adding-families.html",
        "Workflow H — Phase 2."),
    ("developer-notes/source-map.md", "Implemented source map", "Stable", "articles/source-map.html",
        "the `src/` engine map: `sparse_phy`, `takahashi_selinv`, `sparse_aug_plsm`, `fit_q4_sparse_tmb`."),

    # Reference (6 workflow-ordered categories, mirroring drmTMB)
    ("reference/package.md", "Package", "Reference", "reference/index.html",
        "the `DRModels` module overview (1 item in drmTMB)."),
    ("reference/model-specification.md", "Model specification", "Reference", "reference/index.html",
        "`bf()` / `drm_formula()` + family constructors (13 items in drmTMB)."),
    ("reference/structured-effect-markers.md", "Structured-effect markers", "Reference", "reference/index.html",
        "`phylo` / `spatial` / `animal` / `relmat` / `corpair` / `sd*` (6 in drmTMB)."),
    ("reference/deprecated-marker-internals.md", "Deprecated marker internals", "Reference", "reference/index.html",
        "`meta_known_V`, `gr` — kept as parity stubs only (2 in drmTMB)."),
    ("reference/model-fitting-and-postfit.md", "Model fitting and post-fit tools", "Reference", "reference/index.html",
        "`fit`, `check_drm`, `fixef`, `ranef`, `sigma`, `corpairs`, `predict`, `simulate` (23 in drmTMB)."),
    ("reference/visualization.md", "Visualization", "Reference", "reference/index.html",
        "`plot_corpairs`, `plot_parameter_surface` (3 in drmTMB)."),

    # DRModels.jl-specific pages
    ("r-julia-bridge.md", "R ↔ Julia bridge", "Planned or reserved", "",
        "Hopper + Lovelace: `drmTMB(..., engine = \"julia\")` via JuliaCall (Phase 1.5). The bridge glue lives in the drmTMB R repo."),
    ("rosetta.md", "Rosetta — R ↔ Julia", "Planned or reserved", "",
        "the same model side by side in drmTMB (R) and DRModels.jl (Julia) — fills as the `bf()` front end lands."),
]

function stub(title, status, url, today)
    mirror = isempty(url) ? "" : "Mirrors drmTMB's [$title]($BASE$url). "
    """
    # $title

    !!! note "Status — $status"
        $(mirror)**In DRModels.jl today:** $today

    *Phase 0 stub — filled via Workflow D (`mirror-article`). See the [roadmap]($ROADMAP).*
    """
end

created, skipped = 0, 0
for (rel, title, status, url, today) in pages
    path = joinpath(SRC, rel)
    mkpath(dirname(path))
    if isfile(path)
        global skipped += 1
    else
        write(path, stub(title, status, url, today))
        global created += 1
    end
end

println("make_stubs: created $created, skipped $skipped (already existed).")
