#!/usr/bin/env julia
#
# bootstrap-issues.jl — seed the DRModels.jl GitHub work ledger (labels, milestones,
# the ~18 near-term issues). Idempotent for labels (--force) and tolerant for
# milestones (dup titles are skipped). Run once from the repo root:
#
#     julia tools/bootstrap-issues.jl
#
# Requires `gh` authenticated for itchyshin/DRModels.jl. Article (26) and family (8)
# slices are NOT opened here — they live as checklists in the roadmap issues and
# are promoted to real issues when their phase opens (Workflow D / H).

run_ok(cmd) = try; run(cmd); true; catch err; @warn "skipped" cmd err; false; end

# ---- labels (kept in sync with .github/labels.yml) ------------------------
labels = [
    ("roadmap","0e8a16","Per-phase pinned tracking issue"),
    ("enhancement","a2eeef","New capability — family, structured effect, article"),
    ("bug","d73a4a","Defect / regression / parity drift"),
    ("documenter","1d76db","Lands in docs/src/"),
    ("r-bridge","5319e7","Hopper / Lovelace — parity, engine=julia, RCall"),
    ("workflow","fbca04","Changes to .claude/workflows/*.js"),
    ("engine-quality","0052cc","Workflow Q gate work"),
    ("formula-parity","c2e0c6","drmTMB <-> DRModels.jl formula contract (Boole)"),
    ("idea","d4c5f9","Pólya scouting signal / creative combination"),
    ("autoresearch","bfdadc","Workflow R estimator-optimisation experiment"),
    ("phase-0","ededed","Phase 0 — Team & workflows"),
    ("phase-1","ededed","Phase 1.x"),
    ("phase-2","ededed","Phase 2 — Family expansion"),
    ("phase-3","ededed","Phase 3 — Articles"),
    ("status:stable","0e8a16","drmTMB status: Stable"),
    ("status:first-slice","fbca04","drmTMB status: First slice"),
    ("status:opt-in-control","0052cc","drmTMB status: Opt-in control"),
    ("status:planned","ededed","drmTMB status: Planned or reserved"),
    ("status:blocked","b60205","drmTMB status: Unsupported or blocked"),
    ("good-first-codex","7057ff","Hand-off: good for Codex"),
    ("good-first-claude","7057ff","Hand-off: good for Claude"),
]
println("== labels ==")
for (n,c,d) in labels
    run_ok(`gh label create $n --color $c --description $d --force`)
end

# ---- milestones -----------------------------------------------------------
milestones = [
    ("Phase 0 — Team & workflows","Team, scripted workflows, ledger, Documenter shell. No engine changes."),
    ("Phase 1.0 — Hygiene + wire experimental/","Wire src/experimental/ to the API; fill easy docs; complete Q gates."),
    ("Phase 1.1 — bf() front end + R parity","drmTMB-exact bf(); RCall parity live; thread the bootstrap."),
    ("Phase 1.5 — R-side engine=julia","drmTMB(..., engine = \"julia\") via JuliaCall; bridge glue in the R repo."),
    ("Phase 2 — Family expansion","Student/lognormal/Gamma/Tweedie/beta/Poisson/nbinom2/cumulative_logit."),
    ("Phase 3 — Articles to mirror drmTMB","Fill the remaining articles; target drmTMB's 26."),
    ("v0.1.0","Gaussian uni+bivariate + inference + docs + R-bridge; register in General."),
    ("v1.0","Full twin — every drmTMB capability matched."),
]
println("== milestones ==")
ms_path = "repos/{owner}/{repo}/milestones"  # string var: Julia backticks reject literal {}
for (t,d) in milestones
    run_ok(`gh api $ms_path -f title=$t -f description=$d`)
end

# ---- issues (~18 near-term) -----------------------------------------------
# (title, body, comma-labels, milestone)
issues = [
  ("Roadmap: Phase 0 — Team & workflows",
   "Tracking issue for Phase 0. See ROADMAP.md.\n\n- [x] AGENTS.md (12 personas), CLAUDE.md, ROADMAP.md\n- [x] 10 scripted workflows (.claude/workflows/)\n- [x] 12 Codex agents (.codex/agents/)\n- [x] dev-log scaffold + tools/drm-checkpoint.jl\n- [x] Documenter shell (36 status-tagged stubs)\n- [x] GitHub ledger (labels/milestones/issues)\n- [ ] Phase 0 gate verified (engine load + bench logLik -256.51 + docs build)",
   "roadmap,phase-0","Phase 0 — Team & workflows"),
  ("Roadmap: Phase 1.0 — Hygiene + wire experimental/",
   "Wire src/experimental/ into the public API; fill the easy Get-Started docs; complete Workflow Q gates; pin docs/ + bench/ Manifests; first Workflow R run.",
   "roadmap,phase-1","Phase 1.0 — Hygiene + wire experimental/"),
  ("Roadmap: Phase 1.1 — bf() front end + R parity",
   "Workflow B (drmTMB-exact bf() incl. reserved-syntax rejections), Workflow G (RCall parity vs vendored drmTMB v0.1.3), thread+measure the bootstrap, first application articles + Rosetta page.",
   "roadmap,phase-1","Phase 1.1 — bf() front end + R parity"),
  ("Roadmap: Phase 1.5 — R-side engine=julia",
   "Lovelace: drmTMB(formula, ..., engine = \"julia\") via JuliaCall (glue lives in the drmTMB R repo). Hopper guards equivalence.\n\n- [ ] JuliaCall bridge in drmTMB R repo\n- [ ] result-shape parity\n- [ ] round-trip bf() formulas",
   "roadmap,phase-1,r-bridge","Phase 1.5 — R-side engine=julia"),
  ("Roadmap: Phase 2 — Family expansion",
   "Workflow H per family (preserve sigma<->phi mappings):\n\n- [ ] student\n- [ ] lognormal\n- [ ] Gamma\n- [ ] tweedie\n- [ ] beta\n- [ ] poisson\n- [ ] nbinom2\n- [ ] cumulative_logit",
   "roadmap,phase-2","Phase 2 — Family expansion"),
  ("Roadmap: Phase 3 — Articles to mirror drmTMB",
   "Fill the Documenter articles (Workflow D), target drmTMB's 26. Slugs:\n\nModel Guides: model-map, which-scale, distribution-families, model-workflow, convergence, large-data\nTutorials: location-scale, robust-student, count-nbinom2, proportion-beta-binomial, bivariate-coscale, meta-analysis, structural-dependence, animal-models, phylogenetic-models, spatial-models, relmat-known-matrices, phylogenetic-spatial\nDiagnostics: figure-gallery, implementation-map, testing-likelihoods, simulation-plot-grammar\nDeveloper: formula-grammar, adding-families, source-map\n+ get-started",
   "roadmap,phase-3","Phase 3 — Articles to mirror drmTMB"),
  ("Roadmap: v0.1.0",
   "Release gate: Gaussian uni+bivariate (q=4 PLSM headline) + inference + docs published + R-bridge functional. Then register in the Julia General registry.",
   "roadmap","v0.1.0"),
  ("Roadmap: v1.0",
   "Full twin — every drmTMB capability matched, speed edge documented per family.",
   "roadmap","v1.0"),
  ("Wire infer_q4.jl (inference) into the public API",
   "Promote src/experimental/infer_q4.jl (Wald + bootstrap) into a clean inference module; port GLLVM.jl confint* patterns. Owner: Fisher. Thread + measure the bootstrap.",
   "enhancement,phase-1","Phase 1.0 — Hygiene + wire experimental/"),
  ("Wire reml_q4.jl (REML) into the public API",
   "Promote src/experimental/reml_q4.jl (Laplace-REML). ML stays the default; REML an option. Mean-axis bias-correction verified; scale-axis + exact REML gradient are open. Owner: Fisher/Noether.",
   "enhancement,phase-1","Phase 1.0 — Hygiene + wire experimental/"),
  ("Wire location_only.jl (conjugate EM) into the public API",
   "Promote src/experimental/location_only.jl (conjugate EM vs LBFGS; EM 3.1x on the conjugate sub-model). Owner: Noether/Curie.",
   "enhancement,phase-1","Phase 1.0 — Hygiene + wire experimental/"),
  ("Wire fit_em_natgrad.jl (natural-gradient EM) into the public API",
   "Promote src/experimental/fit_em_natgrad.jl. Natural gradient = Fisher scoring = AI-REML (info-geometry scout). Owner: Noether.",
   "enhancement,phase-1","Phase 1.0 — Hygiene + wire experimental/"),
  ("Q-gate: finite-difference gradient check <= 1e-6",
   "Wire the FD-vs-exact gradient gate into Workflow Q. Already validated at 1e-6 in the poc; make it a standing test.",
   "engine-quality,phase-1","Phase 1.0 — Hygiene + wire experimental/"),
  ("Q-gate: Allocs.jl zero-alloc inner mode-finder loop",
   "Wire the Allocs.jl gate; keep the inner mode-finder loop zero-allocation. Owner: Karpinski.",
   "engine-quality,phase-1","Phase 1.0 — Hygiene + wire experimental/"),
  ("Q-gate: multi-shape sweep (balanced + caterpillar, p in {100,1k,10k})",
   "Wire the multi-shape scaling gate using the O(p) precision sampler. Owner: Karpinski/Curie.",
   "engine-quality,phase-1","Phase 1.0 — Hygiene + wire experimental/"),
  ("Q-gate: R-parity vs vendored drmTMB v0.1.3 outputs",
   "Wire Workflow G (DRM_PARITY_TESTS=1) against vendored drmTMB v0.1.3 reference outputs in test/parity/fixtures/. Generated outputs only — never GPL source. Owner: Hopper.",
   "engine-quality,r-bridge,phase-1","Phase 1.1 — bf() front end + R parity"),
  ("Design: public verb — drm() vs fit() vs drmTMB()",
   "Decide DRModels.jl's entry-point verb (the twin's public surface). Paste-and-run feel for R users vs Julia idiom. Resolve before Workflow B. Owner: Boole/Emmy. Record in docs/dev-log/decisions/.",
   "idea,formula-parity,phase-1","Phase 1.1 — bf() front end + R parity"),
  ("Design: phylo-tree + R-object marshalling",
   "Decide Newick/phylo reading (Phylo.jl vs minimal parser) and how pedigree / Ainv / K cross the R<->Julia bridge. Needed for real trees + the bridge. Owner: Hopper/Noether.",
   "idea,r-bridge,phase-1","Phase 1.1 — bf() front end + R parity"),
]
println("== issues ==")
for (t,b,l,m) in issues
    run_ok(`gh issue create --title $t --body $b --label $l --milestone $m`)
end
println("\nDone. Check: gh issue list  /  gh api repos/{owner}/{repo}/milestones --jq '.[].title'")
