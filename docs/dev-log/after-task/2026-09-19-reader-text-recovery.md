# After-task — reader-first documentation recovery

## 1. Scope

Reworked public Documenter routes and prose for readers approaching DRModels.jl
from biology or from its R twin. No model, API, formula, or evidence claim was
changed.

## 2. Reader outcome

The site now starts with a scientific question, makes the optional R bridge
explicit, separates public learning material from development material, and
states uncertainty limits without internal-ledger language.

## 3. Changed surfaces

Getting started, capabilities, bridge/Rosetta pages, phylogenetic and spatial
tutorials, prediction/post-fit guidance, model selection, navigation, and the
reader-surface audit.

## 4. Evidence reviewed

Five independent reader lenses covered community data, phylogeny, the R/Julia
bridge, uncertainty/model selection, and spatial or repeated-measure designs.

## 5. Implementation

Added a public-only prose audit and navigation assertions; moved Development
out of Reference in the Documenter menu.

## 6. Tests

- `python3 tools/reader_surface_audit.py --public-only`
- `julia --project=. tools/tests/test_docs_navigation.jl`
- `python3 -m unittest tools.tests.test_reader_surface_audit`
- `env DRM_DOCS_DEPLOY=false julia --project=docs docs/make.jl`
- `git diff --check`

All passed.

## 7. Boundaries retained

Julia remains optional for R users. A successful fit remains distinct from a
validated uncertainty claim. Figures and visual redesign are outside this
text-only slice.

## 8. Risk check

No source-engine files, public APIs, formula grammar, capability ledgers, or
numeric parity fixtures changed.

## 9. Follow-up

The rendered-doc parity audit has one deployment-only `versions.js` exception
to formalize without weakening the generic auditor.

## 10. Integration

This is a documentation-and-tests-only pull request and is eligible for the
normal review/CI gate.

## 11. Handoff

Next reader work should compare the corresponding journeys in drmTMB and
gllvmTMB, not force their article inventories to be identical.
