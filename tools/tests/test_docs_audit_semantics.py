"""Lexical smoke guards for reviewed documentation wording, not runtime proof.

Source review and the runtime/example receipts establish implementation behavior.
These tests only prevent the specifically corrected prose omissions recurring.
"""
from pathlib import Path
import re
import unittest
ROOT = Path(__file__).resolve().parents[2]


def markdown_table_cells(line):
    """Split a Markdown table row on unescaped pipes, preserving escaped pipes."""
    cells = []
    current = []
    i = 0
    while i < len(line):
        char = line[i]
        if char == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            current.append("|")
            i += 2
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        i += 1
    cells.append("".join(current).strip())
    if cells and cells[0] == "":
        cells.pop(0)
    if cells and cells[-1] == "":
        cells.pop()
    return cells


def assert_markdown_table_arity(markdown):
    """Reject a table body row whose number of cells differs from its header."""
    lines = markdown.splitlines()
    for i, header in enumerate(lines[:-1]):
        if "|" not in header or not re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*", lines[i + 1]):
            continue
        expected = len(markdown_table_cells(header))
        j = i + 2
        while j < len(lines) and "|" in lines[j] and lines[j].strip():
            actual = len(markdown_table_cells(lines[j]))
            if actual != expected:
                raise AssertionError(f"table row {j + 1} has {actual} cells; expected {expected}: {lines[j]}")
            j += 1


def assert_tested_definition_scope(markdown):
    """Keep numerical-check status distinct from worked-analysis/coverage claims."""
    intro = re.sub(r"\s+", " ", markdown.split("## Response families", 1)[0])
    if "numerical checks for the listed" not in intro:
        raise AssertionError("Tested definition must be based on numerical checks for the listed use")
    if not re.search(r"does not imply (?:that every route has a complete worked analysis|a complete worked analysis for every route)", intro):
        raise AssertionError("Tested definition must not imply every route has a complete worked analysis")
    if "calibrated interval coverage" not in intro:
        raise AssertionError("Tested definition must distinguish numerical checks from calibrated coverage")
    for family in ("Beta", "Gamma"):
        row = next((line for line in markdown.splitlines() if line.startswith(f"| {family} |")), "")
        if "numerical tests but no complete worked analysis" not in row:
            raise AssertionError(f"{family} crossed-route worked-analysis qualification was lost")


def assert_recent_capability_corrections(markdown):
    """Reject stale handoff copies that erase already merged route corrections."""
    rows = markdown.splitlines()
    zero_one = next(line for line in rows if line.startswith("| ZeroOneBeta |"))
    assert "mean (`mu`) random intercepts" in zero_one
    default = next(line for line in rows if line.startswith("| `marginal=:LA`, the default integrator"))
    assert "a single node for LogNormal, where it is exact" in default
    meta = next(line for line in rows if line.startswith("| `meta_V(v)` plus random intercepts"))
    assert "correlation scale for every `sigma` formula" in meta
    scale = next(line for line in rows if line.startswith("| Random effect on the Gaussian scale axis"))
    assert "Parametric bootstrap now redraws" in scale
    assert "does not establish interval coverage" in scale


def assert_mixed_margin_identification_is_qualified(overview, guide):
    """Keep mixed-family identifiability conditional on the route and data."""
    mixed = overview.split("For Gaussian ×", 1)[1].split("## The latent-scale correlation", 1)[0]
    normalized = re.sub(r"\s+", " ", mixed).lower()
    if "all parameters are identified" in normalized:
        raise AssertionError("Mixed-family identifiability must not be stated categorically")
    for phrase in (
        "family pair",
        "parameterisation",
        "data",
        "do not assume every loading and dispersion is separately identified",
    ):
        if phrase not in normalized:
            raise AssertionError(f"Mixed-family identification qualification is missing: {phrase}")
    if "depends on the family pair, its parameterisation, and the data" not in " ".join(guide.split()).lower():
        raise AssertionError("The methods guide and overview must agree on identification scope")


class DocumentationSemantics(unittest.TestCase):
    def text(self, path):
        return (ROOT / path).read_text()
    def test_preserves_recently_merged_capability_corrections(self):
        assert_recent_capability_corrections(self.text("docs/src/capabilities.md"))
    def test_rejects_stale_handoff_route_regressions(self):
        text = self.text("docs/src/capabilities.md")
        for old, new in [
            ("mean (`mu`) random intercepts", "fixed effects only"),
            ("a single node for LogNormal, where it is exact", "five nodes for every family"),
            ("correlation scale for every `sigma` formula", "raw scale with constant sigma"),
            ("Parametric bootstrap now redraws", "Parametric bootstrap never redraws"),
        ]:
            self.assertIn(old, text)
            with self.assertRaises(AssertionError):
                assert_recent_capability_corrections(text.replace(old, new))
    def test_mixed_margin_identification_is_conditional(self):
        assert_mixed_margin_identification_is_qualified(
            self.text("docs/src/cross-family.md"),
            self.text("docs/src/model-guides/cross-family-methods.md"),
        )
    def test_family_map_lists_all_public_constructors(self):
        exports = self.text("src/DRModels.jl")
        names = {"Gaussian", "Student", "LogNormal", "Gamma", "Tweedie", "Poisson", "NegBinomial2", "TruncatedNegBinomial2", "TruncatedPoisson", "Beta", "BetaBinomial", "Binomial", "ZeroOneBeta", "CumulativeLogit", "SkewNormal"}
        table = self.text("docs/src/model-guides/model-map.md").split("## Structured and random effects")[0]
        for name in names:
            self.assertRegex(exports, r"\b" + name + r"\b")
            self.assertIn("[`" + name + "`](@ref)", table)
    def test_binomial_slope_distinction(self):
        text = self.text("docs/src/model-guides/model-map.md")
        self.assertNotIn("supports `(1 | g)` on the mean only", text)
        self.assertIn("Binomial", text)
        self.assertIn("independent `(0 + x | g)`", text)
        self.assertIn("correlated `(1 + x | g)`", text)
    def test_parameter_prediction_links(self):
        text = self.text("src/gaussian_core.jl").split("Population-level prediction of **every**")[1].split('"""')[0]
        self.assertNotIn("mean is left on the link scale", text)
        self.assertNotIn("comes\nback as `ν − 2`", text)
        self.assertIn("`2 + exp(η)`", text)
        self.assertIn("SkewNormal", text)
        self.assertIn("identity", text)
        self.assertIn("ZeroOneBeta", text)
        self.assertIn("logistic", text)
    def test_prediction_target_and_laplace_exception(self):
        text = self.text("docs/src/reference/model-fitting-and-postfit.md")
        self.assertNotIn("returns the response (mean) prediction", text)
        self.assertIn("median", text)
        self.assertIn("untruncated", text)
        text = self.text("docs/src/model-guides/marginal-la-vs-va.md")
        self.assertIn("LogNormal", text)
        self.assertIn("single mode-centred Gauss–Hermite node", text)
    def test_simulate_docstring_matches_current_keyword_defaults(self):
        source = self.text("src/gaussian_core.jl")
        doc = source.split('    simulate(fit;', 1)[1].split('"""', 1)[0]
        method_start = source.index('function simulate(fit::DrmFit;')
        method_end = source.index('\n    nsim >=', method_start)
        method = source[method_start:method_end]
        for documented, implemented in (
            ("nsim = 1", "nsim::Integer = 1"),
            ("newdata = nothing", "newdata = nothing"),
            ("re = :marginal", "re::Symbol = :marginal"),
            ("data = nothing", "data = nothing"),
            ("K = nothing", "K = nothing"),
            ("A = nothing", "A = nothing"),
            ("tree = nothing", "tree = nothing"),
            ("coords = nothing", "coords = nothing"),
        ):
            self.assertIn(documented, doc)
            self.assertIn(implemented, method)
    def test_simulate_parity_scopes_newdata_and_temporal_refusal(self):
        text = " ".join(self.text("docs/src/drmtmb-parity.md").split())
        self.assertIn("For non-temporal fits, `simulate(fit; newdata)` stays population level", text)
        self.assertIn("drmTMB does not accept `newdata` for simulation", text)
        self.assertIn("Temporal fits in DRModels.jl reject `newdata`", text)
        self.assertNotIn("differ in three ways", text)
    def test_offset_table_separates_random_effect_and_zi_hu_routes(self):
        text = " ".join(self.text("docs/src/reference/model-specification.md").split())
        self.assertIn("Fixed-effects `NegBinomial2()` routes with `zi` or `hu` also support offsets", text)
        self.assertIn("they cannot be combined with a random effect", text)
        self.assertIn("`Gamma()` and `Beta()` with one `(1 | g)` under `marginal = :Laplace`", text)
    def test_rosetta_simulate_data_note_names_temporal_exception(self):
        text = self.text("docs/src/rosetta.md")
        row = next(line for line in text.splitlines() if "`simulate(fit)`" in line)
        self.assertIn("temporal fits can simulate their full marginal chain without it", row)
    def test_temporal_experimental_scope(self):
        text = self.text("docs/src/api-stability.md").split("## What is still experimental")[1].split("## Engine functions")[0]
        for term in ("AR1", "OU", "Toeplitz", "intercept-only", "maximum likelihood"):
            self.assertIn(term, text)
    def test_capabilities_markdown_table_arity_and_escaped_pipes(self):
        text = self.text("docs/src/capabilities.md")
        section = text.split('## R → Julia bridge (engine = "julia")', 1)[1].split("## ", 1)[0]
        assert_markdown_table_arity(section)
        self.assertEqual(markdown_table_cells(r"| `phylo(1\|g)` | **Tested** |"), ["`phylo(1|g)`", "**Tested**"])
    def test_capabilities_tested_definition_scope(self):
        assert_tested_definition_scope(self.text("docs/src/capabilities.md"))
    def test_profile_ci_formula_refusal_is_documented(self):
        text = re.sub(r"\s+", " ", self.text("docs/src/capabilities.md"))
        self.assertIn("`profile_ci = true` is refused unless the `sigma` formula contains", text)
        self.assertIn("use `confint(fit, method = :profile)` after fitting", text)
        self.assertIn("currently ignores `profile_ci = true`", text)
    def test_capabilities_r_bridge_row_keeps_two_cell_scope_qualification(self):
        text = self.text("docs/src/capabilities.md")
        section = text.split('## R → Julia bridge (engine = "julia")', 1)[1].split("## ", 1)[0]
        row = next((line for line in section.splitlines() if "Full R-side glue" in line), "")
        cells = markdown_table_cells(row)
        self.assertEqual(len(cells), 2, f"R bridge row should have two table cells, got {len(cells)}")
        status = cells[1]
        self.assertIn("Not available here", status)
        self.assertIn("Julia primitive is tested", status)
        self.assertIn("R package glue lives in the drmTMB repository", status)
    def test_semantic_guards_reject_malformed_row_and_old_tested_definition(self):
        with self.assertRaisesRegex(AssertionError, "has 3 cells; expected 2"):
            assert_markdown_table_arity("| Capability | Status |\n|---|---|\n| R bridge | scope | Not available |")
        old = "This is deliberately conservative: ‘tested’ means the route has a worked example and routine checks for the listed use."
        with self.assertRaisesRegex(AssertionError, "numerical checks"):
            assert_tested_definition_scope(old)

    def test_marginal_comparison_claims_are_bound_to_the_retained_smoke(self):
        text = " ".join(self.text("docs/src/model-guides/marginal-la-vs-va.md").split())
        report_url = (
            "https://github.com/itchyshin/DRModels.jl/blob/"
            "64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716/report/va-vs-laplace-bias.md"
        )
        self.assertIn(report_url, text)
        self.assertRegex(text, r"three-seed smoke")
        self.assertIn("does not establish general accuracy or coverage", text)
        self.assertNotIn("295 to 501 units", text)
        self.assertNotIn("3.2 to 20 times", text)
        self.assertNotRegex(text, r"depend on the optimiser, the OS, or the BLAS")
        self.assertIn("Random.seed!(1)", text)
        self.assertIn("using DRModels, Distributions, Random", text)
        self.assertIn("data = (; y, x, g)", text)
        self.assertNotIn("data = dat", text)

    def test_bootstrap_result_documents_route_specific_keywords(self):
        text = " ".join(self.text("docs/src/reference/model-fitting-and-postfit.md").split())
        self.assertIn("bootstrap_result(formula, family::Gaussian; data, B = 300", text)
        self.assertIn("bootstrap_result(formula, family; data, B = 300", text)
        self.assertIn("bootstrap_result(fit::DrmFit{<:Gaussian}; data, B = 300", text)
        self.assertIn("bootstrap_result(fit::DrmFit; data, B = 300", text)
        self.assertIn("algorithm = :auto, g_tol = 1e-8, method = :ML", text)
        self.assertIn("algorithm` and `g_tol` are accepted on the Gaussian routes only", text)

    def test_large_count_and_gaussian_comparisons_are_not_conflated(self):
        text = " ".join(self.text("docs/src/drmtmb-parity.md").split())
        self.assertIn("the p=10,000 count fit is Julia-only", text)
        self.assertIn("paired p=10,000 timing", text)
        self.assertIn("it is a different model, not a count-route comparator", text)

    def test_small_sample_records_keep_their_distinct_denominators(self):
        text = " ".join(self.text("docs/src/diagnostics-and-validation/small-sample-behaviour.md").split())
        self.assertIn("coverage report lists `−1.852` and 986 intervals", text)
        self.assertIn("965 converged fits out of 1000", text)
        self.assertIn("N-ladder TSV reports `−1.8178` from 291 converged fits", text)
        self.assertIn("do not combine their counts", text)
if __name__ == "__main__":
    unittest.main()
