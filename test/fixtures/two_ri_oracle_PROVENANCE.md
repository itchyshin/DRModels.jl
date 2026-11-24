# Two-intercept prior oracle provenance

The fixture `prior_table_oracle.jl` contains numerical inputs and reference outputs from an independently derived dense Gaussian crossed-random-intercept oracle. The project owner confirmed that the identified oracle scripts and fixtures may be made available under the MIT terms in `two_ri_oracle_LICENSE`, subject to final provenance review. This confirmation does not relicense drmTMB GPL implementation source or third-party material.

| Artifact | SHA-256 |
|---|---|
| Julia fixture `prior_table_oracle.jl` | `b64bf2cfaa027996e40865ff1d20cf5be4042d617349ad56ea8bf95d0ffc77b6` |
| Oracle generator `oracle.py` | `d6e272b9993c686a50909ffca6714154503f29a666bb49c3b00f533da064c80e` |
| Asymmetric generator `supplemental.py` | `b11598b421f6f57360b38f6541e76a1614a9a52e7f04585566fb23942d038ec7` |
| Frozen input `frozen.json` | `d2bfbbac2d5e0127c90ef877c01259fda56526b2797240ddf718e0f782f6e1e2` |
| Frozen input `asymmetric-frozen.json` | `91de5613071175a8789a349243230a2023ae6aadd70bb35f9d6acc48938a0b85` |

A retained Codex session records generator creation and one subsequent correction; independent reconstruction reproduced the final generator bytes. A separate mathematical review checked the likelihood, full derivatives, finite-difference comparisons and mutation controls before package comparisons. The initial generator run failed because SciPy was unavailable; the corrected run succeeded. The original SciPy version and original Julia fixture converter were not recorded. An independently written replacement converter reproduces the Julia fixture from the frozen JSON files, but does not establish the missing history. The inspected Julia changes did not contain demonstrated copied drmTMB GPL implementation source. This record covers the inspected files; it does not establish the licensing of every dependency.
